"""Strict read-only postrun verification with explicit window/job receipt bindings.

No model calls, order-based legacy reconstruction, report mutation or SQLite
writes. Missing call_id/source binding is unproven, even if global counts match.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acceptance_common import digest, write_json
from acceptance_model_http import feedback_run_completion
from zhilv_qiling_v2.long_context import Segment, ReviewWindow, coverage
from zhilv_qiling_v2.models import Clause

MODEL = 'legalone-8b'
EVENT_FIELDS = ('event_id', 'created_at', 'trace_id', 'contract_id', 'component', 'model',
                'operation', 'status', 'elapsed_seconds', 'reason', 'details')
PENDING_FIELDS = ('pending_feedback_item_ids', 'pending_subject_verification_request_ids',
                  'pending_feedback_fingerprints')


def source_sha256(segments):
    return digest(json.dumps(segments, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))


def load_sqlite_events(path, trace_id):
    """Open the existing audit database with SQLite mode=ro; never create a DB."""
    path = Path(path).resolve()
    if not path.is_file() or not isinstance(trace_id, str) or not trace_id:
        raise ValueError('existing audit database and exact report trace_id required')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('SELECT * FROM model_calls WHERE trace_id=? ORDER BY created_at,event_id',
                          (trace_id,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item['details'] = json.loads(item['details'])
        result.append(item)
    return result


def _event_core(row):
    return {key: row.get(key) for key in EVENT_FIELDS}


def _known_list(value):
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value) and len(set(value)) == len(value)


def _receipt_checks(events, trace_id, contract_id):
    selected = [row for row in events if row.get('component') == 'llm' and row.get('operation') == 'request_json']
    starts, terminals = {}, {}
    issues = []
    for row in selected:
        details = row.get('details')
        call_id = details.get('call_id') if isinstance(details, dict) else None
        if not isinstance(call_id, str) or not call_id:
            issues.append('request_event_missing_call_id')
            continue
        bucket = starts if row.get('status') == 'started' else terminals
        if call_id in bucket:
            issues.append('duplicate_start_or_terminal:' + call_id)
        bucket[call_id] = row
    if not starts or set(starts) != set(terminals):
        issues.append('request_start_terminal_ids_not_one_to_one')
    verified = {}
    for call_id in set(starts) & set(terminals):
        start, terminal = starts[call_id], terminals[call_id]
        sd, td = start['details'], terminal['details']
        okay = (start.get('model') == MODEL and terminal.get('model') == MODEL
                and sd.get('requested_model') == MODEL and td.get('requested_model') == MODEL
                and terminal.get('status') == 'succeeded'
                and td.get('model_identity_verified') is True and td.get('actual_model') == MODEL
                and td.get('finish_reason') not in ('length', 'time_limit')
                and not terminal.get('reason')
                and start.get('trace_id') == terminal.get('trace_id') == trace_id
                and start.get('contract_id') == terminal.get('contract_id') == contract_id)
        try:
            okay = okay and datetime.fromisoformat(start['created_at']) <= datetime.fromisoformat(terminal['created_at'])
        except (KeyError, TypeError, ValueError):
            okay = False
        if not okay:
            issues.append('request_receipt_not_verified_success:' + call_id)
        verified[call_id] = okay
    return {'all_real_requests_have_verified_success_receipts': not issues,
            'issues': issues, 'request_count': len(starts), 'terminal_count': len(terminals),
            'verified_request_ids': sorted(call_id for call_id, okay in verified.items() if okay)}, starts, terminals


def _source_from_spans(spans, clauses):
    if not isinstance(spans, list) or not spans:
        return None
    result = []
    for span in spans:
        if not isinstance(span, dict):
            return None
        cid, start, end = (span.get(key) for key in ('clause_id', 'start', 'end'))
        if (cid not in clauses or type(start) is not int or type(end) is not int
                or not (0 <= start < end <= len(clauses[cid].text))):
            return None
        result.append({'clause_id': cid, 'start': start, 'end': end, 'text': clauses[cid].text[start:end]})
    return result


def _binding_check(call, starts, terminals, clauses, *, expected_window=None):
    issues = []
    if not isinstance(call, dict):
        return {'verified': False, 'issues': ['call_row_not_object']}
    window_id, job_id, call_id = (call.get(key) for key in ('window_id', 'job_id', 'call_id'))
    source = _source_from_spans(call.get('source_spans'), clauses)
    if not all(isinstance(value, str) and value for value in (window_id, job_id, call_id)):
        issues.append('explicit_window_job_call_id_missing_unproven')
    if source is None:
        issues.append('source_spans_missing_or_invalid_unproven')
    else:
        expected_hash = source_sha256(source)
        if call.get('source_sha256') != expected_hash:
            issues.append('source_digest_mismatch')
        if expected_window is not None:
            wanted = [segment.as_dict() for segment in expected_window.segments]
            if source != wanted or window_id != expected_window.window_id or call.get('kind') != expected_window.kind:
                issues.append('initial_source_does_not_match_exact_prepared_window')
    start, terminal = starts.get(call_id), terminals.get(call_id)
    if not start or not terminal:
        issues.append('call_id_does_not_resolve_to_one_start_and_terminal')
    else:
        for event in (start, terminal):
            if any(event.get('details', {}).get(key) != call.get(key)
                   for key in ('window_id', 'job_id', 'source_sha256')):
                issues.append('audit_window_job_source_binding_mismatch')
                break
        if call.get('model') != terminal.get('details', {}).get('actual_model'):
            issues.append('call_model_does_not_match_actual_receipt_model')
    return {'verified': not issues, 'window_id': window_id, 'job_id': job_id, 'call_id': call_id, 'issues': issues}



def _feedback_job_assessment(run, clauses, *, current_items=None, focus_ids=None):
    """Recompute every declared feedback batch; aggregate acknowledgements cannot hide a missed job."""
    issues, plans = [], {}
    rows = run.get('calls') if isinstance(run.get('calls'), list) else []
    items = run.get('feedback_items')
    item_rows = current_items if current_items is not None else items
    known_items = isinstance(items, list) and isinstance(item_rows, list) and all(isinstance(item, dict) for item in items + item_rows)
    item_map = {item.get('feedback_item_id'): item for item in item_rows} if known_items else {}
    metadata_map = {item.get('feedback_item_id'): item for item in items} if known_items else {}
    if (not known_items or not all(isinstance(key, str) and key for key in item_map)
            or len(item_map) != len(item_rows)
            or Counter(item.get('feedback_item_id') for item in items) != Counter(item_map.keys())):
        issues.append('feedback_input_ids_missing_duplicate_or_not_current')
    window_plans = run.get('feedback_windows')
    if not isinstance(window_plans, list):
        window_plans = []; issues.append('feedback_window_job_plan_missing')
    window_ids, window_sources = [], {}
    for window in window_plans:
        if not isinstance(window, dict) or not isinstance(window.get('batches'), list):
            issues.append('invalid_feedback_window_plan'); continue
        window_ids.append(window.get('window_id'))
        batches = window['batches']
        if type(window.get('batch_count')) is not int or window['batch_count'] != len(batches):
            issues.append('feedback_batch_count_mismatch')
        if window.get('oversize_feedback_item_ids'):
            issues.append('feedback_input_oversize_unreviewed')
        for batch in batches:
            if not isinstance(batch, dict) or not isinstance(batch.get('job_id'), str) or batch['job_id'] in plans:
                issues.append('feedback_job_missing_or_duplicate'); continue
            plans[batch['job_id']] = (window, batch)
    if len(set(window_ids)) != len(window_ids):
        issues.append('duplicate_feedback_window_plan')
    by_job = {}
    for call in rows:
        if not isinstance(call, dict) or not isinstance(call.get('job_id'), str) or call['job_id'] in by_job:
            issues.append('feedback_call_missing_or_duplicate_job'); continue
        by_job[call['job_id']] = call
    if set(by_job) - set(plans): issues.append('feedback_call_not_in_declared_plan')
    completed_jobs, completed_windows, matched_items = set(), set(), set()
    for job_id, (window, batch) in plans.items():
        call = by_job.get(job_id)
        expected = batch.get('feedback_item_ids')
        okay = bool(call and _known_list(expected) and set(expected).issubset(item_map))
        if not okay:
            issues.append('feedback_job_missing_call_or_invalid_item_plan:' + job_id); continue
        matched_items.update(expected)
        okay = (call.get('window_id') == window.get('window_id') and call.get('kind') == window.get('kind')
            and type(call.get('batch_index')) is int and call['batch_index'] == batch.get('batch_index')
            and type(call.get('batch_count')) is int and call['batch_count'] == batch.get('batch_count')
            and call.get('feedback_item_ids') == expected and call.get('status') == 'ok'
            and _known_list(call.get('verified_feedback_item_ids', []))
            and set(call.get('verified_feedback_item_ids', [])) == set(expected)
            and call.get('uncompleted_feedback_item_ids') == [] and call.get('verification_errors', []) == [])
        actual_source = _source_from_spans(call.get('source_spans'), clauses)
        if actual_source is None:
            okay = False
        else:
            previous_source = window_sources.setdefault(window['window_id'], actual_source)
            if previous_source != actual_source: okay = False
        for item_id in expected:
            item = item_map[item_id]
            subject = item.get('feedback_kind') == 'subject_fact'
            root = 'subject_fact_verifications' if subject else 'verifications'
            root_rows = call.get(root, [])
            root_ids = {value for value in expected if (item_map[value].get('feedback_kind') == 'subject_fact') == subject}
            okay = okay and isinstance(root_rows, list) and all(isinstance(value, dict) and value.get('feedback_item_id') in root_ids for value in root_rows) and len(root_rows) == len(root_ids)
            verifications = [row for row in root_rows if isinstance(row, dict) and row.get('feedback_item_id') == item_id] if isinstance(root_rows, list) else []
            truncated = any(item.get(key) or metadata_map.get(item_id, {}).get(key) for key in
                ('article_quote_truncated', 'fact_evidence_text_truncated', 'source_evidence_text_truncated'))
            if current_items is not None:
                truncated = truncated or any(isinstance(item.get(key), str) and len(item[key]) > 1000 for key in
                    (('fact_evidence_text', 'source_evidence_text') if subject else ('article_quote',)))
            okay = okay and not truncated
            scope_ids = item.get('clause_ids')
            scope_ids = set(scope_ids) if _known_list(scope_ids) else set()
            bound_sources = [part for part in (actual_source or []) if part['clause_id'] in scope_ids]
            okay = okay and bool(bound_sources) and len(verifications) == 1
            if len(verifications) != 1: continue
            row = verifications[0]
            identity = ('fact_id', 'source_id', 'subject_name', 'fact_type', 'clause_id') if subject else ('candidate_id', 'article_uid')
            if current_items is None and subject:
                identity = tuple(key for key in identity if key != 'clause_id')
            okay = okay and all(row.get(key) == item.get(key) and item.get(key) is not None for key in identity)
            if subject:
                okay = okay and row.get('clause_id') in scope_ids and any(part['clause_id'] == row.get('clause_id') for part in bound_sources)
            quote = row.get('clause_quote')
            okay = okay and row.get('verdict') in ('confirmed', 'rejected') and isinstance(quote, str) and bool(quote)
            okay = okay and isinstance(row.get('reason'), str) and bool(row['reason'].strip())
            quote_sources = [part for part in bound_sources if not subject or part['clause_id'] == row.get('clause_id')]
            okay = okay and any(quote in part['text'] for part in quote_sources) if isinstance(quote, str) else False
            if current_items is not None:
                for quote_key, input_key in ((('fact_quote', 'fact_evidence_text'), ('source_quote', 'source_evidence_text'))
                                             if subject else (('article_quote', 'article_quote'),)):
                    quoted, supplied = row.get(quote_key), item.get(input_key)
                    if quote_key == 'source_quote' and supplied == '':
                        okay = okay and quoted == ''
                    else:
                        okay = okay and isinstance(quoted, str) and bool(quoted) and isinstance(supplied, str) and quoted in supplied
            if subject:
                okay = okay and item.get('source_complete') is True and item.get('effective_status') not in (None, '', 'unknown', 'conflicting_evidence') and not item.get('upstream_pending')
        if okay: completed_jobs.add(job_id)
        else: issues.append('feedback_job_status_ack_or_item_source_invalid:' + job_id)
    for window in window_plans:
        if isinstance(window, dict) and isinstance(window.get('batches'), list) and window['batches'] and all(batch.get('job_id') in completed_jobs for batch in window['batches']):
            completed_windows.add(window['window_id'])
    cross_count = sum(isinstance(window, dict) and window.get('kind') == 'cross_clause' for window in window_plans)
    cross_done = sum(isinstance(window, dict) and window.get('kind') == 'cross_clause' and window.get('window_id') in completed_windows for window in window_plans)
    counts = (type(run.get('planned_calls')) is int and run['planned_calls'] == len(plans)
        and type(run.get('completed_calls')) is int and run['completed_calls'] == len(completed_jobs)
        and type(run.get('cross_windows')) is int and run['cross_windows'] == cross_count
        and type(run.get('cross_windows_completed')) is int and run['cross_windows_completed'] == cross_done)
    if not counts: issues.append('feedback_declared_counts_disagree_with_actual_jobs')
    independent_scope = None
    if focus_ids is not None and _known_list(focus_ids) and all(cid in clauses for cid in focus_ids):
        source_windows = [ReviewWindow(window['window_id'], window['kind'], tuple(Segment(**part) for part in window_sources[window['window_id']]))
                          for window in window_plans if isinstance(window, dict) and window.get('window_id') in window_sources]
        independent_scope = coverage([clauses[cid] for cid in focus_ids], source_windows, completed_windows)
        if not independent_scope['complete']: issues.append('current_feedback_focus_source_or_reference_coverage_incomplete')
    complete = bool(plans and not issues and len(completed_jobs) == len(plans) and matched_items == set(item_map)
        and run.get('uncompleted_feedback_item_ids') == [] and run.get('unmatched_feedback_item_ids') == []
        and set(run.get('verified_feedback_item_ids') or []) == set(item_map)
        and feedback_run_completion(run)['verification_complete'])
    return {'complete': complete, 'issues': issues, 'planned_jobs': len(plans), 'actual_call_rows': len(rows),
        'independently_completed_jobs': len(completed_jobs), 'independent_focus_coverage': independent_scope,
        'false_complete_summary': bool((run.get('complete') is True and run.get('feedback_complete') is True) and not complete)}


def _current_feedback_assessment(logic, cross, clauses, pending_present):
    manifest, runs = logic.get('feedback_current_request'), logic.get('feedback_runs')
    bindings, attempts, latest = logic.get('feedback_run_bindings'), logic.get('feedback_attempts'), logic.get('feedback_latest_runs')
    known = (isinstance(manifest, dict) and isinstance(runs, list) and isinstance(bindings, list)
             and isinstance(attempts, list) and isinstance(latest, list))
    issues, run_assessments, current_assessment = [], [], None
    if not known:
        return {'manifest_and_attempts_verified': False, 'protocol_ledger_consistent': False, 'complete': False,
                'issues': ['explicit_current_feedback_manifest_or_attempt_bindings_missing_unproven'], 'run_assessments': []}
    items, focus = manifest.get('items'), manifest.get('focus_clause_ids')
    item_ids = [item.get('feedback_item_id') for item in items] if isinstance(items, list) and all(isinstance(item, dict) for item in items) else None
    if not _known_list(item_ids) or not _known_list(focus) or any(cid not in clauses for cid in focus):
        issues.append('current_feedback_item_or_focus_manifest_invalid')
    declared_pending = manifest.get('pending_subject_request_ids')
    if not _known_list(declared_pending) or declared_pending != cross.get('pending_subject_verification_request_ids'):
        issues.append('current_subject_request_ledger_mismatch')
    required = bool(items or declared_pending or manifest.get('subject_verification_complete') is False)
    if items and not focus:
        issues.append('current_feedback_items_have_no_bound_focus')
    item_clause_ids = {cid for item in (items or []) if isinstance(item, dict)
                       for cid in item.get('clause_ids', []) if isinstance(cid, str)}
    if focus and not set(focus).issubset(item_clause_ids):
        issues.append('current_feedback_focus_not_covered_by_input_items')
    if type(manifest.get('required')) is not bool or manifest['required'] != required:
        issues.append('current_feedback_required_flag_mismatch')
    by_id, indices = {}, []
    if len(bindings) != len(runs) or len(attempts) != len(runs): issues.append('feedback_history_binding_count_mismatch')
    attempts_by_id = {row.get('attempt_id'): row for row in attempts if isinstance(row, dict)}
    for binding in bindings:
        if not isinstance(binding, dict): issues.append('feedback_attempt_binding_invalid'); continue
        index, attempt_id, fingerprint = binding.get('attempt_index'), binding.get('attempt_id'), binding.get('fingerprint')
        if (type(index) is not int or not 1 <= index <= len(runs) or not isinstance(attempt_id, str) or not attempt_id
                or attempt_id in by_id or not isinstance(fingerprint, str) or not fingerprint):
            issues.append('feedback_attempt_identity_invalid'); continue
        indices.append(index); by_id[attempt_id] = binding
        run = runs[index-1]
        if not isinstance(run, dict) or binding.get('run_sha256') != source_sha256(run):
            issues.append('feedback_run_snapshot_digest_mismatch'); continue
        attempt = attempts_by_id.get(attempt_id)
        if not attempt or any(attempt.get(key) != binding.get(key) for key in ('attempt_index', 'fingerprint', 'round')):
            issues.append('feedback_attempt_ledger_identity_mismatch')
        assessment = _feedback_job_assessment(run, clauses)
        run_assessments.append({'attempt_id': attempt_id, **assessment})
        if assessment['false_complete_summary']: issues.append('feedback_history_falsely_claims_complete:' + attempt_id)
    if sorted(indices) != list(range(1, len(runs)+1)) or len(attempts_by_id) != len(attempts):
        issues.append('feedback_attempt_indices_or_ids_not_unique_complete')
    fingerprint = manifest.get('fingerprint')
    matching = [binding for binding in by_id.values() if binding.get('fingerprint') == fingerprint]
    current = max(matching, key=lambda binding: binding['attempt_index']) if matching else None
    if required and (not isinstance(fingerprint, str) or not fingerprint): issues.append('current_feedback_fingerprint_missing')
    if not required and fingerprint is not None: issues.append('unexpected_fingerprint_for_no_required_feedback')
    if current:
        if manifest.get('attempt_id') != current['attempt_id'] or manifest.get('attempt_index') != current['attempt_index']:
            issues.append('current_feedback_does_not_select_latest_explicit_attempt')
        run = runs[current['attempt_index']-1]
        if latest != [run]: issues.append('latest_feedback_snapshot_does_not_match_current_attempt')
        current_assessment = _feedback_job_assessment(run, clauses, current_items=items, focus_ids=focus)
    elif manifest.get('attempt_id') is not None or manifest.get('attempt_index') is not None or latest:
        issues.append('current_feedback_attempt_missing_or_unproven')
    current_complete = bool(not required or current_assessment and current_assessment['complete'])
    claim_agrees = bool(logic.get('feedback_runs_complete') is (not pending_present)
        and (pending_present or current_complete))
    return {'manifest_and_attempts_verified': not issues, 'protocol_ledger_consistent': not issues and claim_agrees,
        'complete': bool(not issues and claim_agrees and not pending_present and current_complete),
        'required': required, 'fingerprint': fingerprint, 'current_attempt_id': manifest.get('attempt_id'),
        'issues': issues, 'run_assessments': run_assessments, 'current_assessment': current_assessment}


def check(prepared, report, *, sqlite_events=None):
    """Verify serialized evidence; absent links stay unproven, never guessed."""
    stats = report.get('stats', {})
    logic = stats.get('logic_llm') or {}
    cross = stats.get('cross_validation') or {}
    raw_calls = logic.get('calls')
    calls = raw_calls if isinstance(raw_calls, list) else []
    fields_allowed = {field.name for field in fields(Clause)}
    clauses = [Clause(**{key: value for key, value in row.items() if key in fields_allowed}) for row in prepared['clauses']]
    clause_map = {clause.clause_id: clause for clause in clauses}
    windows = [ReviewWindow(row['window_id'], row['kind'], tuple(Segment(**part) for part in row['segments']),
                            row.get('reason', ''), tuple(row.get('dependency_ids', []))) for row in prepared['windows']]
    window_map = {window.window_id: window for window in windows}
    window_ids = [window.window_id for window in windows]
    actual_ids = [call.get('window_id') if isinstance(call, dict) else None for call in calls]
    done = {call.get('window_id') for call in calls if isinstance(call, dict) and call.get('status') == 'ok'}
    valid_segments = bool(clauses and windows and all(window.kind in ('text', 'cross_clause') and window.segments for window in windows))
    for window in windows:
        spans = [{'clause_id': part.clause_id, 'start': part.start, 'end': part.end} for part in window.segments]
        source = _source_from_spans(spans, clause_map)
        valid_segments = valid_segments and source == [part.as_dict() for part in window.segments]
    independent = coverage(clauses, windows, done) if valid_segments else {'text_complete': False, 'text_coverage': 0,
                                                                                   'explicit_reference_complete': False}
    events = stats.get('model_calls') if isinstance(stats.get('model_calls'), list) else []
    trace_id, contract_id = report.get('trace_id'), prepared.get('contract_id')
    receipt, starts, terminals = _receipt_checks(events, trace_id, contract_id)
    event_ids = [event.get('event_id') for event in events]
    db_ids = [event.get('event_id') for event in sqlite_events] if isinstance(sqlite_events, list) else []
    db_map = {event.get('event_id'): _event_core(event) for event in sqlite_events} if isinstance(sqlite_events, list) else {}
    sqlite_match = bool(isinstance(sqlite_events, list) and events and len(set(event_ids)) == len(event_ids)
                        and all(isinstance(value, str) and value for value in event_ids)
                        and len(set(db_ids)) == len(db_ids) and set(event_ids) == set(db_ids)
                        and all(_event_core(event) == db_map.get(event.get('event_id')) for event in events))
    binding_rows = [_binding_check(call, starts, terminals, clause_map,
                                   expected_window=window_map.get(call.get('window_id')) if isinstance(call, dict) else None)
                    for call in calls]
    binding_call_ids = [item['call_id'] for item in binding_rows if item.get('call_id')]
    initial_jobs = [call.get('job_id') for call in calls if isinstance(call, dict)]
    wanted_jobs = {window_id + ':batch:1' for window_id in window_ids}
    initial_job_shape = bool(isinstance(raw_calls, list) and set(initial_jobs) == wanted_jobs
        and len(initial_jobs) == len(wanted_jobs)
        and all(isinstance(call, dict) and type(call.get('batch_index')) is int and call['batch_index'] == 1
                and type(call.get('batch_count')) is int and call['batch_count'] == 1
                and call.get('feedback_item_ids') == [] for call in calls)
        and type(logic.get('planned_calls')) is int and logic['planned_calls'] == len(windows)
        and type(logic.get('completed_calls')) is int and logic['completed_calls'] == len(windows))
    bindings_complete = bool(binding_rows and all(row['verified'] for row in binding_rows)
                             and len(binding_call_ids) == len(calls) == len(set(binding_call_ids)))
    pending_known = all(_known_list(cross.get(key)) for key in PENDING_FIELDS)
    pending_present = bool(any(cross.get(key) for key in PENDING_FIELDS)) if pending_known else True
    termination, converged = cross.get('termination'), cross.get('converged')
    # Accept only the producer's two canonical states. All synonyms and unknown
    # success-like spellings fail closed instead of bypassing a finite blacklist.
    termination_valid = termination in ('stable', 'round_limit')
    termination_consistent = bool(termination_valid and
        ((termination == 'stable' and converged is True and not pending_present)
         or (termination == 'round_limit' and converged is False)))
    ledger_consistent = bool(pending_known and type(cross.get('feedback_verification_complete')) is bool
                             and cross['feedback_verification_complete'] == (not pending_present))
    feedback_runs = logic.get('feedback_runs')
    latest = logic.get('feedback_latest_runs')
    feedback_bindings, feedback_sources_known = [], isinstance(feedback_runs, list)
    if isinstance(feedback_runs, list):
        for index, run in enumerate(feedback_runs):
            if not isinstance(run, dict) or not isinstance(run.get('calls'), list):
                feedback_sources_known = False
                continue
            for call in run['calls']:
                row = _binding_check(call, starts, terminals, clause_map)
                row['run_index'] = index
                feedback_bindings.append(row)
    all_bound_ids = binding_call_ids + [row.get('call_id') for row in feedback_bindings if row.get('call_id')]
    all_recorded_bindings = bool(bindings_complete and feedback_sources_known
        and all(row['verified'] for row in feedback_bindings)
        and len(all_bound_ids) == len(set(all_bound_ids)))
    current_feedback = _current_feedback_assessment(logic, cross, clause_map, pending_present)
    declared_window_ids = {call_id for call_id, event in starts.items()
        if any(key in event.get('details', {}) for key in ('window_id', 'job_id', 'source_sha256'))}
    audit_window_jobs_have_rows = declared_window_ids.issubset(set(all_bound_ids))
    initial_feedback = feedback_run_completion(logic)
    latest_known = isinstance(latest, list) and all(isinstance(run, dict) for run in latest)
    latest_assessments = [feedback_run_completion(run) for run in latest] if latest_known else []
    latest_belongs_to_history = bool(latest_known and isinstance(feedback_runs, list)
        and all(any(run == previous for previous in feedback_runs) for run in latest))
    feedback_final_complete = bool(initial_feedback['verification_complete'] and latest_known
        and latest_belongs_to_history and all(item['verification_complete'] for item in latest_assessments)
        and logic.get('feedback_runs_complete') is True and ledger_consistent and not pending_present
        and current_feedback['complete'])
    deep = [event for event in events if event.get('component') == 'entailment' and event.get('operation') == 'predict_pair']
    deep_counts = Counter(event.get('status') for event in deep)
    degraded = list(stats.get('degraded') or [])
    llm_degraded = [item for item in degraded if 'llm_' in str(item) or 'output_truncated' in str(item)]
    checks = {
        'document_text_equals_prepared': report.get('document', {}).get('full_text') == prepared.get('parsed_text'),
        'prepared_clause_ids_unique': len(clause_map) == len(clauses),
        'prepared_windows_nonempty_unique_and_source_exact': valid_segments and len(window_map) == len(windows),
        'initial_call_window_ids_match_plan_exactly': Counter(actual_ids) == Counter(window_ids),
        'initial_job_ids_batches_and_counts_match_plan': initial_job_shape,
        'every_prepared_initial_window_completed': bool(windows) and all(window_id in done for window_id in window_ids),
        'independent_text_coverage_complete': independent['text_complete'],
        'reported_text_complete_agrees': logic.get('text_complete') == independent['text_complete'],
        'reported_text_coverage_agrees': logic.get('text_coverage') == independent['text_coverage'],
        'reported_reference_status_agrees': logic.get('explicit_reference_complete') == independent['explicit_reference_complete'],
        'all_real_requests_have_verified_success_receipts': receipt['all_real_requests_have_verified_success_receipts'],
        'initial_window_job_call_ids_and_sources_bound_to_receipts': bindings_complete,
        'all_recorded_feedback_call_ids_and_sources_bound_to_receipts': all_recorded_bindings,
        'all_audit_declared_window_job_requests_have_call_records': audit_window_jobs_have_rows,
        'current_feedback_manifest_and_explicit_attempts_verified': current_feedback['manifest_and_attempts_verified'],
        'current_feedback_protocol_claim_agrees_with_pending_ledger': current_feedback['protocol_ledger_consistent'],
        'report_events_exactly_match_readonly_sqlite_trace': sqlite_match,
        'audit_event_trace_contract_identity_matches_report': bool(isinstance(trace_id, str) and trace_id and isinstance(contract_id, str)
            and contract_id and events and all(event.get('trace_id') == trace_id and event.get('contract_id') == contract_id for event in events)),
        'actual_deep_pair_success_present': deep_counts['succeeded'] > 0,
        'pending_ledger_present_and_consistent': ledger_consistent,
        'termination_canonical_consistent_and_no_false_stability': termination_consistent,
    }
    strict = all(checks.values())
    # Upstream semantic limits do not invalidate a correctly executed feedback
    # protocol. They separately prevent an effective-completion claim.
    semantic_pending = []
    semantic_sources = (('statute_checks', stats.get('statute_checks')),
        ('current_feedback_items', (logic.get('feedback_current_request') or {}).get('items')),
        ('current_feedback_consumption', logic.get('feedback_consumption_current')))
    for source_name, records in semantic_sources:
        if not isinstance(records, list):
            continue
        for row in records:
            if not isinstance(row, dict):
                continue
            reasons = ([key for key in ('input_truncated', 'manual_required', 'disagreement') if row.get(key)]
                       + (['coverage_complete_false'] if row.get('coverage_complete') is False else []))
            if reasons:
                semantic_pending.append({'source': source_name, 'reasons': reasons,
                    **{key: row[key] for key in ('feedback_item_id', 'candidate_id', 'article_uid', 'clause_ids') if key in row}})
    effective = bool(strict and independent['explicit_reference_complete'] and feedback_final_complete
                     and not llm_degraded and not semantic_pending)
    missing_bindings = [row for row in binding_rows + feedback_bindings if not row['verified']]
    return {'status': ('verified_execution_with_pending' if strict and not effective else
                       'verified_effective_scheduled_model_review' if effective else 'failed_or_unproven_strict_checks'),
        'strict_engineering_execution_verified': strict,
        'initial_window_log_and_source_coverage_complete': bool(checks['initial_call_window_ids_match_plan_exactly']
            and checks['every_prepared_initial_window_completed'] and independent['text_complete']),
        'effective_model_review_complete': effective, 'checks': checks,
        'missing_initial_window_ids': sorted(set(window_ids)-set(actual_ids), key=str),
        'unknown_initial_window_ids': sorted(set(actual_ids)-set(window_ids), key=str),
        'independent_source_coverage': independent,
        'explicit_reference_scope_complete': independent['explicit_reference_complete'],
        'receipts': receipt,
        'receipt_bindings': {'status': 'verified' if all_recorded_bindings else 'unproven_or_mismatched',
            'initial_bindings': binding_rows, 'feedback_bindings': feedback_bindings,
            'missing_or_mismatched_bindings': missing_bindings,
            'unassigned_actual_request_ids': sorted(set(starts)-set(all_bound_ids)),
            'legacy_order_or_timing_reconstruction_used': False},
        'sqlite_crosscheck': {'provided': isinstance(sqlite_events, list), 'exact_trace_event_match': sqlite_match,
                             'report_events': len(events), 'sqlite_trace_events': len(db_ids)},
        'feedback_completion': {'ledger_known': pending_known, 'ledger_consistent': ledger_consistent,
            'pending_present': pending_present, **{key: cross.get(key) for key in PENDING_FIELDS},
            'termination': termination, 'converged': converged,
            'final_protocol_feedback_complete': feedback_final_complete,
            'initial_feedback_assessment': initial_feedback, 'latest_feedback_assessments': latest_assessments,
            'independent_current_request': current_feedback},
        'deep_pair_summary': {'status_counts': dict(deep_counts),
            'degradation_reason_counts': dict(Counter(event.get('reason') for event in deep if event.get('status') != 'succeeded')),
            'successful_deep_call_present': deep_counts['succeeded'] > 0,
            'complete_expected_pair_coverage_proven': False},
        'upstream_semantic_review': {'clear_of_reported_limits': not semantic_pending,
            'pending_checks': semantic_pending,
            'meaning': 'confirmed feedback may complete the protocol while truncated, disputed or manual-required upstream relations remain pending; these limits do not make protocol receipts invalid'},
        'llm_degradations': llm_degraded, 'legal_correctness_evaluated': False,
        'meaning': 'requires explicit source/window/job/call_id links and exact SQLite receipts; missing legacy links are unproven; pending and reference ambiguity never establish effective completion; deep success presence is not complete pair coverage or legal quality'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / 'tools/acceptance_results'):
        parser.error('run-dir must remain under tools/acceptance_results')
    output = (args.out or directory / 'postrun_strict_v2.json').resolve()
    if not output.is_relative_to(directory) or output.exists():
        parser.error('new output under this run directory required; preserve earlier evidence')
    paths = {name: directory / name for name in ('prepared.json', 'report.json', 'result.json', 'model_calls.sqlite')}
    try:
        data = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in paths.items() if path.suffix == '.json'}
        sqlite_events = load_sqlite_events(paths['model_calls.sqlite'], data['report.json'].get('trace_id'))
        result = check(data['prepared.json'], data['report.json'], sqlite_events=sqlite_events)
        initial = data['result.json']
        result['checks']['archive_roundtrip_verified_by_run'] = initial.get('http_archive_roundtrip_verified') is True
        result['checks']['saved_report_identity_matches_run'] = (data['report.json'].get('saved') is True
            and bool(data['report.json'].get('report_id')) and data['report.json'].get('report_id') == initial.get('report_id'))
        result['checks']['prepared_bytes_match_run_provenance'] = initial.get('prepared_sha256') == digest(paths['prepared.json'].read_bytes())
        result['strict_engineering_execution_verified'] = all(result['checks'].values())
        result['effective_model_review_complete'] = result['effective_model_review_complete'] and result['strict_engineering_execution_verified']
        if not result['strict_engineering_execution_verified']:
            result['status'] = 'failed_or_unproven_strict_checks'
        result['input_sha256'] = {name: digest(path.read_bytes()) for name, path in paths.items()}
        result['tool_sha256'] = digest(Path(__file__).read_bytes())
        write_json(output, result)
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
        result = {'status': 'failed_closed_invalid_or_missing_evidence', 'strict_engineering_execution_verified': False,
                  'effective_model_review_complete': False, 'legal_correctness_evaluated': False,
                  'error_type': type(exc).__name__, 'error': str(exc)[:300], 'tool_sha256': digest(Path(__file__).read_bytes())}
        write_json(output, result)
    print(json.dumps({'status': result['status'], 'evidence': str(output),
        'strict_engineering_execution_verified': result['strict_engineering_execution_verified'],
        'effective_model_review_complete': result['effective_model_review_complete']}, ensure_ascii=True))
    return 0 if result['strict_engineering_execution_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
