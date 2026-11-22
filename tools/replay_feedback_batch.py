"""Replay one genuinely failed synthetic HTTP feedback batch with current prompts.

This is a targeted real-model regression, not a new complete HTTP review,
accuracy score, or a replacement for the original failure evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acceptance_common import digest, provenance, request, require, write_json


def _legacy_subject_item(item, candidate_rows, requests):
    """Recompute a known V2 input ID using only its archived source records.

    Legacy candidate association is replay evidence, not the current product's
    identity decision. Refuse ambiguous request recovery instead of assigning
    an archived ID to a differently associated current item.
    """
    from zhilv_qiling_v2.models import stable_id
    legacy = dict(item)
    for field in ('feedback_item_id', 'candidate_binding_status', 'candidate_binding_errors'):
        legacy.pop(field, None)
    candidate_ids = {str(row['candidate_id']) for row in candidate_rows
        if isinstance(row, dict) and row.get('candidate_id')
        and legacy.get('clause_id') in (row.get('clause_ids') or [])
        and isinstance(row.get('extra'), dict)
        and row['extra'].get('subject_name') == legacy.get('subject_name')
        and (not legacy.get('subject_node_id') or not row['extra'].get('subject_node_id')
            or row['extra']['subject_node_id'] == legacy['subject_node_id'])}
    if legacy.get('request_id'):
        matches = [request for request in requests
            if isinstance(request, dict) and request.get('request_id') == legacy['request_id']]
        if len(matches) != 1:
            return None
        if matches[0].get('candidate_id'):
            candidate_ids.add(str(matches[0]['candidate_id']))
    legacy['candidate_ids'] = sorted(candidate_ids)
    legacy['feedback_item_id'] = stable_id('subject_feedback',
        json.dumps(legacy, sort_keys=True, ensure_ascii=False), length=24)
    return legacy


def restore_batch(report, kind, *, run_index=0, job_id=None):
    """Restore supplied evidence, never manufacture a replacement model answer."""
    from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, stable_id
    from zhilv_qiling_v2.agents_multi_run import _statute_feedback_item, _subject_fact_feedback
    fields = {field.name for field in dataclasses.fields(Clause)}
    all_clauses = [Clause(**{key:value for key,value in clause.items() if key in fields})
                   for clause in report['document']['clauses']]
    contract = Contract(report['contract_id'], DocumentParse('SYNTHETIC-feedback-replay.txt', 'txt',
        report['document']['full_text'], clauses=all_clauses), contract_type=report.get('contract_type', '通用'))
    run = report['stats']['logic_llm']['feedback_runs'][run_index]
    matches = [call for call in run.get('calls', [])
               if bool(call.get('subject_fact_feedback_items')) == (kind == 'subject')
               and call.get('status') in ('error', 'feedback_incomplete', 'partial', 'invalid_schema')
               and (job_id is None or call.get('job_id') == job_id)]
    if not matches or (job_id is not None and len(matches) != 1):
        raise ValueError(f'cannot uniquely select requested failed {kind} batch')
    # Default is the first archived failure, not whichever item now performs best.
    failed = matches[0]
    descriptions = {item['feedback_item_id']: item for item in run['feedback_items']}
    if len(descriptions) != len(run['feedback_items']) or len(set(failed['feedback_item_ids'])) != len(failed['feedback_item_ids']):
        raise ValueError('duplicate archived feedback IDs cannot identify one exact batch')
    original_indices = {item['feedback_item_id']: index for index, item in enumerate(run['feedback_items'])}
    if kind == 'statute':
        items = []
        for item_id in failed['feedback_item_ids']:
            description = descriptions[item_id]
            restored = []
            for check in report['stats']['statute_checks']:
                if (check.get('candidate_id'), check.get('article_uid')) != (description['candidate_id'], description['article_uid']):
                    continue
                # Recompute either supported content-bound ID from source
                # records. The current producer hashes the full quote before
                # transport bounds it; old V2 archives hashed the bounded
                # record with its original global preparation index.
                current = _statute_feedback_item(check)
                if current['feedback_item_id'] == item_id:
                    restored.append(current)
                clean = dict(check)
                quote = str(clean.get('article_quote') or '')
                clean['article_quote'] = quote[:1000]
                clean['article_quote_truncated'] = bool(clean.get('article_quote_truncated') or len(quote) > 1000)
                derived = stable_id('feedback', original_indices[item_id],
                    json.dumps(clean, sort_keys=True, ensure_ascii=False), length=16)
                if derived == item_id:
                    restored.append({**clean, 'feedback_item_id': item_id})
            if not restored:
                raise ValueError('archived statute record cannot reproduce original content-bound feedback ID: ' + item_id)
            items.append(restored[0])
    else:
        subject = report['subject_checks']
        payload = {**subject['stats'], 'candidates':subject.get('candidates', [])}
        bus = SimpleNamespace(inbox=lambda agent, kinds: [SimpleNamespace(
            payload={'risk_checks':report['stats']['statute_checks']})] if kinds == ['statute.matched'] else [])
        current = _subject_fact_feedback(SimpleNamespace(contract=contract, bus=bus), payload)
        restored = {item['feedback_item_id']: item for item in current}
        candidate_rows = list(payload.get('candidates') or []) + list(report['stats']['statute_checks'])
        for item in current:
            legacy = _legacy_subject_item(item, candidate_rows, payload.get('verification_results') or [])
            if legacy is not None:
                restored.setdefault(legacy['feedback_item_id'], legacy)
        missing = set(failed['feedback_item_ids']) - restored.keys()
        if missing:
            raise ValueError(f'archived subject records cannot reproduce original feedback IDs: {sorted(missing)}')
        items = [restored[item_id] for item_id in failed['feedback_item_ids']]
    for item in items:
        description = descriptions[item['feedback_item_id']]
        for key, value in description.items():
            if key.endswith('_truncated') and value is False and key not in item:
                continue
            if item.get(key) != value:
                raise ValueError(f'archived feedback differs at {key}; cannot claim exact restoration')
    clauses = [clause for clause in all_clauses if clause.clause_id in failed['clause_ids']]
    if not clauses or set(failed['clause_ids']) != {clause.clause_id for clause in clauses}:
        raise ValueError('cannot restore exact source clause IDs')
    return contract, clauses, failed, items


def expected_replay_schema(config, clauses, items, kind):
    """Rebuild the first real window's schema from source text, not receipts."""
    from zhilv_qiling_v2 import nodes as N
    from zhilv_qiling_v2.long_context import build_review_plan
    windows, _ = build_review_plan(clauses,
        chunk_chars=int(getattr(config, 'llm_chunk_chars', 1800)),
        overlap=int(getattr(config, 'llm_chunk_overlap', 160)),
        cross_pairs=int(getattr(config, 'llm_cross_clause_pairs', 12)))
    if not windows:
        raise ValueError('cannot reconstruct a feedback schema without a source window')
    clause_sources = {}
    for segment in windows[0].segments:
        clause_sources.setdefault(segment.clause_id, []).append(segment.text)
    return N._feedback_json_schema(items, subject_fact=kind == 'subject', clause_sources=clause_sources)


def evaluate_replay(meta, calls, items, kind, expected_schema):
    """Evaluate one batch's protocol evidence, never its legal correctness."""
    expected_hash = digest(json.dumps(expected_schema, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False))
    llm_calls = [call for call in calls if call.get('component') == 'llm']
    started = [call for call in llm_calls if call.get('status') == 'started']
    terminals = [call for call in llm_calls if call.get('status') != 'started']
    def details(call):
        try:
            value = json.loads(call.get('details', '{}'))
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}
    paired = (len(started) == 1 and len(terminals) == 1
        and isinstance(details(started[0]).get('call_id'), str) and bool(details(started[0])['call_id'].strip())
        and details(started[0]).get('call_id') == details(terminals[0]).get('call_id'))
    identity_verified = (len(terminals) == 1 and details(terminals[0]).get('model_identity_verified') is True
        and details(terminals[0]).get('actual_model') == 'legalone-8b')
    stopped = (len(terminals) == 1 and terminals[0].get('status') == 'succeeded'
        and details(terminals[0]).get('finish_reason') == 'stop' and identity_verified)
    sent_schema = paired and all(details(call).get('native_json_schema_sent') is True for call in llm_calls)
    hash_verified = sent_schema and all(details(call).get('json_schema_sha256') == expected_hash for call in llm_calls)
    item_ids = [item['feedback_item_id'] for item in items]
    rows_key = 'subject_fact_verifications' if kind == 'subject' else 'verifications'
    returned = [row for call in meta.get('calls', []) for row in call.get(rows_key, [])]
    protocol_valid = stopped and sorted(row.get('feedback_item_id', '') for row in returned) == sorted(item_ids)
    raw_pending = meta.get('uncompleted_feedback_item_ids')
    pending_ids = raw_pending if isinstance(raw_pending, list) else []
    raw_verified = meta.get('verified_feedback_item_ids')
    verified_ids = raw_verified if isinstance(raw_verified, list) else []
    pending = bool(pending_ids)
    required_pending = set()
    for item in items:
        if kind == 'subject':
            unresolved = (item.get('source_complete') is not True or not item.get('source_id')
                or item.get('effective_status') in (None, '', 'unknown', 'conflicting_evidence')
                or item.get('upstream_pending') or item.get('fact_evidence_text_truncated')
                or item.get('source_evidence_text_truncated'))
        else:
            unresolved = bool(item.get('article_quote_truncated'))
        if unresolved:
            required_pending.add(item['feedback_item_id'])
    required_pending.update(row.get('feedback_item_id') for row in returned if row.get('verdict') == 'needs_manual')
    pending_preserved = required_pending.issubset(pending_ids)
    accounting_valid = (isinstance(raw_pending, list) and len(set(pending_ids)) == len(pending_ids)
        and set(pending_ids).issubset(item_ids) and meta.get('feedback_complete') is (not pending)
        and isinstance(raw_verified, list) and len(set(verified_ids)) == len(verified_ids)
        and set(verified_ids) == set(item_ids) - set(pending_ids))
    passed = protocol_valid and sent_schema and hash_verified and pending_preserved and accounting_valid
    return {'passed':passed, 'pending':pending, 'protocol_bindings_and_quotes_valid':protocol_valid,
        'native_schema_sent':sent_schema, 'expected_native_schema_sha256':expected_hash,
        'native_schema_sha256_verified':hash_verified, 'single_call_audit_verified':paired,
        'completion_model_identity_verified':identity_verified,
        'required_pending_feedback_item_ids':sorted(required_pending),
        'unresolved_subject_items_preserved':pending_preserved, 'pending_accounting_consistent':accounting_valid,
        'model_response_not_truncated':stopped, 'full_http_revalidated':False,
        'legal_truth_independently_evaluated':False, 'accuracy_evaluated':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=['statute', 'subject'], required=True)
    parser.add_argument('--run-index', type=int, default=0)
    parser.add_argument('--job-id')
    parser.add_argument('--prepare-only', action='store_true', help='restore and verify evidence without a model call')
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.out_dir.resolve()
    allowed = ROOT / 'tools/acceptance_results'
    if not source.is_relative_to(allowed) or not output.is_relative_to(allowed):
        parser.error('evidence directories must remain under tools/acceptance_results')
    result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    if result.get('synthetic_input') is not True or not result.get('http_archive_roundtrip_verified'):
        raise ValueError('requires a completed archived, explicitly synthetic source run')
    report_bytes = (source / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    contract, clauses, failed, items = restore_batch(report, args.kind,
        run_index=args.run_index, job_id=args.job_id)
    item_ids = failed['feedback_item_ids']
    output.mkdir(parents=True, exist_ok=False)
    os.environ['ZLQL_MODEL_AUDIT_PATH'] = str(output / 'model_calls.sqlite')
    from zhilv_qiling_v2 import nodes as N
    from zhilv_qiling_v2.config import Config
    config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:8620/v1', llm_model='legalone-8b',
        llm_native_thinking='off', llm_native_reasoning_format='deepseek', llm_max_tokens=4096,
        llm_timeout=900, llm_cross_clause_pairs=0, llm_max_calls=1, llm_native_json_schema=True)
    expected_schema = expected_replay_schema(config, clauses, items, args.kind)
    evidence = {'status':'running', 'scope':'one_real_failed_batch_replay_not_full_http_or_accuracy',
        'synthetic_input':True, 'source_run':str(source), 'source_report_sha256':digest(report_bytes),
        'original_failure':failed, 'replayed_feedback_item_ids':item_ids,
        'feedback_kind':args.kind,
        'changes':'independent typed prompt and native schema with source-extract clause-quote selection; restored feedback records and clause IDs; single primary window',
        'clause_quote_mode':'source_extract_selection_not_free_form_or_legal_truth_evaluation',
        'accuracy_evaluated':False, 'provenance':provenance(), 'errors':[]}
    write_json(output / 'input.json', {'clauses':[dataclasses.asdict(clause) for clause in clauses], 'feedback':items})
    if args.prepare_only:
        evidence['status'] = 'evidence_restored_without_model_call'
        write_json(output / 'result.json', evidence)
        print(json.dumps({'status':evidence['status'], 'kind':args.kind, 'job_id':failed['job_id']}, ensure_ascii=True))
        return 0
    try:
        served = require(*request('http://127.0.0.1:8620', '/v1/models', timeout=15)[:2])
        identities = [item for item in served.get('data', []) if item.get('id')=='legalone-8b'
                      and item.get('meta', {}).get('n_params')==8195978240]
        if len(identities)!=1:
            raise ValueError('actual native LegalOne-8B parameter identity not verified')
        evidence['actual_model_identity'] = identities[0]
        print('Replaying the archived failed synthetic batch against the real LegalOne-8B.', flush=True)
        options = {'subject_fact_feedback' if args.kind == 'subject' else 'verification_feedback':items}
        _, meta, degraded = N._llm_logic(contract, config, clauses, **options)
        database = sqlite3.connect(output / 'model_calls.sqlite')
        calls = [dict(zip([column[1] for column in database.execute('pragma table_info(model_calls)')], row))
                 for row in database.execute('select * from model_calls order by rowid')]
        database.close()
        assessment = evaluate_replay(meta, calls, items, args.kind, expected_schema)
        passed, pending = assessment.pop('passed'), assessment.pop('pending')
        evidence.update(status=('bound_records_returned_pending' if pending else 'complete_bound_batch_returned') if passed
            else 'output_reliability_not_resolved', feedback=meta, degraded=degraded, audit_events=calls, **assessment)
    except Exception as exc:
        evidence['status']='failed'
        evidence['errors'].append(f'{type(exc).__name__}: {exc}')
    write_json(output / 'result.json', evidence)
    print(json.dumps({'status':evidence['status'], 'errors':evidence['errors']}, ensure_ascii=True))
    return 0 if evidence['status'] in ('complete_bound_batch_returned','bound_records_returned_pending') else 2


if __name__ == '__main__':
    raise SystemExit(main())
