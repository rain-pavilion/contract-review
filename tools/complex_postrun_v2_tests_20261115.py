"""Offline adversarial tests for v2 postrun; no live model or DB is touched."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import verify_complex_postrun as V1
import verify_complex_postrun_v2 as V2
from zhilv_qiling_v2 import long_context as C
from zhilv_qiling_v2.models import Clause


def fixture():
    clauses = [Clause('a', '第一条', '', 1, '', '', '按第二条支付价款。' + '资料归档。' * 50),
               Clause('b', '第二条', '', 1, '', '', '付款应按约履行。' + '材料接收。' * 50)]
    windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=100)
    coverage = C.coverage(clauses, windows, {w.window_id for w in windows})
    prepared = {'contract_id': 'contract-test', 'parsed_text': '\n'.join(c.text for c in clauses),
                'clauses': [asdict(c) for c in clauses], 'windows': [w.as_dict() for w in windows]}
    calls, events = [], []
    for index, window in enumerate(windows):
        call_id = 'request-' + str(index)
        source = [s.as_dict() for s in window.segments]
        call = {'window_id': window.window_id, 'job_id': window.window_id + ':batch:1',
                'kind': window.kind, 'batch_index': 1, 'batch_count': 1, 'feedback_item_ids': [],
                'status': 'ok', 'model': 'legalone-8b', 'call_id': call_id,
                'source_spans': [{k: s[k] for k in ('clause_id', 'start', 'end')} for s in source],
                'source_sha256': V2.source_sha256(source)}
        calls.append(call)
        details = {key: call[key] for key in ('window_id', 'job_id', 'source_sha256')}
        details.update(call_id=call_id, requested_model='legalone-8b', attempt=1)
        base = {'trace_id': 'trace-test', 'contract_id': 'contract-test', 'component': 'llm',
                'model': 'legalone-8b', 'operation': 'request_json', 'elapsed_seconds': 0, 'reason': ''}
        events.append({**base, 'event_id': call_id + '-start', 'created_at': '2026-11-15T00:00:00+00:00',
                       'status': 'started', 'details': details})
        events.append({**base, 'event_id': call_id + '-end', 'created_at': '2026-11-15T00:00:01+00:00',
                       'status': 'succeeded', 'details': {**details, 'actual_model': 'legalone-8b',
                           'model_identity_verified': True, 'finish_reason': 'stop'}})
    events.append({'event_id': 'deep-test', 'created_at': '2026-11-15T00:00:02+00:00',
                   'trace_id': 'trace-test', 'contract_id': 'contract-test', 'component': 'entailment',
                   'model': 'bert-mamba-test', 'operation': 'predict_pair', 'status': 'succeeded',
                   'elapsed_seconds': 0, 'reason': '', 'details': {}})
    logic = {**plan, **coverage, 'calls': calls, 'planned_calls': len(windows),
             'completed_calls': len(windows), 'cross_windows_completed': plan['cross_windows'],
             'feedback_complete': True, 'uncompleted_feedback_item_ids': [],
             'unmatched_feedback_item_ids': [], 'feedback_runs': [], 'feedback_latest_runs': [],
             'feedback_runs_complete': True, 'feedback_run_bindings': [], 'feedback_attempts': [],
             'feedback_current_request': {'required': False, 'fingerprint': None, 'attempt_id': None,
                 'attempt_index': None, 'items': [], 'focus_clause_ids': [],
                 'pending_subject_request_ids': [], 'subject_verification_complete': None}}
    report = {'trace_id': 'trace-test', 'document': {'full_text': prepared['parsed_text']},
              'stats': {'logic_llm': logic, 'model_calls': events, 'degraded': [],
                        'cross_validation': {'termination': 'stable', 'converged': True,
                            'feedback_verification_complete': True, 'pending_feedback_item_ids': [],
                            'pending_subject_verification_request_ids': [], 'pending_feedback_fingerprints': []}}}
    return prepared, report


def check(prepared, report, events=None):
    return V2.check(prepared, report, sqlite_events=deepcopy(report['stats']['model_calls']) if events is None else events)



def add_feedback_run(prepared, report, *, suffix='one', fingerprint='f'*64):
    """Two current clause-b windows require a separate acknowledgement in each job."""
    logic = report['stats']['logic_llm']
    item = {'feedback_item_id': 'fb-required', 'feedback_kind': 'statute',
            'candidate_id': 'candidate-required', 'article_uid': 'law-required',
            'clause_ids': ['b'], 'article_quote': '合成规范：依约支付价款。'}
    templates = [call for call in logic['calls'] if call['kind'] == 'text'
                 and {part['clause_id'] for part in call['source_spans']} == {'b'}]
    cmap = {row['clause_id']: Clause(**row) for row in prepared['clauses']}
    calls, window_plans, windows = [], [], []
    original_events = deepcopy(report['stats']['model_calls'][:2])
    for index, template in enumerate(templates):
        call = deepcopy(template)
        call.update(call_id='feedback-' + suffix + '-' + str(index), feedback_item_ids=['fb-required'],
            verified_feedback_item_ids=['fb-required'], uncompleted_feedback_item_ids=[], verification_errors=[])
        source = V2._source_from_spans(call['source_spans'], cmap)
        call['verifications'] = [{'feedback_item_id': 'fb-required', 'candidate_id': 'candidate-required',
            'article_uid': 'law-required', 'verdict': 'confirmed', 'reason': '离线逐项协议诊断。',
            'clause_quote': source[0]['text'][:20], 'article_quote': item['article_quote']}]
        calls.append(call)
        batch = {key: call[key] for key in ('job_id', 'batch_index', 'batch_count', 'feedback_item_ids', 'status')}
        window_plans.append({'window_id': call['window_id'], 'kind': call['kind'], 'batch_count': 1,
                             'batches': [batch], 'oversize_feedback_item_ids': []})
        windows.append(C.ReviewWindow(call['window_id'], call['kind'], tuple(C.Segment(**part) for part in source)))
        for number, original in enumerate(original_events):
            event = deepcopy(original)
            event['event_id'] = call['call_id'] + '-event-' + str(number)
            event['details'].update({key: call[key] for key in ('call_id', 'window_id', 'job_id', 'source_sha256')})
            report['stats']['model_calls'].append(event)
    covered = C.coverage([cmap['b']], windows, {window.window_id for window in windows})
    run = {**covered, 'calls': calls, 'feedback_items': [item], 'feedback_windows': window_plans,
        'planned_calls': len(calls), 'completed_calls': len(calls), 'cross_windows': 0,
        'cross_windows_completed': 0, 'feedback_complete': True, 'uncompleted_feedback_item_ids': [],
        'unmatched_feedback_item_ids': [], 'verified_feedback_item_ids': ['fb-required']}
    history = logic.setdefault('feedback_runs', [])
    history.append(run)
    index = len(history)
    binding = {'attempt_id': 'attempt-' + suffix, 'attempt_index': index, 'fingerprint': fingerprint,
               'round': index+1, 'run_sha256': V2.source_sha256(run)}
    logic.setdefault('feedback_run_bindings', []).append(binding)
    logic.setdefault('feedback_attempts', []).append({'attempt_id': binding['attempt_id'], 'attempt_index': index,
        'fingerprint': fingerprint, 'round': index+1, 'complete': True})
    logic['feedback_latest_runs'] = [run]
    logic['feedback_current_request'] = {'required': True, 'fingerprint': fingerprint, 'attempt_id': binding['attempt_id'],
        'attempt_index': index, 'items': [item], 'focus_clause_ids': ['b'],
        'pending_subject_request_ids': [], 'subject_verification_complete': None}
    return run, binding


class StrictV2PostrunTests(unittest.TestCase):
    def test_fully_bound_positive_control(self):
        prepared, report = fixture()
        result = check(prepared, report)
        self.assertTrue(result['strict_engineering_execution_verified'], result['checks'])
        self.assertTrue(result['effective_model_review_complete'])
        self.assertFalse(result['legal_correctness_evaluated'])
        self.assertFalse(result['deep_pair_summary']['complete_expected_pair_coverage_proven'])

    def test_initial_text_call_missing_cannot_be_hidden_by_success_receipts(self):
        prepared, report = fixture()
        calls = report['stats']['logic_llm']['calls']
        calls.pop(next(i for i, c in enumerate(calls) if c['kind'] == 'text'))
        self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])
        self.assertFalse(V1.check(prepared, report)['strict_engineering_execution_verified'])

    def test_cross_window_failed_is_rejected_even_if_text_calls_succeeded(self):
        prepared, report = fixture()
        cross = next(c for c in report['stats']['logic_llm']['calls'] if c['kind'] == 'cross_clause')
        cross['status'] = 'error'
        result = check(prepared, report)
        self.assertFalse(result['strict_engineering_execution_verified'])
        self.assertFalse(result['checks']['every_prepared_initial_window_completed'])
        self.assertFalse(V1.check(prepared, report)['strict_engineering_execution_verified'])

    def test_partial_cross_window_is_not_ok(self):
        prepared, report = fixture()
        next(c for c in report['stats']['logic_llm']['calls'] if c['kind'] == 'cross_clause')['status'] = 'partial'
        self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_duplicate_and_unknown_window_ids_are_rejected(self):
        for mutation in ('duplicate', 'unknown'):
            prepared, report = fixture()
            calls = report['stats']['logic_llm']['calls']
            calls[0]['window_id'] = calls[1]['window_id'] if mutation == 'duplicate' else 'unknown-window'
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_pending_fake_stable_and_convergence_aliases_fail_closed(self):
        for word in ('stable', 'converged', 'convergence', 'completed', 'fixed_point', '稳定', '收敛', 'SUCCESS'):
            with self.subTest(termination=word):
                prepared, report = fixture()
                cross = report['stats']['cross_validation']
                cross.update(termination=word, converged=False, feedback_verification_complete=False,
                             pending_feedback_item_ids=['pending'])
                result = check(prepared, report)
                self.assertFalse(result['strict_engineering_execution_verified'])
                self.assertFalse(result['effective_model_review_complete'])

    def test_canonical_round_limit_with_truthful_pending_is_never_effectively_complete(self):
        prepared, report = fixture()
        report['stats']['cross_validation'].update(termination='round_limit', converged=False,
            feedback_verification_complete=False, pending_feedback_item_ids=['pending'])
        report['stats']['logic_llm']['feedback_runs_complete'] = False
        report['stats']['logic_llm']['feedback_current_request'].update(required=True, fingerprint='current-state',
            subject_verification_complete=False)
        result = check(prepared, report)
        self.assertTrue(result['strict_engineering_execution_verified'], result['checks'])
        self.assertFalse(result['effective_model_review_complete'])
        # Reproduces the prior tool's independent false-positive effective flag.
        legacy_false_claim = deepcopy(report)
        legacy_false_claim['stats']['logic_llm']['feedback_runs_complete'] = True
        self.assertTrue(V1.check(prepared, legacy_false_claim)['effective_model_review_complete'])

    def test_missing_or_inconsistent_pending_ledgers_fail(self):
        for mutation in ('missing', 'wrong_type', 'claims_complete', 'fingerprint_only'):
            with self.subTest(mutation=mutation):
                prepared, report = fixture()
                cross = report['stats']['cross_validation']
                if mutation == 'missing':
                    del cross['pending_feedback_item_ids']
                elif mutation == 'wrong_type':
                    cross['pending_subject_verification_request_ids'] = 'none'
                elif mutation == 'claims_complete':
                    cross.update(termination='round_limit', converged=False, pending_feedback_item_ids=['pending'])
                else:
                    cross['pending_feedback_fingerprints'] = ['unconsumed-state']
                self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_legacy_equal_counts_without_explicit_bindings_are_unproven(self):
        prepared, report = fixture()
        for call in report['stats']['logic_llm']['calls']:
            for key in ('call_id', 'source_sha256', 'source_spans'):
                del call[key]
        result = check(prepared, report)
        self.assertFalse(result['strict_engineering_execution_verified'])
        self.assertEqual(result['receipt_bindings']['status'], 'unproven_or_mismatched')
        self.assertFalse(result['receipt_bindings']['legacy_order_or_timing_reconstruction_used'])
        self.assertTrue(V1.check(prepared, report)['strict_engineering_execution_verified'])

    def test_one_receipt_pair_cannot_stand_for_every_initial_window(self):
        prepared, report = fixture()
        events = report['stats']['model_calls']
        report['stats']['model_calls'] = events[:2] + [events[-1]]
        self.assertTrue(V1.check(prepared, report)['strict_engineering_execution_verified'])
        self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_reused_call_id_or_wrong_job_binding_is_rejected(self):
        for mutation in ('reuse', 'wrong_job', 'missing_context'):
            prepared, report = fixture()
            calls = report['stats']['logic_llm']['calls']
            if mutation == 'reuse':
                calls[1]['call_id'] = calls[0]['call_id']
            elif mutation == 'wrong_job':
                report['stats']['model_calls'][0]['details']['job_id'] = 'wrong-job'
            else:
                del report['stats']['model_calls'][0]['details']['window_id']
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_source_digest_and_exact_initial_prepared_ranges_are_enforced(self):
        for mutation in ('digest', 'range', 'prepared_text'):
            prepared, report = fixture()
            call = report['stats']['logic_llm']['calls'][0]
            if mutation == 'digest':
                call['source_sha256'] = '0' * 64
            elif mutation == 'range':
                call['source_spans'][0]['end'] -= 1
                cmap = {c['clause_id']: Clause(**c) for c in prepared['clauses']}
                value = V2.source_sha256(V2._source_from_spans(call['source_spans'], cmap))
                call['source_sha256'] = value
                for event in report['stats']['model_calls'][:2]:
                    event['details']['source_sha256'] = value
            else:
                prepared['windows'][0]['segments'][0]['text'] = 'fabricated'
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_missing_extra_or_changed_sqlite_events_fail(self):
        for mutation in ('missing', 'extra', 'changed', 'not_supplied'):
            prepared, report = fixture()
            db = deepcopy(report['stats']['model_calls'])
            if mutation == 'missing':
                db.pop()
            elif mutation == 'extra':
                db.append({**db[-1], 'event_id': 'extra-event'})
            elif mutation == 'changed':
                db[0]['details']['call_id'] = 'different-id'
            else:
                db = None
            result = V2.check(prepared, report, sqlite_events=db)
            self.assertFalse(result['strict_engineering_execution_verified'])
            self.assertFalse(result['sqlite_crosscheck']['exact_trace_event_match'])

    def test_identity_is_exact_true_and_truncated_success_cannot_pass(self):
        for mutation in ('identity_false', 'identity_string', 'wrong_model', 'truncated', 'degraded'):
            prepared, report = fixture()
            terminal = report['stats']['model_calls'][1]
            if mutation == 'identity_false': terminal['details']['model_identity_verified'] = False
            elif mutation == 'identity_string': terminal['details']['model_identity_verified'] = 'true'
            elif mutation == 'wrong_model': terminal['details']['actual_model'] = 'another-model'
            elif mutation == 'truncated': terminal['details']['finish_reason'] = 'length'
            else: terminal['status'] = 'degraded'
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_foreign_trace_or_contract_does_not_bind_to_this_run(self):
        for key in ('trace_id', 'contract_id'):
            prepared, report = fixture()
            report['stats']['model_calls'][0][key] = 'foreign'
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_planned_count_and_batch_identity_are_strict(self):
        for mutation in ('count', 'bool_count', 'batch_count', 'job_id'):
            prepared, report = fixture()
            logic = report['stats']['logic_llm']
            if mutation == 'count': logic['planned_calls'] -= 1
            elif mutation == 'bool_count': logic['planned_calls'] = True
            elif mutation == 'batch_count': logic['calls'][0]['batch_count'] = 2
            else: logic['calls'][0]['job_id'] = 'fabricated-job'
            self.assertFalse(check(prepared, report)['strict_engineering_execution_verified'])

    def test_feedback_latest_cannot_claim_completion_if_missing_from_history(self):
        prepared, report = fixture()
        report['stats']['logic_llm']['feedback_latest_runs'] = [{'complete': True}]
        result = check(prepared, report)
        self.assertFalse(result['effective_model_review_complete'])

    def test_deep_degradation_counts_are_preserved_without_claiming_full_coverage(self):
        prepared, report = fixture()
        report['stats']['model_calls'].append({**report['stats']['model_calls'][-1],
            'event_id': 'deep-degraded', 'status': 'degraded', 'reason': 'unknown-label'})
        result = check(prepared, report)
        self.assertTrue(result['strict_engineering_execution_verified'])
        self.assertEqual(result['deep_pair_summary']['status_counts']['degraded'], 1)
        self.assertFalse(result['deep_pair_summary']['complete_expected_pair_coverage_proven'])

    def test_feedback_items_cannot_be_hidden_by_clearing_focus_and_required_flag(self):
        prepared, report = fixture()
        report['stats']['logic_llm']['feedback_current_request']['items'] = [{
            'feedback_item_id': 'unconsumed', 'feedback_kind': 'subject_fact',
            'clause_ids': ['b'], 'fact_id': 'fact-1'}]
        result = check(prepared, report)
        self.assertFalse(result['strict_engineering_execution_verified'])
        self.assertFalse(result['effective_model_review_complete'])
        self.assertIn('current_feedback_items_have_no_bound_focus',
            result['feedback_completion']['independent_current_request']['issues'])

    def test_current_feedback_positive_control_requires_every_job_and_full_focus(self):
        prepared, report = fixture()
        run, _ = add_feedback_run(prepared, report)
        self.assertGreater(len(run['calls']), 1)
        result = check(prepared, report)
        self.assertTrue(result['strict_engineering_execution_verified'], result['checks'])
        self.assertTrue(result['effective_model_review_complete'], result['feedback_completion'])

    def test_feedback_run_summary_cannot_hide_zero_calls_or_missing_one_window_ack(self):
        for mutation in ('zero_calls', 'one_window_ack', 'call_pending', 'wrong_source_scope'):
            with self.subTest(mutation=mutation):
                prepared, report = fixture()
                run, binding = add_feedback_run(prepared, report)
                if mutation == 'zero_calls':
                    run['calls'] = []
                elif mutation == 'one_window_ack':
                    run['calls'][0]['verified_feedback_item_ids'] = []
                    run['calls'][0]['uncompleted_feedback_item_ids'] = ['fb-required']
                elif mutation == 'call_pending':
                    run['calls'][0]['status'] = 'feedback_incomplete'
                else:
                    run['feedback_items'][0]['clause_ids'] = ['a']
                # Keep producer snapshot hashes correct so semantic checks,
                # rather than a stale-hash shortcut, must reject the mutation.
                binding['run_sha256'] = V2.source_sha256(run)
                result = check(prepared, report)
                self.assertFalse(result['strict_engineering_execution_verified'])
                self.assertFalse(result['effective_model_review_complete'])

    def test_feedback_truncation_missing_reason_and_unknown_extra_rows_cannot_complete(self):
        for mutation in ('article_quote_truncated', 'fact_evidence_text_truncated',
                         'source_evidence_text_truncated', 'empty_reason', 'extra_response'):
            with self.subTest(mutation=mutation):
                prepared, report = fixture()
                run, binding = add_feedback_run(prepared, report)
                if mutation.endswith('_truncated'):
                    run['feedback_items'][0][mutation] = True
                elif mutation == 'empty_reason':
                    run['calls'][0]['verifications'][0]['reason'] = '   '
                else:
                    run['calls'][0]['verifications'].append({'feedback_item_id': 'foreign-item'})
                binding['run_sha256'] = V2.source_sha256(run)
                result = check(prepared, report)
                self.assertFalse(result['strict_engineering_execution_verified'])
                self.assertFalse(result['effective_model_review_complete'])

    def test_upstream_semantic_limits_block_effective_but_preserve_valid_execution_and_protocol(self):
        limits = ({'input_truncated': True}, {'coverage_complete': False},
                  {'manual_required': True}, {'disagreement': True})
        for location in ('current_feedback', 'statute_checks', 'current_consumption'):
            for limit in limits:
                with self.subTest(location=location, limit=limit):
                    prepared, report = fixture()
                    run, binding = add_feedback_run(prepared, report)
                    if location == 'current_feedback':
                        run['feedback_items'][0].update(limit)
                        binding['run_sha256'] = V2.source_sha256(run)
                    elif location == 'statute_checks':
                        report['stats']['statute_checks'] = [{'candidate_id': 'candidate-required',
                            'article_uid': 'law-required', **limit}]
                    else:
                        report['stats']['logic_llm']['feedback_consumption_current'] = [{
                            'feedback_item_id': 'fb-required', 'protocol_complete': True, **limit}]
                    result = check(prepared, report)
                    self.assertTrue(result['strict_engineering_execution_verified'], result['checks'])
                    self.assertTrue(result['feedback_completion']['final_protocol_feedback_complete'])
                    self.assertFalse(result['effective_model_review_complete'])
                    self.assertTrue(result['upstream_semantic_review']['pending_checks'])

    def test_explicit_clear_upstream_flags_keep_positive_control_complete(self):
        prepared, report = fixture()
        run, binding = add_feedback_run(prepared, report)
        clear = {'input_truncated': False, 'coverage_complete': True, 'manual_required': False, 'disagreement': False}
        run['feedback_items'][0].update(clear)
        binding['run_sha256'] = V2.source_sha256(run)
        report['stats']['statute_checks'] = [dict(clear)]
        result = check(prepared, report)
        self.assertTrue(result['strict_engineering_execution_verified'])
        self.assertTrue(result['effective_model_review_complete'])
        self.assertTrue(result['upstream_semantic_review']['clear_of_reported_limits'])

    def test_latest_cannot_select_an_older_complete_attempt_for_same_fingerprint(self):
        prepared, report = fixture()
        older, first = add_feedback_run(prepared, report, suffix='older')
        newer, second = add_feedback_run(prepared, report, suffix='newer')
        for call in newer['calls']:
            call.update(status='feedback_incomplete', verified_feedback_item_ids=[],
                        uncompleted_feedback_item_ids=['fb-required'])
        newer.update(complete=False, feedback_complete=False, completed_calls=0,
                     verified_feedback_item_ids=[], uncompleted_feedback_item_ids=['fb-required'])
        second['run_sha256'] = V2.source_sha256(newer)
        logic = report['stats']['logic_llm']
        logic['feedback_attempts'][-1]['complete'] = False
        logic['feedback_latest_runs'] = [older]
        logic['feedback_current_request'].update(attempt_id=first['attempt_id'], attempt_index=first['attempt_index'])
        result = check(prepared, report)
        self.assertFalse(result['strict_engineering_execution_verified'])
        self.assertFalse(result['effective_model_review_complete'])
        self.assertIn('current_feedback_does_not_select_latest_explicit_attempt',
                      result['feedback_completion']['independent_current_request']['issues'])

    def test_audit_declared_window_job_cannot_disappear_as_an_auxiliary_request(self):
        prepared, report = fixture()
        for index, original in enumerate(deepcopy(report['stats']['model_calls'][:2])):
            original['event_id'] = 'extra-window-event-' + str(index)
            original['details']['call_id'] = 'extra-window-call'
            report['stats']['model_calls'].append(original)
        result = check(prepared, report)
        self.assertFalse(result['strict_engineering_execution_verified'])
        self.assertFalse(result['checks']['all_audit_declared_window_job_requests_have_call_records'])

    def test_sqlite_loader_reads_only_exact_trace_without_creating_missing_db(self):
        prepared, report = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'audit.sqlite'
            with self.assertRaises(ValueError): V2.load_sqlite_events(path, 'trace-test')
            self.assertFalse(path.exists())
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE model_calls(event_id TEXT, created_at TEXT, trace_id TEXT, contract_id TEXT, component TEXT, model TEXT, operation TEXT, status TEXT, elapsed_seconds REAL, reason TEXT, details TEXT)')
                for row in report['stats']['model_calls']:
                    values = [row[key] if key != 'details' else json.dumps(row[key]) for key in V2.EVENT_FIELDS]
                    db.execute('INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)', values)
            db.close()
            before = path.read_bytes()
            events = V2.load_sqlite_events(path, 'trace-test')
            self.assertEqual(len(events), len(report['stats']['model_calls']))
            self.assertTrue(check(prepared, report, events)['strict_engineering_execution_verified'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(V2.load_sqlite_events(path, 'foreign-trace'), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
