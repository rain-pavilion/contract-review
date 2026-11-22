"""Offline in-memory tests for the unapplied receipt-binding proposal.

No live model, server or product source is changed. Transport is explicitly
mocked; this is engineering verification, not real LegalOne or legal quality.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import zhilv_qiling_v2 as PACKAGE
from zhilv_qiling_v2 import long_context as C
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse
import verify_complex_postrun_v2 as VERIFY

PRODUCTS = [ROOT / 'zhilv_qiling_v2' / name for name in ('llm_client.py', 'nodes.py', 'model_audit.py', 'agents_multi_run.py')]
BEFORE = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PRODUCTS}


def proposed_sources():
    proposal = (ROOT / 'tools/receipt_binding_patch_20261115.patch').read_text(encoding='utf-8')
    result = {}
    for section in proposal.split('--- a/')[1:]:
        lines = section.splitlines(keepends=True)
        relative = lines[0].strip()
        original = (ROOT / relative).read_text(encoding='utf-8').splitlines(keepends=True)
        out, position, active = [], 0, False
        for line in lines[1:]:
            if line.startswith('+++ '): continue
            if line.startswith('@@'):
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
                if not match: raise AssertionError('malformed proposal hunk')
                start = int(match.group(1)) - 1
                if start < position: raise AssertionError('overlapping proposal hunks')
                out.extend(original[position:start]); position = start; active = True
            elif active and line.startswith(' '):
                if original[position] != line[1:]: raise AssertionError('proposal context drift')
                out.append(line[1:]); position += 1
            elif active and line.startswith('-'):
                if original[position] != line[1:]: raise AssertionError('proposal removal drift')
                position += 1
            elif active and line.startswith('+'): out.append(line[1:])
            elif line.startswith('\\ No newline'): continue
            else: raise AssertionError('unexpected proposal line')
        out.extend(original[position:])
        result[Path(relative).name] = ''.join(out)
    if set(result) != {'llm_client.py', 'nodes.py', 'model_audit.py', 'agents_multi_run.py'}:
        raise AssertionError('unexpected proposal files')
    return result


def load_candidate(name, source):
    fullname = 'zhilv_qiling_v2._unapplied_binding_' + name
    module = types.ModuleType(fullname); module.__package__ = 'zhilv_qiling_v2'
    sys.modules[fullname] = module
    exec(compile(source, '<unapplied-receipt-binding-' + name + '>', 'exec'), module.__dict__)
    return module


SOURCES = proposed_sources()
AUDIT = load_candidate('audit', SOURCES['model_audit.py'])
CLIENT = load_candidate('client', SOURCES['llm_client.py'])
NODES = load_candidate('nodes', SOURCES['nodes.py'])
NODES.L = CLIENT
GRAPH = load_candidate('graph', SOURCES['agents_multi_run.py'])
GRAPH.N = NODES


@contextmanager
def runtime():
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(PACKAGE, 'model_audit', AUDIT, create=True), \
         patch.dict(sys.modules, {'zhilv_qiling_v2.model_audit': AUDIT}), \
         patch.dict(os.environ, {'ZLQL_MODEL_AUDIT_PATH': str(Path(temporary) / 'audit.sqlite')}):
        yield Path(temporary) / 'audit.sqlite'


def envelope(content=None, finish='stop', model='legalone-8b'):
    return (json.dumps({'defects': []} if content is None else content, ensure_ascii=False),
            {'model': model, 'choices': [{'finish_reason': finish}]}, 0.01)


def client_config():
    return CLIENT.LLMConfig(endpoint='http://127.0.0.1:1/v1', model='legalone-8b')


def context():
    return {'window_id': 'window-test', 'job_id': 'window-test:batch:1', 'source_sha256': 'a' * 64}


class ReceiptBindingDraftTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for path in PRODUCTS:
            if hashlib.sha256(path.read_bytes()).hexdigest() != BEFORE[str(path)]:
                raise AssertionError('product source changed while candidate tests ran: ' + path.name)

    def test_client_returns_exact_call_id_and_persists_bound_receipts(self):
        with runtime() as db, patch.object(CLIENT, '_complete', return_value=envelope()):
            with AUDIT.review_audit('trace-test', 'contract-test') as collector:
                pair = CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'PRIVATE-DIAGNOSTIC'}],
                                           audit_context=context(), retries=0)
            self.assertTrue(pair.call_id)
            self.assertEqual([row['status'] for row in collector.events], ['started', 'succeeded'])
            self.assertTrue(all(row['details']['call_id'] == pair.call_id for row in collector.events))
            self.assertTrue(all(all(row['details'][key] == value for key, value in context().items())
                                for row in collector.events))
            persisted = VERIFY.load_sqlite_events(db, 'trace-test')
            self.assertEqual([row['event_id'] for row in persisted], [row['event_id'] for row in collector.events])
            self.assertNotIn('PRIVATE-DIAGNOSTIC', json.dumps(persisted))

    def test_optional_context_keeps_other_callers_compatible_without_inventing_window(self):
        with runtime(), patch.object(CLIENT, '_complete', return_value=envelope()):
            with AUDIT.review_audit('trace', 'contract') as collector:
                pair = CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}], retries=0)
            self.assertTrue(pair.call_id)
            self.assertTrue(all('window_id' not in event['details'] for event in collector.events))

    def test_invalid_context_or_overwrite_attempt_is_rejected_before_transport(self):
        bad = [{}, {**context(), 'source_sha256': 'bad'}, {**context(), 'window_id': 'raw\ntext'},
               {**context(), 'job_id': ''}, {**context(), 'call_id': 'forged'},
               {**context(), 'prompt': 'PRIVATE-PROMPT'}, {**context(), 'source_sha256': 5}]
        for value in bad:
            with self.subTest(context=value), runtime(), patch.object(CLIENT, '_complete') as transport:
                with AUDIT.review_audit('trace', 'contract') as collector:
                    with self.assertRaises(CLIENT.LLMClientError):
                        CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}], audit_context=value)
                transport.assert_not_called()
                self.assertEqual(collector.events, [])

    def test_parse_retry_preserves_job_binding_but_returns_final_attempt_id(self):
        bad = ('not json', {'model': 'legalone-8b', 'choices': [{'finish_reason': 'stop'}]}, 0.01)
        with runtime(), patch.object(CLIENT, '_complete', side_effect=[bad, envelope()]):
            with AUDIT.review_audit('trace', 'contract') as collector:
                pair = CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}],
                                           audit_context=context(), retries=1)
            starts = [event for event in collector.events if event['status'] == 'started']
            self.assertEqual(len(starts), 2)
            self.assertNotEqual(starts[0]['details']['call_id'], starts[1]['details']['call_id'])
            self.assertEqual(pair.call_id, starts[-1]['details']['call_id'])
            self.assertEqual(pair.degraded, 'json_retry')
            self.assertTrue(all(all(event['details'][key] == value for key, value in context().items())
                                for event in collector.events))

    def test_transport_failure_retains_binding_without_fabricating_success(self):
        error = CLIENT.LLMClientError('fixture transport failure', status=503)
        with runtime(), patch.object(CLIENT, '_complete', side_effect=error):
            with AUDIT.review_audit('trace', 'contract') as collector:
                with self.assertRaises(CLIENT.LLMClientError):
                    CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}], audit_context=context())
            self.assertEqual([event['status'] for event in collector.events], ['started', 'failed'])
            self.assertTrue(all(event['details']['window_id'] == 'window-test' for event in collector.events))

    def test_truncation_and_wrong_response_identity_remain_visible(self):
        with runtime(), patch.object(CLIENT, '_complete', return_value=envelope(finish='length')):
            with AUDIT.review_audit('trace', 'contract') as collector:
                pair = CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}], audit_context=context())
            self.assertTrue(pair.call_id)
            self.assertEqual(pair.degraded, 'output_truncated')
            self.assertEqual(collector.events[-1]['status'], 'degraded')
        with runtime(), patch.object(CLIENT, '_complete', return_value=envelope(model='wrong-model')):
            with AUDIT.review_audit('trace', 'contract') as collector:
                with self.assertRaises(CLIENT.LLMClientError):
                    CLIENT.request_json(client_config(), [{'role': 'user', 'content': 'x'}], audit_context=context())
            self.assertEqual(collector.events[-1]['reason'], 'model_identity_mismatch')
            self.assertEqual(collector.events[-1]['details']['source_sha256'], context()['source_sha256'])

    def test_nodes_to_client_to_sqlite_bind_actual_sources_and_pass_v2_engineering(self):
        clauses = [Clause('a', '第一条', '', 1, '', '', '按第二条支付价款。' + 'PRIVATE-ALPHA归档材料。' * 25),
                   Clause('b', '第二条', '', 1, '', '', '收到材料后支付价款。' + 'PRIVATE-BETA接收资料。' * 25)]
        contract = Contract('contract-test', DocumentParse('fixture.txt', 'txt', '\n'.join(c.text for c in clauses), clauses=clauses))
        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='legalone-8b',
                        llm_chunk_chars=200, llm_chunk_overlap=0, llm_cross_clause_pairs=100)
        sent_sources = []
        def complete(_config, messages, _max_tokens, **_kwargs):
            payload = json.loads(messages[-1]['content'])
            sent_sources.append(payload['segments'])
            return envelope()
        with runtime() as db, patch.object(CLIENT, '_complete', side_effect=complete):
            with AUDIT.review_audit('trace-test', 'contract-test') as collector:
                _, logic, _ = NODES._llm_logic(contract, config, clauses)
                AUDIT.record('entailment', 'OFFLINE-DIAGNOSTIC', 'predict_pair', 'succeeded')
            windows, _ = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=100)
            prepared = {'contract_id': 'contract-test', 'parsed_text': contract.document.full_text,
                        'clauses': [asdict(c) for c in clauses], 'windows': [w.as_dict() for w in windows]}
            self.assertEqual(len(logic['calls']), len(sent_sources))
            for call, sent in zip(logic['calls'], sent_sources):
                self.assertEqual(call['source_sha256'], VERIFY.source_sha256(sent))
                self.assertEqual(call['source_spans'], [{k: s[k] for k in ('clause_id', 'start', 'end')} for s in sent])
                self.assertTrue(call['call_id'])
            logic.update(feedback_runs=[], feedback_latest_runs=[], feedback_runs_complete=True,
                feedback_run_bindings=[], feedback_attempts=[], feedback_current_request={
                    'required': False, 'fingerprint': None, 'attempt_id': None, 'attempt_index': None,
                    'items': [], 'focus_clause_ids': [], 'pending_subject_request_ids': [],
                    'subject_verification_complete': None})
            report = {'trace_id': 'trace-test', 'document': {'full_text': contract.document.full_text}, 'stats': {
                'logic_llm': logic, 'model_calls': collector.events, 'degraded': [], 'cross_validation': {
                    'termination': 'stable', 'converged': True, 'feedback_verification_complete': True,
                    'pending_feedback_item_ids': [], 'pending_subject_verification_request_ids': [],
                    'pending_feedback_fingerprints': []}}}
            events = VERIFY.load_sqlite_events(db, 'trace-test')
            result = VERIFY.check(prepared, report, sqlite_events=events)
            self.assertTrue(result['strict_engineering_execution_verified'], result['checks'])
            self.assertTrue(result['effective_model_review_complete'])
            persisted = json.dumps(events)
            self.assertNotIn('PRIVATE-ALPHA', persisted); self.assertNotIn('PRIVATE-BETA', persisted)
            self.assertNotIn('source_spans', persisted)

    def test_nodes_keep_receipt_id_on_partial_output_without_claiming_coverage(self):
        clause = Clause('a', '第一条', '', 1, '', '', '本合同尾款支付需按约履行。')
        contract = Contract('contract-test', DocumentParse('fixture.txt', 'txt', clause.text, clauses=[clause]))
        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='legalone-8b', llm_cross_clause_pairs=0)
        with runtime(), patch.object(CLIENT, '_complete', return_value=envelope(finish='length')):
            with AUDIT.review_audit('trace-test', 'contract-test') as collector:
                _, logic, _ = NODES._llm_logic(contract, config, [clause])
            call = logic['calls'][0]
            self.assertEqual(call['status'], 'partial')
            self.assertEqual(call['call_id'], collector.events[-1]['details']['call_id'])
            self.assertFalse(logic['complete'])

    def test_feedback_jobs_also_preserve_source_binding_and_auditable_receipts(self):
        clause = Clause('a', '第一条', '', 1, '', '', '买方应在验收后十日内支付价款。')
        contract = Contract('contract-test', DocumentParse('fixture.txt', 'txt', clause.text, clauses=[clause]))
        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='legalone-8b', llm_cross_clause_pairs=0)
        item = {'feedback_item_id': 'fb-test', 'candidate_id': 'candidate-test', 'article_uid': 'law-test',
                'clause_ids': ['a'], 'article_quote': '合成规范：按约支付。'}
        def complete(_config, messages, _max_tokens, **_kwargs):
            payload = json.loads(messages[-1]['content'])
            value = payload['verification_feedback'][0]
            content = {'verifications': [{'feedback_item_id': value['feedback_item_id'],
                'candidate_id': value['candidate_id'], 'article_uid': value['article_uid'], 'verdict': 'confirmed',
                'reason': '仅核对离线诊断协议。', 'clause_quote': clause.text, 'article_quote': value['article_quote']}]}
            return envelope(content)
        with runtime() as db, patch.object(CLIENT, '_complete', side_effect=complete):
            with AUDIT.review_audit('trace-test', 'contract-test') as collector:
                _, logic, _ = NODES._llm_logic(contract, config, [clause], verification_feedback=[item])
            call = logic['calls'][0]
            self.assertEqual(call['status'], 'ok')
            self.assertTrue(logic['feedback_complete'])
            starts = {row['details']['call_id']: row for row in collector.events if row['status'] == 'started'}
            terminals = {row['details']['call_id']: row for row in collector.events if row['status'] != 'started'}
            checked = VERIFY._binding_check(call, starts, terminals, {'a': clause})
            self.assertTrue(checked['verified'], checked)
            self.assertEqual(len(VERIFY.load_sqlite_events(db, 'trace-test')), 2)

    def test_graph_exports_explicit_current_attempt_and_input_manifest_without_rewriting_runs(self):
        module = importlib.import_module('zhilv_qiling_v2.tests_v2.test_feedback_retry')
        with patch.object(module, 'N', NODES), patch.object(module, 'G', GRAPH):
            helper = module.FeedbackRetryTests()
            failed = {'complete': False, 'feedback_complete': False, 'uncompleted_feedback_item_ids': ['fb1']}
            success = {'complete': True, 'feedback_complete': True, 'uncompleted_feedback_item_ids': []}
            run, _ = helper.run_loop([failed, success])
        logic = run.diagnostics['logic_llm']
        self.assertEqual(logic['feedback_runs'], [failed, success])
        self.assertEqual(logic['feedback_latest_runs'], [success])
        bindings = logic['feedback_run_bindings']
        self.assertEqual([row['attempt_index'] for row in bindings], [1, 2])
        self.assertNotEqual(bindings[0]['attempt_id'], bindings[1]['attempt_id'])
        self.assertEqual(bindings[1]['run_sha256'], VERIFY.source_sha256(success))
        current = logic['feedback_current_request']
        self.assertEqual(current['attempt_id'], bindings[1]['attempt_id'])
        self.assertEqual(current['fingerprint'], bindings[1]['fingerprint'])
        self.assertTrue(current['items'])

    def test_existing_long_context_and_feedback_source_regressions_stay_compatible(self):
        targets = [('zhilv_qiling_v2.tests_v2.test_core_completion', 'LongContractTest'),
                   ('zhilv_qiling_v2.tests_v2.test_feedback_grounding', 'FeedbackGroundingTests')]
        for module_name, class_name in targets:
            module = importlib.import_module(module_name)
            old = module.N
            try:
                module.N = NODES
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, class_name))
                result = unittest.TestResult(); suite.run(result)
                self.assertEqual(result.errors, [], str(result.errors))
                self.assertEqual(result.failures, [], str(result.failures))
            finally:
                module.N = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
