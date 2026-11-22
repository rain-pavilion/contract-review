"""Offline receipts for passive v3 capture; real network/model imports blocked."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'llama_cpp'}:
            raise AssertionError('real model imports forbidden in offline validation')


sys.meta_path.insert(0, NoModels())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_complex_core_observed_v3 as observed
from zhilv_qiling_v2 import llm_client as client

SOURCE = 'a' * 64
BINDING = {'window_id': 'window.offline.1', 'job_id': 'job.offline.1', 'source_sha256': SOURCE}
SCHEMA = {'type': 'object', 'required': ['risks'], 'properties': {'risks': {'type': 'array'}},
          'additionalProperties': False}


def reply(content='{"risks": []}', finish='stop'):
    return json.dumps({'model': 'legalone-8b', 'choices': [{'message': {'content': content},
        'finish_reason': finish}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}},
        ensure_ascii=False).encode('utf-8')


class Reply:
    def __init__(self, raw):
        self.raw = raw
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return self.raw


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='passive_observed_v3_')
        self.directory = Path(self.temporary.name)
        self.stack = __import__('contextlib').ExitStack()
        self.stack.enter_context(mock.patch.dict('os.environ', {
            'ZLQL_MODEL_AUDIT_PATH': str(self.directory / 'model_calls.sqlite')}, clear=False))
        self.stack.enter_context(mock.patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(mock.patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')))
        self.config = client.LLMConfig(endpoint='http://127.0.0.1:1/v1', model='legalone-8b',
            api_key='OFFLINE_FAKE_SECRET_NEVER_CAPTURE', extra_headers={'X-Private': 'OFFLINE_HEADER_NEVER_CAPTURE'},
            native_json_schema=True, max_tokens=256)
        self.messages = [{'role': 'user', 'content': '合成离线诊断原文。'}]

    def tearDown(self):
        if sys.getprofile() is not None:
            sys.setprofile(None)
        if threading.getprofile() is not None:
            threading.setprofile(None)
        self.stack.close()
        self.temporary.cleanup()

    def capture(self, name='raw_captures', complete=None, request=None):
        return observed.PassiveCapture(self.directory / name,
            (complete or client._complete).__code__, (request or client.request_json).__code__)

    def invoke(self, capture, raw=None, binding=BINDING, **kwargs):
        with mock.patch.object(client.urllib.request, 'urlopen', return_value=Reply(raw or reply())):
            capture.start()
            try:
                return client.request_json(self.config, self.messages, json_schema=SCHEMA,
                                           audit_context=binding, **kwargs)
            finally:
                capture.stop()

    def load(self, name):
        return json.loads((self.directory / 'raw_captures' / name).read_text(encoding='utf-8'))

    def test_original_request_and_full_response_bytes(self):
        capture = self.capture()
        raw = reply('{"risks": [], "中文": "完整原文"}')
        with self.assertRaises(client.LLMClientError):
            self.invoke(capture, raw, retries=0)
        returned = self.load('complete_000001_return.json')
        self.assertEqual(base64.b64decode(returned['raw_bytes']['data']), raw)
        self.assertEqual(returned['raw_bytes']['sha256'], hashlib.sha256(raw).hexdigest())
        body = json.loads(base64.b64decode(returned['body']['data']))
        self.assertEqual(body['messages'], self.messages)
        self.assertEqual(body['response_format']['schema'], SCHEMA)
        self.assertEqual(returned['original_return'][0], '{"risks": [], "中文": "完整原文"}')

    def test_actual_client_parsed_result_unchanged(self):
        capture = self.capture()
        pair = self.invoke(capture)
        row = self.load('request_000001_return.json')
        self.assertEqual(row['parsed_pair'], asdict(pair))
        self.assertEqual(pair.content, {'risks': []})
        self.assertTrue(capture.summary()['evidence_sufficient_for_full_response_analysis'])

    def test_exact_source_call_id_binding_without_headers(self):
        capture = self.capture()
        pair = self.invoke(capture)
        started = self.load('complete_000001_input.json')
        self.assertEqual(started['metadata']['call_id'], pair.call_id)
        for key, value in BINDING.items():
            self.assertEqual(started['metadata'][key], value)
        self.assertEqual(started['request_observation_id'], 'request_000001')
        self.assertEqual(started['source_window_binding'], 'bound')

    def test_config_and_headers_never_serialized(self):
        capture = self.capture()
        self.invoke(capture)
        all_text = ''.join(path.read_text(encoding='utf-8') for path in capture.directory.iterdir())
        self.assertNotIn('OFFLINE_FAKE_SECRET_NEVER_CAPTURE', all_text)
        self.assertNotIn('OFFLINE_HEADER_NEVER_CAPTURE', all_text)
        self.assertNotIn('Authorization', all_text)
        self.assertNotIn('extra_headers', all_text)

    def test_future_thread_is_observed(self):
        capture = self.capture()
        results = []
        failures = []
        def work():
            try:
                results.append(client.request_json(self.config, self.messages, json_schema=SCHEMA,
                                                   audit_context=BINDING))
            except BaseException as exc:
                failures.append(type(exc).__name__)
        with mock.patch.object(client.urllib.request, 'urlopen', return_value=Reply(reply())):
            capture.start()
            try:
                thread = threading.Thread(target=work)
                thread.start()
                thread.join(10)
                self.assertFalse(thread.is_alive())
            finally:
                capture.stop()
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 1)
        self.assertNotIn(threading.get_ident(), capture.summary()['profile_thread_coverage']['observed_thread_ids'])
        self.assertTrue(capture.summary()['capture_integrity_complete'])

    def test_concurrent_calls_keep_distinct_pair_ids(self):
        capture = self.capture()
        results, failures = [], []
        def work(number):
            binding = {**BINDING, 'window_id': f'window.{number}', 'job_id': f'job.{number}'}
            try:
                results.append(client.request_json(self.config, self.messages, json_schema=SCHEMA,
                                                   audit_context=binding))
            except BaseException as exc:
                failures.append(type(exc).__name__)
        with mock.patch.object(client.urllib.request, 'urlopen', side_effect=lambda *a, **k: Reply(reply())):
            capture.start()
            try:
                threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(10)
                    self.assertFalse(thread.is_alive())
            finally:
                capture.stop()
        self.assertEqual(failures, [])
        call_ids = {self.load(f'complete_{i:06d}_input.json')['metadata']['call_id'] for i in range(1, 5)}
        self.assertEqual(call_ids, {pair.call_id for pair in results})
        self.assertEqual(capture.summary()['source_window_binding_counts']['bound'], 4)

    def test_retry_preserves_each_actual_attempt(self):
        self.config.native_json_schema = False
        capture = self.capture()
        with mock.patch.object(client.urllib.request, 'urlopen', side_effect=[Reply(reply('broken json')), Reply(reply())]):
            capture.start()
            try:
                pair = client.request_json(self.config, self.messages, audit_context=BINDING, retries=1)
            finally:
                capture.stop()
        first = self.load('complete_000001_input.json')
        second = self.load('complete_000002_input.json')
        self.assertEqual(first['metadata']['attempt'], 1)
        self.assertEqual(second['metadata']['attempt'], 2)
        self.assertNotEqual(first['metadata']['call_id'], second['metadata']['call_id'])
        self.assertEqual(second['metadata']['call_id'], pair.call_id)
        self.assertEqual(first['messages'], self.messages)
        self.assertEqual(len(second['messages']), 3)
        self.assertEqual(self.load('complete_000001_return.json')['original_return'][0], 'broken json')
        self.assertEqual(pair.degraded, 'json_retry')
        self.assertTrue(capture.summary()['capture_integrity_complete'])

    def test_http_failure_original_exception_and_missing_raw(self):
        capture = self.capture()
        failure = urllib.error.HTTPError('http://127.0.0.1:1', 502, 'offline', {}, io.BytesIO(b'fixture error body'))
        with mock.patch.object(client.urllib.request, 'urlopen', side_effect=failure):
            capture.start()
            try:
                with self.assertRaises(client.LLMClientError) as raised:
                    client.request_json(self.config, self.messages, json_schema=SCHEMA, audit_context=BINDING)
            finally:
                capture.stop()
        self.assertEqual(raised.exception.status, 502)
        self.assertEqual(str(raised.exception), 'HTTP 502: fixture error body')
        self.assertEqual(self.load('complete_000001_return.json')['return_kind'], 'no_return_value_or_unwind')
        self.assertTrue(capture.summary()['capture_integrity_complete'])
        self.assertFalse(capture.summary()['all_raw_model_responses_available'])
        self.assertFalse(capture.summary()['evidence_sufficient_for_full_response_analysis'])

    def test_invalid_outer_json_preserves_original_raw_bytes(self):
        capture = self.capture()
        with self.assertRaises(client.LLMClientError):
            self.invoke(capture, b'not JSON', retries=0)
        self.assertEqual(base64.b64decode(self.load('complete_000001_return.json')['raw_bytes']['data']), b'not JSON')
        self.assertTrue(capture.summary()['all_raw_model_responses_available'])

    def test_write_failure_cannot_change_original_pair(self):
        capture = self.capture()
        with mock.patch.object(capture, '_write', side_effect=OSError('private details must not appear')):
            pair = self.invoke(capture)
        self.assertEqual(pair.content, {'risks': []})
        summary = capture.summary()
        self.assertFalse(summary['capture_integrity_complete'])
        self.assertTrue(summary['capture_errors'])
        self.assertNotIn('private details', json.dumps(summary))

    def test_baseexception_capture_failure_cannot_change_pair(self):
        capture = self.capture()
        with mock.patch.object(capture, '_write', side_effect=KeyboardInterrupt):
            pair = self.invoke(capture)
        self.assertEqual(pair.content, {'risks': []})
        self.assertEqual({row['error_type'] for row in capture.summary()['capture_errors']}, {'KeyboardInterrupt'})

    def test_capture_failure_cannot_mask_original_http_error(self):
        capture = self.capture()
        with mock.patch.object(capture, '_write', side_effect=OSError('offline write error')):
            with mock.patch.object(client.urllib.request, 'urlopen', side_effect=TimeoutError('offline timeout')):
                capture.start()
                try:
                    with self.assertRaises(client.LLMClientError) as raised:
                        client.request_json(self.config, self.messages, json_schema=SCHEMA, audit_context=BINDING)
                finally:
                    capture.stop()
        self.assertIn('offline timeout', str(raised.exception))
        self.assertFalse(capture.summary()['capture_integrity_complete'])

    def test_judge_without_window_is_declared_not_guessed(self):
        capture = self.capture()
        self.invoke(capture, binding=None)
        row = self.load('complete_000001_input.json')
        self.assertNotIn('window_id', row['metadata'])
        self.assertEqual(row['source_window_binding'], 'not_supplied_by_caller')
        self.assertEqual(capture.summary()['source_window_binding_counts']['not_supplied_by_caller'], 1)
        self.assertTrue(capture.summary()['evidence_sufficient_for_full_response_analysis'])

    def test_missing_source_call_id_is_not_reconstructed(self):
        capture = self.capture()
        with mock.patch.object(client.urllib.request, 'urlopen', return_value=Reply(reply())):
            capture.start()
            try:
                client._complete(self.config, self.messages, 256, SCHEMA)
            finally:
                capture.stop()
        row = self.load('complete_000001_input.json')
        self.assertEqual(row['metadata'], {})
        self.assertFalse(capture.summary()['source_call_bindings_complete'])
        self.assertFalse(capture.summary()['evidence_sufficient_for_full_response_analysis'])

    def test_existing_profile_is_preserved_and_rejected(self):
        def prior(frame, event, arg):
            return None
        capture = self.capture()
        sys.setprofile(prior)
        with self.assertRaises(RuntimeError):
            capture.start()
        self.assertIs(sys.getprofile(), prior)
        self.assertFalse(capture.directory.exists())

    def test_existing_thread_profile_is_preserved_and_rejected(self):
        def prior(frame, event, arg):
            return None
        capture = self.capture()
        threading.setprofile(prior)
        with self.assertRaises(RuntimeError):
            capture.start()
        self.assertIs(threading.getprofile(), prior)

    def test_profile_hooks_are_restored(self):
        capture = self.capture()
        self.invoke(capture)
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(threading.getprofile())
        capture.stop()

    def test_existing_capture_directory_is_never_overwritten(self):
        capture = self.capture()
        capture.directory.mkdir()
        marker = capture.directory / 'keep.txt'
        marker.write_text('preserve', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            capture.start()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')

    def test_json_clone_refuses_arbitrary_objects_without_access(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError('private object accessed')
        with self.assertRaises(TypeError):
            observed.plain(Forbidden())
        with self.assertRaises(TypeError):
            observed.plain(float('nan'))

    def test_capture_does_not_read_config_object(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError('private config accessed')
        def complete(config, messages, max_tokens, json_schema=None):
            body = b'{}'
            raw_bytes = reply()
            return '{"risks": []}', json.loads(raw_bytes), 0.0
        def request(config, messages, max_tokens=256, json_schema=None):
            meta = {'call_id': 'offline.call', **BINDING}
            raw, envelope, elapsed = complete(config, messages, max_tokens, json_schema)
            return client.LLMPair(json.loads(raw), envelope['model'], elapsed, raw=raw, call_id=meta['call_id'])
        capture = self.capture(complete=complete, request=request)
        capture.start()
        try:
            pair = request(Forbidden(), self.messages, json_schema=SCHEMA)
        finally:
            capture.stop()
        self.assertEqual(pair.content, {'risks': []})
        self.assertTrue(capture.summary()['capture_integrity_complete'])

    def test_file_hash_inventory_matches_each_capture(self):
        capture = self.capture()
        self.invoke(capture)
        summary = capture.summary()
        self.assertEqual(len(summary['files']), 4)
        for row in summary['files']:
            data = (self.directory / row['path']).read_bytes()
            self.assertEqual(len(data), row['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])

    def test_audit_call_ids_exactly_reconcile(self):
        capture = self.capture()
        self.invoke(capture)
        reconciliation = observed.reconcile_calls(self.directory, capture.summary())
        self.assertTrue(reconciliation['audit_started_calls_exactly_match_captured_attempts'])
        self.assertEqual(reconciliation['audit_started_count'], 1)

    def test_unobserved_audit_call_prevents_complete_evidence(self):
        capture = self.capture()
        self.invoke(capture)
        with mock.patch.object(client.urllib.request, 'urlopen', return_value=Reply(reply())):
            client.request_json(self.config, self.messages, json_schema=SCHEMA, audit_context=BINDING)
        reconciliation = observed.reconcile_calls(self.directory, capture.summary())
        self.assertFalse(reconciliation['audit_started_calls_exactly_match_captured_attempts'])
        self.assertEqual(len(reconciliation['missing_capture_call_ids']), 1)

    def test_duplicate_capture_call_id_cannot_reconcile(self):
        capture = self.capture()
        self.invoke(capture)
        summary = capture.summary()
        summary['source_complete_call_ids'] *= 2
        reconciliation = observed.reconcile_calls(self.directory, summary)
        self.assertFalse(reconciliation['call_ids_unique_per_attempt'])
        self.assertFalse(reconciliation['audit_started_calls_exactly_match_captured_attempts'])

    def test_prelock_failure_stays_inside_hook(self):
        class BrokenLock:
            def __enter__(self):
                raise KeyboardInterrupt
            def __exit__(self, *args):
                return False
        capture = self.capture()
        original_lock = capture.lock
        capture.lock = BrokenLock()
        pair = self.invoke(capture)
        capture.lock = original_lock
        self.assertEqual(pair.content, {'risks': []})
        self.assertFalse(capture.summary()['capture_integrity_complete'])
        self.assertEqual({row['error_type'] for row in capture.summary()['capture_errors']}, {'KeyboardInterrupt'})


class ProvenanceCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='observed_v3_binding_')
        self.directory = Path(self.temporary.name)
        self.expected = {'product_source_sha256': {'nodes.py': 'fixture-sha'}}
        raw = observed.v2.scoped_contract()
        self.prepared = {'synthetic_input': True, 'raw_text': raw,
            'raw_text_sha256': observed.sha(raw.encode('utf-8')),
            'tool_sha256': observed.sha(Path(observed.base.__file__).read_bytes()),
            'provenance': {**self.expected, **observed.wrapper_provenance(),
                'diagnostic_runner': Path(observed.v2.__file__).name,
                'diagnostic_runner_sha256': observed.sha(Path(observed.v2.__file__).read_bytes()),
                'fixture_version': 'scoped_value_tail_v2', 'fixture_is_independent_legal_gold': False},
            'source_manifest': {'mode': 'synthetic_diagnostic_not_real_transaction',
                'fixture_version': 'scoped_value_tail_v2',
                'runner_sha256': observed.sha(Path(observed.v2.__file__).read_bytes())}}

    def tearDown(self):
        self.temporary.cleanup()

    def test_matching_fixed_preparation_accepted(self):
        observed.check_prepared(self.prepared, self.expected)

    def test_each_stale_binding_rejected(self):
        for field in ('observed_runner_sha256', 'diagnostic_runner_sha256', 'product_source_sha256', 'fixture_version'):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(self.prepared))
                changed['provenance'][field] = 'stale'
                with self.assertRaises(ValueError):
                    observed.check_prepared(changed, self.expected)

    def test_legacy_prepared_without_wrapper_rejected(self):
        del self.prepared['provenance']['observed_runner_sha256']
        with self.assertRaises(ValueError):
            observed.check_prepared(self.prepared, self.expected)

    def test_public_or_modified_fixture_rejected(self):
        for key, value in (('synthetic_input', False), ('raw_text', 'different'), ('tool_sha256', 'old')):
            with self.subTest(key=key):
                changed = {**self.prepared, key: value}
                with self.assertRaises(ValueError):
                    observed.check_prepared(changed, self.expected)

    def write_bound_result(self):
        path = self.directory / 'prepared.json'
        path.write_text(json.dumps(self.prepared), encoding='utf-8')
        result = {'prepared_sha256': observed.sha(path.read_bytes()),
            'tool_sha256': self.prepared['tool_sha256'], 'provenance': self.prepared['provenance'],
            'source_manifest': self.prepared['source_manifest']}
        (self.directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        return result

    def test_prepared_and_result_match(self):
        self.write_bound_result()
        with mock.patch.object(observed.base, 'source_provenance', return_value=self.expected):
            result = observed.result_binding(self.directory, self.expected)
        self.assertTrue(result['prepared_result_provenance_match'])

    def test_result_preparation_hash_mismatch_rejected(self):
        result = self.write_bound_result()
        result['prepared_sha256'] = 'stale'
        (self.directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaises(ValueError):
            observed.result_binding(self.directory, self.expected)

    def test_product_change_during_run_rejected(self):
        self.write_bound_result()
        with mock.patch.object(observed.base, 'source_provenance', return_value={'product_source_sha256': {'nodes.py': 'changed'}}):
            with self.assertRaises(ValueError):
                observed.result_binding(self.directory, self.expected)

    def test_public_contract_cli_rejected_before_runner(self):
        with mock.patch.object(observed.v2, 'main') as runner:
            with self.assertRaises(ValueError):
                observed.main(['--run-dir', str(self.directory), '--public-contract-dir', str(self.directory)])
        runner.assert_not_called()

    def test_passes_argv_and_restores_original_base_functions(self):
        temporary_root = self.directory
        run = temporary_root / 'tools/acceptance_results/test'
        originals = {name: getattr(observed.base, name) for name in ('synthetic_contract', 'prepare_case', 'source_provenance')}
        argv = ['--run-dir', str(run), '--max-tokens', '4096', '--cross-windows', '4']
        def runner(args):
            self.assertEqual(args, argv)
            self.assertIn('observed_runner_sha256', observed.base.source_provenance())
            observed.base.synthetic_contract = lambda: 'temporary'
            observed.base.prepare_case = lambda: None
            return 23
        with mock.patch.object(observed, 'ROOT', temporary_root), mock.patch.object(observed.v2, 'main', side_effect=runner), \
                mock.patch.object(observed.base, 'source_provenance', return_value=self.expected):
            self.assertEqual(observed.main(argv), 23)
        for name, function in originals.items():
            self.assertIs(getattr(observed.base, name), function)
        self.assertFalse((run / 'raw_captures').exists())

    def test_exception_from_original_runner_not_masked_and_globals_restored(self):
        temporary_root = self.directory
        run = temporary_root / 'tools/acceptance_results/test'
        original = observed.base.source_provenance
        failure = RuntimeError('original runner failure')
        with mock.patch.object(observed, 'ROOT', temporary_root), mock.patch.object(observed.v2, 'main', side_effect=failure), \
                mock.patch.object(observed.base, 'source_provenance', return_value=self.expected):
            with self.assertRaises(RuntimeError) as raised:
                observed.main(['--run-dir', str(run)])
        self.assertIs(raised.exception, failure)
        self.assertIs(observed.base.source_provenance, original)

    def test_execute_stop_failure_preserves_original_return_and_restores_hooks(self):
        run = self.directory / 'tools/acceptance_results/test'
        original = observed.base.source_provenance
        with mock.patch.object(observed, 'ROOT', self.directory), \
                mock.patch.object(observed.base, 'source_provenance', return_value=self.expected), \
                mock.patch.object(observed.v2, 'main', return_value=19), \
                mock.patch.object(observed.PassiveCapture, 'stop', side_effect=OSError('offline stop failure')):
            self.assertEqual(observed.main(['--run-dir', str(run), '--execute']), 19)
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(threading.getprofile())
        self.assertIs(observed.base.source_provenance, original)
        manifest = json.loads((run / 'capture_manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['evidence_sufficient_for_full_response_analysis'])
        self.assertIn({'stage': 'stop', 'error_type': 'OSError'}, manifest['capture_errors'])

    def test_execute_finalization_and_stderr_failure_preserve_original_exception(self):
        run = self.directory / 'tools/acceptance_results/test'
        failure = RuntimeError('original runner failure')
        with mock.patch.object(observed, 'ROOT', self.directory), \
                mock.patch.object(observed.base, 'source_provenance', return_value=self.expected), \
                mock.patch.object(observed.v2, 'main', side_effect=failure), \
                mock.patch.object(observed.PassiveCapture, 'summary', side_effect=MemoryError), \
                mock.patch.object(sys.stderr, 'write', side_effect=OSError):
            with self.assertRaises(RuntimeError) as raised:
                observed.main(['--run-dir', str(run), '--execute'])
        self.assertIs(raised.exception, failure)
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(threading.getprofile())


def frozen_hashes():
    paths = sorted((ROOT / 'zhilv_qiling_v2').glob('*.py')) + [ROOT / 'zhilv_qiling_v2/webui.html']
    paths += [ROOT / 'tools' / name for name in ('verify_complex_core_workflow.py', 'verify_complex_core_v2.py',
        'verify_complex_postrun_v2.py', 'inspect_scoped_diagnostic_outcomes_20261115.py',
        'replay_native_cross_windows_20261115.py')]
    return {str(path.relative_to(ROOT)).replace('\\', '/'): observed.sha(path.read_bytes()) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-dir', type=Path, required=True)
    args = parser.parse_args()
    before = frozen_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = frozen_hashes()
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'passed' if result.wasSuccessful() and before == after else 'failed',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'no_real_model_execution': True,
        'network_socket_creation_blocked_in_call_tests': True, 'torch_and_model_imports_blocked': True,
        'audit_database_scope': 'temporary directory per call test',
        'capture_outputs_scope': 'temporary directories only; no run_v3 prepared or executed',
        'frozen_sources_unchanged': before == after, 'frozen_source_sha256': after,
        'wrapper_sha256': observed.sha(Path(observed.__file__).read_bytes()),
        'tests_sha256': observed.sha(Path(__file__).read_bytes())}
    (args.receipt_dir / 'test_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('status', 'tests_run', 'failures', 'errors', 'frozen_sources_unchanged')}))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
