"""Offline public-source/passive-observer checks; no model or socket calls."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'llama_cpp'}:
            raise AssertionError('real model imports forbidden in offline validation')


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
sys.meta_path.insert(0, NoModels())
import verify_public_contract_observed as public
import verify_public_contract_observed_receipts_v2 as receipts
import verify_complex_core_observed_v3 as v3


SOURCE = ROOT / 'data/gov_contracts/07ccde130333'


class PublicObservedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = public.public_source_binding(SOURCE)
        cls.expected = public.base.source_provenance()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='public-observed-offline-')
        self.directory = Path(self.temp.name)
        self.no_network = patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden'))
        self.no_network.start()
        self.prepared = {
            'synthetic_input': False, 'legal_quality_evaluated': False,
            'raw_text': self.source['raw_text'], 'raw_text_sha256': self.source['raw_text_sha256'],
            'source_name': self.source['source_name'], 'source_manifest': copy.deepcopy(self.source['source_manifest']),
            'tool_sha256': public.sha(Path(public.base.__file__).read_bytes()),
            'provenance': {**self.expected, **public.wrapper_provenance()},
            'subject_records': [], 'subject_identity_diagnostics': {},
            'parsed_text': self.source['raw_text'], 'parsed_text_sha256': self.source['raw_text_sha256'],
        }

    def tearDown(self):
        self.no_network.stop()
        self.temp.cleanup()

    def write_result(self):
        prepared_path = self.directory / 'prepared.json'
        prepared_path.write_text(json.dumps(self.prepared, ensure_ascii=False), encoding='utf-8')
        (self.directory / 'input-contract.txt').write_text(self.source['raw_text'], encoding='utf-8')
        result = {'synthetic_input': False, 'legal_quality_evaluated': False,
                  'prepared_sha256': public.sha(prepared_path.read_bytes()),
                  'tool_sha256': self.prepared['tool_sha256'], 'provenance': self.prepared['provenance'],
                  'source_manifest': self.prepared['source_manifest']}
        (self.directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        (self.directory / 'report.json').write_text(json.dumps({'document': {'full_text': self.prepared['parsed_text']}}), encoding='utf-8')
        return result

    def test_actual_four_page_local_copy_is_complete_unmodified_and_not_gold(self):
        source = self.source
        pages = sorted((SOURCE / 'text').glob('page_*.txt'))
        self.assertEqual(source['page_text_count'], 4)
        self.assertEqual(source['raw_text'], '\n'.join(path.read_text(encoding='utf-8') for path in pages))
        self.assertEqual(source['source_manifest']['mode'], 'existing_public_procurement_local_copy')
        self.assertTrue(source['all_numbered_page_texts_supplied'])
        self.assertFalse(source['source_refetched_this_run'])
        self.assertFalse(source['input_is_independent_legal_gold'])
        self.assertNotIn('fixture_version', source['source_manifest'])
        self.assertIs(public.PassiveCapture, v3.PassiveCapture)
        self.assertTrue(issubclass(receipts.PublicPassiveCapture, v3.PassiveCapture))

    def test_real_client_stub_return_keeps_full_new_receipts_and_original_network_bytes(self):
        from zhilv_qiling_v2 import llm_client as client
        from complex_observed_v3_offline_tests_20261115 import Reply, reply, BINDING, SCHEMA
        raw = reply()
        config = client.LLMConfig(endpoint='http://127.0.0.1:1/v1', model='legalone-8b', native_json_schema=True)
        capture = receipts.PublicPassiveCapture(self.directory / 'raw_captures',
                                                client._complete.__code__, client.request_json.__code__)
        with patch.dict(os.environ, {'ZLQL_MODEL_AUDIT_PATH': str(self.directory / 'model_calls.sqlite')}), \
                patch.object(client.urllib.request, 'urlopen', return_value=Reply(raw)):
            capture.start()
            try:
                pair = client.request_json(config, [{'role': 'user', 'content': '离线合成原文'}],
                                           json_schema=SCHEMA, audit_context=BINDING)
            finally:
                capture.stop()
        returned = json.loads((capture.directory / 'request_000001_return.json').read_text(encoding='utf-8'))
        self.assertEqual(returned['parsed_pair'], asdict(pair))
        self.assertTrue(pair.request_receipts)
        self.assertTrue(capture.summary()['evidence_sufficient_for_full_response_analysis'])
        self.assertTrue(public.reconcile_calls(self.directory, capture.summary())[
            'audit_started_calls_exactly_match_captured_attempts'])

    def test_numbered_page_gaps_rejected_even_if_metadata_count_matches(self):
        folder = self.directory / 'public'
        (folder / 'text').mkdir(parents=True)
        (folder / 'meta.json').write_text('{"page_count": 2}', encoding='utf-8')
        for number in (1, 3):
            (folder / 'text' / f'page_{number:04d}.txt').write_text('合同原文', encoding='utf-8')
        with patch.object(public.base, 'public_contract_source', return_value=('合同原文', 'public.txt', {})):
            with self.assertRaises(ValueError):
                public.public_source_binding(folder)

    def test_matching_prepared_and_result_bind_all_public_sources(self):
        self.write_result()
        public.check_prepared(self.prepared, self.expected, self.source)
        binding = public.result_binding(self.directory, self.expected, self.source)
        self.assertTrue(binding['prepared_result_provenance_match'])
        self.assertTrue(binding['report_text_matches_prepared'])
        self.assertFalse(binding['source_refetched_this_run'])
        self.assertNotIn('raw_text', binding)

    def test_synthetic_or_stale_preparation_and_injected_facts_rejected(self):
        for change in ({'synthetic_input': True}, {'raw_text': '换掉原合同'}, {'tool_sha256': 'stale'},
                       {'subject_records': [{'subject_name': '合成企业'}]},
                       {'provenance': {**self.prepared['provenance'], 'fixture_version': 'scoped_value_tail_v2'}},
                       {'source_manifest': {**self.prepared['source_manifest'], 'source_refetched_this_run': True}}):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    public.check_prepared({**self.prepared, **change}, self.expected, self.source)

    def test_stale_result_saved_input_report_or_source_change_rejected(self):
        self.write_result()
        for filename in ('result.json', 'input-contract.txt', 'report.json'):
            with self.subTest(filename=filename):
                path = self.directory / filename
                original = path.read_bytes()
                path.write_text('{}' if filename.endswith('.json') else '截断原文', encoding='utf-8')
                with self.assertRaises(ValueError):
                    public.result_binding(self.directory, self.expected, self.source)
                path.write_bytes(original)
        with patch.object(public, 'public_source_binding', return_value={**self.source, 'raw_text_sha256': 'changed'}):
            with self.assertRaises(ValueError):
                public.result_binding(self.directory, self.expected, self.source)

    def test_direct_base_runner_receives_real_public_argument_and_restores_provenance(self):
        run = self.directory / 'tools/acceptance_results/fresh'
        argv = ['--run-dir', str(run), '--public-contract-dir', str(SOURCE), '--max-tokens', '4096']
        original = public.base.source_provenance
        original_synthetic = public.base.synthetic_contract
        original_prepare = public.base.prepare_case
        def runner(args):
            self.assertEqual(args, argv)
            self.assertIn('observed_runner_sha256', public.base.source_provenance())
            self.assertNotIn('fixture_version', public.base.source_provenance())
            self.assertIs(public.base.prepare_case, original_prepare)
            self.assertIs(public.base.synthetic_contract, original_synthetic)
            return 23
        with patch.object(public, 'ROOT', self.directory), patch.object(public.base, 'main', side_effect=runner) as called, \
                patch.object(v3.v2, 'main', side_effect=AssertionError('synthetic wrapper forbidden')):
            self.assertEqual(public.main(argv), 23)
            called.assert_called_once()
        self.assertIs(public.base.source_provenance, original)
        self.assertFalse((run / 'raw_captures').exists())

    def test_existing_evidence_is_preserved_before_runner(self):
        run = self.directory / 'tools/acceptance_results/fresh'
        run.mkdir(parents=True)
        marker = run / 'model_calls.sqlite'
        marker.write_bytes(b'preserve')
        with patch.object(public, 'ROOT', self.directory), patch.object(public.base, 'main') as runner:
            with self.assertRaises(ValueError):
                public.main(['--run-dir', str(run), '--public-contract-dir', str(SOURCE)])
        runner.assert_not_called()
        self.assertEqual(marker.read_bytes(), b'preserve')

    def test_original_runner_failure_restores_provenance_and_is_not_masked(self):
        run = self.directory / 'tools/acceptance_results/fresh'
        original = public.base.source_provenance
        failure = RuntimeError('original failure')
        with patch.object(public, 'ROOT', self.directory), patch.object(public.base, 'main', side_effect=failure):
            with self.assertRaises(RuntimeError) as raised:
                public.main(['--run-dir', str(run), '--public-contract-dir', str(SOURCE)])
        self.assertIs(raised.exception, failure)
        self.assertIs(public.base.source_provenance, original)


if __name__ == '__main__':
    unittest.main()
