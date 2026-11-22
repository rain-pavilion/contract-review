"""Offline product regressions for receipt-bound node2 evidence provenance.

Normally imports the product; never loads or applies a proposal. The transport
is explicitly stubbed and every audit database lives in a temporary directory.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import zhilv_qiling_v2 as PACKAGE
from zhilv_qiling_v2 import agents as A, llm_client as L, model_audit as AUDIT, nodes as N, rules as R
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse


def clause(cid, text):
    return Clause(cid, '', '', 1, '', text, text, ['block_' + cid])


@contextmanager
def isolated_runtime():
    with tempfile.TemporaryDirectory(prefix='zlql_evidence_provenance_') as temporary, \
         patch.object(PACKAGE, 'model_audit', AUDIT, create=True), \
         patch.dict(sys.modules, {'zhilv_qiling_v2.model_audit': AUDIT}), \
         patch.dict(os.environ, {'ZLQL_MODEL_AUDIT_PATH': str(Path(temporary) / 'audit.sqlite')}):
        yield Path(temporary) / 'audit.sqlite'


class LLMEvidenceProvenanceProductTests(unittest.TestCase):
    def review(self, *, partial=False, normalized_only=False):
        primary = '本合同尾款说明：' + '付款明细均以双方签署的清单为准。' * 7 + '约定尾款人民币70000元。'
        if normalized_only:
            primary = '本合同 尾款人民币70000元，应依约支付。'
        related = '本合同尾款人民币80000元，应按补充约定结算。'
        rows = [clause('a', primary), clause('b', related)]
        contract = Contract('evidence-product-contract', DocumentParse('fixture.txt', 'txt', '\n'.join(c.text for c in rows), clauses=rows))
        quote = primary.replace(' ', '') if normalized_only else primary
        message = '模型原始关系声称：两条尾款数值存在差异，需复核。'

        def complete(_config, messages, _max_tokens, **_kwargs):
            payload = json.loads(messages[-1]['content'])
            content = {'defects': []}
            if payload['kind'] == 'cross_clause':
                content['defects'] = [{'clause_id': 'a', 'code': 'fee_mismatch', 'message': message,
                    'quote': quote, 'related_quotes': [{'clause_id': 'b', 'quote': related}]}]
            return json.dumps(content, ensure_ascii=False), {
                'model': 'legalone-8b', 'choices': [{'finish_reason': 'length' if partial else 'stop'}]}, 0.01

        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='legalone-8b',
            llm_chunk_chars=600, llm_chunk_overlap=0, llm_cross_clause_pairs=10)
        with isolated_runtime(), patch.object(L, '_complete', side_effect=complete):
            with AUDIT.review_audit('evidence-product-trace', contract.contract_id) as collector:
                candidates, meta, degraded = N._llm_logic(contract, config, rows)
        self.assertEqual(len(candidates), 1)
        return candidates[0], meta, degraded, contract, config, collector.events, [(rows[0].clause_id, quote), (rows[1].clause_id, related)], message

    def test_full_primary_and_related_quotes_are_bound_to_their_actual_call(self):
        candidate, meta, _, _, _, events, quotes, message = self.review()
        model_evidence = [evidence for evidence in candidate.evidence if evidence.source == 'node2_llm']
        self.assertEqual(len(model_evidence), 2)
        self.assertGreater(len(quotes[0][1]), 120)
        for evidence, (cid, quote) in zip(model_evidence, quotes):
            extra = evidence.extra
            self.assertEqual(extra['candidate_id'], candidate.candidate_id)
            self.assertEqual(extra['clause_id'], cid)
            self.assertEqual(extra['quote'], quote)
            self.assertIs(extra['quote_exact_source_match'], True)
            self.assertIs(extra['partial_model_output'], False)
            self.assertEqual(extra['model_message'], message)
            self.assertEqual(extra['issue_code'], 'fee_mismatch')
            self.assertEqual([(row['clause_id'], row['quote']) for row in extra['bound_model_quotes']], quotes)
            self.assertTrue(all(row['quote_exact_source_match'] is True for row in extra['bound_model_quotes']))
            call = next(row for row in meta['calls'] if row['call_id'] == extra['call_id'])
            for key in ('window_id', 'job_id', 'source_sha256'):
                self.assertEqual(extra[key], call[key])
            receipt = [row for row in events if row.get('details', {}).get('call_id') == extra['call_id']]
            self.assertEqual([row['status'] for row in receipt], ['started', 'succeeded'])
            self.assertTrue(all(row['details']['source_sha256'] == extra['source_sha256'] for row in receipt))
        self.assertNotIn(quotes[0][1], json.dumps(events, ensure_ascii=False))

    def test_final_fusion_preserves_provenance_without_changing_score_or_evidence_weight(self):
        candidate, _, _, contract, config, _, _, _ = self.review()
        baseline = copy.deepcopy(candidate)
        for evidence in baseline.evidence:
            evidence.extra.clear()
        before = A.fuse([baseline], [], contract, config)
        after = A.fuse([copy.deepcopy(candidate)], [], contract, config)
        self.assertEqual(len(before), len(after))
        self.assertEqual(A.score_risks(before, config), A.score_risks(after, config))
        self.assertEqual([(risk.code, risk.message, risk.confidence, risk.severity) for risk in before],
                         [(risk.code, risk.message, risk.confidence, risk.severity) for risk in after])
        self.assertEqual([(e.kind, e.text, e.strength, e.source) for e in before[0].evidence],
                         [(e.kind, e.text, e.strength, e.source) for e in after[0].evidence])
        self.assertEqual([e.extra for e in after[0].evidence], [e.extra for e in candidate.evidence])

    def test_partial_model_output_keeps_bound_evidence_and_never_claims_completion(self):
        candidate, meta, degraded, _, _, _, _, _ = self.review(partial=True)
        self.assertFalse(meta['complete'])
        self.assertFalse(meta['text_complete'])
        self.assertTrue(any('output_truncated' in reason for reason in degraded))
        self.assertTrue(all(e.extra['partial_model_output'] is True and e.extra['call_id'] for e in candidate.evidence))
        self.assertTrue(all(call['status'] == 'partial' for call in meta['calls']))

    def test_rule_evidence_is_never_stamped_as_a_model_observation(self):
        candidate, _, _, contract, config, _, quotes, _ = self.review()
        rule = R._candidate('fee_mismatch', '规则侧独立发现。', contract.document.clauses[0], quotes[0][1], source='rule')
        before_rule = copy.deepcopy(rule.evidence)
        risks = A.fuse([rule, candidate], [], contract, config)
        rule_evidence = [e for risk in risks for e in risk.evidence if e.source == 'rule']
        self.assertEqual(rule.evidence, before_rule)
        self.assertTrue(rule_evidence)
        self.assertTrue(all(not set(e.extra).intersection({'call_id', 'bound_model_quotes', 'model_message'}) for e in rule_evidence))

    def test_retained_related_model_evidence_preserves_the_full_observation_after_text_deduplication(self):
        candidate, _, _, contract, config, _, quotes, _ = self.review()
        rule = R._candidate('fee_mismatch', '规则侧先保留的说明。', contract.document.clauses[0], quotes[0][1], source='rule')
        rule.merge_from(candidate)
        risk = A.fuse([rule], [], contract, config)[0]
        model_evidence = [e for e in risk.evidence if e.source == 'node2_llm']
        self.assertEqual(len(model_evidence), 1)
        self.assertEqual(model_evidence[0].extra['clause_id'], 'b')
        self.assertEqual([(row['clause_id'], row['quote']) for row in model_evidence[0].extra['bound_model_quotes']], quotes)
        self.assertTrue(all(not e.extra for e in risk.evidence if e.source == 'rule'))

    def test_normalized_only_quote_is_preserved_but_never_marked_an_exact_source_match(self):
        candidate, _, _, _, _, _, quotes, _ = self.review(normalized_only=True)
        primary = next(e for e in candidate.evidence if e.extra.get('clause_id') == 'a')
        self.assertEqual(primary.extra['quote'], quotes[0][1])
        self.assertIs(primary.extra['quote_exact_source_match'], False)
        self.assertIs(primary.extra['bound_model_quotes'][0]['quote_exact_source_match'], False)
        self.assertIs(primary.extra['bound_model_quotes'][1]['quote_exact_source_match'], True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    watched = [ROOT / 'zhilv_qiling_v2' / (name + '.py') for name in ('nodes', 'llm_client', 'model_audit', 'models', 'agents', 'rules')]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LLMEvidenceProvenanceProductTests)
    if args.collect_only:
        print(json.dumps({'tests': suite.countTestCases(), 'mode': 'normal_product_imports_no_proposals'}))
        return 0
    def no_network(*_args, **_kwargs):
        raise AssertionError('network is forbidden in evidence provenance product tests')
    with patch('socket.create_connection', side_effect=no_network), \
         patch.object(socket.socket, 'connect', no_network), patch.object(socket.socket, 'connect_ex', no_network):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError('product source changed during regression: ' + str(path))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
