"""Direct-product regressions for the applied entailment input-coverage guard.

Run after applying entailment_input_coverage_20261115.patch. Imports the actual
product modules; never applies or compiles an in-memory proposal. All predictor
responses are explicit stubs, sockets are blocked, and audit writes are isolated
in temporary databases. This is not a model-quality or legal-quality evaluation.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import agents as A, agents_multi_run as G, entailment as E, nodes as N
from zhilv_qiling_v2 import model_audit
from zhilv_qiling_v2.models import RetrievalHit
from zhilv_qiling_v2.tests_v2 import test_feedback_consumption as fixtures


def helper_fixture():
    helper = fixtures.FeedbackConsumptionTests()
    helper.setUp()
    return helper


def predicted(*, truncated=True):
    predictor = Mock()
    predictor.predict.return_value = {'label': 'entail', 'score': 0.99,
        'probabilities': {'entail': 0.99, 'neutral': 0.008, 'contradict': 0.002},
        'input_truncated': truncated}
    classifier = Mock()
    classifier.predict.return_value = ('entail', 0.95, 0.5)
    with patch.object(E, 'backend', return_value='bert'), \
         patch.object(E, '_bert_predictor', return_value=predictor), \
         patch.object(E, 'load_classifier', return_value=classifier):
        return E.judge_pair('本合成合同用于离线覆盖协议测试。' * 50,
            {'article_uid': 'fixture-law', 'article_no': '第一条', 'content': '合成规范原文。'})


def statute(verdict):
    helper = helper_fixture()
    candidate = copy.deepcopy(helper.first)
    hit = RetrievalHit('fixture-law', '合成规范', '第一条', '合成规范原文。', 0.9)
    retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[hit]))
    with patch.object(E, 'annotate', return_value=[verdict]):
        result = N.node_statute(helper.contract, [candidate], helper.config, retriever=retriever)
    return helper, candidate, result


class EntailmentInputCoverageProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.product_hashes = {name: hashlib.sha256((ROOT / 'zhilv_qiling_v2' / name).read_bytes()).hexdigest()
                              for name in ('entailment.py', 'nodes.py', 'agents_multi_run.py')}

    @classmethod
    def tearDownClass(cls):
        current = {name: hashlib.sha256((ROOT / 'zhilv_qiling_v2' / name).read_bytes()).hexdigest()
                   for name in cls.product_hashes}
        if current != cls.product_hashes:
            raise AssertionError('product source changed while direct regression ran')

    def setUp(self):
        self.scope = contextlib.ExitStack()
        self.addCleanup(self.scope.close)
        temporary = self.scope.enter_context(tempfile.TemporaryDirectory(prefix='coverage-product-test-'))
        self.scope.enter_context(patch.dict(os.environ,
            {'ZLQL_MODEL_AUDIT_PATH': str(Path(temporary) / 'offline-stub-model-audit.sqlite')}))
        self.scope.enter_context(patch.object(socket.socket, 'connect',
            side_effect=AssertionError('network and live model calls are forbidden in product regressions')))
        self.scope.enter_context(model_audit.review_audit('OFFLINE-STUB-product-input-coverage-test'))

    def test_prefix_retains_raw_verdict_note_and_audit_reason(self):
        with model_audit.review_audit('OFFLINE-STUB-truncated-pair') as collector:
            verdict = predicted()
        self.assertEqual(verdict.label, 'entail')
        self.assertEqual(verdict.score, 0.99)
        self.assertEqual(verdict.neural_label, 'entail')
        self.assertEqual(verdict.rule_label, 'entail')
        self.assertFalse(verdict.disagreement)
        self.assertTrue(verdict.input_truncated)
        self.assertFalse(verdict.coverage_complete)
        self.assertIn('已编码前缀', verdict.note)
        self.assertTrue(any(row['reason'] == 'bert_input_window_limit' for row in collector.events))

    def test_full_input_does_not_gain_a_new_coverage_warning(self):
        verdict = predicted(truncated=False)
        self.assertFalse(verdict.input_truncated)
        self.assertTrue(verdict.coverage_complete)
        _, candidate, result = statute(verdict)
        self.assertFalse(result.payload['risk_checks'][0]['manual_required'])
        self.assertNotIn('entailment_input_coverage_incomplete', candidate.extra.get('review_flags', []))

    def test_truncated_check_is_manual_without_relabel_or_confidence_change(self):
        helper, candidate, result = statute(predicted())
        check = result.payload['risk_checks'][0]
        self.assertEqual(check['label'], 'entail')
        self.assertEqual(check['score'], 0.99)
        self.assertTrue(check['input_truncated'])
        self.assertFalse(check['coverage_complete'])
        self.assertTrue(check['manual_required'])
        self.assertTrue(candidate.extra['manual_required'])
        self.assertIn('entailment_input_coverage_incomplete', candidate.extra['review_flags'])
        self.assertEqual(candidate.confidence, helper.first.confidence)
        evidence = next(row for row in candidate.evidence if row.kind == 'entailment_input_coverage')
        self.assertEqual(evidence.strength, 0.0)
        self.assertTrue(any(row.startswith('bert_input_window_limit:') for row in result.degraded))

    def test_truncated_check_enters_feedback_focus_and_stays_pending_manual(self):
        helper, candidate, result = statute(predicted())
        ctx = helper.context(logic=[candidate])
        ctx.emit('statute', 'logic', 'statute.matched', {'risk_checks': result.payload['risk_checks']})
        _, _, _, focus, relevant, _ = G._feedback_request(ctx)
        self.assertEqual(focus, {'a'})
        self.assertEqual(len(relevant), 1)
        G._handle_logic(ctx)
        self.assertEqual(ctx.shared['logic'].candidates[0].extra['cross_validation_status'], 'pending_manual')

    def test_confirmed_receipt_keeps_protocol_complete_but_coverage_semantic_pending(self):
        helper = helper_fixture()
        item = {**helper.law, 'input_truncated': True, 'coverage_complete': False, 'manual_required': True}
        ctx = helper.context()
        baseline = A.fuse(copy.deepcopy(ctx.shared['logic'].candidates), [A.AgentResult('node4_statute',
            citations=ctx.shared['statute'].payload['per_candidate'])], helper.contract, helper.config)
        self.assertFalse(baseline[0].review_required)
        ctx.shared['feedback_latest'] = {'current-fingerprint': helper.snapshot([item])}
        with patch.object(G, '_feedback_request', return_value=helper.request(laws=[item])), \
             patch.object(N, 'node_judge', return_value=N.NodeOutcome('node5_judge', payload={'score': 100})):
            G._handle_judge(ctx)
        risks = ctx.shared['fused']
        self.assertEqual(len(risks), len(baseline))
        self.assertEqual(risks[0].confidence, baseline[0].confidence)
        self.assertEqual(A.score_risks(risks, helper.config), A.score_risks(baseline, helper.config))
        self.assertTrue(risks[0].review_required)
        evidence = next(row for row in risks[0].evidence if row.kind == 'model_feedback')
        self.assertTrue(evidence.extra['protocol_complete'])
        self.assertEqual(evidence.extra['semantic_status'], 'pending_manual')
        self.assertEqual(evidence.extra['verdicts'], ['confirmed'])
        self.assertIn('entailment_input_coverage_incomplete', evidence.extra['upstream_review_reasons'])
        self.assertEqual(evidence.strength, 0.0)

    def test_truncated_agreeing_relation_enters_final_queue_without_receipt(self):
        helper, candidate, result = statute(predicted())
        ctx = helper.context(logic=[candidate])
        ctx.shared['statute'].payload['risk_checks'] = result.payload['risk_checks']
        with patch.object(N, 'node_judge', return_value=N.NodeOutcome('node5_judge', payload={'score': 100})):
            G._handle_judge(ctx)
        self.assertTrue(ctx.shared['fused'][0].review_required)
        self.assertIn('输入覆盖不完整', ctx.shared['fused'][0].rationale)
        self.assertEqual(ctx.shared['fused'][0].confidence, candidate.confidence)

    def test_neural_rule_disagreement_stays_semantic_pending_even_if_confirmed(self):
        helper = helper_fixture()
        item = {**helper.law, 'disagreement': True, 'manual_required': True}
        ctx = helper.context()
        ctx.shared['feedback_latest'] = {'current-fingerprint': helper.snapshot([item])}
        with patch.object(G, '_feedback_request', return_value=helper.request(laws=[item])):
            record = G._current_feedback_consumption(ctx)[0]
        self.assertTrue(record['protocol_complete'])
        self.assertTrue(record['pending_manual'])
        self.assertIn('neural_rule_disagreement', record['upstream_review_reasons'])

    def test_complete_confirmed_relation_has_no_new_manual_flag(self):
        helper = helper_fixture()
        item = {**helper.law, 'input_truncated': False, 'coverage_complete': True, 'manual_required': False}
        ctx = helper.context()
        ctx.shared['feedback_latest'] = {'current-fingerprint': helper.snapshot([item])}
        with patch.object(G, '_feedback_request', return_value=helper.request(laws=[item])):
            record = G._current_feedback_consumption(ctx)[0]
        self.assertTrue(record['protocol_complete'])
        self.assertFalse(record['pending_manual'])
        self.assertEqual(record['semantic_status'], 'provided_relation_confirmed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
