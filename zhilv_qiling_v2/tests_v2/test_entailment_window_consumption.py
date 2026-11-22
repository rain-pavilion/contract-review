"""Local neural windows cannot be promoted to whole-input or legal confirmation."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi_run as G, model_audit as A, nodes as N, rules as R
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.retrieve import RetrievalHit
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract
from zhilv_qiling_v2.tests_v2 import test_feedback_consumption as F


class EntailmentWindowConsumptionTests(unittest.TestCase):
    def test_all_executed_local_windows_still_require_human_review(self):
        row = clause('a', '测试条款，不是真实法律金标。')
        document = contract([row])
        candidate = R._candidate('liability_exemption', '合成协议测试', row, row.text, confidence=0.95)
        hit = RetrievalHit('test-law', '测试规范', '第一条', '仅供协议测试。', 0.9)
        audit = {'executed_pair_count': 4, 'planned_pair_count': 4, 'predictions': []}
        verdict = N.E.EntailmentVerdict('test-law', '第一条', 'entail', 0.9,
            source='bert_entail', input_truncated=False, coverage_complete=False,
            token_coverage_complete=True, pair_coverage_complete=True,
            semantic_pending=True, manual_required=True, decision_scope='local_window_only',
            window_audit=audit)
        retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[hit]))
        with patch.object(N.E, 'annotate', return_value=[verdict]):
            outcome = N.node_statute(document, [candidate], Config(), retriever=retriever)
        check = outcome.payload['risk_checks'][0]
        self.assertTrue(check['manual_required'])
        self.assertTrue(check['token_coverage_complete'])
        self.assertTrue(check['pair_coverage_complete'])
        self.assertTrue(check['semantic_pending'])
        self.assertEqual(check['window_audit'], audit)
        self.assertTrue(candidate.extra['manual_required'])
        self.assertIn('entailment_local_window_only', candidate.extra['review_flags'])
        self.assertFalse(any('bert_input_window_limit' in value for value in outcome.degraded))

    def test_confirmed_llm_feedback_cannot_close_local_window_semantics(self):
        fixture = F.FeedbackConsumptionTests(methodName='test_unknown_without_candidate_creates_no_illegality_candidate')
        fixture.setUp()
        ctx = fixture.context()
        local = {**fixture.law, 'label': 'entail', 'input_truncated': False,
                 'coverage_complete': False, 'token_coverage_complete': True,
                 'pair_coverage_complete': True, 'semantic_pending': True,
                 'manual_required': True, 'window_disagreement': True,
                 'decision_scope': 'local_window_only', 'window_audit': {'planned_pair_count': 4}}
        ctx.shared['feedback_latest'] = {'current-fingerprint': fixture.snapshot([local])}
        with patch.object(G, '_feedback_request', return_value=fixture.request(laws=[local])):
            records = G._current_feedback_consumption(ctx)
        self.assertTrue(records[0]['protocol_complete'])
        self.assertTrue(records[0]['pending_manual'])
        self.assertIn('entailment_local_window_only', records[0]['upstream_review_reasons'])
        self.assertIn('entailment_window_disagreement', records[0]['upstream_review_reasons'])
        self.assertNotIn('entailment_input_coverage_incomplete', records[0]['upstream_review_reasons'])
        self.assertTrue(records[0]['token_coverage_complete'])
        self.assertEqual(records[0]['window_audit'], local['window_audit'])

    def test_unexecuted_pairs_have_distinct_coverage_reason(self):
        fixture = F.FeedbackConsumptionTests(methodName='test_unknown_without_candidate_creates_no_illegality_candidate')
        fixture.setUp()
        ctx = fixture.context()
        local = {**fixture.law, 'pair_coverage_complete': False, 'token_coverage_complete': True,
                 'semantic_pending': True, 'coverage_complete': False, 'manual_required': True,
                 'decision_scope': 'local_window_only'}
        ctx.shared['feedback_latest'] = {'current-fingerprint': fixture.snapshot([local])}
        with patch.object(G, '_feedback_request', return_value=fixture.request(laws=[local])):
            result = G._current_feedback_consumption(ctx)[0]
        self.assertTrue(result['pending_manual'])
        self.assertIn('entailment_input_coverage_incomplete', result['upstream_review_reasons'])

    def test_audit_keeps_scope_metadata_and_drops_text(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {'ZLQL_MODEL_AUDIT_PATH': str(Path(directory) / 'calls.sqlite')}):
            with A.review_audit('window-metadata'):
                A.record('entailment', 'bert_entail', 'predict_window', 'succeeded',
                    clause_token_start=190, clause_token_end=380,
                    article_token_start=254, article_token_end=508,
                    planned_pairs=4, executed_pairs=4, token_coverage_complete=True,
                    pair_coverage_complete=True, semantic_pending=True,
                    window_disagreement=True, decision_scope='local_window_only',
                    source_text='PRIVATE CONTENT', api_key='SECRET')
            details = A.recent_events(trace_id='window-metadata')[0]['details']
        self.assertEqual(details['clause_token_end'], 380)
        self.assertTrue(details['semantic_pending'])
        self.assertEqual(details['decision_scope'], 'local_window_only')
        self.assertNotIn('source_text', details)
        self.assertNotIn('api_key', details)


if __name__ == '__main__':
    unittest.main()
