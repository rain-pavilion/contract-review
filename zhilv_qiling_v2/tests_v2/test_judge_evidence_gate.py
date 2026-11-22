"""The aggregation step cannot upgrade unsubstantiated model claims."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from zhilv_qiling_v2 import agents, nodes
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import FinalRisk, Evidence
from zhilv_qiling_v2.judge_evidence import risk_basis, summary_basis, render_focus


def risk(key='r1', status=None, manual=False, model=False):
    evidence = []
    if status is not None:
        evidence.append(Evidence('claim_grounding', 'original assessment', source='claim_grounding',
            extra={'claim_grounding_status': status, 'reasons': ['citation_not_supporting'] if status != 'supported' else []}))
    if model:
        evidence.append(Evidence('llm_quote', 'exact original words', source='node2_llm'))
    return FinalRisk(risk_id=key, code='fee_mismatch', category='履约风险', severity='high',
        message='同一尾款金额存在两项不同约定', confidence=.9, review_required=manual, evidence=evidence)


class JudgeEvidenceGateTest(unittest.TestCase):
    def test_unsubstantiated_claims_retained_but_not_deducted(self):
        items = [risk('u', 'unsupported', True, True), risk('x', 'unknown', True, True)]
        score, detail = agents.score_risks(items, Config())
        self.assertEqual(score, 100)
        self.assertEqual(len(detail['excluded_unsubstantiated']), 2)
        self.assertTrue(detail['not_a_clearance_or_accuracy'])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].evidence[0].extra['claim_grounding_status'], 'unsupported')

    def test_supported_and_deterministic_findings_keep_score_semantics(self):
        for item in (risk(), risk(status='supported', model=True)):
            score, detail = agents.score_risks([item], Config())
            self.assertEqual(score, 84)
            self.assertFalse(detail['excluded_unsubstantiated'])
            self.assertFalse(risk_basis(item)['legal_truth_established'])

    def test_unassessed_model_claim_is_not_treated_as_rule(self):
        item = risk(model=True)
        self.assertEqual(risk_basis(item)['text_support_status'], 'not_assessed')
        self.assertEqual(agents.score_risks([item], Config())[0], 100)

    def test_unknown_or_unsupported_assessment_is_monotone(self):
        item = risk(status='supported', model=True)
        item.evidence.append(Evidence('claim_grounding', 'second evidence', extra={'claim_grounding_status':'unknown'}))
        self.assertFalse(risk_basis(item)['score_eligible'])

    def test_no_model_still_gets_evidence_bounded_summary(self):
        result = nodes.node_judge([risk(status='unsupported', manual=True)], Config(enable_llm=False))
        self.assertIn('1项模型主张的文本支持性未确认', result.payload['summary'])
        self.assertIn('不是准确率', result.payload['summary'])
        self.assertEqual(result.payload['evidence_basis']['unsubstantiated_count'], 1)

    def test_model_free_text_and_unknown_risk_ids_never_become_conclusions(self):
        cfg = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='test')
        result = SimpleNamespace(content={'summary':'该合同已被确认违法', 'focus':['虚构风险'],
                                         'focus_risk_ids':['invented','r2']}, model='test', elapsed=.1)
        with patch.object(nodes.L, 'available', return_value=True), patch.object(nodes.L, 'request_json', return_value=result) as call:
            outcome = nodes.node_judge([risk('r1'),risk('r2','unknown',True,True)],cfg)
        self.assertNotIn('已被确认违法', outcome.payload['summary'])
        self.assertNotIn('虚构风险', str(outcome.payload['focus']))
        self.assertEqual(outcome.llm['accepted_risk_ids'], ['r2'])
        self.assertTrue(outcome.payload['focus'][0].startswith('主张待核实'))
        self.assertIn('evidence_state', call.call_args.args[1][1]['content'])
        self.assertTrue(call.call_args.kwargs['json_schema'])

    def test_no_candidate_notice_does_not_trigger_judge_model(self):
        item = risk(); item.code='no_substantive_risk'
        cfg = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='test')
        with patch.object(nodes.L, 'available', return_value=True), patch.object(nodes.L, 'request_json') as call:
            outcome = nodes.node_judge([item],cfg)
        call.assert_not_called()
        self.assertIn('不代表审查通过', outcome.payload['summary'])

    def test_arbitrary_model_focus_values_are_ignored(self):
        self.assertEqual(len(render_focus([risk()], [None, {}, 'invented'])),1)


if __name__=='__main__': unittest.main()
