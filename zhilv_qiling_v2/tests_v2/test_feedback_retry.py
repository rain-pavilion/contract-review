"""Core retry/convergence contracts. Mocked protocol responses are not gold."""
from types import SimpleNamespace
import copy
import unittest
from unittest.mock import patch
from zhilv_qiling_v2 import agents_multi_run as G, nodes as N, rules as R
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration
from zhilv_qiling_v2.tests_v2.protocol_fixtures import synthetic_feedback_calls


class FeedbackRetryTests(unittest.TestCase):
    def run_loop(self, responses, rounds=4, label='contradict', manual=False):
        c = clause('a', '甲方的付款金额与结算总价不一致。')
        candidate = R._candidate('fee_mismatch', '金额核对', c, c.text)
        initial = N.NodeOutcome('node2_logic', candidates=[candidate], llm={'complete':True})
        check = {'candidate_id': candidate.candidate_id, 'clause_ids':['a'],
            'article_uid':'fixture-law', 'label':label, 'manual_required':manual, 'score':0.8}
        statute = N.NodeOutcome('node4_statute', payload={'risk_checks':[check]})
        def logic(*args, **kwargs):
            if not kwargs.get('verification_feedback'):
                return copy.deepcopy(initial)
            meta = responses.pop(0) if len(responses)>1 else responses[0]
            if meta.get('complete') is True and meta.get('feedback_complete') is True:
                # Explicit SYNTHETIC/non-gold binding fixture. Booleans alone
                # are deliberately not treated as execution evidence.
                items = kwargs.get('verification_feedback') or []
                meta['calls'] = synthetic_feedback_calls(items, index=len(responses))
                meta['verified_feedback_item_ids'] = [item['feedback_item_id'] for item in items]
            return N.NodeOutcome('node2_logic', llm=copy.deepcopy(meta))
        with patch.object(N, 'node_logic', side_effect=logic) as calls, \
             patch.object(N, 'node_subject', return_value=N.NodeOutcome('node3_subject', payload={'stats':{'fact_checks':[]}})), \
             patch.object(N, 'node_statute', return_value=statute), \
             patch.object(N, 'node_judge', return_value=N.NodeOutcome('node5_judge', payload={'score':100})), \
             patch.dict(G.HANDLERS, {'reviewer':lambda ctx:{'notes':[]}}):
            result = G.run_multi_agent(contract([c]), configuration(review_max_rounds=rounds),
                retriever=SimpleNamespace(search=lambda *a, **k:SimpleNamespace(hits=[])))
        return result, calls.call_count

    def test_failed_feedback_is_retried_and_success_not_repeated(self):
        failed = {'complete':False, 'feedback_complete':False, 'uncompleted_feedback_item_ids':['fb1']}
        success = {'complete':True, 'feedback_complete':True, 'uncompleted_feedback_item_ids':[]}
        run, count = self.run_loop([failed, success])
        self.assertEqual(count, 3)  # initial + failed + successful; stable round does not repeat
        self.assertTrue(run.diagnostics['cross_validation']['feedback_verification_complete'])
        self.assertTrue(run.diagnostics['cross_validation']['converged'])
        logic = run.diagnostics['logic_llm']
        self.assertEqual(len(logic['feedback_runs']), 2)
        self.assertEqual(logic['feedback_latest_runs'], [success])
        self.assertTrue(logic['feedback_runs_complete'])
        self.assertFalse(logic['feedback_all_attempts_complete'])

    def test_unchanged_candidates_cannot_converge_with_unverified_feedback(self):
        failed = {'complete':False, 'feedback_complete':False, 'uncompleted_feedback_item_ids':['fb1']}
        run, count = self.run_loop([failed], rounds=3)
        self.assertEqual(count, 3)
        status = run.diagnostics['cross_validation']
        self.assertFalse(status['converged'])
        self.assertEqual(status['termination'], 'round_limit')
        expected = {item['feedback_item_id'] for item in
            run.diagnostics['logic_llm']['feedback_current_request']['items']} | {'fb1'}
        self.assertEqual(set(status['pending_feedback_item_ids']), expected)
        self.assertFalse(run.diagnostics['logic_llm']['feedback_runs_complete'])

    def test_empty_defects_is_not_feedback_acknowledgment(self):
        run, count = self.run_loop([{'complete':True,'feedback_complete':False,
            'uncompleted_feedback_item_ids':['fb1']}], rounds=3)
        self.assertEqual(count, 3)
        self.assertFalse(run.diagnostics['cross_validation']['feedback_verification_complete'])

    def test_manual_required_neutral_checks_also_get_feedback(self):
        run, count = self.run_loop([{'complete':True,'feedback_complete':True,
            'uncompleted_feedback_item_ids':[]}], label='neutral', manual=True)
        self.assertEqual(count, 2)
        # Stricter three-axis semantics: protocol ack does not erase the
        # upstream manual review flag, even for a neutral relation.
        self.assertFalse(run.diagnostics['cross_validation']['converged'])
        self.assertEqual(run.diagnostics['cross_validation']['termination'], 'stalled_with_pending')
        self.assertTrue(run.diagnostics['review_work_items']['pending_work_item_ids'])
