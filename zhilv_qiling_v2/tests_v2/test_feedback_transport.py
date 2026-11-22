"""Synthetic protocol fixtures; actual-loop failure is preserved separately."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zhilv_qiling_v2.feedback_transport import select_feedback_windows
from zhilv_qiling_v2.feedback_transport import project_feedback
from zhilv_qiling_v2.tests_v2 import test_native_feedback_atomic as F


class FeedbackTransportTests(unittest.TestCase):
    def test_selected_context_calls_once_and_does_not_claim_omitted_window_executed(self):
        from zhilv_qiling_v2 import nodes, long_context
        from zhilv_qiling_v2.config import Config
        fixture = F.NativeFeedbackAtomicTests()
        fixture.setUp()
        segments = tuple(long_context.Segment(c.clause_id,0,len(c.text),c.text) for c in fixture.rows)
        windows = [long_context.ReviewWindow("full","text",segments),
                   long_context.ReviewWindow("cross","cross_clause",segments[:1])]
        config = Config(enable_llm=True,llm_endpoint="http://localhost:1/v1",llm_model="OFFLINE-STUB",
                        llm_feedback_window_policy="cover_all_context_v1")
        reply = SimpleNamespace(content={"verifications":[{
            "feedback_item_id":"f-1","candidate_id":"candidate-1","article_uid":"law-1",
            "verdict":"needs_manual","reason":"需人工核对适用关系",
            "clause_quote":fixture.a.text,"article_quote":fixture.item["article_quote"]}]},
            model="OFFLINE-STUB",degraded="")
        with patch.object(long_context,"build_review_plan",return_value=(windows,{"cross_windows":1})), patch.object(
                nodes.L,"request_json",return_value=reply) as request:
            _, meta, _ = nodes._llm_logic(fixture.contract,config,fixture.rows,verification_feedback=[fixture.item])
        self.assertEqual(request.call_count,1)
        self.assertEqual(meta["planned_calls"],1)
        self.assertEqual(meta["cross_windows_completed"],0)
        self.assertIn("f-1",meta["uncompleted_feedback_item_ids"])
        sent = json.loads(request.call_args.args[1][-1]["content"])
        self.assertEqual(set(sent["clause_sources"]),{"a","b"})

    def test_only_literal_superset_context_is_deduplicated(self):
        def window(key, text, start=0):
            return SimpleNamespace(window_id=key, kind="text", segments=[
                SimpleNamespace(clause_id="a", text=text, start=start, end=start+len(text))])
        windows = [window("full", "付款前须验收"), window("part", "须验收", 3)]
        selected, audit = select_feedback_windows([{"feedback_item_id":"f","clause_ids":["a"]}], windows)
        self.assertEqual(selected["f"], {"full"})
        self.assertEqual(audit[0]["omitted_duplicate_context_window_ids"], ["part"])
        self.assertFalse(audit[0]["omitted_windows_executed"])
        for altered in (window("other","先验收",3),window("other","须验收",0)):
            selected, _ = select_feedback_windows([{"feedback_item_id":"f","clause_ids":["a"]}], [windows[0],altered])
            self.assertEqual(selected["f"], {"full","other"})

    def test_unknown_subject_stays_pending_without_invented_model_call(self):
        from zhilv_qiling_v2.tests_v2.test_subject_fact_feedback import SubjectFactProtocolTests
        from zhilv_qiling_v2 import nodes
        fixture = SubjectFactProtocolTests()
        fixture.setUp()
        fixture.config.llm_feedback_window_policy = "cover_all_context_v1"
        item = {**fixture.item,"source_complete":False,"source_id":"","effective_status":"unknown"}
        with patch.object(nodes.L, "request_json") as request:
            candidates,meta,_ = nodes._llm_logic(fixture.contract,fixture.config,[fixture.clause],
                                                  subject_fact_feedback=[item])
        request.assert_not_called()
        self.assertEqual(candidates, [])
        self.assertFalse(meta["feedback_complete"])
        self.assertIn(item["feedback_item_id"],meta["uncompleted_feedback_item_ids"])
        self.assertEqual(meta["calls"],[])

    def test_large_realistic_window_audit_is_sent_and_original_is_retained(self):
        fixture=F.NativeFeedbackAtomicTests(methodName="test_native_batches_are_single_items_with_truthful_plan_and_receipts")
        fixture.setUp()
        item={**fixture.item,"manual_required":True,"semantic_pending":True,"label":"neutral",
            "window_audit":{"planned_pairs":64,"executed_pairs":64,"manual_required":True,"semantic_pending":True,
                "decision_scope":"local_window_only","token_coverage_complete":True,"pair_coverage_complete":True,
                "window_predictions":[{"label":"neutral","tokens":["synthetic"]*50} for _ in range(64)]}}
        prior=copy.deepcopy(item)
        self.assertGreater(len(json.dumps(item)),4000)
        _,meta,degraded,captured,_=fixture.review(items=[item],max_calls=1)
        self.assertEqual(len(captured),1)
        sent=captured[0]["payload"]["verification_feedback"][0]
        self.assertLess(len(json.dumps(sent,ensure_ascii=False)),4000)
        self.assertTrue(sent["manual_required"])
        self.assertTrue(sent["window_audit"]["semantic_pending"])
        self.assertEqual(sent["article_quote"],item["article_quote"])
        self.assertEqual(meta["feedback_transport_audits"][0]["original_feedback"],prior)
        self.assertEqual(item,prior)
        self.assertFalse(any("oversize" in value for value in degraded))

    def test_projection_is_deterministic_and_small_payload_is_untouched(self):
        item={"window_audit":{"manual_required":True}}
        self.assertEqual(project_feedback(item),(item,None))
        large={"window_audit":{"manual_required":True,"predictions":["x"]*2000}}
        self.assertEqual(project_feedback(large),project_feedback(large))

if __name__=="__main__":
    unittest.main()
