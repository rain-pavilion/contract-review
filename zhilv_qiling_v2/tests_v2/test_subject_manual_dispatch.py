"""Missing-source manual work cannot keep redispatching unchanged law feedback."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi_run as G
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.tests_v2.test_review_work_items import receipt_call


class SubjectManualDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fact = {"feedback_kind": "subject_fact", "feedback_item_id": "subject-feedback",
            "fact_id": "fact1", "source_id": "", "subject_name": "测试公司", "subject_node_id": "subject1",
            "fact_type": "license", "clause_id": "a", "clause_ids": ["a"],
            "effective_status": "unknown", "source_complete": False}
        self.statute = {"feedback_item_id": "statute-feedback", "candidate_id": "candidate1",
            "article_uid": "law1", "clause_ids": ["a"], "article_quote": "给定法条",
            "manual_required": True}

    def observe(self, *, fact=None, statutes=(), calls=(), policy="cover_all_context_v1", ctx=None):
        fact = copy.deepcopy(self.fact if fact is None else fact)
        if ctx is None:
            ctx = SimpleNamespace(config=Config(llm_feedback_window_policy=policy),
                contract=SimpleNamespace(contract_id="contract1", document=SimpleNamespace(clauses=[
                    SimpleNamespace(clause_id="a", text="给定合同原文")])), shared={"round": 1})
        ctx.shared["feedback_latest"] = {"request": {"calls": list(calls)}}
        request = ("request", list(statutes), {}, {"a"}, list(statutes), [fact])
        with patch.object(G, "_feedback_request", return_value=request), \
                patch.object(G, "_current_feedback_consumption", return_value=[]):
            items, graph = G._review_work_state(ctx)
        return ctx, items, graph

    def test_missing_source_not_dispatched_pending_preserved(self):
        for changes in ({"source_complete": False, "source_id": "source1"},
                        {"source_complete": True, "source_id": ""},
                        {"source_complete": False, "source_id": ""}):
            with self.subTest(changes=changes):
                fact = {**self.fact, **changes}
                _, baseline, _ = self.observe(fact=fact, policy="all_matching_windows")
                ctx, items, graph = self.observe(fact=fact)
                item = items[0]
                self.assertFalse(item["dispatchable"])
                self.assertTrue(item["pending"])
                self.assertFalse(item["actual_processed"])
                self.assertEqual(item["execution_status"], "not_executed")
                self.assertEqual(item["receipt_call_ids"], [])
                self.assertEqual(item["provided_feedback"], fact)
                for key in ("pending_reasons", "source_status", "semantic_status", "protocol_complete",
                            "input_fingerprint", "provided_feedback_fingerprint"):
                    self.assertEqual(item[key], baseline[0][key])
                plan = ctx.shared["incremental_dispatch"].plan(items, graph, 2)
                self.assertEqual(plan["dispatch_work_item_ids"], [])
                self.assertEqual(plan["reused_work_item_ids"], [])
                self.assertIn(item["work_item_id"], ctx.shared["review_work_ledger"].snapshot()["pending_work_item_ids"])

    def test_complete_source_unknown_without_receipt_still_dispatches(self):
        fact = {**self.fact, "source_complete": True, "source_id": "source1"}
        ctx, items, graph = self.observe(fact=fact)
        self.assertTrue(items[0]["dispatchable"])
        self.assertTrue(items[0]["pending"])
        self.assertNotIn("dispatch_policy_reason", items[0])
        self.assertEqual(ctx.shared["incremental_dispatch"].plan(items, graph, 2)["dispatch_work_item_ids"],
                         [items[0]["work_item_id"]])

    def test_complete_source_needs_manual_keeps_real_receipt_and_reuse(self):
        fact = {**self.fact, "source_complete": True, "source_id": "source1"}
        call = receipt_call(fact, error="subject_fact_status_unresolved")
        ctx, items, graph = self.observe(fact=fact, calls=[call])
        item = items[0]
        self.assertTrue(item["dispatchable"])
        self.assertTrue(item["actual_processed"])
        self.assertTrue(item["pending"])
        self.assertEqual(item["semantic_status"], "pending_manual")
        self.assertEqual(item["receipt_call_ids"], ["actual-call"])
        self.assertEqual(ctx.shared["incremental_dispatch"].plan(items, graph, 2)["reused_work_item_ids"],
                         [item["work_item_id"]])

    def test_default_policy_does_not_omit_missing_source(self):
        ctx, items, graph = self.observe(policy="all_matching_windows")
        self.assertTrue(items[0]["dispatchable"])
        self.assertNotIn("dispatch_policy_reason", items[0])
        self.assertEqual(ctx.shared["incremental_dispatch"].plan(items, graph, 2)["dispatch_work_item_ids"],
                         [items[0]["work_item_id"]])

    def test_missing_fact_cannot_force_processed_statute_to_redispatch(self):
        ctx, items, graph = self.observe(statutes=[self.statute], calls=[receipt_call(self.statute)])
        law = next(item for item in items if item["kind"] == "statute_feedback")
        fact = next(item for item in items if item["kind"] == "subject_fact")
        plan = ctx.shared["incremental_dispatch"].plan(items, graph, 2)
        self.assertEqual(plan["dispatch_work_item_ids"], [])
        self.assertEqual(plan["reused_work_item_ids"], [law["work_item_id"]])
        self.assertTrue(fact["pending"])
        self.assertFalse(fact["actual_processed"])
        self.assertTrue(law["pending"])

    def test_new_source_reenables_same_obligation_without_reusing_fake_receipt(self):
        ctx, previous, _ = self.observe()
        fact = {**self.fact, "source_complete": True, "source_id": "source1"}
        ctx, items, graph = self.observe(fact=fact, ctx=ctx)
        self.assertEqual(items[0]["work_item_id"], previous[0]["work_item_id"])
        self.assertNotEqual(items[0]["input_fingerprint"], previous[0]["input_fingerprint"])
        self.assertTrue(items[0]["dispatchable"])
        self.assertFalse(items[0]["actual_processed"])
        self.assertEqual(ctx.shared["incremental_dispatch"].plan(items, graph, 3)["dispatch_work_item_ids"],
                         [items[0]["work_item_id"]])

    def test_frozen_full_loop_reuses_manual_law_without_missing_subject_redispatch(self):
        """Run the actual handlers/cycle/scheduler through two feedback rounds.

        Only node execution is stubbed. The model stub follows the compact
        transport contract: it emits a real-shaped needs_manual receipt for
        the law, and no invented receipt for source-less subject evidence.
        """
        from zhilv_qiling_v2 import nodes as N, rules as R
        from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration
        c = clause("a", "甲公司应按约支付价款并提交经营许可证明。")
        candidate = R._candidate("fee_mismatch", "价款与资质需核对", c, c.text)
        initial = N.NodeOutcome("node2_logic", candidates=[candidate], llm={"complete": True})
        batches = []

        def node_logic(*args, **kwargs):
            laws = kwargs.get("verification_feedback", [])
            facts = kwargs.get("subject_fact_feedback", [])
            if not laws and not facts:
                return copy.deepcopy(initial)
            batches.append(copy.deepcopy({"laws": laws, "facts": facts}))
            self.assertTrue(laws)
            self.assertTrue(facts)
            self.assertTrue(all(not fact.get("source_complete") or not fact.get("source_id") for fact in facts))
            calls = [receipt_call(law, verdict="needs_manual", error="verification_needs_manual") for law in laws]
            return N.NodeOutcome("node2_logic", llm={"complete": False, "feedback_complete": False,
                "calls": calls, "verified_feedback_item_ids": [],
                "uncompleted_feedback_item_ids": [item["feedback_item_id"] for item in [*laws, *facts]]})

        check = {"candidate_id": candidate.candidate_id, "clause_ids": ["a"], "article_uid": "law1",
            "label": "contradict", "manual_required": True, "article_quote": "按约履行给付义务。", "score": .8}
        fact_checks = [{"subject_name": "甲公司", "subject_node_id": "subject1", "clause_ids": ["a"],
            "checks": {"license": "unknown"}, "records": []}]
        with patch.object(N, "node_logic", side_effect=node_logic) as logic_node, \
                patch.object(N, "node_ner", return_value=N.NodeOutcome("node1_ner")), \
                patch.object(N, "node_subject", return_value=N.NodeOutcome("node3_subject", payload={
                    "stats": {"fact_checks": fact_checks}})), \
                patch.object(N, "node_statute", return_value=N.NodeOutcome("node4_statute", payload={
                    "risk_checks": [check]})), \
                patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})), \
                patch.dict(G.HANDLERS, {"reviewer": lambda ctx: {"notes": []}}):
            run = G.run_multi_agent(contract([c]), configuration(review_max_rounds=6,
                freeze_feedback_cycle=True, llm_feedback_window_policy="cover_all_context_v1"),
                retriever=SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[])))

        self.assertEqual(logic_node.call_count, 2)  # one initial plus one feedback batch
        self.assertEqual(len(batches), 1)
        cross = run.diagnostics["cross_validation"]
        self.assertEqual(cross["termination"], "stalled_with_pending")
        self.assertFalse(cross["converged"])
        self.assertFalse(cross["feedback_verification_complete"])
        self.assertEqual(len(cross["rounds"]), 3)  # initial, actual feedback, reuse
        final_plan = cross["rounds"][-1]["incremental_dispatch"]
        self.assertFalse(final_plan["dispatch_work_item_ids"])
        self.assertTrue(final_plan["reused_work_item_ids"])
        cycle = run.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertFalse(cycle["overall_complete"])
        self.assertFalse(cycle["active_attempt_complete"])
        self.assertFalse(cycle["deferred_next_cycle"])
        work = cycle["active_work_items"]
        law = next(item for item in work if item["kind"] == "statute_feedback")
        facts = [item for item in work if item["kind"] == "subject_fact"]
        self.assertTrue(law["pending"])
        self.assertTrue(law["actual_processed"])
        self.assertEqual(law["receipt_call_ids"], ["actual-call"])
        self.assertEqual(final_plan["reused_work_item_ids"], [law["work_item_id"]])
        self.assertTrue(facts)
        self.assertTrue(all(item["pending"] and not item["dispatchable"] and not item["actual_processed"]
                            and not item["receipt_call_ids"] for item in facts))
        self.assertTrue({item["legacy_feedback_item_id"] for item in facts}.issubset(cross["pending_feedback_item_ids"]))
        self.assertTrue(all(item["provided_feedback"]["effective_status"] == "unknown" for item in facts))


if __name__ == "__main__":
    unittest.main()
