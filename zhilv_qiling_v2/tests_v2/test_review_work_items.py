"""Offline work/dispatch safety tests; fixtures are not legal gold."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.review_work_items import (
    ReviewWorkLedger, feedback_work_items, initial_work_items,
)
from zhilv_qiling_v2.feedback_dependencies import (
    IncrementalDispatch, dependency_graph, strongly_connected_components,
)


def receipt_call(item, *, verdict="needs_manual", error="verification_needs_manual"):
    call = {"call_id": "actual-call", "window_id": "window", "job_id": "window:batch:1",
        "source_sha256": "source-hash", "status": "feedback_incomplete" if error else "ok",
        "feedback_item_ids": [item["feedback_item_id"]],
        "request_receipts": [{"call_id": "actual-call", "window_id": "window",
            "job_id": "window:batch:1", "source_sha256": "source-hash", "status": "succeeded"}],
        "verification_errors": [{"feedback_item_id": item["feedback_item_id"], "reason": error}] if error else [],
        "verified_feedback_item_ids": [] if error else [item["feedback_item_id"]]}
    fields = (("fact_id", "source_id", "subject_name", "fact_type", "clause_id")
              if item.get("feedback_kind") == "subject_fact" else ("candidate_id", "article_uid"))
    row = {key: item.get(key) for key in fields}
    row.update(feedback_item_id=item["feedback_item_id"], verdict=verdict,
               reason="Fixture result only", clause_quote="原文", article_quote="法条")
    call["subject_fact_verifications" if item.get("feedback_kind") == "subject_fact" else "verifications"] = [row]
    return call


class WorkItemTests(unittest.TestCase):
    def setUp(self):
        self.statute = {"feedback_item_id": "old-id", "candidate_id": "candidate", "article_uid": "law",
            "article_quote": "完整法条", "clause_ids": ["a"], "manual_required": True}

    def work(self, item=None, call=None):
        item = copy.deepcopy(item or self.statute)
        call = call if call is not None else receipt_call(item)
        return feedback_work_items("contract", [item], [], {}, {"calls": [call]}, [], {"a": "合同原文"})

    def test_initial_execution_does_not_clear_more_true(self):
        call = receipt_call(self.statute)
        call.update(root_job_id="root", clause_ids=["a"], source_spans=[{"clause_id": "a", "start": 0, "end": 4}])
        root = {"root_job_id": "root", "window_id": "window", "source_sha256": "source-hash",
            "complete": False, "can_continue": False, "pending_reasons": ["initial_output_more_findings_possible"],
            "attempts": [{"content": {"defects": [], "more_findings_possible": True}}]}
        item = initial_work_items("contract", {"initial_review_lineage": [root], "calls": [call]})[0]
        self.assertEqual(item["execution_status"], "executed")
        self.assertEqual(item["semantic_status"], "output_pending")
        self.assertTrue(item["pending"])
        self.assertTrue(item["lineage"]["attempts"][0]["content"]["more_findings_possible"])

    def test_initial_complete_requires_source_and_real_bound_receipt(self):
        call = receipt_call(self.statute, verdict="confirmed", error="")
        call.update(root_job_id="root", clause_ids=["a"])
        root = {"root_job_id": "root", "window_id": "window", "source_sha256": "source-hash",
            "complete": True, "pending_reasons": []}
        def work(current_root, current_call):
            return initial_work_items("contract", {"initial_review_lineage": [current_root], "calls": [current_call]})[0]
        self.assertFalse(work(root, call)["pending"])
        no_receipt = copy.deepcopy(call)
        no_receipt.pop("request_receipts")
        self.assertTrue(work(root, no_receipt)["pending"])
        self.assertEqual(work(root, no_receipt)["lineage"], root)
        wrong_source = {**root, "source_sha256": "different-source"}
        self.assertTrue(work(wrong_source, call)["pending"])
        self.assertTrue(work({**root, "source_sha256": ""}, call)["pending"])
        self.assertTrue(work({**root, "pending_reasons": ["initial_output_more_findings_possible"]}, call)["pending"])

    def test_valid_needs_manual_can_be_processed_but_stays_pending(self):
        item = self.work()[0]
        self.assertTrue(item["actual_processed"])
        self.assertTrue(item["pending"])
        self.assertEqual(item["semantic_status"], "pending_manual")

    def test_missing_or_cross_bound_receipt_never_reusable(self):
        call = receipt_call(self.statute)
        call["request_receipts"][0]["source_sha256"] = "other-source"
        self.assertFalse(self.work(call=call)[0]["actual_processed"])
        call = receipt_call(self.statute)
        call["verifications"][0]["candidate_id"] = "other-candidate"
        self.assertFalse(self.work(call=call)[0]["actual_processed"])

    def test_invalid_quote_response_is_not_processed(self):
        item = self.work(call=receipt_call(self.statute, error="clause_quote_not_in_bound_window"))[0]
        self.assertFalse(item["actual_processed"])

    def test_only_recovered_complete_retry_receipts_are_allowed(self):
        for reason in ("json_retry", "response_format_retry"):
            call = receipt_call(self.statute)
            call["request_receipts"][0].update(status="degraded", reason=reason)
            self.assertTrue(self.work(call=call)[0]["actual_processed"])
        for status, reason in (("degraded", "output_truncated"), ("started", ""), ("failed", "transport_failed")):
            call = receipt_call(self.statute)
            call["request_receipts"][0].update(status=status, reason=reason)
            self.assertFalse(self.work(call=call)[0]["actual_processed"])

    def test_unknown_subject_source_remains_pending_after_real_processing(self):
        fact = {"feedback_kind": "subject_fact", "feedback_item_id": "fact-fb", "fact_id": "missing-fact",
            "source_id": "", "subject_name": "甲公司", "subject_node_id": "subject-a", "fact_type": "license",
            "clause_id": "a", "clause_ids": ["a"], "effective_status": "unknown", "source_complete": False}
        call = receipt_call(fact, error="subject_fact_source_incomplete")
        item = feedback_work_items("contract", [], [fact], {}, {"calls": [call]}, [], {"a": "合同原文"})[0]
        self.assertTrue(item["actual_processed"])
        self.assertEqual(item["source_status"], "external_evidence_pending")
        self.assertTrue(item["pending"])

    def test_changed_source_reopens_same_stable_obligation(self):
        old = self.work()[0]
        new = self.work({**self.statute, "article_quote": "变更后的法条"})[0]
        self.assertEqual(old["work_item_id"], new["work_item_id"])
        self.assertNotEqual(old["input_fingerprint"], new["input_fingerprint"])

    def test_transport_uuid_is_not_new_evidence_but_provided_input_kept(self):
        first = {**self.statute, "window_audit": {"call_id": "one", "window_predictions": [{"label": "neutral", "elapsed_seconds": 1}]}}
        second = {**self.statute, "feedback_item_id": "new-audit-id", "window_audit": {"call_id": "two", "window_predictions": [{"label": "neutral", "elapsed_seconds": 2}]}}
        a, b = self.work(first)[0], self.work(second)[0]
        self.assertEqual(a["input_fingerprint"], b["input_fingerprint"])
        self.assertNotEqual(a["provided_feedback_fingerprint"], b["provided_feedback_fingerprint"])

    def test_ledger_roundtrip_including_retired_items(self):
        ledger = ReviewWorkLedger()
        ledger.observe(self.work(), 1)
        ledger.observe([], 2)
        self.assertEqual(ReviewWorkLedger.replay(ledger.events).snapshot(), ledger.snapshot())


class DependencyTests(WorkItemTests):
    def test_scc_is_deterministic(self):
        self.assertEqual(strongly_connected_components(["c", "a", "b"], [("a", "b"), ("b", "a"), ("b", "c")]), [["a", "b"], ["c"]])

    def test_only_real_unchanged_work_is_reused(self):
        scheduler = IncrementalDispatch()
        items = self.work()
        graph = dependency_graph(items)
        self.assertEqual(len(scheduler.plan(items, graph, 1)["dispatch_work_item_ids"]), 1)
        scheduler.record_processed(items, graph)
        self.assertEqual(len(scheduler.plan(items, graph, 2)["reused_work_item_ids"]), 1)
        self.assertTrue(items[0]["pending"])
        changed = self.work({**self.statute, "article_quote": "新法条原文"})
        self.assertEqual(len(scheduler.plan(changed, dependency_graph(changed), 3)["dispatch_work_item_ids"]), 1)

    def test_new_or_changed_scc_dependency_invalidates_reuse(self):
        scheduler = IncrementalDispatch()
        items = self.work()
        scheduler.record_processed(items, dependency_graph(items))
        dependency = copy.deepcopy(items[0])
        dependency.update(work_item_id="new-dependency", kind="subject_fact", input_fingerprint="new-source")
        with_dependency = [*items, dependency]
        plan = scheduler.plan(with_dependency, dependency_graph(with_dependency), 2)
        self.assertEqual(set(plan["dispatch_work_item_ids"]), {item["work_item_id"] for item in with_dependency})


class IntegratedDispatchTests(unittest.TestCase):
    def run_fixture(self, *, changed=False, confirmed=False, omit_receipts=False):
        from zhilv_qiling_v2 import agents_multi_run as G, nodes as N, rules as R
        from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration
        c = clause("a", "甲方应当按时支付合同价款。")
        candidate = R._candidate("fee_mismatch", "核对价款", c, c.text)
        initial = N.NodeOutcome("node2_logic", candidates=[candidate], llm={"complete": True})
        check = {"candidate_id": candidate.candidate_id, "clause_ids": ["a"], "article_uid": "law",
            "label": "contradict", "manual_required": not confirmed, "article_quote": "第一版法条", "score": 0.8}
        generated = []
        def logic(*args, **kwargs):
            if not kwargs.get("verification_feedback"):
                return copy.deepcopy(initial)
            item = kwargs["verification_feedback"][0]
            generated.append(copy.deepcopy(item))
            call = receipt_call(item, verdict="confirmed" if confirmed else "needs_manual",
                error="" if confirmed else "verification_needs_manual")
            if omit_receipts:
                call.pop("request_receipts")
            return N.NodeOutcome("node2_logic", llm={"complete": confirmed, "feedback_complete": confirmed,
                "calls": [call], "verified_feedback_item_ids": [item["feedback_item_id"]] if confirmed else [],
                "uncompleted_feedback_item_ids": [] if confirmed else [item["feedback_item_id"]]})
        visits = []
        def statute(*args, **kwargs):
            visits.append(1)
            row = {**check, "article_quote": "第二版法条" if changed and len(visits) > 1 else "第一版法条"}
            return N.NodeOutcome("node4_statute", payload={"risk_checks": [row]})
        with patch.object(N, "node_logic", side_effect=logic), \
             patch.object(N, "node_ner", return_value=N.NodeOutcome("node1_ner")), \
             patch.object(N, "node_subject", return_value=N.NodeOutcome("node3_subject", payload={"stats": {"fact_checks": []}})), \
             patch.object(N, "node_statute", side_effect=statute), \
             patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})), \
             patch.dict(G.HANDLERS, {"reviewer": lambda ctx: {"notes": []}}):
            run = G.run_multi_agent(contract([c]), configuration(review_max_rounds=6),
                retriever=SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[])))
        return run, generated

    def test_same_pending_input_reused_without_fake_convergence(self):
        run, generated = self.run_fixture()
        self.assertEqual(len(generated), 1)
        state = run.diagnostics["cross_validation"]
        self.assertEqual(state["termination"], "stalled_with_pending")
        self.assertFalse(state["converged"])
        self.assertTrue(state["pending_feedback_item_ids"])
        self.assertFalse(run.diagnostics["logic_llm"]["feedback_runs_complete"])
        self.assertTrue(run.diagnostics["incremental_dispatch"]["events"][-1]["reused_work_item_ids"])

    def test_changed_source_is_really_redispatched(self):
        run, generated = self.run_fixture(changed=True)
        self.assertEqual([item["article_quote"] for item in generated], ["第一版法条", "第二版法条"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])

    def test_bound_complete_confirmation_can_converge(self):
        run, generated = self.run_fixture(confirmed=True)
        self.assertEqual(len(generated), 1)
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])
        self.assertFalse(run.diagnostics["review_work_items"]["pending_work_item_ids"])

    def test_complete_booleans_without_receipts_cannot_converge(self):
        run, _ = self.run_fixture(confirmed=True, omit_receipts=True)
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertTrue(run.diagnostics["review_work_items"]["pending_work_item_ids"])


if __name__ == "__main__":
    unittest.main()
