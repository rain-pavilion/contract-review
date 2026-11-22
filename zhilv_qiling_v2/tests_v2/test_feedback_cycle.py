"""Frozen-cycle safety fixtures. No model, network, or legal-quality claims."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.feedback_cycle import FeedbackCycle
from zhilv_qiling_v2.feedback_dependencies import dependency_graph, IncrementalDispatch
from zhilv_qiling_v2.review_work_items import feedback_work_items
from zhilv_qiling_v2.tests_v2.test_review_work_items import receipt_call


class CycleStateTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"a": "合同原文"}
        self.law = {"feedback_item_id": "law-1", "candidate_id": "candidate-1",
            "article_uid": "law", "clause_ids": ["a"], "article_quote": "第一版法条"}
        self.request = ("live-1", [self.law], {}, {"a"}, [self.law], [])

    def items(self, cycle, calls=(), consumption=()):
        _, _, subject, _, laws, facts = cycle.request()
        return feedback_work_items("contract", laws, facts, subject,
            {"calls": list(calls)}, list(consumption), self.sources)

    def test_snapshot_is_deep_and_live_new_node_does_not_expand_active_scc(self):
        fact = {"feedback_kind": "subject_fact", "feedback_item_id": "fact-1", "fact_id": "f",
            "source_id": "s", "subject_name": "甲", "fact_type": "license",
            "clause_id": "a", "clause_ids": ["a"]}
        request = (*self.request[:5], [fact])
        cycle = FeedbackCycle("contract", self.sources, request)
        active = self.items(cycle)
        graph = dependency_graph(active)
        cycle.bind_graph(active, graph)
        before = cycle.request()
        changed = copy.deepcopy(self.law)
        changed.update(candidate_id="candidate-2", feedback_item_id="law-2")
        cycle.observe(("live-2", [self.law, changed], {}, {"a"}, [self.law, changed], [fact]))
        cycle.bind_graph(self.items(cycle), dependency_graph(self.items(cycle)))
        self.assertEqual(cycle.request(), before)
        self.assertEqual(cycle.snapshot()["active_dependency_graph"]["sccs"], graph["sccs"])
        self.assertEqual(len(cycle.deferred_items()), 1)
        queued = cycle.deferred_items()[0]
        self.assertTrue(queued["pending"])
        self.assertFalse(queued["dispatchable"])
        self.assertNotIn(queued["work_item_id"], graph["nodes"])
        before[4][0]["article_quote"] = "caller mutation"
        self.assertEqual(cycle.request()[4][0]["article_quote"], "第一版法条")

    def test_source_change_is_pending_new_version_not_replacement_receipt(self):
        cycle = FeedbackCycle("contract", self.sources, self.request)
        changed = {**self.law, "article_quote": "第二版真实法条", "feedback_item_id": "law-new"}
        cycle.observe(("new", [changed], {}, {"a"}, [changed], []))
        self.assertEqual(cycle.request()[4][0]["article_quote"], "第一版法条")
        queued = cycle.deferred_items()[0]
        self.assertTrue(queued["supersedes_active_version"])
        self.assertEqual(queued["provided_feedback"]["article_quote"], "第二版真实法条")
        self.assertFalse(queued["actual_processed"])
        self.assertEqual(queued["receipt_call_ids"], [])
        self.assertTrue(cycle.deferred_affects(self.law))
        self.assertFalse(cycle.snapshot()["overall_complete"])

    def test_transport_only_reaudit_is_not_another_obligation(self):
        original = {**self.law, "window_audit": {"call_id": "one", "elapsed_seconds": 1}}
        cycle = FeedbackCycle("contract", self.sources, ("a", [], {}, {"a"}, [original], []))
        newer = {**original, "feedback_item_id": "transport-2", "window_audit": {"call_id": "two", "elapsed_seconds": 2}}
        cycle.observe(("b", [], {}, {"a"}, [newer], []))
        self.assertEqual(cycle.deferred_items(), [])

    def test_only_exact_bound_receipts_are_reusable(self):
        cycle = FeedbackCycle("contract", self.sources, self.request)
        call = receipt_call(self.law)
        good = self.items(cycle, [call])
        scheduler = IncrementalDispatch()
        graph = dependency_graph(good)
        scheduler.record_processed(good, graph)
        self.assertTrue(scheduler.plan(good, graph, 2)["reused_work_item_ids"])
        call["request_receipts"][0]["source_sha256"] = "wrong-source"
        bad = self.items(cycle, [call])
        self.assertFalse(bad[0]["actual_processed"])
        fresh_scheduler = IncrementalDispatch()
        fresh_scheduler.record_processed(bad, graph)
        self.assertFalse(fresh_scheduler.plan(bad, graph, 2)["reused_work_item_ids"])
        call.pop("request_receipts")
        self.assertFalse(self.items(cycle, [call])[0]["actual_processed"])

    def test_explicit_advance_keeps_queued_newer_version_with_no_receipt_transfer(self):
        cycle = FeedbackCycle("contract", self.sources, self.request)
        call = receipt_call(self.law)
        cycle.update_status(self.items(cycle, [call]))
        changed = {**self.law, "article_quote": "新证据", "feedback_item_id": "law-new"}
        cycle.observe(("new", [changed], {}, {"a"}, [changed], []))
        advanced = FeedbackCycle.advance(cycle.snapshot(), contract_id="contract",
            clause_sources=self.sources, request=self.request)  # deliberately stale live view
        self.assertEqual(advanced.cycle_number, 2)
        self.assertNotEqual(advanced.key, cycle.key)
        self.assertEqual(advanced.request()[4][0]["article_quote"], "新证据")
        self.assertFalse(advanced.snapshot()["active_attempt_complete"])
        self.assertFalse(self.items(advanced)[0]["actual_processed"])
        self.assertTrue(advanced.snapshot()["prior_cycle_archive"]["active_work_items"][0]["receipt_call_ids"])

    def test_advance_rejects_changed_contract_source_or_tampered_snapshot(self):
        snapshot = FeedbackCycle("contract", self.sources, self.request).snapshot()
        with self.assertRaisesRegex(ValueError, "scope_mismatch"):
            FeedbackCycle.advance(snapshot, contract_id="contract", clause_sources={"a": "new"}, request=self.request)
        snapshot["active_request"]["statutes"][0]["article_quote"] = "tampered"
        with self.assertRaisesRegex(ValueError, "snapshot_mismatch"):
            FeedbackCycle.advance(snapshot, contract_id="contract", clause_sources=self.sources, request=self.request)


class CycleIntegrationTests(unittest.TestCase):
    def fixture(self, *, freeze=True, manual=False, omit_receipts=False, new_candidate=False,
                change_source=False, initial_pending=False, unknown_subject=False, no_responses=False,
                continuation=None):
        from zhilv_qiling_v2 import agents_multi_run as G, nodes as N, rules as R
        from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration
        c = clause("a", "甲公司应当按时支付合同价款。")
        candidate = R._candidate("fee_mismatch", "价款核对", c, c.text)
        new = R._candidate("date_conflict", "新候选", c, c.text)
        initial_meta = {"complete": not initial_pending}
        if initial_pending:
            initial_meta["initial_review_lineage"] = [{"root_job_id": "root", "window_id": "w",
                "source_sha256": "source", "complete": False,
                "pending_reasons": ["initial_output_more_findings_possible"],
                "attempts": [{"content": {"more_findings_possible": True}}]}]
        initial = N.NodeOutcome("node2_logic", candidates=[candidate], llm=initial_meta)
        generated, visits = [], []
        def logic(*args, **kwargs):
            inputs = [*kwargs.get("verification_feedback", []), *kwargs.get("subject_fact_feedback", [])]
            if not inputs:
                return copy.deepcopy(initial)
            generated.append(copy.deepcopy(inputs))
            calls, verified, uncompleted = [], [], []
            if not no_responses:
                for index, item in enumerate(inputs):
                    needs_manual = manual or item.get("feedback_kind") == "subject_fact"
                    call = receipt_call(item, verdict="needs_manual" if needs_manual else "confirmed",
                        error="verification_needs_manual" if needs_manual else "")
                    call_id = f"synthetic-cycle-{len(generated)}-{index}"
                    call["call_id"] = call_id
                    call["request_receipts"][0]["call_id"] = call_id
                    if omit_receipts:
                        call.pop("request_receipts")
                    calls.append(call)
                    (uncompleted if needs_manual else verified).append(item["feedback_item_id"])
            else:
                uncompleted = [item["feedback_item_id"] for item in inputs]
            complete = not uncompleted and not no_responses
            return N.NodeOutcome("node2_logic", candidates=[new] if new_candidate else [], llm={
                "complete": complete, "feedback_complete": complete, "calls": calls,
                "verified_feedback_item_ids": verified, "uncompleted_feedback_item_ids": uncompleted})
        def statute(_contract, candidates, *_args, **_kwargs):
            visits.append(1)
            quote = "新来源法条" if change_source and len(visits) > 1 else "第一版法条"
            return N.NodeOutcome("node4_statute", payload={"risk_checks": [{
                "candidate_id": row.candidate_id, "clause_ids": ["a"], "article_uid": "law",
                "label": "contradict", "manual_required": manual,
                "article_quote": quote, "score": .8} for row in candidates]})
        facts = [{"subject_name": "甲公司", "subject_node_id": "subject-1", "clause_ids": ["a"],
            "checks": {"license": "unknown"}, "records": []}] if unknown_subject else []
        with patch.object(N, "node_logic", side_effect=logic), \
             patch.object(N, "node_ner", return_value=N.NodeOutcome("node1_ner")), \
             patch.object(N, "node_subject", return_value=N.NodeOutcome("node3_subject", payload={"stats": {"fact_checks": facts}})), \
             patch.object(N, "node_statute", side_effect=statute), \
             patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})), \
             patch.dict(G.HANDLERS, {"reviewer": lambda ctx: {"notes": []}}):
            run = G.run_multi_agent(contract([c]), configuration(review_max_rounds=5,
                freeze_feedback_cycle=freeze), retriever=SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[])),
                feedback_cycle_state=continuation)
        return run, generated

    def test_opt_in_default_is_legacy_and_legacy_can_converge(self):
        self.assertFalse(Config().freeze_feedback_cycle)
        run, calls = self.fixture(freeze=False)
        self.assertEqual(len(calls), 1)
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])
        self.assertNotIn("feedback_cycle", run.diagnostics["logic_llm"])

    def test_manual_response_is_processed_once_but_stalls_open(self):
        run, calls = self.fixture(manual=True)
        cycle = run.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(cycle["active_attempt_complete"])
        self.assertFalse(cycle["active_cycle_complete"])
        self.assertEqual(run.diagnostics["cross_validation"]["termination"], "stalled_with_pending")
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertTrue(run.diagnostics["review_work_items"]["pending_work_item_ids"])

    def test_new_candidate_is_deferred_not_convergence_or_active_key_reset(self):
        run, calls = self.fixture(new_candidate=True)
        cycle = run.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(cycle["active_attempt_complete"])
        self.assertTrue(cycle["deferred_next_cycle"])
        self.assertFalse(cycle["overall_complete"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertFalse(run.diagnostics["cross_validation"]["feedback_verification_complete"])
        all_pending = set(run.diagnostics["review_work_items"]["pending_work_item_ids"])
        self.assertTrue(set(cycle["deferred_pending_work_item_ids"]).issubset(all_pending))

    def test_changed_source_confirmation_is_only_historical_and_stays_manual(self):
        run, calls = self.fixture(change_source=True)
        cycle = run.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["article_quote"], "第一版法条")
        self.assertTrue(cycle["deferred_next_cycle"])
        active = [item for item in cycle["active_work_items"] if item["kind"] == "statute_feedback"][0]
        self.assertTrue(active["actual_processed"])
        self.assertTrue(active["pending"])
        self.assertIn("newer_evidence_deferred_next_cycle", active["pending_reasons"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])

    def test_missing_receipt_cannot_complete_or_reuse(self):
        run, _ = self.fixture(omit_receipts=True)
        cycle = run.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertFalse(cycle["active_attempt_complete"])
        self.assertFalse(run.diagnostics["logic_llm"]["feedback_runs_complete"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertFalse(any(event["reused_work_item_ids"] for event in run.diagnostics["incremental_dispatch"]["events"]))

    def test_exhausted_response_budget_keeps_all_unprocessed_work_open(self):
        run, _ = self.fixture(no_responses=True)
        self.assertEqual(run.diagnostics["cross_validation"]["termination"], "round_limit")
        self.assertFalse(run.diagnostics["logic_llm"]["feedback_cycle"]["active_attempt_complete"])
        self.assertTrue(run.diagnostics["review_work_items"]["pending_work_item_ids"])

    def test_initial_more_and_unknown_external_facts_cannot_pass(self):
        for kwargs in ({"initial_pending": True}, {"unknown_subject": True}):
            with self.subTest(kwargs=kwargs):
                run, _ = self.fixture(**kwargs)
                self.assertFalse(run.diagnostics["cross_validation"]["converged"])
                self.assertFalse(run.diagnostics["logic_llm"]["feedback_cycle"]["overall_complete"])
                self.assertTrue(run.diagnostics["review_work_items"]["pending_work_item_ids"])

    def test_explicit_run_continuation_rebinds_new_cycle(self):
        first, _ = self.fixture(change_source=True)
        state = first.diagnostics["logic_llm"]["feedback_cycle"]
        second, calls = self.fixture(continuation=state)
        current = second.diagnostics["logic_llm"]["feedback_cycle"]
        self.assertEqual(current["cycle_number"], 2)
        self.assertEqual(current["parent_cycle_id"], state["cycle_id"])
        self.assertTrue(calls)  # no inherited processed/ack set; a new bound response is required
        self.assertEqual(calls[0][0]["article_quote"], "新来源法条")


class CycleContinuationEvidenceTests(unittest.TestCase):
    def setUp(self):
        from zhilv_qiling_v2.subject_facts import validate_fact
        self.sources = {"a": "甲公司营业执照核验条款", "b": "另一个条款"}
        self.pending = {"request_id": "request-1", "candidate_id": "candidate-1",
            "clause_ids": ["a"], "article_uid": "law-1", "risk_code": "need_license",
            "subject_name": "甲公司", "subject_node_id": "subject-1", "subject_id": "12345678",
            "fact_type": "license", "status": "pending", "fact_status": "unknown",
            "records": [], "fact_ids": []}
        subject = {"pending_verification_request_ids": ["request-1"],
            "verification_complete": False, "verification_results": [self.pending]}
        self.cycle = FeedbackCycle("contract", self.sources, ("old", [], subject, {"a"}, [], []))
        _, _, saved_subject, _, _, _ = self.cycle.request()
        self.cycle.update_status(feedback_work_items("contract", [], [], saved_subject,
            {}, [], self.sources))
        record = validate_fact({"subject_name": "甲公司", "subject_id": "12345678",
            "fact_type": "license", "status": "invalid", "source": {
                "url": "https://source.invalid/official-record", "publisher": "离线测试来源",
                "reviewer": "fixture-reviewer", "verification_method": "合成协议测试",
                "verified_at": "2026-11-18", "actor_kind": "human"}})
        record.update(effective_status="invalid", identity_match="name_and_identifier",
            review_reasons=[], review_reason="", identity_review_reason="")
        self.completed = {**copy.deepcopy(self.pending), "status": "completed",
            "fact_status": "invalid", "fact_ids": [record["fact_id"]], "records": [record],
            "identity_binding": {"status": "bound", "identifiers": ["12345678"]},
            "reason": "bound_reviewed_local_evidence", "reasons": []}

    def advance(self, completed=None, *, pending_ids=()):
        row = copy.deepcopy(self.completed if completed is None else completed)
        subject = {"pending_verification_request_ids": list(pending_ids),
            "verification_complete": not pending_ids, "verification_results": [row]}
        return FeedbackCycle.advance(self.cycle.snapshot(), contract_id="contract",
            clause_sources=self.sources, request=("new", [], subject, {"a"}, [], []))

    def request_ids(self, cycle):
        return cycle.request()[2].get("pending_verification_request_ids", [])

    def test_bound_reviewed_completed_result_releases_only_prior_request(self):
        advanced = self.advance()
        self.assertEqual(self.request_ids(advanced), [])
        self.assertEqual(advanced.active_versions, {})
        self.assertFalse(advanced.snapshot()["active_attempt_complete"])
        self.assertTrue(advanced.snapshot()["prior_cycle_archive"]["active_pending_work_item_ids"])

    def test_bare_completed_or_missing_source_cannot_release_prior_request(self):
        mutations = [lambda row: row.update(records=[], fact_ids=[]),
            lambda row: row["records"][0].pop("source"),
            lambda row: row["records"][0]["source"].pop("reviewer"),
            lambda row: row["records"][0]["source"].update(url="https://"),
            lambda row: row["records"][0].update(effective_status="unknown"),
            lambda row: row.update(fact_ids=["unbound-fact"]),
            lambda row: row["records"][0].update(record_sha256="wrong-hash"),
            lambda row: row.update(identity_binding=None),
            lambda row: row.update(fact_ids=[{}]),
            lambda row: row.update(clause_ids=[{}]),
            lambda row: row.update(fact_type=[]),
            lambda row: row.update(fact_status=[])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(self.completed)
                mutate(row)
                self.assertEqual(self.request_ids(self.advance(row)), ["request-1"])

    def test_wrong_identity_or_scope_and_explicit_pending_cannot_release(self):
        mutations = [lambda row: row.update(subject_id="87654321"),
            lambda row: row.update(candidate_id="candidate-other"),
            lambda row: row.update(clause_ids=["b"]),
            lambda row: row.update(subject_name="乙公司"),
            lambda row: row["identity_binding"].update(status="ambiguous"),
            lambda row: row["records"][0].update(identity_match="unresolved")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(self.completed)
                mutate(row)
                self.assertEqual(self.request_ids(self.advance(row)), ["request-1"])
        self.assertEqual(self.request_ids(self.advance(pending_ids=["request-1"])), ["request-1"])

    def test_unselected_opposite_record_cannot_be_hidden_by_fact_ids(self):
        from zhilv_qiling_v2.subject_facts import validate_fact
        row = copy.deepcopy(self.completed)
        opposite = validate_fact({"subject_name": "甲公司", "subject_id": "12345678",
            "fact_type": "license", "status": "valid", "valid_until": "2027-11-18",
            "source": copy.deepcopy(row["records"][0]["source"])})
        opposite.update(effective_status="valid", identity_match="name_and_identifier",
            review_reasons=[], review_reason="", identity_review_reason="")
        row["records"].append(opposite)
        self.assertNotIn(opposite["fact_id"], row["fact_ids"])
        self.assertEqual(self.request_ids(self.advance(row)), ["request-1"])
        opposite["effective_status"] = "conflicting_evidence"
        self.assertEqual(self.request_ids(self.advance(row)), ["request-1"])
        # Unknown observations do not contradict known, reviewed evidence.
        opposite["effective_status"] = "unknown"
        self.assertEqual(self.request_ids(self.advance(row)), [])

    def test_no_feedback_cannot_claim_an_execution_attempt(self):
        for statuses in ([], [{"kind": "initial_review", "work_item_id": "initial", "pending": False}],
                         [{"kind": "subject_request", "work_item_id": "request", "pending": True}]):
            with self.subTest(statuses=statuses):
                self.cycle.update_status(statuses)
                self.assertFalse(self.cycle.snapshot()["active_attempt_complete"])


if __name__ == "__main__":
    unittest.main()
