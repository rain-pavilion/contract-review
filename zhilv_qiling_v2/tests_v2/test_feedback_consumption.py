"""Current typed receipts reach candidate evidence and human queues; no model calls."""
from __future__ import annotations

import copy
import json
import time
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents as A, agents_multi as M, agents_multi_run as G, nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import Citation, Evidence, RiskCandidate
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract


class FeedbackConsumptionTests(unittest.TestCase):
    def setUp(self):
        self.clause = clause("a", "甲公司与乙公司在同一条款约定付款和主体资质。")
        self.contract = contract([self.clause])
        self.config = Config()
        self.config.enable_llm = False
        self.config.review_threshold = 0.8
        self.first = RiskCandidate("candidate-a", "fee_mismatch", "付款关系待核对", "medium",
            clause_ids=["a"], confidence=0.96, extra={"subject_name": "甲公司"})
        self.second = RiskCandidate("candidate-b", "need_license", "资质关系待核对", "medium",
            clause_ids=["a"], confidence=0.96, extra={"subject_name": "乙公司"})
        self.law = {"feedback_item_id": "law-feedback-1", "candidate_id": self.first.candidate_id,
            "article_uid": "fixture-law", "clause_ids": ["a"], "article_quote": "合成规范仅用于协议核对。",
            "article_quote_truncated": False}
        self.fact = {"feedback_kind": "subject_fact", "feedback_item_id": "fact-feedback-1",
            "candidate_ids": [self.second.candidate_id], "clause_id": "a", "clause_ids": ["a"],
            "fact_id": "fact-1", "source_id": "source-1", "subject_name": "乙公司", "fact_type": "license",
            "effective_status": "invalid", "source_complete": True,
            "fact_evidence_text": "乙公司的合成资质记录待人工确认。", "source_evidence_text": "合成来源，不是真实企业查询。"}

    def context(self, logic=None, subject=None):
        ctx = M.AgentContext(self.contract, self.config, M.MessageBus(strict=False))
        logic = [copy.deepcopy(self.first)] if logic is None else logic
        subject = [] if subject is None else subject
        ctx.shared["logic"] = N.NodeOutcome("node2_logic", candidates=logic)
        ctx.shared["subject"] = N.NodeOutcome("node3_subject", candidates=subject)
        citations = {candidate.candidate_id: [Citation("合成规范", "第1条", "仅作测试。", article_uid="fixture-law")]
                     for candidate in logic + subject}
        for candidate in logic + subject:
            for alias in candidate.extra.get("alias_candidate_ids", []):
                citations[alias] = citations[candidate.candidate_id]
        ctx.shared["statute"] = N.NodeOutcome("node4_statute", payload={"per_candidate": citations, "risk_checks": []})
        ctx.shared["round"] = 3
        return ctx

    def request(self, fingerprint="current-fingerprint", laws=None, facts=None):
        laws = [copy.deepcopy(self.law)] if laws is None else laws
        facts = [] if facts is None else facts
        return (fingerprint, [], {}, {"a"}, laws, facts)

    def row(self, item, verdict="confirmed"):
        fields = ("feedback_item_id", "fact_id", "source_id", "subject_name", "fact_type", "clause_id") if item.get("feedback_kind") == "subject_fact" else (
            "feedback_item_id", "candidate_id", "article_uid")
        row = {field: item[field] for field in fields}
        row.update(verdict=verdict, reason="已核对给定材料关系，法律后果仍须人工判断。", clause_quote=self.clause.text)
        if item.get("feedback_kind") == "subject_fact":
            row.update(fact_quote=item["fact_evidence_text"], source_quote=item["source_evidence_text"])
        else:
            row["article_quote"] = item["article_quote"]
        return row

    def snapshot(self, items, verdict="confirmed"):
        calls = []
        for index, item in enumerate(items):
            root = "subject_fact_verifications" if item.get("feedback_kind") == "subject_fact" else "verifications"
            item_id = item["feedback_item_id"]
            calls.append({"job_id": f"job-{index}", "window_id": f"window-{index}", "status": "ok",
                "feedback_item_ids": [item_id], root: [self.row(item, verdict)],
                "verified_feedback_item_ids": [item_id], "verification_errors": []})
        return {"calls": calls, "verified_feedback_item_ids": [item["feedback_item_id"] for item in items],
            "uncompleted_feedback_item_ids": [], "unmatched_feedback_item_ids": [],
            "complete": True, "feedback_complete": True}

    def judge(self, ctx, request):
        with patch.object(G, "_feedback_request", return_value=request), \
             patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})):
            G._handle_judge(ctx)
        return ctx.shared["fused"]

    def test_confirmed_receipt_reaches_candidate_final_evidence_and_json_without_score_change(self):
        ctx = self.context()
        baseline = A.fuse(copy.deepcopy(ctx.shared["logic"].candidates), [A.AgentResult("node4_statute",
            citations=ctx.shared["statute"].payload["per_candidate"])], self.contract, self.config)
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot([self.law])}
        risks = self.judge(ctx, self.request())
        self.assertEqual(len(risks), 1)
        self.assertFalse(risks[0].review_required)
        self.assertEqual(A.score_risks(risks, self.config), A.score_risks(baseline, self.config))
        self.assertEqual(risks[0].confidence, baseline[0].confidence)
        evidence = [value for value in risks[0].evidence if value.kind == "model_feedback"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].strength, 0.0)
        self.assertIn("confirmed", evidence[0].text)
        self.assertIn(self.law["article_quote"], evidence[0].text)
        self.assertTrue(evidence[0].extra["protocol_complete"])
        self.assertEqual(evidence[0].extra["observations"][0]["clause_quote_mode"], "unspecified")
        self.assertEqual(ctx.shared["logic"].candidates[0].extra["model_feedback_current"][0]["article_uid"], "fixture-law")
        serialized = json.dumps(risks[0].to_dict(), ensure_ascii=False)
        self.assertIn("clause_quote", serialized)
        self.assertIn("已核对给定材料关系", serialized)

    def test_native_source_selection_mode_is_visible_without_promoting_model_verdict(self):
        ctx = self.context()
        meta = self.snapshot([self.law], "rejected")
        meta["calls"][0]["clause_quote_mode"] = "native_source_extract_selection"
        ctx.shared["feedback_latest"] = {"current-fingerprint": meta}
        risks = self.judge(ctx, self.request())
        evidence = risks[0].evidence[-1]
        self.assertEqual(evidence.extra["observations"][0]["clause_quote_mode"], "native_source_extract_selection")
        self.assertIn("合同引文来自原文片段选择，相关性仍需复核", evidence.text)
        self.assertEqual(evidence.extra["verdicts"], ["rejected"])
        self.assertTrue(risks[0].review_required)

    def test_rejected_protocol_stays_complete_and_enters_actual_report_review_queue(self):
        ctx = self.context()
        meta = self.snapshot([self.law], "rejected")
        ctx.shared["feedback_latest"] = {"current-fingerprint": meta}
        ctx.shared["processed_feedback"] = {"current-fingerprint"}
        risks = self.judge(ctx, self.request())
        self.assertTrue(risks[0].review_required)
        self.assertTrue(meta["feedback_complete"])
        self.assertEqual(ctx.shared["processed_feedback"], {"current-fingerprint"})
        detail = risks[0].evidence[-1].extra
        self.assertTrue(detail["protocol_complete"])
        self.assertEqual(detail["semantic_status"], "pending_manual")
        run = G.MultiAgentRun(self.contract.contract_id, [], 100, risks=risks)
        with patch.object(G, "run_multi_agent", return_value=run):
            report = ReviewEngine(self.config, retriever=False)._run_multi_agent(self.contract, time.time())
        self.assertEqual(report.review_queue, [risks[0].risk_id])

    def test_candidate_alias_receives_the_exact_receipt(self):
        kept = copy.deepcopy(self.first)
        kept.candidate_id = "kept-candidate"
        kept.extra["alias_candidate_ids"] = [self.first.candidate_id]
        ctx = self.context(logic=[kept])
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot([self.law], "rejected")}
        risks = self.judge(ctx, self.request())
        self.assertTrue(risks[0].review_required)
        self.assertEqual(risks[0].evidence[-1].extra["candidate_ids"], [self.first.candidate_id])

    def test_refreshed_subject_candidate_gets_fact_only_by_explicit_candidate_ids(self):
        ctx = self.context(subject=[copy.deepcopy(self.second)])
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot([self.fact], "rejected")}
        risks = self.judge(ctx, self.request(laws=[], facts=[self.fact]))
        first = next(risk for risk in risks if risk.code == self.first.code)
        second = next(risk for risk in risks if risk.code == self.second.code)
        self.assertFalse(any(e.kind == "model_feedback" for e in first.evidence))
        evidence = next(e for e in second.evidence if e.kind == "model_feedback")
        self.assertEqual(evidence.extra["fact_id"], "fact-1")
        self.assertEqual(evidence.extra["source_id"], "source-1")
        self.assertIn("乙公司", evidence.text)
        self.assertTrue(second.review_required)
        self.assertFalse(first.review_required)

    def test_needs_manual_wrong_binding_and_missing_rows_remain_visible_and_queued(self):
        for kind in ("needs_manual", "wrong_binding", "missing"):
            with self.subTest(kind=kind):
                ctx = self.context()
                meta = self.snapshot([self.law])
                call = meta["calls"][0]
                if kind == "needs_manual":
                    call["verifications"][0]["verdict"] = "needs_manual"
                    call["verification_errors"] = [{"feedback_item_id": self.law["feedback_item_id"], "reason": "verification_needs_manual"}]
                    call["verified_feedback_item_ids"] = []
                elif kind == "wrong_binding":
                    call["verifications"][0]["candidate_id"] = "another-candidate"
                else:
                    call["verifications"] = []
                ctx.shared["feedback_latest"] = {"current-fingerprint": meta}
                risks = self.judge(ctx, self.request())
                evidence = risks[0].evidence[-1]
                self.assertTrue(risks[0].review_required)
                self.assertFalse(evidence.extra["protocol_complete"])
                self.assertTrue(evidence.extra["validation_errors"])
                self.assertIn("人工待复核", evidence.text)

    def test_cross_window_disagreement_is_semantic_pending_without_rewriting_protocol(self):
        ctx = self.context()
        meta = self.snapshot([self.law])
        second = copy.deepcopy(meta["calls"][0])
        second.update(job_id="second-job", window_id="second-window")
        second["verifications"][0]["verdict"] = "rejected"
        meta["calls"].append(second)
        ctx.shared["feedback_latest"] = {"current-fingerprint": meta}
        risks = self.judge(ctx, self.request())
        record = risks[0].evidence[-1].extra
        self.assertTrue(record["protocol_complete"])
        self.assertTrue(record["disagreement"])
        self.assertEqual(record["verdicts"], ["confirmed", "rejected"])
        self.assertTrue(risks[0].review_required)
        self.assertTrue(meta["complete"])
        self.assertTrue(meta["feedback_complete"])

    def test_old_fingerprint_cannot_release_new_input_and_old_candidate_evidence_is_removed(self):
        ctx = self.context()
        old = self.snapshot([self.law])
        ctx.shared["feedback_latest"] = {"old-fingerprint": old}
        self.judge(ctx, self.request("old-fingerprint"))
        changed = {**self.law, "feedback_item_id": "law-new-source", "article_quote": "新的来源材料。"}
        risks = self.judge(ctx, self.request("new-fingerprint", laws=[changed]))
        evidence = [e for e in risks[0].evidence if e.kind == "model_feedback"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].extra["feedback_fingerprint"], "new-fingerprint")
        self.assertFalse(evidence[0].extra["protocol_complete"])
        self.assertIn("current_fingerprint_not_attempted", evidence[0].extra["validation_errors"])
        self.assertTrue(risks[0].review_required)
        self.assertEqual(ctx.shared["feedback_latest"]["old-fingerprint"], old)

    def test_repeated_projection_is_idempotent_and_confirmed_does_not_clear_other_review_reasons(self):
        ctx = self.context()
        ctx.shared["statute"].payload["risk_checks"] = [{"candidate_id": self.first.candidate_id, "label": "contradict"}]
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot([self.law])}
        self.judge(ctx, self.request())
        risks = self.judge(ctx, self.request())
        self.assertEqual(len([e for e in risks[0].evidence if e.kind == "model_feedback"]), 1)
        self.assertTrue(risks[0].review_required)

    def test_unknown_without_candidate_creates_no_illegality_candidate(self):
        ctx = self.context(logic=[], subject=[])
        item = {**self.fact, "candidate_ids": [], "source_id": "", "source_complete": False,
                "effective_status": "unknown", "source_evidence_text": ""}
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot([item])}
        risks = self.judge(ctx, self.request(laws=[], facts=[item]))
        self.assertEqual(ctx.shared["logic"].candidates, [])
        self.assertEqual(ctx.shared["subject"].candidates, [])
        self.assertTrue(all(risk.code == "no_substantive_risk" for risk in risks))
        record = ctx.shared["logic"].llm["feedback_consumption_current"][0]
        self.assertFalse(record["protocol_complete"])
        self.assertTrue(record["pending_manual"])

    def test_statute_id_uses_all_normalized_content_not_upstream_stale_id(self):
        original = {**self.law, "source": {"url": "https://source.invalid/a"}, "feedback_item_id": "stale"}
        first = G._statute_feedback_item(original)
        self.assertEqual(first["feedback_item_id"], G._statute_feedback_item(dict(reversed(list(original.items()))))["feedback_item_id"])
        for changes in ({"article_quote": "新引文"}, {"candidate_id": "other"},
                        {"source": {"url": "https://source.invalid/b"}}, {"label": "neutral"}):
            with self.subTest(changes=changes):
                self.assertNotEqual(first["feedback_item_id"], G._statute_feedback_item({**original, **changes})["feedback_item_id"])
        long = {**original, "article_quote": "前" * 1100}
        self.assertNotEqual(G._statute_feedback_item(long)["feedback_item_id"],
                            G._statute_feedback_item({**long, "article_quote": "前" * 1099 + "后"})["feedback_item_id"])

    def test_subject_feedback_builder_does_not_assign_every_candidate_on_shared_clause(self):
        ctx = self.context()
        record = {"fact_id": "record", "fact_type": "license", "effective_status": "invalid", "source": {}}
        checked = {"subject_name": "乙公司", "subject_node_id": "S:seller:乙公司", "clause_ids": ["a"],
                   "checks": {"license": "invalid"}, "records": [record]}
        second = copy.deepcopy(self.second)
        second.extra["subject_node_id"] = "S:seller:乙公司"
        payload = {"fact_checks": [checked], "candidates": [self.first.to_dict(), second.to_dict()]}
        items = G._subject_fact_feedback(ctx, payload)
        self.assertEqual(items[0]["candidate_ids"], [self.second.candidate_id])
        payload["candidates"] = [self.first.to_dict()]
        self.assertEqual(G._subject_fact_feedback(ctx, payload)[0]["candidate_ids"], [])

    def same_named_payload(self, *, bound=True, candidate_nodes=False, collision=False):
        name = "同名科技有限公司"
        checks, candidates = [], []
        for role, suffix in (("buyer", "A"), ("seller", "B")):
            record = {"fact_id": "fact-" + suffix, "fact_type": "license", "effective_status": "invalid",
                "subject_name": name, "subject_id": "ID-" + suffix, "source": {}}
            checked = {"subject_name": name, "subject_node_id": f"S:{role}:{name}", "subject_role": role,
                "clause_ids": ["a"], "checks": {"license": "invalid"}, "records": [record]}
            if bound:
                checked.update(identity_binding={"status": "bound", "identifiers": ["ID-" + suffix]},
                    subject_identifiers=["ID-" + suffix], identity_binding_status="bound")
                record["identity_match"] = "name_and_identifier"
            extra = {"subject_name": name, "fact_id": record["fact_id"], "fact_type": "license"}
            if candidate_nodes:
                extra.update(subject_node_id=checked["subject_node_id"], subject_role=role, subject_id=record["subject_id"])
            candidates.append({"candidate_id": "collision" if collision else "candidate-" + suffix,
                "clause_ids": ["a"], "extra": extra})
            checks.append(checked)
        return {"fact_checks": checks, "candidates": candidates}

    def test_same_name_legacy_candidates_need_exact_fact_and_bound_identifier(self):
        items = G._subject_fact_feedback(self.context(), self.same_named_payload())
        self.assertEqual({item["fact_id"]: item["candidate_ids"] for item in items},
                         {"fact-A": ["candidate-A"], "fact-B": ["candidate-B"]})
        unbound = G._subject_fact_feedback(self.context(), self.same_named_payload(bound=False))
        self.assertTrue(all(not item["candidate_ids"] for item in unbound))
        self.assertTrue(all(item["candidate_binding_status"] == "pending_manual" for item in unbound))

    def test_same_name_explicit_nodes_do_not_cross_role_or_identifier(self):
        payload = self.same_named_payload(candidate_nodes=True)
        # Matching a different fact/node is never rescued by the shared name.
        items = G._subject_fact_feedback(self.context(), payload)
        for item in items:
            suffix = item["fact_id"][-1]
            self.assertEqual(item["candidate_ids"], ["candidate-" + suffix])
            self.assertEqual(item["subject_id"], "ID-" + suffix)
        payload["candidates"][0]["extra"]["subject_node_id"] = payload["fact_checks"][1]["subject_node_id"]
        changed = G._subject_fact_feedback(self.context(), payload)
        self.assertEqual(next(item for item in changed if item["fact_id"] == "fact-A")["candidate_ids"], [])

    def test_colliding_legacy_candidate_id_is_not_attached_and_reason_is_retained(self):
        items = G._subject_fact_feedback(self.context(), self.same_named_payload(collision=True))
        self.assertTrue(all(item["candidate_ids"] == [] for item in items))
        self.assertTrue(all("ambiguous_candidate_identity:collision" in item["candidate_binding_errors"] for item in items))
        ctx = self.context()
        ctx.shared["feedback_latest"] = {"current-fingerprint": self.snapshot(items)}
        with patch.object(G, "_feedback_request", return_value=self.request(laws=[], facts=items)):
            records = G._current_feedback_consumption(ctx)
        self.assertTrue(all(record["pending_manual"] for record in records))
        self.assertTrue(all(record["candidate_binding_errors"] for record in records))

    def test_explicit_request_candidate_is_preserved_without_same_clause_expansion(self):
        payload = self.same_named_payload()
        record = payload["fact_checks"][0]["records"][0]
        payload["fact_checks"] = []
        payload["verification_results"] = [{"request_id": "request-A", "candidate_id": "selected-only",
            "subject_name": record["subject_name"], "subject_node_id": "S:buyer:" + record["subject_name"],
            "fact_type": "license", "fact_status": "invalid", "status": "completed", "clause_ids": ["a"],
            "records": [record]}]
        items = G._subject_fact_feedback(self.context(), payload)
        # The record's legacy candidate lacks an explicit node and the request
        # does not supply its identity binding, so only the exact request ID is used.
        self.assertEqual(items[0]["candidate_ids"], ["selected-only"])

    def test_real_same_name_negative_subject_facts_keep_distinct_ids_scope_and_fusion(self):
        from zhilv_qiling_v2.tests_v2.test_subject_identity_binding import make_contract, fact, check, NAME_A, ID_A, ID_B
        scoped = make_contract([("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_A}，统一社会信用代码：{ID_B}。"),
            ("service", f"{NAME_A}的资质许可证应持续有效。")], [(NAME_A, "buyer"), (NAME_A, "seller")])
        result = check(scoped, [fact(status="invalid"), fact(identifier=ID_B, status="invalid")])
        candidates = [candidate for candidate in result.candidates
                      if any(evidence.kind == "reviewed_subject_fact" for evidence in candidate.evidence)]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 2)
        self.assertEqual(len({candidate.dedupe_key for candidate in candidates}), 2)
        self.assertEqual({candidate.extra["subject_role"]: candidate.clause_ids for candidate in candidates},
                         {"buyer": ["a"], "seller": ["b"]})
        fused = A.fuse(copy.deepcopy(candidates), [], scoped, self.config)
        self.assertEqual(len(fused), 2)
        ctx = M.AgentContext(scoped, self.config, M.MessageBus(strict=False))
        items = G._subject_fact_feedback(ctx, {"fact_checks": result.stats["fact_checks"],
            "candidates": [candidate.to_dict() for candidate in candidates]})
        targets = {candidate.extra["fact_id"]: candidate.candidate_id for candidate in candidates}
        for item in items:
            if item["candidate_ids"]:
                self.assertEqual(item["candidate_ids"], [targets[item["fact_id"]]])
                target = next(candidate for candidate in candidates if candidate.candidate_id == item["candidate_ids"][0])
                self.assertEqual(item["subject_node_id"], target.extra["subject_node_id"])
                self.assertEqual(item["subject_id"], target.extra["subject_id"])


if __name__ == "__main__":
    unittest.main()
