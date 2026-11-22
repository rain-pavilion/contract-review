"""Subject→logic protocol and convergence fixtures; no real model or gold labels."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi_run as G, nodes as N
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration


class SubjectFactProtocolTests(unittest.TestCase):
    def setUp(self):
        self.clause = clause("a", "测试科技有限公司提供技术服务，并对其资质真实性负责。")
        self.contract = contract([self.clause])
        self.config = configuration(llm_cross_clause_pairs=0)
        self.item = {"feedback_item_id": "sf1", "fact_id": "fact1", "source_id": "source1",
            "subject_name": "测试科技有限公司", "fact_type": "license", "effective_status": "invalid",
            "clause_id": "a", "clause_ids": ["a"], "candidate_ids": ["candidate1"],
            "source_complete": True, "source": {"url": "https://source.invalid/record"},
            "fact_evidence_text": "导入事实fact1：测试科技有限公司的license记录为invalid。",
            "source_evidence_text": "测试来源机构；核验时间2026-11-14；人工核验记录。"}

    def ack(self, item=None, **changes):
        item = item or self.item
        row = {key: item[key] for key in ("feedback_item_id", "fact_id", "source_id", "subject_name", "fact_type", "clause_id")}
        row.update(verdict="confirmed", reason="已比对给定记录和对应条款；仍需人工判断法律后果。",
            clause_quote=self.clause.text, fact_quote=item["fact_evidence_text"],
            source_quote=item["source_evidence_text"])
        row.update(changes)
        return row

    def run_reply(self, content, items=None, degraded=""):
        reply = SimpleNamespace(content=content, model="fixture", degraded=degraded)
        with patch.object(N.L, "request_json", return_value=reply) as request:
            values = N._llm_logic(self.contract, self.config, [self.clause],
                subject_fact_feedback=[self.item] if items is None else items)
        return (*values, request)

    def test_fact_source_and_clause_are_actually_in_prompt_without_statute_ids(self):
        _, meta, _, request = self.run_reply({"subject_fact_verifications": [self.ack()]})
        payload = json.loads(request.call_args.args[1][-1]["content"])
        sent = payload["subject_fact_feedback"][0]
        for key in ("fact_id", "source_id", "subject_name", "fact_type", "clause_id", "candidate_ids", "source"):
            self.assertEqual(sent[key], self.item[key])
        self.assertNotIn("article_uid", sent)
        self.assertNotIn("article_uid", meta["feedback_items"][0])
        self.assertEqual(payload["verification_feedback"], [])
        self.assertIn("subject_fact_verifications:array", request.call_args.kwargs["schema_hint"])
        self.assertTrue(meta["feedback_complete"])
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["verified_feedback_item_ids"], ["sf1"])
        self.assertEqual(meta["feedback_protocol"], "bound-subject-fact-verifications-v1")
        self.assertFalse(meta["calls"][0]["subject_fact_verifications"][0]["independently_verified"])

    def test_wrong_source_fact_subject_type_and_clause_remain_pending(self):
        for field in ("source_id", "fact_id", "subject_name", "fact_type", "clause_id"):
            with self.subTest(field=field):
                _, meta, _, _ = self.run_reply({"subject_fact_verifications": [self.ack(**{field: "wrong"})]})
                self.assertFalse(meta["feedback_complete"])
                self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf1"])
                self.assertIn("subject_fact_" + field + "_mismatch",
                    [row["reason"] for row in meta["calls"][0]["verification_errors"]])

    def test_missing_ack_or_statute_ack_does_not_verify_subject_fact(self):
        for content in ({"defects": []}, {"verifications": [{"feedback_item_id": "sf1"}]},
                        {"subject_fact_verifications": []}):
            with self.subTest(content=content):
                _, meta, _, _ = self.run_reply(content)
                self.assertFalse(meta["feedback_complete"])
                self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf1"])

    def test_missing_item_does_not_discard_successful_other_item(self):
        second = {**self.item, "feedback_item_id": "sf2", "fact_id": "fact2"}
        _, meta, _, _ = self.run_reply({"subject_fact_verifications": [self.ack()]}, [self.item, second])
        self.assertEqual(meta["verified_feedback_item_ids"], ["sf1"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf2"])
        self.assertEqual(meta["subject_fact_max_items_per_batch"], 1)

    def test_unknown_without_source_cannot_become_illegality(self):
        item = {**self.item, "source_id": "", "source_complete": False,
                "effective_status": "unknown", "source_evidence_text": "", "source": {}}
        content = {"subject_fact_verifications": [self.ack(item)], "defects": [
            {"clause_id": "a", "code": "illegal_terms", "message": "无证违法", "quote": self.clause.text}]}
        candidates, meta, degraded, request = self.run_reply(content, [item])
        self.assertEqual(candidates, [])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf1"])
        self.assertIn("llm_subject_fact_defects_blocked_pending_evidence", degraded)
        self.assertIn("不能推断违法", request.call_args.args[1][0]["content"])

    def test_needs_manual_upstream_pending_and_truncation_stay_pending(self):
        cases = [({"upstream_pending": True}, {}, "subject_fact_upstream_request_pending"),
                 ({}, {"verdict": "needs_manual"}, "subject_fact_verification_needs_manual"),
                 ({"fact_evidence_text_truncated": True}, {}, "subject_fact_evidence_truncated")]
        for changes, ack_changes, reason in cases:
            with self.subTest(reason=reason):
                item = {**self.item, **changes}
                _, meta, _, _ = self.run_reply({"subject_fact_verifications": [self.ack(item, **ack_changes)]}, [item])
                self.assertFalse(meta["complete"])
                self.assertIn(reason, [row["reason"] for row in meta["calls"][0]["verification_errors"]])

    def test_quotes_must_match_the_bound_fact_source_and_window(self):
        for field in ("clause_quote", "fact_quote", "source_quote"):
            with self.subTest(field=field):
                _, meta, _, _ = self.run_reply({"subject_fact_verifications": [self.ack(**{field: "其他记录的引文"})]})
                self.assertFalse(meta["feedback_complete"])
        other = clause("b", "其他主体条款。")
        with patch.object(N.L, "request_json", return_value=SimpleNamespace(content={"subject_fact_verifications": [self.ack()]}, model="fixture")):
            _, meta, _ = N._llm_logic(self.contract, self.config, [other], subject_fact_feedback=[self.item])
        self.assertEqual(meta["unmatched_feedback_item_ids"], ["sf1"])

    def test_duplicate_or_unexpected_ack_and_partial_output_are_pending(self):
        contents = [({"subject_fact_verifications": [self.ack(), self.ack()]}, ""),
                    ({"subject_fact_verifications": [self.ack(), self.ack(feedback_item_id="other")]}, ""),
                    ({"subject_fact_verifications": [self.ack()]}, "output_truncated"),
                    ({"subject_fact_verifications": [self.ack(verdict=[])]}, "")]
        for content, degraded in contents:
            with self.subTest(content=content, degraded=degraded):
                _, meta, _, _ = self.run_reply(content, degraded=degraded)
                self.assertFalse(meta["feedback_complete"])

    def test_statute_and_fact_protocols_keep_separate_prompts_and_shared_accounting(self):
        statute = {"feedback_item_id": "law-feedback", "candidate_id": "law-candidate", "article_uid": "law-1",
            "clause_ids": ["a"], "article_quote": "合成规范只用于协议测试。"}
        sent = []
        def reply(config, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            sent.append(payload)
            if payload.get("subject_fact_feedback"):
                content = {"subject_fact_verifications": [self.ack()]}
            else:
                content = {"verifications": [{"feedback_item_id": "law-feedback", "candidate_id": "law-candidate",
                    "article_uid": "law-1", "verdict": "confirmed", "reason": "合成协议核对。",
                    "clause_quote": self.clause.text, "article_quote": statute["article_quote"]}]}
            return SimpleNamespace(content=content, model="fixture")
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, _ = N._llm_logic(self.contract, self.config, [self.clause],
                verification_feedback=[statute], subject_fact_feedback=[self.item])
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(not (payload.get("verification_feedback") and payload.get("subject_fact_feedback")) for payload in sent))
        self.assertTrue(meta["feedback_complete"])
        self.assertEqual(set(meta["verified_feedback_item_ids"]), {"law-feedback", "sf1"})
        self.assertEqual(meta["feedback_protocol"], "mixed-bound-item-verifications-v1")

    def test_subject_batch_budget_preserves_all_unreviewed_ids(self):
        items = [{**self.item, "feedback_item_id": f"sf{i}", "fact_id": f"fact{i}"} for i in range(5)]
        seen = []
        self.config.llm_max_calls = 2
        def reply(config, messages, **kwargs):
            batch = json.loads(messages[-1]["content"])["subject_fact_feedback"]
            seen.extend(batch)
            self.assertEqual(len(batch), 1)
            return SimpleNamespace(content={"subject_fact_verifications": [self.ack(batch[0])]}, model="fixture")
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, degraded = N._llm_logic(self.contract, self.config, [self.clause], subject_fact_feedback=items)
        self.assertEqual(len(seen), 2)
        self.assertEqual(meta["verified_feedback_item_ids"], ["sf0", "sf1"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf2", "sf3", "sf4"])
        self.assertFalse(meta["feedback_complete"])
        self.assertIn("llm_call_budget_exhausted", degraded)


class SubjectFeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.clause = clause("a", "测试科技有限公司提供技术服务。")
        self.source = {"url": "https://source.invalid/evidence", "publisher": "合成来源机构", "verified_at": "2026-11-14",
            "reviewer": "fixture-reviewer", "verification_method": "合成协议测试，不是真实查验"}
        self.record = {"fact_id": "record-1", "subject_name": "测试科技有限公司", "fact_type": "license",
                       "effective_status": "invalid", "source": self.source}

    def checked(self, record=None):
        record = self.record if record is None else record
        return {"subject_name": "测试科技有限公司", "subject_node_id": "subject-1", "clause_ids": ["a"],
                "checks": {"license": record.get("effective_status", "unknown")}, "records": [record] if record else []}

    def run_loop(self, subject_values, *, fail_once=False, rounds=5, enable=True):
        seen = []
        synthetic_calls = []
        def subject(*args, **kwargs):
            value = subject_values.pop(0) if len(subject_values)>1 else subject_values[0]
            return N.NodeOutcome("node3_subject", payload={"stats": copy.deepcopy(value)})
        def reply(config, messages, **kwargs):
            from zhilv_qiling_v2.tests_v2.protocol_fixtures import synthetic_pair
            payload = json.loads(messages[-1]["content"])
            content = {"defects": []}
            if payload.get("subject_fact_feedback"):
                items = payload["subject_fact_feedback"]
                seen.extend(items)
                rows = []
                for item in items:
                    row = {key: item[key] for key in ("feedback_item_id", "fact_id", "source_id", "subject_name", "fact_type", "clause_id")}
                    row.update(verdict="confirmed", reason="合成协议测试比对给定记录。", clause_quote=self.clause.text,
                        fact_quote=item["fact_evidence_text"], source_quote=item["source_evidence_text"])
                    if fail_once and len(seen)==1:
                        row["source_id"] = "wrong-source"
                    rows.append(row)
                content["subject_fact_verifications"] = rows
            synthetic_calls.append(1)
            return synthetic_pair(content, audit_context=kwargs['audit_context'], index=len(synthetic_calls))
        with patch.object(N.L, "request_json", side_effect=reply), \
             patch.object(N, "node_subject", side_effect=subject), \
             patch.object(N, "node_statute", return_value=N.NodeOutcome("node4_statute", payload={"risk_checks": []})), \
             patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})), \
             patch.dict(G.HANDLERS, {"reviewer": lambda ctx: {"notes": []}}):
            run = G.run_multi_agent(contract([self.clause]), configuration(enable_llm=enable,
                llm_cross_clause_pairs=0, review_max_rounds=rounds),
                retriever=SimpleNamespace(search=lambda *a, **kw: SimpleNamespace(hits=[])))
        return run, seen

    def test_subject_fact_triggers_without_any_statute_conflict(self):
        run, seen = self.run_loop([{"fact_checks": [self.checked()]}])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["fact_id"], "record-1")
        self.assertEqual(seen[0]["source"], self.source)
        self.assertEqual(seen[0]["clause_ids"], ["a"])
        self.assertTrue(run.diagnostics["cross_validation"]["feedback_verification_complete"])
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])

    def test_fact_change_reopens_review_with_new_bound_item(self):
        first = {"fact_checks": [self.checked()]}
        changed = {"fact_checks": [self.checked({**self.record, "effective_status": "valid",
            "source": {**self.source, "url": "https://source.invalid/new-evidence"}})]}
        run, seen = self.run_loop([first, changed])
        self.assertEqual([item["effective_status"] for item in seen], ["invalid", "valid"])
        self.assertNotEqual(seen[0]["feedback_item_id"], seen[1]["feedback_item_id"])
        self.assertNotEqual(seen[0]["source_id"], seen[1]["source_id"])
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])

    def test_failed_ack_is_retried_before_stable(self):
        run, seen = self.run_loop([{"fact_checks": [self.checked()]}], fail_once=True)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["feedback_item_id"], seen[1]["feedback_item_id"])
        logic = run.diagnostics["logic_llm"]
        self.assertFalse(logic["feedback_all_attempts_complete"])
        self.assertTrue(logic["feedback_runs_complete"])
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])

    def test_no_source_unknowns_are_one_summary_not_four_facts(self):
        checked = {"subject_name": "测试科技有限公司", "clause_ids": ["a"], "records": [],
            "checks": dict.fromkeys(("registration", "license", "authorization", "credit"), "unknown")}
        run, seen = self.run_loop([{"fact_checks": [checked]}], rounds=3)
        self.assertEqual(len(seen), 2)  # one summary in each of two retry rounds
        self.assertEqual({item["fact_type"] for item in seen}, {"unresolved_subject_checks"})
        self.assertTrue(all(len(item["check_states"]) == 4 for item in seen))
        self.assertTrue(all(item["fact_id"].startswith("missing-evidence:") for item in seen))
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertTrue(run.diagnostics["cross_validation"]["pending_feedback_item_ids"])

    def test_unresolved_subject_request_is_forwarded_and_blocks_stability(self):
        request = {"request_id": "request-1", "subject_name": "", "clause_ids": ["a"],
            "fact_type": "license", "status": "pending", "fact_status": "unknown", "reason": "subject_not_unique", "records": []}
        stats = {"fact_checks": [], "verification_results": [request], "verification_complete": False,
                 "pending_verification_request_ids": ["request-1"]}
        run, seen = self.run_loop([stats], rounds=3)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["request_id"], "request-1")
        self.assertTrue(seen[0]["upstream_pending"])
        self.assertFalse(run.diagnostics["cross_validation"]["feedback_verification_complete"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertEqual(run.diagnostics["cross_validation"]["pending_subject_verification_request_ids"], ["request-1"])

    def test_resolved_request_reopens_then_completes_current_review(self):
        pending = {"request_id": "request-1", "candidate_id": "risk-request", "subject_name": "测试科技有限公司",
            "clause_ids": ["a"], "fact_type": "license", "status": "pending", "fact_status": "unknown", "records": []}
        completed = {**pending, "status": "completed", "fact_status": "invalid", "records": [self.record]}
        first = {"fact_checks": [], "verification_results": [pending], "verification_complete": False,
                 "pending_verification_request_ids": ["request-1"]}
        second = {"fact_checks": [], "verification_results": [completed], "verification_complete": True,
                  "pending_verification_request_ids": []}
        run, seen = self.run_loop([first, second])
        self.assertEqual([item["upstream_pending"] for item in seen], [True, False])
        self.assertEqual(seen[1]["candidate_ids"], ["risk-request"])
        self.assertNotEqual(seen[0]["feedback_item_id"], seen[1]["feedback_item_id"])
        self.assertFalse(run.diagnostics["logic_llm"]["feedback_all_attempts_complete"])
        self.assertTrue(run.diagnostics["logic_llm"]["feedback_runs_complete"])
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])

    def test_unmapped_request_and_disabled_model_remain_pending_without_call(self):
        request = {"request_id": "request-1", "clause_ids": [], "fact_type": "license",
                   "status": "pending", "fact_status": "unknown", "records": []}
        stats = {"fact_checks": [], "verification_results": [request], "verification_complete": False,
                 "pending_verification_request_ids": ["request-1"]}
        run, seen = self.run_loop([stats], rounds=3, enable=False)
        self.assertEqual(seen, [])
        self.assertFalse(run.diagnostics["cross_validation"]["feedback_verification_complete"])
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])

    def test_unmapped_request_never_falls_back_to_whole_contract_review(self):
        request = {"request_id": "request-1", "clause_ids": ["not-a-clause"], "fact_type": "license",
                   "status": "pending", "fact_status": "unknown", "records": []}
        stats = {"fact_checks": [], "verification_results": [request], "verification_complete": False,
                 "pending_verification_request_ids": ["request-1"]}
        run, seen = self.run_loop([stats], rounds=3)
        self.assertEqual(seen, [])
        self.assertEqual(len(run.diagnostics["logic_llm"]["calls"]), 1)  # initial ordinary review only
        self.assertTrue(run.diagnostics["logic_llm"]["feedback_runs"])
        self.assertTrue(all(not item["calls"] for item in run.diagnostics["logic_llm"]["feedback_runs"]))
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])


if __name__ == "__main__":
    unittest.main()
