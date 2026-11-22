"""Coverage, item-level feedback and provenance contracts; no quality claims."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi as M
from zhilv_qiling_v2 import agents_multi_run as G
from zhilv_qiling_v2 import long_context as C
from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2 import agents as A
from zhilv_qiling_v2 import pipeline
from zhilv_qiling_v2 import risk_identity as I
from zhilv_qiling_v2 import rules as R
from zhilv_qiling_v2 import subject_facts as F
from zhilv_qiling_v2 import subject_graph as S
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, ContractElements, DocumentParse, Element, RiskCandidate, Citation, RetrievalHit, dedupe_candidates


def clause(cid, text, label=""):
    return Clause(cid, label, "", 1, "", text, text, ["b" + cid])


def contract(clauses):
    return Contract("test-contract", DocumentParse("test.txt", "txt", "\n".join(c.text for c in clauses), clauses=clauses))


def configuration(**values):
    config = Config()
    config.enable_llm = True
    config.llm_endpoint = "http://127.0.0.1:9999/v1"
    config.llm_model = "test-grounded-model"
    for key, value in values.items():
        setattr(config, key, value)
    return config


def sent_segments(payload):
    """Read either ordinary windows or shared feedback sources without changing evidence."""
    return payload["segments"] if "segments" in payload else [
        {"clause_id": cid, **segment} for cid, segments in payload.get("clause_sources", {}).items()
        for segment in segments]


def empty_reply(_config, messages, **kwargs):
    # Protocol fixture: no discovered defects still requires one bound response
    # for every requested feedback item. It is never model-quality evidence.
    payload = json.loads(messages[-1]["content"])
    content = {"defects": []}
    if payload.get("verification_feedback"):
        content["verifications"] = [{"feedback_item_id": item["feedback_item_id"],
            "candidate_id": item["candidate_id"], "article_uid": item["article_uid"],
            "verdict": "confirmed", "reason": "隔离协议测试已核对输入证据。",
            "clause_quote": next(segment["text"][:24] for segment in sent_segments(payload)
                if segment["clause_id"] in item["clause_ids"]),
            "article_quote": item["article_quote"][:32]}
            for item in payload["verification_feedback"]]
    return SimpleNamespace(content=content, model="test-model", elapsed=0.01, raw="{}")


class LongContractTest(unittest.TestCase):
    def test_truncated_output_keeps_grounded_candidate_without_claiming_coverage(self):
        c = clause("a", "乙方放弃全部尾款请求权。")
        reply = SimpleNamespace(content={"defects": [{"clause_id": "a", "code": "waiver_of_rights",
            "message": "尾款请求权需复核", "quote": c.text}]}, model="test", degraded="output_truncated")
        with patch.object(N.L, "request_json", return_value=reply):
            candidates, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0), [c])
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].extra["partial_model_output"])
        self.assertEqual(meta["calls"][0]["status"], "partial")
        self.assertEqual(meta["text_coverage"], 0)
        self.assertFalse(meta["complete"])
        self.assertTrue(any(reason.startswith("llm_output_truncated:") for reason in degraded))

    def test_valid_defects_after_twelfth_are_not_silently_lost(self):
        quotes = [f"第{i}项尾款请求权需要核对。" for i in range(1, 15)]
        c = clause("a", "".join(quotes))
        reply = SimpleNamespace(content={"defects": [{"clause_id": "a", "code": "waiver_of_rights",
            "message": quote, "quote": quote} for quote in quotes]}, model="test", degraded="")
        with patch.object(N.L, "request_json", return_value=reply):
            candidates, meta, _ = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0), [c])
        self.assertEqual(len(candidates), 14)
        self.assertEqual(meta["accepted"], 14)
        self.assertTrue(meta["complete"])

    def test_all_tail_characters_and_short_clauses_are_scheduled(self):
        clauses = [clause(f"c{i}", "本条内容" * 80) for i in range(35)] + [clause("tail", "尾款")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=500, overlap=80, cross_pairs=0)
        complete = C.coverage(clauses, windows, {w.window_id for w in windows})
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["total_chars"], complete["reviewed_chars"])
        self.assertEqual(len(complete["completed_clause_ids"]), 36)
        self.assertEqual(plan["cross_scope"], "targeted_dependency_pairs_not_exhaustive")

    def test_unreviewed_tail_is_explicit_when_call_budget_runs_out(self):
        clauses = [clause(f"c{i}", "本条付款条件由双方约定。" * 30) for i in range(30)]
        cfg = configuration(llm_chunk_chars=500, llm_chunk_overlap=40, llm_max_calls=1, llm_cross_clause_pairs=0)
        with patch.object(N.L, "request_json", side_effect=empty_reply):
            _, meta, degraded = N._llm_logic(contract(clauses), cfg, clauses)
        self.assertFalse(meta["complete"])
        self.assertLess(meta["text_coverage"], 1)
        self.assertIn("c29", meta["unreviewed_clause_ids"])
        self.assertIn("llm_call_budget_exhausted", degraded)

    def test_no_legacy_24_clause_cutoff(self):
        clauses = [clause(f"c{i}", "条款内容与权利义务由双方约定。" * 6) for i in range(35)]
        cfg = configuration(llm_max_clauses=24, llm_chunk_chars=500, llm_max_calls=0, llm_cross_clause_pairs=0)
        with patch.object(N.L, "request_json", side_effect=empty_reply):
            _, meta, _ = N._llm_logic(contract(clauses), cfg, clauses)
        self.assertTrue(meta["complete"])
        self.assertIn("c34", meta["completed_clause_ids"])

    def test_oversize_clause_tail_can_generate_grounded_finding(self):
        c = clause("long", "正常履约约定。" * 600 + "尾款全部无条件免除。")
        cfg = configuration(llm_chunk_chars=500, llm_chunk_overlap=40, llm_max_calls=0, llm_cross_clause_pairs=0)
        def reply(_cfg, messages, **kwargs):
            sent = json.loads(messages[-1]["content"])
            defects = []
            if any("尾款全部无条件免除。" in s["text"] for s in sent["segments"]):
                defects = [{"clause_id": "long", "code": "waiver_of_rights", "message": "尾款权利约定需核对", "quote": "尾款全部无条件免除。"}]
            return SimpleNamespace(content={"defects": defects}, model="test", elapsed=0.01)
        with patch.object(N.L, "request_json", side_effect=reply):
            candidates, meta, _ = N._llm_logic(contract([c]), cfg, [c])
        self.assertTrue(meta["complete"])
        self.assertEqual(len(candidates), 1)
        self.assertIn("尾款", candidates[0].spans[0].quote)

    def test_cross_clause_findings_require_both_exact_quotes(self):
        clauses = [clause("a", "合同总价为十万元。"), clause("b", "合同结算总价为二十万元。")]
        cfg = configuration(llm_cross_clause_pairs=5)
        def reply(_cfg, messages, **kwargs):
            sent = json.loads(messages[-1]["content"])
            defects = []
            if sent["kind"] == "cross_clause":
                defects = [{"clause_id": "a", "code": "fee_mismatch", "message": "两处总价需要核对", "quote": clauses[0].text,
                            "related_quotes": [{"clause_id": "b", "quote": clauses[1].text}]}]
            return SimpleNamespace(content={"defects": defects}, model="test", elapsed=0.01)
        with patch.object(N.L, "request_json", side_effect=reply):
            candidates, meta, _ = N._llm_logic(contract(clauses), cfg, clauses)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(set(candidates[0].clause_ids), {"a", "b"})
        self.assertTrue(candidates[0].extra["cross_clause_verified"])
        self.assertEqual(meta["cross_windows_completed"], 1)

    def test_invalid_schema_does_not_count_as_reviewed_text(self):
        c = clause("a", "合同价款与支付条件。")
        with patch.object(N.L, "request_json", return_value=SimpleNamespace(content={"wrong": []}, model="test")):
            _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0), [c])
        self.assertEqual(meta["text_coverage"], 0)
        self.assertIn("llm_text_coverage_incomplete", degraded)

    def test_unavailable_model_reports_zero_coverage(self):
        c = clause("a", "合同价款与支付条件。")
        cfg = configuration()
        cfg.llm_endpoint = ""
        with patch.object(N.L, "available", return_value=False):
            _, meta, degraded = N._llm_logic(contract([c]), cfg, [c])
        self.assertEqual(meta["text_coverage"], 0)
        self.assertIn("a", meta["unreviewed_clause_ids"])
        self.assertIn("llm_unavailable", degraded)


class FeedbackTest(unittest.TestCase):
    @staticmethod
    def feedback(count=10, quote_size=900):
        return [{"feedback_item_id": f"f{i}", "candidate_id": f"risk{i}",
                 "clause_ids": ["a"], "article_uid": f"law{i}", "label": "contradict",
                 "article_quote": "法" * quote_size} for i in range(count)]

    def test_all_feedback_is_sent_in_bounded_numbered_batches(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        feedback = self.feedback()
        sent = []
        def reply(cfg, messages, **kwargs):
            sent.append(json.loads(messages[-1]["content"]))
            return empty_reply(cfg, messages, **kwargs)
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0),
                                               [c], verification_feedback=feedback)
        self.assertGreater(len(sent), 1)
        self.assertEqual([item["feedback_item_id"] for batch in sent for item in batch["verification_feedback"]],
                         [item["feedback_item_id"] for item in feedback])
        self.assertEqual([call["batch_index"] for call in meta["calls"]], list(range(1, len(sent) + 1)))
        self.assertEqual(len({call["job_id"] for call in meta["calls"]}), len(sent))
        for batch, call in zip(sent, meta["calls"]):
            self.assertEqual(call["batch_count"], len(sent))
            self.assertEqual(call["feedback_item_ids"], batch["feedback_batch"]["feedback_item_ids"])
            self.assertLessEqual(len(json.dumps(batch["verification_feedback"], ensure_ascii=False)), 4000)
            self.assertEqual(call["uncompleted_feedback_item_ids"], [])
            for item in batch["verification_feedback"]:
                self.assertEqual(len(item["article_quote"]), 900)
                self.assertFalse(item["article_quote_truncated"])
        self.assertTrue(meta["complete"])
        self.assertTrue(meta["feedback_complete"])
        self.assertEqual(meta["completed_windows"], 1)
        self.assertEqual(meta["uncompleted_feedback_item_ids"], [])
        self.assertFalse(any("llm_feedback_context_partial" in reason for reason in degraded))

    def test_any_failed_or_partial_batch_keeps_its_window_incomplete(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        for failure in ("error", "partial", "invalid_schema"):
            with self.subTest(failure=failure):
                sent = []
                def reply(cfg, messages, **kwargs):
                    batch = json.loads(messages[-1]["content"])
                    sent.append(batch)
                    if len(sent) == 2:
                        if failure == "error":
                            raise N.L.LLMClientError("fixture error")
                        if failure == "partial":
                            return SimpleNamespace(content={"defects": []}, model="test", degraded="output_truncated")
                        return SimpleNamespace(content={"wrong": []}, model="test")
                    return empty_reply(cfg, messages, **kwargs)
                with patch.object(N.L, "request_json", side_effect=reply):
                    _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0),
                        [c], verification_feedback=self.feedback())
                failed_ids = [item["feedback_item_id"] for item in sent[1]["verification_feedback"]]
                self.assertGreater(len(sent), 2)
                self.assertFalse(meta["complete"])
                self.assertFalse(meta["feedback_complete"])
                self.assertEqual(meta["text_coverage"], 0)
                self.assertEqual(meta["completed_windows"], 0)
                self.assertEqual(meta["uncompleted_feedback_item_ids"], failed_ids)
                self.assertEqual(meta["calls"][1]["uncompleted_feedback_item_ids"], failed_ids)
                self.assertIn("llm_feedback_verification_incomplete", degraded)

    def test_call_budget_lists_all_unexecuted_feedback(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        feedback = self.feedback()
        cfg = configuration(llm_cross_clause_pairs=0, llm_max_calls=1)
        with patch.object(N.L, "request_json", side_effect=empty_reply) as mock:
            _, meta, degraded = N._llm_logic(contract([c]), cfg, [c], verification_feedback=feedback)
        sent_ids = meta["calls"][0]["feedback_item_ids"]
        self.assertEqual(mock.call_count, 1)
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"],
                         [item["feedback_item_id"] for item in feedback if item["feedback_item_id"] not in sent_ids])
        self.assertTrue(all(batch["status"] == "not_started" for batch in meta["feedback_windows"][0]["batches"][1:]))
        self.assertIn("llm_call_budget_exhausted", degraded)

    def test_three_failures_leave_failed_and_unexecuted_feedback_explicit(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        feedback = self.feedback(20)
        with patch.object(N.L, "request_json", side_effect=N.L.LLMClientError("fixture failure")) as mock:
            _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0),
                                               [c], verification_feedback=feedback)
        self.assertEqual(mock.call_count, 3)
        self.assertGreater(meta["planned_calls"], 3)
        self.assertEqual(meta["uncompleted_feedback_item_ids"], [item["feedback_item_id"] for item in feedback])
        self.assertTrue(all(call["uncompleted_feedback_item_ids"] for call in meta["calls"]))
        self.assertIn("llm_circuit_open_after_3_errors", degraded)

    def test_unavailable_model_still_lists_every_pending_feedback_item(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        feedback = self.feedback()
        with patch.object(N.L, "available", return_value=False), patch.object(N.L, "request_json") as mock:
            _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0),
                                               [c], verification_feedback=feedback)
        mock.assert_not_called()
        self.assertEqual(meta["uncompleted_feedback_item_ids"], [item["feedback_item_id"] for item in feedback])
        self.assertFalse(meta["feedback_complete"])
        self.assertIn("llm_feedback_verification_incomplete", degraded)

    def test_feedback_must_succeed_in_every_relevant_window(self):
        c = clause("a", "甲方按时付款，乙方依约交付。" * 90)
        cfg = configuration(llm_cross_clause_pairs=0, llm_chunk_chars=500, llm_chunk_overlap=40)
        count = 0
        def reply(config, messages, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise N.L.LLMClientError("fixture failure")
            return empty_reply(config, messages, **kwargs)
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, _ = N._llm_logic(contract([c]), cfg, [c], verification_feedback=self.feedback(1))
        self.assertGreater(meta["completed_windows"], 0)
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f0"])
        self.assertFalse(meta["complete"])

    def test_oversize_metadata_and_unmatched_items_are_not_silently_verified(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        feedback = self.feedback(2)
        feedback[0]["note"] = "异" * 5000
        feedback[1]["clause_ids"] = ["not-in-scope"]
        with patch.object(N.L, "request_json") as mock:
            _, meta, degraded = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0),
                                               [c], verification_feedback=feedback)
        mock.assert_not_called()
        self.assertFalse(meta["complete"])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f0", "f1"])
        self.assertEqual(meta["unmatched_feedback_item_ids"], ["f1"])
        self.assertIn("llm_feedback_item_oversize:f0", degraded)
        self.assertIn("llm_feedback_no_matching_window", degraded)

    def test_no_feedback_keeps_one_call_per_window(self):
        c = clause("a", "甲方按时付款，乙方依约交付。")
        with patch.object(N.L, "request_json", side_effect=empty_reply) as mock:
            _, meta, _ = N._llm_logic(contract([c]), configuration(llm_cross_clause_pairs=0), [c])
        self.assertEqual(mock.call_count, 1)
        sent = json.loads(mock.call_args.args[1][-1]["content"])
        self.assertEqual(sent["verification_feedback"], [])
        self.assertNotIn("feedback_batch", sent)
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["calls"][0]["batch_count"], 1)

    def test_feedback_runs_reach_final_diagnostics(self):
        c = clause("a", "甲方付款与结算总价不一致。")
        candidate = R._candidate("fee_mismatch", "金额核对", c, c.text)
        initial = N.NodeOutcome("node2_logic", candidates=[candidate], llm={"complete": True})
        feedback_meta = {"complete": False, "feedback_complete": False,
            "uncompleted_feedback_item_ids": ["last-item"], "calls": [{"batch_index": 1, "batch_count": 2}]}
        fresh = N.NodeOutcome("node2_logic", llm=feedback_meta)
        statute = N.NodeOutcome("node4_statute", payload={"risk_checks": [{
            "candidate_id": candidate.candidate_id, "clause_ids": ["a"],
            "article_uid": "law-A", "label": "contradict", "score": 0.8}]})
        cfg = configuration(review_max_rounds=2)
        retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[]))
        with patch.object(N, "node_logic", side_effect=[initial, fresh]), \
             patch.object(N, "node_statute", return_value=statute), \
             patch.object(N, "node_judge", return_value=N.NodeOutcome("node5_judge", payload={"score": 100})), \
             patch.dict(G.HANDLERS, {"reviewer": lambda ctx: {"notes": []}}):
            run = G.run_multi_agent(contract([c]), cfg, retriever=retriever)
        logic_meta = run.diagnostics["logic_llm"]
        self.assertEqual(logic_meta["feedback_runs"], [feedback_meta])
        self.assertFalse(logic_meta["feedback_runs_complete"])
        self.assertEqual(run.summary()["diagnostics"]["logic_llm"]["feedback_runs"][0]["uncompleted_feedback_item_ids"],
                         ["last-item"])

    def test_local_feedback_scope_keeps_all_focus_results_and_preserves_other_audit(self):
        clauses = [clause("a", "甲方应按时付款。"), clause("b", "双方应妥善保密。"),
                   clause("c", "乙方应依约交付。")]
        candidates = [R._candidate("fee_mismatch", f"核对{c.clause_id}", c, c.text) for c in clauses]
        checks = []
        for item_id, cid, label, disagreement in (
                ("a-conflict", "a", "contradict", False),
                ("a-neutral", "a", "neutral", False),
                ("a-support", "a", "support", False),
                ("b-neutral", "b", "neutral", False),
                ("c-disagreement", "c", "neutral", True),
                ("c-support", "c", "support", False)):
            candidate = candidates[[c.clause_id for c in clauses].index(cid)]
            checks.append({"feedback_item_id": item_id, "candidate_id": candidate.candidate_id,
                "clause_ids": [cid], "article_uid": item_id, "label": label, "score": 0.8,
                "disagreement": disagreement, "article_quote": "测试法律依据。"})
        original_checks = copy.deepcopy(checks)
        ctx = M.AgentContext(contract(clauses), configuration(llm_cross_clause_pairs=0), M.MessageBus())
        ctx.shared["logic"] = N.NodeOutcome("node2_logic", candidates=candidates)
        ctx.bus.publish(M.Message("statute", "logic", "statute.matched", {"risk_checks": checks}))
        with patch.object(N.L, "request_json", side_effect=empty_reply) as mock:
            G._handle_logic(ctx)
        sent = [json.loads(call.args[1][-1]["content"]) for call in mock.call_args_list]
        self.assertTrue(sent)
        sent_items = [item for batch in sent for item in batch["verification_feedback"]]
        expected_articles = {"a-conflict", "a-neutral", "a-support", "c-disagreement", "c-support"}
        self.assertEqual(len(sent_items), len(expected_articles))
        self.assertEqual({item["article_uid"] for item in sent_items}, expected_articles)
        original_by_article = {item["article_uid"]: item for item in original_checks}
        for item in sent_items:
            original = original_by_article[item["article_uid"]]
            self.assertEqual(item["feedback_item_id"], G._statute_feedback_item(original)["feedback_item_id"])
            self.assertNotEqual(item["feedback_item_id"], original["feedback_item_id"])
            for field in ("candidate_id", "article_uid", "clause_ids", "label", "disagreement", "article_quote"):
                self.assertEqual(item[field], original[field])
        self.assertEqual({segment["clause_id"] for batch in sent for segment in sent_segments(batch)}, {"a", "c"})
        outcome = ctx.shared["logic"]
        self.assertTrue({candidate.candidate_id for candidate in candidates}.issubset(
            {candidate.candidate_id for candidate in outcome.candidates}))
        feedback_run = outcome.payload["feedback_runs"][0]
        self.assertTrue(feedback_run["feedback_complete"])
        self.assertEqual(feedback_run["unmatched_feedback_item_ids"], [])
        self.assertEqual(feedback_run["uncompleted_feedback_item_ids"], [])
        b_result = next(candidate for candidate in outcome.candidates if candidate.candidate_id == candidates[1].candidate_id)
        self.assertEqual([check["feedback_item_id"] for check in b_result.extra["statute_checks"]], ["b-neutral"])
        self.assertEqual(checks, original_checks)
        self.assertEqual([item["feedback_item_id"] for item in checks],
                         ["a-conflict", "a-neutral", "a-support", "b-neutral", "c-disagreement", "c-support"])
        self.assertEqual(ctx.bus.inbox("logic", kinds=["statute.matched"])[0].payload["risk_checks"], checks)

    def test_neural_rule_disagreement_keeps_both_verdicts_and_enters_manual_review(self):
        c = clause("a", "乙方无需承担任何违约责任。")
        ct = contract([c])
        candidate = R._candidate("liability_exemption", "免责约定需复核", c, c.text, confidence=0.95)
        hit = RetrievalHit("law-A", "测试法", "第一条", "违约应当承担相应责任。", 0.9)
        verdict = N.E.EntailmentVerdict("law-A", "第一条", "neutral", 0.999,
            source="bert_entail", neural_label="neutral", neural_score=0.999,
            rule_label="contradict", rule_score=0.991, disagreement=True)
        cfg = Config()
        retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[hit]))
        with patch.object(N.E, "annotate", return_value=[verdict]):
            statute = N.node_statute(ct, [candidate], cfg, retriever=retriever)
        check = statute.payload["risk_checks"][0]
        self.assertEqual(check["label"], "neutral")
        self.assertEqual(check["neural_label"], "neutral")
        self.assertEqual(check["rule_label"], "contradict")
        self.assertTrue(check["manual_required"])
        self.assertTrue(any(e.kind == "neural_rule_disagreement" for e in candidate.evidence))
        ctx = M.AgentContext(ct, cfg, M.MessageBus())
        ctx.shared.update(logic=N.NodeOutcome("node2_logic", candidates=[candidate]),
            subject=N.NodeOutcome("node3_subject"), statute=statute)
        G._handle_judge(ctx)
        risk = ctx.shared["fused"][0]
        self.assertTrue(risk.review_required)
        self.assertIn("神经模型与规则判别分歧", risk.rationale)

    def test_three_node_rounds_stop_when_state_stabilizes(self):
        document = contract([clause("a", "甲方应按时支付合同价款，乙方应依约交付货物。")])
        cfg = Config()
        cfg.review_max_rounds = 3
        retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[]))
        run = G.run_multi_agent(document, cfg, retriever=retriever)
        for name in ("logic", "subject", "statute"):
            self.assertEqual(run.order.count(name), 2)
        self.assertTrue(run.diagnostics["cross_validation"]["converged"])
        self.assertEqual(run.diagnostics["cross_validation"]["termination"], "stable")
        self.assertEqual(run.bus.stats()["rejected"], 0)

    def test_round_limit_is_explicit(self):
        document = contract([clause("a", "甲方应按时支付合同价款，乙方应依约交付货物。")])
        cfg = Config()
        cfg.review_max_rounds = 1
        retriever = SimpleNamespace(search=lambda *a, **k: SimpleNamespace(hits=[]))
        run = G.run_multi_agent(document, cfg, retriever=retriever)
        self.assertFalse(run.diagnostics["cross_validation"]["converged"])
        self.assertIn("cross_validation_round_limit", run.degraded)

    def test_hybrid_judge_keeps_enabled_model(self):
        cfg = configuration()
        self.assertTrue(G._cfg_for("judge", cfg).enable_llm)
        self.assertFalse(G._cfg_for("subject", cfg).enable_llm)

    def test_all_three_nodes_can_exchange_messages(self):
        bus = M.MessageBus()
        for left in ("logic", "subject", "statute"):
            for right in ("logic", "subject", "statute"):
                if left != right:
                    self.assertTrue(bus.allowed(left, right), (left, right))

    def test_feedback_is_not_broadcast_to_unrelated_risk(self):
        clauses = [clause("a", "甲方支付金额与总价不一致。"), clause("b", "乙方承担的保密义务不明确。")]
        document = contract(clauses)
        ca = R._candidate("fee_mismatch", "金额核对", clauses[0], clauses[0].text)
        cb = R._candidate("confidentiality_missing", "保密核对", clauses[1], clauses[1].text)
        outcome = N.NodeOutcome("node2_logic", candidates=[ca, cb])
        ctx = M.AgentContext(document, Config(), M.MessageBus())
        ctx.bus.publish(M.Message("statute", "logic", "statute.matched", {"risk_checks": [{
            "candidate_id": ca.candidate_id, "clause_ids": ["a"], "article_uid": "law-A", "label": "contradict", "score": 0.8}]}))
        with patch.object(N, "node_logic", return_value=outcome):
            G._handle_logic(ctx)
        self.assertTrue(ca.extra.get("statute_checks"))
        self.assertFalse(cb.extra.get("statute_checks"))
        self.assertFalse(any(e.kind == "cross_validation" for e in cb.evidence))


class FactIdentityTest(unittest.TestCase):
    def test_real_pipeline_merges_same_date_fact_and_locates_both_dates(self):
        text = "采购合同\n甲方：星海科技有限公司\n乙方：云图技术有限公司\n第一条 期限\n合同起始日期为2026年12月1日，终止日期为2026年11月1日。"
        cfg = Config()
        document = pipeline.prepare("date.txt", text.encode(), config=cfg)
        ct = pipeline.build_contract(document, cfg)
        findings = [c for c in N.node_logic(ct, cfg).candidates if c.code == "date_reversed"]
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].clause_ids)
        self.assertGreaterEqual(len(findings[0].extra["alias_candidate_ids"]), 2)
        self.assertEqual({s.quote for s in findings[0].spans}, {"2026年12月1日", "2026年11月1日"})
        blocks = document.block_map()
        for span in findings[0].spans:
            self.assertIn(span.clause_id, findings[0].clause_ids)
            self.assertEqual(blocks[span.block_id].text[span.char_start:span.char_end], span.quote)

    def test_two_different_date_pairs_in_same_clause_are_not_merged(self):
        c = clause("a", "阶段一自2026年12月1日至2026年11月1日；阶段二自2027年8月1日至2027年7月1日。")
        ct = contract([c])
        findings = [RiskCandidate("one", "date_reversed", "2026-12-01晚于2026-11-01", clause_ids=["a"]),
                    RiskCandidate("two", "date_reversed", "2027-08-01晚于2027-07-01", clause_ids=["a"])]
        result = dedupe_candidates(I.normalize_fact_candidates(findings, ct))
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0].extra["fact_key"], result[1].extra["fact_key"])

    def test_same_dates_in_distinct_milestones_are_not_globally_merged(self):
        clauses = [clause("a", "阶段一自2026年12月1日至2026年11月1日。"),
                   clause("b", "阶段二自2026年12月1日至2026年11月1日。")]
        ct = contract(clauses)
        findings = [RiskCandidate("one", "date_reversed", "2026-12-01晚于2026-11-01", clause_ids=["a"]),
                    RiskCandidate("two", "date_reversed", "2026-12-01晚于2026-11-01", clause_ids=["b"]),
                    RiskCandidate("ambiguous", "date_reversed", "2026-12-01晚于2026-11-01")]
        result = I.normalize_fact_candidates(findings, ct)
        self.assertEqual(len(result), 3)
        self.assertNotIn("fact_key", result[-1].extra)

    def test_fusion_keeps_citations_from_all_merged_channel_aliases(self):
        c = clause("a", "自2026年12月1日至2026年11月1日。")
        ct = contract([c])
        one = RiskCandidate("one", "date_reversed", "2026-12-01晚于2026-11-01", confidence=0.85)
        two = RiskCandidate("two", "date_reversed", "结束2026-11-01早于开始2026-12-01", confidence=0.82,
                            extra={"issue_code": "date_reversed"})
        citations = A.AgentResult("test", citations={
            "one": [Citation("测试法", "第一条", "测试依据甲", article_uid="A")],
            "two": [Citation("测试法", "第二条", "测试依据乙", article_uid="B")]})
        result = A.fuse([one, two], [citations], ct, Config())
        self.assertEqual(len(result), 1)
        self.assertEqual({c.article_uid for c in result[0].citations}, {"A", "B"})
        _, breakdown = A.score_risks(result, Config())
        self.assertEqual(set(breakdown["by_code"]), {"date_reversed"})


class SubjectFactsTest(unittest.TestCase):
    def record(self, **changes):
        result = {"subject_name": "测试科技有限公司", "subject_id": "91310101TEST123456",
            "fact_type": "license", "status": "valid", "scope": "测试许可",
            "valid_until": (date.today() + timedelta(days=90)).isoformat(),
            "source": {"url": "https://example.gov.cn/record/1", "publisher": "测试核验来源",
                       "reviewer": "测试核验员", "verification_method": "人工比对证书和来源记录",
                       "verified_at": date.today().isoformat()}}
        result.update(changes)
        return result

    def test_provenance_fields_are_required(self):
        item = self.record()
        del item["source"]["reviewer"]
        with self.assertRaises(ValueError):
            F.validate_fact(item)

    def test_source_url_is_never_fetched(self):
        # localhost URL may be recorded as a source, but no network implementation
        # exists in this registry and import cannot cause an SSRF request.
        item = self.record()
        item["source"]["url"] = "http://127.0.0.1:1/private"
        fact = F.validate_fact(item)
        self.assertIn("not_independently_verified", fact["authenticity"])

    def test_expired_or_identifier_mismatch_stays_unknown(self):
        fact = F.validate_fact(self.record(valid_until=(date.today() - timedelta(days=1)).isoformat()))
        result = F.resolve_subject(fact["subject_name"], [fact], full_text=fact["subject_id"])
        self.assertEqual(result["checks"]["license"], "unknown")
        self.assertEqual(result["records"][0]["review_reason"], "expired")
        valid = F.validate_fact(self.record())
        mismatch = F.resolve_subject(valid["subject_name"], [valid], full_text="另一个主体代码")
        self.assertEqual(mismatch["checks"]["license"], "unknown")

    def test_authorization_must_match_this_contract(self):
        with self.assertRaises(ValueError):
            F.validate_fact(self.record(fact_type="authorization", contract_id=""))
        fact = F.validate_fact(self.record(fact_type="authorization", contract_id="contract-A"))
        checked = F.resolve_subject(fact["subject_name"], [fact], full_text=fact["subject_id"], contract_id="contract-B")
        self.assertEqual(checked["checks"]["authorization"], "unknown")

    def test_atomic_import_deduplicates_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "facts.json"
            first = F.import_facts([self.record()], path)
            second = F.import_facts([self.record()], path)
            self.assertEqual(first["total"], second["total"])
            records, errors = F.load_facts(path)
            self.assertEqual(errors, [])
            self.assertEqual(len(records), 1)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["records"][0]["status"] = "invalid"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(F.load_facts(path)[0], [])

    def test_contract_claim_does_not_become_a_verified_fact(self):
        c = clause("party", "供应商：测试科技有限公司，声称已具备全部资质许可证和授权。")
        document = contract([c])
        document.elements.parties.append(Element("p1", "party", "测试科技有限公司", "测试科技有限公司", source="test"))
        cfg = Config()
        cfg.subject_facts_path = ""
        result = S.check_subjects(document, cfg)
        self.assertTrue(result.stats["has_license_token"])
        self.assertEqual(result.stats["fact_checks"][0]["checks"]["license"], "unknown")
        self.assertNotIn("holds", {edge.kind for edge in result.edges})
        self.assertFalse(result.stats["contract_claims_independently_verified"])


if __name__ == "__main__":
    unittest.main()
