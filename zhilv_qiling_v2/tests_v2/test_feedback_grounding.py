"""Actual outgoing feedback sources stay bound to each item and current window."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import long_context as C, nodes as N
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration


class FeedbackGroundingTests(unittest.TestCase):
    def setUp(self):
        self.a = clause("a", "第一条 交付。卖方应在约定地点交付设备。", label="第一条")
        self.b = clause("b", "第二条 付款。买方应在验收后十日内支付价款。", label="第二条")
        self.contract = contract([self.a, self.b])
        self.config = configuration(llm_cross_clause_pairs=0)
        self.item = {"feedback_item_id": "feedback-b", "candidate_id": "candidate-b", "article_uid": "law-b",
            "clause_ids": ["b"], "article_quote": "合成规范：按约定期限付款。",
            "note": "这只是模型概述附注，不能作为合同引文。", "title": "这只是元数据标题。"}

    def run_review(self, *, feedback=None, clauses=None, quote=None):
        captured = []
        scope = clauses or [self.a, self.b]
        def reply(config, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            captured.append((payload, messages[0]["content"], kwargs))
            content = {"defects": []}
            if payload.get("verification_feedback"):
                rows = []
                for item in payload["verification_feedback"]:
                    cid = item["bound_clause_ids"][0]
                    text = payload["clause_sources"][cid][0]["text"]
                    selected = quote(item, payload) if callable(quote) else quote or text[:80]
                    rows.append({"feedback_item_id": item["feedback_item_id"], "candidate_id": item["candidate_id"],
                        "article_uid": item["article_uid"], "verdict": "confirmed", "reason": "合成协议核对。",
                        "clause_quote": selected, "article_quote": item["article_quote"][:80]})
                content = {"verifications": rows}
            return SimpleNamespace(content=content, model="fixture")
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, degraded = N._llm_logic(self.contract, self.config, scope,
                verification_feedback=[self.item] if feedback is None else feedback)
        return captured, meta, degraded

    def test_shared_sources_keep_all_window_text_once_and_bind_each_item(self):
        feedback = [{**self.item, "feedback_item_id": f"feedback-{i}", "article_uid": f"law-{i}"} for i in range(3)]
        captured, meta, _ = self.run_review(feedback=feedback)
        payload, system, _ = captured[0]
        self.assertNotIn("segments", payload)
        self.assertEqual(set(payload["clause_sources"]), {"a", "b"})
        self.assertEqual(payload["clause_sources"]["a"], [{"start": 0, "end": len(self.a.text), "text": self.a.text}])
        self.assertEqual(payload["clause_sources"]["b"], [{"start": 0, "end": len(self.b.text), "text": self.b.text}])
        self.assertTrue(all(item["bound_clause_ids"] == ["b"] for item in payload["verification_feedback"]))
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(raw.count(json.dumps(self.a.text, ensure_ascii=False)), 1)
        self.assertEqual(raw.count(json.dumps(self.b.text, ensure_ascii=False)), 1)
        self.assertIn("同一项bound_clause_ids", system)
        self.assertIn("逐字复制连续片段", system)
        self.assertIn("note/title/heading/summary/reason", system)
        self.assertTrue(meta["feedback_complete"])

    def test_wrong_clause_stays_pending_even_when_its_text_is_shared_context(self):
        captured, meta, _ = self.run_review(quote=self.a.text)
        self.assertIn("a", captured[0][0]["clause_sources"])
        self.assertEqual(captured[0][0]["verification_feedback"][0]["bound_clause_ids"], ["b"])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["feedback-b"])
        self.assertIn("clause_quote_not_in_feedback_window", [row["reason"] for row in meta["calls"][0]["verification_errors"]])

    def test_note_and_title_cannot_be_used_as_locatable_contract_quotes(self):
        for field in ("note", "title"):
            with self.subTest(field=field):
                _, meta, _ = self.run_review(quote=self.item[field])
                self.assertFalse(meta["feedback_complete"])
                self.assertEqual(meta["uncompleted_feedback_item_ids"], ["feedback-b"])

    def test_binding_intersects_item_ids_with_actual_window_without_adding_clauses(self):
        item = {**self.item, "clause_ids": ["b", "not-in-contract"]}
        captured, meta, _ = self.run_review(feedback=[item])
        self.assertEqual(captured[0][0]["verification_feedback"][0]["bound_clause_ids"], ["b"])
        self.assertNotIn("not-in-contract", captured[0][0]["clause_sources"])
        self.assertTrue(meta["feedback_complete"])

    def test_chunk_sources_never_include_unseen_tail_and_cannot_verify_it(self):
        body = "前段付款约定。" * 60 + "尾部专属条文：验收后五日内付清。"
        long_clause = clause("b", body)
        self.config.llm_chunk_chars = 150
        self.config.llm_chunk_overlap = 0
        captured, meta, _ = self.run_review(clauses=[long_clause], quote="尾部专属条文：验收后五日内付清。")
        self.assertGreater(len(captured), 1)
        first = captured[0][0]["clause_sources"]["b"]
        self.assertNotIn("尾部专属条文", "".join(part["text"] for part in first))
        self.assertFalse(meta["feedback_complete"])
        self.assertTrue(meta["uncompleted_feedback_item_ids"])
        self.assertIn("clause_quote_not_in_feedback_window", [row["reason"] for row in meta["calls"][0]["verification_errors"]])

    def test_text_and_cross_window_context_and_coverage_are_preserved(self):
        self.a = clause("a", "第一条 付款期限为验收后五日。", label="第一条")
        self.b = clause("b", "第二条 付款期限以第一条为准，验收后十日付款。", label="第二条")
        clauses = [self.a, self.b]
        self.contract = contract(clauses)
        self.config.llm_cross_clause_pairs = 4
        windows, plan = C.build_review_plan(clauses, chunk_chars=self.config.llm_chunk_chars,
            overlap=self.config.llm_chunk_overlap, cross_pairs=self.config.llm_cross_clause_pairs)
        expected = {window.window_id: window for window in windows}
        captured, meta, _ = self.run_review(clauses=clauses)
        self.assertTrue(any(payload["kind"] == "cross_clause" for payload, _, _ in captured))
        for payload, _, _ in captured:
            window = expected[payload["window_id"]]
            actual = [(cid, part["start"], part["end"], part["text"]) for cid, parts in payload["clause_sources"].items() for part in parts]
            self.assertEqual(set(actual), {(s.clause_id, s.start, s.end, s.text) for s in window.segments})
        self.assertEqual(meta["text_coverage"], 1)
        self.assertEqual(meta["cross_windows_completed"], plan["cross_windows"])
        self.assertTrue(meta["complete"])

    def test_feedback_budget_includes_binding_metadata_and_does_not_drop_items(self):
        feedback = [{**self.item, "feedback_item_id": f"feedback-{i}", "article_uid": f"law-{i}",
                     "note": "合成附注" * 220} for i in range(7)]
        captured, meta, _ = self.run_review(feedback=feedback)
        self.assertGreater(len(captured), 1)
        sent_ids = []
        for payload, _, _ in captured:
            batch = payload["verification_feedback"]
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False)), 4000)
            self.assertLessEqual(len(batch), 3)
            self.assertEqual(set(payload["clause_sources"]), {"a", "b"})
            sent_ids.extend(item["feedback_item_id"] for item in batch)
        self.assertEqual(sent_ids, [item["feedback_item_id"] for item in feedback])
        self.assertTrue(meta["feedback_complete"])

    def test_ordinary_review_still_uses_original_segment_input(self):
        captured, meta, _ = self.run_review(feedback=[])
        payload, system, _ = captured[0]
        self.assertIn("segments", payload)
        self.assertNotIn("clause_sources", payload)
        self.assertTrue(system.startswith(N.LOGIC_SYSTEM_PROMPT))
        self.assertTrue(meta["complete"])

    def test_legacy_schema_without_sources_does_not_force_quote_or_verdict(self):
        schema = N._feedback_json_schema([self.item])
        props = schema["properties"]["verifications"]["items"]["properties"]
        for field in ("clause_quote", "article_quote"):
            self.assertNotIn("enum", props[field])
            self.assertNotIn("const", props[field])
        self.assertEqual(set(props["verdict"]["enum"]), {"confirmed", "rejected", "needs_manual"})


if __name__ == "__main__":
    unittest.main()
