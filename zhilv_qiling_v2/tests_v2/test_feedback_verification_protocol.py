"""Fail-closed item acknowledgements; synthetic fixtures are not legal gold."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse


class FeedbackVerificationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.clause = Clause("a", "", "", 1, "", "买方应在验收后十日内付款。", "买方应在验收后十日内付款。", ["ba"])
        self.contract = Contract("synthetic-contract", DocumentParse("fixture.txt", "txt", self.clause.text, clauses=[self.clause]))
        self.config = Config(enable_llm=True, llm_endpoint="http://unused.invalid/v1", llm_model="test-only",
                             llm_cross_clause_pairs=0)
        self.feedback = [{"feedback_item_id": "f1", "candidate_id": "risk-1", "article_uid": "law-1",
            "clause_ids": ["a"], "label": "contradict", "article_quote": "合成规范：买方须在验收后五日内付款。"}]

    def ack(self, item=None, **changes):
        item = item or self.feedback[0]
        row = {"feedback_item_id": item["feedback_item_id"], "candidate_id": item["candidate_id"],
            "article_uid": item["article_uid"], "verdict": "confirmed", "reason": "期限与所附合成规范不一致。",
            "clause_quote": self.clause.text, "article_quote": item["article_quote"]}
        row.update(changes)
        return row

    def run_reply(self, content, *, feedback=None, degraded=""):
        result = SimpleNamespace(content=content, model="fixture-model", degraded=degraded)
        with patch.object(N.L, "request_json", return_value=result) as request:
            values = N._llm_logic(self.contract, self.config, [self.clause],
                verification_feedback=self.feedback if feedback is None else feedback)
        return (*values, request)

    def assert_pending(self, content, expected_reason, *, feedback=None):
        _, meta, degraded, _ = self.run_reply(content, feedback=feedback)
        self.assertFalse(meta["feedback_complete"])
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["text_coverage"], 0)
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f1"])
        self.assertIn(expected_reason, {row["reason"] for row in meta["calls"][0].get("verification_errors", [])})
        self.assertIn("llm_feedback_verification_incomplete", degraded)
        return meta

    def test_empty_defects_does_not_acknowledge_feedback(self):
        self.assert_pending({"defects": []}, "verifications_array_required")

    def test_missing_or_nonarray_verifications_remain_pending(self):
        for values in (None, {}, "verified"):
            with self.subTest(values=values):
                self.assert_pending({"defects": [], "verifications": values}, "verifications_array_required")

    def test_valid_bound_ack_completes_and_exposes_evidence(self):
        _, meta, degraded, request = self.run_reply({"defects": [], "verifications": [self.ack()]})
        self.assertTrue(meta["feedback_complete"])
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["verified_feedback_item_ids"], ["f1"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], [])
        returned = meta["calls"][0]["verifications"][0]
        self.assertEqual({key: returned[key] for key in self.ack()}, self.ack())
        self.assertEqual(returned["verified_scope"], "bound_provided_text")
        self.assertTrue(returned["article_source_complete"])
        self.assertEqual(meta["calls"][0]["verification_errors"], [])
        self.assertEqual(meta["feedback_protocol"], "bound-item-verifications-v1")
        self.assertNotIn("llm_feedback_verification_incomplete", degraded)
        self.assertIn("verifications", request.call_args.args[1][0]["content"])
        self.assertIn("verifications:array", request.call_args.kwargs["schema_hint"])

    def test_feedback_only_protocol_does_not_require_defects(self):
        _, meta, _, _ = self.run_reply({"verifications": [self.ack()]})
        self.assertTrue(meta["feedback_complete"])

    def test_rejected_is_completed_review_without_new_risk(self):
        candidates, meta, _, _ = self.run_reply({"verifications": [self.ack(verdict="rejected")]})
        self.assertEqual(candidates, [])
        self.assertTrue(meta["feedback_complete"])

    def test_needs_manual_is_not_completed(self):
        self.assert_pending({"verifications": [self.ack(verdict="needs_manual")]}, "verification_needs_manual")

    def test_missing_ack_does_not_complete_other_items(self):
        second = {**self.feedback[0], "feedback_item_id": "f2", "candidate_id": "risk-2", "article_uid": "law-2"}
        _, meta, _, _ = self.run_reply({"verifications": [self.ack(second)]}, feedback=self.feedback + [second])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f1"])
        self.assertEqual(meta["verified_feedback_item_ids"], ["f2"])
        self.assertFalse(meta["complete"])

    def test_duplicate_ack_invalidates_that_id_but_keeps_other_valid_ack(self):
        second = {**self.feedback[0], "feedback_item_id": "f2", "candidate_id": "risk-2", "article_uid": "law-2"}
        _, meta, _, _ = self.run_reply({"verifications": [self.ack(), self.ack(), self.ack(second)]}, feedback=self.feedback + [second])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f1"])
        self.assertEqual(meta["verified_feedback_item_ids"], ["f2"])
        self.assertIn("duplicate_verification", {row["reason"] for row in meta["calls"][0]["verification_errors"]})

    def test_wrong_id_never_resolves_expected_item(self):
        self.assert_pending({"verifications": [self.ack(feedback_item_id="invented-id")]}, "missing_verification")

    def test_extra_unknown_id_prevents_successful_batch_coverage(self):
        self.assert_pending({"verifications": [self.ack(), self.ack(feedback_item_id="invented-id")]}, "unexpected_verification_rows_in_batch")

    def test_wrong_candidate_or_article_binding_is_rejected(self):
        for field, reason in (("candidate_id", "candidate_id_mismatch"), ("article_uid", "article_uid_mismatch")):
            with self.subTest(field=field):
                self.assert_pending({"verifications": [self.ack(**{field: "another-item"})]}, reason)

    def test_article_quote_from_another_source_is_rejected(self):
        self.assert_pending({"verifications": [self.ack(article_quote="合成规范：卖方应当永久免费维修。") ]}, "article_quote_not_in_bound_source")

    def test_clause_quote_must_belong_to_feedback_clause_and_current_window(self):
        _, errors = N._validate_feedback_verifications({"verifications": [self.ack(clause_quote="另一条款") ]},
            self.feedback, {"a": ["当前窗口"], "b": ["另一条款"]})
        self.assertEqual(errors[0]["reason"], "clause_quote_not_in_feedback_window")
        _, errors = N._validate_feedback_verifications({"verifications": [self.ack()]}, self.feedback, {"a": ["只有前缀"]})
        self.assertEqual(errors[0]["reason"], "clause_quote_not_in_feedback_window")

    def test_rewritten_or_ellipsis_quotes_are_rejected(self):
        for quote in ("买方…付款。", "买方应当立即付款。", "", self.clause.text.replace("。", "！")):
            with self.subTest(quote=quote):
                self.assert_pending({"verifications": [self.ack(clause_quote=quote)]}, "clause_quote_not_in_feedback_window")

    def test_whitespace_only_differences_are_locatable(self):
        _, meta, _, _ = self.run_reply({"verifications": [self.ack(clause_quote="买方 应在验收后十日内付款。", article_quote="合成规范： 买方须在验收后五日内付款。")]})
        self.assertTrue(meta["feedback_complete"])

    def test_reason_and_verdict_are_required_and_whitelisted(self):
        for changes, reason in (({"reason": ""}, "verification_reason_required"),
                ({"reason": 12}, "verification_reason_required"), ({"verdict": "ok"}, "invalid_verdict"),
                ({"verdict": []}, "invalid_verdict"), ({"candidate_id": 1}, "candidate_id_mismatch")):
            with self.subTest(changes=changes):
                self.assert_pending({"verifications": [self.ack(**changes)]}, reason)

    def test_duplicate_input_id_is_not_silently_renamed_or_verified(self):
        meta = self.assert_pending({"verifications": [self.ack()]}, "duplicate_feedback_input_id",
            feedback=self.feedback + [copy.deepcopy(self.feedback[0])])
        self.assertEqual([item["feedback_item_id"] for item in meta["feedback_items"]], ["f1", "f1"])

    def test_truncated_output_cannot_complete_even_valid_ack(self):
        _, meta, _, _ = self.run_reply({"verifications": [self.ack()]}, degraded="output_truncated")
        self.assertFalse(meta["feedback_complete"])
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f1"])
        self.assertEqual(meta["calls"][0]["status"], "partial")

    def test_truncated_article_source_never_completes_legal_relation(self):
        for input_truncated in (True, False):
            with self.subTest(input_truncated=input_truncated):
                source = "法条正文" * (5 if input_truncated else 300)
                feedback = [{**self.feedback[0], "article_quote": source,
                             "article_quote_truncated": input_truncated}]
                row = self.ack(feedback[0], article_quote="法条正文")
                meta = self.assert_pending({"verifications": [row]}, "article_source_truncated", feedback=feedback)
                verification = meta["calls"][0]["verifications"][0]
                self.assertEqual(verification["verified_scope"], "sent_excerpt")
                self.assertFalse(verification["article_source_complete"])

    def test_at_most_three_feedback_items_per_call_without_dropping_ids(self):
        feedback = [{**self.feedback[0], "feedback_item_id": f"f{index}", "candidate_id": f"risk-{index}"}
                    for index in range(10)]
        sent = []
        def reply(config, messages, **kwargs):
            batch = json.loads(messages[-1]["content"])["verification_feedback"]
            sent.append(batch)
            return SimpleNamespace(model="fixture-model", content={"verifications": [self.ack(item) for item in batch]})
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, _ = N._llm_logic(self.contract, self.config, [self.clause], verification_feedback=feedback)
        self.assertEqual([len(batch) for batch in sent], [3, 3, 3, 1])
        self.assertEqual([item["feedback_item_id"] for batch in sent for item in batch], [item["feedback_item_id"] for item in feedback])
        self.assertEqual(meta["feedback_max_items_per_batch"], 3)
        self.assertTrue(meta["feedback_complete"])

    def test_malformed_extra_rows_do_not_count_as_complete(self):
        self.assert_pending({"verifications": [self.ack(), "not-an-object"]}, "unexpected_verification_rows_in_batch")

    def test_ordinary_review_still_accepts_empty_defects(self):
        _, meta, _, request = self.run_reply({"defects": []}, feedback=[])
        self.assertTrue(meta["complete"])
        self.assertTrue(meta["feedback_complete"])
        self.assertEqual(meta["feedback_protocol"], "not_applicable")
        self.assertNotIn("每个feedback_item_id恰好返回一次", request.call_args.args[1][0]["content"])
        self.assertEqual(request.call_args.kwargs["schema_hint"], "defects:array")


if __name__ == "__main__":
    unittest.main()
