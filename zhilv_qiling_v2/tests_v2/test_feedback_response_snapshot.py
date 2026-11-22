"""Rejected response diagnostics are bounded copies, never replacement evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import model_audit as A, nodes as N
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration


class FeedbackResponseSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.clause = clause("a", "合同原文：买方应在验收后十日内付款。")
        self.contract = contract([self.clause])
        self.config = configuration(llm_cross_clause_pairs=0)
        self.item = {"feedback_item_id": "feedback-1", "candidate_id": "candidate-1", "article_uid": "law-1",
            "clause_ids": ["a"], "article_quote": "合成法条：依照约定支付价款。"}
        self.row = {"feedback_item_id": "feedback-1", "candidate_id": "candidate-1", "article_uid": "law-1",
            "verdict": "confirmed", "reason": "仅作协议诊断测试。", "clause_quote": "被拒绝的模型概述引文-PRIVATE-RESPONSE",
            "article_quote": self.item["article_quote"]}

    def test_report_keeps_rejected_quote_but_validator_still_uses_original_response(self):
        content = {"verifications": [self.row]}
        before = copy.deepcopy(content)
        with patch.object(N.L, "request_json", return_value=SimpleNamespace(content=content, model="fixture")), \
             patch.object(N, "_validate_feedback_verifications", wraps=N._validate_feedback_verifications) as validate:
            _, meta, _ = N._llm_logic(self.contract, self.config, [self.clause], verification_feedback=[self.item])
        call = meta["calls"][0]
        self.assertIs(validate.call_args.args[0], content)
        self.assertEqual(content, before)
        self.assertEqual(json.loads(call["feedback_response"]["json"])["verifications"], [self.row])
        self.assertFalse(call["feedback_response"]["truncated"])
        self.assertEqual(call["verifications"], [])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["feedback-1"])
        self.assertEqual(call["verification_errors"][0]["reason"], "clause_quote_not_in_feedback_window")

    def test_large_both_root_snapshot_has_total_row_and_json_bounds_without_mutation(self):
        huge = {**self.row, "reason": "超长理由" * 1000, "clause_quote": "超长引文\"\\\n" * 1000,
                "article_quote": "超长法条" * 1000}
        subject = {"feedback_item_id": "subject-1", "fact_id": "fact-1", "source_id": "source-1",
            "subject_name": "主体名称" * 300, "fact_type": "license", "clause_id": "a", "verdict": "needs_manual",
            "reason": "待人工核验" * 1000, "clause_quote": "主体合同引文" * 1000,
            "fact_quote": "事实引文" * 1000, "source_quote": "来源引文" * 1000}
        content = {"verifications": [copy.deepcopy(huge) for _ in range(3)],
                   "subject_fact_verifications": [copy.deepcopy(subject) for _ in range(8)]}
        before = copy.deepcopy(content)
        snapshot = N._feedback_response_snapshot(content, 3)
        self.assertEqual(content, before)
        self.assertEqual(sum(snapshot["kept_counts"].values()), 4)
        self.assertLessEqual(len(snapshot["json"]), 16000)
        self.assertTrue(snapshot["truncated"])
        self.assertEqual(snapshot["original_counts"], {"verifications": 3, "subject_fact_verifications": 8})
        expected = json.dumps({"verifications": content["verifications"][:3],
                               "subject_fact_verifications": content["subject_fact_verifications"][:1]}, ensure_ascii=False)
        self.assertEqual(snapshot["json"], expected[:16000])
        self.assertEqual(snapshot["selected_json_chars"], len(expected))
        self.assertEqual(snapshot["original_json_chars"], len(json.dumps(content, ensure_ascii=False)))

    def test_snapshot_is_feedback_report_only_and_never_persisted_to_model_audit(self):
        content = {"verifications": [self.row]}
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"ZLQL_MODEL_AUDIT_PATH": str(Path(folder) / "audit.sqlite")}), \
             patch.object(N.L, "_complete", return_value=(json.dumps(content), {"model": "fixture"}, 0.01)), \
             A.review_audit("snapshot-audit"):
            _, meta, _ = N._llm_logic(self.contract, self.config, [self.clause], verification_feedback=[self.item])
            events = A.recent_events(trace_id="snapshot-audit")
        self.assertTrue(events)
        self.assertIn("feedback_response", meta["calls"][0])
        self.assertNotIn("PRIVATE-RESPONSE", json.dumps(events))
        self.assertNotIn("feedback_response", json.dumps(events))
        with patch.object(N.L, "request_json", return_value=SimpleNamespace(content={"defects": []}, model="fixture")):
            _, plain, _ = N._llm_logic(self.contract, self.config, [self.clause])
        self.assertNotIn("feedback_response", plain["calls"][0])


if __name__ == "__main__":
    unittest.main()
