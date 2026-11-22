"""Adversarial display checks: explanations cannot upgrade pending evidence."""
import copy
import unittest

from zhilv_qiling_v2.review_explanations import build_review_explanations


def report_fixture():
    feedback = {"feedback_item_id": "f1", "candidate_id": "c1", "article_uid": "a1",
        "clause_ids": ["p1"], "law_name": "测试法", "article_no": "第一条",
        "article_quote": "应当按约定履行义务。", "manual_required": False}
    record = {"feedback_item_id": "f1", "feedback_kind": "statute", "candidate_id": "c1",
        "candidate_ids": ["c1"], "article_uid": "a1", "clause_ids": ["p1"],
        "protocol_complete": True, "pending_manual": False, "semantic_status": "provided_relation_confirmed",
        "verdicts": ["confirmed"], "observations": [{"job_id": "j1", "window_id": "w1", "verdict": "confirmed",
            "reason": "双方均承诺履约", "quotes": {"clause_quote": "双方应按约履行义务", "article_quote": "按约定履行义务"}}]}
    call = {"call_id": "call1", "job_id": "j1", "window_id": "w1", "source_sha256": "hash1",
        "status": "ok", "feedback_item_ids": ["f1"], "verified_feedback_item_ids": ["f1"],
        "verifications": [{"feedback_item_id": "f1", "candidate_id": "c1", "article_uid": "a1",
            "verdict": "confirmed", "clause_quote": "双方应按约履行义务", "article_quote": "按约定履行义务"}],
        "request_receipts": [{"call_id": "call1", "job_id": "j1", "window_id": "w1", "source_sha256": "hash1", "status": "succeeded"}]}
    work = {"work_item_id": "work1", "kind": "statute_feedback", "legacy_feedback_item_id": "f1",
        "provided_feedback": feedback, "pending": False, "protocol_complete": True, "pending_reasons": [],
        "semantic_status": "provided_relation_confirmed", "source_status": "bound_provided_evidence",
        "execution_status": "executed", "actual_processed": True}
    return {"contract_id": "contract1", "document": {"clauses": [{"clause_id": "p1", "body": "双方应按约履行义务。"}]},
        "risks": [{"risk_id": "risk1", "message": "履行义务主张", "clause_ids": ["p1"],
            "spans": [{"clause_id": "p1", "quote": "双方应按约履行义务"}], "citations": [], "review_required": False,
            "evidence": [{"kind": "text", "extra": {"candidate_id": "c1"}}]}],
        "feedback": {"risk1": {"status": "confirmed", "note": "人工意见"}},
        "stats": {"logic_llm": {"feedback_consumption_current": [record], "feedback_latest_runs": [{"calls": [call]}],
            "review_work_items": {"items": [work]}}}}


def feedback_row(report):
    return next(row for row in build_review_explanations(report)["items"] if row["item_id"] == "f1")


class ReviewExplanationsTests(unittest.TestCase):
    def test_ocr_date_conflict_requires_original_page_check_and_keeps_machine_record(self):
        report = report_fixture()
        report["document"]["clauses"][0]["metadata"] = {"source_id": "original-source",
            "source_page_start": 13, "source_page_end": 13, "original_sha256": "stored-hash",
            "extraction_quality": {"status": "machine_ocr_unverified", "human_verified": False}}
        report["risks"][0].update(code="date_reversed", message="日期倒置", review_required=False)
        before = copy.deepcopy(report)
        row = next(row for row in build_review_explanations(report)["items"] if row["kind"] == "risk")
        self.assertTrue(row["manual_required"])
        self.assertIn("ocr_numeric_field_unverified", row["manual_reasons"])
        self.assertIn("原始PDF", row["next_action"])
        self.assertIn("不将识别异常", row["next_action"])
        self.assertEqual(row["contract_evidence"][0]["source_id"], "original-source")
        self.assertEqual(row["contract_evidence"][0]["source_page_start"], 13)
        self.assertEqual(report, before)

    def test_verified_numeric_source_does_not_receive_unverified_ocr_reason(self):
        report = report_fixture()
        report["document"]["clauses"][0]["metadata"] = {
            "extraction_quality": {"status": "machine_ocr_unverified", "human_verified": True}}
        report["risks"][0].update(code="amount_inconsistent", message="金额大小写不一致")
        row = next(row for row in build_review_explanations(report)["items"] if row["kind"] == "risk")
        self.assertNotIn("ocr_numeric_field_unverified", row["manual_reasons"])
        self.assertFalse(row["legal_truth_established"])

    def test_bound_relation_is_limited_and_input_immutable(self):
        report = report_fixture()
        before = copy.deepcopy(report)
        row = feedback_row(report)
        self.assertEqual(row["status"], "provided_relation_checked")
        self.assertFalse(row["manual_required"])
        self.assertFalse(row["legal_truth_established"])
        self.assertTrue(row["calls"][0]["receipt_bound"])
        self.assertTrue(row["calls"][0]["response_bound"])
        self.assertEqual(report, before)

    def test_confirmed_without_actual_receipt_remains_manual(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]["request_receipts"] = []
        row = feedback_row(report)
        self.assertTrue(row["manual_required"])
        self.assertIn("bound_processing_receipt_missing_or_invalid", row["manual_reasons"])

    def test_another_calls_receipt_cannot_acknowledge_current(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]["request_receipts"][0]["source_sha256"] = "other"
        self.assertTrue(feedback_row(report)["manual_required"])

    def test_forged_consumption_without_bound_response_stays_manual(self):
        report = report_fixture()
        call = report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]
        call["verifications"][0]["candidate_id"] = "another-candidate"
        row = feedback_row(report)
        self.assertFalse(row["calls"][0]["response_bound"])
        self.assertTrue(row["manual_required"])

    def test_no_historical_fallback_for_explicitly_empty_current(self):
        report = report_fixture()
        logic = report["stats"]["logic_llm"]
        logic["feedback_runs"] = logic.pop("feedback_latest_runs")
        logic["feedback_latest_runs"] = []
        row = feedback_row(report)
        self.assertEqual(row["calls"], [])
        self.assertTrue(row["manual_required"])

    def test_confirmed_projection_cannot_rewrite_actual_needs_manual(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]["verifications"][0]["verdict"] = "needs_manual"
        self.assertTrue(feedback_row(report)["manual_required"])

    def test_needs_manual_pending_and_disagreement_cannot_be_cleared(self):
        for mutation in ("pending_manual", "disagreement", "newer_evidence"):
            report = report_fixture()
            record = report["stats"]["logic_llm"]["feedback_consumption_current"][0]
            if mutation == "newer_evidence":
                record["upstream_review_reasons"] = ["newer_evidence_deferred_next_cycle"]
            else:
                record[mutation] = True
            self.assertTrue(feedback_row(report)["manual_required"], mutation)
        report = report_fixture()
        record = report["stats"]["logic_llm"]["feedback_consumption_current"][0]
        record["verdicts"] = ["needs_manual"]
        self.assertTrue(feedback_row(report)["manual_required"])

    def test_quote_must_belong_to_bound_contract_and_law(self):
        for key in ("clause_quote", "article_quote"):
            report = report_fixture()
            report["stats"]["logic_llm"]["feedback_consumption_current"][0]["observations"][0]["quotes"][key] = "虚构引文"
            self.assertTrue(feedback_row(report)["manual_required"], key)

    def test_exact_quote_does_not_establish_unsupported_claim(self):
        report = report_fixture()
        report["risks"][0]["evidence"].append({"kind": "claim_grounding", "extra": {
            "claim_grounding_status": "unsupported", "manual_required": True}})
        row = build_review_explanations(report)["items"][0]
        self.assertEqual(row["evidence_status"], "contract_quote_located")
        self.assertTrue(row["manual_required"])
        self.assertIn("claim_grounding_pending", row["manual_reasons"])
        self.assertEqual(row["human_feedback"]["status"], "confirmed")

    def test_unknown_subject_fact_never_promoted_by_imported_record(self):
        report = report_fixture()
        report["stats"]["subject"] = {"stats": {"fact_checks": [{"subject_name": "测试公司",
            "checks": {"registration": "unknown", "license": "valid"}, "independently_verified": False,
            "records": [{"source": {"publisher": "人工导入"}}]}]}}
        rows = [row for row in build_review_explanations(report)["items"] if row["kind"] == "subject_fact"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["manual_required"] for row in rows))
        self.assertEqual(rows[0]["fact_status"], "unknown")

    def test_missing_response_work_remains_visible(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_consumption_current"] = []
        row = feedback_row(report)
        self.assertTrue(row["manual_required"])
        self.assertEqual(row["model_verdicts"], [])

    def test_legacy_and_empty_reports_do_not_invent_completion(self):
        self.assertEqual(build_review_explanations({})["summary"]["total"], 0)
        report = {"risks": [{"risk_id": "old", "message": "旧报告"}]}
        row = build_review_explanations(report)["items"][0]
        self.assertEqual(row["status"], "manual_review")
        self.assertFalse(row["legal_truth_established"])

    def test_raw_pending_axes_override_optimistic_derived_summaries(self):
        mutations = {
            "semantic_pending": True, "decision_scope": "local_window_only",
            "input_truncated": True, "token_coverage_complete": False,
            "pair_coverage_complete": False, "coverage_complete": False,
            "window_disagreement": True, "upstream_claim_grounding_pending": True,
            "article_quote_truncated": True, "fact_evidence_text_truncated": True,
            "source_evidence_text_truncated": True, "source_complete": False,
            "upstream_pending": True, "newer_input_pending": True,
        }
        for location in ("supplied", "record", "window_audit"):
            for key, value in mutations.items():
                with self.subTest(location=location, key=key):
                    report = report_fixture()
                    logic = report["stats"]["logic_llm"]
                    if location == "supplied":
                        target = logic["review_work_items"]["items"][0]["provided_feedback"]
                    elif location == "record":
                        target = logic["feedback_consumption_current"][0]
                    else:
                        target = logic["review_work_items"]["items"][0]["provided_feedback"].setdefault("window_audit", {})
                    target[key] = value
                    before = copy.deepcopy(report)
                    self.assertTrue(feedback_row(report)["manual_required"])
                    self.assertEqual(report, before)

    def test_unknown_flags_are_not_treated_as_resolved(self):
        for key in ("semantic_pending", "input_truncated", "coverage_complete", "decision_scope"):
            for value in (None, "unknown", "false", 0):
                with self.subTest(key=key, value=value):
                    report = report_fixture()
                    report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"][key] = value
                    self.assertTrue(feedback_row(report)["manual_required"])

    def test_explicitly_resolved_axes_still_allow_given_relation_support(self):
        report = report_fixture()
        supplied = report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"]
        supplied.update(semantic_pending=False, decision_scope="whole_input", input_truncated=False,
                        token_coverage_complete=True, pair_coverage_complete=True, coverage_complete=True,
                        window_disagreement=False, upstream_claim_grounding_pending=False,
                        article_quote_truncated=False, source_complete=True,
                        window_audit={"input_valid": True, "semantic_pending": False})
        self.assertEqual(feedback_row(report)["status"], "provided_relation_checked")

    def test_work_axes_cannot_be_overruled_by_pending_false(self):
        for key, values in {
            "semantic_status": ("pending_manual", "unknown", None),
            "source_status": ("external_evidence_pending", "missing_source_binding", "unknown", None),
            "execution_status": ("not_executed", "unknown", None),
            "actual_processed": (False, None, "true"),
            "protocol_complete": (False, None),
            "pending": (True, None),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    report = report_fixture()
                    report["stats"]["logic_llm"]["review_work_items"]["items"][0][key] = value
                    self.assertTrue(feedback_row(report)["manual_required"])

    def test_missing_work_ledger_is_not_success_evidence(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["review_work_items"] = {"items": []}
        report["risks"][0]["citations"] = [{"article_uid": "a1", "quote": "应当按约定履行义务。"}]
        self.assertTrue(feedback_row(report)["manual_required"])

    def test_global_pending_ledger_cannot_be_overruled_by_local_summary(self):
        for location in ("latest", "cross"):
            report = report_fixture()
            if location == "latest":
                report["stats"]["logic_llm"]["feedback_latest_runs"][0]["uncompleted_feedback_item_ids"] = ["f1"]
            else:
                report["stats"]["cross_validation"] = {"pending_feedback_item_ids": ["f1"]}
            self.assertTrue(feedback_row(report)["manual_required"])

    def test_unknown_call_status_and_replacement_article_do_not_pass(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]["status"] = "unknown"
        self.assertTrue(feedback_row(report)["manual_required"])
        report = report_fixture()
        report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"]["article_quote"] = "另一段提供给模型的条文。"
        report["risks"][0]["citations"] = [{"article_uid": "a1", "quote": "应当按约定履行义务。"}]
        self.assertTrue(feedback_row(report)["manual_required"])

    def test_manual_instructions_follow_actual_missing_context_and_claim(self):
        report = report_fixture()
        supplied = report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"]
        supplied.update(semantic_pending=True, pair_coverage_complete=False, upstream_claim_grounding_pending=True)
        before = copy.deepcopy(report)
        row = feedback_row(report)
        self.assertTrue(row["manual_required"])
        self.assertIn("原合同附件", row["next_action"])
        self.assertIn("同一主体", row["next_action"])
        self.assertIn("同一标的", row["next_action"])
        self.assertIn("同一履约阶段", row["next_action"])
        self.assertIn("已触发的条件", row["next_action"])
        self.assertIn("完整主张", row["next_action"])
        self.assertEqual(row["next_action"].count("打开关联条款"), 1)
        self.assertNotIn("统一社会信用代码", row["next_action"])
        self.assertEqual(report, before)

    def test_manual_instructions_explain_disagreement_and_subject_sources(self):
        report = report_fixture()
        supplied = report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"]
        supplied["disagreement"] = True
        text = feedback_row(report)["next_action"]
        self.assertIn("规则判断、神经模型判断", text)
        self.assertIn("适用条件、例外", text)
        report["stats"]["subject"] = {"stats": {"fact_checks": [{"subject_name": "测试公司",
            "checks": {"authorization": "unknown"}, "independently_verified": False, "records": []}]}}
        row = next(item for item in build_review_explanations(report)["items"] if item["kind"] == "subject_fact")
        self.assertIn("统一社会信用代码", row["next_action"])
        self.assertIn("有效期和适用范围", row["next_action"])
        self.assertIn("对应本合同", row["next_action"])
        self.assertFalse(row["independently_verified"])

    def test_manual_instructions_distinguish_receipt_recovery_and_truncation(self):
        report = report_fixture()
        report["stats"]["logic_llm"]["feedback_latest_runs"][0]["calls"][0]["request_receipts"] = []
        supplied = report["stats"]["logic_llm"]["review_work_items"]["items"][0]["provided_feedback"]
        supplied["article_quote_truncated"] = True
        row = feedback_row(report)
        self.assertIn("call_id、job_id、window_id、source_sha256", row["next_action"])
        self.assertIn("保留原失败记录", row["next_action"])
        self.assertIn("完整原文和模型输出", row["next_action"])
        self.assertIn("保持 pending", row["next_action"])
        self.assertTrue(row["manual_required"])
        self.assertEqual(len(row["next_actions"]), len(set(row["next_actions"])))

    def test_supported_relation_does_not_invent_missing_evidence_instructions(self):
        row = feedback_row(report_fixture())
        self.assertFalse(row["manual_required"])
        self.assertIn("最终法律意见仍由复核人员确认", row["next_action"])
        self.assertNotIn("补充对应主体", row["next_action"])
        self.assertFalse(row["legal_truth_established"])


if __name__ == "__main__":
    unittest.main()
