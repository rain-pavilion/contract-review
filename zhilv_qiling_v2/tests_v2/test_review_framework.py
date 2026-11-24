"""Three-layer projection checks: missing detections never become clearance."""
import copy
import unittest

from zhilv_qiling_v2.review_points import review_point_catalog
from zhilv_qiling_v2.review_assessment import build_review_framework, review_framework_markdown


def payment_report():
    return {
        "contract_id": "contract-test",
        "document": {"clauses": [{"clause_id": "p1", "body": "甲方应于验收后十日内支付价款。"}]},
        "risks": [],
        "stats": {"review_context": {"contract_type": "买卖", "party_position": "买方"},
            "review_point_checks": [{
                "check_id": "payment-1", "point_id": "commercial.payment_terms",
                "status": "no_issue_detected", "execution_status": "executed",
                "scope": "仅核对已提供文本中识别出的付款句是否存在直接绑定的期限。",
                "scope_complete": True, "clause_ids": ["p1"],
                "evidence": [{"clause_id": "p1", "quote": "验收后十日内支付价款", "located": True}],
                "actual_issue": "在上述限定范围内未检出缺少付款期限的问题。",
                "impact": "", "recommendation": "结合完整交易背景核对起算条件。",
                "severity": "info", "manual_required": False, "pending_reasons": [],
                "rule_version": "payment-screen-v1", "legal_truth_established": False,
            }]},
    }


class ReviewFrameworkTests(unittest.TestCase):
    def assessment(self, report, point_id):
        return next(row for row in build_review_framework(report)["assessments"] if row["point_id"] == point_id)

    def test_catalog_gives_independent_actionable_criteria_for_all_areas(self):
        self.assertTrue(callable(review_point_catalog))
        catalog = review_point_catalog()
        expected = {"subject.identity", "subject.authorization", "subject.qualification", "subject.title",
            "commercial.amount", "commercial.payment_terms", "performance.delivery", "performance.acceptance",
            "liability.balance", "legal.compliance", "legal.dispute", "consistency.references", "confidentiality",
            "subject.scope"}
        self.assertEqual({row["point_id"] for row in catalog}, expected)
        for row in catalog:
            for key in ("domain", "question", "criterion", "responsible_agent", "required_evidence", "scope", "version",
                        "impact_template", "recommendation_template"):
                self.assertTrue(row[key], (row["point_id"], key))
        catalog[0]["risk_codes"].append("forged")
        self.assertNotIn("forged", review_point_catalog()[0]["risk_codes"])

    def test_empty_report_keeps_every_unperformed_point_unchecked(self):
        framework = build_review_framework({})
        self.assertEqual(framework["schema_version"], "contract-review-three-layers-v1")
        self.assertEqual(len(framework["assessments"]), 14)
        self.assertEqual({row["status"] for row in framework["assessments"]}, {"not_checked"})
        self.assertTrue(all(row["execution_status"] == "not_recorded" for row in framework["assessments"]))
        self.assertTrue(all(row["legal_truth_established"] is False for row in framework["assessments"]))
        self.assertEqual(framework["workflow"]["status"], "not_recorded")

    def test_placeholder_risk_never_means_any_point_passed(self):
        framework = build_review_framework({"risks": [{"risk_id": "placeholder", "code": "no_substantive_risk"}]})
        self.assertEqual({row["status"] for row in framework["assessments"]}, {"not_checked"})
        self.assertEqual(framework["summary"]["by_status"]["no_issue_detected"], 0)

    def test_existing_risk_retains_actual_ids_and_conditional_impact(self):
        report = payment_report()
        report["risks"] = [{"risk_id": "r1", "code": "amount_inconsistent", "message": "同一价款大小写不一致",
            "clause_ids": ["p1"], "spans": [{"clause_id": "p1", "quote": "支付价款"}]}]
        report["review_explanations"] = {"items": [{"item_id": "r1", "kind": "risk", "risk_id": "r1",
            "manual_required": True, "manual_reasons": ["ocr_numeric_field_unverified"]}]}
        row = self.assessment(report, "commercial.amount")
        self.assertEqual(row["status"], "issue_found")
        self.assertEqual(row["risk_ids"], ["r1"])
        self.assertEqual(row["explanation_item_ids"], ["r1"])
        self.assertIn("同一价款大小写不一致", row["actual_issue"])
        self.assertIn("可能", row["impact"])
        self.assertEqual(row["impact_basis"], "point_template")
        self.assertTrue(row["manual_required"])
        self.assertIn("ocr_numeric_field_unverified", row["pending_reasons"])
        self.assertTrue(row["evidence"][0]["located"])

    def test_unlocated_risk_quote_is_not_promoted_by_self_reported_location(self):
        report = payment_report()
        report["risks"] = [{"risk_id": "r1", "code": "amount_missing", "message": "待补价款",
            "clause_ids": ["p1"], "spans": [{"clause_id": "p1", "quote": "虚构引文", "located": True}]}]
        row = self.assessment(report, "commercial.amount")
        self.assertFalse(row["evidence"][0]["located"])
        self.assertTrue(row["manual_required"])
        self.assertIn("contract_quote_missing_or_unlocated", row["pending_reasons"])

    def test_subject_fact_pending_maps_to_its_review_question(self):
        report = {"review_explanations": {"items": [{"item_id": "fact-1", "kind": "subject_fact",
            "fact_type": "authorization", "fact_status": "unknown", "manual_required": True,
            "manual_reasons": ["source_status_unknown"], "contract_evidence": []}]}}
        row = self.assessment(report, "subject.authorization")
        self.assertEqual(row["status"], "needs_manual")
        self.assertEqual(row["explanation_item_ids"], ["fact-1"])
        self.assertEqual(row["risk_ids"], [])

    def test_valid_explicit_check_yields_only_a_bounded_negative_result(self):
        report = payment_report()
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "no_issue_detected")
        self.assertEqual(row["execution_status"], "executed")
        self.assertIn("直接绑定", row["scope"])
        self.assertFalse(row["manual_required"])
        self.assertFalse(row["legal_truth_established"])
        self.assertEqual(row["source_status"], "provided_text_located")

    def test_completed_execution_receipt_normalizes_to_executed(self):
        report = payment_report()
        report["stats"]["review_point_checks"][0]["execution_status"] = "completed"
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "no_issue_detected")
        self.assertEqual(row["execution_status"], "executed")

    def test_explicit_unperformed_check_remains_unchecked(self):
        report = payment_report()
        report["stats"]["review_point_checks"] = [{"check_id": "not-run", "point_id": "commercial.payment_terms",
            "status": "not_checked", "execution_status": "not_run", "manual_required": True,
            "pending_reasons": ["document_missing"]}]
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "not_checked")
        self.assertEqual(row["execution_status"], "not_recorded")
        self.assertTrue(row["manual_required"])
        self.assertIn("document_missing", row["pending_reasons"])

    def test_receipt_location_is_recomputed_from_document_not_input_flag(self):
        report = payment_report()
        report["stats"]["review_point_checks"][0]["evidence"][0]["located"] = False
        row = self.assessment(report, "commercial.payment_terms")
        self.assertTrue(row["evidence"][0]["located"])
        self.assertEqual(row["status"], "no_issue_detected")

    def test_negative_result_without_supplied_document_requires_manual(self):
        report = payment_report()
        report.pop("document")
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "needs_manual")
        self.assertFalse(any(item["located"] for item in row["evidence"]))
        self.assertIn("contract_quote_missing_or_unlocated", row["pending_reasons"])

    def test_negative_result_must_bind_quote_to_its_own_clause(self):
        report = payment_report()
        report["document"]["clauses"].append({"clause_id": "p2", "body": "虚构引文"})
        report["stats"]["review_point_checks"][0]["evidence"][0]["quote"] = "虚构引文"
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "needs_manual")
        self.assertFalse(row["evidence"][0]["located"])

    def test_incomplete_scope_never_becomes_negative_clearance(self):
        for complete in (False, None, "true"):
            with self.subTest(complete=complete):
                report = payment_report()
                report["stats"]["review_point_checks"][0]["scope_complete"] = complete
                row = self.assessment(report, "commercial.payment_terms")
                self.assertEqual(row["status"], "needs_manual")
                self.assertIn("review_point_scope_incomplete", row["pending_reasons"])

    def test_missing_execution_or_identity_makes_receipt_unusable_for_clearance(self):
        for field in ("check_id", "execution_status", "rule_version", "scope"):
            with self.subTest(field=field):
                report = payment_report()
                report["stats"]["review_point_checks"][0].pop(field)
                row = self.assessment(report, "commercial.payment_terms")
                self.assertNotIn(row["status"], ("no_issue_detected", "not_applicable"))
                self.assertTrue(row["manual_required"])

    def test_negative_receipt_cannot_hide_pending_or_an_existing_risk(self):
        report = payment_report()
        report["review_explanations"] = {"items": [{"item_id": "payment-pending", "point_id": "commercial.payment_terms",
            "kind": "review_work", "manual_required": True, "manual_reasons": ["input_truncated"]}]}
        self.assertEqual(self.assessment(report, "commercial.payment_terms")["status"], "needs_manual")
        report["risks"] = [{"risk_id": "r1", "code": "amount_missing", "message": "金额未定", "clause_ids": ["p1"]}]
        report["stats"]["review_point_checks"][0]["point_id"] = "commercial.amount"
        row = self.assessment(report, "commercial.amount")
        self.assertEqual(row["status"], "issue_found")
        self.assertEqual(row["risk_ids"], ["r1"])

    def test_receipt_pending_reasons_and_manual_flag_override_negative_status(self):
        for changes in ({"pending_reasons": ["payment_reference_unresolved"]}, {"manual_required": True},
                        {"execution_status": "failed"}, {"legal_truth_established": True}):
            with self.subTest(changes=changes):
                report = payment_report()
                report["stats"]["review_point_checks"][0].update(changes)
                self.assertEqual(self.assessment(report, "commercial.payment_terms")["status"], "needs_manual")

    def test_malformed_status_is_rejected_without_crashing_projection(self):
        report = payment_report()
        report["stats"]["review_point_checks"][0]["status"] = ["no_issue_detected"]
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "needs_manual")
        self.assertIn("review_point_receipt_invalid", row["pending_reasons"])

    def test_malformed_receipt_list_entries_cannot_silently_disappear(self):
        for field, bad in (("pending_reasons", {"reason": "unresolved"}), ("clause_ids", None), ("evidence", "unreadable")):
            with self.subTest(field=field):
                report = payment_report()
                report["stats"]["review_point_checks"][0][field].append(bad)
                row = self.assessment(report, "commercial.payment_terms")
                self.assertEqual(row["status"], "needs_manual")
                self.assertIn("review_point_receipt_invalid", row["pending_reasons"])

    def test_each_clause_in_a_completed_check_requires_its_bound_quote(self):
        report = payment_report()
        report["document"]["clauses"].append({"clause_id": "p2", "body": "尾款另议。"})
        report["stats"]["review_point_checks"][0]["clause_ids"].append("p2")
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "needs_manual")

    def test_one_located_quote_cannot_clear_another_unlocated_risk_quote(self):
        report = payment_report()
        report["risks"] = [{"risk_id": "r1", "code": "amount_missing", "message": "待补价款", "clause_ids": ["p1"],
            "spans": [{"clause_id": "p1", "quote": "支付价款"}, {"clause_id": "p1", "quote": "虚构引文"}]}]
        row = self.assessment(report, "commercial.amount")
        self.assertEqual(row["status"], "issue_found")
        self.assertTrue(row["manual_required"])
        self.assertIn("contract_quote_missing_or_unlocated", row["pending_reasons"])

    def test_duplicate_conflicting_receipts_cannot_erase_pending(self):
        report = payment_report()
        pending = copy.deepcopy(report["stats"]["review_point_checks"][0])
        pending.update(status="needs_manual", manual_required=True, pending_reasons=["unresolved_reference"])
        report["stats"]["review_point_checks"].insert(0, pending)
        row = self.assessment(report, "commercial.payment_terms")
        self.assertEqual(row["status"], "needs_manual")
        self.assertIn("unresolved_reference", row["pending_reasons"])

    def test_unknown_or_malformed_receipts_never_invent_an_assessment_or_success(self):
        report = payment_report()
        report["stats"]["review_point_checks"] = [None, "done", {"point_id": "forged", "status": "no_issue_detected"}]
        framework = build_review_framework(report)
        self.assertEqual(len(framework["assessments"]), 14)
        self.assertEqual({row["status"] for row in framework["assessments"]}, {"not_checked"})

    def test_absence_of_trace_does_not_claim_six_agent_execution(self):
        framework = build_review_framework(payment_report())
        stages = {row["id"]: row for row in framework["workflow"]["stages"]}
        self.assertEqual(stages["analysis"]["status"], "not_recorded")
        self.assertEqual(stages["verification"]["status"], "not_recorded")
        self.assertEqual(stages["assessment"]["status"], "not_recorded")
        self.assertEqual(stages["analysis"]["agents"], ["ner", "logic"])
        self.assertEqual([(row["id"], row["number"]) for row in framework["workflow"]["core_nodes"]],
            [("orchestrator", 1), ("ner", 2), ("logic", 3), ("subject", 4), ("statute", 5), ("judge", 6)])

    def test_failed_agent_trace_and_actual_pending_gate_workflow(self):
        report = payment_report()
        report["trace"] = [{"node": "ner", "status": "ok"}, {"node": "logic", "status": "error"},
                           {"node": "subject", "status": "ok"}, {"node": "statute", "status": "ok"}]
        report["stats"]["cross_validation"] = {"converged": True, "pending_work_item_ids": ["unresolved-1"]}
        framework = build_review_framework(report)
        stages = {row["id"]: row for row in framework["workflow"]["stages"]}
        self.assertEqual(stages["analysis"]["status"], "failed")
        self.assertEqual(stages["verification"]["status"], "needs_manual")
        self.assertGreaterEqual(framework["workflow"]["pending_count"], 1)
        self.assertTrue(framework["workflow"]["manual_required"])
        self.assertEqual(stages["human_review"]["status"], "required")

    def test_legacy_subject_trace_alone_cannot_complete_statute_verification(self):
        report = payment_report()
        report["trace"] = [{"node": "subject_check", "status": "ok"}]
        stages = {row["id"]: row for row in build_review_framework(report)["workflow"]["stages"]}
        self.assertEqual(stages["verification"]["status"], "partial")

    def test_unlabelled_agent_summary_cannot_hide_an_explicit_execution_trace(self):
        report = payment_report()
        report["trace"] = [{"node": "understand", "status": "ok"}, {"node": "subject_check", "status": "ok"},
            {"node": "legal", "status": "ok"}, {"node": "judge", "status": "ok"}]
        report["stats"]["agents"] = [{"agent": "understand", "degraded": []}, {"agent": "legal", "degraded": []},
            {"agent": "judge", "degraded": []}]
        stages = {row["id"]: row for row in build_review_framework(report)["workflow"]["stages"]}
        self.assertEqual(stages["parse"]["status"], "completed")
        self.assertEqual(stages["verification"]["status"], "completed")
        self.assertEqual(stages["assessment"]["status"], "completed")

    def test_real_agent_degradation_is_retained_alongside_success_trace(self):
        report = payment_report()
        report["trace"] = [{"node": "ner", "status": "ok"}, {"node": "logic", "status": "ok"}]
        report["stats"]["agents"] = [{"agent": "logic", "degraded": ["model_response_incomplete"]}]
        stages = {row["id"]: row for row in build_review_framework(report)["workflow"]["stages"]}
        self.assertEqual(stages["analysis"]["status"], "needs_manual")

    def test_subject_unknown_and_review_queue_are_not_lost_without_explanations(self):
        report = payment_report()
        report["review_queue"] = ["r-manual"]
        report["stats"]["subject"] = {"stats": {"fact_checks": [{"subject_name": "测试主体", "checks": {"authorization": "unknown"},
            "independently_verified": False}]}}
        framework = build_review_framework(report)
        self.assertTrue(framework["workflow"]["manual_required"])
        row = next(row for row in framework["assessments"] if row["point_id"] == "subject.authorization")
        self.assertEqual(row["status"], "needs_manual")

    def test_framework_projection_does_not_mutate_input(self):
        report = payment_report()
        before = copy.deepcopy(report)
        framework = build_review_framework(report)
        framework["context"]["contract_type"] = "changed"
        framework["review_points"][0]["risk_codes"].append("changed")
        framework["assessments"][0]["clause_ids"].append("changed")
        self.assertEqual(report, before)

    def test_markdown_exposes_distinct_layers_and_scope_limits(self):
        framework = build_review_framework(payment_report())
        lines = review_framework_markdown(framework)
        self.assertIsInstance(lines, list)
        rendered = "\n".join(lines)
        for label in ("工作流", "审查点", "评估结果", "仅核对", "未执行", "法律结论"):
            self.assertIn(label, rendered)
        self.assertNotIn("审查通过", rendered)


if __name__ == "__main__":
    unittest.main()
