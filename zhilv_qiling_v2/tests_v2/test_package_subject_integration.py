"""Offline integration and installed-package regressions, not accuracy scores."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi_run as G, nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.contract_packages import DEFAULT_STORE, prepare_package_document
from zhilv_qiling_v2.pipeline import build_contract
from zhilv_qiling_v2.review_work_items import feedback_work_items
from zhilv_qiling_v2.review_explanations import build_review_explanations
from zhilv_qiling_v2.tests_v2.test_core_completion import configuration
from zhilv_qiling_v2.tests_v2.test_subject_identity_binding import (
    ID_A, ID_B, NAME_A, NAME_B, applicability, fact, make_contract,
)
from zhilv_qiling_v2.tests_v2.test_subject_package_availability import check


class PackageSubjectIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.contract = make_contract([
            ("buyer", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("seller", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("service", "乙方应提供本合同签约授权。"),
        ])
        self.gap = {"document_id": "authorization_documents", "title": "签约授权资料",
            "availability": "unavailable_private", "reason_source": "user_statement",
            "fact_types": ["authorization"], "subject_roles": ["seller"]}
        self.request = {"request_id": "auth", "risk_code": "need_authorization", "clause_ids": ["service"]}

    def feedback(self, *, gap=None, records=()):
        result = check(self.contract, context={"missing_documents": [self.gap if gap is None else gap]},
                       records=records, requests=[self.request])
        subject = {**result.stats, "candidates": [candidate.to_dict() for candidate in result.candidates]}
        ctx = SimpleNamespace(contract=self.contract, bus=SimpleNamespace(inbox=lambda *args, **kwargs: []))
        return subject, G._subject_fact_feedback(ctx, subject)

    def test_private_request_reaches_feedback_and_work_ledger_without_becoming_completed(self):
        subject, facts = self.feedback()
        request_items = [item for item in facts if item.get("request_id") == "auth"]
        self.assertEqual(len(request_items), 1)
        item = request_items[0]
        self.assertFalse(item["auto_retry"])
        self.assertEqual(item["effective_status"], "unknown")
        work = feedback_work_items(self.contract.contract_id, [], [item], subject, {"calls": []},
            [{"feedback_item_id": item["feedback_item_id"], "protocol_complete": True,
              "semantic_status": "provided_relation_confirmed"}],
            {clause.clause_id: clause.text for clause in self.contract.document.clauses})
        fact_work = next(row for row in work if row["kind"] == "subject_fact")
        self.assertFalse(fact_work["dispatchable"])
        self.assertTrue(fact_work["pending"])
        self.assertFalse(fact_work["actual_processed"])
        self.assertEqual(fact_work["receipt_call_ids"], [])
        self.assertEqual(fact_work["semantic_status"], "pending_manual")
        self.assertIn("auth", subject["pending_verification_request_ids"])
        explanation = build_review_explanations({"contract_id": self.contract.contract_id,
            "document": {"clauses": [clause.to_dict() for clause in self.contract.document.clauses]},
            "stats": {"logic_llm": {"review_work_items": {"items": work}}}})
        row = next(row for row in explanation["items"] if row["item_id"] == item["feedback_item_id"])
        self.assertIn("保留用户", row["next_action"])
        self.assertNotIn("补充对应主体的登记", row["next_action"])
        self.assertFalse(row["auto_retry"])

    def test_public_unknown_source_does_not_become_private_no_retry(self):
        _, facts = self.feedback(gap={**self.gap, "availability": "unverified_public"})
        self.assertTrue(facts)
        self.assertFalse(any(item.get("auto_retry") is False for item in facts))

    def test_private_authorization_cannot_suspend_other_missing_fact_types(self):
        _, facts = self.feedback()
        private = [item for item in facts if item.get("auto_retry") is False]
        self.assertTrue(private)
        for item in private:
            if item["fact_type"] == "unresolved_subject_checks":
                self.assertLessEqual(set(item["check_states"]), {"authorization"})
            else:
                self.assertEqual(item["fact_type"], "authorization")
        public = [item for item in facts if item["subject_role"] == "seller"
                  and item.get("fact_type") == "unresolved_subject_checks"
                  and {"registration", "credit"}.intersection(item.get("check_states", {}))]
        self.assertTrue(public)
        self.assertTrue(all(item.get("auto_retry") is not False for item in public))

    def test_clause_limited_private_gap_does_not_suspend_another_clause(self):
        _, facts = self.feedback(gap={**self.gap, "clause_ids": ["service"]})
        self.assertTrue(any(item.get("auto_retry") is False for item in facts))
        for item in facts:
            if item.get("auto_retry") is False:
                self.assertEqual(item["clause_ids"], ["service"])

    def test_valid_authorization_does_not_leave_other_unknowns_privately_suspended(self):
        record = fact(NAME_B, ID_B, fact_type="authorization", contract_id=self.contract.contract_id,
                      applicability=applicability())
        subject, facts = self.feedback(records=[record])
        self.assertEqual(subject["verification_results"][0]["status"], "completed")
        self.assertFalse(any(item.get("auto_retry") is False for item in facts))
        auth = [item for item in facts if item["fact_type"] == "authorization"]
        self.assertTrue(auth)
        self.assertTrue(all(item["effective_status"] == "valid" for item in auth))

    def test_private_request_does_not_call_model_under_either_feedback_policy(self):
        _, facts = self.feedback()
        item = next(row for row in facts if row.get("request_id") == "auth")
        clause = next(row for row in self.contract.document.clauses if row.clause_id == "service")
        for policy in ("all_matching_windows", "cover_all_context_v1"):
            with self.subTest(policy=policy), patch.object(N.L, "request_json") as request:
                _, meta, _ = N._llm_logic(self.contract, configuration(llm_feedback_window_policy=policy),
                                          [clause], subject_fact_feedback=[item])
                request.assert_not_called()
                self.assertFalse(meta["feedback_complete"])
                self.assertIn(item["feedback_item_id"], meta["uncompleted_feedback_item_ids"])
                self.assertEqual(meta["calls"], [])

    def test_stale_no_retry_flag_cannot_suspend_effective_available_evidence(self):
        item = {"feedback_item_id": "known-source", "feedback_kind": "subject_fact",
            "fact_id": "fact", "source_id": "source", "subject_name": NAME_B,
            "fact_type": "authorization", "clause_id": "service", "clause_ids": ["service"],
            "effective_status": "valid", "source_complete": True, "upstream_pending": False,
            "auto_retry": False, "fact_evidence_text": "绑定记录", "source_evidence_text": "绑定来源"}
        clause = next(row for row in self.contract.document.clauses if row.clause_id == "service")
        reply = SimpleNamespace(content={"subject_fact_verifications": []}, model="offline-fixture")
        with patch.object(N.L, "request_json", return_value=reply) as request:
            N._llm_logic(self.contract, configuration(), [clause], subject_fact_feedback=[item])
            self.assertTrue(request.called)


class InstalledRealPackageRegressionTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_STORE / "beijing_procurement_20261121" / "manifest.json").is_file(),
                         "Local real package not installed; no synthetic replacement")
    def test_actual_fifteen_page_contract_has_two_parties_and_no_false_licensed_industry(self):
        cfg = Config(enable_llm=False, ner_enable_model=False, ner_use_bert=False)
        document = prepare_package_document("beijing_procurement_20261121", cfg)
        result = check(build_contract(document, cfg))
        self.assertEqual(document.stats["pages"], 15)
        self.assertEqual({(node.name, node.role) for node in result.nodes}, {
            ("北京市统计局", "buyer"), ("北京力信诚印务科技有限公司", "seller")})
        self.assertEqual(result.stats["requires"], 0)
        self.assertTrue(all(check_["checks"]["authorization"] == "unknown"
                            for check_ in result.stats["fact_checks"]))
        self.assertEqual(sum(gap["availability"] == "unavailable_private"
                             for gap in result.stats["evidence_gaps"]), 7)


if __name__ == "__main__":
    unittest.main()
