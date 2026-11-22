"""Offline scoping/privacy regressions; these fixtures are not human gold data."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import subject_facts as F, subject_graph as S
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.review_explanations import build_review_explanations
from zhilv_qiling_v2.tests_v2.test_subject_identity_binding import (
    ID_A, ID_B, NAME_A, NAME_B, applicability, fact, make_contract,
)


def check(contract, *, context=None, records=(), requests=()):
    with patch.object(F, "load_facts", return_value=(list(records), [])):
        return S.check_subjects(contract, Config(), package_context=context, verification_requests=requests)


class SubjectActivityScopeTests(unittest.TestCase):
    def test_goods_procurement_and_supplier_name_do_not_create_printing_or_cma_requirements(self):
        contract = make_contract([
            ("buyer", "甲方：北京市统计局。"),
            ("seller", "乙方：北京力信诚印务科技有限公司。"),
            ("goods", "乙方提供调查包、夹灯、鞋套、触屏手套、雨伞。"),
            ("quality", "货物应符合检验检测报告中的质量标准，产品认证资料随货交付。"),
            ("annex", "相关法律法规包括《印刷业管理条例》。"),
        ], parties=[("北京市统计局", "buyer"), ("北京力信诚印务科技有", "seller"),
                    ("北京力信诚印务科技有限公司", "seller")])
        result = check(contract)
        self.assertEqual([n.name for n in result.nodes], ["北京市统计局", "北京力信诚印务科技有限公司"])
        self.assertEqual(result.nodes[1].attrs["aliases"], ["北京力信诚印务科技有"])
        self.assertEqual(result.stats["requires"], 0)

    def test_actual_printing_activity_only_applies_to_performing_supplier(self):
        contract = make_contract([("parties", f"甲方：{NAME_A}。乙方：{NAME_B}。"),
                                  ("scope", "甲方采购印刷服务，由乙方负责印刷。")])
        result = check(contract)
        required = [edge for edge in result.edges if edge.kind == "requires"]
        self.assertTrue(required)
        self.assertTrue(all(edge.source == f"S:seller:{NAME_B}" for edge in required))
        self.assertTrue(all("scope" in edge.path for edge in required))

    def test_actual_testing_service_is_not_silently_removed(self):
        result = check(make_contract([("scope", "乙方提供检验检测服务，出具检验检测报告。")]))
        self.assertTrue(any("CMA" in edge.target for edge in result.edges if edge.kind == "requires"))

    def test_abolished_requirement_is_not_reintroduced_by_keyword_fallback(self):
        result = check(make_contract([("scope", "乙方提供物业服务。")]))
        self.assertFalse(any(edge.kind == "requires" for edge in result.edges))

    def test_qualification_authorization_and_credit_claims_do_not_spread_to_other_party(self):
        result = check(make_contract([("party", f"甲方：{NAME_A}，乙方：{NAME_B}。"),
            ("buyer", "甲方的法定代表人为某某，甲方提供营业执照，甲方查询信用中国。"),
            ("service", "乙方提供印刷服务。")]))
        claims = [edge for edge in result.edges if edge.kind in
                  ("claimed_holds", "claimed_authorized_by", "credit_mention")]
        self.assertTrue(claims)
        self.assertTrue(all(edge.source == f"S:buyer:{NAME_A}" for edge in claims))
        self.assertTrue(any(candidate.code == "need_license" for candidate in result.candidates))

    def test_similar_but_complete_legal_entities_are_not_merged(self):
        names = [("北京甲乙科技有限公司", "seller"), ("北京甲乙科技有限公司分公司", "seller")]
        result = check(make_contract([("names", "；".join(name for name, _ in names))], parties=names))
        self.assertEqual(len(result.nodes), 2)

    def test_truncated_name_without_original_full_name_is_not_repaired(self):
        names = [("北京力信诚印务科技有", "seller"), ("北京力信诚印务科技有限公司", "seller")]
        result = check(make_contract([("names", "乙方：北京力信诚印务科技有。")], parties=names))
        self.assertEqual(len(result.nodes), 2)

    def test_only_pure_role_definition_is_excluded_and_audit_is_retained(self):
        names = [("本合同乙方系指：_中标人", "seller"), ("北京中标人技术有限公司", "seller")]
        result = check(make_contract([("names", "；".join(name for name, _ in names))], parties=names))
        self.assertEqual([node.name for node in result.nodes], ["北京中标人技术有限公司"])
        self.assertEqual(result.stats["excluded_subject_mentions"][0]["source_name"], names[0][0])

    def test_scoped_business_delegation_is_not_signature_delegation(self):
        result = check(make_contract([("service", "乙方委托第三方配送本合同货物。")]))
        self.assertNotIn("need_authorization", {candidate.code for candidate in result.candidates})

    def test_another_partys_authorization_claim_cannot_hide_actual_signer_gap(self):
        result = check(make_contract([("buyer", "甲方的法定代表人签订本合同。"),
                                      ("seller", "乙方委托项目经理签署本合同。")]))
        candidates = [candidate for candidate in result.candidates if candidate.code == "need_authorization"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].extra["subject_role"], "seller")


class PrivateEvidenceAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.contract = make_contract([
            ("buyer", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("seller", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("service", "乙方应提供本合同签约授权。"),
        ])
        self.request = {"request_id": "auth", "risk_code": "need_authorization", "clause_ids": ["service"]}
        self.gap = {"document_id": "authorization_documents", "title": "签约授权资料",
            "availability": "unavailable_private", "reason_source": "user_statement",
            "fact_types": ["authorization"], "subject_roles": ["seller"],
            "blocked_fact_scope": "不能确认签署人对本合同的实际代理权限"}
        self.context = {"missing_documents": [self.gap]}

    def row(self, **kwargs):
        return check(self.contract, context=kwargs.pop("context", self.context),
                     requests=[kwargs.pop("request", self.request)], **kwargs).stats["verification_results"][0]

    def test_private_missing_evidence_stays_pending_and_waits_for_new_evidence(self):
        before = copy.deepcopy(self.context)
        row = self.row()
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["fact_status"], "unknown")
        self.assertFalse(row["auto_retry"])
        self.assertEqual(row["retry_trigger"], "new_evidence")
        self.assertEqual(row["reason_source"], "user_statement")
        self.assertIn("unavailable_private", row["reasons"])
        self.assertFalse(row["independently_verified"])
        self.assertEqual(self.context, before)

    def test_document_stats_context_is_supported(self):
        self.contract.document.stats["contract_package"] = self.context
        row = self.row(context=None)
        self.assertFalse(row["auto_retry"])

    def test_unbound_file_or_another_subject_or_fact_type_does_not_block_request(self):
        for changes in ({"subject_roles": []}, {"subject_roles": ["buyer"]}, {"fact_types": ["license"]},
                        {"subject_names": [NAME_A]}, {"clause_ids": ["buyer"]},
                        {"availability": "unverified_public"}):
            with self.subTest(changes=changes):
                row = self.row(context={"missing_documents": [{**self.gap, **changes}]})
                self.assertEqual(row["status"], "pending")
                self.assertNotIn("auto_retry", row)
                self.assertNotIn("unavailable_private", row["reasons"])

    def test_private_declaration_cannot_override_available_bound_valid_evidence(self):
        record = fact(NAME_B, ID_B, fact_type="authorization", contract_id=self.contract.contract_id,
                      applicability=applicability())
        row = self.row(records=[record])
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["fact_status"], "valid")
        self.assertNotIn("auto_retry", row)
        self.assertFalse(row["independently_verified"])

    def test_partial_clause_availability_does_not_suspend_a_broader_request(self):
        row = self.row(context={"missing_documents": [{**self.gap, "clause_ids": ["service"]}]},
                       request={**self.request, "clause_ids": ["service", "seller"]})
        self.assertNotIn("auto_retry", row)
        self.assertEqual(row["status"], "pending")

    def test_readable_projection_keeps_gap_and_fact_pending_without_mutation(self):
        result = check(self.contract, context=self.context, requests=[self.request])
        payload = {"contract_id": self.contract.contract_id, "subject_checks": {"stats": result.stats}}
        before = copy.deepcopy(payload)
        explanation = build_review_explanations(payload)
        private = [row for row in explanation["items"] if "unavailable_private" in row["manual_reasons"]]
        self.assertEqual(len(private), 2)
        self.assertEqual({row["kind"] for row in private}, {"subject_fact", "document_evidence_gap"})
        for row in private:
            self.assertTrue(row["manual_required"])
            self.assertEqual(row["fact_status"], "unknown")
            self.assertFalse(row["auto_retry"])
            self.assertIn("保留用户", row["next_action"])
            self.assertNotIn("补充对应主体的登记", row["next_action"])
            self.assertFalse(row["legal_truth_established"])
        self.assertEqual(payload, before)


if __name__ == "__main__":
    unittest.main()
