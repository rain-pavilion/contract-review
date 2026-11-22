"""Offline identity and statute-request fixtures, not independent gold labels."""
from __future__ import annotations

from datetime import date, timedelta
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import subject_facts as F, subject_graph as S
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, Element


NAME_A = "甲辰科技有限公司"
NAME_B = "乙星技术有限公司"
ID_A = "91310101TEST000001"
ID_B = "91310101TEST000002"
ID_C = "91310101TEST000003"


def make_contract(texts, parties=None):
    clauses = [Clause(cid, "", "", 1, "", text, text, ["b" + cid]) for cid, text in texts]
    result = Contract("identity-fixture", DocumentParse("identity.txt", "txt",
        "\n".join(clause.text for clause in clauses), clauses=clauses))
    for index, (name, role) in enumerate(parties or [(NAME_A, "buyer"), (NAME_B, "seller")]):
        result.elements.parties.append(Element(f"party{index}", "party", name, name,
            role=role, confidence=1.0, source="offline_fixture"))
    return result


def fact(name=NAME_A, identifier=ID_A, **changes):
    item = {"subject_name": name, "subject_id": identifier,
        "fact_type": "license", "status": "valid", "scope": "合成身份绑定测试",
        "valid_until": (date.today() + timedelta(days=90)).isoformat(),
        "source": {"url": "https://source.invalid/fixture", "publisher": "合成测试来源",
            "reviewer": "fixture-reviewer", "verification_method": "仅测试身份绑定协议，未真实核验",
            "verified_at": date.today().isoformat()}}
    item.update(changes)
    return F.validate_fact(item)


def applicability(**changes):
    """Synthetic human-claim fixture; never a real applicability judgement."""
    item = {"actor_kind": "human", "contract_id": "identity-fixture", "clause_ids": ["service"],
        "verdict": "applicable", "reviewer": "fixture-not-a-person",
        "reviewed_at": date.today().isoformat(), "basis": "离线协议测试中明确指定本条款适用，不是真实核验"}
    item.update(changes)
    return item


def check(contract, records=(), requests=()):
    with patch.object(F, "load_facts", return_value=(list(records), [])):
        return S.check_subjects(contract, Config(), verification_requests=requests)


def facts_by_role(result):
    return {row["subject_role"]: row for row in result.stats["fact_checks"]}


class SubjectIdentityBindingTests(unittest.TestCase):
    def setUp(self):
        self.contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("service", "乙方提供技术服务，并保证资质许可证处于有效期内。"),
        ])

    def test_identifier_from_another_subject_cannot_verify_same_named_evidence(self):
        result = check(self.contract, [fact(identifier=ID_B)])
        rows = facts_by_role(result)
        self.assertEqual(rows["buyer"]["checks"]["license"], "unknown")
        self.assertEqual(rows["buyer"]["records"][0]["identity_review_reason"], "subject_identifier_mismatch")
        self.assertEqual(rows["buyer"]["subject_identifiers"], [ID_A])
        self.assertEqual(rows["seller"]["records"], [])

    def test_each_explicit_identity_receives_its_own_evidence(self):
        result = check(self.contract, [fact(), fact(NAME_B, ID_B)])
        rows = facts_by_role(result)
        for role, identifier in (("buyer", ID_A), ("seller", ID_B)):
            with self.subTest(role=role):
                self.assertEqual(rows[role]["checks"]["license"], "valid")
                self.assertEqual(rows[role]["identity_binding"]["status"], "bound")
                self.assertEqual(rows[role]["subject_identifiers"], [identifier])
                self.assertEqual(rows[role]["records"][0]["identity_match"], "name_and_identifier")
                self.assertFalse(rows[role]["independently_verified"])

    def test_evidence_without_identifier_stays_unknown(self):
        row = facts_by_role(check(self.contract, [fact(identifier="")]))["buyer"]
        self.assertEqual(row["checks"]["license"], "unknown")
        self.assertEqual(row["records"][0]["identity_review_reason"], "evidence_identifier_not_supplied")

    def test_contract_without_bound_identifier_does_not_borrow_arbitrary_text(self):
        contract = make_contract([("a", f"甲方：{NAME_A}。"),
            ("b", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("appendix", f"资料编号：{ID_A}，仅为附件示例。")])
        row = facts_by_role(check(contract, [fact()]))["buyer"]
        self.assertEqual(row["checks"]["license"], "unknown")
        self.assertEqual(row["identity_binding"]["status"], "unbound")
        self.assertEqual(row["records"][0]["identity_review_reason"], "subject_identifier_not_bound")

    def test_full_text_compatibility_argument_cannot_supply_identity(self):
        row = F.resolve_subject(NAME_A, [fact()], full_text=f"{NAME_A} {ID_A}")
        self.assertEqual(row["checks"]["license"], "unknown")
        self.assertEqual(row["identity_binding_status"], "unbound")
        bound = F.resolve_subject(NAME_A, [fact()], subject_ids=[ID_A])
        self.assertEqual(bound["checks"]["license"], "valid")

    def test_same_name_role_fields_are_distinct_and_unselected_request_is_ambiguous(self):
        contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_A}，统一社会信用代码：{ID_B}。"),
            ("service", f"{NAME_A}的资质许可证应持续有效。"),
        ], [(NAME_A, "buyer"), (NAME_A, "seller")])
        request = {"candidate_id": "c1", "clause_ids": ["service"], "risk_code": "need_license"}
        result = check(contract, [fact(), fact(identifier=ID_B)], [request])
        rows = facts_by_role(result)
        self.assertEqual(rows["buyer"]["identity_binding"]["identifiers"], [ID_A])
        self.assertEqual(rows["seller"]["identity_binding"]["identifiers"], [ID_B])
        self.assertEqual(rows["buyer"]["checks"]["license"], "valid")
        self.assertEqual(rows["seller"]["checks"]["license"], "valid")
        self.assertEqual(result.stats["verification_results"][0]["reason"], "request_subject_ambiguous")
        selected = check(contract, [fact(), fact(identifier=ID_B, applicability=applicability())],
            [{**request, "subject_id": ID_B}])
        response = selected.stats["verification_results"][0]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["subject_id"], ID_B)

    def test_same_name_without_role_binding_is_ambiguous(self):
        contract = make_contract([("a", f"{NAME_A}，统一社会信用代码：{ID_A}。")],
            [(NAME_A, "buyer"), (NAME_A, "seller")])
        result = check(contract, [fact()])
        self.assertTrue(all(row["identity_binding"]["status"] == "ambiguous"
            for row in result.stats["fact_checks"]))
        self.assertTrue(all(row["checks"]["license"] == "unknown" for row in result.stats["fact_checks"]))

    def test_multiple_explicit_identifiers_for_one_subject_are_ambiguous(self):
        contract = make_contract([("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"甲方统一社会信用代码：{ID_C}。")], [(NAME_A, "buyer")])
        row = check(contract, [fact()]).stats["fact_checks"][0]
        self.assertEqual(row["identity_binding"]["status"], "ambiguous")
        self.assertEqual(row["checks"]["license"], "unknown")

    def test_alternative_identifiers_in_a_single_field_are_ambiguous(self):
        contract = make_contract([("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}或{ID_C}。")], [(NAME_A, "buyer")])
        row = check(contract, [fact()]).stats["fact_checks"][0]
        self.assertEqual(row["identity_binding"]["status"], "ambiguous")
        self.assertEqual(row["checks"]["license"], "unknown")

    def test_different_names_cannot_share_an_identity(self):
        contract = make_contract([("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_B}，统一社会信用代码：{ID_A}。")])
        result = check(contract, [fact(), fact(NAME_B, ID_A)])
        for row in result.stats["fact_checks"]:
            self.assertEqual(row["identity_binding"]["reason"], "identifier_shared_by_different_subjects")
            self.assertEqual(row["checks"]["license"], "unknown")

    def test_generic_identifier_on_next_line_is_not_a_name_binding(self):
        contract = make_contract([("a", f"甲方：{NAME_A}\n统一社会信用代码：{ID_A}。")], [(NAME_A, "buyer")])
        row = check(contract, [fact()]).stats["fact_checks"][0]
        self.assertEqual(row["identity_binding"]["status"], "unbound")
        self.assertEqual(row["checks"]["license"], "unknown")

    def test_role_named_identifier_can_bind_a_unique_role(self):
        contract = make_contract([("a", f"甲方：{NAME_A}；乙方：{NAME_B}。"),
            ("b", f"乙方统一社会信用代码：{ID_B}。")])
        rows = facts_by_role(check(contract, [fact(NAME_B, ID_B)]))
        self.assertEqual(rows["seller"]["checks"]["license"], "valid")
        self.assertEqual(rows["seller"]["identity_binding"]["evidence"][0]["binding"], "role_identifier_field")
        self.assertEqual(rows["buyer"]["identity_binding"]["status"], "unbound")

    def test_fourth_party_label_matches_the_existing_third_role_mapping(self):
        contract = make_contract([("a", f"丁方：{NAME_A}。"),
            ("b", f"丁方统一社会信用代码：{ID_A}。")], [(NAME_A, "third")])
        row = check(contract, [fact()]).stats["fact_checks"][0]
        self.assertEqual(row["checks"]["license"], "valid")

    def test_role_identifier_with_several_subjects_is_ambiguous(self):
        contract = make_contract([("a", f"乙方：{NAME_A}、{NAME_B}。"),
            ("b", f"乙方统一社会信用代码：{ID_B}。")], [(NAME_A, "seller"), (NAME_B, "seller")])
        result = check(contract, [fact(), fact(NAME_B, ID_B)])
        self.assertTrue(all(row["identity_binding"]["status"] == "ambiguous" for row in result.stats["fact_checks"]))

    def test_name_with_conflicting_role_does_not_take_other_role_identifier(self):
        contract = make_contract([("a", f"乙方：{NAME_A}，统一社会信用代码：{ID_B}。")])
        result = check(contract, [fact(identifier=ID_B)])
        self.assertTrue(all(row["identity_binding"]["status"] == "unbound" for row in result.stats["fact_checks"]))

    def test_wrong_identity_negative_evidence_does_not_create_reviewed_fact_risk(self):
        result = check(self.contract, [fact(identifier=ID_B, status="invalid")])
        self.assertFalse(any(e.kind == "reviewed_subject_fact" for candidate in result.candidates for e in candidate.evidence))

    def test_expiry_staleness_contract_and_validity_requirements_still_apply(self):
        cases = [({"valid_until": (date.today() - timedelta(days=1)).isoformat()}, "expired"),
            ({"valid_from": (date.today() + timedelta(days=1)).isoformat()}, "not_yet_effective"),
            ({"valid_until": ""}, "validity_end_not_supplied"),
            ({"fact_type": "authorization", "contract_id": "other-contract"}, "different_contract")]
        old_source = dict(fact()["source"], verified_at=(date.today() - timedelta(days=31)).isoformat())
        cases.append(({"source": old_source}, "source_stale"))
        for changes, reason in cases:
            with self.subTest(reason=reason):
                row = F.resolve_subject(NAME_A, [fact(**changes)], subject_ids=[ID_A], contract_id=self.contract.contract_id)
                self.assertEqual(row["records"][0]["effective_status"], "unknown")
                self.assertIn(reason, row["records"][0]["review_reasons"])


class SubjectVerificationRequestTests(unittest.TestCase):
    def setUp(self):
        self.contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("both", f"{NAME_A}与{NAME_B}分别保证其资质许可有效。"),
            ("service", "乙方提供技术服务并承担资质保证责任。"),
            ("unknown", "交付时间按附件约定。"),
        ])
        self.request = {"request_id": "request-license", "candidate_id": "candidate-license",
            "article_uid": "law-fixture-1", "risk_code": "need_license", "clause_ids": ["service"]}
        self.record = fact(NAME_B, ID_B, applicability=applicability())

    def response(self, request=None, records=None):
        result = check(self.contract, [self.record] if records is None else records,
            [self.request if request is None else request])
        return result, result.stats["verification_results"][0]

    def test_explicit_scoped_request_completes_with_bound_source_and_origins(self):
        result, response = self.response()
        self.assertEqual(response["status"], "completed")
        for field in ("request_id", "candidate_id", "article_uid", "risk_code", "clause_ids"):
            self.assertEqual(response[field], self.request[field])
        self.assertEqual(response["subject_name"], NAME_B)
        self.assertEqual(response["subject_id"], ID_B)
        self.assertEqual(response["fact_type"], "license")
        self.assertEqual(response["fact_status"], "valid")
        self.assertEqual(response["fact_ids"], [self.record["fact_id"]])
        self.assertEqual(response["records"][0]["source"], self.record["source"])
        self.assertFalse(response["independently_verified"])
        self.assertTrue(result.stats["verification_complete"])
        self.assertEqual(result.stats["pending_verification_request_ids"], [])

    def test_no_source_unresolved_subject_type_or_scope_remains_pending(self):
        cases = [({}, [], "no_effective_reviewed_evidence"),
            ({"clause_ids": ["both"]}, None, "request_subject_ambiguous"),
            ({"clause_ids": ["unknown"]}, None, "request_subject_not_bound_in_clause_scope"),
            ({"fact_type": "invented"}, None, "request_fact_type_unresolved"),
            ({"risk_code": "unrelated"}, None, "request_fact_type_unresolved"),
            ({"clause_ids": []}, None, "request_clause_scope_missing"),
            ({"clause_ids": ["missing"]}, None, "request_clause_not_found")]
        for changes, records, reason in cases:
            with self.subTest(reason=reason):
                result, response = self.response({**self.request, **changes}, records)
                self.assertEqual(response["status"], "pending")
                self.assertIn(reason, response["reasons"])
                self.assertFalse(result.stats["verification_complete"])
                self.assertEqual(result.stats["pending_verification_request_ids"], [self.request["request_id"]])
                self.assertTrue(any("subject_verification_pending:" in item for item in result.degraded))

    def test_subject_selector_must_also_match_clause_scope_and_actual_identity(self):
        for changes in ({"subject_name": NAME_A}, {"subject_id": ID_A},
                        {"subject_name": NAME_B, "subject_id": ID_A}, {"subject_node_id": f"S:buyer:{NAME_A}"}):
            with self.subTest(changes=changes):
                _, response = self.response({**self.request, **changes})
                self.assertEqual(response["status"], "pending")
                self.assertIn("request_subject_not_bound_in_clause_scope", response["reasons"])

    def test_short_name_inside_another_party_name_is_not_in_scope(self):
        longer_name = "上海" + NAME_A
        contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{longer_name}，统一社会信用代码：{ID_B}。"),
            ("service", f"{longer_name}的资质许可证应持续有效。"),
        ], [(NAME_A, "buyer"), (longer_name, "seller")])
        request = {**self.request, "subject_name": NAME_A}
        result = check(contract, [fact()], [request])
        self.assertEqual(result.stats["verification_results"][0]["status"], "pending")
        self.assertIn("request_subject_not_bound_in_clause_scope", result.stats["verification_results"][0]["reasons"])

    def test_same_name_explicit_role_scope_cannot_select_the_other_role(self):
        contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_A}，统一社会信用代码：{ID_B}。"),
        ], [(NAME_A, "buyer"), (NAME_A, "seller")])
        request = {**self.request, "clause_ids": ["b"], "subject_id": ID_A}
        result = check(contract, [fact(), fact(identifier=ID_B)], [request])
        self.assertEqual(result.stats["verification_results"][0]["status"], "pending")
        correct = check(contract, [fact(), fact(identifier=ID_B, applicability=applicability(clause_ids=["b"]))],
            [{**request, "subject_id": ID_B}])
        self.assertEqual(correct.stats["verification_results"][0]["status"], "completed")

    def test_conflicting_role_name_header_cannot_bind_a_scope_selector(self):
        self.contract.document.clauses.append(Clause("conflict", "", "", 1, "",
            f"乙方：{NAME_A}。", f"乙方：{NAME_A}。", ["bconflict"]))
        for name, identifier in ((NAME_A, ID_A), (NAME_B, ID_B)):
            with self.subTest(name=name):
                _, response = self.response({**self.request, "clause_ids": ["conflict"], "subject_name": name,
                    "subject_id": identifier}, [fact(), self.record])
                self.assertEqual(response["status"], "pending")

    def test_no_bound_identity_and_conflicting_sources_stay_pending(self):
        no_id = make_contract([("service", f"乙方：{NAME_B}提供服务。")], [(NAME_B, "seller")])
        result = check(no_id, [self.record], [self.request])
        self.assertEqual(result.stats["verification_results"][0]["status"], "pending")
        _, response = self.response(records=[self.record,
            fact(NAME_B, ID_B, status="invalid", applicability=applicability())])
        self.assertEqual(response["fact_status"], "conflicting_evidence")
        self.assertIn("conflicting_evidence", response["reasons"])

    def test_legacy_request_ids_are_stable_and_keep_every_origin_and_duplicate(self):
        request = {key: value for key, value in self.request.items() if key != "request_id"}
        other = {**request, "candidate_id": "other-candidate", "article_uid": "law-fixture-2"}
        requests = [request, other, dict(request)]
        first = check(self.contract, [self.record], requests).stats["verification_results"]
        second = check(self.contract, [self.record], requests).stats["verification_results"]
        self.assertEqual(len(first), len(requests))
        self.assertEqual([row["request_id"] for row in first], [row["request_id"] for row in second])
        self.assertEqual(len({row["request_id"] for row in first}), 3)
        self.assertEqual([row["candidate_id"] for row in first], ["candidate-license", "other-candidate", "candidate-license"])
        self.assertTrue(all(row["status"] == "completed" for row in first))

    def test_duplicate_explicit_request_ids_remain_present_and_pending(self):
        result = check(self.contract, [self.record], [self.request, dict(self.request)])
        rows = result.stats["verification_results"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertTrue(all("duplicate_verification_request_id" in row["reasons"] for row in rows))
        self.assertFalse(result.stats["verification_complete"])

    def test_registration_and_credit_are_not_inferred_from_generic_risk_code(self):
        for code in ("subject_unspecified", "illegal_terms"):
            with self.subTest(code=code):
                _, response = self.response({**self.request, "risk_code": code})
                self.assertIn("request_fact_type_unresolved", response["reasons"])
        request = {**self.request, "risk_code": "subject_unspecified", "clause_ids": ["b"]}
        record = fact(NAME_B, ID_B, fact_type="registration", status="active")
        _, response = self.response(request, [record])
        self.assertEqual(response["fact_type"], "registration")
        self.assertEqual(response["status"], "completed")

    def test_invalid_input_request_is_preserved_as_pending(self):
        result = check(self.contract, [self.record], [None, "invalid"])
        rows = result.stats["verification_results"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertTrue(all("invalid_verification_request" in row["reasons"] for row in rows))


if __name__ == "__main__":
    unittest.main()
