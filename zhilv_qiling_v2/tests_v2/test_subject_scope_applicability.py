"""Scope binding regressions using offline fixtures, not legal-quality evidence."""
from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from zhilv_qiling_v2 import api, subject_facts as F
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.tests_v2.test_subject_identity_binding import (
    NAME_A, NAME_B, ID_A, ID_B, applicability, check, fact, make_contract,
)


class SubjectScopeApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.contract = make_contract([
            ("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
            ("b", f"乙方：{NAME_B}，统一社会信用代码：{ID_B}。"),
            ("service", "乙方承担建筑施工，应有建筑业企业资质证书和本合同签约授权。"),
        ])
        self.request = {"request_id": "construction", "candidate_id": "construction-license",
            "risk_code": "need_license", "clause_ids": ["service"]}

    def response(self, records, request=None):
        result = check(self.contract, records, [request or self.request])
        return result, result.stats["verification_results"][0]

    def test_free_text_scope_cannot_complete_a_different_or_same_named_requirement(self):
        for scope in ("仅限食品经营许可，不包含建筑施工资质", "建筑业企业资质证书"):
            with self.subTest(scope=scope):
                record = fact(NAME_B, ID_B, scope=scope)
                result, row = self.response([record])
                self.assertEqual(row["status"], "pending")
                self.assertEqual(row["fact_status"], "unknown")
                self.assertEqual(row["fact_ids"], [])
                self.assertIn("evidence_scope_applicability_missing", row["reasons"])
                self.assertIn("construction", result.stats["pending_verification_request_ids"])
                # Validity of the supplied license record is retained separately.
                self.assertEqual(row["records"][0]["effective_status"], "valid")
                self.assertNotIn("applicability", row["records"][0])

    def test_explicit_human_scope_judgement_can_complete_only_bound_license_or_authorization(self):
        for kind, risk_code in (("license", "need_license"), ("authorization", "need_authorization")):
            with self.subTest(kind=kind):
                record = fact(NAME_B, ID_B, fact_type=kind, contract_id=self.contract.contract_id,
                    applicability=applicability())
                _, row = self.response([record], {**self.request, "risk_code": risk_code})
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["fact_status"], "valid")
                self.assertEqual(row["fact_ids"], [record["fact_id"]])
                self.assertEqual(row["scope_status"], "human_attested_applicable")
                self.assertFalse(row["scope_independently_verified"])
                self.assertFalse(row["independently_verified"])
                self.assertFalse(row["records"][0]["applicability"]["independently_verified"])

    def test_negative_unknown_and_mismatched_scope_never_complete_license_or_authorization(self):
        cases = [({"verdict": "not_applicable"}, "evidence_scope_not_applicable"),
            ({"verdict": "needs_manual"}, "evidence_scope_needs_manual"),
            ({"contract_id": "other-contract"}, "evidence_scope_different_contract"),
            ({"clause_ids": ["b"]}, "evidence_scope_clause_not_bound")]
        for kind, risk_code in (("license", "need_license"), ("authorization", "need_authorization")):
            for changes, reason in cases:
                with self.subTest(kind=kind, reason=reason):
                    record = fact(NAME_B, ID_B, fact_type=kind, contract_id=self.contract.contract_id,
                        applicability=applicability(**changes))
                    _, row = self.response([record], {**self.request, "risk_code": risk_code})
                    self.assertEqual(row["status"], "pending")
                    self.assertEqual(row["fact_status"], "unknown")
                    self.assertEqual(row["fact_ids"], [])
                    self.assertIn(reason, row["reasons"])

    def test_every_requested_clause_must_be_bound(self):
        record = fact(NAME_B, ID_B, applicability=applicability())
        _, row = self.response([record], {**self.request, "clause_ids": ["service", "b"]})
        self.assertEqual(row["status"], "pending")
        self.assertIn("evidence_scope_clause_not_bound", row["reasons"])

    def test_other_license_status_does_not_replace_the_applicable_record(self):
        construction = fact(NAME_B, ID_B, applicability=applicability())
        food = fact(NAME_B, ID_B, status="invalid", scope="食品经营许可",
            applicability=applicability(clause_ids=["b"]))
        _, row = self.response([construction, food])
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["fact_status"], "valid")
        self.assertEqual(row["fact_ids"], [construction["fact_id"]])
        self.assertIn("evidence_scope_clause_not_bound", row["scope_review_reasons"])
        _, conflict = self.response([construction,
            fact(NAME_B, ID_B, status="invalid", applicability=applicability())])
        self.assertEqual(conflict["status"], "pending")
        self.assertEqual(conflict["fact_status"], "conflicting_evidence")

    def test_scope_judgement_cannot_override_identity_expiry_or_unknown_source_status(self):
        cases = [{"identifier": ID_A},
            {"valid_until": (date.today() - timedelta(days=1)).isoformat()},
            {"status": "unknown"}]
        for changes in cases:
            with self.subTest(changes=changes):
                record = fact(NAME_B, changes.pop("identifier", ID_B),
                    applicability=applicability(), **changes)
                _, row = self.response([record])
                self.assertEqual(row["status"], "pending")
                self.assertEqual(row["fact_status"], "unknown")
                self.assertEqual(row["fact_ids"], [])


class ScopeImportCompatibilityTests(unittest.TestCase):
    def test_legacy_hash_and_new_scope_hash_survive_roundtrip_and_detect_tampering(self):
        legacy = fact(NAME_B, ID_B)
        self.assertEqual(F.validate_fact(legacy)["record_sha256"], legacy["record_sha256"])
        self.assertEqual(legacy["record_hash_schema"], "canonical-v1-legacy")
        scoped = fact(NAME_B, ID_B, applicability=applicability())
        self.assertNotEqual(scoped["fact_id"], legacy["fact_id"])
        self.assertEqual(scoped["record_hash_schema"], "canonical-v3-scope-applicability")
        self.assertEqual(F.validate_fact(scoped)["record_sha256"], scoped["record_sha256"])
        changed = copy.deepcopy(scoped)
        changed["applicability"]["clause_ids"] = ["other-clause"]
        with self.assertRaisesRegex(ValueError, "校验和"):
            F.validate_fact(changed)

    def test_incomplete_or_automatic_applicability_is_rejected(self):
        cases = [None, {}, applicability(actor_kind="adapter"), applicability(reviewer=""),
            applicability(basis=""), applicability(clause_ids=[]), applicability(clause_ids=[{}]),
            applicability(verdict=[]),
            applicability(reviewed_at=(date.today() + timedelta(days=1)).isoformat())]
        for review in cases:
            with self.subTest(review=review), self.assertRaises(ValueError):
                fact(NAME_B, ID_B, applicability=review)

    def test_web_json_import_persists_scope_and_keeps_legacy_readable(self):
        with TemporaryDirectory() as folder:
            config = Config(subject_facts_path=str(Path(folder) / "facts.json"))
            legacy, scoped = fact(NAME_B, ID_B), fact(NAME_B, ID_B, applicability=applicability())
            post = SimpleNamespace(config=config, method="POST", json_body={"records": [legacy, scoped]})
            imported = api.handle_subject_facts(post)
            self.assertEqual(imported["total"], 2)
            self.assertFalse(imported["independently_verified"])
            loaded = api.handle_subject_facts(SimpleNamespace(config=config, method="GET"))
            self.assertEqual(loaded["degraded"], [])
            by_id = {record["fact_id"]: record for record in loaded["records"]}
            self.assertNotIn("applicability", by_id[legacy["fact_id"]])
            self.assertEqual(by_id[scoped["fact_id"]]["applicability"], scoped["applicability"])
            self.assertFalse(by_id[scoped["fact_id"]]["independently_verified"])


if __name__ == "__main__":
    unittest.main()
