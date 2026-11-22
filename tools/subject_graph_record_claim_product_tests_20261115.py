"""Direct product tests after actor and graph-claim patches; no proposal loading.

Source/fact fixtures are synthetic. No model, network, external registry or real
business fact is used. Run normally only after applying the product patches, or
through the isolated integration candidate bootstrap before product application.
"""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import subject_facts as F, subject_graph as S
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import stable_id
from zhilv_qiling_v2.tests_v2.test_subject_identity_binding import make_contract, NAME_A, ID_A, ID_B


def fact(actor=None, *, status="invalid", identifier=ID_A, expired=False):
    source = {"url": "https://example.invalid/synthetic", "publisher": "Synthetic offline fixture",
        "reviewer": "fixture supplier", "verification_method": "Protocol fixture, no actual attestation",
        "verified_at": date.today().isoformat()}
    if actor is not None: source["actor_kind"] = actor
    return F.validate_fact({"subject_name": NAME_A, "subject_id": identifier,
        "fact_type": "license", "status": status, "scope": "Synthetic test scope",
        "valid_until": (date.today() + timedelta(days=-1 if expired else 30)).isoformat(), "source": source})


def check(rows):
    contract = make_contract([("a", f"甲方：{NAME_A}，统一社会信用代码：{ID_A}。"),
        ("service", "甲方提供技术服务，并保证资质许可证持续有效。")], [(NAME_A, "buyer")])
    with patch.object(F, "load_facts", return_value=(deepcopy(rows), [])):
        return S.check_subjects(contract, Config())


def fact_candidates(result):
    return [candidate for candidate in result.candidates if any(e.kind == "reviewed_subject_fact" for e in candidate.evidence)]


class DirectRecordClaimProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = {Path(module.__file__): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (F, S)}

    @classmethod
    def tearDownClass(cls):
        for path, expected in cls.before.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise AssertionError("product source changed during direct graph-claim tests")

    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("network is forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def test_legacy_negative_record_claim_preserves_business_status_ids_and_scores(self):
        row = fact(); result = check([row]); candidate = fact_candidates(result)[0]
        self.assertEqual(candidate.extra["verification_status"], "supplied_record_claim")
        self.assertEqual(candidate.extra["source_actor_kind"], "legacy_unspecified")
        self.assertFalse(candidate.extra["independently_verified"])
        checked = result.stats["fact_checks"][0]
        self.assertEqual(checked["checks"]["license"], "invalid")
        key = stable_id("reviewed_subject_fact", checked["subject_node_id"], ID_A,
                        row["fact_id"], "license", "a", length=32)
        self.assertEqual(candidate.extra["fact_key"], key)
        self.assertEqual(candidate.candidate_id, stable_id("rc_subject_fact", key, length=24))
        self.assertEqual(candidate.code, "need_license")
        self.assertEqual(candidate.confidence, .8)
        evidence = next(item for item in candidate.evidence if item.kind == "reviewed_subject_fact")
        self.assertEqual(evidence.strength, .85)
        self.assertEqual(evidence.extra["effective_status"], "invalid")

    def test_explicit_human_negative_is_still_a_supplied_claim(self):
        candidate = fact_candidates(check([fact("human")]))[0]
        self.assertEqual(candidate.extra["source_actor_kind"], "human")
        self.assertEqual(candidate.extra["verification_status"], "supplied_record_claim")
        self.assertFalse(candidate.extra["independently_verified"])

    def test_adapter_unknown_does_not_create_negative_fact_candidate(self):
        result = check([fact("adapter", status="unknown")])
        self.assertEqual(fact_candidates(result), [])
        self.assertEqual(result.stats["fact_checks"][0]["checks"]["license"], "unknown")

    def test_wrong_identity_still_unknown_and_no_negative_fact_candidate(self):
        result = check([fact(identifier=ID_B)])
        self.assertEqual(fact_candidates(result), [])
        self.assertEqual(result.stats["fact_checks"][0]["checks"]["license"], "unknown")

    def test_expired_fact_still_unknown_and_no_negative_fact_candidate(self):
        result = check([fact(expired=True)])
        self.assertEqual(fact_candidates(result), [])
        self.assertEqual(result.stats["fact_checks"][0]["checks"]["license"], "unknown")

    def test_canonical_source_fact_and_digest_are_not_mutated(self):
        row = fact("human"); original = deepcopy(row)
        check([row])
        self.assertEqual(row, original)
        self.assertEqual(F.validate_fact(row)["record_sha256"], original["record_sha256"])
        self.assertFalse(row["independently_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
