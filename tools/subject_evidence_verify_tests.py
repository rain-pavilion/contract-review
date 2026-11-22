"""Offline adversarial protocol fixtures, not genuine corporate evidence or gold labels."""
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import subject_evidence_verify as V


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = V.SOURCES["csrc_cms_2018"]
        self.text = (f"合成协议测试，不是真实证据。{self.cfg['subject_name']} "
                     f"统一社会信用代码{self.cfg['subject_id']}。"
                     "核准你公司设立16家分支机构；自批复下发之日起6个月内。2018年6月1日。")
        self.raw = ("<html><meta charset='utf-8'><body><p>" + self.text + "</p></body></html>").encode()
        self.manifest = {"schema": "fixed-official-subject-evidence-v2", "source_key": "csrc_cms_2018",
            "source_url": self.cfg["url"], "publisher": self.cfg["publisher"],
            "source_role": self.cfg["source_role"],
            "subject_name": self.cfg["subject_name"], "subject_id": self.cfg["subject_id"],
            "document_date": self.cfg["document_date"], "fact_type": self.cfg["fact_type"],
            "retrieved_at_utc": "2026-11-14T08:00:00+00:00", "raw_file": "source.html",
            "visible_text_file": "source.txt", "raw_sha256": V.sha256(self.raw), "raw_bytes": len(self.raw),
            "visible_text_sha256": V.sha256(V.extract_text(self.raw).encode()),
            "decoding": V.extract_text_with_metadata(self.raw)[1],
            "source_date_metadata": V.source_date_metadata(self.cfg, V.extract_text(self.raw)),
            "identity_quote_whitespace_normalized": V.compact(self.text),
            "source_claim_scope": self.cfg["claim"]}
        self.bundle = self.root / "capture.json"
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        (self.root / "source.html").write_bytes(self.raw)
        (self.root / "source.txt").write_bytes(V.extract_text(self.raw).encode("utf-8"))
        V.write_json(self.bundle, self.manifest)

    def test_self_consistent_forged_fixture_does_not_reauthenticate_or_confirm_facts(self):
        result = V.verify(self.bundle, today=date(2026, 11, 15))
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["source_identity_and_quote_bound"])
        self.assertEqual(result["source_authenticity"], "offline_manifest_claim_not_reauthenticated")
        self.assertFalse(result["network_reauthentication_performed"])
        self.assertFalse(result["human_review_completed"])
        self.assertEqual(result["current_fact_status"], "unknown")

    def test_actual_product_import_readback_and_identity_isolation_stay_unknown(self):
        result = V.exercise_import(self.bundle, self.root / "isolated", today=date(2026, 11, 15))
        self.assertTrue(result["product_import_exercised"])
        self.assertTrue((self.root / "isolated/subject_facts.isolated.json").is_file())
        self.assertTrue(all(v == "unknown" for v in result["correct_identity_resolution"]["checks"].values()))
        self.assertEqual(result["wrong_identity_resolution"]["records"][0]["identity_review_reason"], "subject_identifier_mismatch")
        self.assertFalse(result["correct_identity_resolution"]["independently_verified"])

    def test_raw_tampering_rejected(self):
        with (self.root / "source.html").open("ab") as stream:
            stream.write(b"tampered")
        self.assertIn("raw_file_integrity_mismatch", V.verify(self.bundle)["errors"])

    def test_text_file_tampering_rejected(self):
        (self.root / "source.txt").write_text("changed", encoding="utf-8")
        self.assertIn("visible_text_integrity_mismatch", V.verify(self.bundle)["errors"])

    def test_arbitrary_url_cannot_be_accepted_even_if_local_hashes_match(self):
        self.manifest["source_url"] = "http://127.0.0.1/private"
        self.save()
        self.assertIn("manifest_binding_mismatch:source_url", V.verify(self.bundle)["errors"])
        with self.assertRaises(ValueError):
            V.source_config("https://169.254.169.254/")

    def test_subject_manifest_mismatch_rejected(self):
        self.manifest["subject_id"] = "9144030010001686XA"
        self.save()
        self.assertIn("manifest_binding_mismatch:subject_id", V.verify(self.bundle)["errors"])

    def test_quote_not_present_or_from_different_subject_rejected(self):
        self.manifest["identity_quote_whitespace_normalized"] = "其他主体9144030010001686XA"
        self.save()
        self.assertIn("source_identity_quote_not_bound", V.verify(self.bundle)["errors"])

    def test_hidden_script_is_not_evidence(self):
        self.raw = ("<html><script>" + self.text + "</script><p>nothing</p></html>").encode()
        self.manifest.update(raw_sha256=V.sha256(self.raw), raw_bytes=len(self.raw),
            visible_text_sha256=V.sha256(V.extract_text(self.raw).encode()))
        self.save()
        self.assertIn("source_identity_quote_not_bound", V.verify(self.bundle)["errors"])

    def test_path_traversal_rejected(self):
        self.manifest["raw_file"] = "../outside.html"
        self.save()
        with self.assertRaises(ValueError):
            V.verify(self.bundle)

    def test_future_capture_rejected(self):
        self.assertIn("capture_or_document_date_invalid", V.verify(self.bundle, today=date(2026, 11, 13))["errors"])

    def test_uscc_check_digit(self):
        self.assertTrue(V.valid_uscc(self.cfg["subject_id"]))
        self.assertTrue(V.valid_uscc(V.SOURCES["szcredit_cmb_public_report"]["subject_id"]))
        self.assertFalse(V.valid_uscc(self.cfg["subject_id"][:-1] + "0"))

    def test_misdeclared_utf8_strict_gb18030_records_conflict_without_ignoring(self):
        raw = ('<meta charset="utf-8"><title>企业信用报告</title>').encode("gbk")
        text, meta = V.extract_text_with_metadata(raw)
        self.assertEqual(text, "企业信用报告")
        self.assertEqual(meta["declared_encoding"], "utf-8")
        self.assertEqual(meta["actual_decoding"], "gb18030")
        self.assertTrue(meta["declared_encoding_conflict"])
        self.assertFalse(meta["replacement_or_ignored_bytes"])
        self.assertEqual(meta["strict_decode_failures"][0]["encoding"], "utf-8-sig")

    def test_visible_text_write_is_lf_exact_on_windows(self):
        self.raw = self.raw.replace(b"</p>", b"</p><p>second line</p>")
        text, decoding = V.extract_text_with_metadata(self.raw)
        self.manifest.update(raw_sha256=V.sha256(self.raw), raw_bytes=len(self.raw),
            visible_text_sha256=V.sha256(text.encode()), decoding=decoding)
        self.save()
        self.assertIn(b"\n", (self.root / "source.txt").read_bytes())
        self.assertNotIn(b"\r", (self.root / "source.txt").read_bytes())
        self.assertEqual(V.verify(self.bundle)["errors"], [])

    def test_old_unsupported_claim_is_not_silently_reinterpreted(self):
        self.manifest["source_claim_scope"] = "获准设立21家分支机构"
        self.save()
        self.assertIn("manifest_binding_mismatch:source_claim_scope", V.verify(self.bundle)["errors"])

    def test_dynamic_report_date_is_extracted_from_actual_field(self):
        cfg = V.SOURCES["szcredit_cmb_public_report"]
        result = V.source_date_metadata(cfg, "报告生产时间2026/11/15 13:02:19\nother text")
        self.assertTrue(result["supported"])
        self.assertEqual(result["document_date"], "2026-11-15")
        self.assertEqual(result["reported_time_text"], "2026/11/15 13:02:19")
        self.assertFalse(V.source_date_metadata(cfg, "2026/11/14 unrelated date")["supported"])

    def test_manifest_cannot_claim_another_source_date(self):
        self.manifest["document_date"] = "2026-11-15"
        self.save()
        self.assertIn("manifest_binding_mismatch:source_date", V.verify(self.bundle)["errors"])

    def test_expired_and_stale_attested_fixture_does_not_resolve_positive(self):
        # Separate synthetic attestation fixture checks actual product safeguards.
        today = date(2026, 11, 15)
        source_day = min(date.today(), today - timedelta(days=31))
        item = V.F.validate_fact({"subject_name": self.cfg["subject_name"],
            "subject_id": self.cfg["subject_id"], "fact_type": "license", "status": "valid",
            "valid_until": "2026-11-13", "scope": "synthetic guard fixture only",
            "source": {"url": self.cfg["url"], "publisher": "synthetic fixture",
                "reviewer": "synthetic-test-reviewer", "verification_method": "not genuine evidence",
                "verified_at": source_day.isoformat()}})
        result = V.F.resolve_subject(self.cfg["subject_name"], [item],
                                     subject_ids=[self.cfg["subject_id"]], today=today)
        self.assertEqual(result["checks"]["license"], "unknown")
        self.assertIn("expired", result["records"][0]["review_reasons"])
        self.assertIn("source_stale", result["records"][0]["review_reasons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
