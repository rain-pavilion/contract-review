"""Engineering regressions; fixtures are not independent legal gold."""
from datetime import date, timedelta
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents, ner
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import (RiskCandidate, Evidence, Contract, DocumentParse,
    SourceBlock, Clause, Element)
from zhilv_qiling_v2.ner_constraints import constrain_hits
from zhilv_qiling_v2.subject_materials import SubjectMaterialStore, decode_content, check_material_fields
from zhilv_qiling_v2.subject_facts import load_facts, validate_fact


def document(text):
    return DocumentParse("fixture.txt", "txt", text, blocks=[SourceBlock("b", 0, "paragraph", text)],
        clauses=[Clause("c", "", "", 1, "", text, text, ["b"])])


class ManualFusionTests(unittest.TestCase):
    def test_merge_cannot_erase_pending_or_evidence_with_same_text(self):
        a = RiskCandidate("a", "illegal_terms", "fixture", extra={"manual_required": False,
            "claim_grounding_status": "supported", "review_flags": ["old"]})
        b = RiskCandidate("b", "illegal_terms", "fixture", extra={"manual_required": True,
            "semantic_pending": True, "claim_grounding_status": "unsupported", "review_flags": ["new"]})
        a.evidence = [Evidence("claim_grounding", "", extra={"claim_grounding_status": "supported"})]
        b.evidence = [Evidence("claim_grounding", "", extra={"claim_grounding_status": "unsupported"})]
        a.merge_from(b)
        self.assertTrue(a.extra["manual_required"])
        self.assertEqual(a.extra["claim_grounding_status"], "unsupported")
        self.assertEqual(a.extra["review_flags"], ["old", "new"])
        self.assertEqual(len(a.evidence), 2)

    def test_high_confidence_with_citation_still_requires_review(self):
        from zhilv_qiling_v2.agents import AgentResult
        from zhilv_qiling_v2.models import Citation
        c = RiskCandidate("a", "illegal_terms", "fixture", clause_ids=["c"], confidence=.99,
                          extra={"manual_required": True})
        result = AgentResult(agent="fixture", candidates=[c], citations={"a": [Citation("fixture", "1", "fixture")]})
        risks = agents.fuse([c], [result], Contract("fixture", document("fixture")), Config())
        self.assertTrue(risks[0].review_required)
        self.assertTrue(risks[0].citations)


class NerConstraintTests(unittest.TestCase):
    def test_equal_values_different_roles_and_occurrences_survive(self):
        doc = document("合同总价100元，定金100元，尾款100元。")
        hits = ner.extract_amounts(doc)
        hits, rejected = constrain_hits(doc, hits)
        result = ner.fuse_elements(hits, [])
        self.assertFalse(rejected)
        self.assertEqual(len(result.amounts), 3)
        self.assertEqual({x.role for x in result.amounts}, {"total", "deposit", "balance"})
        self.assertEqual(len({x.element_id for x in result.amounts}), 3)

    def test_bank_account_is_not_model_amount(self):
        doc = document("银行账号：1234567890123456。")
        hit = Element("a", "amount", "1234567890123456", "1234567890123456.00", source="model",
                      span=doc.locate("b", 5, 21))
        accepted, rejected = constrain_hits(doc, [hit])
        self.assertFalse(accepted)
        self.assertIn("numeric_identifier_or_missing_currency_context", rejected[0]["reasons"])

    def test_total_and_stages_are_not_unscoped_sum(self):
        doc = document("合同总价100元，预付款80元，定金30元，其中定金抵扣预付款。")
        elements = ner.fuse_elements(ner.extract_amounts(doc), [])
        self.assertNotIn("fee_mismatch", [signal.kind for signal in ner.element_signals(elements, doc)])

    def test_repeated_buyer_is_not_two_sided_identity(self):
        doc = document("甲方：甲辰科技有限公司。签字盖章甲方：甲辰科技有限公司。")
        elements = ner.fuse_elements(ner.extract_parties(doc), [])
        self.assertGreaterEqual(len(elements.parties),2)
        self.assertNotIn("party_duplicated", [signal.kind for signal in ner.element_signals(elements,doc)])
        elements.parties[1].role="seller"
        self.assertIn("party_duplicated", [signal.kind for signal in ner.element_signals(elements,doc)])

    def test_invalid_date_and_ambiguous_legacy_span_rejected(self):
        doc = document("日期2026-02-30，100元，100元。")
        invalid = Element("d", "date", "2026-02-30", "2026-02-30", span=doc.locate("b",2,12))
        ambiguous = Element("a", "amount", "100元", "100.00", span=ner.SpanRef("b", "100元"))
        accepted, rejected = constrain_hits(doc, [invalid, ambiguous])
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 2)

    def test_same_span_conflicting_roles_do_not_increase_confidence(self):
        doc = document("定金100元。")
        rule = ner.extract_amounts(doc)[0]
        model = Element("m", "amount", "100元", "100.00", role="total", source="model", confidence=.91,
                        span=doc.locate("b",2,6))
        result = ner.fuse_elements([rule], [model])
        self.assertEqual(len(result.amounts), 2)
        self.assertTrue(all(x.extra["manual_required"] for x in result.amounts))
        self.assertEqual(sorted(x.confidence for x in result.amounts), [.91, .92])

    def test_model_trim_keeps_real_offsets(self):
        doc = document("采购标的文字足够十二个字符， 金额100元 。")
        class Tagger:
            def decode(self, text):
                start=text.index(" 金额") + 3
                return [(start, text.index(" 。"), "AMOUNT", 4)]
        hits = ner.model_elements(doc, Tagger())
        for hit in hits:
            self.assertEqual(doc.blocks[0].text[hit.span.char_start:hit.span.char_end], hit.value)


class MaterialImportTests(unittest.TestCase):
    def metadata(self):
        return {"document_number": "合成夹具-001", "issuer": "测试机构",
            "fact": {"subject_name": "测试主体有限公司", "subject_id": "91310101TEST000001",
                     "fact_type": "license", "status": "valid", "scope": "合成程序测试，非真实证照",
                     "valid_until": (date.today()+timedelta(days=30)).isoformat(),
                     "source": {"url": "https://fixture.invalid/source", "publisher": "测试机构",
                                "reviewer": "fixture-reviewer", "verification_method": "仅程序测试",
                                "verified_at": date.today().isoformat()}}}

    def test_bytes_bound_to_unknown_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SubjectMaterialStore(Path(folder)/"materials", Path(folder)/"facts.json")
            result = store.import_material("fixture.txt", b"synthetic, not official evidence", self.metadata())
            records, errors = load_facts(store.facts_path)
            self.assertFalse(errors)
            self.assertEqual(records[0]["status"], "unknown")
            self.assertFalse(records[0]["independently_verified"])
            self.assertEqual(store.list_materials()["records"][0]["storage_integrity"], "valid")
            (store.root/(result["material"]["sha256"]+".blob")).write_bytes(b"modified")
            self.assertEqual(store.list_materials()["records"][0]["storage_integrity"], "failed")
            with self.assertRaises(ValueError):
                store.import_material("fixture.txt", b"synthetic, not official evidence", self.metadata())

    def test_material_reference_cannot_claim_positive_status(self):
        record = self.metadata()["fact"]
        record["evidence_ref"] = "material:unverified"
        with self.assertRaises(ValueError):
            validate_fact(record)

    def test_path_and_base64_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                SubjectMaterialStore(folder, Path(folder)/"facts.json").import_material("../a.txt", b"x", self.metadata())
        with self.assertRaises(ValueError):
            decode_content("***bad***")

    def test_literal_material_field_match_never_establishes_authenticity(self):
        fields={"subject_name":"测试主体有限公司", "subject_id":"91310101TEST000001", "document_number":"夹具001",
                "issuer":"测试机构", "scope":"合成技术服务", "valid_from":"2026-01-01", "valid_until":"2026-12-31"}
        text="测试主体有限公司\n91310101TEST000001\n夹具001\n测试机构\n合成技术服务\n2026年1月1日至2026年12月31日"
        result=check_material_fields("fixture.txt",text.encode(),fields)
        self.assertTrue(result["all_fields_literally_matched"])
        self.assertFalse(result["authenticity_established"])
        fields["subject_id"]="91310101OTHER00001"
        result=check_material_fields("fixture.txt",text.encode(),fields)
        self.assertFalse(result["all_fields_literally_matched"])

    def test_invalid_record_and_metadata_tamper_fail_closed(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            store=SubjectMaterialStore(Path(folder)/"materials",Path(folder)/"facts.json")
            result=store.import_material("fixture.txt",b"synthetic",self.metadata())
            bad=store.root/(result["material"]["binding_id"]+".json")
            value=json.loads(bad.read_text(encoding="utf-8"))
            value["issuer_declared"]="modified"
            bad.write_text(json.dumps(value),encoding="utf-8")
            (store.root/"array.json").write_text("[]",encoding="utf-8")
            self.assertEqual(len(store.list_materials()["errors"]),2)
            with self.assertRaises(ValueError):
                store.import_material("fixture.txt",b"synthetic",self.metadata())


if __name__ == "__main__":
    unittest.main()
