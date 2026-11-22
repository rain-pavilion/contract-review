"""Replay observed NER outputs; engineering checks, not independent legal gold."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from zhilv_qiling_v2 import ner
from zhilv_qiling_v2.models import Clause, Contract, ContractElements, DocumentParse, Element, SourceBlock, SpanRef
from zhilv_qiling_v2.ner_constraints import constrain_hits
from zhilv_qiling_v2.subject_graph import build_subject_graph
from zhilv_qiling_v2.tests_v2.test_selected_safety_guards import document

ROOT = Path(__file__).resolve().parents[2]
OBSERVED = ROOT / "tools/package_core_delivery_20261121/final_profile_run/real_report_readback.json"


def element(row):
    row = deepcopy(row)
    row["span"] = SpanRef(**row["span"]) if row.get("span") else None
    return Element(**row)


def observed_contract():
    report = json.loads(OBSERVED.read_text(encoding="utf-8"))
    data = report["document"]
    doc = DocumentParse(data["source_name"], data["file_type"], data["full_text"],
        blocks=[SourceBlock(**row) for row in data["blocks"]],
        clauses=[Clause(**row) for row in data["clauses"]], stats=data["stats"])
    elements = ContractElements(**{key: [element(row) for row in values]
                                  for key, values in report["elements"].items()})
    return Contract(report["contract_id"], doc, elements, contract_type=report["contract_type"])


class EntityBoundaryTests(unittest.TestCase):
    def test_definition_fragments_are_rejected_with_source_audit(self):
        for raw in ("本合同中方系指：_北", "本合同乙方系指：", "本合同乙方系指：_中标人", "甲方是指：示例公司"):
            with self.subTest(raw=raw):
                doc = document(raw)
                hit = Element("e", "party", raw, raw, span=doc.locate("b", 0, len(raw)), source="bert_ner")
                accepted, rejected = constrain_hits(doc, [hit])
                self.assertFalse(accepted)
                self.assertIn("role_definition_fragment_not_entity", rejected[0]["reasons"])
                self.assertEqual(rejected[0]["value"], raw)

    def test_multiple_numbers_are_not_a_single_amount(self):
        for raw in ("9500  40.00元", "240000  14.19元", "10元至20元"):
            doc = document(raw)
            hit = Element("e", "amount", raw, "9500.00", span=doc.locate("b", 0, len(raw)))
            accepted, rejected = constrain_hits(doc, [hit])
            self.assertFalse(accepted)
            self.assertIn("multiple_numeric_values_in_amount_span", rejected[0]["reasons"])
        for raw in ("40.00元", "6.30元", "人民币1,234.56元", "2万元"):
            doc = document(raw)
            accepted, rejected = constrain_hits(doc, ner.extract_amounts(doc))
            self.assertEqual(len(accepted), 1)
            self.assertFalse(rejected)

    def test_adjustment_is_not_a_new_total(self):
        doc = document("较原合同总金额增加人民币222130.26元。")
        hits = ner.extract_amounts(doc)
        self.assertEqual(hits[0].role, "total")
        accepted, rejected = constrain_hits(doc, hits)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0].role, "adjustment")
        self.assertEqual(accepted[0].extra["constraint_audit"]["adjustments"][0]["from"], "total")
        doc = document("合同总金额增加至人民币222130.26元。")
        accepted, _ = constrain_hits(doc, ner.extract_amounts(doc))
        self.assertEqual(accepted[0].role, "total")

    def test_unverified_ocr_chinese_amount_is_not_comparison_evidence(self):
        doc = document("合同总价壹万元整。")
        doc.blocks[0].metadata["extraction_quality"] = {"status": "machine_ocr_unverified", "human_verified": False}
        accepted, rejected = constrain_hits(doc, ner.extract_amounts(doc))
        self.assertFalse(rejected)
        self.assertEqual(accepted[0].normalized, "10000.00")
        self.assertFalse(accepted[0].extra["numeric_comparison_eligible"])
        self.assertTrue(accepted[0].extra["manual_required"])

    def test_generic_exact_name_merges_only_with_unique_role_and_keeps_mentions(self):
        raw = "示例科技有限公司"
        doc = document(raw)
        generic = Element("g", "party", "_" + raw, raw, role="other", source="bert_ner")
        seller = Element("s", "party", raw, raw, role="seller", source="rule")
        for mentions in ([generic, seller], [seller, generic]):
            nodes, _ = build_subject_graph(Contract("c", doc, ContractElements(parties=mentions)))
            self.assertEqual([(n.name, n.role) for n in nodes], [(raw, "seller")])
            self.assertEqual({m["element_id"] for m in nodes[0].attrs["source_mentions"]}, {"g", "s"})
            self.assertIn("_" + raw, nodes[0].attrs["aliases"])
            self.assertEqual(generic.role, "other")
        buyer = Element("b", "party", raw, raw, role="buyer")
        nodes, _ = build_subject_graph(Contract("c", doc, ContractElements(parties=[generic, seller, buyer])))
        self.assertEqual({n.role for n in nodes}, {"other", "buyer", "seller"})

    def test_unverified_chinese_fragment_cannot_trigger_notation_conflict(self):
        doc = document("合同总价10000.26元（大写：壹万元）。")
        doc.blocks[0].metadata["extraction_quality"] = {"status": "machine_ocr_unverified"}
        hits, _ = constrain_hits(doc, ner.extract_amounts(doc))
        signals = ner.element_signals(ner.fuse_elements(hits, []), doc)
        self.assertNotIn("amount_inconsistent", [signal.kind for signal in signals])
        doc.blocks[0].metadata = {}
        hits, _ = constrain_hits(doc, ner.extract_amounts(doc))
        signals = ner.element_signals(ner.fuse_elements(hits, []), doc)
        self.assertIn("amount_inconsistent", [signal.kind for signal in signals])

    def test_similar_company_names_are_not_merged(self):
        names = ["示例科技有限公司", "示例科技有限公司分公司"]
        doc = document("；".join(names))
        parties = [Element(str(i), "party", name, name, role="seller") for i, name in enumerate(names)]
        nodes, _ = build_subject_graph(Contract("c", doc, ContractElements(parties=parties)))
        self.assertEqual({n.name for n in nodes}, set(names))


@unittest.skipUnless(OBSERVED.is_file(), "Recorded real BERT output is unavailable; no synthetic replacement")
class ObservedEntityReplayTests(unittest.TestCase):
    def test_recorded_real_bert_mentions_resolve_to_two_subjects(self):
        contract = observed_contract()
        self.assertTrue(any(e.source == "bert_ner" for e in contract.elements.parties))
        self.assertGreater(len(contract.elements.parties), 2)
        before = deepcopy(contract.elements.to_dict())
        nodes, _ = build_subject_graph(contract)
        self.assertEqual({(n.name, n.role) for n in nodes}, {
            ("北京市统计局", "buyer"), ("北京力信诚印务科技有限公司", "seller")})
        self.assertEqual(contract.elements.to_dict(), before)
        mentions = [m for n in nodes for m in n.attrs["source_mentions"]]
        self.assertEqual(len(mentions), len(contract.elements.parties) - 3)
        self.assertTrue(any(m["role_binding"] for m in mentions))

    def test_recorded_real_model_span_rejections_preserve_usable_amounts(self):
        contract = observed_contract()
        amounts, rejected = constrain_hits(contract.document, contract.elements.amounts)
        mixed = [r for r in rejected if "multiple_numeric_values_in_amount_span" in r["reasons"]]
        self.assertEqual(len(mixed), 5)
        self.assertTrue(all(r["source"] == "bert_ner" for r in mixed))
        self.assertTrue({"40.00", "13.00", "12.00", "6.30", "14.19"}.issubset({e.normalized for e in amounts}))
        adjustment = [e for e in amounts if e.normalized == "222130.26"]
        self.assertTrue(adjustment)
        self.assertTrue(all(e.role == "adjustment" for e in adjustment))
        parties, rejected = constrain_hits(contract.document, contract.elements.parties)
        self.assertEqual(len(rejected), 3)
        contract.elements.parties = parties
        nodes, _ = build_subject_graph(contract)
        self.assertEqual(len(nodes), 2)


if __name__ == "__main__":
    unittest.main()
