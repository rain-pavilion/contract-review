"""Preserve complete source groups and keep unverified OCR findings provisional."""
import unittest
from zhilv_qiling_v2 import long_context as C
from zhilv_qiling_v2 import agents
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, RiskCandidate, Citation, SpanRef, Evidence
from zhilv_qiling_v2.risk_identity import normalize_fact_candidates


def clause(cid, text, group="", source="original"):
    item = Clause(cid, "", "", 1, "", text, text, [cid])
    if group:
        item.metadata.update(source_id=source, article_group_id=group)
    return item


class PackageSourceContextTests(unittest.TestCase):
    def test_payment_subclauses_stay_together_when_they_fit(self):
        clauses = [clause("intro", "背景" * 80), clause("a", "预付款" * 20, "payment"),
                   clause("b", "验收与尾款" * 20, "payment")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=0)
        self.assertTrue(plan["source_context_group_plan_complete"])
        grouped = [w for w in windows if {"a", "b"} <= {s.clause_id for s in w.segments}]
        self.assertEqual(len(grouped), 1)
        for w in windows:
            self.assertLessEqual(sum(len(s.text) for s in w.segments), 200)
        complete = C.coverage(clauses, windows, {w.window_id for w in windows})
        self.assertTrue(complete["source_context_group_complete"])

    def test_separate_completed_fragments_do_not_acknowledge_whole_table(self):
        clauses = [clause("a", "货物名称数量" * 20, "table"), clause("b", "单价与金额" * 25, "table")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=0)
        coverage = C.coverage(clauses, windows, {w.window_id for w in windows})
        self.assertTrue(coverage["text_complete"])
        self.assertFalse(coverage["source_context_group_complete"])
        self.assertFalse(coverage["complete"])
        self.assertFalse(plan["source_context_group_plan_complete"])
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=1)
        self.assertTrue(plan["source_context_group_plan_complete"])
        primary_only = {w.window_id for w in windows if w.kind == "text"}
        self.assertFalse(C.coverage(clauses, windows, primary_only)["source_context_group_complete"])
        self.assertTrue(C.coverage(clauses, windows, {w.window_id for w in windows})["source_context_group_complete"])

    def test_same_group_label_from_different_documents_never_merges(self):
        clauses = [clause("a", "原合同付款", "第三条", "original"),
                   clause("b", "补充协议付款", "第三条", "supplement")]
        _, plan = C.build_review_plan(clauses, cross_pairs=0)
        self.assertEqual(plan["source_context_groups"], [])

    def test_long_single_clause_full_context_group_does_not_pass_by_fragments(self):
        c = clause("a", "复杂条款。" * 200, "single_article")
        c.metadata["requires_full_context"] = True
        windows, plan = C.build_review_plan([c], chunk_chars=200, overlap=0, cross_pairs=0)
        self.assertFalse(plan["source_context_group_plan_complete"])
        result = C.coverage([c], windows, {window.window_id for window in windows})
        self.assertTrue(result["text_complete"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["source_context_group_pending"], ["single_article"])

    def test_ocr_numeric_candidate_remains_manual_even_with_high_confidence_citation(self):
        c = clause("a", "OCR识别的合同金额1元。")
        c.metadata.update(source_id="scan", source_title="合同扫描件", source_page_start=2,
            source_page_end=2, original_sha256="a" * 64,
            extraction_quality={"status": "machine_ocr_unverified", "human_verified": False})
        contract = Contract("contract", DocumentParse("scan", "pdf", c.text, clauses=[c]))
        candidate = RiskCandidate("r", "fee_mismatch", "金额候选", clause_ids=["a"], confidence=.99)
        normalized = normalize_fact_candidates([candidate], contract)
        result = agents.AgentResult("fixture", candidates=normalized,
            citations={"r": [Citation("测试法", "第一条", "fixture")]})
        risks = agents.fuse(normalized, [result], contract, Config())
        self.assertTrue(risks[0].review_required)
        self.assertTrue(candidate.extra["source_text_requires_review"])
        self.assertEqual(candidate.extra["source_review_sources"][0]["source_page_start"], 2)
        normalize_fact_candidates([candidate], contract)
        self.assertEqual(sum(ev.kind == "source_text_quality" for ev in candidate.evidence), 1)

    def test_ocr_span_only_candidate_cannot_bypass_source_gate(self):
        c = clause("a", "识别金额1元")
        c.metadata.update(source_id="scan", extraction_quality={"status": "machine_ocr_unverified"})
        contract = Contract("contract", DocumentParse("scan", "pdf", c.text, clauses=[c]))
        for mode in ("candidate_span", "evidence_span"):
            candidate = RiskCandidate("r", "fee_mismatch", "金额候选")
            span = SpanRef("a", "1元", 4, 6)
            if mode == "candidate_span":
                candidate.spans.append(span)
            else:
                candidate.evidence.append(Evidence("source", "1元", span=span))
            normalize_fact_candidates([candidate], contract)
            self.assertTrue(candidate.extra["manual_required"])
            self.assertFalse(candidate.extra["automatic_confirmation_allowed"])
            self.assertEqual(candidate.clause_ids, [])


if __name__ == "__main__":
    unittest.main()
