"""Observe the true model result and consume semantic pending independently."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents as A, nodes as N, rules as R
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Contract, DocumentParse
from zhilv_qiling_v2.retrieve import RetrievalHit
from zhilv_qiling_v2.tests_v2 import test_initial_review_nodes as legacy

clause, finding, page = legacy.clause, legacy.finding, legacy.page


class ClaimGroundingNodeTests(unittest.TestCase):
    def review(self, *responses, rows=None, **settings):
        return legacy.InitialReviewNodeTests().review(list(responses), rows=rows, **settings)

    def test_source_transport_complete_cannot_become_semantic_complete(self):
        quote = "卖方账号：62123456789012345678"
        raw = finding(quote=quote, code="one_sided_breach", message="合同仅约定卖方违约责任。")
        result = self.review(page([raw]), rows=[clause("a", quote)])
        candidates, meta, _, _, _, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertFalse(meta["semantic_complete"])
        self.assertTrue(meta["semantic_pending"])
        self.assertEqual(meta["initial_review_lineage"][0]["attempts"][0]["content"]["defects"][0], raw)
        self.assertEqual(candidates[0].message, raw["message"])
        self.assertEqual(candidates[0].extra["claim_grounding_status"], "unsupported")
        self.assertTrue(candidates[0].extra["manual_required"])
        evidence = [row for row in candidates[0].evidence if row.kind == "claim_grounding"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].strength, 0)
        self.assertEqual(evidence[0].extra["finding_unchanged"], raw)
        self.assertTrue(legacy.InitialReviewNodeTests().checked(result)["protocol_lineage_acceptance_passed"])

    def test_duplicate_and_invalid_citation_observations_stay_in_ledger(self):
        raw = finding()
        _, meta, _, _, _, _, _ = self.review(page([raw, raw]))
        self.assertEqual(len(meta["initial_semantic_ledger"]), 2)
        self.assertTrue(meta["initial_semantic_ledger"][1]["deduplicated_candidate"])
        self.assertEqual(meta["accepted"], 1)
        _, invalid, _, _, _, _, _ = self.review(page([finding(quote="不存在的引文。")]), llm_initial_max_attempts=1)
        row = invalid["initial_semantic_ledger"][0]
        self.assertFalse(row["candidate_materialized"])
        self.assertEqual(row["assessment"]["claim_grounding_status"], "unsupported")
        self.assertFalse(invalid["complete"])

    def test_supported_literal_positive_control_and_source_payload_unchanged(self):
        quote = "乙方不承担任何赔偿责任。"
        raw = finding(quote=quote, code="liability_exemption", message="乙方被约定不承担赔偿责任。")
        candidates, meta, _, captured, _, _, _ = self.review(page([raw]), rows=[clause("a", quote)])
        self.assertEqual(candidates[0].extra["claim_grounding_status"], "supported")
        self.assertTrue(meta["semantic_complete"])
        self.assertFalse(meta["semantic_automatic_legal_confirmation_allowed"])
        self.assertEqual(captured[0]["payload"]["segments"][0]["text"], quote)
        slots = next(iter(meta["semantic_contexts"].values()))["fact_slots"]
        self.assertEqual(slots[0]["source"]["text"], quote)

    def statute(self, candidates, row):
        doc = Contract("claim-test", DocumentParse("claim.txt", "txt", row.text, clauses=[row]))
        hit = RetrievalHit("test-law", "合成测试规范", "第一条", "仅供协议测试的条文。", .99)
        verdict = N.E.EntailmentVerdict("test-law", "第一条", "contradict", .99, source="bert_entail")
        retriever = SimpleNamespace(search=lambda *args, **kwargs: SimpleNamespace(hits=[hit]))
        with patch.object(N.E, "annotate", return_value=[verdict]):
            outcome = N.node_statute(doc, candidates, Config(), retriever=retriever)
        return doc, outcome

    def test_actual_relation_label_is_preserved_but_blocked_claim_not_confirmed(self):
        row = clause("a", "卖方账号：62123456789012345678")
        candidate = self.review(page([finding(quote=row.text, code="one_sided_breach",
            message="合同仅约定卖方违约责任。")]), rows=[row])[0][0]
        _, outcome = self.statute([candidate], row)
        check = outcome.payload["risk_checks"][0]
        self.assertEqual(check["label"], "contradict")
        self.assertEqual(check["raw_relation_label"], "contradict")
        self.assertIsNone(check["effective_relation_label"])
        self.assertFalse(check["relation_usable_for_confirmation"])
        self.assertTrue(check["upstream_claim_grounding_pending"])
        self.assertEqual(outcome.payload["conflicts"], 0)
        self.assertFalse(any(item.kind == "statute_relation" for item in candidate.evidence))
        self.assertTrue(outcome.citations)
        self.assertFalse(any(item.kind == "entailment_window_scope" for item in candidate.evidence))

    def test_independent_candidate_is_not_suppressed_and_fusion_obeys_manual(self):
        row = clause("a", "乙方不承担任何赔偿责任。")
        pending = R._candidate("one_sided_breach", "合成未支持主张。", row, row.text, confidence=.99, source="node2_llm")
        pending.extra.update(claim_grounding_status="unsupported", manual_required=True, semantic_pending=True)
        independent = R._candidate("liability_exemption", "合成独立规则依据。", row, row.text, confidence=.99)
        doc, outcome = self.statute([pending, independent], row)
        checks = outcome.payload["risk_checks"]
        self.assertFalse(checks[0]["relation_usable_for_confirmation"])
        self.assertTrue(checks[1]["relation_usable_for_confirmation"])
        self.assertEqual(checks[1]["effective_relation_label"], "contradict")
        result = A.AgentResult("statute", citations=outcome.payload["per_candidate"])
        risks = A.fuse([pending, independent], [result], doc, Config())
        self.assertTrue(next(item for item in risks if item.code == pending.code).review_required)
        self.assertFalse(next(item for item in risks if item.code == independent.code).review_required)

    def test_existing_manual_requirement_survives_supported_text_and_relation(self):
        row = clause("a", "乙方不承担任何赔偿责任。")
        candidate = R._candidate("liability_exemption", "合成文本主张。", row, row.text, confidence=.99)
        candidate.extra.update(claim_grounding_status="supported", manual_required=True)
        _, outcome = self.statute([candidate], row)
        check = outcome.payload["risk_checks"][0]
        self.assertEqual(check["raw_relation_label"], "contradict")
        self.assertTrue(check["upstream_manual_required"])
        self.assertIsNone(check["effective_relation_label"])
        self.assertFalse(check["relation_usable_for_confirmation"])


if __name__ == "__main__":
    unittest.main()
