"""Literal grounding controls, not a labelled legal-accuracy benchmark."""
import copy
import json
from pathlib import Path
import unittest

from zhilv_qiling_v2.claim_grounding import assess_claim


def finding(quote, code="fee_mismatch", message="需核验。", cid="a", related=()):
    return {"clause_id": cid, "code": code, "severity": "medium", "message": message,
            "quote": quote, "related_quotes": list(related)}


def segment(text, cid="a", start=0):
    return {"clause_id": cid, "text": text, "start": start, "end": start + len(text)}


class ClaimGroundingTests(unittest.TestCase):
    def test_account_excerpt_cannot_support_substantive_claim(self):
        quote = "卖方账号：62123456789012345678"
        for code, message in (("one_sided_breach", "仅约束卖方违约责任。"),
                              ("vague_delivery", "交付期限不明确。"),
                              ("unclear_standard", "验收条件不明确。")):
            result = assess_claim(finding(quote, code, message), [segment(quote)])
            self.assertEqual(result["claim_grounding_status"], "unsupported")
            self.assertIn("account_extract_does_not_support_substantive_claim", result["reasons"])

    def test_purchase_source_cannot_support_invented_rent_requirement(self):
        quote = "采购合同总价170万元。"
        result = assess_claim(finding(quote, message="合同未约定租金，应补充租金条款。"),
                              [segment(quote)], contract_type="政府采购")
        self.assertEqual(result["claim_grounding_status"], "unsupported")
        self.assertIn("unfounded_lease_assumption_in_purchase_scope", result["reasons"])

    def test_local_absence_is_unknown_and_never_full_contract_proof(self):
        quote = "验收标准另见附件。"
        result = assess_claim(finding(quote, "unclear_standard", "合同没有约定验收标准。"), [segment(quote)])
        self.assertEqual(result["claim_grounding_status"], "unknown")
        self.assertIn("local_absence_cannot_establish_whole_contract_absence", result["reasons"])

    def test_total_and_tail_are_not_a_conflict_or_added_together(self):
        quote = "甲方支付本合同总价100000元，尾款70000元。"
        result = assess_claim(finding(quote, message="总价与尾款相加合计170000元，存在冲突。"), [segment(quote)])
        self.assertEqual(result["claim_grounding_status"], "unsupported")
        self.assertIn("contract_total_and_tail_are_not_additive_or_same_role", result["reasons"])

    def test_uppercase_lowercase_equal_unequal_and_missing_pairs(self):
        for quote, expected, reason in (
            ("合同总价大写壹佰柒拾万元整（小写1700000元）。", "unsupported", "explicit_uppercase_lowercase_values_are_equal"),
            ("合同总价大写壹佰柒拾万元整（小写1800000元）。", "supported", "explicit_uppercase_lowercase_literal_value_difference"),
            ("合同总价1700000元。", "unknown", "no_unambiguous_explicit_uppercase_lowercase_pair")):
            result = assess_claim(finding(quote, "amount_inconsistent", "金额大小写存在差异。"), [segment(quote)])
            self.assertEqual(result["claim_grounding_status"], expected, result)
            self.assertIn(reason, result["reasons"] + result["literal_support_reasons"])
            self.assertFalse(result["automatic_confirmation_allowed"])

    def test_literal_exemption_is_supported_but_illegality_is_unknown(self):
        quote = "乙方不承担任何赔偿责任。"
        literal = assess_claim(finding(quote, "liability_exemption", "乙方被约定不承担赔偿责任。"), [segment(quote)])
        self.assertEqual(literal["claim_grounding_status"], "supported")
        legal = assess_claim(finding(quote, "liability_exemption", "该免责约定违法无效。"), [segment(quote)])
        self.assertEqual(legal["claim_grounding_status"], "unknown")
        self.assertFalse(literal["legal_truth_established"])

    def test_exemption_wording_does_not_support_different_actor_or_unrelated_claim(self):
        quote = "乙方不承担任何赔偿责任。"
        changed_actor = assess_claim(finding(quote, "liability_exemption", "甲方不承担赔偿责任。"), [segment(quote)])
        self.assertEqual(changed_actor["claim_grounding_status"], "unsupported")
        self.assertIn("claimed_actor_not_in_cited_evidence", changed_actor["reasons"])
        unrelated = assess_claim(finding(quote, "liability_exemption", "付款应在次月完成。"), [segment(quote)])
        self.assertEqual(unrelated["claim_grounding_status"], "unknown")

    def test_same_role_difference_needs_both_citations(self):
        first, second = "甲方支付本合同尾款70000元。", "甲方支付本合同尾款80000元。"
        sources = [segment(first), segment(second, "b")]
        unbound = assess_claim(finding(first, message="同一尾款存在数值差异。"), sources)
        self.assertEqual(unbound["claim_grounding_status"], "unknown")
        bound = assess_claim(finding(first, message="同一尾款存在数值差异。",
                              related=[{"clause_id": "b", "quote": second}]), sources)
        self.assertEqual(bound["claim_grounding_status"], "supported")
        self.assertTrue(bound["manual_required"])
        self.assertFalse(bound["automatic_confirmation_allowed"])

    def test_raw_finding_and_source_are_immutable_with_exact_unicode_offsets(self):
        quote = "乙方不承担任何赔偿责任。"
        claim, sources = finding(quote, "liability_exemption", "乙方不承担赔偿责任。"), [segment("😃" + quote, start=17)]
        before = copy.deepcopy((claim, sources))
        result = assess_claim(claim, sources)
        self.assertEqual((claim, sources), before)
        self.assertEqual(result["finding_unchanged"], claim)
        self.assertEqual(result["source_binding_matches"][0]["matches"][0]["start"], 18)
        bad = assess_claim(finding("乙方 不承担任何赔偿责任。"), sources)
        self.assertEqual(bad["claim_grounding_status"], "unsupported")

    def test_prior_actual_eight_findings_remain_unconfirmed_offline(self):
        root = Path(__file__).resolve().parents[2]
        directory = root / "tools/acceptance_results/initial_source_selection_real_20261118_v2"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8-sig"))
        prepared = json.loads((directory / "prepared.json").read_text(encoding="utf-8-sig"))
        # These are historical model observations, expressly not human legal gold.
        meta = report["stats"]["logic_llm"]
        calls = [row for row in meta["calls"] if row.get("source_selection")]
        window = prepared["windows"][0]
        findings = calls[0]["source_selection"]["resolved_content"]["defects"]
        self.assertEqual(len(findings), 8)
        for row in findings:
            assessed = assess_claim(row, window["segments"], contract_type="政府采购")
            self.assertNotEqual(assessed["claim_grounding_status"], "supported", row)
            self.assertTrue(assessed["manual_required"])


if __name__ == "__main__":
    unittest.main()
