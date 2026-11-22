"""Source-bound literal amount relations and extraction-coverage gates.

These tests use hand-written text only; they do not read the frozen development
set and make no legal-accuracy claim.
"""
from __future__ import annotations

import unittest

from zhilv_qiling_v2 import agents as A
from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2 import rules as R
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import (Clause, Contract, ContractElements, DocumentParse,
                                    ElementSignal, Evidence, FinalRisk, SpanRef)
from zhilv_qiling_v2.source_relations import (amount_relation_candidates,
                                              extraction_gap_metadata)


def clause(cid, text, label=""):
    return Clause(cid, label, "", 1, "", text, text, ["b" + cid])


def contract(clauses, signals=()):
    return Contract("test-contract", DocumentParse("test.txt", "txt",
        "\n".join(c.text for c in clauses), clauses=clauses), signals=list(signals))


def detect(*texts, complete=True):
    return amount_relation_candidates(
        [{"clause_id": "clause-" + str(index), "text": text} for index, text in enumerate(texts)],
        contract_id="self-authored-test", scope_complete=complete)


class ExtractionGapGateTest(unittest.TestCase):
    def test_spanless_extraction_notice_kept_but_not_deducted(self):
        signal = ElementSignal(kind="subject_unspecified", note="未识别到标的/项目名称")
        candidate = R.signals_to_candidates([signal])[0]
        self.assertEqual(candidate.extra["claim_grounding_status"], "extraction_gap")
        self.assertFalse(candidate.extra["score_eligible"])
        self.assertTrue(candidate.extra["manual_required"])
        self.assertEqual(len(candidate.evidence), 2)
        final = FinalRisk("risk", candidate.code, "事实不足", candidate.severity, candidate.message,
                          evidence=candidate.evidence, confidence=.9, review_required=True)
        score, detail = A.score_risks([final], Config())
        self.assertEqual(score, 100)
        self.assertEqual(len(detail["excluded_unsubstantiated"]), 1)
        self.assertEqual(detail["excluded_unsubstantiated"][0]["text_support_status"], "extraction_gap")

    def test_other_extraction_categories_are_absence_notices_too(self):
        for kind, note in [("party_missing", "未抽取到任何当事人名称"),
                           ("amount_missing", "全文未识别到任何金额"),
                           ("date_missing", "全文未识别到关键日期"),
                           ("missing_party_info", "当事人角色不完整：已识别 ['buyer']，缺少卖方")]:
            with self.subTest(kind=kind):
                candidate = R.signals_to_candidates([ElementSignal(kind=kind, note=note)])[0]
                self.assertEqual(candidate.extra["claim_grounding_status"], "extraction_gap")

    def test_real_source_backed_or_non_extraction_signals_are_not_blanket_excluded(self):
        span = SpanRef("block", "数量为0", 0, 4, "clause")
        for signal in [ElementSignal("subject_unspecified", "标的数量为0", spans=[span]),
                       ElementSignal("subject_unspecified", "图上有 1 处交叉引用指向不存在的条款"),
                       ElementSignal("date_reversed", "结束日期早于开始日期"),
                       ElementSignal("subject_unspecified", "正文明确约定标的待定")]:
            with self.subTest(note=signal.note):
                self.assertEqual(extraction_gap_metadata(signal), {})
                candidate = R.signals_to_candidates([signal])[0]
                self.assertNotEqual(candidate.extra.get("claim_grounding_status"), "extraction_gap")

    def test_node_logic_keeps_gap_candidate_but_judge_does_not_deduct(self):
        signal = ElementSignal(kind="subject_unspecified", note="未识别到标的/项目名称")
        contract_ = contract([clause("a", "第一条 标的。")], signals=[signal])
        outcome = N.node_logic(contract_, Config(enable_llm=False))
        gaps = [c for c in outcome.candidates
                if c.extra.get("claim_grounding_status") == "extraction_gap"]
        self.assertEqual(len(gaps), 1)
        baseline = [c for c in outcome.candidates if c not in gaps]
        base = A.judge(contract_, baseline, [], Config(enable_llm=False))
        judged = A.judge(contract_, outcome.candidates, [], Config(enable_llm=False))
        self.assertEqual(judged.payload["score"], base.payload["score"])
        excluded = judged.payload["detail"]["excluded_unsubstantiated"]
        self.assertTrue(any(row["text_support_status"] == "extraction_gap" for row in excluded))


class AmountRelationTest(unittest.TestCase):
    def assert_bound(self, result):
        documents = {row["clause_id"]: row["source"]["text"] for row in result["slot_documents"]}
        for candidate in result["candidates"]:
            self.assertFalse(candidate.extra["legal_conflict_established"])
            self.assertTrue(candidate.extra["manual_required"])
            for span in candidate.spans:
                self.assertEqual(documents[span.clause_id][span.char_start:span.char_end], span.quote)
            for binding in candidate.extra["source_relation"]["explicit_item_references"]:
                for span in binding["spans"]:
                    self.assertEqual(documents[span["clause_id"]][span["start"]:span["end"]], span["quote"])

    def test_repeated_tail_values_use_their_own_roles_not_total(self):
        result = detect("甲方向乙方支付本合同总价100000元，尾款70000元。",
                        "甲方向乙方支付本合同尾款80000元。")
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertIn("70000元", row.message)
        self.assertIn("80000元", row.message)
        self.assertNotIn("100000元", row.message)
        self.assert_bound(result)

    def test_same_item_phrase_with_relative_deadline_is_one_payment_node(self):
        result = detect(
            "第三条 价款 甲方向乙方支付本合同总价100000元，大写人民币壹拾万元整，"
            "含货物、包装、运输、安装及税费。甲方应支付的预付款为30000元，"
            "本合同最终验收后的同一笔尾款为70000元，总价由上述预付款与尾款构成。",
            "第六条 付款 甲方于2026年12月8日前向乙方支付预付款30000元；"
            "甲方在本合同最终验收单由双方签署后五日内，向乙方支付本合同最终验收后的同一笔尾款80000元。")
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertIn("70000元", row.message)
        self.assertIn("80000元", row.message)
        self.assertIn("五日内", row.message)
        self.assertEqual(row.extra["claim_grounding_status"], "supported")
        self.assertEqual(row.code, "cross_clause_inconsistency")
        relation = row.extra["source_relation"]
        self.assertEqual(relation["role"], "tail")
        self.assertTrue(relation["explicit_item_references"])
        self.assertFalse(relation["conditions_assumed_true"])
        self.assert_bound(result)

    def test_equal_tail_values_with_contract_total_are_not_a_conflict(self):
        result = detect(
            "甲方向乙方支付本合同总价100000元，预付款30000元，尾款70000元。",
            "甲方在本合同最终验收后支付本合同尾款70000元。")
        self.assertFalse(result["candidates"])
        self.assertFalse(result["observations"])

    def test_contract_total_and_instalments_are_never_compared(self):
        result = detect("甲方向乙方支付本合同总价100000元。",
                        "甲方支付本合同尾款80000元。")
        self.assertFalse(result["candidates"])

    def test_multiple_supported_wordings_and_chinese_amounts(self):
        pairs = [("甲方支付本合同尾款7万元。", "甲方须支付本合同尾款人民币80000元。"),
                 ("乙方应向甲方支付本项目预付款伍万元整。", "乙方向甲方支付本项目预付款60000元。"),
                 ("甲方支付本设备质保金1000元。", "甲方支付本设备质保金为1500元。")]
        for texts in pairs:
            with self.subTest(texts=texts):
                result = detect(*texts)
                self.assertEqual(len(result["candidates"]), 1)
                self.assert_bound(result)

    def test_decimal_units_and_equivalent_chinese_writings_are_not_differences(self):
        for other in ["70000.00元", "7万元", "柒万元整"]:
            self.assertFalse(detect("甲方支付本合同尾款70000元。",
                                    "甲方支付本合同尾款" + other + "。")["candidates"])

    def test_same_explicit_condition_supports_literal_difference_without_satisfaction(self):
        result = detect("甲方在乙方完成本合同最终验收后支付尾款50000元。",
                        "甲方在乙方完成本合同最终验收后支付尾款60000元。")
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertFalse(row.extra["source_relation"]["conditions_assumed_true"])
        self.assertIn("不代表条件已经满足", row.message)
        self.assert_bound(result)

    def test_different_actor_recipient_object_or_payment_role_is_not_same_obligation(self):
        pairs = [("甲方支付本合同尾款50000元。", "乙方支付本合同尾款60000元。"),
                 ("甲方向乙方支付本合同尾款50000元。", "甲方向卖方支付本合同尾款60000元。"),
                 ("甲方支付项目A尾款50000元。", "甲方支付项目B尾款60000元。"),
                 ("甲方支付本合同预付款50000元。", "甲方支付本合同尾款60000元。"),
                 ("甲方支付本合同总价100000元。", "甲方支付本合同尾款60000元。")]
        for texts in pairs:
            with self.subTest(texts=texts):
                self.assertFalse(detect(*texts)["candidates"])

    def test_different_phases_dates_and_rate_bases_remain_distinct(self):
        pairs = [("甲方支付本合同首批尾款50000元。", "甲方支付本合同第二批尾款60000元。"),
                 ("甲方在本合同初步验收后支付尾款50000元。", "甲方在本合同最终验收后支付尾款60000元。"),
                 ("甲方于2027年1月1日支付本合同尾款50000元。", "甲方于2027年2月1日支付本合同尾款60000元。"),
                 ("甲方支付本合同每月租金5000元。", "甲方支付本合同每年租金60000元。")]
        for texts in pairs:
            with self.subTest(texts=texts):
                self.assertFalse(detect(*texts)["candidates"])

    def test_amendments_exceptions_qualified_negative_and_range_amounts_are_not_flattened(self):
        base = "甲方支付本合同尾款50000元。"
        others = ["经变更，甲方支付本合同尾款调整为60000元。",
                  "补充协议约定甲方支付本合同尾款60000元。",
                  "除甲方书面同意外，甲方支付本合同尾款60000元。",
                  "甲方支付本合同尾款不超过60000元。",
                  "甲方无需支付本合同尾款60000元。",
                  "甲方支付本合同尾款50000元至60000元。"]
        for other in others:
            with self.subTest(other=other):
                self.assertFalse(detect(base, other)["candidates"])

    def test_nearby_unlabeled_fee_is_not_inherited_as_tail(self):
        result = detect("甲方支付本合同尾款50000元，另付运输费6000元。",
                        "甲方支付本合同尾款50000元。")
        self.assertFalse(result["candidates"])
        self.assertTrue(any("nearest_role_does_not_bind_additional_amount" in row["reasons"]
                            for row in result["diagnostics"]))

    def test_unbound_subject_and_object_stay_observations_without_scoring(self):
        result = detect("尾款50000元。", "尾款60000元。")
        self.assertFalse(result["candidates"])
        self.assertEqual(len(result["observations"]), 1)
        self.assertFalse(result["observations"][0]["score_eligible"])

    def test_incomplete_review_scope_does_not_become_score_supported(self):
        result = detect("甲方支付本合同尾款50000元。", "甲方支付本合同尾款60000元。", complete=False)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0].extra["claim_grounding_status"], "unknown")

    def test_node_logic_adds_bound_relation_candidate_for_repeated_tail(self):
        contract_ = contract([
            clause("b", "第三条 价款 甲方向乙方支付本合同总价100000元，大写人民币壹拾万元整，"
                        "含货物、包装、运输、安装及税费。甲方应支付的预付款为30000元，"
                        "本合同最终验收后的同一笔尾款为70000元，总价由上述预付款与尾款构成。"),
            clause("c", "第六条 付款 甲方于2026年12月8日前向乙方支付预付款30000元；"
                        "甲方在本合同最终验收单由双方签署后五日内，向乙方支付本合同最终验收后的同一笔尾款80000元。"),
        ])
        outcome = N.node_logic(contract_, Config(enable_llm=False))
        rows = [c for c in outcome.candidates if c.source == "literal_amount_relation"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.code, "cross_clause_inconsistency")
        self.assertEqual(row.extra["claim_grounding_status"], "supported")
        self.assertEqual(set(row.clause_ids), {"b", "c"})
        texts = {c.clause_id: c.text for c in contract_.document.clauses}
        for span in row.spans:
            self.assertEqual(texts[span.clause_id][span.char_start:span.char_end], span.quote)
        self.assertEqual(outcome.payload["literal_amount_relations"]["legal_truth_established"], False)

    def test_node_logic_does_not_add_relation_for_equal_tail(self):
        contract_ = contract([
            clause("b", "第三条 价款 甲方向乙方支付本合同总价100000元，尾款70000元。"),
            clause("c", "第六条 付款 甲方在本合同最终验收后支付本合同尾款70000元。"),
        ])
        outcome = N.node_logic(contract_, Config(enable_llm=False))
        self.assertFalse([c for c in outcome.candidates if c.source == "literal_amount_relation"])


if __name__ == "__main__":
    unittest.main()
