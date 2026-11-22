"""Exact literal-slot tests; these are not independent legal gold."""
from decimal import Decimal
import unittest

from zhilv_qiling_v2.fact_slots import extract_fact_slots, financial_yuan


class FactSlotsTests(unittest.TestCase):
    def test_distinct_payment_roles_subject_recipient_and_scope(self):
        text = "甲方应向乙方支付本合同总价人民币100000元，预付款30000元，尾款70000元。"
        result = extract_fact_slots("clause-a", text)
        self.assertEqual([row["amount_role"] for row in result["amounts"]], ["contract_total", "advance", "tail"])
        self.assertEqual([row["value_yuan"] for row in result["amounts"]], ["100000", "30000", "70000"])
        self.assertTrue(all(row["subject"] == "甲方" and row["recipient"] == "乙方" for row in result["amounts"]))
        self.assertTrue(all(row["object_scope"]["name"] == "本合同" for row in result["amounts"]))
        self.assertFalse(any(row["fact_verified"] for row in result["amounts"]))

    def test_unicode_and_original_clause_offsets_are_exact(self):
        text = "😀甲方支付本合同尾款人民币1，700,000.00元。"
        result = extract_fact_slots("c", text, start_offset=120)
        row = result["amounts"][0]
        span = row["source_span"]
        self.assertEqual(span["quote"], "1，700,000.00元")
        self.assertEqual(text[span["start"] - 120:span["end"] - 120], span["quote"])
        self.assertEqual(row["value_yuan"], "1700000.00")

    def test_uppercase_and_arabic_amounts_are_separate_literal_writings(self):
        text = "本合同总价大写壹佰柒拾万元整（小写：￥1，700,000.00元）。"
        rows = extract_fact_slots("c", text)["amounts"]
        self.assertEqual([row["writing"] for row in rows], ["chinese", "arabic"])
        self.assertEqual({Decimal(row["value_yuan"]) for row in rows}, {Decimal("1700000")})
        self.assertEqual({row["amount_role"] for row in rows}, {"contract_total"})

    def test_unknown_subject_and_role_are_not_inherited(self):
        first = extract_fact_slots("a", "甲方支付本合同尾款70000元。")
        second = extract_fact_slots("b", "另列人民币30000元。")
        self.assertEqual(first["amounts"][0]["subject"], "甲方")
        self.assertIsNone(second["amounts"][0]["subject"])
        self.assertEqual(second["amounts"][0]["amount_role"], "unknown")
        self.assertIsNone(second["amounts"][0]["object_scope"]["name"])

    def test_condition_subject_does_not_replace_operative_payer(self):
        text = "甲方在乙方完成本合同最终验收后支付尾款70000元。"
        row = extract_fact_slots("a", text)["amounts"][0]
        self.assertEqual(row["subject"], "甲方")
        self.assertTrue(row["conditions"])
        self.assertIn("乙方", row["conditions"][0]["span"]["quote"])
        self.assertEqual(row["conditions"][0]["timing"], "after")

    def test_negation_exceptions_and_qualified_value_remain_explicit(self):
        result = extract_fact_slots("a", "除非甲方书面同意，乙方不得支付本合同尾款70000元。乙方本合同保证金不超过1000元。")
        first, second = result["amounts"]
        self.assertEqual(first["polarity"], "negative")
        self.assertTrue(first["exceptions"])
        self.assertEqual(second["value_kind"], "qualified")
        self.assertIn("不超过", second["qualifiers"])

    def test_section_title_and_account_numbers_are_not_payment_facts(self):
        result = extract_fact_slots("a", "4、付款方式\n账号：110958642310000，开户行号：308100005545。")
        self.assertEqual(result["amounts"], [])
        self.assertEqual(result["obligations"], [])

    def test_chinese_financial_grammar_is_deterministic(self):
        self.assertEqual(financial_yuan("壹佰零伍元伍角陆分"), Decimal("105.56"))
        self.assertEqual(financial_yuan("壹亿零壹万元"), Decimal("100010000"))
        self.assertEqual(financial_yuan("拾元"), Decimal("10"))
        self.assertIsNone(financial_yuan("金额大约若干元"))

    def test_invalid_scope_offsets_are_rejected(self):
        for kwargs in ({"start_offset": -1}, {"start_offset": True}):
            with self.assertRaises(ValueError):
                extract_fact_slots("a", "100元", **kwargs)


if __name__ == "__main__":
    unittest.main()
