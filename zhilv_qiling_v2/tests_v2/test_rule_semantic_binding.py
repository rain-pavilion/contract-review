"""Evidence-bound rule checks, including replay of the real procurement report."""
import json
import unittest
from dataclasses import fields
from datetime import date
from pathlib import Path

from zhilv_qiling_v2 import detectors, rules
from zhilv_qiling_v2.models import Clause, Contract, ContractElements, DocumentParse, Element, SourceBlock, SpanRef


def clause(cid, text, **metadata):
    return Clause(cid, "", "", 1, "", text, text, block_ids=[cid], metadata=metadata)


def element(eid, value, *, kind="amount", role="other", cid=None, **extra):
    return Element(eid, kind, str(value), str(value), role=role, clause_id=cid, extra=extra)


def contract(clauses, *, amounts=(), dates=(), contract_type="通用"):
    return Contract("fixture", DocumentParse("fixture", "txt", "\n".join(c.text for c in clauses), clauses=clauses),
                    elements=ContractElements(amounts=list(amounts), dates=list(dates)), contract_type=contract_type)


def report_contract(payload):
    def select(cls, row):
        return {f.name: row[f.name] for f in fields(cls) if f.name in row}
    source = payload["document"]
    doc = DocumentParse(source["source_name"], source["file_type"], source["full_text"],
                        clauses=[Clause(**select(Clause, row)) for row in source["clauses"]],
                        blocks=[SourceBlock(**select(SourceBlock, row)) for row in source.get("blocks", [])],
                        stats=source.get("stats", {}))
    elements = ContractElements()
    for name in ("amounts", "dates", "parties", "subjects", "identifiers", "durations"):
        values = []
        for row in payload["elements"].get(name, []):
            data = select(Element, row)
            if isinstance(data.get("span"), dict):
                data["span"] = SpanRef(**select(SpanRef, data["span"]))
            values.append(Element(**data))
        setattr(elements, name, values)
    return Contract(payload["contract_id"], doc, elements=elements, contract_type=payload.get("contract_type", "通用"))


class RuleSemanticBindingTests(unittest.TestCase):
    def test_procurement_signatures_do_not_inherit_rental_twenty_year_rule(self):
        doc = contract([clause("a", "采购合同签署日2026年10月26日。"), clause("b", "扫描签署日2076年10月26日。")],
                       dates=[element("d1", "2026-10-26", kind="date", cid="a"), element("d2", "2076-10-26", kind="date", cid="b")],
                       contract_type="政府采购")
        self.assertFalse(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(doc)))

    def test_explicit_long_lease_term_still_triggers(self):
        doc = contract([clause("lease", "租赁期限自2026年1月1日起至2051年1月1日止。")], contract_type="房屋租赁")
        risks = detectors.detect_dates(doc)
        issue = next(r for r in risks if r.extra.get("issue_code") == "term_over_20y")
        self.assertEqual(issue.clause_ids, ["lease"])
        self.assertEqual(issue.extra["date_binding"], "explicit_lease_term_range")

    def test_unrelated_lease_signature_range_is_not_lease_period(self):
        doc = contract([clause("a", "两份历史文件签署日期为2026年1月1日至2051年1月1日。")], contract_type="房屋租赁")
        self.assertFalse(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(doc)))

    def test_exact_twenty_year_anniversary_is_allowed_and_next_day_is_flagged(self):
        exact = contract([clause("lease", "租赁期限自2020年1月1日起至2040年1月1日止。")], contract_type="房屋租赁")
        extra = contract([clause("lease", "租赁期限自2020年1月1日起至2040年1月2日止。")], contract_type="房屋租赁")
        self.assertFalse(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(exact)))
        self.assertTrue(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(extra)))

    def test_anniversary_handles_leap_day_and_calendar_upper_bound(self):
        self.assertEqual(detectors._twentieth_anniversary(date(2080, 2, 29)), date(2100, 2, 28))
        self.assertEqual(detectors._twentieth_anniversary(date(9990, 3, 1)), date.max)
        exact = contract([clause("lease", "租期自2080年2月29日至2100年2月28日。")])
        extra = contract([clause("lease", "租期自2080年2月29日至2100年3月1日。")])
        self.assertFalse(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(exact)))
        self.assertTrue(any(r.extra.get("issue_code") == "term_over_20y" for r in detectors.detect_dates(extra)))

    def test_fullwidth_source_quotes_are_preserved(self):
        bond = clause("bond", "履约保证金为６０００元，合同总价为１００００元。")
        doc = contract([bond])
        risk = rules.rule_abnormal_price(bond, rules.RuleContext.build(doc))[0]
        self.assertEqual(risk.extra["amount_binding"]["deposit"]["quote"], "保证金为６０００元")
        self.assertEqual(risk.spans[0].quote, "保证金为６０００元")
        rent = contract([clause("lease", "租期自２０２０年１月１日至２０４０年１月２日。")])
        dated = next(r for r in detectors.detect_dates(rent) if r.extra.get("issue_code") == "term_over_20y")
        self.assertEqual(dated.spans[0].quote, "２０２０年１月１日至２０４０年１月２日")

    def test_explicit_reversed_range_still_triggers(self):
        doc = contract([clause("a", "项目期限自2027年1月1日至2026年1月1日。")])
        self.assertIn("date_reversed", [r.code for r in detectors.detect_dates(doc)])

    def test_signing_and_payment_dates_are_not_work_period(self):
        doc = contract([clause("a", "采购合同签署日2026年10月26日。"), clause("b", "2026年10月27日付款，验收按原约定。")],
                       dates=[element("d1", "2026-10-26", kind="date", cid="a"), element("d2", "2026-10-27", kind="date", cid="b")])
        self.assertFalse(any(r.extra.get("issue_code") == "impossible_deadline" for r in detectors.detect_dates(doc)))
        positive = contract([clause("a", "开发服务期限自2026年10月26日至2026年10月27日，完成全部开发工作。")])
        self.assertTrue(any(r.extra.get("issue_code") == "impossible_deadline" for r in detectors.detect_dates(positive)))

    def test_unit_price_and_contract_total_are_not_comparable_obligations(self):
        doc = contract([clause("unit", "手套单价6.30元。"), clause("total", "合同总价4305080.26元。")],
                       amounts=[element("a", 6.3, role="unit", cid="unit"), element("b", 4305080.26, role="total", cid="total")])
        self.assertFalse(any(r.extra.get("issue_code") == "fee_mismatch" for r in detectors.detect_amounts(doc)))

    def test_explicit_quantity_times_price_large_mismatch_still_triggers(self):
        doc = contract([clause("calc", "数量为10件，单价为100元，总价为200000元。")],
                       amounts=[element("u", 100, role="unit", cid="calc"), element("t", 200000, role="total", cid="calc")])
        issue = next(r for r in detectors.detect_amounts(doc) if r.extra.get("issue_code") == "fee_mismatch")
        self.assertEqual(issue.clause_ids, ["calc"])
        self.assertEqual(issue.extra["amount_binding"]["expected_total"], 1000)

    def test_amount_exclusions_and_ocr_table_context_remain_unasserted(self):
        doc = contract([clause("calc", "数量为10件，单价为100元，总价为200000元。")],
                       amounts=[element("u", 100, role="unit", cid="calc", numeric_comparison_eligible=False)])
        self.assertFalse(any(r.extra.get("issue_code") == "fee_mismatch" for r in detectors.detect_amounts(doc)))
        doc.document.clauses[0].metadata["table_group_id"] = "ocr_table"
        doc.elements.amounts = []
        self.assertFalse(any(r.extra.get("issue_code") == "fee_mismatch" for r in detectors.detect_amounts(doc)))

    def test_unquantified_guarantee_cannot_borrow_goods_line_amount(self):
        guarantee = clause("bond", "合同在乙方递交履约保证金后开始生效。", source_id="original")
        doc = contract([guarantee, clause("goods", "雨伞货款3405600元。", source_id="original"),
                        clause("total", "合同总价4305080.26元。", source_id="original")],
                       amounts=[element("x", 3405600, cid="goods"), element("y", 4305080.26, role="total", cid="total")])
        self.assertEqual(rules.rule_abnormal_price(guarantee, rules.RuleContext.build(doc)), [])

    def test_explicit_deposit_uses_same_source_total_and_original_threshold(self):
        guarantee = clause("bond", "履约保证金为6000元。", source_id="original")
        total = clause("total", "本合同总金额为10000元。", source_id="original")
        unrelated = clause("other", "合同总价1000000元。", source_id="different")
        doc = contract([guarantee, total, unrelated])
        risks = rules.rule_abnormal_price(guarantee, rules.RuleContext.build(doc))
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].extra["amount_binding"]["contract_total"]["value"], 10000)
        self.assertEqual(risks[0].clause_ids, ["bond", "total"])
        guarantee.text = "履约保证金为5000元。"
        self.assertEqual(rules.rule_abnormal_price(guarantee, rules.RuleContext.build(doc)), [])

    def test_ambiguous_totals_and_excluded_deposit_cannot_be_compared(self):
        bond = clause("bond", "履约保证金为6000元。")
        doc = contract([bond, clause("one", "原合同总价10000元。"), clause("two", "调整后合同总价20000元。")])
        self.assertEqual(rules.rule_abnormal_price(bond, rules.RuleContext.build(doc)), [])
        doc.document.clauses.pop()
        doc.elements.amounts = [element("d", 6000, role="deposit", cid="bond", numeric_comparison_eligible=False)]
        self.assertEqual(rules.rule_abnormal_price(bond, rules.RuleContext.build(doc)), [])

    def delivery_fixture(self, *, date_present=True, conditional=True, same_package=True):
        origin = clause("original", "交货时间：2026年10月27日前。" if date_present else "交货时间：年月日前。",
                        source_id="original", source_role="original_contract", package_id="package" if same_package else "unrelated")
        retained = clause("retained", "交货时间、交货地点及质量标准仍按原合同执行。", source_id="supplement",
                          source_role="supplemental_agreement", package_id="package", article_group_id="delivery")
        target = clause("target", "如因数量调整导致交货时间需要变更的，双方另行协商确定，并以书面形式确认。" if conditional else
                        "交货时间另行协商确定，并以书面形式确认。", source_id="supplement",
                        source_role="supplemental_agreement", package_id="package", article_group_id="delivery")
        return target, contract([origin, retained, target])

    def test_conditional_amendment_resolves_against_retained_original_date(self):
        target, doc = self.delivery_fixture()
        self.assertEqual(rules.rule_vague_delivery(target, rules.RuleContext.build(doc)), [])
        audit = target.metadata["rule_binding_audit"]["vague_delivery"]
        self.assertEqual(audit["retention_clause_ids"], ["retained"])
        self.assertEqual(audit["original_delivery_evidence"][0]["dates"], ["2026-10-27"])

    def test_missing_date_unconditional_change_or_other_package_keeps_vagueness_risk(self):
        for options in ({"date_present": False}, {"conditional": False}, {"same_package": False}):
            with self.subTest(options=options):
                target, doc = self.delivery_fixture(**options)
                self.assertEqual(len(rules.rule_vague_delivery(target, rules.RuleContext.build(doc))), 1)

    def test_real_report_replay_removes_four_misbound_claims_without_editing_source(self):
        path = Path(__file__).resolve().parents[2] / "tools/package_core_delivery_20261121/final_profile_run/real_report_readback.json"
        if not path.exists():
            self.skipTest("Archived real procurement report is not installed in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc = report_contract(payload)
        original = [(c.clause_id, c.text) for c in doc.document.clauses]
        self.assertTrue(any("超过租赁期限二十年" in r["message"] for r in payload["risks"]))
        dates = detectors.detect_dates(doc)
        amounts = detectors.detect_amounts(doc)
        context = rules.RuleContext.build(doc)
        abnormal = [r for c in context.clauses for r in rules.rule_abnormal_price(c, context)]
        vague = [r for c in context.clauses for r in rules.rule_vague_delivery(c, context)]
        self.assertFalse(any(r.extra.get("issue_code") == "term_over_20y" for r in dates))
        self.assertFalse(any(r.extra.get("issue_code") == "fee_mismatch" for r in amounts))
        self.assertFalse(any("3405600" in r.message for r in abnormal))
        conditional = next(c for c in doc.document.clauses if "如因数量调整导致交货时间需要变更" in c.text)
        self.assertFalse(any(conditional.clause_id in r.clause_ids for r in vague))
        self.assertEqual(original, [(c.clause_id, c.text) for c in doc.document.clauses])
        self.assertIn("rule_binding_audit", conditional.metadata)


if __name__ == "__main__":
    unittest.main()
