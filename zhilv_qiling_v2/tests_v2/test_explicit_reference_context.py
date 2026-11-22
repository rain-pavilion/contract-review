"""Anchored long-clause references; all fixtures run locally without a model."""
from __future__ import annotations

import unittest

from zhilv_qiling_v2 import long_context as C
from zhilv_qiling_v2.models import Clause


def clause(cid, text, label=""):
    return Clause(cid, label, "", 1, "", text, text, ["block_" + cid])


def long_reference():
    # Previously both cross-clause sides selected this topic-rich head and
    # omitted the actual reference + contradictory final amounts at the tail.
    head = "价款总价金额付款支付结算预付款尾款交付交货验收履行期限完成时间交期违约赔偿免责责任损失授权委托代理签署代表人资质许可解除终止撤销退还退款。"
    head = (head + "背景约定。" * 50)[:200]
    left_quote = "依第2条结算，甲方应支付的最终价款固定为十万元。"
    right_quote = "甲方应支付的最终结算价款固定为二十万元。"
    clauses = [clause("a", head + "普通说明。" * 70 + left_quote, "第1条"),
               clause("b", head + "普通说明。" * 70 + right_quote, "第2条")]
    return clauses, left_quote, right_quote


def complete(windows):
    return {window.window_id for window in windows}


class ExplicitReferenceContextTest(unittest.TestCase):
    def test_long_tails_meet_and_every_source_target_segment_is_checked(self):
        clauses, left, right = long_reference()
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=100)
        cross = [window for window in windows if window.kind == "cross_clause"]
        self.assertTrue(any(any(left in part.text for part in window.segments)
                            and any(right in part.text for part in window.segments) for window in cross))
        self.assertTrue(plan["explicit_reference_plan_complete"])
        dependency = plan["explicit_reference_dependencies"][0]
        self.assertEqual(dependency["reference_text"], "第2条")
        self.assertGreater(dependency["source_anchor"]["start"], 0)
        self.assertGreater(dependency["required_checks"], len(C.split_clause(clauses[1], 200, 0)))
        self.assertEqual(dependency["scheduled_checks"], dependency["required_checks"])
        self.assertEqual(dependency["pending_source_segments"], [])
        self.assertEqual(dependency["pending_target_segments"], [])
        for window in cross:
            self.assertIn(dependency["dependency_id"], window.as_dict()["dependency_ids"])
            self.assertTrue(any(part.clause_id == "a" and part.start <= dependency["reference_start"]
                                and part.end >= dependency["reference_end"] for part in window.segments))
            self.assertLessEqual(sum(len(part.text) for part in window.segments), 600)
        coverage = C.coverage(clauses, windows, complete(windows))
        self.assertTrue(coverage["text_complete"])
        self.assertTrue(coverage["explicit_reference_complete"])
        self.assertTrue(coverage["complete"])

    def test_budget_exhaustion_is_pending_despite_full_primary_text_coverage(self):
        clauses, _, _ = long_reference()
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=1)
        cross = [window for window in windows if window.kind == "cross_clause"]
        self.assertEqual(len(cross), 1)
        self.assertEqual(plan["cross_windows"], 1)
        self.assertEqual(plan["cross_window_budget"], 1)
        self.assertEqual(plan["cross_budget_unit"], "bounded_cross_clause_windows")
        self.assertFalse(plan["explicit_reference_plan_complete"])
        self.assertGreater(plan["explicit_reference_pending_checks"], 0)
        dependency = plan["explicit_reference_dependencies"][0]
        self.assertTrue(any(part["end"] == len(clauses[1].text) for part in dependency["pending_target_segments"]))
        # Even the sole budget-limited window must carry the real source anchor.
        self.assertTrue(any("第2条" in part.text and part.clause_id == "a" for part in cross[0].segments))
        coverage = C.coverage(clauses, windows, complete(windows))
        self.assertEqual(coverage["text_coverage"], 1)
        self.assertTrue(coverage["text_complete"])
        self.assertFalse(coverage["explicit_reference_complete"])
        self.assertFalse(coverage["complete"])
        self.assertGreater(coverage["explicit_reference_pending_checks"], 0)

    def test_zero_cross_budget_cannot_claim_explicit_reference_complete(self):
        clauses, _, _ = long_reference()
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=0)
        self.assertFalse(any(window.kind == "cross_clause" for window in windows))
        self.assertFalse(plan["explicit_reference_plan_complete"])
        coverage = C.coverage(clauses, windows, complete(windows))
        self.assertTrue(coverage["text_complete"])
        self.assertFalse(coverage["complete"])

    def test_failed_cross_call_does_not_become_complete_via_primary_windows(self):
        clauses, _, _ = long_reference()
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=100)
        self.assertTrue(plan["explicit_reference_plan_complete"])
        # Remove the unique joint check for the first source segment and last
        # target segment, while keeping every primary call successful.
        missing = next(window for window in windows if window.kind == "cross_clause"
                       and any(part.clause_id == "a" and part.start == 0 for part in window.segments)
                       and any(part.clause_id == "b" and part.end == len(clauses[1].text) for part in window.segments))
        coverage = C.coverage(clauses, windows, complete(windows) - {missing.window_id})
        self.assertTrue(coverage["text_complete"])
        self.assertFalse(coverage["explicit_reference_complete"])
        self.assertGreater(coverage["explicit_reference_pending_checks"], 0)
        self.assertFalse(coverage["complete"])

    def test_reference_token_straddling_chunk_boundary_retains_actual_anchor(self):
        source = clause("a", "甲" * 199 + "第2条" + "付款以约定为准。" * 25, "第1条")
        target = clause("b", "乙方应按约支付对应价款。", "第2条")
        windows, plan = C.build_review_plan([source, target], chunk_chars=200, overlap=0, cross_pairs=100)
        dependency = plan["explicit_reference_dependencies"][0]
        self.assertEqual(dependency["reference_start"], 199)
        for window in (w for w in windows if w.reason == "explicit_reference"):
            self.assertTrue(any(part.clause_id == "a" and part.start <= 199 and part.end >= 202
                                and "第2条" in part.text for part in window.segments))
            self.assertLessEqual(sum(len(part.text) for part in window.segments), 600)
        self.assertTrue(C.coverage([source, target], windows, complete(windows))["complete"])

    def test_repeated_references_share_one_real_window_without_duplicate_call_ids(self):
        clauses = [clause("a", "付款按第2条执行，结算也依第2条办理。", "第1条"),
                   clause("b", "甲方应于验收后十日支付价款。", "第2条")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, overlap=0, cross_pairs=1)
        cross = [w for w in windows if w.kind == "cross_clause"]
        self.assertEqual(len(cross), 1)
        self.assertEqual(len({w.window_id for w in windows}), len(windows))
        self.assertEqual(len(cross[0].dependency_ids), 2)
        self.assertEqual(len(plan["explicit_reference_dependencies"]), 2)
        self.assertTrue(plan["explicit_reference_plan_complete"])
        self.assertTrue(C.coverage(clauses, windows, complete(windows))["complete"])

    def test_heading_titles_chinese_arabic_aliases_and_decimal_labels_resolve(self):
        for left_label, right_label, reference in (("第一条 付款安排", "第二条 结算", "第2条"),
                                                  ("第1条 付款安排", "第2条 结算", "第二条"),
                                                  ("1.1 合同事项", "2.1 结算", "2.1")):
            with self.subTest(reference=reference):
                clauses = [clause("a", f"价款支付依{reference}执行。", left_label),
                           clause("b", "甲方收到交付物后支付价款。", right_label)]
                windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=1)
                self.assertEqual(len(plan["explicit_reference_dependencies"]), 1)
                self.assertEqual(plan["explicit_reference_dependencies"][0]["target_clause_id"], "b")
                self.assertTrue(C.coverage(clauses, windows, complete(windows))["complete"])

    def test_ambiguous_target_labels_remain_pending_instead_of_choosing_one(self):
        clauses = [clause("a", "依第2条执行。", "第1条"), clause("b", "内容乙。", "第2条"),
                   clause("c", "内容丙。", "第2条")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=12)
        self.assertFalse(plan["explicit_reference_plan_complete"])
        self.assertEqual(plan["explicit_reference_ambiguous_targets"][0]["candidate_target_clause_ids"], ["b", "c"])
        coverage = C.coverage(clauses, windows, complete(windows))
        self.assertTrue(coverage["text_complete"])
        self.assertFalse(coverage["explicit_reference_complete"])
        self.assertFalse(coverage["complete"])

    def test_unresolved_article_or_explicit_decimal_is_pending_without_invented_id(self):
        for text, reference in (("付款依第99条执行。", "第99条"), ("付款按9.9约定办理。", "9.9")):
            with self.subTest(reference=reference):
                clauses = [clause("a", text, "第1条"), clause("b", "交付物为设备。", "第2条")]
                windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=12)
                self.assertEqual(plan["explicit_reference_dependencies"], [])
                self.assertFalse(plan["explicit_reference_plan_complete"])
                pending = plan["explicit_reference_unresolved_targets"]
                self.assertEqual(pending[0]["reference_text"], reference)
                self.assertEqual(pending[0]["candidate_target_clause_ids"], [])
                self.assertEqual(pending[0]["status"], "pending_unresolved_target")
                coverage = C.coverage(clauses, windows, complete(windows))
                self.assertTrue(coverage["text_complete"])
                self.assertFalse(coverage["explicit_reference_complete"])
                self.assertFalse(coverage["complete"])

    def test_unresolved_external_law_article_is_not_fabricated_as_contract_id(self):
        clauses = [clause("a", "依据《某法》第99条处理。", "第1条")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=12)
        self.assertEqual(plan["cross_windows"], 0)
        self.assertEqual(plan["explicit_reference_dependencies"], [])
        self.assertIn("external law", plan["explicit_reference_unresolved_targets"][0]["note"])
        self.assertFalse(C.coverage(clauses, windows, complete(windows))["complete"])

    def test_bare_decimal_amount_and_rate_are_not_unresolved_clause_references(self):
        clauses = [clause("a", "违约金每日按价款0.5%计算，设备单价2.50万元。", "第1条")]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=12)
        self.assertEqual(plan["explicit_reference_unresolved_targets"], [])
        self.assertTrue(plan["explicit_reference_plan_complete"])
        self.assertTrue(C.coverage(clauses, windows, complete(windows))["complete"])

    def test_explicit_anchor_has_priority_over_ordinary_topic_pairs(self):
        clauses = [clause("a", "本合同价款十万元。"), clause("b", "本合同付款方式为转账。"),
                   clause("x", "依第9条办理交付。", "第7条"), clause("y", "交付物为两台设备。", "第9条")]
        windows, _ = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=1)
        cross = [w for w in windows if w.kind == "cross_clause"]
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0].reason, "explicit_reference")
        self.assertEqual({s.clause_id for s in cross[0].segments}, {"x", "y"})

    def test_ordinary_topic_checks_remain_bounded_and_explicitly_nonexhaustive(self):
        clauses = [clause(f"c{i}", f"第{i}阶段合同价款与付款时间由双方约定。") for i in range(15)]
        windows, plan = C.build_review_plan(clauses, chunk_chars=200, cross_pairs=2)
        self.assertEqual(sum(w.kind == "cross_clause" for w in windows), 2)
        self.assertEqual(plan["cross_scope"], "targeted_dependency_pairs_not_exhaustive")
        self.assertGreater(plan["topic_pairs_not_scheduled"], 0)
        self.assertEqual(plan["explicit_reference_dependencies"], [])
        self.assertTrue(plan["explicit_reference_plan_complete"])
        self.assertTrue(C.coverage(clauses, windows, complete(windows))["complete"])


if __name__ == "__main__":
    unittest.main()
