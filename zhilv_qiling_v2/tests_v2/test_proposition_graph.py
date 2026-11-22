"""Combinations retain conditions and unknowns, never infer legal truth."""
import copy
import unittest

from zhilv_qiling_v2.fact_slots import extract_fact_slots
from zhilv_qiling_v2.proposition_graph import build_proposition_graph


class PropositionGraphTests(unittest.TestCase):
    def graph(self, *texts, **options):
        return build_proposition_graph([extract_fact_slots(str(index), text) for index, text in enumerate(texts)], **options)

    def test_total_and_tail_are_never_blindly_added_or_compared(self):
        graph = self.graph("甲方支付本合同总价100000元，尾款70000元。", scope_complete=True)
        self.assertTrue(graph["non_additive_amount_role_pairs"])
        self.assertFalse(graph["relations"])
        self.assertTrue(graph["manual_required"])
        self.assertEqual(graph["status"], "unknown")

    def test_only_explicit_fully_bound_partition_permits_component_arithmetic(self):
        for tail, consistent in ((70000, True), (80000, False)):
            graph = self.graph(f"甲方应支付本合同总价100000元，由预付款30000元和尾款{tail}元组成。")
            relation = next(row for row in graph["relations"] if row["kind"] == "explicit_advance_tail_partition")
            self.assertIs(relation["arithmetic_consistent"], consistent)
            self.assertEqual(relation["component_sum_yuan"], str(30000 + tail))
            self.assertFalse(relation["legal_conflict_established"])

    def test_unknown_subject_or_condition_prevents_partition_confirmation(self):
        graph = self.graph("本合同总价100000元，由预付款30000元和尾款80000元组成。")
        self.assertFalse(graph["relations"])
        self.assertTrue(any(row["kind"] == "explicit_partition" for row in graph["unknowns"]))

    def test_same_literal_role_needs_same_subject_object_and_stage(self):
        good = self.graph("甲方支付本合同尾款70000元。", "甲方支付本合同尾款80000元。")
        self.assertTrue(any(row["kind"] == "same_role_literal_value_difference" for row in good["relations"]))
        different_actor = self.graph("甲方支付本合同尾款70000元。", "乙方支付本合同尾款80000元。")
        self.assertFalse(different_actor["relations"])
        self.assertTrue(different_actor["unknowns"])

    def test_explicit_prerequisite_cycle_is_literal_only_not_legal_deadlock(self):
        graph = self.graph("甲方在乙方完成本合同最终验收后支付尾款70000元。",
                           "乙方在甲方支付本合同尾款70000元后完成最终验收。")
        self.assertTrue(any(row["resolved"] for row in graph["edges"]))
        cycles = [row for row in graph["relations"] if row["kind"] == "literal_prerequisite_cycle"]
        self.assertEqual(len(cycles), 1, graph)
        self.assertFalse(cycles[0]["legal_deadlock_established"])
        self.assertIsNone(cycles[0]["conditions_satisfied"])
        self.assertTrue(graph["manual_required"])

    def test_exception_negation_and_unparsed_cross_window_scope_stay_manual(self):
        graph = self.graph("除非甲方书面同意，乙方不得交付本合同设备。", "乙方收到全部材料后支付尾款70000元。",
                           unresolved_cross_window_relations=["article-9"])
        self.assertTrue(graph["manual_required"])
        self.assertIn("unexamined_cross_window_scope", graph["manual_reasons"])
        self.assertIn("unresolved_cross_window_relations", graph["manual_reasons"])
        self.assertTrue(any(row["reason"] == "exception_or_negation_retained_not_flattened" for row in graph["unknowns"]))
        self.assertFalse(graph["automatic_legal_confirmation_allowed"])

    def test_tampered_source_span_fails_closed(self):
        document = extract_fact_slots("c", "甲方支付本合同总价100000元。")
        changed = copy.deepcopy(document)
        changed["amounts"][0]["source_span"]["quote"] = "伪造金额"
        graph = build_proposition_graph([changed], scope_complete=True)
        self.assertEqual(graph["invalid_source_clause_ids"], ["c"])
        self.assertEqual(graph["amount_slot_ids"], [])
        self.assertEqual(graph["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
