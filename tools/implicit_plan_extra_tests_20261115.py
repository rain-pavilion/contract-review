"""Product regression tests for conservative numeric candidate priorities.

These fixtures have no legal truth labels and never call a model.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2.long_context import build_review_plan
from zhilv_qiling_v2.models import Clause


def candidates(left, right):
    clauses = [Clause('a', '第一条', '', 1, '', left, left),
               Clause('b', '第二条', '', 1, '', right, right)]
    _, plan = build_review_plan(clauses, chunk_chars=300, cross_pairs=4)
    return plan.get('obligation_value_candidates', [])


class ConservativeValueCandidates(unittest.TestCase):
    def test_before_tax_and_after_tax_are_different_bases(self):
        self.assertEqual(candidates('本合同税前合同总价人民币10000元。',
                                    '本合同税后合同总价人民币11000元。'), [])

    def test_same_explicit_tax_basis_can_propose_literal_difference(self):
        self.assertEqual(len(candidates('本合同税前合同总价人民币10000元。',
                                         '本合同不含税合同总价人民币11000元。')), 1)
        self.assertEqual(len(candidates('本合同税后合同总价人民币10000元。',
                                         '本合同含税合同总价人民币11000元。')), 1)

    def test_example_or_hypothetical_is_not_promoted_as_a_payment_obligation(self):
        for marker in ('仅作示例', '例如', '假设', '演示', '样例'):
            with self.subTest(marker=marker):
                self.assertEqual(candidates('本项目尾款人民币70000元' + marker + '。',
                                            '本项目尾款人民币80000元' + marker + '。'), [])

    def test_example_or_hypothetical_is_not_promoted_as_a_date_obligation(self):
        for marker in ('仅作示例', '例如', '假设', '演示', '样例'):
            with self.subTest(marker=marker):
                self.assertEqual(candidates('本项目最终交付日期为2026年12月1日' + marker + '。',
                                            '本项目最终交付日期为2027年1月15日' + marker + '。'), [])

    def test_unambiguous_real_slot_is_not_removed(self):
        rows = candidates('本合同尾款人民币70000元。', '本合同尾款人民币80000元。')
        self.assertEqual(len(rows), 1)
        self.assertIs(rows[0]['legal_conflict_proven'], False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
