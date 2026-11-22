"""Explicit-run draft tests for the UNAPPLIED implicit planning patch.

Run: python -B tools/implicit_plan_tests_20261115_draft.py
The proposal is applied to an in-memory module only. This file makes no model,
network or server requests and never writes product source or validation runs.
These diagnostic examples are engineering tests, not human legal-quality gold.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import importlib
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import long_context as CURRENT
from zhilv_qiling_v2.models import Clause

PRODUCT = ROOT / 'zhilv_qiling_v2/long_context.py'
BEFORE_SHA256 = hashlib.sha256(PRODUCT.read_bytes()).hexdigest()


def candidate_module():
    original = PRODUCT.read_text(encoding='utf-8').splitlines(keepends=True)
    proposal = (ROOT / 'tools/implicit_plan_patch_20261115.patch').read_text(encoding='utf-8').splitlines(keepends=True)
    result, position, active = [], 0, False
    for line in proposal:
        if line.startswith(('--- ', '+++ ')):
            continue
        if line.startswith('@@'):
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
            if not match:
                raise AssertionError('malformed patch header')
            start = int(match.group(1)) - 1
            if start < position:
                raise AssertionError('overlapping or reordered patch hunks')
            result.extend(original[position:start]); position = start; active = True
        elif active and line.startswith(' '):
            if original[position] != line[1:]:
                raise AssertionError('proposal context drift; do not apply blindly')
            result.append(line[1:]); position += 1
        elif active and line.startswith('-'):
            if original[position] != line[1:]:
                raise AssertionError('proposal removal drift')
            position += 1
        elif active and line.startswith('+'):
            result.append(line[1:])
        elif line.startswith('\\ No newline'):
            continue
        else:
            raise AssertionError('unexpected patch line')
    if not active:
        raise AssertionError('empty proposal')
    result.extend(original[position:])
    name = 'zhilv_qiling_v2._unapplied_implicit_plan_candidate'
    module = types.ModuleType(name)
    module.__package__ = 'zhilv_qiling_v2'
    sys.modules[name] = module
    exec(compile(''.join(result), '<unapplied-implicit-plan-candidate>', 'exec'), module.__dict__)
    return module


CANDIDATE = candidate_module()


def clause(cid, text, label=''):
    return Clause(cid, label, '', 1, '', text, text, ['block_' + cid])


def plan_texts(left, right, **kwargs):
    return CANDIDATE.build_review_plan([clause('left', left), clause('right', right)], **kwargs)


def differences(windows):
    return [window for window in windows if window.reason.startswith('shared_obligation_value_difference:')]


def crosses(windows):
    return [window for window in windows if window.kind == 'cross_clause']


class UnappliedPlanDraftTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(PRODUCT.read_bytes()).hexdigest() != BEFORE_SHA256:
            raise AssertionError('product source changed while draft tests ran')

    def test_reciprocal_reference_dependencies_share_one_real_window(self):
        rows = [clause('one', '按第二条支付价款。', '第一条'),
                clause('two', '依据第一条完成付款。', '第二条')]
        windows, plan = CANDIDATE.build_review_plan(rows, cross_pairs=1)
        self.assertEqual(plan['cross_windows'], 1)
        self.assertEqual(len(plan['explicit_reference_dependencies']), 2)
        self.assertTrue(plan['explicit_reference_plan_complete'])
        self.assertEqual(len(crosses(windows)[0].dependency_ids), 2)
        self.assertTrue(CANDIDATE.coverage(rows, windows, {w.window_id for w in windows})['complete'])

    def test_cross_identity_uses_unordered_complete_segment_set_only(self):
        a = CANDIDATE.Segment('a', 0, 3, '甲乙丙')
        b = CANDIDATE.Segment('b', 4, 7, '丁戊己')
        other_range = CANDIDATE.Segment('b', 7, 10, '庚辛壬')
        self.assertEqual(CANDIDATE._window('cross_clause', [a, b]).window_id,
                         CANDIDATE._window('cross_clause', [b, a, a]).window_id)
        self.assertNotEqual(CANDIDATE._window('cross_clause', [a, b]).window_id,
                            CANDIDATE._window('cross_clause', [a, other_range]).window_id)
        self.assertNotEqual(CANDIDATE._window('text', [a, b]).window_id,
                            CANDIDATE._window('text', [b, a]).window_id)

    def test_multisegment_reciprocal_references_have_no_duplicate_segment_sets(self):
        rows = [clause('a', '普通材料。' * 90 + '按第二条支付。', '第一条'),
                clause('b', '依据第一条结算。' + '接收记录。' * 100, '第二条')]
        windows, plan = CANDIDATE.build_review_plan(rows, chunk_chars=200, overlap=0, cross_pairs=100)
        sets = [frozenset((s.clause_id, s.start, s.end) for s in w.segments) for w in crosses(windows)]
        self.assertEqual(len(sets), len(set(sets)))
        self.assertTrue(plan['explicit_reference_plan_complete'])
        self.assertTrue(CANDIDATE.coverage(rows, windows, {w.window_id for w in windows})['complete'])

    def test_primary_windows_and_full_text_coverage_remain_unchanged(self):
        rows = [clause('a', '付款材料。' * 100 + '本项目尾款为70000元。'),
                clause('b', '交付记录。' * 100 + '本项目尾款为80000元。')]
        old, _ = CURRENT.build_review_plan(rows, chunk_chars=300)
        new, _ = CANDIDATE.build_review_plan(rows, chunk_chars=300)
        self.assertEqual([asdict(w) for w in old if w.kind == 'text'],
                         [asdict(w) for w in new if w.kind == 'text'])
        self.assertTrue(CANDIDATE.coverage(rows, new, {w.window_id for w in new})['text_complete'])

    def test_distant_payment_values_anchor_the_actual_tail_not_topic_rich_head(self):
        head = '价款总价金额付款支付结算预付款尾款交付交货验收违约赔偿责任。'
        rows = [clause('a', head + '资料归档记录。' * 150 + '本项目尾款人民币70000元。'),
                clause('b', head + '材料接收记录。' * 150 + '本项目尾款人民币80000元。')]
        windows, plan = CANDIDATE.build_review_plan(rows, chunk_chars=300, cross_pairs=1)
        targeted = differences(windows)
        self.assertEqual(len(targeted), 1)
        self.assertTrue(all(segment.start > 600 for segment in targeted[0].segments))
        self.assertTrue(any('70000元' in segment.text for segment in targeted[0].segments))
        self.assertTrue(any('80000元' in segment.text for segment in targeted[0].segments))
        self.assertEqual(plan['obligation_value_windows_pending'], 0)

    def test_multiple_amount_grammars_and_named_scope_styles_are_supported(self):
        examples = [
            ('合同总价：人民币1,280,000.00元。', '本合同合同总金额为130万元。', 'payment:total'),
            ('设备“冷链一号”尾款为人民币12.5万元。', '设备「冷链一号」尾款：人民币130000元。', 'payment:tail'),
            ('系统CRM-2首期款金额为35,000元。', '系统CRM-2首期款：人民币4万元。', 'payment:first_instalment'),
            ('服务A预付款为8000元。', '服务A预付款金额为9500元。', 'payment:advance'),
            ('“数据迁移”项目质保金：1000元。', '项目“数据迁移”质保金共计2000元。', 'payment:retention'),
        ]
        for left, right, slot in examples:
            with self.subTest(left=left, right=right):
                windows, plan = plan_texts(left, right, cross_pairs=1)
                self.assertEqual(len(differences(windows)), 1)
                self.assertIn(slot, differences(windows)[0].reason)
                self.assertEqual(plan['obligation_value_windows_candidates'], 1)

    def test_multiple_date_grammars_events_and_milestones_are_supported(self):
        examples = [
            ('项目“云迁移”交货日期为2027年03月09日。', '项目“云迁移”交付时间：2027-03-16。', 'date:delivery:unspecified'),
            ('设备A于2028/2/29前完成最终交付。', '设备A最终交货截止日期为2028年3月1日。', 'date:delivery:final'),
            ('本系统初步验收应在2026年12月1日。', '本系统初步验收日期定为2026-12-03。', 'date:acceptance:preliminary'),
            ('本项目首批交货时间为2026年12月5日。', '本项目第1批交付日期为2026/12/7。', 'date:delivery:batch:1'),
        ]
        for left, right, slot in examples:
            with self.subTest(left=left, right=right):
                windows, plan = plan_texts(left, right, cross_pairs=1)
                self.assertEqual(len(differences(windows)), 1)
                self.assertIn(slot, differences(windows)[0].reason)
                self.assertEqual(plan['obligation_value_windows_pending'], 0)

    def test_equal_normalized_amounts_and_dates_do_not_get_difference_priority(self):
        examples = [
            ('本项目尾款人民币7万元。', '本项目尾款人民币70,000.00元。'),
            ('合同总价为1亿元。', '合同总金额为10000万元。'),
            ('项目A交货日期为2026年12月01日。', '项目A交付日期为2026-12-1。'),
            ('本设备首批交付日期为2026/12/1。', '本设备第一批交货日期为2026年12月1日。'),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                windows, plan = plan_texts(left, right)
                self.assertEqual(differences(windows), [])
                self.assertEqual(plan['obligation_value_windows_candidates'], 0)

    def test_different_payment_slots_tax_bases_events_and_milestones_stay_distinct(self):
        examples = [
            ('本项目预付款30000元。', '本项目尾款70000元。'),
            ('本系统首付款为1000元。', '本系统首期款为2000元。'),
            ('本合同含税合同总价10000元。', '本合同不含税合同总价9000元。'),
            ('设备A交付日期为2026年12月1日。', '设备A验收日期为2026年12月15日。'),
            ('本项目最终交付日期为2026年12月1日。', '本项目交付日期为2026年12月15日。'),
            ('项目A第一批交货日期为2026年12月1日。', '项目A第二批交货日期为2026年12月15日。'),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                windows, _ = plan_texts(left, right)
                self.assertEqual(differences(windows), [])

    def test_different_or_ambiguous_object_scopes_do_not_cross_bind(self):
        examples = [
            ('项目A尾款70000元。', '项目B尾款80000元。'),
            ('设备甲尾款70000元。', '设备乙尾款80000元。'),
            ('项目“海河治理”尾款70000元。', '项目“海河维护”尾款80000元。'),
            ('系统CRM尾款70000元。', '系统crm尾款80000元。'),
            ('本项目设备A及设备B尾款70000元。', '设备A尾款80000元。'),
            ('本项目尾款70000元。', '本合同尾款80000元。'),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                windows, _ = plan_texts(left, right)
                self.assertEqual(differences(windows), [])

    def test_unknown_scope_bounds_and_currency_cause_conservative_abstention(self):
        examples = [
            ('尾款70000元。', '尾款80000元。'),
            ('项目海河尾款70000元。', '项目海河尾款80000元。'),
            ('本项目尾款人民币10000元为上限。', '本项目尾款人民币8000元。'),
            ('本合同尾款为100美元。', '本合同尾款为200美元。'),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                windows, _ = plan_texts(left, right)
                self.assertEqual(differences(windows), [])

    def test_date_ranges_invalid_dates_mixed_events_and_unrelated_dates_abstain(self):
        examples = [
            ('本项目交付期限为2026年12月1日至2026年12月5日。', '本项目交付日期为2026年12月8日。'),
            ('设备A交货日期为2027年2月29日。', '设备A交货日期为2027年3月1日。'),
            ('本项目交付和验收日期为2026年12月1日。', '本项目交付日期为2026年12月8日。'),
            ('本项目交付违约金自2026年12月1日起计算。', '本项目交付日期为2026年12月8日。'),
            ('本项目首批与第二批交付日期为2026年12月1日。', '本项目交付日期为2026年12月8日。'),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                windows, _ = plan_texts(left, right)
                self.assertEqual(differences(windows), [])

    def test_date_values_crossing_a_primary_boundary_get_a_real_literal_anchor(self):
        rows = [clause('a', '甲' * 190 + '。本项目交付日期为2026年12月1日。'),
                clause('b', '乙' * 190 + '。本项目交付日期为2026年12月8日。')]
        windows, plan = CANDIDATE.build_review_plan(rows, chunk_chars=200, overlap=0, cross_pairs=1)
        self.assertEqual(len(differences(windows)), 1)
        self.assertTrue(all(s.start > 0 for s in differences(windows)[0].segments))
        self.assertTrue(all(any(s.clause_id == value['clause_id'] and s.start <= value['start']
                                 and s.end >= value['end'] for s in differences(windows)[0].segments)
                            for value in plan['obligation_value_candidates'][0]['values']))

    def test_priority_is_explicit_then_literal_value_then_generic_topic(self):
        rows = [clause('a', '合同价款付款事项另行约定。'), clause('b', '付款结算材料由双方保存。'),
                clause('x', '按第九条办理交付。', '第七条'), clause('y', '交付物为设备。', '第九条'),
                clause('z1', '项目A尾款人民币10000元。'), clause('z2', '项目A尾款人民币20000元。')]
        one, plan = CANDIDATE.build_review_plan(rows, cross_pairs=1)
        self.assertEqual(crosses(one)[0].reason, 'explicit_reference')
        self.assertEqual(plan['obligation_value_windows_pending'], 1)
        two, plan = CANDIDATE.build_review_plan(rows, cross_pairs=2)
        self.assertEqual([w.reason for w in crosses(two)][0], 'explicit_reference')
        self.assertEqual(len(differences(two)), 1)
        self.assertEqual({s.clause_id for s in differences(two)[0].segments}, {'z1', 'z2'})
        self.assertEqual(plan['obligation_value_windows_pending'], 0)

    def test_explicit_and_value_candidate_same_segments_do_not_spend_two_calls(self):
        rows = [clause('a', '按第二条结算；本项目尾款10000元。', '第一条'),
                clause('b', '本项目尾款20000元。', '第二条')]
        windows, plan = CANDIDATE.build_review_plan(rows, cross_pairs=1)
        self.assertEqual(len(crosses(windows)), 1)
        self.assertEqual(crosses(windows)[0].reason, 'explicit_reference')
        self.assertEqual(plan['obligation_value_windows_candidates'], 1)
        self.assertEqual(plan['obligation_value_windows_scheduled'], 1)
        self.assertEqual(plan['obligation_value_windows_pending'], 0)

    def test_budget_zero_is_pending_and_never_claims_cross_plan_complete(self):
        windows, plan = plan_texts('项目A尾款10000元。', '项目A尾款20000元。', cross_pairs=0)
        self.assertEqual(crosses(windows), [])
        self.assertEqual(plan['obligation_value_windows_candidates'], 1)
        self.assertEqual(plan['obligation_value_windows_pending'], 1)
        self.assertFalse(plan['cross_plan_complete'])
        self.assertEqual(plan['cross_scope'], 'targeted_dependency_pairs_not_exhaustive')

    def test_multiple_value_reasons_in_one_segment_set_retain_auditable_sources(self):
        rows = [clause('a', '本项目尾款10000元；本项目交付日期为2026年12月1日。'),
                clause('b', '本项目尾款20000元；本项目交付日期为2026年12月8日。')]
        windows, plan = CANDIDATE.build_review_plan(rows, cross_pairs=3)
        self.assertEqual(len(crosses(windows)), 1)
        self.assertEqual(plan['obligation_value_windows_candidates'], 1)
        self.assertEqual(len(plan['obligation_value_candidates']), 2)
        originals = {row.clause_id: row.text for row in rows}
        for item in plan['obligation_value_candidates']:
            self.assertTrue(item['candidate_only'])
            self.assertIs(item['legal_conflict_proven'], False)
            self.assertNotIn('verdict', item)
            for value in item['values']:
                self.assertEqual(originals[value['clause_id']][value['start']:value['end']], value['source_value'])
                self.assertLessEqual(value['anchor']['start'], value['start'])
                self.assertGreaterEqual(value['anchor']['end'], value['end'])

    def test_same_clause_value_changes_are_not_promoted_to_cross_clause_checks(self):
        rows = [clause('a', '本项目尾款10000元；本项目尾款20000元。')]
        windows, plan = CANDIDATE.build_review_plan(rows)
        self.assertEqual(differences(windows), [])
        self.assertEqual(plan['obligation_value_candidates'], [])

    def test_planning_is_deterministic_and_bounded(self):
        rows = [clause('a', '本项目尾款10000元。'), clause('b', '本项目尾款20000元。'),
                clause('c', '本项目尾款30000元。')]
        first, left = CANDIDATE.build_review_plan(rows, chunk_chars=200, cross_pairs=2)
        second, right = CANDIDATE.build_review_plan(rows, chunk_chars=200, cross_pairs=2)
        self.assertEqual([asdict(w) for w in first], [asdict(w) for w in second])
        self.assertEqual(left, right)
        self.assertLessEqual(len(crosses(first)), 2)
        self.assertTrue(all(sum(len(s.text) for s in w.segments) <= 600 for w in crosses(first)))
        self.assertGreater(left['obligation_value_windows_pending'], 0)

    def test_existing_reference_regression_suite_against_in_memory_candidate(self):
        module = importlib.import_module('zhilv_qiling_v2.tests_v2.test_explicit_reference_context')
        original = module.C
        try:
            module.C = CANDIDATE
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.ExplicitReferenceContextTest)
            result = unittest.TestResult()
            suite.run(result)
            self.assertEqual(result.errors, [], str(result.errors))
            self.assertEqual(result.failures, [], str(result.failures))
            self.assertGreaterEqual(result.testsRun, 10)
        finally:
            module.C = original


if __name__ == '__main__':
    unittest.main(verbosity=2)
