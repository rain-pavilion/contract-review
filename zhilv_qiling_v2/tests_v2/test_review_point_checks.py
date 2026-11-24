"""Payment screening binds every observation to actual provided text."""
import unittest
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse
from zhilv_qiling_v2.review_point_checks import evaluate_review_points


def fixture(text, warnings=None):
    clauses = [Clause(clause_id='payment-1', label='第一条', title='付款', level=1,
                      heading='付款', body=text, text=text, block_ids=['b1'])]
    return Contract(contract_id='payment-contract', document=DocumentParse(
        source_name='合同.txt', file_type='txt', full_text=text, clauses=clauses, warnings=warnings or []))


class PaymentReviewTests(unittest.TestCase):
    def test_explicit_vague_tail_payment_is_source_bound_finding(self):
        contract = fixture('甲方在验收合格后支付尾款，具体付款期限由双方另行协商。')
        result = evaluate_review_points(contract)[0]
        self.assertIn(result['status'], ('issue_found', 'needs_manual'))
        self.assertTrue(result['manual_required'])
        self.assertIn('期限', result['actual_issue'])
        self.assertIn('付款期限', result['recommendation'])
        self.assertTrue(all(e['quote'] in contract.document.full_text for e in result['evidence']))
        self.assertFalse(result['legal_truth_established'])

    def test_direct_relative_deadline_only_clears_bounded_rule(self):
        result = evaluate_review_points(fixture('甲方应在验收合格后10个工作日内支付全部尾款。'))[0]
        self.assertEqual(result['status'], 'no_issue_detected')
        self.assertEqual(result['scope'], 'provided_contract_text_payment_sentences_only')
        self.assertFalse(result['legal_truth_established'])

    def test_clear_first_instalment_does_not_clear_undated_tail(self):
        result = evaluate_review_points(fixture('甲方在签订后5日内支付首款；尾款支付时间由双方另行协商。'))[0]
        self.assertNotEqual(result['status'], 'no_issue_detected')
        self.assertTrue(result['manual_required'])

    def test_dates_for_other_events_do_not_clear_payment_deadline(self):
        result = evaluate_review_points(fixture('合同签订日期为2026年11月24日，甲方支付尾款六万元。'))[0]
        self.assertNotEqual(result['status'], 'no_issue_detected')

    def test_reference_or_missing_ocr_never_implies_clean_check(self):
        for text, warnings in [
            ('甲方在签订后5日内支付首款，其余按附件付款计划执行。', []),
            ('甲方在验收后5日内支付尾款。', ['部分页面未识别'])]:
            with self.subTest(text=text):
                result = evaluate_review_points(fixture(text, warnings))[0]
                self.assertNotEqual(result['status'], 'no_issue_detected')
                self.assertTrue(result['pending_reasons'])

    def test_absent_payment_text_does_not_pass(self):
        result = evaluate_review_points(fixture('双方应当保守商业秘密。'))[0]
        self.assertEqual(result['status'], 'not_checked')
        self.assertFalse(result['scope_complete'])

    def test_stale_payment_clause_cannot_hide_behind_one_located_clause(self):
        item = fixture('甲方在验收合格后5日内支付首款。')
        item.document.clauses.append(Clause('stale', '第二条', '尾款', 1, '尾款',
            '尾款另行协商。', '尾款另行协商。'))
        result = evaluate_review_points(item)[0]
        self.assertNotEqual(result['status'], 'no_issue_detected')
        self.assertTrue(result['pending_reasons'])

    def test_machine_ocr_without_human_confirmation_cannot_clear_check(self):
        item = fixture('甲方在验收合格后5日内支付尾款。')
        item.document.clauses[0].metadata['extraction_quality'] = {
            'status': 'machine_ocr_unverified', 'human_verified': False}
        result = evaluate_review_points(item)[0]
        self.assertNotEqual(result['status'], 'no_issue_detected')
        self.assertTrue(result['pending_reasons'])

    def test_bad_date_and_unknown_trigger_cannot_clear_payment(self):
        for text in ('甲方于2026年2月30日前支付尾款。', '甲方在条件满足后5日内支付尾款。',
                     '甲方不得在验收合格后5日内支付尾款。'):
            with self.subTest(text=text):
                self.assertNotEqual(evaluate_review_points(fixture(text))[0]['status'], 'no_issue_detected')

    def test_deadline_for_invoice_is_not_a_payment_deadline(self):
        for text in ('甲方在验收后10日内开具发票后择日支付尾款。',
                     '甲方在验收后十日内完成结算后支付尾款。',
                     '甲方在验收后一二三四日内支付尾款。'):
            with self.subTest(text=text):
                self.assertNotEqual(evaluate_review_points(fixture(text))[0]['status'], 'no_issue_detected')

    def test_full_text_payment_sentence_missing_from_clauses_blocks_clean_receipt(self):
        item = fixture('甲方在验收后5日内支付首款。')
        item.document.full_text += '尾款支付时间另行协商。'
        result = evaluate_review_points(item)[0]
        self.assertNotEqual(result['status'], 'no_issue_detected')
        self.assertTrue(result['pending_reasons'])

    def test_payment_heading_is_not_a_separate_undated_obligation(self):
        item = fixture('第二条 付款\n甲方应在验收合格后10个工作日内支付全部尾款。')
        result = evaluate_review_points(item)[0]
        self.assertEqual(result['status'], 'no_issue_detected')


if __name__ == '__main__':
    unittest.main()
