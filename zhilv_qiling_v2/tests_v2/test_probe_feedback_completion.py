"""Receipt accounting is execution evidence only, never legal acceptance."""
import copy
import unittest

from tools.probe_feedback_completion import summarize_feedback


class ProbeFeedbackAccountingTests(unittest.TestCase):
    def item(self, **changes):
        value = {"work_item_id": "work-one", "kind": "statute_feedback",
                 "execution_status": "executed", "actual_processed": True,
                 "receipt_call_ids": ["bound-real-receipt-fixture"],
                 "protocol_complete": False, "pending": True,
                 "semantic_status": "pending_manual", "pending_reasons": ["verification_needs_manual"]}
        value.update(changes)
        return value

    def test_status_without_bound_receipt_is_not_complete(self):
        for changes in ({"actual_processed": False}, {"receipt_call_ids": []},
                        {"execution_status": "attempted_without_valid_result", "actual_processed": False,
                         "receipt_call_ids": [], "pending_reasons": ["bound_processing_receipt_missing_or_invalid"]}):
            with self.subTest(changes=changes):
                summary = summarize_feedback({"items": [self.item(**changes)]})
                self.assertFalse(summary["attempt_complete"])
                self.assertFalse(summary["execution_coverage_complete"])
                self.assertEqual(summary["current_fingerprint_attempted_work_items"], 0)

    def test_valid_manual_response_is_processed_but_remains_open(self):
        item = self.item()
        original = copy.deepcopy(item)
        summary = summarize_feedback({"items": [item]})
        self.assertTrue(summary["attempt_complete"])
        self.assertEqual(summary["pending_after_response"], 1)
        self.assertEqual(summary["protocol_complete_count"], 0)
        self.assertEqual(item, original)

    def test_current_fingerprint_missing_never_complete(self):
        summary = summarize_feedback({"items": [self.item(pending_reasons=["current_fingerprint_not_attempted"])]})
        self.assertTrue(summary["execution_coverage_complete"])
        self.assertFalse(summary["attempt_complete"])

    def test_deferred_cannot_inherit_processed_status(self):
        summary = summarize_feedback({"items": [self.item(), self.item(work_item_id="next-cycle",
            execution_status="deferred_next_cycle", actual_processed=False, receipt_call_ids=[])]})
        self.assertFalse(summary["attempt_complete"])
        self.assertEqual(summary["pending_after_response"], 2)
        self.assertEqual(summary["current_fingerprint_attempted_work_items"], 1)
        self.assertEqual(summary["current_fingerprint_not_attempted_work_items"], 1)
        self.assertEqual(summary["explicit_current_fingerprint_not_attempted_reason_count"], 0)

    def test_empty_feedback_does_not_manufacture_completion(self):
        self.assertFalse(summarize_feedback({"items": []})["attempt_complete"])


if __name__ == "__main__":
    unittest.main()
