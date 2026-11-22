"""All current obligations remain visible until bound feedback acknowledges them."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi_run as G


class FeedbackPendingLedgerTests(unittest.TestCase):
    def request(self, *, key="current", subject=None, laws=None, facts=None):
        return (key, [], subject or {}, {"clause-a"},
                [{"feedback_item_id": "law-a"}, {"feedback_item_id": "law-b"}] if laws is None else laws,
                [{"feedback_item_id": "fact-a"}] if facts is None else facts)

    def pending(self, request, shared=None):
        ctx = SimpleNamespace(shared={} if shared is None else shared)
        with patch.object(G, "_feedback_request", return_value=request):
            return G._feedback_pending(ctx)

    def test_new_fingerprint_lists_unstarted_law_and_fact_obligations(self):
        status = self.pending(self.request())
        self.assertTrue(status["required"])
        self.assertTrue(status["pending"])
        self.assertEqual(status["item_ids"], ["law-a", "law-b", "fact-a"])

    def test_partial_current_acknowledgements_leave_all_other_items_pending(self):
        shared = {"feedback_latest": {"current": {
            "verified_feedback_item_ids": ["law-a"],
            "uncompleted_feedback_item_ids": ["law-b"],
            "unmatched_feedback_item_ids": ["unmatched-current"],
        }}}
        before = copy.deepcopy(shared)
        status = self.pending(self.request(), shared)
        self.assertEqual(status["item_ids"], ["law-b", "unmatched-current", "fact-a"])
        self.assertTrue(status["pending"])
        self.assertEqual(shared, before)

    def test_unknown_subject_stays_pending_after_law_acknowledgement(self):
        subject = {"verification_complete": False,
                   "pending_verification_request_ids": ["unknown-registry-fact"]}
        shared = {"feedback_latest": {"current": {
            "verified_feedback_item_ids": ["law-a", "law-b"],
            "uncompleted_feedback_item_ids": ["fact-a"],
        }}}
        status = self.pending(self.request(subject=subject), shared)
        self.assertTrue(status["pending"])
        self.assertEqual(status["item_ids"], ["fact-a"])
        self.assertEqual(status["pending_subject_request_ids"], ["unknown-registry-fact"])

    def test_old_fingerprint_receipts_do_not_complete_changed_obligations(self):
        shared = {"processed_feedback": {"old"}, "feedback_latest": {"old": {
            "verified_feedback_item_ids": ["law-a", "law-b", "fact-a"],
            "uncompleted_feedback_item_ids": [], "feedback_complete": True,
        }}}
        status = self.pending(self.request(key="current"), shared)
        self.assertTrue(status["pending"])
        self.assertEqual(status["item_ids"], ["law-a", "law-b", "fact-a"])

    def test_processed_current_fingerprint_has_no_pending_ids(self):
        status = self.pending(self.request(), {"processed_feedback": {"current"}})
        self.assertFalse(status["pending"])
        self.assertEqual(status["item_ids"], [])

    def test_unresolved_request_without_bound_items_is_still_pending(self):
        subject = {"verification_complete": False,
                   "pending_verification_request_ids": ["missing-subject-scope"]}
        status = self.pending(self.request(subject=subject, laws=[], facts=[]))
        self.assertTrue(status["pending"])
        self.assertEqual(status["item_ids"], [])
        self.assertEqual(status["pending_subject_request_ids"], ["missing-subject-scope"])


if __name__ == "__main__":
    unittest.main()
