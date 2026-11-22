"""Continuation preserves exact scope, failures and bounded real-attempt lineage."""
import copy
import unittest

from zhilv_qiling_v2.initial_review import (
    InitialReviewContinuation, continuation_json_schema, source_sha256,
)


POLICY = {"max_findings": 8, "message_max_chars": 160,
          "quote_max_chars": 160, "max_related_quotes": 2}


class InitialReviewContinuationTests(unittest.TestCase):
    def setUp(self):
        self.window = {"window_id": "original-window", "kind": "cross_clause", "segments": [
            {"clause_id": "a", "start": 10, "end": 23, "text": "甲方验收完成后支付尾款。"},
            {"clause_id": "b", "start": 20, "end": 35, "text": "乙方收齐全部价款后开始验收。"},
        ]}

    def finding(self, quote="验收完成后支付尾款"):
        return {"clause_id": "a", "code": "circular_dependency", "severity": "high",
                "message": "付款与验收互为前提，需核对履行顺序。", "quote": quote,
                "related_quotes": [{"clause_id": "b", "quote": "收齐全部价款后开始验收"}]}

    def state(self, **kwargs):
        return InitialReviewContinuation(self.window, policy=POLICY, **kwargs)

    def test_more_findings_continuation_keeps_exact_sources_and_seen_findings(self):
        state = self.state(max_attempts=3)
        state.record_attempt({"defects": [self.finding()], "more_findings_possible": True},
                             call_id="real-call-1", job_id="root-job")
        request = state.next_request()
        self.assertEqual(request["payload"]["segments"], self.window["segments"])
        self.assertEqual(request["window_id"], "original-window")
        self.assertEqual(request["source_sha256"], source_sha256(self.window["segments"]))
        context = request["payload"]["initial_review_continuation"]
        self.assertEqual(context["seen_findings"][0]["defect"], self.finding())
        self.assertEqual(context["prior_attempts"][0]["call_id"], "real-call-1")
        self.assertEqual(context["attempt_index"], 2)

    def test_completed_second_page_retains_prior_valid_findings_and_both_calls(self):
        state = self.state()
        state.record_attempt({"defects": [self.finding()], "more_findings_possible": True}, call_id="first")
        final = state.record_attempt({"defects": [], "corrections": [], "more_findings_possible": False}, call_id="second")
        self.assertTrue(final["complete"])
        self.assertEqual(final["accepted_findings"], [self.finding()])
        self.assertEqual([x["call_id"] for x in final["attempts"]], ["first", "second"])
        self.assertIsNone(state.next_request())

    def test_empty_page_cannot_hide_prior_misbound_quote(self):
        state = self.state(max_attempts=3)
        invalid = self.finding("原文中完全不存在的引文")
        state.record_attempt({"defects": [invalid], "more_findings_possible": True}, call_id="first")
        repair_id = state.pending_finding_ids[0]
        second = state.record_attempt({"defects": [], "corrections": [], "more_findings_possible": False}, call_id="second")
        self.assertFalse(second["complete"])
        self.assertEqual(state.pending_finding_ids, [repair_id])
        self.assertEqual(second["attempts"][0]["content"]["defects"], [invalid])

    def test_explicit_corrected_quote_can_resolve_the_named_prior_finding(self):
        state = self.state()
        state.record_attempt({"defects": [self.finding("错误引文")], "more_findings_possible": True}, call_id="first")
        repair_id = state.pending_finding_ids[0]
        final = state.record_attempt({"defects": [], "corrections": [
            {"prior_finding_id": repair_id, "defect": self.finding()}], "more_findings_possible": False},
            call_id="second")
        self.assertTrue(final["complete"])
        self.assertEqual(final["attempts"][1]["repaired_finding_ids"], [repair_id])
        self.assertEqual(final["attempts"][0]["content"]["defects"][0]["quote"], "错误引文")

    def test_invalid_or_unknown_correction_does_not_clear_repair(self):
        for unknown in (True, False):
            with self.subTest(unknown=unknown):
                state = self.state()
                state.record_attempt({"defects": [self.finding("错误引文")], "more_findings_possible": True}, call_id="first")
                repair_id = state.pending_finding_ids[0]
                final = state.record_attempt({"defects": [], "corrections": [{
                    "prior_finding_id": "other-finding" if unknown else repair_id,
                    "defect": self.finding() if unknown else self.finding("仍然错误的引文")}],
                    "more_findings_possible": False}, call_id="second")
                self.assertFalse(final["complete"])
                self.assertEqual(state.pending_finding_ids, [repair_id])

    def test_duplicate_correction_cannot_acknowledge_one_finding_twice(self):
        state = self.state()
        state.record_attempt({"defects": [self.finding("错误引文")], "more_findings_possible": True}, call_id="first")
        correction = {"prior_finding_id": state.pending_finding_ids[0], "defect": self.finding()}
        result = state.record_attempt({"defects": [], "corrections": [correction, correction],
                                       "more_findings_possible": False}, call_id="second")
        self.assertFalse(result["complete"])
        self.assertIn("initial_output_correction_duplicate", result["pending_reasons"])

    def test_budget_limit_keeps_more_findings_pending_without_fabricating_completion(self):
        state = self.state(max_attempts=2)
        response = {"defects": [self.finding()], "more_findings_possible": True}
        state.record_attempt(response, call_id="first")
        final = state.record_attempt(response, call_id="second")
        self.assertFalse(final["complete"])
        self.assertEqual(final["status"], "budget_exhausted")
        self.assertIn("initial_review_continuation_budget_exhausted", final["pending_reasons"])
        self.assertTrue(all(row["content"]["more_findings_possible"] for row in final["attempts"]))
        self.assertIsNone(state.next_request())
        with self.assertRaises(ValueError):
            state.record_attempt(response, call_id="third")

    def test_original_input_and_returned_payload_are_defensive_copies(self):
        original = copy.deepcopy(self.window)
        state = self.state()
        self.window["segments"][0]["text"] = "篡改的原文"
        request = state.next_request()
        self.assertEqual(request["payload"]["segments"], original["segments"])
        request["payload"]["segments"][0]["text"] = "再次篡改"
        self.assertEqual(state.next_request()["payload"]["segments"], original["segments"])

    def test_scope_hash_and_duplicate_call_identity_are_enforced(self):
        state = self.state()
        response = {"defects": [], "more_findings_possible": True}
        for args in ({"window_id": "different"}, {"source_hash": "different"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                state.record_attempt(response, call_id="call", **args)
        self.assertEqual(state.attempts, [])
        state.record_attempt(response, call_id="same-call")
        with self.assertRaises(ValueError):
            state.record_attempt(response, call_id="same-call")
        self.assertEqual(len(state.attempts), 1)

    def test_missing_call_id_transport_truncation_and_external_validation_stay_pending(self):
        for args, reason in (({"call_id": ""}, "initial_review_call_id_missing"),
                             ({"call_id": "call", "transport_partial": True}, "initial_review_transport_partial"),
                             ({"call_id": "call", "pending_reasons": ["schema_validation_failed"]}, "schema_validation_failed")):
            with self.subTest(reason=reason):
                final = self.state(max_attempts=1).record_attempt(
                    {"defects": [], "more_findings_possible": False}, **args)
                self.assertFalse(final["complete"])
                self.assertIn(reason, final["pending_reasons"])

    def test_existing_quote_capacity_boundary_is_not_relaxed(self):
        self.window["segments"][0]["text"] = "甲" * 160
        state = self.state()
        value = self.finding("甲" * 160)
        value["related_quotes"] = []
        result = state.record_attempt({"defects": [value], "more_findings_possible": False}, call_id="call")
        self.assertFalse(result["complete"])
        self.assertIn("initial_output_quote_capacity_reached", result["pending_reasons"])
        self.assertEqual(len(result["pending_findings"]), 1)

    def test_invalid_response_cannot_claim_empty_complete(self):
        for content in (None, {}, {"defects": [], "more_findings_possible": "false"}):
            with self.subTest(content=content):
                result = self.state().record_attempt(content, call_id="call")
                self.assertFalse(result["complete"])
                self.assertIn("initial_output_schema_invalid", result["pending_reasons"])

    def test_continuation_schema_preserves_legal_options_and_binds_only_repair_id(self):
        defect = {"type": "object", "properties": {"code": {"enum": ["A", "B"]}}}
        original = {"required": ["defects", "more_findings_possible"],
                    "properties": {"defects": {"type": "array", "items": defect},
                                   "more_findings_possible": {"type": "boolean"}}}
        untouched = copy.deepcopy(original)
        result = continuation_json_schema(original, ["repair-A"])
        self.assertEqual(original, untouched)
        self.assertEqual(result["properties"]["defects"], original["properties"]["defects"])
        correction = result["properties"]["corrections"]["items"]["properties"]
        self.assertEqual(correction["defect"], defect)
        self.assertEqual(correction["prior_finding_id"]["enum"], ["repair-A"])
        self.assertEqual(result["properties"]["more_findings_possible"], {"type": "boolean"})


if __name__ == "__main__":
    unittest.main()
