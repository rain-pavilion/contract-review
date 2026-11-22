"""No-issue observations are auditable status data, never substantive defects.

Offline stubs verify routing and retained uncertainty, not legal accuracy.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2.initial_review import InitialReviewContinuation, continuation_json_schema
from zhilv_qiling_v2.long_context import build_review_plan
from zhilv_qiling_v2.models import RISK_CODES
from zhilv_qiling_v2.tests_v2 import test_initial_native_output as fixtures


def status(**changes):
    return fixtures.finding(code="no_substantive_risk", severity="info",
                            message="本窗口未确认实质缺陷。", **changes)


class NoIssueObservationTests(unittest.TestCase):
    def review(self, content, **kwargs):
        return fixtures.InitialNativeOutputTests().review(content, **kwargs)

    def test_legacy_status_retains_raw_and_call_sources_without_creating_candidate(self):
        observed = status()
        values, meta, _, _, _, contract, config = self.review({"defects": [observed]}, native=False)
        self.assertEqual(values, [])
        self.assertEqual(meta["accepted"], 0)
        self.assertEqual(meta["no_issue_observations"][0]["raw_observation"], observed)
        audit = meta["no_issue_observations"][0]
        self.assertEqual(audit["call_id"], meta["calls"][0]["call_id"])
        self.assertEqual(audit["source_sha256"], meta["calls"][0]["source_sha256"])
        self.assertTrue(audit["source_binding_checks"][0]["quote_exact_source_match"])
        self.assertFalse(audit["establishes_review_completion"])
        self.assertEqual(meta["calls"][0]["no_issue_observations"], meta["no_issue_observations"])
        retriever = Mock()
        N.node_statute(contract, values, config, retriever=retriever)
        retriever.search.assert_not_called()

    def test_mixed_legacy_page_preserves_true_risk_and_only_it_reaches_statute(self):
        true_finding = fixtures.finding()
        values, meta, _, _, _, contract, config = self.review(
            {"defects": [status(), true_finding]}, native=False)
        self.assertEqual([candidate.code for candidate in values], [true_finding["code"]])
        self.assertEqual(meta["accepted"], 1)
        self.assertEqual(len(meta["no_issue_observations"]), 1)
        retriever = Mock()
        retriever.search.return_value = SimpleNamespace(hits=[])
        with patch.object(N, "A_RETRIEVE_PLAN", return_value="only-real-risk") as plan:
            N.node_statute(contract, values, config, retriever=retriever)
        self.assertEqual(plan.call_args.args[2].code, true_finding["code"])
        retriever.search.assert_called_once()

    def test_unbound_legacy_status_is_audited_as_unbound_without_inventing_evidence(self):
        observed = status(quote="发送原文没有这一段。")
        values, meta, _, _, _, _, _ = self.review({"defects": [observed]}, native=False)
        self.assertEqual(values, [])
        self.assertEqual(meta["no_issue_observations"][0]["raw_observation"], observed)
        self.assertFalse(meta["no_issue_observations"][0]["source_binding_checks"][0]["quote_exact_source_match"])

    def test_native_direct_sentinel_stays_invalid_pending_even_with_false_more_flag(self):
        values, meta, _, _, _, _, _ = self.review(
            {"defects": [status()], "more_findings_possible": False}, direct=True)
        self.assertEqual(values, [])
        self.assertFalse(meta["complete"])
        self.assertIn("initial_output_schema_invalid", meta["initial_output_pending"][0]["reasons"])
        self.assertTrue(meta["initial_review_lineage"][0]["pending_findings"])
        self.assertEqual(len(meta["no_issue_observations"]), 1)

    def test_native_transport_rejects_status_sentinel_and_prompt_keeps_status_out_of_codes(self):
        values, meta, _, sent, _, _, _ = self.review(
            {"defects": [status()], "more_findings_possible": False})
        self.assertEqual(values, [])
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["calls"][0]["status"], "error")
        self.assertNotIn("no_substantive_risk", sent[0]["messages"][0]["content"])
        self.assertIn("no_substantive_risk", RISK_CODES)  # final no-risk display remains available

    def test_status_observation_cannot_clear_old_wrong_quote_repair_or_more_flag(self):
        rows = [fixtures.clause("alpha", "约定付款条件应与验收记录一致。")]
        windows, _ = build_review_plan(rows, chunk_chars=6000, overlap=0, cross_pairs=0)
        state = InitialReviewContinuation(windows[0].as_dict(), policy=N.LOGIC_OUTPUT_POLICY,
                                          allowed_codes=N.LOGIC_DEFECT_CODES)
        original = fixtures.finding(quote="历史错误引文。")
        state.record_attempt({"defects": [original], "more_findings_possible": True}, call_id="old")
        repair_id = state.pending_finding_ids[0]
        response = {"defects": [], "corrections": [{"prior_finding_id": repair_id, "defect": status()}],
                    "more_findings_possible": False}
        untouched = copy.deepcopy(response)
        final = state.record_attempt(response, call_id="next")
        self.assertEqual(response, untouched)
        self.assertFalse(final["complete"])
        self.assertEqual(state.pending_finding_ids, [repair_id])
        self.assertEqual(final["attempts"][0]["content"]["defects"], [original])
        self.assertEqual(final["attempts"][1]["repaired_finding_ids"], [])
        schema = continuation_json_schema(N._logic_output_json_schema(windows[0]), [repair_id])
        correction_codes = schema["properties"]["corrections"]["items"]["properties"]["defect"]["properties"]["code"]["enum"]
        self.assertNotIn("no_substantive_risk", correction_codes)

    def test_direct_more_flag_remains_pending_when_sentinel_is_routed_to_status(self):
        _, meta, _, _, _, _, _ = self.review(
            {"defects": [status()], "more_findings_possible": True}, direct=True)
        self.assertFalse(meta["complete"])
        self.assertIn("initial_output_more_findings_possible", meta["initial_output_pending"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
