"""Versioned source-selection node integration with offline model responses."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.verify_initial_review_lineage import verify as verify_legacy
from tools.verify_initial_source_selection import verify as verify_selection
from zhilv_qiling_v2 import llm_client as L, nodes as N
from zhilv_qiling_v2.initial_source_selection import PROTOCOL_VERSION, replay_source_selection
from zhilv_qiling_v2.tests_v2 import test_initial_review_nodes as legacy_nodes

clause, page = legacy_nodes.clause, legacy_nodes.page


def selected(payload, *, related_count=0, **changes):
    extracts = payload["source_extract_catalogue"]["extracts"]
    finding = {"source_ref": extracts[0]["source_ref"], "code": "fee_mismatch",
        "severity": "medium", "message": "需核对付款条件。",
        "related_source_refs": [row["source_ref"] for row in extracts[1:1 + related_count]]}
    finding.update(changes)
    return finding


class InitialSourceSelectionNodeTests(unittest.TestCase):
    def review(self, response, **settings):
        observed = []
        original_request = L.request_json

        def observe(*args, **kwargs):
            pair = original_request(*args, **kwargs)
            observed.append((pair, copy.deepcopy(pair.content)))
            return pair

        with patch.object(L, "request_json", side_effect=observe):
            result = legacy_nodes.InitialReviewNodeTests().review(response,
                llm_initial_source_protocol=PROTOCOL_VERSION, **settings)
        for pair, snapshot in observed:
            self.assertEqual(pair.content, snapshot, "Product changed the observed original model JSON")
        return result, {pair.call_id: snapshot for pair, snapshot in observed}

    def checked(self, result, observed):
        _, meta, _, _, events, windows, clauses = result
        return verify_selection(meta, windows, clauses, observed, events,
            trace_id="offline-continuation-trace", contract_id="offline-continuation")

    def test_source_refs_are_native_constrained_with_unambiguous_prompt_and_raw_capture(self):
        result, observed = self.review(lambda payload, _: page([selected(payload)]))
        candidates, meta, _, sent, _, windows, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual(len(candidates), 1)
        call = meta["calls"][0]
        self.assertEqual(call["model_output_protocol"], PROTOCOL_VERSION)
        self.assertIn("source_ref", call["initial_raw_response"]["defects"][0])
        self.assertNotIn("quote", call["initial_raw_response"]["defects"][0])
        self.assertIn("quote", meta["initial_review_lineage"][0]["attempts"][0]["content"]["defects"][0])
        system = sent[0]["messages"][0]["content"]
        self.assertIn("source_ref", system)
        self.assertNotIn("related_quotes", system)
        self.assertNotIn('"quote"', system)
        fields = sent[0]["schema"]["properties"]["defects"]["items"]["properties"]
        self.assertEqual(set(fields), {"source_ref", "code", "severity", "message", "related_source_refs"})
        self.assertEqual(fields["source_ref"]["enum"], [x["source_ref"] for x in sent[0]["payload"]["source_extract_catalogue"]["extracts"]])
        self.assertTrue(replay_source_selection(windows[0], observed[call["call_id"]],
                                               call["source_selection"])["replay_verified"])
        self.assertEqual(candidates[0].evidence[0].extra["model_output_protocol"], PROTOCOL_VERSION)
        checked = self.checked(result, observed)
        self.assertTrue(checked["source_selection_lineage_acceptance_passed"], checked)
        self.assertFalse(checked["canonical_projection_is_original_model_json"])
        self.assertFalse(checked["historical_strict_acceptance_established"])

    def test_more_true_stays_pending_and_audited_continuation_uses_same_catalogue(self):
        def response(payload, index):
            return page([selected(payload)], more=True) if index == 1 else page(corrections=[])
        result, observed = self.review(response)
        _, meta, _, sent, _, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["actual_attempt_calls"], 2)
        self.assertTrue(meta["calls"][0]["source_selection"]["raw_content"]["more_findings_possible"])
        self.assertTrue(meta["calls"][0]["source_selection"]["resolved_content"]["more_findings_possible"])
        self.assertIn("initial_output_more_findings_possible", meta["calls"][0]["output_pending_reasons"])
        self.assertEqual(sent[0]["payload"]["segments"], sent[1]["payload"]["segments"])
        self.assertEqual(sent[0]["payload"]["source_extract_catalogue"], sent[1]["payload"]["source_extract_catalogue"])
        self.assertEqual(meta["calls"][1]["source_selection"]["pending_finding_ids"], [])
        checked = self.checked(result, observed)
        self.assertTrue(checked["source_selection_lineage_acceptance_passed"], checked)

    def test_exhausted_more_never_becomes_complete_from_source_binding(self):
        result, observed = self.review(lambda payload, index: page([selected(payload)], more=True,
                                       corrections=[] if index > 1 else None),
                                       llm_initial_max_attempts=2)
        _, meta, degraded, _, _, _, _ = result
        self.assertFalse(meta["complete"])
        self.assertEqual(len(meta["calls"]), 2)
        self.assertTrue(all(call["source_selection"]["all_source_refs_resolved"] for call in meta["calls"]))
        self.assertIn("initial_review_continuation_budget_exhausted", meta["initial_output_pending"][0]["reasons"])
        self.assertTrue(any("llm_text_coverage_incomplete" == reason for reason in degraded))
        self.assertFalse(self.checked(result, observed)["source_selection_lineage_acceptance_passed"])

    def test_related_capacity_needs_explicit_repair_and_original_selected_json_survives(self):
        rows = [clause("a", "主付款条款。"), clause("b", "相关验收条款。"), clause("c", "相关交付条款。")]
        def response(payload, index):
            if index == 1:
                return page([selected(payload, related_count=2)])
            prior_id = payload["initial_review_continuation"]["pending_findings"][0]["finding_id"]
            return page(corrections=[{"prior_finding_id": prior_id, "defect": selected(payload, related_count=1)}])
        result, observed = self.review(response, rows=rows)
        _, meta, _, _, _, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertIn("initial_output_related_capacity_reached", meta["calls"][0]["output_pending_reasons"])
        self.assertEqual(len(meta["calls"][0]["initial_raw_response"]["defects"][0]["related_source_refs"]), 2)
        self.assertTrue(meta["calls"][1]["repaired_finding_ids"])
        self.assertTrue(self.checked(result, observed)["source_selection_lineage_acceptance_passed"])

    def test_old_checker_does_not_silently_accept_the_new_raw_protocol(self):
        result, observed = self.review(lambda payload, _: page([selected(payload)]))
        _, meta, _, _, events, windows, clauses = result
        old = verify_legacy(meta, windows, clauses, events,
            trace_id="offline-continuation-trace", contract_id="offline-continuation")
        self.assertFalse(old["protocol_lineage_acceptance_passed"])
        self.assertFalse(old["checks"]["policy_unchanged"])
        self.assertTrue(self.checked(result, observed)["source_selection_lineage_acceptance_passed"])

    def test_checker_rejects_source_receipt_lineage_and_external_raw_tampering(self):
        original, observed = self.review(lambda payload, _: page([selected(payload)]))
        for mutate in (
            lambda meta: meta["calls"][0]["source_selection"]["bindings"][0].update(clause_start=999),
            lambda meta: meta["calls"][0]["source_selection"]["resolved_content"]["defects"][0].update(quote="伪造引文"),
            lambda meta: meta["initial_review_lineage"][0]["attempts"][0]["content"]["defects"][0].update(message="篡改"),
            lambda meta: meta["calls"][0]["initial_raw_response"]["defects"][0].update(message="篡改"),
            lambda meta: meta["calls"][0]["source_selection"].update(pending_finding_ids=["unseen"]),
            lambda meta: meta["calls"][0].update(source_selection_catalogue_sha256="wrong"),
        ):
            changed = copy.deepcopy(original)
            mutate(changed[1])
            self.assertFalse(self.checked(changed, observed)["source_selection_lineage_acceptance_passed"])
        self.assertFalse(self.checked(original, {})["source_selection_lineage_acceptance_passed"])
        forged_raw = copy.deepcopy(observed)
        next(iter(forged_raw.values()))["more_findings_possible"] = True
        self.assertFalse(self.checked(original, forged_raw)["source_selection_lineage_acceptance_passed"])

    def test_unknown_source_protocol_or_missing_native_mode_makes_no_model_request(self):
        for settings, reason in (({"llm_initial_source_protocol": "unknown"}, "llm_initial_source_protocol_unsupported"),
                                 ({"llm_initial_source_protocol": PROTOCOL_VERSION,
                                   "llm_native_json_schema": False}, "llm_initial_source_selection_requires_native_schema")):
            result = legacy_nodes.InitialReviewNodeTests().review([], **settings)
            _, meta, degraded, captured, _, _, _ = result
            self.assertFalse(meta["complete"])
            self.assertEqual(captured, [])
            self.assertIn(reason, degraded)
            self.assertEqual(meta["initial_source_protocol_error"], reason)

    def test_injected_unknown_reference_survives_into_pending_ledger_without_raw_mutation(self):
        original_content = {"defects": [{"source_ref": "unknown", "code": "fee_mismatch",
            "severity": "medium", "message": "需复核。", "related_source_refs": []}], "more_findings_possible": False}
        pair = SimpleNamespace(content=copy.deepcopy(original_content), model="offline-direct-stub", call_id="offline-direct-stub", degraded="")
        with patch.object(N.L, "request_json", return_value=pair):
            result = legacy_nodes.InitialReviewNodeTests().review([], llm_initial_source_protocol=PROTOCOL_VERSION,
                                                     llm_initial_max_attempts=1)
        _, meta, _, _, _, _, _ = result
        self.assertEqual(pair.content, original_content)
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["initial_review_lineage"][0]["pending_findings"][0]["defect"], original_content["defects"][0])
        self.assertEqual(meta["calls"][0]["initial_raw_response"], original_content)


if __name__ == "__main__":
    unittest.main()
