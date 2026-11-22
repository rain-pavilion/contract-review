"""Explicit v1 adapter for source-selection receipts and canonical lineage.

The historical checker and historical raw model format are unchanged.  This
checker first binds externally retained model JSON to deterministic source
selection receipts.  It then labels and checks an explicit canonical projection
using the old continuation algorithm.  It cannot certify legal correctness,
feedback closure, or passage of the project's older strict acceptance suite.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from tools.verify_initial_review_lineage import POLICY, replay, verify as verify_canonical
from zhilv_qiling_v2.initial_source_selection import PROTOCOL_VERSION, replay_source_selection


VERSION = "initial-source-selection-lineage-v1"
EXPECTED_POLICY = {**POLICY, "version": "logic-native-source-selection-v1",
    "model_output_protocol": PROTOCOL_VERSION, "source_extract_chars": 80,
    "source_extract_stride_chars": 40,
    "lineage_content": "derived_canonical_with_separate_raw_model_response"}


def verify(meta: dict[str, Any], windows: list[dict[str, Any]], clauses: list[dict[str, Any]],
           raw_responses: Mapping[str, Any], events=None, *, trace_id=None,
           contract_id=None) -> dict[str, Any]:
    """Require original model JSON retained outside the product's derived meta.

    ``raw_responses`` maps actual call IDs to JSON observed at request_json's
    return or decoded from the original transport capture.  Constructing it from
    ``meta.calls.source_selection`` would not provide independent raw evidence.
    A caller must separately establish the capture files' provenance/integrity.
    """
    checks = {"version_and_unchanged_capacity_policy": True,
        "external_raw_responses_bound_to_calls": True,
        "source_receipts_independently_replay": True,
        "source_receipts_match_derived_lineage": True,
        "canonical_projection_lineage_verified": False}
    issues: list[str] = []

    def fail(key, reason):
        checks[key] = False
        issues.append(reason)

    if (meta.get("initial_model_output_protocol") != PROTOCOL_VERSION
            or meta.get("initial_output_policy") != EXPECTED_POLICY
            or meta.get("initial_review_lineage_content") != "derived_canonical_with_separate_raw_model_response"):
        fail("version_and_unchanged_capacity_policy", "source_selection_protocol_or_policy_mismatch")
    window_map = {window.get("window_id"): window for window in windows if isinstance(window, dict)}
    calls = meta.get("calls", [])
    if not isinstance(calls, list):
        calls = []
        fail("external_raw_responses_bound_to_calls", "calls_invalid")
    calls_by_job = {call.get("job_id"): call for call in calls if isinstance(call, dict)}
    call_ids = {call.get("call_id") for call in calls if isinstance(call, dict)}
    if not isinstance(raw_responses, Mapping) or set(raw_responses) - call_ids:
        fail("external_raw_responses_bound_to_calls", "external_raw_map_invalid_or_has_unknown_calls")
        raw_responses = {} if not isinstance(raw_responses, Mapping) else raw_responses
    rows = []
    lineages = meta.get("initial_review_lineage", [])
    if not isinstance(lineages, list):
        lineages = []
        fail("source_receipts_match_derived_lineage", "lineages_invalid")
    for lineage in lineages:
        window = window_map.get(lineage.get("window_id")) if isinstance(lineage, dict) else None
        if window is None:
            fail("source_receipts_match_derived_lineage", "lineage_has_no_prepared_source_window")
            continue
        attempts = lineage.get("attempts", [])
        if not isinstance(attempts, list):
            fail("source_receipts_match_derived_lineage", "lineage_attempts_invalid")
            continue
        for index, attempt in enumerate(attempts):
            call = calls_by_job.get(attempt.get("job_id")) if isinstance(attempt, dict) else None
            if not call or call.get("model_output_protocol") != PROTOCOL_VERSION:
                fail("version_and_unchanged_capacity_policy", "attempt_call_missing_or_wrong_protocol")
                continue
            if call.get("output_policy") != EXPECTED_POLICY:
                fail("version_and_unchanged_capacity_policy", "call_source_policy_mismatch")
            receipt = call.get("source_selection")
            if receipt is None and call.get("status") == "error" and attempt.get("content") is None:
                rows.append({"call_id": call.get("call_id"), "status": "recorded_request_failure_no_derived_output"})
                continue
            if not isinstance(receipt, dict):
                fail("source_receipts_independently_replay", "successful_response_missing_source_receipt")
                continue
            if call.get("source_selection_catalogue_sha256") != receipt.get("catalogue_sha256"):
                fail("source_receipts_match_derived_lineage", "call_catalogue_hash_differs_from_receipt")
            call_id = call.get("call_id")
            if call_id not in raw_responses:
                fail("external_raw_responses_bound_to_calls", "external_original_model_json_missing:" + str(call_id))
                continue
            raw = raw_responses[call_id]
            if call.get("initial_raw_response") != raw or receipt.get("raw_content") != raw:
                fail("external_raw_responses_bound_to_calls", "original_model_json_changed:" + str(call_id))
            pending_ids = receipt.get("pending_finding_ids")
            expected_pending = replay(attempts[:index], window)["pending_finding_ids"] if index else None
            if ((index == 0 and pending_ids is not None)
                    or index > 0 and (not isinstance(pending_ids, list)
                        or any(not isinstance(value, str) for value in pending_ids)
                        or len(set(pending_ids)) != len(pending_ids)
                        or sorted(pending_ids) != expected_pending)):
                fail("source_receipts_match_derived_lineage", "source_correction_ids_disagree_with_prior_ledger:" + str(call_id))
                continue
            checked = replay_source_selection(window, raw, receipt, pending_finding_ids=pending_ids)
            rows.append({"call_id": call_id, **checked})
            if not checked["replay_verified"]:
                fail("source_receipts_independently_replay", "source_receipt_replay_failed:" + str(call_id))
            if (attempt.get("content") != receipt.get("resolved_content")
                    or any(reason not in attempt.get("pending_reasons", []) for reason in checked["pending_reasons"])):
                fail("source_receipts_match_derived_lineage", "derived_content_or_pending_reasons_changed:" + str(call_id))
    # This explicit, labelled projection is NOT the original raw-response
    # protocol.  The old checker is called unchanged, and its version is never
    # claimed as proof that source_ref was an old free-generated quote.
    canonical_projection = copy.deepcopy(meta)
    canonical_projection["initial_output_policy"] = copy.deepcopy(POLICY)
    canonical_report = verify_canonical(canonical_projection, windows, clauses, events,
        trace_id=trace_id, contract_id=contract_id)
    checks["canonical_projection_lineage_verified"] = canonical_report["protocol_lineage_acceptance_passed"]
    if not checks["canonical_projection_lineage_verified"]:
        issues.append("canonical_projection_failed_independent_continuation_checks")
    return {"version": VERSION, "checks": checks,
        "source_selection_lineage_acceptance_passed": all(checks.values()),
        "issues": list(dict.fromkeys(issues)), "source_receipt_replays": rows,
        "canonical_projection_check": canonical_report,
        "canonical_projection_is_original_model_json": False,
        "historical_strict_acceptance_established": False,
        "raw_capture_file_integrity_evaluated": False,
        "legal_accuracy_evaluated": False, "subject_authenticity_evaluated": False,
        "scope": "Versioned source-selection bindings and initial continuation only; not full legal/feedback acceptance"}
