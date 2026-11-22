"""Replayable work ledger; execution, source sufficiency and semantics differ.

This ledger observes existing node receipts. It cannot acknowledge legacy
feedback, clear a pending finding, change a source claim, or change a score.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _work_id(contract_id: str, kind: str, identity: Any) -> str:
    return "work_" + fingerprint([contract_id, kind, identity])[:24]


def _success_receipts(call: dict[str, Any]) -> list[dict[str, Any]]:
    """Never treat a bare UUID or another window's receipt as execution proof."""
    return [receipt for receipt in call.get("request_receipts", [])
        if isinstance(receipt, dict) and (receipt.get("status") == "succeeded" or
            (receipt.get("status") == "degraded" and receipt.get("reason") in ("json_retry", "response_format_retry")))
        and bool(call.get("call_id")) and receipt.get("call_id") == call.get("call_id")
        and all(bool(call.get(key)) and receipt.get(key) == call.get(key)
                for key in ("window_id", "job_id", "source_sha256"))]


def _item(contract_id: str, kind: str, identity: Any, inputs: dict[str, Any],
          *, execution: str = "not_executed", source: str = "bound",
          semantic: str = "pending", pending: Iterable[str] = (),
          receipts: Iterable[str] = (), actual_processed: bool = False,
          clause_ids: Iterable[str] = (), dispatchable: bool = True) -> dict[str, Any]:
    return {"work_item_id": _work_id(contract_id, kind, identity), "kind": kind,
        "input_fingerprint": fingerprint(inputs), "inputs": copy.deepcopy(inputs),
        "clause_ids": sorted(set(str(cid) for cid in clause_ids if cid)),
        "execution_status": execution, "source_status": source,
        "semantic_status": semantic, "pending_reasons": sorted(set(pending)),
        "receipt_call_ids": sorted(set(str(cid) for cid in receipts if cid)),
        "actual_processed": bool(actual_processed), "dispatchable": dispatchable,
        "pending": semantic not in ("provided_relation_confirmed", "output_complete"),
        "meaning": "Execution is not source authenticity or legal correctness."}


def initial_work_items(contract_id: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep root/attempt lineage untouched and expose its distinct pending states."""
    result = []
    calls = [call for call in meta.get("calls", []) if isinstance(call, dict)]
    for root in meta.get("initial_review_lineage", []):
        if not isinstance(root, dict):
            continue
        root_id = str(root.get("root_job_id") or root.get("window_id") or "")
        bound = [call for call in calls if str(call.get("root_job_id") or call.get("job_id")) == root_id]
        source_bound = bool(root.get("source_sha256") and root.get("window_id") and bound
            and all(call.get("source_sha256") == root.get("source_sha256")
                    and call.get("window_id") == root.get("window_id") for call in bound))
        receipts = [receipt for call in bound for receipt in _success_receipts(call)
                    if call.get("source_sha256") == root.get("source_sha256")
                    and call.get("window_id") == root.get("window_id")]
        reasons = list(root.get("pending_reasons") or [])
        source_pending = any("source" in str(reason) or "quote" in str(reason) for reason in reasons)
        if not source_bound:
            reasons.append("initial_source_binding_missing_or_mismatched")
        if not receipts:
            reasons.append("initial_bound_execution_receipt_missing")
        complete = bool(root.get("complete") is True and receipts and source_bound
                        and not source_pending and not reasons)
        inputs = {"root_job_id": root_id, "window_id": root.get("window_id"),
            "source_sha256": root.get("source_sha256"),
            "source_spans": [span for call in bound for span in call.get("source_spans", [])],
            "max_attempts": root.get("max_attempts")}
        item = _item(contract_id, "initial_review", root_id, inputs,
            execution="executed" if receipts else "attempted_without_success" if bound else "not_executed",
            source="missing_source_binding" if not source_bound else "citation_binding_pending" if source_pending else "bound",
            semantic="output_complete" if complete else "output_pending",
            pending=reasons or ([] if complete else ["initial_output_not_completed"]),
            receipts=[r.get("call_id", "") for r in receipts], actual_processed=bool(receipts),
            clause_ids=[cid for call in bound for cid in call.get("clause_ids", [])], dispatchable=False)
        item["lineage"] = copy.deepcopy(root)
        item["continuation_owner"] = "node2_initial_review_protocol"
        result.append(item)
    if not result and meta and meta.get("complete") is False:
        result.append(_item(contract_id, "initial_review", "unavailable_lineage",
            {"initial_review_continuation_version": meta.get("initial_review_continuation_version"),
             "planned_windows": meta.get("planned_windows"), "unreviewed_clause_ids": meta.get("unreviewed_clause_ids", [])},
            source="missing_source_binding", semantic="output_pending",
            pending=["initial_lineage_not_available"], dispatchable=False))
    return result


_SOURCE_PENDING_ERRORS = {
    "subject_fact_source_incomplete", "subject_fact_status_unresolved",
    "subject_fact_upstream_request_pending", "subject_fact_verification_needs_manual",
    "feedback_verification_needs_manual", "verification_needs_manual",
}


def feedback_work_items(contract_id: str, statutes: list[dict[str, Any]],
                        facts: list[dict[str, Any]], subject: dict[str, Any],
                        latest: dict[str, Any], consumption: list[dict[str, Any]],
                        clause_sources: dict[str, str]) -> list[dict[str, Any]]:
    records = {str(row.get("feedback_item_id")): row for row in consumption}
    calls = [call for call in latest.get("calls", []) if isinstance(call, dict)]
    result = []
    for original in [*statutes, *facts]:
        item = copy.deepcopy(original)
        legacy_id = str(item.get("feedback_item_id") or "")
        fact = item.get("feedback_kind") == "subject_fact"
        kind = "subject_fact" if fact else "statute_feedback"
        identity_fields = (("subject_node_id", "subject_id", "subject_name", "subject_role", "fact_type", "fact_id", "clause_id", "request_id")
                           if fact else ("candidate_id", "article_uid", "clause_ids"))
        identity = {key: item.get(key) for key in identity_fields}
        clause_ids = list(item.get("clause_ids") or [])
        # Exact evidence and the legacy item ID remain input-bound. Changed IDs,
        # quotes, fact dates or upstream results cannot reuse old processing.
        stable_feedback = copy.deepcopy(item)
        stable_feedback.pop("feedback_item_id", None)
        # Per-execution NLI audit UUIDs/timings are observations, not new source
        # evidence. Keep the original below, but do not create a new obligation
        # merely because the exact same model input was audited again.
        def without_transport(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: without_transport(child) for key, child in value.items()
                        if key not in ("call_id", "elapsed_seconds")}
            if isinstance(value, list):
                return [without_transport(child) for child in value]
            return value
        if isinstance(stable_feedback.get("window_audit"), dict):
            stable_feedback["window_audit"] = without_transport(stable_feedback["window_audit"])
        inputs = {"feedback": stable_feedback, "contract_clause_sources": {
            cid: clause_sources.get(cid) for cid in sorted(clause_ids)}}
        record = records.get(legacy_id, {})
        bound = [call for call in calls if legacy_id in call.get("feedback_item_ids", [])]
        receipts = []
        valid_rows = []
        source_pending = bool(fact and (not item.get("source_complete") or not item.get("source_id")
            or item.get("effective_status") in (None, "", "unknown", "conflicting_evidence") or item.get("upstream_pending")))
        for call in bound:
            rows = [row for row in call.get("subject_fact_verifications" if fact else "verifications", [])
                    if isinstance(row, dict) and row.get("feedback_item_id") == legacy_id]
            errors = {str(error.get("reason")) for error in call.get("verification_errors", [])
                      if isinstance(error, dict) and error.get("feedback_item_id") in (legacy_id, "", None)}
            binding_fields = (("fact_id", "source_id", "subject_name", "fact_type", "clause_id")
                              if fact else ("candidate_id", "article_uid"))
            actual_receipts = _success_receipts(call)
            if (actual_receipts and len(rows) == 1
                    and all(rows[0].get(key) == item.get(key) for key in binding_fields)
                    and rows[0].get("verdict") in ("confirmed", "rejected", "needs_manual")
                    and (not source_pending or rows[0].get("verdict") == "needs_manual")
                    and not (errors - _SOURCE_PENDING_ERRORS)
                    and call.get("status") not in ("error", "partial", "invalid_schema")):
                valid_rows.extend(rows)
                receipts.extend(actual_receipts)
        processed = bool(receipts and valid_rows)
        reasons = list(record.get("validation_errors") or []) + list(record.get("upstream_review_reasons") or [])
        if source_pending:
            reasons.append("external_subject_evidence_pending")
        wait_for_evidence = bool(fact and source_pending and item.get("auto_retry") is False)
        if wait_for_evidence:
            reasons.append("private_evidence_unavailable_wait_for_new_evidence")
        if not processed:
            reasons.append("bound_processing_receipt_missing_or_invalid")
        semantic = str(record.get("semantic_status") or "pending_manual")
        if source_pending or not processed:
            semantic = "pending_manual"
        work = _item(contract_id, kind, identity, inputs,
            execution="executed" if processed else "attempted_without_valid_result" if bound else "not_executed",
            source="external_evidence_pending" if source_pending else "bound_provided_evidence",
            semantic=semantic, pending=reasons,
            receipts=[r.get("call_id", "") for r in receipts], actual_processed=processed, clause_ids=clause_ids,
            dispatchable=not wait_for_evidence)
        if wait_for_evidence:
            work.update(auto_retry=False, retry_trigger="new_evidence", unavailable_evidence=item.get("unavailable_evidence", []))
        work["legacy_feedback_item_id"] = legacy_id
        work["provided_feedback"] = item
        work["provided_feedback_fingerprint"] = fingerprint(item)
        work["protocol_complete"] = record.get("protocol_complete") is True
        result.append(work)
    for request_id in sorted(set(str(value) for value in subject.get("pending_verification_request_ids", []) if value)):
        result.append(_item(contract_id, "subject_request", request_id,
            {"request_id": request_id, "verification_results": [row for row in subject.get("verification_results", [])
                if isinstance(row, dict) and row.get("request_id") == request_id]},
            source="external_evidence_pending", semantic="pending_manual",
            pending=["subject_verification_request_pending"], dispatchable=False))
    return result


class ReviewWorkLedger:
    """Append-only observations; a reused task retains all unresolved axes."""
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def observe(self, items: list[dict[str, Any]], round_no: int) -> None:
        current_ids = {item["work_item_id"] for item in items}
        for old in self.items.values():
            if old.get("current") and old["work_item_id"] not in current_ids:
                old["current"] = False
                self.events.append({"seq": len(self.events) + 1, "round": round_no,
                    "work_item_id": old["work_item_id"], "observation": copy.deepcopy(old)})
        for original in items:
            item = copy.deepcopy(original)
            item["current"] = True
            old = self.items.get(item["work_item_id"])
            if old != item:
                self.events.append({"seq": len(self.events) + 1, "round": round_no,
                    "work_item_id": item["work_item_id"], "observation": copy.deepcopy(item)})
            self.items[item["work_item_id"]] = item

    def snapshot(self) -> dict[str, Any]:
        current = sorted((copy.deepcopy(item) for item in self.items.values() if item.get("current")),
                         key=lambda item: item["work_item_id"])
        return {"version": "review-work-items-v1", "items": current,
            "pending_work_item_ids": [item["work_item_id"] for item in current if item["pending"]],
            "events": copy.deepcopy(self.events),
            "meaning": "Historical observations do not acknowledge or replace legacy pending ledgers."}

    @classmethod
    def replay(cls, events: list[dict[str, Any]]) -> "ReviewWorkLedger":
        ledger = cls()
        for expected, event in enumerate(events, 1):
            if event.get("seq") != expected or not isinstance(event.get("observation"), dict):
                raise ValueError("invalid_work_item_event_sequence")
            item = copy.deepcopy(event["observation"])
            if item.get("work_item_id") != event.get("work_item_id"):
                raise ValueError("work_item_event_identity_mismatch")
            ledger.items[item["work_item_id"]] = item
            ledger.events.append(copy.deepcopy(event))
        return ledger
