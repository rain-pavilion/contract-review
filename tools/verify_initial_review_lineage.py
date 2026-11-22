"""Independent, read-only v1 verification of source-bound initial continuations.

This does not alter the legacy strict checker, infer lost receipts, run a model,
or establish legal accuracy. Failure history remains part of the result even
when a later, independently checked attempt completes the same original window.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2.models import RISK_CODES

VERSION = "initial-review-lineage-v1"
POLICY = {"version": "logic-native-output-v1", "max_findings": 8,
          "message_max_chars": 160, "quote_max_chars": 160,
          "max_related_quotes": 2, "token_completion_guaranteed": False}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finding_id(window_id, source_hash, finding):
    return "initial_finding_" + hashlib.sha256(
        (window_id + source_hash + canonical(finding)).encode("utf-8")).hexdigest()[:24]


def finding_valid(finding, sources):
    """Separately replay exact source and unchanged strict capacity checks."""
    if (not isinstance(finding, dict)
            or set(finding) != {"clause_id", "code", "severity", "message", "quote", "related_quotes"}
            or not isinstance(finding.get("code"), str) or finding["code"] not in RISK_CODES
            or finding.get("severity") not in ("high", "medium", "low", "info")):
        return False
    message = finding.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) >= 160:
        return False

    def bound(row):
        return (isinstance(row, dict) and isinstance(row.get("quote"), str)
                and isinstance(row.get("clause_id"), str)
                and bool(row["quote"].strip()) and len(row["quote"]) < 160
                and any(row["quote"] in text for text in sources.get(row.get("clause_id"), [])))

    related = finding.get("related_quotes")
    return bool(bound(finding) and isinstance(related, list) and len(related) < 2
                and all(set(row) == {"clause_id", "quote"} and bound(row)
                        for row in related if isinstance(row, dict))
                and all(isinstance(row, dict) for row in related))


def replay(attempts, window):
    """No imports from the product continuation state or product completion flag."""
    source_hash = sha(window["segments"])
    sources = {}
    for row in window["segments"]:
        sources.setdefault(row["clause_id"], []).append(row["text"])
    pending, accepted, states = {}, {}, []
    complete = False
    for index, attempt in enumerate(attempts, 1):
        content = attempt.get("content")
        valid = (isinstance(content, dict) and isinstance(content.get("defects"), list)
                 and type(content.get("more_findings_possible")) is bool)
        fields = {"defects", "more_findings_possible"} | ({"corrections"} if index > 1 else set())
        valid = valid and set(content) == fields
        defects = content.get("defects", []) if isinstance(content, dict) else []
        corrections = content.get("corrections", []) if isinstance(content, dict) else []
        current_pending = not valid or attempt.get("transport_partial") is True or not attempt.get("call_id")
        if isinstance(defects, list):
            current_pending |= len(defects) >= 8
        else:
            defects = []
            current_pending = True
        if not isinstance(corrections, list):
            corrections = []
            current_pending = True
        correction_counts = Counter(row.get("prior_finding_id") for row in corrections
                                    if isinstance(row, dict) and isinstance(row.get("prior_finding_id"), str))
        for correction in corrections:
            prior = correction.get("prior_finding_id") if isinstance(correction, dict) else None
            defect = correction.get("defect") if isinstance(correction, dict) else None
            if (not isinstance(correction, dict) or set(correction) != {"prior_finding_id", "defect"}
                    or not isinstance(prior, str) or prior not in pending or correction_counts[prior] != 1
                    or not finding_valid(defect, sources)):
                current_pending = True
                continue
            del pending[prior]
            accepted[finding_id(window["window_id"], source_hash, defect)] = defect
        for defect in defects:
            key = finding_id(window["window_id"], source_hash, defect)
            if finding_valid(defect, sources):
                accepted[key] = defect
            else:
                pending[key] = defect
                current_pending = True
        current_pending |= bool(pending) or isinstance(content, dict) and content.get("more_findings_possible") is True
        # An external schema or transport failure is recorded as a failed
        # attempt, and can only be recovered by a new audited request.
        current_pending |= any(reason in attempt.get("pending_reasons", []) for reason in (
            "initial_review_request_failed", "initial_output_schema_invalid", "initial_review_lineage_invalid"))
        complete = not current_pending
        states.append({"attempt_index": index, "complete": complete,
                       "pending_finding_ids": sorted(pending)})
    return {"complete": complete, "pending_finding_ids": sorted(pending),
            "accepted_findings": list(accepted.values()), "states": states}


def verify(meta, windows, clauses, events=None, *, trace_id=None, contract_id=None):
    """Verify original preparation against actual calls; absent evidence fails."""
    checks = {key: True for key in (
        "prepared_sources_match_original_clauses", "policy_unchanged",
        "root_jobs_exactly_match_prepared_windows", "attempts_are_contiguous_and_source_bound",
        "every_actual_call_has_unique_identity", "actual_request_budget_and_counts_match",
        "every_attempt_has_matching_audit_receipts", "completion_and_pending_replay_match",
        "every_prepared_window_completed", "root_completion_counts_match",
        "original_text_coverage_matches_completed_roots")}
    issues = []

    def fail(check, reason):
        checks[check] = False
        issues.append(reason)

    clause_map = {row.get("clause_id"): row.get("text") for row in clauses if isinstance(row, dict)}
    window_map = {row.get("window_id"): row for row in windows if isinstance(row, dict)}
    if (not windows or len(window_map) != len(windows) or any(not isinstance(key, str) or not key for key in window_map)):
        fail("prepared_sources_match_original_clauses", "missing_or_duplicate_prepared_windows")
    for window in window_map.values():
        if not window.get("segments") or window.get("kind") not in ("text", "cross_clause"):
            fail("prepared_sources_match_original_clauses", "prepared_window_has_no_original_segments")
        for segment in window.get("segments", []):
            text = clause_map.get(segment.get("clause_id"))
            start, end = segment.get("start"), segment.get("end")
            if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text) or text[start:end] != segment.get("text")):
                fail("prepared_sources_match_original_clauses", "prepared_segment_not_exact_original:" + str(window["window_id"]))
    if meta.get("initial_output_policy") != POLICY:
        fail("policy_unchanged", "initial_output_policy_missing_or_changed")
    lineages = meta.get("initial_review_lineage", [])
    if not isinstance(lineages, list):
        lineages = []
    lineage_map = {row.get("root_job_id"): row for row in lineages if isinstance(row, dict)}
    expected_roots = {window_id + ":batch:1" for window_id in window_map}
    if set(lineage_map) != expected_roots or len(lineages) != len(expected_roots):
        fail("root_jobs_exactly_match_prepared_windows", "root_manifest_does_not_match_original_preparation")
        fail("every_prepared_window_completed", "missing_or_unproven_prepared_window_lineage")
        fail("completion_and_pending_replay_match", "cannot_replay_missing_original_lineage")
    calls = meta.get("calls", []) if isinstance(meta.get("calls"), list) else []
    calls_by_root = {}
    all_ids, actual_count, recovered_failures = [], 0, []
    starts, terminals = {}, {}
    for event in events or []:
        if event.get("component") != "llm" or event.get("operation") != "request_json":
            continue
        details = event.get("details", {})
        call_id = details.get("call_id") if isinstance(details, dict) else None
        if not call_id:
            continue
        bucket = starts if event.get("status") == "started" else terminals
        if call_id in bucket:
            fail("every_attempt_has_matching_audit_receipts", "duplicate_audit_start_or_terminal:" + call_id)
        bucket[call_id] = event
    for call in calls:
        root = call.get("root_job_id")
        if root not in expected_roots:
            fail("attempts_are_contiguous_and_source_bound", "call_outside_initial_root_plan:" + str(root))
        calls_by_root.setdefault(root, []).append(call)
        all_ids.append(call.get("call_id"))
        count = call.get("actual_attempt_count")
        actual_count += count if type(count) is int else 0
        # Initial continuations deliberately disable hidden parser retries.
        if count != 1 or len(call.get("request_receipts", [])) != 1:
            fail("actual_request_budget_and_counts_match", "initial_attempt_not_exactly_one_real_request:" + str(call.get("job_id")))
        start, terminal = starts.get(call.get("call_id")), terminals.get(call.get("call_id"))
        if not start or not terminal:
            fail("every_attempt_has_matching_audit_receipts", "missing_actual_start_or_terminal:" + str(call.get("call_id")))
            continue
        binding = {key: call.get(key) for key in ("window_id", "job_id", "source_sha256")}
        for event in (start, terminal):
            details = event["details"]
            if (any(details.get(key) != value for key, value in binding.items())
                    or trace_id is not None and event.get("trace_id") != trace_id
                    or contract_id is not None and event.get("contract_id") != contract_id
                    or details.get("native_json_schema_sent") is not True
                    or not isinstance(details.get("json_schema_sha256"), str)
                    or len(details["json_schema_sha256"]) != 64):
                fail("every_attempt_has_matching_audit_receipts", "audit_binding_mismatch:" + str(call.get("call_id")))
        terminal_status = terminal.get("status")
        if terminal_status not in ("succeeded", "degraded", "failed"):
            fail("every_attempt_has_matching_audit_receipts", "invalid_terminal_status")
        if call.get("output_scope_complete") is True and (terminal_status != "succeeded"
                or terminal["details"].get("model_identity_verified") is not True
                or terminal["details"].get("finish_reason") in ("length", "time_limit")):
            fail("every_attempt_has_matching_audit_receipts", "unverified_response_claims_completion")
        if call.get("output_scope_complete") is True and any(event.get("model") != "legalone-8b"
                or event["details"].get("requested_model") != "legalone-8b" for event in (start, terminal)):
            fail("every_attempt_has_matching_audit_receipts", "completed_response_is_not_requested_legalone_8b")
        if terminal_status != "succeeded":
            recovered_failures.append({"call_id": call.get("call_id"), "status": terminal_status,
                                       "reason": terminal.get("reason")})
        receipts = call.get("request_receipts", [])
        if (len(receipts) == 1 and any(receipts[0].get(key) != call.get(key)
                for key in ("call_id", "window_id", "job_id", "source_sha256"))):
            fail("every_attempt_has_matching_audit_receipts", "local_receipt_binding_mismatch")
    if (not all_ids or any(not isinstance(value, str) or not value for value in all_ids)
            or len(set(all_ids)) != len(all_ids)):
        fail("every_actual_call_has_unique_identity", "missing_or_duplicate_call_id")
    audit_initial_ids = {call_id for call_id, event in starts.items()
                         if any(event["details"].get("job_id") == root
                                or str(event["details"].get("job_id", "")).startswith(root + ":continue:")
                                for root in expected_roots)}
    if audit_initial_ids != set(all_ids):
        fail("every_attempt_has_matching_audit_receipts", "initial_audit_request_ids_not_exactly_retained_in_calls")
    budget = meta.get("actual_request_budget")
    if (meta.get("actual_attempt_calls") != actual_count or actual_count != len(calls)
            or type(budget) is not int or budget < 0 or budget and actual_count > budget):
        fail("actual_request_budget_and_counts_match", "actual_attempt_count_does_not_match_evidence")
    completed_roots = 0
    completed_window_ids = set()
    replayed = []
    for root, lineage in lineage_map.items():
        window_id = lineage.get("window_id")
        window = window_map.get(window_id)
        if not window:
            fail("root_jobs_exactly_match_prepared_windows", "unknown_lineage_window")
            continue
        root_calls = calls_by_root.get(root, [])
        attempts = lineage.get("attempts", [])
        max_attempts = lineage.get("max_attempts")
        source_hash = sha(window["segments"])
        expected_spans = [{key: segment[key] for key in ("clause_id", "start", "end")}
                          for segment in window["segments"]]
        if (lineage.get("source_sha256") != source_hash or type(max_attempts) is not int
                or max_attempts < 1 or len(attempts) > max_attempts
                or lineage.get("attempt_count") != len(attempts) or len(root_calls) != len(attempts)):
            fail("attempts_are_contiguous_and_source_bound", "lineage_source_count_or_budget_mismatch:" + root)
        for index, (attempt, call) in enumerate(zip(attempts, root_calls), 1):
            job = root if index == 1 else f"{root}:continue:{index}"
            for row in (attempt, call):
                if (row.get("window_id") != window_id or row.get("source_sha256") != source_hash
                        or row.get("attempt_index") != index or row.get("job_id") != job):
                    fail("attempts_are_contiguous_and_source_bound", "attempt_changed_identity_or_source:" + job)
            if (call.get("source_spans") != expected_spans or attempt.get("call_id") != call.get("call_id")
                    or call.get("root_job_id") != root):
                fail("attempts_are_contiguous_and_source_bound", "attempt_call_manifest_mismatch:" + job)
        result = replay(attempts, window)
        if (lineage.get("complete") is not result["complete"]
                or sorted(row.get("finding_id") for row in lineage.get("pending_findings", [])) != result["pending_finding_ids"]
                or lineage.get("accepted_findings") != result["accepted_findings"]
                or any(attempt.get("complete") is not state["complete"]
                       or call.get("output_scope_complete") is not state["complete"]
                       for attempt, call, state in zip(attempts, root_calls, result["states"]))):
            fail("completion_and_pending_replay_match", "independent_replay_disagrees:" + root)
        if result["complete"]:
            completed_roots += 1
            completed_window_ids.add(window_id)
        else:
            fail("every_prepared_window_completed", "initial_window_still_pending:" + window_id)
        replayed.append({"root_job_id": root, **result})
    if (meta.get("planned_calls") != len(expected_roots) or meta.get("completed_calls") != completed_roots
            or meta.get("completed_windows") != completed_roots):
        fail("root_completion_counts_match", "reported_root_or_coverage_counts_disagree")
    intervals = {}
    for window_id in completed_window_ids:
        window = window_map[window_id]
        if window.get("kind") == "text":
            for segment in window["segments"]:
                intervals.setdefault(segment["clause_id"], []).append((segment["start"], segment["end"]))
    total_chars = sum(len(text) for text in clause_map.values() if isinstance(text, str))
    reviewed_chars = 0
    for spans in intervals.values():
        end = 0
        for left, right in sorted(spans):
            reviewed_chars += max(0, right - max(left, end))
            end = max(end, right)
    text_complete = total_chars == reviewed_chars
    if (meta.get("total_chars") != total_chars or meta.get("reviewed_chars") != reviewed_chars
            or meta.get("text_complete") is not text_complete
            or meta.get("text_coverage") != (round(reviewed_chars / total_chars, 4) if total_chars else 1.0)
            or meta.get("complete") is not (text_complete and meta.get("explicit_reference_complete") is True)):
        fail("original_text_coverage_matches_completed_roots", "independent_text_coverage_disagrees")
    return {"version": VERSION, "checks": checks, "protocol_lineage_acceptance_passed": all(checks.values()),
            "issues": list(dict.fromkeys(issues)), "independent_replay": replayed,
            "actual_attempt_calls": actual_count, "prepared_root_jobs": len(expected_roots),
            "completed_root_jobs": completed_roots, "recorded_non_success_attempts": recovered_failures,
            "independent_text_coverage": {"total_chars": total_chars, "reviewed_chars": reviewed_chars,
                                           "text_complete": text_complete},
            "reported_explicit_reference_complete": meta.get("explicit_reference_complete"),
            "all_attempts_succeeded": not recovered_failures and bool(all_ids)
                and checks["every_attempt_has_matching_audit_receipts"],
            "legal_accuracy_evaluated": False, "subject_authenticity_evaluated": False,
            "scope": "Initial source-bound continuation execution only; feedback closure and legal correctness require separate acceptance"}


def read_events(path, trace_id):
    if not path.is_file() or not trace_id:
        raise ValueError("Existing audit SQLite and explicit report trace_id required")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM model_calls WHERE trace_id=? ORDER BY created_at,event_id", (trace_id,)).fetchall()
    return [{**dict(row), "details": json.loads(row["details"])} for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    prepared = json.loads((args.run_dir / "prepared.json").read_text(encoding="utf-8"))
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    events = read_events(args.run_dir / "model_calls.sqlite", report.get("trace_id"))
    result = verify(report.get("stats", {}).get("logic_llm", {}), prepared["windows"], prepared["clauses"],
                    events, trace_id=report.get("trace_id"), contract_id=prepared.get("contract_id"))
    report_events = report.get("stats", {}).get("model_calls", [])
    report_event_map = {event.get("event_id"): event for event in report_events}
    sqlite_event_map = {event.get("event_id"): event for event in events}
    event_fields = ("event_id", "created_at", "trace_id", "contract_id", "component", "model",
                    "operation", "status", "elapsed_seconds", "reason", "details")
    same_events = bool(report_events and len(report_event_map) == len(report_events)
                      and set(report_event_map) == set(sqlite_event_map)
                      and all(all(row.get(key) == sqlite_event_map[event_id].get(key) for key in event_fields)
                              for event_id, row in report_event_map.items()))
    result["checks"]["serialized_audit_matches_readonly_sqlite"] = same_events
    if not same_events:
        result["issues"].append("serialized_audit_and_sqlite_differ_or_missing")
        result["protocol_lineage_acceptance_passed"] = False
    # This helper reads local revision metadata; it never launches git.exe.
    from acceptance_common import provenance, write_json
    result["provenance"] = provenance()
    result["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["protocol_lineage_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
