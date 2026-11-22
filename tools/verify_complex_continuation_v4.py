"""Strict full-loop postrun checks with explicit initial continuation lineage.

Reuse the unchanged v2 receipt, current-feedback, subject, semantic and ledger
conditions. Replace only its one-call-per-initial-window plan assumption with
independently replayed same-source root/attempt completion. No model or Git runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import verify_complex_postrun_v2 as V2
import verify_initial_review_lineage as LINEAGE
from acceptance_common import digest, write_json
from zhilv_qiling_v2.long_context import ReviewWindow, Segment, coverage
from zhilv_qiling_v2.models import Clause

VERSION = "complex-continuation-v4"
LEGACY_SHA256 = "4814d84faf926ed762438cc697f98412ab46e1c77c1bc29b844c7008a5ceed24"
REPLACED_CHECKS = ("initial_call_window_ids_match_plan_exactly",
                   "initial_job_ids_batches_and_counts_match_plan",
                   "every_prepared_initial_window_completed")


def check(prepared, report, *, sqlite_events=None):
    """No mutation, fabricated final rows, ordering guesses or feedback projection."""
    original = V2.check(prepared, report, sqlite_events=sqlite_events)
    result = {**original, "checks": dict(original["checks"])}
    logic = report.get("stats", {}).get("logic_llm") or {}
    events = report.get("stats", {}).get("model_calls") or []
    # Feedback can revisit the identical source window/job ID. Exclude its
    # explicitly recorded call IDs from the initial-only audit selection; v2's
    # unchanged global bindings and fingerprint checks still validate every row.
    initial_ids = {call.get("call_id") for call in logic.get("calls", [])}
    feedback_ids = {call.get("call_id") for run in logic.get("feedback_runs", [])
                    if isinstance(run, dict) for call in run.get("calls", []) if isinstance(call, dict)} - initial_ids
    initial_events = [event for event in events if event.get("details", {}).get("call_id") not in feedback_ids]
    lineage = LINEAGE.verify(logic, prepared["windows"], prepared["clauses"], initial_events,
                             trace_id=report.get("trace_id"), contract_id=prepared.get("contract_id"))
    checks = result["checks"]
    lineage_checks = lineage["checks"]
    calls = logic.get("calls", [])
    roots = logic.get("initial_review_lineage", [])
    root_window_ids = [row.get("window_id") for row in roots]
    planned_ids = [row["window_id"] for row in prepared["windows"]]
    done = {row["root_job_id"].removesuffix(":batch:1")
            for row in lineage["independent_replay"] if row["complete"]}
    checks[REPLACED_CHECKS[0]] = bool(lineage_checks["root_jobs_exactly_match_prepared_windows"]
                                     and Counter(root_window_ids) == Counter(planned_ids))
    checks[REPLACED_CHECKS[1]] = bool(lineage_checks["root_completion_counts_match"]
        and lineage_checks["attempts_are_contiguous_and_source_bound"]
        and lineage_checks["actual_request_budget_and_counts_match"]
        and all(type(logic.get(key)) is int for key in
                ("planned_calls", "completed_calls", "completed_windows", "actual_attempt_calls"))
        and all(type(call.get("batch_index")) is int and call["batch_index"] == 1
                and type(call.get("batch_count")) is int and call["batch_count"] == 1
                and call.get("feedback_item_ids") == [] for call in calls))
    checks[REPLACED_CHECKS[2]] = bool(planned_ids and set(planned_ids) == done
        and lineage_checks["every_prepared_window_completed"])
    fields_allowed = {field.name for field in fields(Clause)}
    clauses = [Clause(**{key: value for key, value in row.items() if key in fields_allowed})
               for row in prepared["clauses"]]
    windows = [ReviewWindow(row["window_id"], row["kind"], tuple(Segment(**part) for part in row["segments"]),
                             row.get("reason", ""), tuple(row.get("dependency_ids", [])))
               for row in prepared["windows"]]
    independent = coverage(clauses, windows, done)
    # Same coverage requirements, now contributed only by replay-complete roots.
    checks["independent_text_coverage_complete"] = independent["text_complete"]
    checks["reported_text_complete_agrees"] = logic.get("text_complete") == independent["text_complete"]
    checks["reported_text_coverage_agrees"] = logic.get("text_coverage") == independent["text_coverage"]
    checks["reported_reference_status_agrees"] = logic.get("explicit_reference_complete") == independent["explicit_reference_complete"]
    checks["initial_continuation_lineage_independently_verified"] = lineage["protocol_lineage_acceptance_passed"]
    checks["initial_continuation_version_explicit"] = logic.get("initial_review_continuation_version") == "initial-review-continuation-v1"
    status_consistent = True
    for root in roots:
        attempts = root.get("attempts", [])
        # A producer cannot continue after it already declared this root done,
        # nor erase an old partial by relabelling it as an ok call.
        if any(attempt.get("complete") is True for attempt in attempts[:-1]):
            status_consistent = False
        for attempt in attempts:
            matching = [call for call in calls if call.get("call_id") == attempt.get("call_id")]
            if len(matching) != 1:
                status_consistent = False
                continue
            call = matching[0]
            if (attempt.get("complete") is True and call.get("status") != "ok"
                    or attempt.get("complete") is False and call.get("status") == "ok"):
                status_consistent = False
    checks["initial_partial_history_status_and_attempt_order_consistent"] = status_consistent
    checks["unchanged_legacy_strict_checker_identity"] = digest(Path(V2.__file__).read_bytes()) == LEGACY_SHA256
    strict = all(checks.values())
    effective = bool(strict and independent["explicit_reference_complete"]
        and original["feedback_completion"]["final_protocol_feedback_complete"]
        and not original["llm_degradations"]
        and original["upstream_semantic_review"]["clear_of_reported_limits"])
    result.update(version=VERSION,
        status="verified_effective_scheduled_model_review" if effective else
               "verified_execution_with_pending" if strict else "failed_or_unproven_strict_checks",
        strict_engineering_execution_verified=strict, effective_model_review_complete=effective,
        initial_window_log_and_source_coverage_complete=bool(checks[REPLACED_CHECKS[0]]
            and checks[REPLACED_CHECKS[2]] and independent["text_complete"]),
        independent_source_coverage=independent, explicit_reference_scope_complete=independent["explicit_reference_complete"],
        missing_initial_window_ids=sorted(set(planned_ids)-set(root_window_ids), key=str),
        unknown_initial_window_ids=sorted(set(root_window_ids)-set(planned_ids), key=str),
        initial_lineage_verification=lineage,
        original_single_call_assumption_checks={name: original["checks"][name] for name in REPLACED_CHECKS},
        preserved_legacy_checks={name: value for name, value in original["checks"].items()
                                 if name not in REPLACED_CHECKS},
        partial_attempt_history=[{key: call.get(key) for key in
            ("window_id", "root_job_id", "job_id", "attempt_index", "call_id", "status", "output_pending_reasons")}
            for call in calls if call.get("status") != "ok"],
        legacy_strict_checker_sha256=digest(Path(V2.__file__).read_bytes()),
        subject_authenticity_evaluated=False,
        meaning="All legacy receipt/feedback/subject/semantic/ledger requirements retained; only single-call initial plan shape is replaced by audited same-source continuation roots. Old partials remain evidence; an unresolved prior quote cannot disappear into an empty later page. This is not independent legal accuracy or subject authenticity.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    if not directory.is_relative_to((ROOT / "tools/acceptance_results").resolve()):
        parser.error("run-dir must remain under tools/acceptance_results")
    output = (args.out or directory / "postrun_strict_continuation_v4.json").resolve()
    if not output.is_relative_to(directory) or output.exists():
        parser.error("new output under this run directory required; preserve previous evidence")
    paths = {name: directory / name for name in ("prepared.json", "report.json", "result.json", "model_calls.sqlite")}
    # A running review is not a failed acceptance and cannot be accepted early.
    if not paths["report.json"].is_file():
        print(json.dumps({"status": "awaiting_completed_report", "strict_engineering_execution_verified": False,
            "effective_model_review_complete": False, "output_written": False}, ensure_ascii=True))
        return 3
    try:
        data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if path.suffix == ".json"}
        run = data["result.json"]
        if run.get("status") == "running":
            print(json.dumps({"status": "awaiting_completed_run_result", "strict_engineering_execution_verified": False,
                "effective_model_review_complete": False, "output_written": False}, ensure_ascii=True))
            return 3
        sqlite_events = V2.load_sqlite_events(paths["model_calls.sqlite"], data["report.json"].get("trace_id"))
        result = check(data["prepared.json"], data["report.json"], sqlite_events=sqlite_events)
        result["checks"].update(
            archive_roundtrip_verified_by_run=run.get("http_archive_roundtrip_verified") is True,
            saved_report_identity_matches_run=(data["report.json"].get("saved") is True
                and bool(data["report.json"].get("report_id"))
                and data["report.json"].get("report_id") == run.get("report_id")),
            prepared_bytes_match_run_provenance=run.get("prepared_sha256") == digest(paths["prepared.json"].read_bytes()))
        result["strict_engineering_execution_verified"] = all(result["checks"].values())
        result["effective_model_review_complete"] &= result["strict_engineering_execution_verified"]
        if not result["strict_engineering_execution_verified"]:
            result["status"] = "failed_or_unproven_strict_checks"
        result["input_sha256"] = {name: digest(path.read_bytes()) for name, path in paths.items()}
        result["tool_sha256"] = digest(Path(__file__).read_bytes())
        result["lineage_checker_sha256"] = digest(Path(LINEAGE.__file__).read_bytes())
        write_json(output, result)
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
        result = {"version": VERSION, "status": "failed_closed_invalid_or_missing_evidence",
            "strict_engineering_execution_verified": False, "effective_model_review_complete": False,
            "legal_correctness_evaluated": False, "error_type": type(exc).__name__, "error": str(exc)[:300],
            "tool_sha256": digest(Path(__file__).read_bytes())}
        write_json(output, result)
    print(json.dumps({"status": result["status"], "evidence": str(output),
        "strict_engineering_execution_verified": result["strict_engineering_execution_verified"],
        "effective_model_review_complete": result["effective_model_review_complete"]}, ensure_ascii=True))
    return 0 if result["strict_engineering_execution_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
