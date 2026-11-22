"""Independent offline diagnostics for serialized v2 feedback evidence.

Does not call a model, touch SQLite, alter product/verifier code, or invent gold.
The baseline fixture is synthetic. Passing these cases would prove a verifier
acceptance gap, not establish that a production report was actually corrupted.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from complex_postrun_v2_tests_20261115 import fixture, check


def feedback_run(*, calls=None, complete=True, pending=False):
    """A run with one required feedback item on clause b."""
    return {
        "complete": complete, "feedback_complete": complete,
        "planned_calls": 1, "completed_calls": 1 if complete else 0,
        "cross_windows": 0, "cross_windows_completed": 0,
        "uncompleted_feedback_item_ids": ["fb-required"] if pending else [],
        "unmatched_feedback_item_ids": [], "unreviewed_clause_ids": [],
        "feedback_items": [{"feedback_item_id": "fb-required", "clause_ids": ["b"],
                            "candidate_id": "candidate-required", "article_uid": "law-required"}],
        "verified_feedback_item_ids": [] if pending else ["fb-required"],
        "calls": [] if calls is None else calls,
    }


def add_bound_feedback_call(report, *, status="ok", acknowledged=True, suffix="1"):
    """Reuse an exact valid source with a new real-request-shaped receipt pair."""
    logic = report["stats"]["logic_llm"]
    call = deepcopy(logic["calls"][0])
    call.update(call_id="feedback-request-" + suffix, job_id="feedback-job-" + suffix, status=status,
                feedback_item_ids=["fb-required"],
                verified_feedback_item_ids=["fb-required"] if acknowledged else [],
                uncompleted_feedback_item_ids=[] if acknowledged else ["fb-required"])
    for index, original in enumerate(report["stats"]["model_calls"][:2]):
        event = deepcopy(original)
        event["event_id"] = "feedback-event-" + suffix + "-" + str(index)
        event["details"].update({key: call[key] for key in
                                 ("call_id", "window_id", "job_id", "source_sha256")})
        report["stats"]["model_calls"].append(event)
    return call


def summarize(name, result, **inputs):
    return {
        "case": name, "synthetic_input": True, "new_model_calls": 0,
        "strict_engineering_execution_verified": result["strict_engineering_execution_verified"],
        "effective_model_review_complete": result["effective_model_review_complete"],
        "feedback_completion": result["feedback_completion"],
        "receipt_binding_status": result["receipt_bindings"]["status"],
        "failed_checks": [key for key, value in result["checks"].items() if not value],
        "mutated_evidence": inputs,
    }


def diagnostics():
    results = []
    prepared, report = fixture()
    results.append(summarize("positive_control", check(prepared, report)))

    prepared, report = fixture()
    run = feedback_run()
    report["stats"]["logic_llm"].update(feedback_runs=[run], feedback_latest_runs=[run])
    results.append(summarize("required_feedback_job_has_zero_calls", check(prepared, report),
                             planned_calls=1, completed_calls=1, actual_feedback_call_rows=0,
                             required_item="fb-required"))

    prepared, report = fixture()
    call = add_bound_feedback_call(report, status="feedback_incomplete", acknowledged=False)
    run = feedback_run(calls=[call])
    run["feedback_items"][0]["clause_ids"] = [call["source_spans"][0]["clause_id"]]
    report["stats"]["logic_llm"].update(feedback_runs=[run], feedback_latest_runs=[run])
    results.append(summarize("bound_call_pending_but_run_summary_claims_complete", check(prepared, report),
                             call_status=call["status"],
                             call_verified_feedback_item_ids=call["verified_feedback_item_ids"],
                             call_uncompleted_feedback_item_ids=call["uncompleted_feedback_item_ids"],
                             run_uncompleted_feedback_item_ids=run["uncompleted_feedback_item_ids"]))

    prepared, report = fixture()
    call = add_bound_feedback_call(report)
    # Correct receipts and source locatability alone do not prove this source
    # belongs to the feedback item's required clause.
    source_clause_ids = {span["clause_id"] for span in call["source_spans"]}
    missing_clause = next(cid for cid in ("a", "b") if cid not in source_clause_ids)
    run = feedback_run(calls=[call])
    run["feedback_items"][0]["clause_ids"] = [missing_clause]
    report["stats"]["logic_llm"].update(feedback_runs=[run], feedback_latest_runs=[run])
    results.append(summarize("feedback_item_clause_not_in_bound_call_source", check(prepared, report),
                             source_clause_ids=sorted(source_clause_ids), required_clause_id=missing_clause))

    prepared, report = fixture()
    success_call = add_bound_feedback_call(report, suffix="older")
    failed_call = add_bound_feedback_call(report, status="feedback_incomplete", acknowledged=False,
                                          suffix="newer")
    success = feedback_run(calls=[success_call])
    failed = feedback_run(calls=[failed_call], complete=False, pending=True)
    for run in (success, failed):
        run["feedback_items"][0]["clause_ids"] = [success_call["source_spans"][0]["clause_id"]]
    report["stats"]["logic_llm"].update(
        feedback_runs=[success, failed], feedback_latest_runs=[success],
        feedback_attempts=[{"fingerprint": "same-current-state", "round": 2, "complete": True},
                           {"fingerprint": "same-current-state", "round": 3, "complete": False}])
    results.append(summarize("latest_selects_older_complete_attempt_for_same_fingerprint", check(prepared, report),
                             history_complete=[True, False], latest_history_index=0,
                             same_fingerprint_last_attempt_complete=False))
    return results


if __name__ == "__main__":
    print(json.dumps({"scope": "synthetic_readonly_verifier_audit", "cases": diagnostics()},
                     ensure_ascii=True, indent=2))
