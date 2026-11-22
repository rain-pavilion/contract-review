"""Adversarial offline checks; synthetic receipts never establish model quality."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import complex_postrun_v2_tests_20261115 as FIXTURE
import verify_complex_continuation_v4 as V4
import verify_complex_postrun_v2 as V2
import verify_initial_review_lineage as LINEAGE


def fixture():
    prepared, report = FIXTURE.fixture()
    logic = report["stats"]["logic_llm"]
    original_calls = deepcopy(logic["calls"])
    original_events = deepcopy(report["stats"]["model_calls"])
    first_root = original_calls[0]["job_id"]
    final_call = deepcopy(original_calls[0])
    final_call.update(job_id=first_root + ":continue:2", call_id="request-continuation-2")
    calls = original_calls + [final_call]
    root_map = {}
    for call in calls:
        is_final = call is final_call
        root = first_root if is_final else call["job_id"]
        index = 2 if is_final else 1
        partial = call is calls[0]
        content = {"defects": [], "more_findings_possible": partial}
        if is_final:
            content["corrections"] = []
        reasons = ["initial_output_more_findings_possible"] if partial else []
        call.update(root_job_id=root, attempt_index=index, actual_attempt_count=1,
                    status="partial" if partial else "ok", output_scope_complete=not partial,
                    output_pending_reasons=reasons, output_policy=deepcopy(LINEAGE.POLICY),
                    request_receipts=[{"call_id": call["call_id"], "transport_attempt": 1,
                        "status": "succeeded", **{key: call[key] for key in
                            ("window_id", "job_id", "source_sha256")}}])
        lineage = root_map.setdefault(root, {"root_job_id": root, "window_id": call["window_id"],
            "source_sha256": call["source_sha256"], "max_attempts": 3, "complete": True,
            "pending_reasons": [], "pending_findings": [], "accepted_findings": [], "attempts": []})
        lineage["attempts"].append({"attempt_index": index, "window_id": call["window_id"],
            "source_sha256": call["source_sha256"], "call_id": call["call_id"], "job_id": call["job_id"],
            "content": content, "transport_partial": False, "pending_reasons": reasons,
            "repaired_finding_ids": [], "complete": not partial})
        lineage["attempt_count"] = len(lineage["attempts"])
    for event in original_events:
        if event["component"] == "llm":
            event["details"].update(native_json_schema_sent=True, json_schema_sha256="a" * 64)
    for original in original_events[:2]:
        event = deepcopy(original)
        event["event_id"] = final_call["call_id"] + "-" + event["status"]
        event["details"].update({key: final_call[key] for key in ("call_id", "window_id", "job_id", "source_sha256")})
        original_events.append(event)
    logic.update(calls=calls, initial_review_lineage=list(root_map.values()),
        initial_review_continuation_version="initial-review-continuation-v1",
        initial_output_policy=deepcopy(LINEAGE.POLICY), initial_output_pending=[],
        actual_request_budget=0, actual_attempt_calls=len(calls), completed_windows=len(prepared["windows"]))
    report["stats"]["model_calls"] = original_events
    return prepared, report


def check(prepared, report):
    return V4.check(prepared, report, sqlite_events=deepcopy(report["stats"]["model_calls"]))


class ContinuationV4Tests(unittest.TestCase):
    def test_two_actual_attempts_complete_one_root_without_erasing_partial_history(self):
        prepared, report = fixture()
        before = deepcopy(report)
        old = V2.check(prepared, report, sqlite_events=deepcopy(report["stats"]["model_calls"]))
        result = check(prepared, report)
        self.assertFalse(old["strict_engineering_execution_verified"])
        self.assertTrue(result["strict_engineering_execution_verified"], result["checks"])
        self.assertTrue(result["effective_model_review_complete"], result)
        self.assertEqual(len(result["partial_attempt_history"]), 1)
        self.assertGreater(result["initial_lineage_verification"]["actual_attempt_calls"], len(prepared["windows"]))
        self.assertEqual(report, before)
        for key, value in old["checks"].items():
            if key not in V4.REPLACED_CHECKS:
                self.assertEqual(result["checks"][key], value, key)
        self.assertFalse(result["legal_correctness_evaluated"])

    def test_deleted_attempt_or_original_window_cannot_be_hidden_by_summary_counts(self):
        for mutation in ("first_call", "first_attempt", "window"):
            with self.subTest(mutation=mutation):
                prepared, report = fixture()
                logic = report["stats"]["logic_llm"]
                if mutation == "first_call":
                    logic["calls"].pop(0)
                    logic["actual_attempt_calls"] -= 1
                elif mutation == "first_attempt":
                    logic["initial_review_lineage"][0]["attempts"].pop(0)
                else:
                    root = logic["initial_review_lineage"].pop()["root_job_id"]
                    logic["calls"] = [call for call in logic["calls"] if call["root_job_id"] != root]
                    logic["planned_calls"] -= 1
                    logic["completed_calls"] -= 1
                    logic["completed_windows"] -= 1
                    logic["actual_attempt_calls"] = len(logic["calls"])
                self.assertFalse(check(prepared, report)["strict_engineering_execution_verified"])

    def test_bad_prior_quote_is_not_cleared_by_empty_continuation_or_forged_summary(self):
        prepared, report = fixture()
        first = report["stats"]["logic_llm"]["initial_review_lineage"][0]
        cid = prepared["windows"][0]["segments"][0]["clause_id"]
        first["attempts"][0]["content"]["defects"] = [{"clause_id": cid,
            "code": "fee_mismatch", "severity": "high", "message": "合成诊断待核验。",
            "quote": "原窗口完全不存在的文字", "related_quotes": []}]
        result = check(prepared, report)
        self.assertFalse(result["strict_engineering_execution_verified"])
        self.assertFalse(result["checks"]["initial_continuation_lineage_independently_verified"])
        replay = result["initial_lineage_verification"]["independent_replay"][0]
        self.assertFalse(replay["complete"])
        self.assertEqual(len(replay["pending_finding_ids"]), 1)

    def test_forged_early_complete_or_relabelled_partial_is_rejected(self):
        for mutation in ("relabel", "erase_partial"):
            prepared, report = fixture()
            logic = report["stats"]["logic_llm"]
            logic["calls"][0]["status"] = "ok"
            if mutation == "erase_partial":
                attempt = logic["initial_review_lineage"][0]["attempts"][0]
                attempt.update(complete=True, pending_reasons=[], content={"defects": [], "more_findings_possible": False})
                logic["calls"][0].update(output_scope_complete=True, output_pending_reasons=[])
            result = check(prepared, report)
            self.assertFalse(result["checks"]["initial_partial_history_status_and_attempt_order_consistent"])
            self.assertFalse(result["strict_engineering_execution_verified"])

    def test_changed_source_duplicate_real_call_and_missing_sqlite_remain_unproven(self):
        for mutation in ("source", "call_id", "sqlite"):
            prepared, report = fixture()
            if mutation == "source":
                report["stats"]["logic_llm"]["calls"][-1]["source_sha256"] = "0" * 64
            elif mutation == "call_id":
                report["stats"]["logic_llm"]["calls"][-1]["call_id"] = "request-0"
            result = V4.check(prepared, report, sqlite_events=None if mutation == "sqlite" else deepcopy(report["stats"]["model_calls"]))
            self.assertFalse(result["strict_engineering_execution_verified"])

    def test_actual_failed_or_truncated_request_remains_strict_failure_after_later_success(self):
        prepared, report = fixture()
        report["stats"]["model_calls"][1].update(status="degraded", reason="output_truncated")
        report["stats"]["model_calls"][1]["details"]["finish_reason"] = "length"
        result = check(prepared, report)
        self.assertFalse(result["checks"]["all_real_requests_have_verified_success_receipts"])
        self.assertFalse(result["strict_engineering_execution_verified"])

    def test_manual_semantic_pending_still_prevents_effective_completion(self):
        for field in ("manual_required", "input_truncated", "disagreement"):
            prepared, report = fixture()
            report["stats"]["statute_checks"] = [{"candidate_id": "pending", field: True}]
            result = check(prepared, report)
            self.assertTrue(result["strict_engineering_execution_verified"], result["checks"])
            self.assertFalse(result["effective_model_review_complete"])
            self.assertFalse(result["upstream_semantic_review"]["clear_of_reported_limits"])

    def test_subject_pending_cannot_claim_stability_or_effective_completion(self):
        prepared, report = fixture()
        report["stats"]["cross_validation"].update(termination="round_limit", converged=False,
            feedback_verification_complete=False, pending_subject_verification_request_ids=["subject-unknown"])
        logic = report["stats"]["logic_llm"]
        logic["feedback_runs_complete"] = False
        logic["feedback_current_request"].update(required=True, fingerprint="current-state",
            subject_verification_complete=False, pending_subject_request_ids=["subject-unknown"])
        result = check(prepared, report)
        self.assertFalse(result["effective_model_review_complete"])
        report["stats"]["cross_validation"].update(termination="stable", converged=True)
        self.assertFalse(check(prepared, report)["strict_engineering_execution_verified"])

    def test_feedback_fingerprint_and_each_item_source_checks_are_retained(self):
        prepared, report = fixture()
        run, binding = FIXTURE.add_feedback_run(prepared, report)
        result = check(prepared, report)
        self.assertTrue(result["strict_engineering_execution_verified"], result["checks"])
        run["calls"][0]["verifications"][0]["clause_quote"] = "错误绑定的引文"
        binding["run_sha256"] = V2.source_sha256(run)
        self.assertFalse(check(prepared, report)["strict_engineering_execution_verified"])

    def test_running_directory_without_report_is_not_accepted_or_written(self):
        with TemporaryDirectory() as temporary, patch.object(V4, "ROOT", Path(temporary)):
            directory = Path(temporary) / "tools/acceptance_results/running"
            directory.mkdir(parents=True)
            (directory / "result.json").write_text('{"status":"running"}', encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = V4.main(["--run-dir", str(directory)])
            self.assertEqual(code, 3)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "awaiting_completed_report")
            self.assertFalse(result["output_written"])
            self.assertFalse((directory / "postrun_strict_continuation_v4.json").exists())


if __name__ == "__main__":
    unittest.main()
