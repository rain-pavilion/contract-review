"""Full-trace audit counts and read-only running-task progress projection."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import api, model_audit as A


class TraceProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "audit.sqlite"
        self.path_patch = patch.object(A, "_path", return_value=self.path)
        self.path_patch.start()
        self.db = A._connect()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.db.close)
        self.seq = 0

    def event(self, trace, component="llm", operation="request_json", status="succeeded", reason="", **details):
        self.seq += 1
        row = (str(self.seq), f"2026-11-19T00:00:{self.seq:06d}+00:00", trace, "contract",
               component, "test-model", operation, status, 0.1, reason, json.dumps(details))
        self.db.execute("INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
        self.db.commit()

    def test_counts_all_events_beyond_recent_500(self):
        # Every request has two audit events. Neither is lost to a display limit.
        for index in range(620):
            self.event("trace-long", status="started", call_id=str(index))
            self.event("trace-long", status="succeeded", call_id=str(index), job_id=f"job-{index}")
        with patch.object(A, "recent_events", side_effect=AssertionError("must aggregate SQL")):
            result = A.trace_progress("trace-long")
        self.assertTrue(result["available"])
        self.assertEqual(result["total_events"], 1240)
        self.assertEqual(result["llm"]["started"], 620)
        self.assertEqual(result["llm"]["succeeded"], 620)
        self.assertEqual(result["inference_attempts"], 620)
        self.assertEqual(result["latest_event"]["job_id"], "job-619")

    def test_exact_trace_not_prefix_or_truncated_match(self):
        self.event("sameprefix12-A", status="started")
        self.event("sameprefix12-A", status="succeeded", job_id="selected-job")
        self.event("sameprefix12-B", status="failed")
        self.event("sameprefix12", status="skipped")
        result = A.trace_progress("sameprefix12-A")
        self.assertEqual(result["total_events"], 2)
        self.assertEqual(result["llm"], {"started": 1, "succeeded": 1, "failed": 0, "degraded": 0, "skipped": 0})
        self.assertEqual(A.trace_progress("sameprefix12") ["llm"]["skipped"], 1)
        self.assertEqual(A.trace_progress("' OR 1=1 --")["total_events"], 0)

    def test_start_terminal_skipped_and_retry_have_distinct_meaning(self):
        self.event("trace", status="started", call_id="one")
        self.event("trace", status="failed", call_id="one", reason="invalid_json")
        self.event("trace", status="started", call_id="two")
        self.event("trace", status="degraded", call_id="two", reason="json_retry")
        self.event("trace", status="skipped", call_id="never-sent", generation_sent=False,
                   reason="context_budget_exceeded")
        self.event("trace", status="succeeded", operation="load")
        result = A.trace_progress("trace")
        self.assertEqual(result["llm"], {"started": 2, "succeeded": 0, "failed": 1, "degraded": 1, "skipped": 1})
        self.assertEqual(result["inference_attempts"], 2)
        self.event("only-preflight", status="skipped", generation_sent=False)
        self.assertEqual(A.trace_progress("only-preflight")["inference_attempts"], 0)

    def test_pair_counts_are_not_window_counts_or_all_degradations(self):
        self.event("trace", "entailment", "predict_pair", "started")
        self.event("trace", "entailment", "predict_pair", "succeeded")
        self.event("trace", "entailment", "predict_window", "succeeded")
        self.event("trace", "entailment", "predict_pair", "degraded")
        self.event("trace", "entailment", "cross_check", "degraded", "neural_rule_disagreement")
        self.event("trace", "entailment", "cross_check", "degraded", "other_reason")
        result = A.trace_progress("trace")
        self.assertEqual(result["entailment"], {"predict_pair_succeeded": 1, "neural_rule_disagreement": 1})
        self.assertEqual(result["inference_attempts"], 0)

    def test_read_only_does_not_create_db_schema_or_events_or_expose_details(self):
        self.event("trace", job_id="job-safe", arbitrary_secret="must-not-leak", source_text="must-not-read")
        before = self.db.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0]
        with patch.object(A, "_connect", side_effect=AssertionError("read-only connection required")):
            result = A.trace_progress("trace")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0], before)
        self.assertFalse(result["contains_contract_text"])
        self.assertNotIn("must-not", json.dumps(result))
        missing = Path(self.temp.name) / "new-directory" / "absent.sqlite"
        with patch.object(A, "_path", return_value=missing):
            result = A.trace_progress("trace")
        self.assertFalse(result["available"])
        self.assertFalse(missing.parent.exists())

    def test_missing_schema_returns_unavailable_without_mutation(self):
        self.db.execute("DROP TABLE model_calls")
        self.db.commit()
        result = A.trace_progress("trace")
        self.assertFalse(result["available"])
        self.assertEqual(self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_missing_exact_trace_does_not_return_other_run(self):
        self.event("other", job_id="other-job")
        result = A.trace_progress("new-trace")
        self.assertTrue(result["available"])
        self.assertEqual(result["total_events"], 0)
        self.assertIsNone(result["latest_event"])
        for value in ("", None, 42):
            with self.assertRaises(ValueError):
                A.trace_progress(value)


class RuntimeProgressApiTests(unittest.TestCase):
    def test_only_running_batch_items_receive_exact_trace_progress(self):
        store = SimpleNamespace(get_batch=lambda *a, **k: copy.deepcopy({"batch_id": "batch", "items": [
            {"task_id": "1" * 32, "status": "running"},
            {"task_id": "2" * 32, "status": "pending"},
            {"task_id": "3" * 32, "status": "completed"},
            {"task_id": "4" * 32, "status": "failed"},
        ]}))
        ctx = SimpleNamespace(method="GET", params={"batch_id": "batch"}, query={}, config=SimpleNamespace(),
                              handler=SimpleNamespace(server=object()))
        with patch.object(api, "_ensure_queue_worker", return_value=SimpleNamespace(store=store)), \
                patch.object(A, "trace_progress", return_value={"available": True, "inference_attempts": 7}) as progress, \
                patch.object(api, "_run_workspace_task", side_effect=AssertionError("must not execute")) as run:
            result = api.handle_workspace_batch(ctx)
        progress.assert_called_once_with("1" * 12)
        run.assert_not_called()
        self.assertEqual(result["items"][0]["runtime_progress"]["inference_attempts"], 7)
        self.assertTrue(all("runtime_progress" not in item for item in result["items"][1:]))


if __name__ == "__main__":
    unittest.main()
