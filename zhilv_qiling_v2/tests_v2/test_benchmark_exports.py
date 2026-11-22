"""Offline split-workflow tests. Fake transport stays in temporary directories.

No model is loaded/called and no synthetic run is saved as acceptance evidence.
Official input/source files are read-only; simulated HTTP content is not gold.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from zhilv_qiling_v2 import benchmark_exports as B


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"model": "legalone-8b", "usage": {"prompt_tokens": 3},
            "choices": [{"message": {"role": "assistant", "content": "SYNTHETIC_TEST_RESPONSE"},
                         "finish_reason": "stop"}]}).encode("utf-8")


class BenchmarkExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.adapter = B._adapter()
        self.snapshot = B.DEFAULT_SNAPSHOT
        if not (self.snapshot / "official_snapshot.json").is_file():
            self.skipTest("read-only pinned official snapshot unavailable")

    def prepared(self, **changes):
        args = dict(tasks="1-2", shot="zero_shot", limit=1)
        args.update(changes)
        return B.prepare(self.snapshot, self.run, **args)

    def synthetic_infer(self, **changes):
        args = dict(tasks="1-2", shot="zero_shot", limit=1,
            endpoint="http://synthetic.invalid/v1", model_revision="SYNTHETIC_UNIT_TEST_NOT_ACTUAL_WEIGHTS",
            max_new_items=1, max_tokens=16)
        args.update(changes)
        identity = self.adapter.native_identity({"data": [{"id": "legalone-8b", "owned_by": "llamacpp",
            "meta": {"n_params": 8195978240, "n_ctx": 8192}}]}, "legalone-8b")
        budget = {"fits": True, "input_truncation_permitted": False, "reserved_output_tokens": args["max_tokens"],
            "server_context_tokens": 8192, "prompt_tokens": 3,
            "required_context_tokens": 3 + args["max_tokens"] + 1}
        with patch.object(self.adapter, "verify_native_identity", return_value=identity), \
             patch.object(self.adapter, "preflight_context", return_value=budget), \
             patch.object(self.adapter, "provenance", return_value={"scope": "SYNTHETIC_UNIT_TEST_ONLY"}), \
             patch.object(self.adapter.urllib.request, "urlopen", return_value=FakeResponse()) as transport, \
             patch.object(self.adapter.subprocess, "run", side_effect=AssertionError("no scoring or subprocess")):
            result = B.infer(self.snapshot, self.run, **args)
        return result, transport.call_count

    def test_prepare_is_read_only_and_idempotent_without_inference_state(self):
        with patch.object(self.adapter.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            result = self.prepared()
            path = self.run / "split_selection.json"
            before = path.read_bytes()
            self.assertEqual(self.prepared(), result)
            self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.run / "state.json").exists())
        self.assertTrue(result["selection"]["is_subset"])
        self.assertEqual(result["selection"]["selected_items"], 1)
        self.assertIsNone(result["formal_metrics"])

    def test_fixed_selection_cannot_silently_extend_or_change_task(self):
        self.prepared()
        for change in ({"limit": 2}, {"tasks": "3-6"}, {"shot": "one_shot"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "selection changed"):
                self.prepared(**change)

    def test_exact_official_manifest_is_pinned_not_just_its_self_declared_revision(self):
        with patch.object(B, "OFFICIAL_SNAPSHOT_MANIFEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "exact pinned snapshot"):
                self.prepared()

    def test_limit_and_tasks_are_explicit_validated(self):
        for limit in (0, 501, True, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "limit"):
                self.prepared(limit=limit)
        for tasks in ("", "1-2,1-2", "not-a-task"):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                self.prepared(tasks=tasks)

    def test_inference_reuses_existing_atomic_receipts_without_duplicate_generation(self):
        first, first_calls = self.synthetic_infer()
        second, second_calls = self.synthetic_infer()
        self.assertEqual((first_calls, second_calls), (1, 0))
        self.assertEqual(first["new_actual_model_responses"], 1)
        self.assertEqual(second["new_actual_model_responses"], 0)
        self.assertFalse(first["full_official_benchmark"])
        self.assertIsNone(first["formal_metrics"])
        receipts = list((self.run / "receipts").glob("*/*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_text(encoding="utf-8"))["status"], "succeeded")
        self.assertFalse((self.run / "official_results.csv").exists())

    def test_recovered_or_failed_identity_cannot_change_configuration(self):
        self.synthetic_infer()
        with self.assertRaisesRegex(ValueError, "resume configuration changed"):
            self.synthetic_infer(max_tokens=17)

    def test_export_contains_unchanged_official_sources_and_unscored_subset(self):
        self.synthetic_infer()
        output = Path(self.temp.name) / "linux.zip"
        source_before = (self.snapshot / "evaluation/main.py").read_bytes()
        state_before = (self.run / "state.json").read_bytes()
        result = B.export_linux(self.snapshot, self.run, output)
        self.assertEqual(result["status"], "exported_not_scored")
        self.assertFalse(result["full_official_benchmark"])
        self.assertFalse(result["official_evaluator_executed"])
        self.assertEqual(result["prediction_scope"]["generated_items_per_task"], {"1-2": 1})
        self.assertIsNone(result["formal_metrics"])
        extracted = Path(self.temp.name) / "extracted"
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read("LawBench/evaluation/main.py"), source_before)
            official = json.loads((self.snapshot / "official_snapshot.json").read_text(encoding="utf-8"))
            for entry in official["files"]:
                self.assertEqual(self.adapter.digest(archive.read("LawBench/" + entry["path"])), entry["sha256"])
            manifest = json.loads(archive.read("export_manifest.json"))
            self.assertEqual(manifest["export_scoring_status"], "not_scored")
            self.assertEqual(manifest["configuration_sha256"], self.adapter.json_digest(manifest["configuration"]))
            self.assertEqual(json.loads(archive.read("NOT_SCORED.json"))["status"], "not_scored_at_export")
            self.assertFalse(any(name.endswith("official_results.csv") for name in archive.namelist()))
            archive.extractall(extracted)
        self.assertEqual((self.snapshot / "evaluation/main.py").read_bytes(), source_before)
        self.assertEqual((self.run / "state.json").read_bytes(), state_before)
        namespace = runpy.run_path(str(extracted / "verify_bundle.py"), run_name="offline_test_verifier")
        self.assertEqual(namespace["verify"]()["status"], "verified")
        compile((extracted / "run_official_linux.py").read_text(encoding="utf-8"), "linux_wrapper", "exec")
        target = extracted / "run/predictions/legalone-8b/1-2.json"
        target.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            namespace["verify"]()
        with self.assertRaisesRegex(ValueError, "fresh output"):
            B.export_linux(self.snapshot, self.run, output)

    def test_export_refuses_fixture_and_unknown_outcome_receipts(self):
        self.synthetic_infer()
        state_path = self.run / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["fixture_only"] = True
        self.adapter.atomic_json(state_path, state)
        with self.assertRaisesRegex(ValueError, "fixtures are not model evidence"):
            B.export_linux(self.snapshot, self.run, Path(self.temp.name) / "fixture.zip")
        state.pop("fixture_only")
        self.adapter.atomic_json(state_path, state)
        receipt_path = next((self.run / "receipts").glob("*/*.json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["status"] = "started"
        self.adapter.atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "outcome unknown"):
            B.export_linux(self.snapshot, self.run, Path(self.temp.name) / "unknown.zip")

    def test_export_refuses_changed_prediction(self):
        self.synthetic_infer()
        path = self.run / "predictions/legalone-8b/1-2.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["0"]["prediction"] = "changed"
        self.adapter.atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, "prediction lacks a matching successful receipt"):
            B.export_linux(self.snapshot, self.run, Path(self.temp.name) / "changed.zip")

    def test_prepare_refuses_adopting_an_unpinned_existing_run(self):
        self.run.mkdir()
        (self.run / "state.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no split selection"):
            self.prepared()

    def test_cli_requires_explicit_limit_and_prepare_does_not_score(self):
        spec = importlib.util.spec_from_file_location("lawbench_split_cli_test", B.ROOT / "tools/selected_solutions_20261118/lawbench_split.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with redirect_stdout(io.StringIO()) as stdout:
            code = cli.main(["prepare", "--run-dir", str(self.run), "--tasks", "1-2", "--limit", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["actual_generation_calls"], 0)
        with patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(["infer", "--run-dir", str(self.run), "--tasks", "1-2", "--model-revision", "TEST_ONLY"])
        self.assertEqual(error.exception.code, 2)

    def linux_namespace(self):
        root = Path(self.temp.name) / "linux_wrapper_unit_only"
        root.mkdir()
        (root / "export_manifest.json").write_text(json.dumps({
            "prediction_scope": {"is_official_subset": True, "generated_items_per_task": {"1-2": 1}},
            "official_revision": self.adapter.REVISION, "generated_official_tasks": ["1-2"],
            "full_official_shot_predictions": False}), encoding="utf-8")
        verifier = Mock(return_value={"status": "verified"})
        namespace = {"__name__": "SYNTHETIC_WRAPPER_UNIT_TEST", "__file__": str(root / "run_official_linux.py")}
        with patch.dict(sys.modules, {"verify_bundle": SimpleNamespace(verify=verifier)}):
            exec(compile(B.LINUX_SCRIPT, "SYNTHETIC_WRAPPER_UNIT_TEST", "exec"), namespace)
        return root, namespace, verifier

    def test_linux_wrapper_refuses_windows_without_verification_or_scoring(self):
        root, namespace, verifier = self.linux_namespace()
        with patch.object(sys, "platform", "win32"), patch.object(self.adapter.subprocess, "run", side_effect=AssertionError("no scoring")):
            with self.assertRaisesRegex(SystemExit, "no Windows-equivalent score"):
                namespace["main"]()
        verifier.assert_not_called()
        self.assertFalse((root / "results").exists())

    def test_linux_missing_dependencies_preserve_unscored_state(self):
        root, namespace, _ = self.linux_namespace()
        with patch.object(sys, "platform", "linux"), patch.object(sys, "path", list(sys.path)), \
             patch.object(self.adapter, "evaluation_preflight", return_value={"ready": False, "errors": ["SYNTHETIC_MISSING_DEPENDENCY"]}), \
             patch.object(self.adapter.subprocess, "run", side_effect=AssertionError("no scoring")), redirect_stdout(io.StringIO()):
            self.assertEqual(namespace["main"](), 1)
        status = json.loads((root / "results/scoring_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "blocked_before_official_scoring")
        self.assertFalse(status["official_evaluator_executed"])
        self.assertIsNone(status["formal_metrics"])

    def test_linux_wrapper_calls_exact_official_entry_and_keeps_subset_scope(self):
        root, namespace, verifier = self.linux_namespace()
        def simulated_official_process(command, **kwargs):
            self.assertEqual(command[1:4], ["main.py", "-i", str(root / "run/predictions")])
            self.assertEqual(kwargs["cwd"], root / "LawBench/evaluation")
            # Temporary protocol fixture only; this is never an actual score.
            Path(command[-1]).write_text("task,model_name,score,abstention_rate\n1-2,legalone-8b,0.0,1.0\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="SYNTHETIC_SCORER_NOT_ACTUAL", stderr="")
        with patch.object(sys, "platform", "linux"), patch.object(sys, "path", list(sys.path)), \
             patch.object(self.adapter, "evaluation_preflight", return_value={"ready": True}), \
             patch.object(self.adapter.subprocess, "run", side_effect=simulated_official_process) as process, redirect_stdout(io.StringIO()):
            self.assertEqual(namespace["main"](), 0)
            with self.assertRaisesRegex(SystemExit, "overwrite scoring evidence"):
                namespace["main"]()
        self.assertEqual(process.call_count, 1)
        self.assertEqual(verifier.call_count, 3)
        status = json.loads((root / "results/scoring_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "official_evaluator_executed")
        self.assertFalse(status["full_official_benchmark"])
        self.assertTrue(status["prediction_scope"]["is_official_subset"])


if __name__ == "__main__":
    unittest.main()
