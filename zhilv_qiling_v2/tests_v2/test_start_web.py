"""Launcher lifecycle tests: mocked HTTP/processes only, no model inference."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import start_web as W
import zhilv_qiling_v2 as package


class Response:
    def __init__(self, data, server="ZhilvQiling/test"):
        self.data = data
        self.headers = {"Server": server}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode()


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.keep(patch.object(W, "ROOT", self.root))
        self.keep(patch.object(W, "model_port_in_use", return_value=False))
        self.keep(patch.dict(os.environ, {}, clear=False))
        self.keep(patch.object(W.time, "sleep"))
        self.keep(patch.object(W.time, "monotonic", return_value=0))
        self.keep(patch("sys.stdout", new_callable=io.StringIO))
        self.keep(patch("sys.stderr", new_callable=io.StringIO))
        self.addCleanup(self.temp.cleanup)
        self.args = argparse.Namespace(with_model=True, model_port=8620, model_backend="auto",
            llama_server="", gguf_model="", cpu_threads=0, model_device="cpu", quant="4bit", model_dtype="auto")

    def keep(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def native(self):
        return {"id": "legalone-8b", "aliases": ["legalone-8b"],
                "meta": {"n_params": W.LEGALONE_PARAMETERS, "n_embd": 4096, "n_ctx": 8192}}

    def assets(self):
        binary = self.root / ".model-runtime/current/cpu/llama-server.exe"
        model = self.root / ".model-runtime/models/legalone-8b/current-Q4_K_M.gguf"
        for path in (binary, model):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not executable: mocked test fixture")
        receipts = self.root / "tools/acceptance_results"
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / "llama_runtime_preparation.json").write_text(json.dumps({
            "status": "prepared_not_executed", "server": str(binary.relative_to(self.root))}))
        (receipts / "legalone8b_gguf_conversion.json").write_text(json.dumps({
            "status": "quantized_not_inference_validated", "gguf": {"path": str(model.relative_to(self.root))}}))
        return binary, model

    def child(self):
        child = Mock()
        child.poll.return_value = None
        return child

    def web_health(self):
        return {"tag": "test", "multi_agent": {}, "llm": {"enabled": True,
                "model": "legalone-8b", "endpoint": "http://127.0.0.1:8620/v1"}}

    def call_main(self, api, child, workspace=None):
        with patch.object(W, "workspace_status", side_effect=workspace or [None]), \
                patch.object(W, "connect_local_model", return_value=child), \
                patch.object(package, "api", api, create=True), \
                patch.object(sys, "argv", ["start_web.py", "--with-model", "--no-browser"]):
            return W.main()

    def test_native_requires_exact_alias_and_known_parameter_count(self):
        cases = [(self.native(), True), ({**self.native(), "id": "legalone-8b-fake"}, False),
                 ({**self.native(), "id": "LegalOne-8B"}, False),
                 ({**self.native(), "meta": {"n_params": 1700000000}}, False),
                 ({"id": "legalone-8b"}, False)]
        for model, expected in cases:
            with self.subTest(model=model), patch.object(W.urllib.request, "urlopen", side_effect=[
                    Response({"status": "ok"}), Response({"data": [model]})]):
                self.assertEqual(bool(W.local_model_status()), expected)

    def test_transformers_health_remains_supported(self):
        item = {"id": "legalone-8b", "model_dir": "models/legalone-8b", "ready": True}
        with patch.object(W.urllib.request, "urlopen", return_value=Response({"data": [item]})) as http:
            self.assertEqual(W.local_model_status(), item)
        http.assert_called_once()

    def test_owned_native_ready_with_wrong_identity_fails_without_waiting(self):
        with patch.object(W.urllib.request, "urlopen", side_effect=[Response({"status": "ok"}),
                Response({"data": [{"id": "legalone-8b", "meta": {"n_params": 1700000000}}]})]):
            with self.assertRaisesRegex(RuntimeError, "identity did not match"):
                W.local_model_status(strict_identity=True)

    def test_manifest_is_preferred_and_missing_manifest_target_falls_back(self):
        binary, model = self.assets()
        old = self.root / ".model-runtime/aaa-old/llama-server.exe"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"mock old runtime")
        self.assertEqual(W.llama_runtime_paths(self.args), (binary, model))
        binary.unlink()
        self.assertEqual(W.llama_runtime_paths(self.args), (old, model))

    def test_explicit_relative_paths_are_resolved_before_child_cwd_changes(self):
        binary, model = self.assets()
        self.args.llama_server = str(binary.relative_to(self.root))
        self.args.gguf_model = str(model.relative_to(self.root))
        with patch("os.getcwd", return_value=str(self.root)):
            self.assertEqual(W.llama_runtime_paths(self.args), (binary, model))

    def test_auto_gguf_start_passes_native_arguments_and_returns_owned_child(self):
        binary, model = self.assets()
        child = self.child()
        with patch.object(W, "local_model_status", side_effect=[None, self.native()]), \
                patch.object(W.subprocess, "Popen", return_value=child) as popen:
            self.assertIs(W.connect_local_model(self.args), child)
        command, kwargs = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual(command[:5], [str(binary), "--model", str(model), "--alias", "legalone-8b"])
        for flag, value in (("--ctx-size", "16384"), ("--threads", "8"), ("--n-gpu-layers", "0"),
                            ("--parallel", "1"), ("--reasoning", "off")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--no-context-shift", command)
        self.assertEqual(kwargs["cwd"], str(binary.parent))
        self.assertEqual(Path(kwargs["stdout"].name), self.root / "zhilv_qiling_v2/reports/model-server.log")
        self.assertEqual(os.environ["ZLQL_LLM_ENDPOINT"], "http://127.0.0.1:8620/v1")
        child.terminate.assert_not_called()

    def test_explicit_transformers_backend_is_retained(self):
        self.assets()
        self.args.model_backend = "transformers"
        child = self.child()
        with patch.object(W, "local_model_status", side_effect=[None, self.native()]), \
                patch.object(W, "start_transformers_model", return_value=child) as hf, \
                patch.object(W.subprocess, "Popen") as native:
            self.assertIs(W.connect_local_model(self.args), child)
        hf.assert_called_once()
        native.assert_not_called()

    def test_existing_verified_model_is_reused_without_process_ownership(self):
        with patch.object(W, "local_model_status", return_value=self.native()), \
                patch.object(W.subprocess, "Popen") as popen:
            self.assertIsNone(W.connect_local_model(self.args))
        popen.assert_not_called()

    def test_unknown_occupied_model_port_does_not_start_another_model(self):
        with patch.object(W, "local_model_status", return_value=None), \
                patch.object(W, "model_port_in_use", return_value=True), \
                patch.object(W.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "identity/readiness"):
                W.connect_local_model(self.args)
        popen.assert_not_called()

    def test_interrupt_and_timeout_during_readiness_reap_owned_child(self):
        self.assets()
        for interrupted in (True, False):
            child = self.child()
            statuses = [None, KeyboardInterrupt()] if interrupted else [None]
            with self.subTest(interrupted=interrupted), \
                    patch.object(W, "local_model_status", side_effect=statuses), \
                    patch.object(W.subprocess, "Popen", return_value=child), \
                    patch.object(W.time, "monotonic", side_effect=None if interrupted else [0, 901], return_value=0):
                with self.assertRaises(KeyboardInterrupt if interrupted else RuntimeError):
                    W.connect_local_model(self.args)
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=10)

    def test_api_initialization_failure_reaps_owned_model(self):
        child = self.child()
        api = types.SimpleNamespace(serve=Mock(side_effect=RuntimeError("mock init failure")))
        self.assertEqual(self.call_main(api, child), 1)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)

    def test_api_import_failure_reaps_owned_model(self):
        child = self.child()
        real_import = __import__
        def fail_api_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "zhilv_qiling_v2" and "api" in fromlist:
                raise ImportError("mock API dependency missing")
            return real_import(name, globals, locals, fromlist, level)
        with patch.object(W, "workspace_status", return_value=None), \
                patch.object(W, "connect_local_model", return_value=child), \
                patch.object(sys, "argv", ["start_web.py", "--with-model", "--no-browser"]), \
                patch("builtins.__import__", side_effect=fail_api_import):
            with self.assertRaises(ImportError):
                W.main()
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)

    def test_normal_interrupt_closes_web_and_only_owned_model(self):
        for owned in (True, False):
            child = self.child() if owned else None
            server = Mock()
            server.serve_forever.side_effect = KeyboardInterrupt
            with self.subTest(owned=owned):
                self.assertEqual(self.call_main(types.SimpleNamespace(serve=Mock(return_value=server)), child), 0)
            server.server_close.assert_called_once()
            if owned:
                child.terminate.assert_called_once()

    def test_web_bind_race_return_still_reaps_owned_model(self):
        child = self.child()
        api = types.SimpleNamespace(serve=Mock(side_effect=OSError("mock race")))
        with patch.object(W, "existing_web_model_ready", return_value=True):
            self.assertEqual(self.call_main(api, child, [None, self.web_health()]), 0)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)

    def test_existing_web_requires_actual_model_health(self):
        for status in (None, self.native()):
            with self.subTest(status=status), patch.object(W, "workspace_status", return_value=self.web_health()), \
                    patch.object(W, "local_model_status", return_value=status), \
                    patch.object(W, "connect_local_model") as connect, \
                    patch.object(sys, "argv", ["start_web.py", "--with-model", "--no-browser"]):
                self.assertEqual(W.main(), 0 if status else 1)
            connect.assert_not_called()

    def test_cleanup_escalates_only_when_owned_child_does_not_exit(self):
        child = self.child()
        child.wait.side_effect = [subprocess.TimeoutExpired("mock", 10), None]
        W.stop_owned_model(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_args_list[-1].kwargs, {"timeout": 5})


if __name__ == "__main__":
    unittest.main()
