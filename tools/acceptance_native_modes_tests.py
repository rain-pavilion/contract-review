"""Isolated native reasoning/response integrity checks; no inference or scores."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import acceptance_lawbench as L


class Response:
    def __init__(self, message):
        self.message = message
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def read(self):
        return json.dumps({"model": "legalone-8b", "choices": [{"message": self.message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2}}).encode()


class NativeReasoningGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data_file = self.folder / "fixture.json"
        self.rows = [{"instruction": "选择一项。", "question": "隔离合成测试问题", "answer": "A"}]
        self.data_file.write_text(json.dumps(self.rows), encoding="utf-8")
        self.a = SimpleNamespace(snapshot=self.folder, run_dir=self.folder / "run", model="legalone-8b",
            model_revision="isolated-test-not-real-weights", endpoint="http://unused.invalid/v1", shot="zero_shot",
            tasks="1-2", max_items=1, item_limit=1, max_tokens=8, temperature=0, timeout=1)
        identity = L.native_identity({"data": [{"id": "legalone-8b", "owned_by": "llamacpp",
            "meta": {"n_params": 8195978240, "n_ctx": 8192, "ftype": "Q4_K - Medium"}}]}, self.a.model)
        for p in (patch.object(L, "source", return_value={}),
                  patch.object(L, "load_task", return_value=(self.data_file, self.rows)),
                  patch.object(L, "provenance", return_value={"isolated_test": True}),
                  patch.object(L, "verify_native_identity", return_value=identity),
                  patch.object(L, "preflight_context", return_value={"method":"isolated_native_budget_fixture",
                    "prompt_tokens":8,"server_context_tokens":8192,"reserved_output_tokens":8,
                    "reserved_stop_token":1,"required_context_tokens":17,"fits":True,"input_truncation_permitted":False})):
            p.start()
            self.addCleanup(p.stop)

    def state(self):
        return json.loads((self.a.run_dir / "state.json").read_text(encoding="utf-8"))

    def receipt(self):
        path = next((self.a.run_dir / "receipts").glob("*/*.json"))
        return path, json.loads(path.read_text(encoding="utf-8"))

    def infer(self, message=None):
        with patch.object(L.urllib.request, "urlopen", return_value=Response(message or {"content": "A"})):
            return L.infer(self.a)

    def test_native_modes_change_only_explicit_config_and_request(self):
        default = L.config_for(self.a)
        self.assertNotIn("chat_template_kwargs", default)
        self.assertNotIn("reasoning_format", default)
        self.a.native_thinking = "off"
        self.a.native_reasoning_format = "deepseek"
        config = L.config_for(self.a)
        without_modes = {key: value for key, value in config.items() if key not in ("chat_template_kwargs", "reasoning_format")}
        self.assertEqual(without_modes, default)
        captured = []
        def reply(request, timeout):
            captured.append(json.loads(request.data))
            return Response({"content": "A", "reasoning_content": "analysis only"})
        with patch.object(L.urllib.request, "urlopen", side_effect=reply):
            L.infer(self.a)
        _, receipt = self.receipt()
        for target in (captured[0], receipt, self.state()["configuration"]):
            self.assertEqual(target["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(target["reasoning_format"], "deepseek")
        self.assertEqual(captured[0]["messages"], [{"role": "user", "content": L.prompt(self.rows[0])}])

    def test_raw_message_and_analysis_retained_but_prediction_unchanged(self):
        message = {"content": " B ", "reasoning_content": "example A is not the final answer", "role": "assistant"}
        self.infer(message)
        _, receipt = self.receipt()
        self.assertEqual(receipt["response_message"], message)
        self.assertEqual(receipt["response_message_sha256"], L.json_digest(message))
        self.assertEqual(receipt["prediction_entry"]["prediction"], " B ")

    def test_receipt_native_mode_tampering_refuses_resume(self):
        self.a.native_thinking = "off"
        self.infer()
        path, receipt = self.receipt()
        receipt["chat_template_kwargs"]["enable_thinking"] = True
        L.atomic_json(path, receipt)
        with patch.object(L.urllib.request, "urlopen", side_effect=AssertionError("must not repeat inference")):
            with self.assertRaisesRegex(ValueError, "native reasoning mode"):
                L.infer(self.a)

    def test_config_native_mode_drift_refuses_resume(self):
        self.a.native_reasoning_format = "deepseek"
        self.infer()
        self.a.native_reasoning_format = "none"
        with patch.object(L.urllib.request, "urlopen", side_effect=AssertionError("must not repeat inference")):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                L.infer(self.a)

    def test_response_message_hash_tampering_refuses_resume(self):
        self.infer({"content": "A", "reasoning_content": "original"})
        path, receipt = self.receipt()
        receipt["response_message"]["reasoning_content"] = "tampered"
        L.atomic_json(path, receipt)
        with patch.object(L.urllib.request, "urlopen", side_effect=AssertionError("must not repeat inference")):
            with self.assertRaisesRegex(ValueError, "response|message"):
                L.infer(self.a)

    def test_prediction_must_equal_raw_response_content(self):
        self.infer({"content": "A", "reasoning_content": "original"})
        path, receipt = self.receipt()
        receipt["response_message"]["content"] = "B"
        receipt["response_message_sha256"] = L.json_digest(receipt["response_message"])
        L.atomic_json(path, receipt)
        with patch.object(L.urllib.request, "urlopen", side_effect=AssertionError("must not repeat inference")):
            with self.assertRaisesRegex(ValueError, "response|message|prediction"):
                L.infer(self.a)


if __name__ == "__main__":
    unittest.main()
