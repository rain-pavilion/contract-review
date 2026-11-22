"""Model integration contracts; synthetic servers are never model accuracy evidence."""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch, Mock

from zhilv_qiling_v2 import entailment as E, llm_client as C, local_llm as L, model_audit as A, model_runtime as R


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"ZLQL_MODEL_AUDIT_PATH": str(pathlib.Path(self.temp.name) / "calls.sqlite")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_explicit_missing_model_does_not_fallback(self):
        self.assertIsNone(L.find_model_dir(pathlib.Path(self.temp.name) / "missing-8b"))

    def test_incomplete_shards_are_not_ready(self):
        folder = pathlib.Path(self.temp.name)
        (folder / "config.json").write_text("{}")
        (folder / "model-00001-of-00002.safetensors").write_bytes(b"x")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
        self.assertIsNone(L.find_model_dir(folder))

    def test_serve_cli_passes_quantization_and_offload(self):
        with patch.object(L, "serve") as serve:
            self.assertEqual(L.main(["--serve", "--model-dir", "requested-8b", "--quant", "4bit", "--device", "cuda",
                                     "--dtype", "bf16", "--cpu-offload", "--max-gpu-memory", "3GiB"]), 0)
        self.assertEqual(serve.call_args.kwargs["quant"], "4bit")
        self.assertEqual(serve.call_args.kwargs["device"], "cuda")
        self.assertEqual(serve.call_args.kwargs["model_dir"], "requested-8b")
        self.assertTrue(serve.call_args.kwargs["cpu_offload"])

    def test_audit_scope_and_persistent_receipts_do_not_store_contract_text(self):
        with A.review_audit("trace-1", "contract-hash") as collector:
            A.record("llm", "legalone-8b", "request_json", "succeeded", .12,
                     prompt_chars=30, prompt="PRIVATE CONTRACT", api_key="SECRET")
        self.assertEqual(len(collector.events), 1)
        events = A.recent_events(trace_id="trace-1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["contract_id"], "contract-hash")
        self.assertNotIn("PRIVATE", json.dumps(events))
        self.assertNotIn("SECRET", json.dumps(events))

    def test_missing_endpoint_is_audited(self):
        with A.review_audit("missing") as collector:
            with self.assertRaises(C.LLMClientError):
                C.request_json(C.LLMConfig(), [{"role": "user", "content": "private"}])
        self.assertEqual(collector.events[0]["status"], "skipped")

    def test_interrupted_transport_leaves_a_persistent_started_receipt(self):
        with A.review_audit("interrupted"), patch.object(C, "_complete", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                C.request_json(C.LLMConfig(endpoint="http://unused", model="legalone-8b"),
                               [{"role": "user", "content": "private"}])
        events = A.recent_events(trace_id="interrupted")
        self.assertEqual([e["status"] for e in events], ["started"])
        self.assertTrue(events[0]["details"]["call_id"])

    def test_every_parse_attempt_and_recovery_is_audited(self):
        replies = [("not JSON", {"model": "legalone-8b"}, .01), ('{"defects":[]}', {"model": "legalone-8b"}, .01)]
        with patch.object(C, "_complete", side_effect=replies), A.review_audit("retry") as collector:
            receipt = C.request_json(C.LLMConfig(endpoint="http://unused", model="legalone-8b"),
                                     [{"role": "user", "content": "private contract"}], retries=1)
        self.assertEqual([e["status"] for e in collector.events], ["started", "failed", "started", "degraded"])
        self.assertEqual(collector.events[0]["details"]["call_id"], collector.events[1]["details"]["call_id"])
        self.assertEqual(receipt.degraded, "json_retry")
        self.assertNotIn("private contract", json.dumps(collector.events))

    def test_unparseable_truncated_json_is_not_retried_with_same_budget(self):
        response = ('{"defects":[{"message":"unfinished',
                    {'model': 'legalone-8b', 'choices': [{'finish_reason': 'length'}],
                     'usage': {'completion_tokens': 512}}, .01)
        with patch.object(C, '_complete', return_value=response) as complete, A.review_audit('truncated') as collector:
            # A cut-off escape sequence cannot be reconstructed as valid JSON.
            complete.return_value = ('{"defects":[{"message":"unfinished\\', response[1], .01)
            with self.assertRaises(C.LLMClientError):
                C.request_json(C.LLMConfig(endpoint='http://unused', model='legalone-8b'),
                               [{'role': 'user', 'content': 'private'}], retries=2)
        self.assertEqual(complete.call_count, 1)
        self.assertEqual(collector.events[-1]['reason'], 'output_truncated')
        self.assertEqual(collector.events[-1]['details']['completion_tokens'], 512)

    def test_actual_model_identity_mismatch_is_rejected_and_audited(self):
        reply = ('{"ok":true}', {"model": "legalone-1.7b", "usage": {"prompt_tokens": 10, "completion_tokens": 4}}, .01)
        with patch.object(C, "_complete", return_value=reply), A.review_audit("identity") as collector:
            with self.assertRaisesRegex(C.LLMClientError, "不一致"):
                C.request_json(C.LLMConfig(endpoint="http://unused", model="legalone-8b"),
                               [{"role": "user", "content": "private"}], retries=0)
        self.assertEqual([e["status"] for e in collector.events], ["started", "failed"])
        self.assertEqual(collector.events[-1]["reason"], "model_identity_mismatch")
        self.assertEqual(collector.events[-1]["model"], "legalone-1.7b")
        self.assertEqual(collector.events[-1]["details"]["completion_tokens"], 4)

    def test_actual_usage_and_token_limit_are_audited(self):
        reply = ('{"ok":true}', {"model": "legalone-8b", "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                                  "choices": [{"finish_reason": "length"}]}, .01)
        with patch.object(C, "_complete", return_value=reply), A.review_audit("usage") as collector:
            result = C.request_json(C.LLMConfig(endpoint="http://unused", model="legalone-8b"),
                                    [{"role": "user", "content": "private"}], retries=0)
        self.assertEqual(result.reason_code, "token_limit")
        self.assertEqual(collector.events[-1]["status"], "degraded")
        self.assertEqual(collector.events[-1]["details"]["total_tokens"], 14)
        self.assertTrue(collector.events[-1]["details"]["model_identity_verified"])

    def test_generic_gateway_alias_to_snapshot_is_not_rejected(self):
        reply = ('{"ok":true}', {"model": "gateway-model-2026-11-01"}, .01)
        with patch.object(C, "_complete", return_value=reply):
            result = C.request_json(C.LLMConfig(endpoint="http://unused", model="gateway-alias"),
                                    [{"role": "user", "content": "private"}], retries=0)
        self.assertTrue(result.content["ok"])

    def test_http_response_format_retry_is_real_and_audited(self):
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                status = 400 if "response_format" in body else 200
                payload = {"error": "unsupported format"} if status == 400 else {"model": "legalone-8b", "choices": [{"message": {"content": '{"ok":true}'}}]}
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with A.review_audit("format") as collector:
                result = C.request_json(C.LLMConfig(endpoint=f"http://127.0.0.1:{server.server_port}/v1", model="legalone-8b"),
                                        [{"role": "user", "content": "x"}], retries=0)
            self.assertTrue(result.content["ok"])
            self.assertEqual(len(bodies), 2)
            self.assertNotIn("response_format", bodies[1])
            self.assertEqual([e["status"] for e in collector.events], ["started", "failed", "started", "degraded"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_annotate_actually_uses_selected_bert(self):
        predictor = Mock()
        predictor.predict.return_value = {"label": "contradict", "score": .9, "probabilities": {"contradict": .9, "neutral": .1}}
        with patch.object(E, "backend", return_value="bert"), patch.object(E, "_bert_predictor", return_value=predictor):
            result = E.annotate("甲方概不负责", [{"content": "应当赔偿", "article_no": "1"}])
        predictor.predict.assert_called_once()
        self.assertEqual(result[0].source, "bert_entail")

    def test_bert_failure_is_visible_and_rules_remain_available(self):
        predictor = Mock()
        predictor.predict.side_effect = RuntimeError("private input must not be persisted")
        with patch.object(E, "backend", return_value="bert"), patch.object(E, "_bert_predictor", return_value=predictor), A.review_audit("bert-failure") as collector:
            result = E.judge_pair("甲方概不负责", {"content": "应当赔偿", "article_no": "1"})
        self.assertEqual(result.source, "rules")
        self.assertEqual(collector.events[0]["reason"], "bert_prediction_failed")
        self.assertNotIn("private input", json.dumps(collector.events))

    def test_health_status_does_not_trigger_model_loading(self):
        with patch.object(E, "_bert_predictor") as loader:
            json.dumps(R.runtime_status())
        loader.assert_not_called()

    def test_neural_rule_disagreement_preserves_neural_verdict(self):
        predictor = Mock()
        predictor.predict.return_value = {"label": "neutral", "score": .999, "probabilities": {"neutral": .999, "contradict": .001}}
        with patch.object(E, "backend", return_value="bert"), patch.object(E, "_bert_predictor", return_value=predictor), A.review_audit("dispute") as collector:
            result = E.judge_pair("甲方无论因何原因违约，均不承担任何赔偿责任。", {
                "content": "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。", "article_no": "577"})
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.neural_label, "neutral")
        self.assertEqual(result.rule_label, "contradict")
        self.assertTrue(result.disagreement)
        self.assertIn("人工复核", result.note)
        self.assertTrue(any(e["reason"] == "neural_rule_disagreement" for e in collector.events))

    def test_occupied_port_prevents_second_model_load(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        llm = Mock()
        try:
            with patch.object(L, "engine", return_value=llm):
                with self.assertRaises(OSError):
                    L.serve(port=server.server_port, model_dir="legalone-8b")
            llm.load.assert_not_called()
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
