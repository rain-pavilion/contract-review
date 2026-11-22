"""Ensure unavailable capacity never produces a fabricated model call or pass."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from zhilv_qiling_v2 import llm_client as L, model_audit as A, nodes
from zhilv_qiling_v2.config import config_from_env


class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.value).encode()


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = patch.dict(os.environ, {"ZLQL_MODEL_AUDIT_PATH": str(Path(self.temp.name)/"audit.sqlite")})
        p.start(); self.addCleanup(p.stop)
        self.config = L.LLMConfig(endpoint="http://native/v1", model="legalone-8b", max_tokens=20,
                                  native_context_preflight=True)
        self.messages = [{"role": "user", "content": "完整合同及未决事项"}]
        self.routes = []
        self.bodies = []
        self.context = 100
        self.tokens = 79
        self.generations = 0
        self.invalid_first = False

    def transport(self, request, timeout):
        route = request.full_url.replace("http://native", "")
        self.routes.append(route)
        body = json.loads(request.data) if request.data else None
        self.bodies.append(body)
        if route == "/v1/models":
            return Response({"data": [{"id": "legalone-8b", "meta": {"n_ctx": self.context}}]})
        if route == "/apply-template": return Response({"prompt": json.dumps(body, ensure_ascii=False)})
        if route == "/tokenize": return Response({"tokens": [1] * self.tokens})
        if route == "/v1/chat/completions":
            self.generations += 1
            value = "invalid" if self.invalid_first and self.generations == 1 else '{"ok":true}'
            return Response({"model": "legalone-8b", "choices": [{"finish_reason": "stop",
                             "message": {"content": value}}]})
        self.fail(route)

    def test_exact_fit_preserves_request_and_binds_real_count(self):
        with patch.object(L.urllib.request, "urlopen", side_effect=self.transport), A.review_audit("fit") as audit:
            pair = L.request_json(self.config, self.messages)
        self.assertEqual(self.generations, 1)
        self.assertEqual(self.bodies[1], self.bodies[3])
        self.assertEqual(self.bodies[3]["messages"], self.messages)
        self.assertEqual(self.bodies[3]["max_tokens"], 20)
        self.assertTrue(pair.request_receipts[0]["generation_sent"])
        self.assertEqual(audit.events[-1]["details"]["context_required_context_tokens"], 100)

    def test_one_token_over_never_generates_retries_or_drops_sources(self):
        self.tokens = 80
        with patch.object(L.urllib.request, "urlopen", side_effect=self.transport), A.review_audit("over") as audit:
            with self.assertRaises(L.LLMContextError) as caught:
                L.request_json(self.config, self.messages, retries=4)
        self.assertEqual(self.generations, 0)
        self.assertEqual(self.messages[0]["content"], "完整合同及未决事项")
        self.assertEqual(len(audit.events), 1)
        self.assertEqual(audit.events[0]["status"], "skipped")
        self.assertEqual(audit.events[0]["reason"], "context_budget_exceeded")
        self.assertFalse(caught.exception.request_receipts[0]["generation_sent"])
        self.assertTrue(caught.exception.call_id)

    def test_missing_context_and_failed_tokenizer_fail_closed(self):
        for failure in ("context", "tokenizer"):
            with self.subTest(failure=failure):
                self.context = None if failure == "context" else 100
                def fail(request, timeout):
                    if request.full_url.endswith("/tokenize"):
                        raise urllib.error.URLError("unavailable")
                    return self.transport(request, timeout)
                with patch.object(L.urllib.request, "urlopen", side_effect=fail):
                    with self.assertRaises(L.LLMContextError) as caught:
                        L.request_json(self.config, self.messages)
                self.assertEqual(caught.exception.reason, "context_preflight_unavailable")
                self.assertEqual(self.generations, 0)

    def test_retry_is_recounted_and_can_be_blocked(self):
        self.invalid_first = True
        def grow(request, timeout):
            if self.generations: self.tokens = 80
            return self.transport(request, timeout)
        with patch.object(L.urllib.request, "urlopen", side_effect=grow):
            with self.assertRaises(L.LLMContextError) as caught:
                L.request_json(self.config, self.messages, retries=2)
        self.assertEqual(self.generations, 1)
        receipts = caught.exception.request_receipts
        self.assertEqual([r["generation_sent"] for r in receipts], [True, False])
        self.assertEqual(self.routes.count("/tokenize"), 2)

    def test_generic_or_nonboolean_opt_in_never_uses_native_routes(self):
        for enabled in (False, "true"):
            self.config.native_context_preflight = enabled
            self.routes.clear()
            with patch.object(L.urllib.request, "urlopen", side_effect=self.transport):
                L.request_json(self.config, self.messages)
            self.assertEqual(self.routes, ["/v1/chat/completions"])

    def test_native_schema_and_thinking_are_counted_with_same_template(self):
        self.config.native_json_schema = True
        self.config.native_thinking = "off"
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}
        with patch.object(L.urllib.request, "urlopen", side_effect=self.transport):
            L.request_json(self.config, self.messages, json_schema=schema)
        self.assertEqual(self.bodies[1], self.bodies[3])
        self.assertEqual(self.bodies[1]["response_format"]["schema"], schema)
        self.assertEqual(self.bodies[1]["chat_template_kwargs"], {"enable_thinking": False})

    def test_environment_reaches_node_config(self):
        with patch.dict(os.environ, {"ZLQL_LLM_NATIVE_CONTEXT_PREFLIGHT": "1"}):
            self.assertTrue(L.llm_config_from_env().native_context_preflight)
            self.assertTrue(nodes.llm_settings(config_from_env()).native_context_preflight)


if __name__ == "__main__": unittest.main()
