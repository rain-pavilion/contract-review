"""Native schema transport/config/audit fixtures; never call an actual model."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import start_web as W
from zhilv_qiling_v2 import api, llm_client as L, model_audit as A
from zhilv_qiling_v2.config import Config, config_from_env


class Response:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.value, ensure_ascii=False).encode("utf-8")


def envelope(value):
    return {"model": "legalone-8b", "choices": [{"finish_reason": "stop", "message": {
        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}}]}


def schema():
    return {"type": "object", "properties": {"verifications": {"type": "array", "minItems": 0,
        "maxItems": 1, "items": {"type": "object", "properties": {
            "feedback_item_id": {"type": "string", "enum": ["private-fixture-id"]},
            "verdict": {"type": "string", "enum": ["confirmed", "rejected", "needs_manual"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 40},
            "quote": {"type": "string", "minLength": 1, "maxLength": 80}},
            "required": ["feedback_item_id", "verdict", "reason", "quote"], "additionalProperties": False}}},
        "required": ["verifications"], "additionalProperties": False}


def acknowledgement():
    return {"verifications": [{"feedback_item_id": "private-fixture-id", "verdict": "confirmed",
        "reason": "合成测试结构已核对", "quote": "仅为结构协议测试"}]}


class NativeSchemaTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.dict(os.environ, {"ZLQL_MODEL_AUDIT_PATH": str(Path(self.temp.name)/"audit.sqlite")})
        patcher.start(); self.addCleanup(patcher.stop)
        self.config = L.LLMConfig(endpoint="http://unused/v1", model="legalone-8b", native_json_schema=True)
        self.messages = [{"role": "user", "content": "合成输入，不是真实合同"}]

    def run_reply(self, value, *, config=None, supplied_schema=None, retries=0):
        captured = []
        def transport(request, timeout):
            captured.append(json.loads(request.data))
            return Response(envelope(value))
        with patch.object(L.urllib.request, "urlopen", side_effect=transport), A.review_audit("schema-fixture") as audit:
            pair = L.request_json(config or self.config, self.messages,
                json_schema=schema() if supplied_schema is None else supplied_schema, retries=retries)
        return pair, captured, audit.events

    def test_explicit_flag_sends_exact_native_schema_without_prompt_changes(self):
        pair, sent, events = self.run_reply(acknowledgement())
        self.assertEqual(sent[0]["response_format"], {"type": "json_object", "schema": schema()})
        self.assertEqual(sent[0]["messages"], self.messages)
        self.assertEqual(pair.content, acknowledgement())
        self.assertEqual(events[-1]["status"], "succeeded")
        self.assertTrue(events[-1]["details"]["native_json_schema_sent"])

    def test_generic_endpoint_does_not_send_schema_even_when_argument_supplied(self):
        config = L.LLMConfig(endpoint=self.config.endpoint, model=self.config.model)
        _, sent, events = self.run_reply({"defects": []}, config=config)
        self.assertEqual(sent[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("schema", sent[0]["response_format"])
        self.assertTrue(events[-1]["details"]["native_json_schema_requested"])
        self.assertFalse(events[-1]["details"]["native_json_schema_sent"])
        self.assertEqual(events[-1]["details"]["json_schema_sha256"], "")

    def test_truthy_non_boolean_flag_cannot_enable_native_extension(self):
        self.config.native_json_schema = "true"
        _, sent, _ = self.run_reply({"defects": []})
        self.assertNotIn("schema", sent[0]["response_format"])

    def test_native_flag_without_per_call_schema_preserves_ordinary_json_request(self):
        with patch.object(L.urllib.request, "urlopen", return_value=Response(envelope({"ok": True}))) as transport:
            pair = L.request_json(self.config, self.messages, retries=0)
        self.assertEqual(pair.content, {"ok": True})
        self.assertEqual(json.loads(transport.call_args.args[0].data)["response_format"], {"type": "json_object"})

    def test_schema_takes_precedence_even_when_ordinary_response_format_disabled(self):
        self.config.response_format = ""
        _, sent, _ = self.run_reply(acknowledgement())
        self.assertEqual(sent[0]["response_format"], {"type": "json_object", "schema": schema()})

    def test_schema_rejection_never_retries_without_the_schema(self):
        for status in (400, 422):
            with self.subTest(status=status):
                error = urllib.error.HTTPError("http://unused", status, "rejected", {},
                    io.BytesIO(b"unsupported response_format schema"))
                with patch.object(L.urllib.request, "urlopen", side_effect=error) as transport, A.review_audit("rejected") as audit:
                    with self.assertRaises(L.LLMClientError):
                        L.request_json(self.config, self.messages, json_schema=schema(), retries=2)
                transport.assert_called_once()
                self.assertEqual(audit.events[-1]["reason"], "native_json_schema_rejected")
                self.assertTrue(all(event["details"]["native_json_schema_sent"] for event in audit.events))

    def test_ordinary_endpoint_still_has_existing_response_format_fallback(self):
        error = urllib.error.HTTPError("http://unused", 400, "unsupported", {}, io.BytesIO(b"unsupported response_format"))
        config = L.LLMConfig(endpoint=self.config.endpoint, model=self.config.model)
        with patch.object(L.urllib.request, "urlopen", side_effect=[error, Response(envelope({"ok": True}))]) as transport:
            pair = L.request_json(config, self.messages, json_schema=schema(), retries=0)
        self.assertEqual(transport.call_count, 2)
        self.assertNotIn("response_format", json.loads(transport.call_args_list[1].args[0].data))
        self.assertEqual(pair.degraded, "response_format_retry")

    def test_parse_retry_keeps_the_same_schema_and_audit_hash(self):
        with patch.object(L.urllib.request, "urlopen", side_effect=[Response(envelope("not json")),
                Response(envelope(acknowledgement()))]) as transport, A.review_audit("retry") as audit:
            pair = L.request_json(self.config, self.messages, json_schema=schema(), retries=1)
        sent = [json.loads(call.args[0].data) for call in transport.call_args_list]
        self.assertEqual([body["response_format"]["schema"] for body in sent], [schema(), schema()])
        self.assertEqual(pair.degraded, "json_retry")
        hashes = {event["details"]["json_schema_sha256"] for event in audit.events}
        self.assertEqual(len(hashes), 1)

    def test_real_request_budget_limits_parse_retries_and_exposes_failed_receipt(self):
        with patch.object(L.urllib.request, "urlopen", return_value=Response(envelope("not json"))) as transport, \
                A.review_audit("limited-retry") as audit:
            with self.assertRaises(L.LLMClientError) as caught:
                L.request_json(self.config, self.messages, json_schema=schema(), retries=2,
                               max_transport_attempts=1)
        transport.assert_called_once()
        self.assertEqual(caught.exception.call_id, audit.events[0]["details"]["call_id"])
        self.assertEqual(len(caught.exception.request_receipts), 1)
        self.assertEqual(caught.exception.request_receipts[0]["status"], "failed")

    def test_real_request_budget_limits_ordinary_format_fallback(self):
        error = urllib.error.HTTPError("http://unused", 400, "unsupported", {}, io.BytesIO(b"unsupported response_format"))
        config = L.LLMConfig(endpoint=self.config.endpoint, model=self.config.model)
        with patch.object(L.urllib.request, "urlopen", side_effect=error) as transport:
            with self.assertRaises(L.LLMClientError) as caught:
                L.request_json(config, self.messages, retries=2, max_transport_attempts=1)
        transport.assert_called_once()
        self.assertTrue(caught.exception.call_id)
        self.assertEqual(len(caught.exception.request_receipts), 1)

    def test_successful_parse_retry_retains_both_actual_request_receipts(self):
        with patch.object(L.urllib.request, "urlopen", side_effect=[Response(envelope("not json")),
                Response(envelope(acknowledgement()))]), A.review_audit("retained-retry") as audit:
            pair = L.request_json(self.config, self.messages, json_schema=schema(), retries=1,
                                  max_transport_attempts=2)
        self.assertEqual([row["status"] for row in pair.request_receipts], ["failed", "degraded"])
        self.assertEqual(pair.call_id, pair.request_receipts[-1]["call_id"])
        self.assertEqual({row["call_id"] for row in pair.request_receipts},
                         {event["details"]["call_id"] for event in audit.events})

    def test_ignored_schema_wrong_root_extra_fields_and_unbound_ids_fail(self):
        invalid = [{"defects": []}, {**acknowledgement(), "defects": []},
            {"verifications": {"feedback_item_id": "private-fixture-id"}}]
        wrong_id = acknowledgement(); wrong_id["verifications"][0]["feedback_item_id"] = "another-private-id"
        invalid.append(wrong_id)
        for value in invalid:
            with self.subTest(value=value), patch.object(L.urllib.request, "urlopen", return_value=Response(envelope(value))):
                with self.assertRaises(L.LLMClientError):
                    L.request_json(self.config, self.messages, json_schema=schema(), retries=0)

    def test_overlong_reason_quote_or_array_fails_local_schema(self):
        for field, size in (("reason", 41), ("quote", 81)):
            value = acknowledgement(); value["verifications"][0][field] = "测"*size
            with self.subTest(field=field), self.assertRaises(L.LLMClientError):
                self.run_reply(value)
        value = acknowledgement(); value["verifications"] *= 2
        with self.assertRaises(L.LLMClientError): self.run_reply(value)

    def test_strict_native_path_does_not_repair_truncated_or_aliased_json(self):
        with self.assertRaises(L.LLMClientError): self.run_reply('{"verifications":[')
        custom = {"type": "object", "properties": {"risk_code": {"type": "string"}},
            "required": ["risk_code"], "additionalProperties": False}
        pair, _, _ = self.run_reply({"risk_code": "fixture"}, supplied_schema=custom)
        self.assertEqual(pair.content, {"risk_code": "fixture"})

    def test_audit_persists_only_schema_boolean_and_canonical_hash(self):
        _, _, events = self.run_reply(acknowledgement())
        expected = hashlib.sha256(json.dumps(schema(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(events[-1]["details"]["json_schema_sha256"], expected)
        text = json.dumps(events, ensure_ascii=False)
        self.assertNotIn("private-fixture-id", text)
        self.assertNotIn("verifications", text)
        self.assertNotIn("合成输入", text)

    def test_bad_schema_root_or_non_json_value_is_rejected_before_transport(self):
        for value in ([1], {"type": "array"}, {"type": "object", "private": object()}):
            with self.subTest(value=value), patch.object(L.urllib.request, "urlopen") as transport:
                with self.assertRaises(L.LLMClientError):
                    L.request_json(self.config, self.messages, json_schema=value, retries=0)
                transport.assert_not_called()

    def test_local_schema_subset_checks_boolean_integer_required_and_unknown_keys(self):
        for value, specification in ((True, {"type": "integer"}), ({}, {"type": "object", "required": ["missing"]}),
                ("x", {"type": ["string"]}), ("x", {"pattern": "x"}), ("x", {"type": "string", "minLength": -1})):
            with self.subTest(specification=specification), self.assertRaises(L.LLMClientError):
                L._validate_json_schema(value, specification)


class NativeSchemaConfigurationTests(unittest.TestCase):
    def test_defaults_and_environment_are_explicit(self):
        self.assertFalse(Config().llm_native_json_schema)
        self.assertFalse(L.LLMConfig().native_json_schema)
        for raw, expected in (("1", True), ("true", True), ("ON", True), ("0", False),
                              ("false", False), ("garbage", False), ("", False)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"ZLQL_LLM_NATIVE_JSON_SCHEMA": raw}, clear=True):
                self.assertEqual(config_from_env().llm_native_json_schema, expected)
                self.assertEqual(L.llm_config_from_env().native_json_schema, expected)

    def test_env_prefix_and_redacted_configuration(self):
        with patch.dict(os.environ, {"TEST_LLM_NATIVE_JSON_SCHEMA": "1"}, clear=True):
            self.assertTrue(L.llm_config_from_env("TEST").native_json_schema)
        config = L.LLMConfig(api_key="private-key", native_json_schema=True)
        self.assertTrue(config.redacted()["native_json_schema"])
        self.assertNotIn("private-key", str(config.redacted()))

    def test_verified_llama_backend_is_the_only_automatic_enablement(self):
        model = {"id": W.LEGALONE_ALIAS, "meta": {"n_params": W.LEGALONE_PARAMETERS}}
        with patch.object(W.urllib.request, "urlopen", side_effect=[Response({"status": "ok"}), Response({"data": [model]})]):
            verified = W.local_model_status()
        transformer = {"id": W.LEGALONE_ALIAS, "ready": True, "backend": "llama.cpp", "_native_identity_verified": True}
        with patch.object(W.urllib.request, "urlopen", return_value=Response({"data": [transformer]})):
            unverified = W.local_model_status()
        self.assertNotIn("_native_identity_verified", unverified)
        args = SimpleNamespace(model_port=8620, with_model=True)
        for status, expected in ((verified, True), (unverified, False),
                                  ({"id": W.LEGALONE_ALIAS, "ready": True}, False)):
            with self.subTest(status=status), patch.dict(os.environ, {}, clear=True), \
                    patch.object(W, "local_model_status", return_value=status), patch("sys.stdout", new_callable=io.StringIO):
                W.connect_local_model(args)
                self.assertEqual(os.environ.get("ZLQL_LLM_NATIVE_JSON_SCHEMA") == "1", expected)

    def test_launcher_preserves_explicit_opt_out(self):
        status = {"ready": True, "backend": "llama.cpp", "_native_identity_verified": True}
        with patch.dict(os.environ, {"ZLQL_LLM_NATIVE_JSON_SCHEMA": "0"}, clear=True), \
                patch.object(W, "local_model_status", return_value=status), patch("sys.stdout", new_callable=io.StringIO):
            W.connect_local_model(SimpleNamespace(model_port=8620, with_model=True))
            self.assertEqual(os.environ["ZLQL_LLM_NATIVE_JSON_SCHEMA"], "0")

    def test_health_exposes_flag_without_loading_a_model(self):
        from zhilv_qiling_v2 import config as C, learned, model_runtime
        with tempfile.TemporaryDirectory() as temporary, patch.object(C, "MODEL_DIR", Path(temporary)), \
                patch.object(api, "capabilities", return_value=SimpleNamespace(to_dict=lambda: {})), \
                patch.object(learned, "registry", return_value=SimpleNamespace(summary=lambda: {})), \
                patch.object(model_runtime, "runtime_status", return_value={}):
            for enabled in (False, True):
                config = Config(); config.llm_native_json_schema = enabled
                value = api.handle_health(SimpleNamespace(config=config, trace_id="fixture"))
                self.assertIs(value["llm"]["native_json_schema"], enabled)


if __name__ == "__main__":
    unittest.main()
