"""Supervisor contracts: executed routes, bounded planning and safe fallbacks."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.config import Config, config_from_env
from zhilv_qiling_v2.llm_client import LLMClientError, LLMPair
from zhilv_qiling_v2.supervisor import Supervisor


def planner_config(**values):
    defaults = dict(enable_llm=True, planner_endpoint="https://planner.example/v1",
                    planner_model="planner-test", planner_api_key="test-planner-secret",
                    llm_endpoint="http://localhost:9001/v1", llm_model="legal-default",
                    llm_api_key="test-default-secret", llm_strong_endpoint="http://localhost:9002/v1",
                    llm_strong_model="legal-strong", planner_max_calls=2)
    defaults.update(values)
    return Config(**defaults)


def output(**values):
    payload = {"action": "continue", "reason": "Use the legal specialist.",
               "assignments": [{"agent": "logic", "model_id": "default", "reason": "Initial review"}],
               "clause_ids": []}
    payload.update(values)
    return LLMPair(payload, "planner-test", 0.01)


class SupervisorTest(unittest.TestCase):
    def test_named_route_changes_real_execution_configuration_without_mutating_base(self):
        config = planner_config(model_pool_json=json.dumps({"cloud": {
            "endpoint": "https://expert.example/v1", "model": "legal-cloud",
            "agents": ["logic", "reviewer"], "api_key_env": "TEST_EXPERT_KEY",
            "timeout": 25, "max_tokens": 1300}}), llm_native_json_schema=True,
            llm_native_thinking="off", llm_native_context_preflight=True)
        with patch.dict(os.environ, {"TEST_EXPERT_KEY": "test-expert-secret"}):
            supervisor = Supervisor(config)
            selected = supervisor.config_for("logic", "cloud")
        self.assertEqual(selected.llm_endpoint, "https://expert.example/v1")
        self.assertEqual(selected.llm_model, "legal-cloud")
        self.assertEqual(selected.llm_api_key, "test-expert-secret")
        self.assertEqual(selected.llm_timeout, 25)
        self.assertEqual(selected.llm_max_tokens, 1300)
        self.assertFalse(selected.llm_native_json_schema)
        self.assertFalse(selected.llm_native_context_preflight)
        self.assertEqual(selected.llm_native_thinking, "")
        self.assertEqual(config.llm_model, "legal-default")

    def test_unknown_route_and_wrong_capability_cannot_execute(self):
        supervisor = Supervisor(planner_config())
        with self.assertRaises(ValueError):
            supervisor.config_for("logic", "unregistered")
        with self.assertRaises(ValueError):
            supervisor.config_for("ner", "default")

    def test_llm_decision_selects_registered_model_and_keeps_ordinal_metadata(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output(assignments=[
                {"agent": "logic", "model_id": "strong", "reason": "Difficult clause"}])):
            decision = supervisor.decide("initial", {"clause_ids": ["c1", "c2"]})
        self.assertEqual(decision["source"], "llm")
        self.assertEqual(decision["stage"], "initial")
        self.assertEqual(decision["assignments"][0]["model_id"], "strong")
        self.assertTrue(decision["decision_id"])
        self.assertEqual(supervisor.snapshot()["calls"], 1)

    def test_unregistered_model_response_falls_back_to_rules(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output(assignments=[
                {"agent": "logic", "model_id": "invented-model", "reason": "Try it"}])):
            decision = supervisor.decide("initial", {"clause_ids": ["c1"]})
        self.assertEqual(decision["source"], "rules")
        self.assertEqual(decision["assignments"][0]["model_id"], "default")
        self.assertIn("planner_invalid_decision", supervisor.snapshot()["degraded"])

    def test_planner_cannot_inject_endpoint_or_skip_business_node(self):
        for extra in ({"endpoint": "https://attacker.example"}, {"skip": ["statute"]}):
            supervisor = Supervisor(planner_config())
            reply = output()
            reply.content.update(extra)
            with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=reply):
                result = supervisor.decide("initial", {"clause_ids": ["c1"]})
            self.assertEqual(result["source"], "rules")

    def test_unknown_clause_and_duplicate_agent_are_rejected(self):
        for reply in (output(clause_ids=["fabricated"]), output(assignments=[
                {"agent": "logic", "model_id": "default", "reason": "first"},
                {"agent": "logic", "model_id": "strong", "reason": "second"}])):
            supervisor = Supervisor(planner_config())
            with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=reply):
                result = supervisor.decide("review", {"clause_ids": ["c1"]})
            self.assertEqual(result["source"], "rules")

    def test_review_requires_real_scope_and_review_capable_assignment(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output(
                action="review", clause_ids=[], assignments=[])):
            result = supervisor.decide("review", {"clause_ids": ["c1"]})
        self.assertEqual(result["source"], "rules")

    def test_manual_decision_retains_request_without_declaring_completion(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output(
                action="manual", assignments=[], clause_ids=["c1"], reason="Evidence unavailable")):
            result = supervisor.decide("feedback", {"clause_ids": ["c1"], "pending": True})
        self.assertEqual(result["action"], "manual")
        self.assertEqual(result["source"], "llm")
        self.assertNotIn("converged", result)
        self.assertNotIn("complete", result)

    def test_failed_planner_falls_back_without_echoing_exception_or_key(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", side_effect=LLMClientError(
                "Authorization test-planner-secret test-default-secret")):
            decision = supervisor.decide("initial", {"clause_ids": ["c1"]})
        self.assertEqual(decision["source"], "rules")
        data = json.dumps(supervisor.snapshot())
        self.assertNotIn("test-planner-secret", data)
        self.assertNotIn("test-default-secret", data)
        self.assertIn("planner_request_failed", data)

    def test_planner_budget_counts_failures_and_never_sends_third_request(self):
        supervisor = Supervisor(planner_config(planner_max_calls=2))
        with patch("zhilv_qiling_v2.supervisor.L.request_json", side_effect=LLMClientError("timeout")) as request:
            for _ in range(3):
                supervisor.decide("feedback", {"clause_ids": ["c1"]})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(supervisor.snapshot()["calls"], 2)
        self.assertIn("planner_budget_exhausted", supervisor.snapshot()["degraded"])

    def test_disabled_llm_and_missing_planner_never_call_network(self):
        for config in (planner_config(enable_llm=False), Config(enable_llm=True),
                       planner_config(enable_planner=False)):
            with patch("zhilv_qiling_v2.supervisor.L.request_json") as request:
                supervisor = Supervisor(config)
                result = supervisor.decide("initial", {"clause_ids": ["c1"]})
            self.assertEqual(result["source"], "rules")
            self.assertEqual(request.call_count, 0)

    def test_rules_review_prioritizes_high_risk_and_caps_scope(self):
        supervisor = Supervisor(planner_config(enable_planner=False, llm_review_top_k=1))
        result = supervisor.decide("review", {"clause_ids": ["c1", "c2"], "defects": [
            {"severity": "low", "confidence": 0.9, "clause_ids": ["c1"]},
            {"severity": "high", "confidence": 0.6, "clause_ids": ["c2"]}]})
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["clause_ids"], ["c2"])
        self.assertEqual(result["assignments"][0]["model_id"], "strong")

    def test_public_registry_and_prompt_never_contain_endpoint_or_secrets(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output()) as request:
            supervisor.decide("initial", {"clause_ids": ["c1"], "api_key": "test-default-secret"})
        prompt = json.dumps(request.call_args.args[1], ensure_ascii=False)
        public = json.dumps(supervisor.snapshot(), ensure_ascii=False)
        for value in ("test-planner-secret", "test-default-secret", "https://planner.example",
                      "http://localhost:9001", "http://localhost:9002"):
            self.assertNotIn(value, prompt)
            self.assertNotIn(value, public)

    def test_context_summary_and_transport_attempts_are_bounded(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output()) as request:
            supervisor.decide("initial", {"clause_ids": ["c1"], "text": "合同内容" * 50000,
                                          "errors": ["x" * 10000] * 100})
        self.assertLess(len(json.dumps(request.call_args.args[1], ensure_ascii=False)), 18000)
        self.assertEqual(request.call_args.kwargs["max_transport_attempts"], 1)
        self.assertEqual(request.call_args.kwargs["retries"], 0)

    def test_large_capability_pool_still_receives_complete_json_summary(self):
        pool = {f"expert{i}": {"endpoint": "https://expert.example/v1", "model": "m" * 200,
                "description": "擅长复杂合同" * 80, "agents": ["logic"]} for i in range(24)}
        supervisor = Supervisor(planner_config(model_pool_json=json.dumps(pool)))
        with patch("zhilv_qiling_v2.supervisor.L.request_json", return_value=output()) as request:
            supervisor.decide("initial", {"clause_ids": ["c" + str(i) + "x" * 190 for i in range(120)],
                "defects": [{"message": "x" * 1000, "clause_ids": ["a" * 180] * 8}] * 24})
        prompt = request.call_args.args[1][1]["content"]
        self.assertLessEqual(len(prompt), 15000)
        self.assertIsInstance(json.loads(prompt), dict)

    def test_missing_named_credential_is_not_an_executable_route(self):
        with patch.dict(os.environ, {}, clear=True):
            supervisor = Supervisor(planner_config(model_pool_json=json.dumps({"cloud": {
                "endpoint": "https://expert.example/v1", "model": "expert", "api_key_env": "UNSET_KEY"}})))
        with self.assertRaises(ValueError):
            supervisor.config_for("logic", "cloud")
        self.assertFalse(next(row for row in supervisor.snapshot()["model_pool"]
                              if row["model_id"] == "cloud")["ready"])

    def test_unexpected_transport_failure_still_preserves_rules_workflow(self):
        supervisor = Supervisor(planner_config())
        with patch("zhilv_qiling_v2.supervisor.L.request_json", side_effect=RuntimeError("transport broke")):
            decision = supervisor.decide("initial", {"clause_ids": ["c1"]})
        self.assertEqual(decision["source"], "rules")
        self.assertEqual(decision["assignments"][0]["model_id"], "default")

    def test_registry_with_plaintext_key_or_credential_url_is_not_registered(self):
        for route in ({"endpoint": "https://host/v1", "api_key": "test-raw-secret"},
                      {"endpoint": "https://user:password@host/v1"}):
            route.update(model="specialist", agents=["logic"])
            supervisor = Supervisor(planner_config(model_pool_json=json.dumps({"unsafe": route})))
            with self.assertRaises(ValueError):
                supervisor.config_for("logic", "unsafe")
            self.assertNotIn("test-raw-secret", json.dumps(supervisor.snapshot()))
            self.assertIn("model_pool_invalid", supervisor.snapshot()["degraded"])

    def test_snapshot_cannot_mutate_execution_history(self):
        supervisor = Supervisor(Config())
        supervisor.decide("initial", {"clause_ids": ["c1"]})
        public = supervisor.snapshot()
        public["decisions"][0]["reason"] = "changed"
        self.assertNotEqual(supervisor.snapshot()["decisions"][0]["reason"], "changed")

    def test_invalid_stage_is_rejected_before_any_request(self):
        with patch("zhilv_qiling_v2.supervisor.L.request_json") as request:
            with self.assertRaises(ValueError):
                Supervisor(planner_config()).decide("skip-all", {})
        self.assertEqual(request.call_count, 0)

    def test_environment_configures_planner_independently_of_expert(self):
        with patch.dict(os.environ, {"ZLQL_PLANNER_ENDPOINT": "https://planner.example/v1",
                                    "ZLQL_PLANNER_MODEL": "planner-test",
                                    "ZLQL_PLANNER_MAX_CALLS": "3", "ZLQL_ENABLE_PLANNER": "0",
                                    "ZLQL_MODEL_POOL": "{}"}):
            config = config_from_env(Config())
        self.assertEqual(config.planner_model, "planner-test")
        self.assertEqual(config.planner_max_calls, 3)
        self.assertFalse(config.enable_planner)
        self.assertEqual(config.llm_endpoint, "")

    def test_primary_credential_is_not_forwarded_to_strong_endpoint(self):
        supervisor = Supervisor(planner_config(llm_strong_api_key=""))
        selected = supervisor.config_for("reviewer", "strong")
        self.assertEqual(selected.llm_api_key, "")
        self.assertEqual(selected.llm_endpoint, "http://localhost:9002/v1")
        self.assertTrue(selected.llm_route_explicit)

    def test_default_route_preserves_environment_fallback_semantics(self):
        supervisor = Supervisor(planner_config())
        self.assertFalse(supervisor.config_for("logic", "default").llm_route_explicit)
        explicitly_bound = Supervisor(planner_config(llm_route_explicit=True))
        self.assertTrue(explicitly_bound.config_for("logic", "default").llm_route_explicit)

    def test_strong_route_uses_explicit_independent_credential(self):
        supervisor = Supervisor(planner_config(llm_strong_api_key="test-strong-secret"))
        selected = supervisor.config_for("reviewer", "strong")
        self.assertEqual(selected.llm_api_key, "test-strong-secret")
        self.assertNotIn("test-strong-secret", json.dumps(supervisor.snapshot()))

    def test_planner_only_environment_enables_model_channel_unless_explicitly_disabled(self):
        env = {"ZLQL_PLANNER_ENDPOINT": "https://planner.example/v1", "ZLQL_PLANNER_MODEL": "planner"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(config_from_env(Config()).enable_llm)
        with patch.dict(os.environ, {**env, "ZLQL_ENABLE_LLM": "0"}, clear=True):
            self.assertFalse(config_from_env(Config(enable_llm=True)).enable_llm)


if __name__ == "__main__":
    unittest.main()
