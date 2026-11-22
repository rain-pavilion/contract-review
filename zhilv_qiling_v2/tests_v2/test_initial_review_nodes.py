"""Actual node/transport integration using only offline synthetic responses."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import llm_client as L, model_audit as AUDIT, nodes as N
from zhilv_qiling_v2.config import Config, config_from_env
from zhilv_qiling_v2.long_context import build_review_plan
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse
from tools.verify_initial_review_lineage import verify


def clause(cid="a", text="约定付款条件应与验收记录一致。"):
    return Clause(cid, "", "", 1, "", text, text, ["block-" + cid])


def finding(cid="a", quote="约定付款条件应与验收记录一致。", **fields):
    return {"clause_id": cid, "code": "fee_mismatch", "severity": "medium",
            "message": "需核对付款条件。", "quote": quote, "related_quotes": [], **fields}


def page(defects=(), more=False, corrections=None):
    result = {"defects": list(defects), "more_findings_possible": more}
    if corrections is not None:
        result["corrections"] = corrections
    return result


class InitialReviewNodeTests(unittest.TestCase):
    def review(self, responses, rows=None, **settings):
        rows = rows or [clause()]
        config = Config(enable_llm=True, llm_endpoint="http://127.0.0.1:1/v1", llm_model="legalone-8b",
                        llm_native_json_schema=True, llm_chunk_chars=6000, llm_chunk_overlap=0,
                        llm_cross_clause_pairs=0, llm_max_tokens=4096)
        for key, value in settings.items():
            setattr(config, key, value)
        contract = Contract("offline-continuation", DocumentParse(
            "synthetic-continuation.txt", "txt", "\n".join(row.text for row in rows), clauses=rows))
        captured = []

        def complete(_config, messages, _max_tokens, **kwargs):
            payload = json.loads(messages[-1]["content"])
            captured.append({"payload": payload, "schema": copy.deepcopy(kwargs.get("json_schema")),
                             "messages": copy.deepcopy(messages)})
            response = responses(payload, len(captured)) if callable(responses) else responses[len(captured)-1]
            if isinstance(response, Exception):
                raise response
            content, finish = response if isinstance(response, tuple) else (response, "stop")
            return json.dumps(content, ensure_ascii=False), {
                "model": "legalone-8b", "choices": [{"finish_reason": finish}],
                "usage": {"completion_tokens": 100}}, .001

        with TemporaryDirectory() as directory, patch.dict(os.environ, {
                "ZLQL_MODEL_AUDIT_PATH": str(Path(directory) / "audit.sqlite")}):
            with patch.object(L, "_complete", side_effect=complete):
                with AUDIT.review_audit("offline-continuation-trace", contract.contract_id) as audit:
                    candidates, meta, degraded = N._llm_logic(contract, config, rows)
        windows, _ = build_review_plan(rows, chunk_chars=config.llm_chunk_chars,
                                      overlap=config.llm_chunk_overlap, cross_pairs=config.llm_cross_clause_pairs)
        return candidates, meta, degraded, captured, audit.events, [w.as_dict() for w in windows], [r.to_dict() for r in rows]

    def checked(self, result):
        _, meta, _, _, events, windows, rows = result
        return verify(meta, windows, rows, events,
                      trace_id="offline-continuation-trace", contract_id="offline-continuation")

    def test_second_real_call_completes_same_source_and_retains_first_finding(self):
        result = self.review([page([finding()], more=True), page(corrections=[])])
        candidates, meta, degraded, sent, events, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual((meta["planned_calls"], meta["completed_calls"], meta["actual_attempt_calls"]), (1, 1, 2))
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].evidence[0].extra["partial_model_output"])
        self.assertEqual(meta["initial_output_pending"], [])
        self.assertFalse(any("more_findings_possible" in item for item in degraded))
        self.assertEqual(sent[0]["payload"]["segments"], sent[1]["payload"]["segments"])
        history = sent[1]["payload"]["initial_review_continuation"]
        self.assertEqual(history["prior_attempts"][0]["call_id"], meta["calls"][0]["call_id"])
        self.assertEqual(history["seen_findings"][0]["defect"], finding())
        self.assertNotEqual(meta["calls"][0]["job_id"], meta["calls"][1]["job_id"])
        self.assertEqual(meta["calls"][0]["source_sha256"], meta["calls"][1]["source_sha256"])
        self.assertEqual(len({event["details"]["call_id"] for event in events}), 2)
        checked = self.checked(result)
        self.assertTrue(checked["protocol_lineage_acceptance_passed"], checked)
        self.assertFalse(checked["legal_accuracy_evaluated"])

    def test_empty_later_pages_cannot_clear_prior_wrong_quote(self):
        result = self.review([page([finding(quote="原文不存在。")]), page(corrections=[]), page(corrections=[])])
        candidates, meta, degraded, sent, _, _, _ = result
        self.assertEqual(candidates, [])
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["actual_attempt_calls"], 3)
        lineage = meta["initial_review_lineage"][0]
        self.assertEqual(len(lineage["pending_findings"]), 1)
        self.assertEqual(lineage["status"], "budget_exhausted")
        self.assertIn("initial_output_prior_findings_unrepaired", lineage["pending_reasons"])
        self.assertTrue(sent[2]["payload"]["initial_review_continuation"]["pending_findings"])
        checked = self.checked(result)
        self.assertTrue(checked["checks"]["completion_and_pending_replay_match"], checked)
        self.assertFalse(checked["protocol_lineage_acceptance_passed"])

    def test_explicit_model_correction_retains_correct_attempt_provenance(self):
        def response(payload, index):
            if index == 1:
                return page([finding(quote="原文不存在。")])
            repair_id = payload["initial_review_continuation"]["pending_findings"][0]["finding_id"]
            return page(corrections=[{"prior_finding_id": repair_id, "defect": finding()}])
        result = self.review(response)
        candidates, meta, _, _, _, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual(len(candidates), 1)
        evidence = candidates[0].evidence[0].extra
        self.assertEqual(evidence["call_id"], meta["calls"][1]["call_id"])
        self.assertEqual(evidence["attempt_index"], 2)
        self.assertTrue(evidence["correction_of_finding_id"])
        checked = self.checked(result)
        self.assertTrue(checked["protocol_lineage_acceptance_passed"], checked)

    def test_global_budget_one_stops_real_requests_and_preserves_pending(self):
        result = self.review([page(more=True)], llm_max_calls=1)
        _, meta, degraded, sent, events, _, _ = result
        self.assertFalse(meta["complete"])
        self.assertEqual(len(sent), meta["actual_attempt_calls"])
        self.assertEqual(sum(event["status"] == "started" for event in events), 1)
        self.assertIn("llm_call_budget_exhausted", degraded)
        self.assertEqual(meta["initial_review_lineage"][0]["attempt_count"], 1)

    def test_second_window_first_attempt_precedes_first_window_continuation(self):
        rows = [clause("a", "甲" * 200), clause("b", "乙" * 200)]
        def response(payload, index):
            return page(more=True) if "initial_review_continuation" not in payload else page(corrections=[])
        result = self.review(response, rows=rows, llm_chunk_chars=200)
        _, meta, _, sent, _, _, _ = result
        self.assertEqual(len(sent), 4)
        self.assertNotEqual(sent[0]["payload"]["window_id"], sent[1]["payload"]["window_id"])
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["completed_calls"], 2)
        checked = self.checked(result)
        self.assertTrue(checked["protocol_lineage_acceptance_passed"], checked)

    def test_transport_error_keeps_actual_failed_call_then_new_attempt_can_complete(self):
        result = self.review([L.LLMClientError("offline transport failure"), page(corrections=[])])
        _, meta, degraded, _, events, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["actual_attempt_calls"], 2)
        self.assertEqual(meta["calls"][0]["status"], "error")
        self.assertEqual(meta["calls"][0]["call_id"], events[0]["details"]["call_id"])
        self.assertEqual(meta["calls"][0]["request_receipts"][0]["status"], "failed")
        self.assertFalse(any(item.startswith("llm_window_error") for item in degraded))
        checked = self.checked(result)
        self.assertTrue(checked["protocol_lineage_acceptance_passed"], checked)
        self.assertFalse(checked["all_attempts_succeeded"])
        self.assertEqual(len(checked["recorded_non_success_attempts"]), 1)

    def test_bad_schema_is_one_actual_request_then_explicit_continuation(self):
        result = self.review([{"defects": [], "more_findings_possible": "false"}, page(corrections=[])])
        _, meta, _, sent, events, _, _ = result
        self.assertTrue(meta["complete"])
        self.assertEqual(len(sent), 2)
        self.assertEqual(meta["calls"][0]["status"], "error")
        self.assertEqual(meta["calls"][0]["call_id"], events[0]["details"]["call_id"])
        self.assertTrue(self.checked(result)["protocol_lineage_acceptance_passed"])

    def test_strict_capacity_remains_pending_without_explicit_repair(self):
        result = self.review([page([finding(quote="甲" * 160)]), page(corrections=[]), page(corrections=[])],
                             rows=[clause(text="甲" * 160)])
        self.assertFalse(result[1]["complete"])
        self.assertIn("initial_output_prior_findings_unrepaired", result[1]["initial_output_pending"][0]["reasons"])
        self.assertFalse(self.checked(result)["protocol_lineage_acceptance_passed"])

    def test_checker_rejects_deleted_attempt_changed_source_and_forged_complete(self):
        original = self.review([page(more=True), page(corrections=[])])
        self.assertTrue(self.checked(original)["protocol_lineage_acceptance_passed"])
        for mutate in (
            lambda meta: meta["calls"].pop(0),
            lambda meta: meta["calls"][1].update(source_sha256="0" * 64),
            lambda meta: meta["initial_review_lineage"][0]["attempts"][0].update(content=page()),
            lambda meta: meta["calls"][0]["source_spans"][0].update(start=1),
            lambda meta: meta["calls"][0].update(call_id=meta["calls"][1]["call_id"]),
        ):
            changed = list(copy.deepcopy(original))
            mutate(changed[1])
            self.assertFalse(self.checked(changed)["protocol_lineage_acceptance_passed"])

    def test_config_defaults_three_with_explicit_environment_override(self):
        self.assertEqual(Config().llm_initial_max_attempts, 3)
        with patch.dict(os.environ, {"ZLQL_LLM_INITIAL_MAX_ATTEMPTS": "5"}):
            self.assertEqual(config_from_env().llm_initial_max_attempts, 5)

    def test_context_block_is_not_inference_or_a_completed_initial_attempt(self):
        error = L.LLMContextError("capacity unavailable", "context_budget_exceeded", {"fits": False})
        with patch.object(L, "preflight_native_context", side_effect=error):
            result = self.review([], llm_native_context_preflight=True)
        _, meta, degraded, sent, events, _, _ = result
        self.assertEqual(sent, [])
        self.assertEqual(meta["actual_attempt_calls"], 0)
        self.assertEqual(meta["completed_calls"], 0)
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["initial_review_lineage"][0]["attempt_count"], 0)
        self.assertEqual(meta["calls"][0]["status"], "context_blocked")
        self.assertEqual(meta["calls"][0]["request_receipts"][0]["generation_sent"], False)
        self.assertEqual(events[0]["status"], "skipped")
        self.assertTrue(any("context_budget_exceeded" in item for item in degraded))


if __name__ == "__main__":
    unittest.main()
