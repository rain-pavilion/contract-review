"""Offline adversarial examples for the scoped outcome inspector; no product imports."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from inspect_scoped_diagnostic_outcomes_20261115 import CASES, digest, inspect


def fixture():
    definitions = (
        ("第二条", "本合同尾款人民币70000元；甲方依据第九条完成最终验收后支付该尾款。"),
        ("第三条", "本项目最终交付日期为2026年12月1日。"),
        ("第七条", "本合同尾款人民币80000元；该笔为同一合同的尾款约定，未写明替代前款。"),
        ("第八条", "本项目最终交付日期为2027年1月15日。"),
        ("第九条", "验收以第二条所列付款安排为前提，甲方支付全部合同价款后乙方才启动最终验收。"),
    )
    clauses = [{"clause_id": f"cl_{i}", "label": label, "text": label + "\n" + body,
                "block_ids": [f"sb_{i}"]} for i, (label, body) in enumerate(definitions)]
    parsed = "\n".join(row["text"] for row in clauses)
    prepared = {"source_manifest": {"fixture_version": "scoped_value_tail_v2"},
                "clauses": clauses, "contract_id": "ct_offline_test", "parsed_text": parsed,
                "parsed_text_sha256": digest(parsed),
                "plan": {"coverage": "planner-only is not a detection"},
                "planned_coverage": {"text_complete": True}}
    sources = [{"clause_id": row["clause_id"], "start": 0, "end": len(row["text"]), "text": row["text"]}
               for row in clauses]
    source_hash = digest(json.dumps(sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    call = {"call_id": "call_exact", "window_id": "window_exact", "job_id": "job_exact",
            "source_sha256": source_hash, "model": "legalone-8b", "status": "ok",
            "source_spans": [{key: row[key] for key in ("clause_id", "start", "end")} for row in sources]}
    events = []
    for index, status in enumerate(("started", "succeeded")):
        events.append({"event_id": "evt_" + str(index), "created_at": f"2026-11-15T01:00:0{index}+00:00",
                       "trace_id": "trace_offline_test", "contract_id": prepared["contract_id"],
                       "component": "llm", "operation": "request_json", "model": "legalone-8b", "status": status,
                       "reason": "", "details": {key: call[key] for key in ("call_id", "window_id", "job_id", "source_sha256")}})
        events[-1]["details"].update(requested_model="legalone-8b", actual_model="legalone-8b",
                                     model_identity_verified=True, finish_reason="stop")
    claims = ("第二条与第七条的尾款70000元和80000元存在不一致。",
              "本项目最终交付日期2026年12月1日与2027年1月15日存在冲突。",
              "付款以验收为前提、验收又以全部付款为前提，构成付款与验收循环。")
    risks = []
    by_label = {row["label"]: row for row in clauses}
    for index, (case, message) in enumerate(zip(CASES, claims)):
        rows = [by_label[label] for label, _ in case["sides"]]
        quotes = [{"clause_id": row["clause_id"], "quote": row["text"].partition("\n")[2],
                   "quote_exact_source_match": True} for row in rows]
        spans, evidence = [], []
        for position, (row, quote) in enumerate(zip(rows, quotes)):
            start = row["text"].index(quote["quote"])
            span = {"clause_id": row["clause_id"], "block_id": row["block_ids"][0], "quote": quote["quote"][:80],
                    "char_start": start, "char_end": start + len(quote["quote"])}
            spans.append(span)
            extra = {key: call[key] for key in ("call_id", "window_id", "job_id", "source_sha256")}
            extra.update(candidate_id=f"candidate_{index}", partial_model_output=False,
                         model_message=message, issue_code="synthetic_test_issue", **quote, bound_model_quotes=deepcopy(quotes))
            evidence.append({"source": "node2_llm", "kind": "text" if position == 0 else "cross_clause_quote",
                             "text": quote["quote"] if position == 0 else quote["clause_id"] + ": " + quote["quote"],
                             "span": deepcopy(span) if position == 0 else None, "extra": extra})
        risks.append({"risk_id": f"risk_{index}", "code": "synthetic_test_issue", "message": message,
                      "agent_sources": ["node2_logic"], "clause_ids": [row["clause_id"] for row in rows],
                      "spans": spans, "evidence": evidence})
    report = {"contract_id": prepared["contract_id"], "trace_id": "trace_offline_test",
              "document": {"full_text": parsed}, "risks": risks,
              "stats": {"candidates": 3, "logic_llm": {"calls": [call], "feedback_runs": [], "text_complete": True},
                        "model_calls": events}}
    return prepared, report


def status(prepared, report, number=0):
    return inspect(prepared, report)["outcomes"][number]


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.prepared, self.report = fixture()

    def test_exact_retained_claim_pair_and_receipts_are_observed_not_accuracy(self):
        result = inspect(self.prepared, self.report)
        self.assertEqual(result["counts"], {"missing": 0, "partial": 0, "observed": 3})
        self.assertFalse(result["legal_accuracy_evaluated"])
        self.assertFalse(result["independent_human_gold_used"])

    def test_planner_and_coverage_alone_are_missing(self):
        self.report["risks"] = []
        self.assertEqual(inspect(self.prepared, self.report)["counts"]["missing"], 3)

    def test_rule_and_agent_name_cannot_substitute_for_model_evidence(self):
        for risk in self.report["risks"]:
            for row in risk["evidence"]:
                row["source"] = "rule"
        result = status(self.prepared, self.report)
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["non_model_final_claims_not_counted"], 1)

    def test_one_clause_model_quote_is_only_partial(self):
        risk = self.report["risks"][0]
        risk["evidence"] = risk["evidence"][:1]
        risk["evidence"][0]["extra"].pop("bound_model_quotes")
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_two_different_candidates_are_not_stitched(self):
        for index, row in enumerate(self.report["risks"][0]["evidence"]):
            row["extra"].pop("bound_model_quotes")
            row["extra"]["candidate_id"] = "different_" + str(index)
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_partial_output_stays_partial(self):
        for row in self.report["risks"][0]["evidence"]:
            row["extra"]["partial_model_output"] = True
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_old_report_missing_candidate_metadata_stays_partial(self):
        for row in self.report["risks"][0]["evidence"]:
            row["extra"] = {}
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_rule_message_does_not_fill_missing_model_claim(self):
        for row in self.report["risks"][0]["evidence"]:
            row["extra"].pop("model_message")
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_negated_model_relationship_does_not_count(self):
        for row in self.report["risks"][0]["evidence"]:
            row["extra"]["model_message"] = "尾款70000元与80000元不存在冲突，可以协调。"
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_cycle_clause_numbers_alone_are_insufficient(self):
        risk = self.report["risks"][2]
        package = [{"clause_id": risk["clause_ids"][0], "quote": "第九条", "quote_exact_source_match": True},
                   {"clause_id": risk["clause_ids"][1], "quote": "第二条", "quote_exact_source_match": True}]
        for row, quote in zip(risk["evidence"], package):
            row["extra"].update(**quote, bound_model_quotes=deepcopy(package))
        self.assertEqual(status(self.prepared, self.report, 2)["status"], "partial")

    def test_forged_quote_or_false_exactness_is_partial(self):
        for row in self.report["risks"][0]["evidence"]:
            row["extra"]["bound_model_quotes"][1]["quote"] = "本合同尾款人民币90000元。"
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")
        self.prepared, self.report = fixture()
        for row in self.report["risks"][0]["evidence"]:
            row["extra"]["bound_model_quotes"][1]["quote_exact_source_match"] = False
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_own_exactness_cannot_disagree_with_its_bundle(self):
        self.report["risks"][0]["evidence"][0]["extra"]["quote_exact_source_match"] = False
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_source_sent_only_one_side_cannot_claim_two_sides(self):
        self.report["stats"]["logic_llm"]["calls"][0]["source_spans"] = self.report["stats"]["logic_llm"]["calls"][0]["source_spans"][:1]
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_wrong_receipt_identity_or_truncated_receipt_stays_partial(self):
        self.report["stats"]["model_calls"][1]["details"]["actual_model"] = "some-other-model"
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")
        self.prepared, self.report = fixture()
        self.report["stats"]["model_calls"][1]["details"]["finish_reason"] = "length"
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_duplicate_call_id_does_not_get_guessed_by_order(self):
        self.report["stats"]["logic_llm"]["feedback_runs"] = [{"calls": deepcopy(self.report["stats"]["logic_llm"]["calls"])}]
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_receipt_completion_cannot_be_unknown_or_have_duplicate_ids(self):
        self.report["stats"]["model_calls"][1]["details"].pop("finish_reason")
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")
        self.prepared, self.report = fixture()
        self.report["stats"]["model_calls"][1]["event_id"] = self.report["stats"]["model_calls"][0]["event_id"]
        self.assertEqual(status(self.prepared, self.report)["status"], "partial")

    def test_surviving_model_bundle_preserves_pair_with_display_problem_visible(self):
        risk = self.report["risks"][0]
        risk["evidence"] = risk["evidence"][1:]
        risk["spans"][0]["char_start"] = 0
        result = status(self.prepared, self.report)
        self.assertEqual(result["status"], "observed")
        self.assertTrue(result["final_model_evidence_assessments"][0]["final_display_location_issues"])

    def test_source_mismatch_and_missing_report_are_explicit(self):
        self.report["document"]["full_text"] += " changed"
        result = inspect(self.prepared, self.report)
        self.assertEqual(result["counts"]["observed"], 0)
        self.assertIn("report_source_does_not_match_prepared", result["source_issues"])
        missing = inspect(self.prepared, None)
        self.assertEqual(missing["inspection_status"], "report_missing")
        self.assertEqual(missing["counts"]["missing"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
