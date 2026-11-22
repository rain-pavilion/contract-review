"""Run the bounded feedback task to attempt-completion under real model calls.

This probe answers one narrow question: does every feedback work item in the
chosen bounded source actually reach a bound model response, instead of the
review stopping on an exhausted call budget or feedback round limit?

It deliberately does NOT claim:
  * full-contract review completion,
  * convergence of disagreement/pending semantics,
  * legal quality or accuracy of any verdict.

Completion here means ``attempt_complete``: every statute_feedback and
subject_fact work item is actually processed with bound receipt IDs for
the current input fingerprint; an attempted status alone is insufficient.
Items that remain pending after an actual response (needs_manual, unknown
facts, window disagreement) are reported as pending, never rewritten as pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

SOURCE = ROOT / "tools/acceptance_results/initial_source_selection_real_20261118_v2/prepared.json"
FEEDBACK_KINDS = ("statute_feedback", "subject_fact")
ATTEMPTED_STATES = ("executed", "attempted_without_valid_result", "reused_prior_processed_input")


def execution_status(item: dict) -> str:
    value = item.get("execution_status")
    if value is None:
        value = item.get("execution")
    return str(value or "not_executed")


def summarize_feedback(ledger: dict) -> dict:
    items = [item for item in ledger.get("items", []) if isinstance(item, dict)]
    feedback = [item for item in items if item.get("kind") in FEEDBACK_KINDS]
    statuses = Counter(execution_status(item) for item in feedback)
    attempted = [item for item in feedback if execution_status(item) in ATTEMPTED_STATES]
    not_executed = [item for item in feedback if execution_status(item) not in ATTEMPTED_STATES]
    reused = [item for item in feedback if execution_status(item) == "reused_prior_processed_input"]
    receipt_bound = [item for item in feedback
                     if item.get("actual_processed") is True and item.get("receipt_call_ids")
                     and execution_status(item) in ("executed", "reused_prior_processed_input")]
    receipt_unbound = [item for item in feedback if item not in receipt_bound]
    unexecuted_fingerprint = [item for item in feedback
                              if "current_fingerprint_not_attempted" in (item.get("pending_reasons") or [])]
    current_bound = [item for item in receipt_bound if item not in unexecuted_fingerprint]
    pending = [item for item in feedback if item.get("pending")]
    pending_reasons = Counter(reason for item in pending for reason in item.get("pending_reasons", []))
    pending_ids = sorted(str(item.get("work_item_id")) for item in pending)
    return {
        "kind_scope": list(FEEDBACK_KINDS),
        "total": len(feedback),
        "attempted_or_reused": len(attempted),
        "executed": statuses.get("executed", 0),
        "reused_prior_processed_input": len(reused),
        "not_executed": len(not_executed),
        "not_executed_work_item_ids": sorted(str(item.get("work_item_id")) for item in not_executed),
        "execution_status_counts": dict(sorted(statuses.items())),
        "execution_status_coverage_complete": bool(feedback) and not not_executed,
        "bound_processed_count": len(receipt_bound),
        "missing_bound_processing_receipt_work_item_ids": sorted(str(item.get("work_item_id")) for item in receipt_unbound),
        "execution_coverage_complete": bool(feedback) and not receipt_unbound,
        "current_fingerprint_attempted_work_items": len(current_bound),
        "current_fingerprint_not_attempted_work_items": len(feedback) - len(current_bound),
        "explicit_current_fingerprint_not_attempted_reason_count": len(unexecuted_fingerprint),
        "attempt_complete": bool(feedback) and not receipt_unbound and not unexecuted_fingerprint,
        "pending_after_response": len(pending),
        "pending_work_item_ids": pending_ids,
        "pending_reasons": dict(sorted(pending_reasons.items(), key=lambda pair: (-pair[1], pair[0]))),
        "protocol_complete_count": sum(item.get("protocol_complete") is True for item in feedback),
        "actual_processed_count": sum(item.get("actual_processed") is True for item in feedback),
        "dispatchable_pending": [str(item.get("work_item_id")) for item in pending if item.get("dispatchable")],
        "meaning": ("attempt_complete requires every feedback item to be actually processed with non-empty bound "
                    "receipt IDs for the current input fingerprint; an attempted status alone is insufficient. "
                    "Manual and deferred obligations stay pending independently of execution coverage."),
    }


def summarize_initial(ledger: dict) -> dict:
    items = [item for item in ledger.get("items", []) if isinstance(item, dict)
             and item.get("kind") == "initial_review"]
    pending_reasons = Counter(reason for item in items for reason in item.get("pending_reasons", []))
    return {
        "items": len(items),
        "pending": sum(bool(item.get("pending")) for item in items),
        "pending_reasons": dict(sorted(pending_reasons.items(), key=lambda pair: (-pair[1], pair[0]))),
        "note": "initial output_pending is recorded as-is; this probe does not claim initial completeness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=0,
                        help="0 = no per-run call budget cut; the round plan still bounds work")
    parser.add_argument("--freeze-cycle", action="store_true",
                        help="opt in to an immutable feedback obligation cycle; discoveries stay pending for the next cycle")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT / "tools/acceptance_results"):
        raise ValueError("fresh directory required; actual calls are never overwritten")
    out.mkdir()
    source_files = ["nodes.py", "agents_multi_run.py", "agents.py", "models.py", "review_work_items.py",
                    "feedback_dependencies.py", "feedback_transport.py", "claim_grounding.py", "fact_slots.py",
                    "proposition_graph.py", "ner.py", "ner_constraints.py", "entailment.py", "bert_entail.py",
                    "config.py", "feedback_cycle.py"]
    before = {name: hashlib.sha256((ROOT / "zhilv_qiling_v2" / name).read_bytes()).hexdigest()
              for name in source_files}
    (out / "sources").mkdir()
    for name in source_files:
        (out / "sources" / name).write_bytes((ROOT / "zhilv_qiling_v2" / name).read_bytes())
    (out / "executed_runner.py").write_bytes(Path(__file__).read_bytes())

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ZLQL_ENTAILMENT_DEVICE="cpu",
                      ZLQL_MODEL_AUDIT_PATH=str(out / "model_calls.sqlite"), ZLQL_LLM_API_KEY="",
                      ZLQL_ENTAILMENT_MODEL_DIR=str(ROOT / "zhilv_qiling_v2/models/bert_mamba_entail"))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from zhilv_qiling_v2 import llm_client as L, model_audit as A, model_runtime as R
    from zhilv_qiling_v2.agents_multi_run import run_multi_agent
    from zhilv_qiling_v2.config import Config
    from zhilv_qiling_v2.models import Contract, DocumentParse, Clause, SourceBlock
    from verify_complex_core_observed_v3 import PassiveCapture, reconcile_calls

    raw = SOURCE.read_bytes()
    prior = json.loads(raw)
    clauses = [Clause(**item) for item in prior["clauses"]]
    blocks = []
    for index, clause in enumerate(clauses):
        block_id = "scope_block_" + str(index)
        blocks.append(SourceBlock(block_id, index, "paragraph", clause.text))
        clause.block_ids = [block_id]
    contract = Contract("feedback-completion-20261118", DocumentParse(
        "existing-public-source-partial.txt", "txt",
        "\n".join(c.text for c in clauses), blocks=blocks, clauses=clauses))
    config = Config(
        enable_llm=True, llm_endpoint="http://127.0.0.1:8621/v1", llm_model="legalone-8b",
        llm_timeout=600, llm_max_tokens=2048, llm_native_thinking="off",
        llm_native_reasoning_format="deepseek", llm_native_json_schema=True,
        llm_initial_source_protocol="initial-exact-source-selection-v1",
        llm_chunk_chars=6000, llm_chunk_overlap=0, llm_cross_clause_pairs=0,
        llm_max_calls=args.max_calls, llm_initial_max_attempts=1,
        review_max_rounds=args.rounds, freeze_feedback_cycle=args.freeze_cycle,
        ner_enable_model=False, ner_use_bert=False,
        use_lsa=False, use_rerank=False, top_k=1, entailment_backend="bert",
        subject_facts_path=str(out / "no_external_facts_supplied.json"))

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    write("prepared.json", {
        "source_path": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_characters": len(contract.document.full_text), "partial_contract": True,
        "human_gold": False, "config": asdict(config),
        "blocks_regrouped_from_exact_existing_clauses": True,
        "initial_calls_per_scope": 1, "feedback_round_limit": args.rounds,
        "max_calls": args.max_calls,
        "freeze_feedback_cycle": args.freeze_cycle,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion_scope": "bounded subset feedback attempt execution; not full contract; not legal quality"})
    runtime = R.initialize_backend(config, force=True)
    if runtime.get("active_backend") != "bert":
        raise RuntimeError("actual trained backend did not activate")
    capture = PassiveCapture(out / "raw_captures", L._complete.__code__, L.request_json.__code__)
    start = time.perf_counter()
    error = None
    run = None
    with A.review_audit("feedback-completion-20261118", contract.contract_id) as audit:
        capture.start()
        try:
            run = run_multi_agent(contract, config)
        except Exception as exc:  # keep partial artifacts and the real failure
            error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            capture.stop()
    write("events.json", audit.events)
    manifest = capture.summary()
    write("capture_manifest.json", manifest)
    write("capture_reconciliation.json", reconcile_calls(out, manifest))
    receipt = {
        "status": "executed" if run else "execution_failed",
        "error": error,
        "completion_scope": "bounded subset feedback attempt execution; not full contract; not legal quality",
        "runtime": runtime,
        "source_sha256_before": before,
        "source_sha256_after": {name: hashlib.sha256((ROOT / "zhilv_qiling_v2" / name).read_bytes()).hexdigest()
                                for name in source_files},
        "human_gold": False,
        "full_contract_acceptance": False,
        "legal_accuracy_evaluated": False,
        "elapsed_seconds_including_capture": time.perf_counter() - start,
        "llm_statuses": dict(Counter(e.get("status") for e in audit.events
                                     if e.get("component") == "llm" and e.get("operation") == "request_json")),
        "entailment_window_successes": sum(
            e.get("operation") == "predict_window" and e.get("status") == "succeeded" for e in audit.events),
        "source_unchanged": SOURCE.read_bytes() == raw,
        "raw_capture_integrity": manifest.get("capture_integrity_complete"),
    }
    if run:
        write("run.json", {"summary": run.summary(), "diagnostics": run.diagnostics, "steps": run.steps,
                           "candidates": [item.to_dict() for item in run.candidates],
                           "risks": [item.to_dict() for item in run.risks],
                           "degraded": run.degraded})
        (out / "trace.json").write_text(run.dump_trace(), encoding="utf-8")
        ledger = run.diagnostics["review_work_items"]
        logic_llm = run.diagnostics["logic_llm"]
        feedback = summarize_feedback(ledger)
        initial = summarize_initial(ledger)
        cross = run.diagnostics["cross_validation"]
        receipt.update({
            "feedback_execution": feedback,
            "initial_review": initial,
            "cross_validation": cross,
            "llm_feedback_budget": {
                "planned_calls": logic_llm.get("planned_calls"),
                "actual_request_budget": logic_llm.get("actual_request_budget"),
                "actual_attempt_calls": logic_llm.get("actual_attempt_calls"),
                "feedback_runs": len(logic_llm.get("feedback_runs") or []),
                "feedback_latest_runs": len(logic_llm.get("feedback_latest_runs") or []),
                "feedback_attempts": len(logic_llm.get("feedback_attempts") or []),
                "feedback_all_attempts_complete": logic_llm.get("feedback_all_attempts_complete"),
                "feedback_runs_complete": logic_llm.get("feedback_runs_complete"),
                "verified_feedback_item_ids": len(logic_llm.get("verified_feedback_item_ids") or []),
                "uncompleted_feedback_item_ids": len(logic_llm.get("uncompleted_feedback_item_ids") or []),
                "unmatched_feedback_item_ids": len(logic_llm.get("unmatched_feedback_item_ids") or []),
            },
            "pending_work_items": len(ledger.get("pending_work_item_ids") or []),
            "risk_count": len(run.risks),
            "review_required_count": sum(item.review_required for item in run.risks),
        })
        cycle = logic_llm.get("feedback_cycle")
        if isinstance(cycle, dict):
            write("feedback_cycle.json", cycle)
            receipt["frozen_cycle"] = {
                "cycle_id": cycle.get("cycle_id"),
                "active_request_fingerprint": cycle.get("active_request_fingerprint"),
                "active_feedback_execution": summarize_feedback({"items": cycle.get("active_work_items", [])}),
                "active_attempt_complete": cycle.get("active_attempt_complete"),
                "active_cycle_complete": cycle.get("active_cycle_complete"),
                "deferred_pending_count": len(cycle.get("deferred_pending_work_item_ids", [])),
                "overall_complete": cycle.get("overall_complete"),
                "meaning": "Active execution coverage is not legal acceptance; deferred and manual work remain pending.",
            }
        checks = {
            "run_executed": True,
            "attempt_complete": feedback["attempt_complete"],
            "source_unchanged": receipt["source_unchanged"],
            "raw_capture_integrity": bool(receipt["raw_capture_integrity"]),
            "no_legal_quality_claim": (receipt["human_gold"] is False and receipt["full_contract_acceptance"] is False
                                       and receipt["legal_accuracy_evaluated"] is False),
        }
        receipt["checks"] = checks
        receipt["attempt_complete"] = feedback["attempt_complete"]
        receipt["pending_still_open"] = feedback["pending_after_response"]
        receipt["pending_are_not_failures"] = True
    write("receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in (
        "status", "error", "attempt_complete", "feedback_execution", "initial_review",
        "llm_statuses", "entailment_window_successes", "pending_still_open",
        "elapsed_seconds_including_capture", "checks")}, ensure_ascii=False, default=str))
    return 0 if run and all(receipt.get("checks", {"run_executed": False}).values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
