"""Offline re-evaluation of preserved real observations; no new model request."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2.claim_grounding import assess_claim
from zhilv_qiling_v2.fact_slots import extract_fact_slots
from zhilv_qiling_v2.initial_source_selection import InitialSourceCatalogue
from zhilv_qiling_v2.proposition_graph import build_proposition_graph


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(directory: Path) -> dict:
    prepared_path = directory / "prepared.json"
    raw_path = directory / "external_original_model_json.json"
    report_path = directory / "report.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8-sig"))
    observed = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    before = copy.deepcopy((prepared, observed, report))
    original_calls = report["stats"]["logic_llm"]["calls"]
    windows = {row["window_id"]: row for row in prepared["windows"]}
    rows = []
    for call in original_calls:
        if call.get("call_id") not in observed:
            continue
        raw = observed[call["call_id"]]
        window = windows[call["window_id"]]
        catalogue = InitialSourceCatalogue(window)
        resolution = catalogue.resolve(raw, window_payload=window)
        canonical = resolution["resolved_content"]
        previous = call.get("source_selection", {}).get("resolved_content")
        if canonical != previous:
            raise ValueError("Recomputed source projection differs from preserved historical receipt")
        slots = [extract_fact_slots(segment["clause_id"], segment["text"],
            start_offset=segment["start"]) for segment in window["segments"]]
        graph = build_proposition_graph(slots, scope_complete=False)
        selected_source_refs = {ref for finding in raw["defects"] for ref in
            [finding["source_ref"], *finding["related_source_refs"]]}
        selected_entries = [entry for entry in catalogue.entries if entry["source_ref"] in selected_source_refs]
        unselected_payment_segments = [segment for segment in window["segments"]
            if "付款方式" in segment["text"] and not any(entry["clause_id"] == segment["clause_id"] for entry in selected_entries)]
        for index, finding in enumerate(canonical["defects"]):
            assessment = assess_claim(finding, window["segments"], contract_type="政府采购",
                scope_complete=False, slot_documents=slots, proposition_graph=graph)
            citations = [finding, *finding.get("related_quotes", [])]
            rows.append({"call_id": call["call_id"], "window_id": window["window_id"],
                "finding_index": index, "raw_finding_unchanged": copy.deepcopy(raw["defects"][index]),
                "source_projection": copy.deepcopy(finding),
                "claim_grounding_status": assessment["claim_grounding_status"],
                "manual_required": assessment["manual_required"],
                "automatic_claim_confirmation_blocked": assessment["claim_grounding_status"] != "supported" or assessment["manual_required"],
                "block_reason_scope": "Citation does not establish this assertion; this is not a proof of its opposite or human legal gold",
                "reasons": assessment["reasons"],
                "literal_support_reasons": assessment["literal_support_reasons"],
                "source_content_difference": {
                    "original_claim_message": finding["message"],
                    "actual_chosen_citations": [{"clause_id": row["clause_id"], "quote": row["quote"]} for row in citations],
                    "main_quote_is_account_dominant": assessment["account_dominant_primary_quote"],
                    "supplied_payment_source_not_selected_by_any_of_eight_findings": unselected_payment_segments,
                    "source_binding_is_not_support_or_legal_truth": True,
                    "whole_contract_absence_not_established": True},
                "assessment": assessment})
    if (prepared, observed, report) != before:
        raise AssertionError("Replay changed its historical inputs")
    counts = {status: sum(row["claim_grounding_status"] == status for row in rows)
        for status in ("supported", "unsupported", "unknown")}
    return {"replayed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_replay_of_preserved_actual_model_json", "new_model_calls": 0,
        "independent_human_legal_gold": False, "legal_accuracy_evaluated": False,
        "source_contract_is_complete": False, "prior_raw_model_json_unchanged": True,
        "prior_source_projection_recomputed_exactly": True,
        "input_files": [{"path": str(path), "sha256": digest(path)}
            for path in (prepared_path, raw_path, report_path)],
        "finding_count": len(rows), "claim_grounding_counts": counts,
        "automatic_confirmation_blocked_count": sum(row["automatic_claim_confirmation_blocked"] for row in rows),
        "account_dominant_primary_quote_count": sum(row["source_content_difference"]["main_quote_is_account_dominant"] for row in rows),
        "interpretation": "The guard makes unsupported observations explicit and blocks automatic promotion; it does not repair the raw model reasoning or prove legal accuracy.",
        "findings": rows}


if __name__ == "__main__":
    source = ROOT / "tools/acceptance_results/initial_source_selection_real_20261118_v2"
    target = ROOT / "tools/selected_solutions_20261118/claim_replay_real_outputs.json"
    result = replay(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readback = json.loads(target.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(target), "findings": readback["finding_count"],
        "statuses": readback["claim_grounding_counts"],
        "automatic_confirmation_blocked": readback["automatic_confirmation_blocked_count"],
        "account_primary": readback["account_dominant_primary_quote_count"]}, ensure_ascii=False))
