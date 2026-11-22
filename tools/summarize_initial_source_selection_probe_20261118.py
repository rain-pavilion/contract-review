"""Read-only textual diagnostics of saved real runs; no inference or network."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tools/acceptance_results"
V1 = BASE / "initial_source_selection_real_20261118_v1"
V2 = BASE / "initial_source_selection_real_20261118_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (V2 / name).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


bodies = [json.loads(base64.b64decode(read(directory / "raw_captures/complete_000001_return.json")["body"]["data"]))
          for directory in (V1, V2)]
differences = [key for key in sorted(set(bodies[0]) | set(bodies[1])) if bodies[0].get(key) != bodies[1].get(key)]
assert differences == ["reasoning_format"]
write("native_configuration_comparison.json", {"v1_first_request_body_vs_v2_first_request_body_changed_fields": differences,
    "reasoning_format_before": bodies[0]["reasoning_format"], "reasoning_format_after": bodies[1]["reasoning_format"],
    "all_other_top_level_native_request_fields_identical": True,
    "v1_result_sha256": sha(V1 / "result.json"), "v2_result_sha256": sha(V2 / "result.json"),
    "v1_actual_requests": 3, "v1_successful_model_returns": 0,
    "v2_actual_requests": 1, "v2_successful_model_returns": 1,
    "total_real_http_generation_requests_across_both_versioned_runs": 4})
report = read(V2 / "report.json")
meta = report["stats"]["logic_llm"]
call = meta["calls"][0]
raw = call["initial_raw_response"]
derived = call["source_selection"]["resolved_content"]
prepared = read(V2 / "prepared.json")
assert raw == next(iter(read(V2 / "external_original_model_json.json").values()))
text = prepared["raw_text"]
rows = []
for index, (model, bound) in enumerate(zip(raw["defects"], derived["defects"])):
    rows.append({"finding_index": index, "code": model["code"], "model_message": model["message"],
        "source_ref": model["source_ref"], "bound_clause_id": bound["clause_id"],
        "bound_quote": bound["quote"], "source_exactly_locatable": True,
        "selected_primary_quote_contains_account_label": "账号" in bound["quote"]})
primary_account_count = sum(row["selected_primary_quote_contains_account_label"] for row in rows)
assert primary_account_count == 6
assert "违约" not in text
diagnostic = {"status": "source_binding_mechanism_operated_but_initial_completion_failed",
    "human_legal_gold": False, "legal_accuracy_measured": False,
    "exact_source_references": len(call["source_selection"]["bindings"]),
    "unresolvable_source_references": 0, "raw_findings": len(raw["defects"]),
    "primary_quotes_containing_account_label": primary_account_count,
    "distinct_selected_source_refs": len({row["source_ref"] for row in call["source_selection"]["bindings"]}),
    "selected_primary_clause_ids": sorted({row["clause_id"] for row in derived["defects"]}),
    "payment_clause_selected_as_evidence": any(row["clause_id"] == "cl_217e94035b"
        for row in call["source_selection"]["bindings"]),
    "raw_code_counts": dict(Counter(row["code"] for row in raw["defects"])),
    "textual_observations_not_legal_verdicts": [
        "Six findings select an account-information extract. Exact location does not show support for subject specifications, acceptance, breach, party completeness or delivery conclusions.",
        "fee_mismatch discusses missing rent in a medical-equipment procurement subset; rent requirements are not established by the supplied text.",
        "one_sided_breach asserts that only seller breach obligations exist, but the supplied 509 characters contain no occurrence of 违约.",
        "vague_delivery treats an introductory signing phrase as a delivery-time description without a source passage tying it to delivery.",
        "Two unclear_standard findings use the same primary source and repeat acceptance/payment concerns.",
        "The first finding repeats its primary source ID in related_source_refs and reaches the unchanged related-capacity boundary.",
        "None of the chosen source references points to the supplied payment clause despite several messages relying on its total price."],
    "rows": rows,
    "strict_pending_reasons": call["output_pending_reasons"],
    "remaining_scope": "Semantic evidence relevance, conservative handling of missing context, duplicate findings and honest completion output still require work",
    "controlled_improvement_over_old_model_established": False}
write("textual_diagnostic_not_legal_gold.json", diagnostic)
write("supplemental_manifest.json", {"original_artifact_manifest_sha256": sha(V2 / "artifact_manifest.json"),
    "files": [{"path": name, "sha256": sha(V2 / name)} for name in (
        "native_configuration_comparison.json", "textual_diagnostic_not_legal_gold.json")],
    "supplemental_runner_sha256": sha(Path(__file__))})
print(json.dumps({key: diagnostic[key] for key in ("raw_findings", "exact_source_references", "unresolvable_source_references", "primary_quotes_containing_account_label", "distinct_selected_source_refs", "payment_clause_selected_as_evidence")}, ensure_ascii=True, indent=2))
