"""Read back the auxiliary CLI smoke receipt; no torch, network or model calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import predict_legal_conflict as P

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "tools/acceptance_results/real_legal_prediction_20261115_v1"
DEFAULT_TRAINING = ROOT / "tools/acceptance_results/real_legal_training_20261115_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, training_directory):
    source_path = directory / "input_source.json"
    input_path = directory / "input_heldout3.json"
    prediction_path = directory / "prediction_heldout3.json"
    training_path = training_directory / "run.json"
    used_path = training_directory / "used_pairs.jsonl"
    provenance, document, result, training = [P.read_json(path) for path in
                                            (source_path, input_path, prediction_path, training_path)]
    rows = P.validate_input(document)
    require(len(rows) == 3, "this smoke verification expects exactly the first three reused test rows")
    require(provenance["source_sha256"] == P.file_sha256(used_path), "used training-pair source hash changed")
    # JSONL parsing uses the same duplicate/non-finite-safe decoder as JSON.
    import json
    used = [json.loads(line, object_pairs_hook=P._object_no_duplicates)
            for line in used_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout = [row for row in used if row["derived_split"] == "test"][:3]
    require([row["id"] for row in rows] == [row["pair_id"] for row in heldout], "input is not the original first-three test smoke")
    for item, original in zip(rows, heldout):
        require(item["query"] == original["clause"] and item["reference"] == original["article"], "heldout input text changed")
    require(result.get("status") == "completed_auxiliary_legislative_conflict_prediction", "CLI did not complete")
    require(result.get("task") == P.TASK and result.get("num_labels") == 5
            and result.get("class_names") == list(P.CLASS_NAMES), "wrong prediction task or classes")
    require(result["input"]["sha256"] == P.file_sha256(input_path), "prediction input hash differs")
    require(result["tool_sha256"] == P.file_sha256(ROOT / "tools/predict_legal_conflict.py"), "prediction tool changed since smoke")
    require(result["model_builder_sha256"] == P.BUILDER_SHA256, "prediction builder hash differs")
    checkpoint, _, hashes = P.inspect_checkpoint(result["checkpoint"]["directory"])
    require(result["checkpoint"]["sha256"] == hashes, "prediction checkpoint receipt differs from pinned files")
    require(training.get("status") == "trained_strict_reloaded_predictions_completed", "training receipt incomplete")
    require(training["checkpoint"]["sha256"] == hashes["model.pt"], "prediction did not use the verified trained checkpoint")
    require(training["strict_reload_predictions_match_after_test"] is True, "training strict-reload proof unavailable")
    for key in ("contract_acceptance", "compatible_with_contract_entailment_runtime", "contract_law_quality_accepted",
                "independent_project_legal_accuracy_accepted", "official_full_benchmark_score", "product_default_replaced",
                "external_facts_verified"):
        require(result.get(key) is False, "auxiliary-task boundary missing: " + key)
    runtime = result["runtime"]
    require(all(runtime.get(key) is True for key in
                ("strict_state_dict_load", "weights_only", "offline_local_files_only")), "strict offline load flags missing")
    require(runtime.get("truncation") is False and runtime.get("max_tokens_each_tower") == 384, "input truncation/budget mismatch")
    after = {row["pair_id"]: row for row in training["after_test"]["predictions"]}
    reloaded = {row["pair_id"]: row for row in training["reload_predictions"]}
    predictions = result["predictions"]
    require([row["id"] for row in predictions] == [row["id"] for row in rows], "prediction IDs/order differ")
    per_id, max_probability_difference = [], 0.
    for item, output, original in zip(rows, predictions, heldout):
        identifier = item["id"]
        require(output["predicted_conflict_type"] == after[identifier]["predicted"]
                == reloaded[identifier]["predicted_conflict_type"], "prediction differs from training after_test/strict reload")
        require(set(output["probabilities"]) == set(P.CLASS_NAMES), "probability classes differ")
        values = P.normalized_probabilities([output["probabilities"][name] for name in P.CLASS_NAMES])
        require(abs(math.fsum(values) - 1) <= 1e-12 and abs(output["probability_sum"] - 1) <= 1e-12, "probability sum invalid")
        require(output["class_index"] == max(range(5), key=lambda index: values[index])
                and P.CLASS_NAMES[output["class_index"]] == output["predicted_conflict_type"], "argmax label invalid")
        require(output.get("probabilities_are_calibrated") is False and output.get("truncated") is False, "probability/truncation boundary missing")
        require(all(type(value) is int and 0 < value <= 384 for value in output["token_lengths"].values()), "tower token budget invalid")
        require(math.isfinite(output["cosine"]), "non-finite cosine")
        for key in ("query", "reference"):
            require(output[key + "_sha256"] == hashlib.sha256(item[key].encode("utf-8")).hexdigest(), "prediction text hash differs")
        difference = max(abs(output["probabilities"][name] - reloaded[identifier]["probabilities"][name]) for name in P.CLASS_NAMES)
        require(difference <= 1e-6, "CLI probabilities differ materially from original strict reload")
        max_probability_difference = max(max_probability_difference, difference)
        per_id.append({"id": identifier, "original_id": original["original_id"],
            "predicted_conflict_type": output["predicted_conflict_type"],
            "training_after_test_prediction": after[identifier]["predicted"],
            "training_strict_reload_prediction": reloaded[identifier]["predicted_conflict_type"],
            "matches_both_prior_predictions": True, "probability_sum": math.fsum(values),
            "max_absolute_probability_difference_vs_prior_reload": difference})
    return {"status": "verified_reused_heldout_cli_smoke", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": P.TASK, "rows": 3, "checkpoint_sha256": hashes["model.pt"],
        "strict_local_five_class_load_verified": True, "finite_normalized_probabilities_verified": True,
        "input_is_first_three_original_used_test_rows": True, "predictions_match_training_after_test": True,
        "predictions_match_training_strict_reload": True, "probability_comparison_tolerance": 1e-6,
        "maximum_absolute_probability_difference": max_probability_difference, "per_id": per_id,
        "input_sha256": {path.name: P.file_sha256(path) for path in (source_path, input_path, prediction_path)},
        "training_run_sha256": P.file_sha256(training_path), "used_pairs_sha256": P.file_sha256(used_path),
        "verifier_sha256": P.file_sha256(__file__), "new_model_calls": 0, "new_training": False,
        "new_independent_test": False, "new_accuracy_estimate": False, "contract_acceptance": False,
        "product_default_replaced": False,
        "meaning": "重复已评估留出集首3条，仅核验独立CLI可加载既有五类权重并复现既有预测；不是新独立测试、准确率估计、合同法律质量验收或完整公开基准。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    args = parser.parse_args()
    target = args.run_dir / "verification.json"
    if target.exists():
        parser.error("preserve existing inference verification")
    result = verify(args.run_dir, args.training_dir)
    P.write_new_json(target, result)
    print(result["status"] + ": " + str(target))


if __name__ == "__main__":
    main()
