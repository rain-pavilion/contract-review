"""Run the isolated, native five-class LCR-CN legislative-conflict model offline.

Input JSON: {"task": TASK, "items": [{"id": "...", "query": "...", "reference": "..."}]}.
query is the supplied lower-level legislative provision; reference is its supplied
superior-law text. This auxiliary classifier does not accept the contract-law
entailment task, verify external facts, or produce a contract legal conclusion.
The exact trained checkpoint, metadata, tokenizer and builder are hash-pinned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASK = "original_legislative_conflict_type_classification"
CLASS_NAMES = (
    "增设或变更适用条件", "处罚幅度或范围不符", "无冲突",
    "概念或定义范围不符", "职权或责任划分不符",
)
DEFAULT_CHECKPOINT = ROOT / "zhilv_qiling_v2/models/bert_mamba_legal_20261115"
MAX_TOKENS = 384
MAX_ITEMS = 128
PINNED_FILES = {
    "model.pt": "f1c21a4fbda91e5da44953ff42469ede72e93047b1480dd383580ebc2cab23cc",
    "meta.json": "5cb114b36277ab0edf8370a6c4a8ecd4f0c996280bb974b508e371d1d52fec6e",
    "tokenizer.json": "8844e2158f1473ec45388fd3187068c04dad2c3735c6eee28056a55f9aab24b2",
    "tokenizer_config.json": "e91d961e17342c45412552d113f949c8b68942d77182af6186b06c6fca7ba985",
    "special_tokens_map.json": "5aa43c2f985a25296d5a5ce621c2f77376ca8091a47993c492fd5460b895b140",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}
BUILDER_RELATIVE = "zhilv_qiling_v2/bert_entail.py"
BUILDER_SHA256 = "31724b4ebc09f5963fd059e340ba2aec21bdbc07ce8643a3fb26fd1d34aa7be0"
SOURCE_MANIFEST_SHA256 = "ba40416dcd89edd606e116a03b251ad2682fa8b385f64731220bea2f239be481"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path, *, max_bytes=4 * 1024 * 1024):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("existing bounded JSON file required: " + str(path))
    def invalid_constant(value):
        raise ValueError("non-finite JSON constant is forbidden: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_object_no_duplicates, parse_constant=invalid_constant)


def validate_meta(meta):
    if not isinstance(meta, dict) or meta.get("task") != TASK:
        raise ValueError("checkpoint task must be the original legislative-conflict classification task")
    if type(meta.get("num_labels")) is not int or meta["num_labels"] != 5 or meta.get("class_names") != list(CLASS_NAMES):
        raise ValueError("exact native five-class metadata in trained class order is required")
    for key, expected in (("compatible_with_contract_entailment_runtime", False),
                          ("contract_law_quality_accepted", False), ("human_gold", False),
                          ("product_default_unchanged", True), ("auxiliary_legal_dataset", True)):
        if meta.get(key) is not expected:
            raise ValueError("auxiliary-task boundary metadata missing or invalid: " + key)
    if meta.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest differs from the verified training source")
    architecture = meta.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("mamba") is not True or architecture.get("rule_residual") is not False:
        raise ValueError("native Mamba architecture without a rule residual is required")
    mamba = architecture.get("mamba_config")
    if not isinstance(mamba, dict) or any(mamba.get(key) != expected for key, expected in
            (("hidden_size", 64), ("state_size", 8), ("num_hidden_layers", 1))):
        raise ValueError("trained Mamba configuration missing or incompatible")
    if architecture.get("encoder_hidden_size") != 768 or architecture.get("dropout") != .1:
        raise ValueError("trained encoder/head configuration missing or incompatible")
    return meta


def inspect_checkpoint(path):
    """Validate metadata and bytes before importing torch or transformers."""
    path = Path(path).resolve()
    if path.is_file():
        if path.name != "model.pt":
            raise ValueError("checkpoint must name its directory or model.pt")
        path = path.parent
    if not path.is_dir():
        raise ValueError("checkpoint directory is missing")
    meta = validate_meta(read_json(path / "meta.json"))
    hashes = {}
    for name, expected in PINNED_FILES.items():
        target = path / name
        if not target.is_file():
            raise ValueError("required checkpoint file missing: " + name)
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError("verified checkpoint/tokenizer hash mismatch: " + name)
        hashes[name] = actual
    if file_sha256(ROOT / BUILDER_RELATIVE) != BUILDER_SHA256:
        raise ValueError("model builder differs from the strict-reload-verified training implementation")
    return path, meta, hashes


def validate_input(document):
    if not isinstance(document, dict) or set(document) != {"task", "items"} or document.get("task") != TASK:
        raise ValueError("input must contain only the exact auxiliary task and items")
    rows = document.get("items")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ITEMS:
        raise ValueError("items must be a nonempty list of at most " + str(MAX_ITEMS))
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "query", "reference"}:
            raise ValueError("each item requires only id/query/reference; contract/clause/article tasks are unsupported")
        if not isinstance(row["id"], str) or not row["id"].strip() or len(row["id"]) > 200 or row["id"] in ids:
            raise ValueError("each item requires a unique nonempty string id of at most 200 characters")
        ids.add(row["id"])
        for key in ("query", "reference"):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 20000:
                raise ValueError("each " + key + " must be nonempty legislative text of at most 20000 characters")
    return rows


def normalized_probabilities(values):
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError("model must return exactly five class probabilities")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("non-finite or out-of-range model probability")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0 or abs(total - 1) > 1e-5:
        raise ValueError("model probabilities are not normalized")
    return [float(value / total) for value in values]


def write_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("refuse to overwrite existing output: " + str(path))
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def predict(rows, checkpoint, meta, *, device_name):
    """Import heavyweight local dependencies only after all input/hash guards."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers import AutoTokenizer
    from zhilv_qiling_v2 import bert_entail as builder

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; no fallback was made")
    device = torch.device(device_name)
    if device_name == "cuda":
        torch.cuda.set_per_process_memory_fraction(.24)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    # Preflight every item before loading the model. Never silently truncate or
    # split legislative evidence, including on the shorter query tower.
    token_lengths = []
    for row in rows:
        lengths = {key: len(tokenizer(row[key], truncation=False, add_special_tokens=True)["input_ids"])
                   for key in ("query", "reference")}
        if any(length > MAX_TOKENS for length in lengths.values()):
            raise ValueError("item " + row["id"] + " exceeds the trained 384-token tower budget; input was not truncated")
        token_lengths.append(lengths)
    architecture = meta["architecture"]
    model = builder.build_model(enable_mamba=True, enable_rule_head=False,
        mamba_config=architecture["mamba_config"], dropout=architecture["dropout"])
    model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 5)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if model.mamba is None or model.rule_head is not None or model.head[-1].out_features != 5:
        raise RuntimeError("strict native five-class model architecture check failed")
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for row, lengths in zip(rows, token_lengths):
            query = tokenizer(row["query"], truncation=False, return_tensors="pt")
            reference = tokenizer(row["reference"], truncation=False, return_tensors="pt")
            logits, cosine, _ = model(query["input_ids"].to(device), query["attention_mask"].to(device),
                reference["input_ids"].to(device), reference["attention_mask"].to(device))
            if tuple(logits.shape) != (1, 5) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("model returned invalid or non-finite five-class logits")
            probabilities = normalized_probabilities(logits.float().softmax(dim=-1)[0].cpu().tolist())
            cosine_value = float(cosine[0].cpu())
            if not math.isfinite(cosine_value):
                raise RuntimeError("model returned a non-finite auxiliary cosine")
            selected = max(range(5), key=lambda index: probabilities[index])
            predictions.append({"id": row["id"], "predicted_conflict_type": CLASS_NAMES[selected],
                "class_index": selected, "probabilities": dict(zip(CLASS_NAMES, probabilities)),
                "probability_sum": math.fsum(probabilities), "probabilities_are_calibrated": False,
                "cosine": cosine_value, "token_lengths": lengths, "truncated": False,
                "query_sha256": hashlib.sha256(row["query"].encode("utf-8")).hexdigest(),
                "reference_sha256": hashlib.sha256(row["reference"].encode("utf-8")).hexdigest()})
    return predictions, {"device": str(device), "torch": torch.__version__,
        "strict_state_dict_load": True, "weights_only": True, "offline_local_files_only": True,
        "max_tokens_each_tower": MAX_TOKENS, "truncation": False, "cpu_threads": 2}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        if args.out.exists():
            raise ValueError("refuse to overwrite existing output: " + str(args.out))
        rows = validate_input(read_json(args.input_json))
        checkpoint, meta, hashes = inspect_checkpoint(args.checkpoint)
        predictions, runtime = predict(rows, checkpoint, meta, device_name=args.device)
        result = {"status": "completed_auxiliary_legislative_conflict_prediction", "task": TASK,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "class_names": list(CLASS_NAMES),
            "num_labels": 5, "predictions": predictions, "runtime": runtime,
            "checkpoint": {"directory": str(checkpoint), "sha256": hashes},
            "model_builder_sha256": BUILDER_SHA256,
            "input": {"path": str(args.input_json.resolve()), "sha256": file_sha256(args.input_json),
                      "normalization": "none", "query_role": "supplied_lower_level_legislative_provision",
                      "reference_role": "supplied_superior_law_text"},
            "tool_sha256": file_sha256(__file__), "elapsed_seconds": time.perf_counter() - started,
            "auxiliary_task_only": True, "contract_acceptance": False,
            "compatible_with_contract_entailment_runtime": False, "contract_law_quality_accepted": False,
            "independent_project_legal_accuracy_accepted": False, "official_full_benchmark_score": False,
            "product_default_replaced": False, "external_facts_verified": False,
            "meaning": "仅对给定下位法规条文与给定上位法文本预测原始五类立法冲突类型；概率未经校准，不给出合同合法性结论，不核实外部事实，不替代人工法律审查。"}
        write_new_json(args.out, result)
    except Exception as exc:
        print(json.dumps({"status": "failed_closed_no_prediction", "error_type": type(exc).__name__,
                          "error": str(exc), "contract_acceptance": False}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "items": len(predictions), "out": str(args.out.resolve()),
                      "contract_acceptance": False}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
