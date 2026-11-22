"""Fixed, CPU-only incremental legal-transfer ablation on existing weak labels.

Released LCR five-class heads are never mapped into contract three-class heads.
All arms use one identical freshly initialized target head, target rows, order,
steps and optimizer. Existing inspected synthetic validation/test remain only
engineering regressions, not independent legal accuracy evidence.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from .acceptance_dataset import digest, read_jsonl, write_new_json

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("entail", "contradict", "neutral")
RAW_DATA_SHA = "e50f7109db8db62645b281528cd459006b81ae452f37c5d471335cee37f68def"
CHECKPOINTS = {
    "base": ("zhilv_qiling_v2/models/bert_mamba_entail", "7cceae24d823c7cafc29a3d93d5b574a8a5114962ff8cb6280538d830c5d9b54"),
    "legal_increment": ("zhilv_qiling_v2/models/bert_mamba_legal_20261115", "f1c21a4fbda91e5da44953ff42469ede72e93047b1480dd383580ebc2cab23cc"),
}
SHARED = ("encoder.", "mamba.", "mamba_input_projection.", "mamba_output_projection.", "attn.", "norm.")
PROFILES = ("head_only", "encoder_frozen_mamba_attention_head")


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def is_shared(name: str) -> bool:
    return name == "mamba_gate" or name.startswith(SHARED)


def trainable_parameter(name: str, profile: str) -> bool:
    if profile not in PROFILES:
        raise ValueError("unknown_fixed_freeze_profile")
    return name.startswith("head.") or (profile == PROFILES[1] and not name.startswith("encoder.") and is_shared(name))


def build_ablation_plan(pair_file: Path | None = None) -> dict:
    path = pair_file or ROOT / "tools/acceptance_results/mamba_entail_training/pairs.jsonl"
    if file_sha(path) != RAW_DATA_SHA:
        raise ValueError("original_fixed_weak_target_data_hash_changed")
    rows = read_jsonl(path)
    if len(rows) != 288 or any(row.get("supervision") != "synthetic_weak_supervision_not_human_gold" for row in rows):
        raise ValueError("original_288_synthetic_weak_pairs_required")
    parts = {split: [r for r in rows if r["split"] == split] for split in ("train", "valid", "test")}
    for field in ("source_id", "group_id", "template_family"):
        groups = {split: {r[field] for r in part} for split, part in parts.items()}
        if any(groups[a] & groups[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
            raise ValueError("original_target_group_leakage")
    selection = {}
    for split, count in (("train", 8), ("valid", 4), ("test", 4)):
        selection[split] = sum(([r["pair_id"] for r in sorted(parts[split], key=lambda r: r["pair_id"])
                                 if r["label"] == label][:count] for label in LABELS), [])
        if len(selection[split]) != count * 3:
            raise ValueError("insufficient_predeclared_balanced_target_rows")
    order = list(selection["train"])
    random.Random(20261118).shuffle(order)
    return {"schema_version": "zlql-controlled-transfer-v1", "data_path": str(path.resolve()),
            "target_data_raw_sha256": RAW_DATA_SHA, "seed": 20261118,
            "target_labels": list(LABELS), "target_pair_ids": selection, "training_order": order,
            "batch_size": 2, "optimizer_steps_per_arm": 12, "learning_rate": 1e-4,
            "optimizer": {"name": "AdamW", "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8},
            "arms": [{"id": init + "__" + profile, "initialization": init, "freeze_profile": profile}
                     for init in CHECKPOINTS for profile in PROFILES],
            "device": "cpu", "cpu_threads": 2, "cpu_interop_threads": 1,
            "fresh_identical_target_head": True, "five_to_three_label_mapping": False,
            "selection_policy": "fixed final step 12; no validation/test checkpoint or hyperparameter selection",
            "project_default_changed": False, "human_gold": False, "quality_acceptance": False,
            "limitations": ["All target labels are existing synthetic weak supervision; validation/test were already inspected.",
                            "The legal source inherited the base checkpoint: this isolates incremental legal-task updates, not clean pretraining causality.",
                            "Small CPU diagnostic, one seed and 24 training rows; retain all negative results, no independent legal-quality claim."]}


def _tensor_sha(state: dict) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run_ablation(output: Path) -> dict:
    if output.exists():
        raise ValueError("fresh_ablation_directory_required")
    plan = build_ablation_plan()
    source_info = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        path = ROOT / relative
        actual = file_sha(path / "model.pt")
        if actual != expected:
            raise ValueError("pinned_source_checkpoint_changed:" + name)
        source_info[name] = {"path": str(path), "weights_sha256": actual,
                             "meta_sha256": file_sha(path / "meta.json")}
    output.mkdir(parents=True)
    write_new_json(output / "plan.json", {**plan, "source_checkpoints": source_info})
    # No dotenv loader, endpoint, CUDA allocation, or default-checkpoint write.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from transformers import AutoTokenizer
    from . import bert_entail as B
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    meta = json.loads((ROOT / CHECKPOINTS["base"][0] / "meta.json").read_text(encoding="utf-8"))
    legal_meta = json.loads((ROOT / CHECKPOINTS["legal_increment"][0] / "meta.json").read_text(encoding="utf-8"))
    if meta["architecture"]["mamba_config"] != legal_meta["architecture"]["mamba_config"] or legal_meta.get("num_labels") != 5:
        raise ValueError("matching_shared_architecture_and_distinct_five_class_source_required")
    tokenizer = AutoTokenizer.from_pretrained(source_info["base"]["path"], local_files_only=True)
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    rows = {r["pair_id"]: r for r in read_jsonl(Path(plan["data_path"]))}
    parts = {split: [rows[pid] for pid in ids] for split, ids in plan["target_pair_ids"].items()}
    lengths = {split: {field: max(len(tokenizer(r[field], truncation=False)["input_ids"]) for r in part)
                      for field in ("clause", "article")} for split, part in parts.items()}
    if max(v for split in lengths.values() for v in split.values()) > 384:
        raise ValueError("complete_input_exceeds_fixed_384_budget_no_truncation")
    receipt = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "plan_sha256": file_sha(output / "plan.json"), "arms": [], "actual_training": True,
               "human_gold": False, "quality_acceptance": False, "product_default_changed": False,
               "input_lengths": lengths, "device": "cpu", "torch": torch.__version__,
               "limitations": plan["limitations"], "implementation_sha256": file_sha(Path(__file__))}
    def save():
        path = output / "run.json"
        temp = output / "run.json.tmp"
        temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    def encode(batch):
        left = tokenizer([r["clause"] for r in batch], padding=True, truncation=False, return_tensors="pt")
        right = tokenizer([r["article"] for r in batch], padding=True, truncation=False, return_tensors="pt")
        return left, right, torch.tensor([LABELS.index(r["label"]) for r in batch])
    def evaluate(model, part):
        model.eval()
        confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
        predictions = []
        with torch.no_grad():
            for start in range(0, len(part), 2):
                batch = part[start:start + 2]
                left, right, _ = encode(batch)
                logits, _, _ = model(left["input_ids"], left["attention_mask"], right["input_ids"], right["attention_mask"])
                for row, probs in zip(batch, logits.softmax(-1).tolist()):
                    label = LABELS[max(range(3), key=lambda i: probs[i])]
                    confusion[row["label"]][label] += 1
                    predictions.append({"pair_id": row["pair_id"], "weak_expected": row["label"], "predicted": label,
                                        "probabilities": dict(zip(LABELS, probs))})
        f1 = []
        for label in LABELS:
            tp = confusion[label][label]
            fp = sum(confusion[o][label] for o in LABELS if o != label)
            fn = sum(confusion[label][o] for o in LABELS if o != label)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        return {"accuracy": sum(confusion[a][a] for a in LABELS) / len(part), "macro_f1": sum(f1) / 3,
                "confusion": confusion, "predictions": predictions, "rows": len(part), "human_gold": False}
    save()
    common_head_hash = None
    try:
        for arm_plan in plan["arms"]:
            started = time.perf_counter()
            # CPU generator only; no GPU service/model is accessed.
            torch.random.default_generator.manual_seed(plan["seed"])
            model = B.build_model(enable_mamba=True, enable_rule_head=False, dropout=.10,
                                  mamba_config=meta["architecture"]["mamba_config"])
            source = torch.load(Path(source_info[arm_plan["initialization"]]["path"]) / "model.pt", map_location="cpu", weights_only=True, mmap=True)
            shared = {name: value for name, value in source.items() if is_shared(name)}
            expected_shared = {name for name in model.state_dict() if is_shared(name)}
            if set(shared) != expected_shared:
                raise ValueError("shared_source_state_keys_incomplete")
            target_state = model.state_dict()
            target_state.update(shared)
            model.load_state_dict(target_state, strict=True)
            head_hash = _tensor_sha({n: p for n, p in model.named_parameters() if n.startswith("head.")})
            if common_head_hash is not None and head_hash != common_head_hash:
                raise ValueError("target_head_initialization_not_identical")
            common_head_hash = head_hash
            for name, parameter in model.named_parameters():
                parameter.requires_grad = trainable_parameter(name, arm_plan["freeze_profile"])
            trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
            initial = {n: p.detach().clone() for n, p in trainable.items()}
            frozen_before = _tensor_sha({n: p for n, p in model.named_parameters() if not p.requires_grad})
            arm = {**arm_plan, "status": "training", "target_head_initial_sha256": head_hash,
                   "source_five_class_head_loaded": False, "label_mapping": None,
                   "trainable_parameters": {n: p.numel() for n, p in trainable.items()},
                   "completed_optimizer_steps": 0, "history": [], "gradient_nonzero_steps": {n: 0 for n in trainable},
                   "before": {split: evaluate(model, parts[split]) for split in ("valid", "test")}}
            receipt["arms"].append(arm)
            save()
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0., betas=(.9, .999), eps=1e-8)
            for step in range(12):
                model.eval()
                model.head.train()
                if arm_plan["freeze_profile"] == PROFILES[1]:
                    for module in (model.mamba, model.attn, model.norm, model.dropout):
                        module.train()
                torch.random.default_generator.manual_seed(plan["seed"] + step)
                ids = plan["training_order"][step * 2:step * 2 + 2]
                left, right, targets = encode([rows[pid] for pid in ids])
                optimizer.zero_grad(set_to_none=True)
                logits, _, _ = model(left["input_ids"], left["attention_mask"], right["input_ids"], right["attention_mask"])
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                for name, parameter in trainable.items():
                    if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
                        arm["gradient_nonzero_steps"][name] += 1
                torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.)
                optimizer.step()
                arm["completed_optimizer_steps"] += 1
                arm["history"].append({"step": step + 1, "pair_ids": ids, "loss": float(loss.detach())})
                save()
            arm["after"] = {split: evaluate(model, parts[split]) for split in ("valid", "test")}
            arm["parameter_l2_deltas"] = {n: float((p.detach() - initial[n]).float().norm()) for n, p in trainable.items()}
            arm["frozen_parameters_unchanged"] = frozen_before == _tensor_sha({n: p for n, p in model.named_parameters() if not p.requires_grad})
            if not arm["frozen_parameters_unchanged"] or not any(v > 0 for v in arm["parameter_l2_deltas"].values()):
                raise ValueError("actual_parameter_update_or_freeze_assertion_failed")
            arm_path = output / (arm_plan["id"] + ".trainable.pt")
            torch.save({n: p.detach().cpu() for n, p in trainable.items()}, arm_path)
            saved = torch.load(arm_path, map_location="cpu", weights_only=True)
            if _tensor_sha(saved) != _tensor_sha(trainable):
                raise ValueError("saved_trainable_state_roundtrip_failed")
            arm.update(status="completed_fixed_step_diagnostic", trainable_state_sha256=file_sha(arm_path),
                       trainable_state_file=arm_path.name, elapsed_seconds=time.perf_counter() - started,
                       save_roundtrip_exact=True, checkpoint_recipe="strict shared-source load + common seed fresh 3-class head + saved trainable tensors")
            del saved, optimizer, initial, trainable, shared, source, target_state, model, logits, loss
            gc.collect()
            save()
        comparisons = []
        indexed = {arm["id"]: arm for arm in receipt["arms"]}
        for profile in PROFILES:
            base, transfer = indexed["base__" + profile], indexed["legal_increment__" + profile]
            comparisons.append({"freeze_profile": profile,
                "weak_test_accuracy_delta": transfer["after"]["test"]["accuracy"] - base["after"]["test"]["accuracy"],
                "weak_test_macro_f1_delta": transfer["after"]["test"]["macro_f1"] - base["after"]["test"]["macro_f1"],
                "negative_or_zero_results_retained": True, "independent_quality_conclusion": None})
        receipt.update(status="four_arm_controlled_weak_diagnostic_completed", comparisons=comparisons,
                       default_checkpoint_sha256_unchanged=all(file_sha(Path(v["path"]) / "model.pt") == v["weights_sha256"] for v in source_info.values()),
                       source_checkpoints=source_info)
        if not receipt["default_checkpoint_sha256_unchanged"]:
            raise ValueError("source_checkpoint_changed_during_ablation")
    except BaseException as exc:
        receipt.update(status="interrupted_or_failed_diagnostic", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    save()
    return receipt
