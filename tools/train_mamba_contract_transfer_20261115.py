"""Isolated LCR-CN shared-parameter transfer into a three-class contract candidate.

Never maps the five released legal labels into entail/contradict/neutral. Uses
the original fixed synthetic split, preserves the entire old three-class head,
and leaves all existing checkpoints, product sources and web defaults intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ForbiddenExternalOperation(BaseException):
    pass


def deny(*args, **kwargs):
    raise ForbiddenExternalOperation("Network and 8B access are forbidden in this isolated training run")


socket.create_connection = socket.socket.connect = socket.socket.connect_ex = deny


def audit_guard(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0]))
        if path.name.lower() == ".env" or path.name.lower().startswith(".env."):
            raise ForbiddenExternalOperation("No environment secret files")
        if "legalone" in str(path).lower() and path.suffix.lower() in {".gguf", ".safetensors", ".bin", ".pt", ".pth"}:
            raise ForbiddenExternalOperation("No LegalOne model weight access")


sys.addaudithook(audit_guard)
SHARED_PREFIXES = ("encoder.", "mamba.", "mamba_input_projection.", "mamba_output_projection.", "attn.", "norm.")
CANONICAL_DATA_SHA = "9a8d60f5facaf45f33eb1d8f827eeb7441c83bf19509ff4814efb3b2dea7c62e"
RAW_DATA_SHA = "e50f7109db8db62645b281528cd459006b81ae452f37c5d471335cee37f68def"
OLD_THREE_SHA = "7cceae24d823c7cafc29a3d93d5b574a8a5114962ff8cb6280538d830c5d9b54"
LEGAL_FIVE_SHA = "f1c21a4fbda91e5da44953ff42469ede72e93047b1480dd383580ebc2cab23cc"
LABELS = ("entail", "contradict", "neutral")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def available_ram():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Cannot verify the necessary host memory budget")
    return status.available_physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "zhilv_qiling_v2/models/bert_mamba_contract_transfer_20261115")
    parser.add_argument("--evidence", type=Path, default=ROOT / "tools/acceptance_results/mamba_contract_transfer_20261115_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    if args.epochs != 3 or args.batch_size not in (1, 2) or args.lr <= 0 or args.lr > 3e-5:
        parser.error("This scoped experiment uses 3 epochs, batch <=2 and learning rate <=3e-5")
    if args.output.exists():
        parser.error("Refuse to overwrite any existing checkpoint directory; select a new version")
    args.evidence.mkdir(parents=True, exist_ok=True)
    receipt_path = args.evidence / "run.json"
    if receipt_path.exists():
        parser.error("Refuse to overwrite an existing training receipt")
    temporary = args.evidence / "temporary"
    temporary.mkdir(exist_ok=False)
    tempfile.tempdir = str(temporary.resolve())
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary.resolve())
    os.environ["ZLQL_MODEL_AUDIT_PATH"] = str((temporary / "isolated_audit.sqlite").resolve())
    old = ROOT / "zhilv_qiling_v2/models/bert_mamba_entail"
    legal = ROOT / "zhilv_qiling_v2/models/bert_mamba_legal_20261115"
    pair_path = ROOT / "tools/acceptance_results/mamba_entail_training/pairs.jsonl"
    integration = ROOT / "tools/acceptance_results/cross_clause_semantics_integration_20261115_v1/receipt.json"
    frozen = json.loads(integration.read_text(encoding="utf-8"))["after_sha256"]
    started = time.perf_counter()
    receipt = {"status": "initializing", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "experimental_three_class_contract_candidate_real_legal_shared_initialization_then_synthetic_weak_finetuning",
        "supervision": "real_released_legal_labels_shared_initialization_then_synthetic_weak_supervision_not_human_gold",
        "source_task": "LCR-CN original five-class legislative conflict classification",
        "target_task": "contract_synthetic_entail_contradict_neutral", "target_labels": list(LABELS),
        "five_to_three_label_mapping": False, "required_old_three_class_entire_head_preservation": True,
        "training_data_seed": "original fixed saved split from 20261114; never regenerated",
        "optimizer_seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "gpu_allocator_fraction_limit": 0.24, "cpu_threads": 2, "cpu_interop_threads": 1,
        "network_blocked": True, "env_files_forbidden": True, "real_8b_execution": False,
        "product_default_replaced": False, "independent_human_gold": False, "official_benchmark_score": False,
        "test_set_reused_for_regression_only": True, "checkpoint_selection": "fixed final epoch 3; no test-based selection or hyperparameter search",
        "limitations": ["All 288 target pairs and their norms are synthetic weak supervision, not current law or human annotation.",
            "Existing target valid/test have already been inspected in earlier engineering runs; scores are reused regression observations.",
            "LCR source itself inherited the old three-class checkpoint; this is a sequential transfer experiment, not a clean independent pretraining ablation.",
            "No evidence yet of improved real contract legal accuracy; web default remains unchanged."],
        "implementation_sha256": {"training_tool": file_hash(__file__), "bert_entail.py": file_hash(ROOT / "zhilv_qiling_v2/bert_entail.py")}}

    def save():
        receipt["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    torch = None
    try:
        receipt["product_source_hashes_before"] = {name: file_hash(ROOT / name) for name in frozen}
        if receipt["product_source_hashes_before"] != frozen:
            raise RuntimeError("Product source drifted from the frozen integration")
        receipt["source_checkpoints"] = {"three_class": {"path": str(old), "sha256": file_hash(old / "model.pt")},
                                         "legal_five_class": {"path": str(legal), "sha256": file_hash(legal / "model.pt")}}
        for name, expected in (("three_class", OLD_THREE_SHA), ("legal_five_class", LEGAL_FIVE_SHA)):
            receipt["source_checkpoints"][name]["verified_training_receipt_expected_sha256"] = expected
            if receipt["source_checkpoints"][name]["sha256"] != expected:
                raise RuntimeError("Source weights do not match the verified original training receipt: " + name)
        receipt["source_checkpoint_provenance_pinned"] = True
        raw = pair_path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
        canonical = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
        if hashlib.sha256(raw).hexdigest() != RAW_DATA_SHA or hashlib.sha256(canonical.encode()).hexdigest() != CANONICAL_DATA_SHA:
            raise RuntimeError("The exact original target data or its fixed split changed")
        splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "valid", "test")}
        if {name: len(part) for name, part in splits.items()} != {"train": 216, "valid": 36, "test": 36}:
            raise RuntimeError("Target split sizes changed")
        for field in ("source_id", "template_family", "group_id"):
            groups = {name: {row[field] for row in part} for name, part in splits.items()}
            if any(groups[left] & groups[right] for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))):
                raise RuntimeError("Target group leakage: " + field)
        if any(row["supervision"] != "synthetic_weak_supervision_not_human_gold" for row in rows):
            raise RuntimeError("Target label provenance changed")
        receipt["data"] = {"path": str(pair_path), "raw_sha256": RAW_DATA_SHA, "canonical_sha256": CANONICAL_DATA_SHA,
            "raw_crlf_lines": raw.count(b"\r\n"), "counts": {name: dict(Counter(row["label"] for row in part)) for name, part in splits.items()},
            "fixed_pair_ids": {name: [row["pair_id"] for row in part] for name, part in splits.items()},
            "groups_disjoint": True, "human_gold": False, "regenerated": False}
        (args.evidence / "fixed_pairs.jsonl").write_bytes(raw)
        old_meta = json.loads((old / "meta.json").read_text(encoding="utf-8"))
        legal_meta = json.loads((legal / "meta.json").read_text(encoding="utf-8"))
        if old_meta["architecture"]["mamba_config"] != legal_meta["architecture"]["mamba_config"] or legal_meta["num_labels"] != 5:
            raise RuntimeError("Source architecture mismatch")
        receipt["source_legal_pretraining"] = {"num_labels": 5, "class_names": legal_meta["class_names"],
            "source_manifest_sha256": legal_meta["source_manifest_sha256"], "optimizer_steps": legal_meta["optimizer_steps"],
            "transfer_uses_labels": False, "encoder_was_frozen": "encoder" in legal_meta["architecture"]["frozen"]}
        free_ram = available_ram()
        receipt["free_host_ram_before_bytes"] = free_ram
        if free_ram < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB host RAM is available; no applications will be closed")
        save()
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from zhilv_qiling_v2 import bert_entail as B
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if tuple(B.LABELS) != LABELS or not torch.cuda.is_available():
            raise RuntimeError("Matching three-class CUDA runtime required; no CPU training fallback")
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        receipt["resources"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "free_gpu_before_bytes": free_gpu, "total_gpu_bytes": total_gpu, "allocator_fraction_limit": 0.24,
            "no_user_applications_closed": True}
        if free_gpu < 2500 * 1024**2:
            raise RuntimeError("Less than 2500 MiB GPU memory available; stop without closing applications")
        torch.cuda.set_per_process_memory_fraction(0.24)
        torch.cuda.reset_peak_memory_stats()
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(str(old), local_files_only=True)
        tokenizer.backend_tokenizer.no_truncation()
        tokenizer.backend_tokenizer.no_padding()
        lengths = {name: {field: max(len(tokenizer(row[field], truncation=False)["input_ids"]) for row in part)
                          for field in ("clause", "article")} for name, part in splits.items()}
        largest = max(value for part in lengths.values() for value in part.values())
        if largest > 384:
            receipt["rejected_sequence_lengths"] = [{"pair_id": row["pair_id"], "lengths": {field: len(tokenizer(row[field], truncation=False)["input_ids"]) for field in ("clause", "article")}}
                for row in rows if any(len(tokenizer(row[field], truncation=False)["input_ids"]) > 384 for field in ("clause", "article"))]
            raise RuntimeError("Complete input exceeds the allowed 384-token transfer budget; no rows silently removed")
        max_length = max(96, largest)
        receipt["sequence_budget"] = {"legacy_training_tokens_per_tower": 96, "actual_max_lengths_by_split": lengths,
            "chosen_complete_input_budget": max_length, "changed_from_legacy_budget": max_length != 96,
            "truncation": False, "persistent_truncation_disabled": True, "rows_excluded": [],
            "all_288_inputs_complete": True}
        runtime_oversize = [{"pair_id": row["pair_id"], "clause_tokens": len(tokenizer(row["clause"], truncation=False)["input_ids"]),
                            "article_tokens": len(tokenizer(row["article"], truncation=False)["input_ids"])}
                           for row in rows if len(tokenizer(row["clause"], truncation=False)["input_ids"]) > B.MAX_CLAUSE
                           or len(tokenizer(row["article"], truncation=False)["input_ids"]) > B.MAX_ARTICLE]
        receipt["existing_runtime_input_limits"] = {"clause": B.MAX_CLAUSE, "article": B.MAX_ARTICLE,
            "all_fixed_inputs_within_limits": not runtime_oversize, "oversize_fixed_inputs": runtime_oversize,
            "training_budget_does_not_expand_product_runtime_limits": True}
        old_state = torch.load(old / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        legal_state = torch.load(legal / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        if set(old_state) != set(legal_state) or len(old_state) != 226:
            raise RuntimeError("Unexpected source state keys")
        shared = sorted(key for key in old_state if key == "mamba_gate" or key.startswith(SHARED_PREFIXES))
        head_keys = sorted(key for key in old_state if key.startswith("head."))
        if len(shared) != 222 or len(head_keys) != 4 or set(shared) | set(head_keys) != set(old_state):
            raise RuntimeError("Only the 222 shared tensors and four complete original head tensors are allowed")
        if any(old_state[key].shape != legal_state[key].shape or old_state[key].dtype != legal_state[key].dtype for key in shared):
            raise RuntimeError("Shared state shape/dtype mismatch")
        mismatches = {key: {"three": list(old_state[key].shape), "five": list(legal_state[key].shape)}
                      for key in old_state if old_state[key].shape != legal_state[key].shape}
        if set(mismatches) != {"head.3.weight", "head.3.bias"}:
            raise RuntimeError("Unexpected task-specific shape mismatch")
        receipt["transfer"] = {"shared_tensor_count": 222, "preserved_head_tensor_count": 4,
            "shared_keys": shared, "excluded_legal_keys": head_keys, "shape_mismatches": mismatches,
            "all_226_target_keys_complete": True, "strict_load": True,
            "old_head_sha256": tensor_hash({key: old_state[key] for key in head_keys}),
            "legal_shared_sha256": tensor_hash({key: legal_state[key] for key in shared}),
            "encoder_five_equals_original_three": tensor_hash({k: v for k, v in old_state.items() if k.startswith("encoder.")}) == tensor_hash({k: v for k, v in legal_state.items() if k.startswith("encoder.")})}
        model = B.build_model(enable_mamba=True, enable_rule_head=False, dropout=0.10,
                              mamba_config=old_meta["architecture"]["mamba_config"])
        model.load_state_dict(old_state, strict=True)
        model.to(device)

        def encode(part):
            left = tokenizer([row["clause"] for row in part], padding=True, truncation=False, return_tensors="pt")
            right = tokenizer([row["article"] for row in part], padding=True, truncation=False, return_tensors="pt")
            if left["input_ids"].shape[1] > max_length or right["input_ids"].shape[1] > max_length:
                raise RuntimeError("Unexpected complete sequence length; no truncation fallback")
            targets = torch.tensor([LABELS.index(row["label"]) for row in part], device=device)
            return left, right, targets

        def evaluate(part):
            model.eval()
            confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
            predictions = []
            with torch.no_grad():
                for start in range(0, len(part), args.batch_size):
                    batch = part[start:start + args.batch_size]
                    left, right, _ = encode(batch)
                    logits, cosine, _ = model(left["input_ids"].to(device), left["attention_mask"].to(device), right["input_ids"].to(device), right["attention_mask"].to(device))
                    for row, probs, cos in zip(batch, logits.softmax(-1).tolist(), cosine.tolist()):
                        predicted = LABELS[max(range(3), key=lambda i: probs[i])]
                        confusion[row["label"]][predicted] += 1
                        predictions.append({"pair_id": row["pair_id"], "expected": row["label"], "predicted": predicted,
                            "probabilities": dict(zip(LABELS, probs)), "cosine": cos, "complete_input": True})
            f1 = []
            for label in LABELS:
                tp = confusion[label][label]
                fp = sum(confusion[other][label] for other in LABELS if other != label)
                fn = sum(confusion[label][other] for other in LABELS if other != label)
                f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            return {"rows": len(part), "accuracy": sum(confusion[label][label] for label in LABELS) / len(part),
                "macro_f1": sum(f1) / 3, "confusion": confusion, "predictions": predictions,
                "scope": "reused_grouped_synthetic_weak_supervision_regression_not_human_gold"}

        receipt["original_three_class_before_transfer"] = {name: evaluate(splits[name]) for name in ("valid", "test")}
        merged = dict(old_state)
        merged.update({key: legal_state[key] for key in shared})
        model.load_state_dict(merged, strict=True)
        state = model.state_dict()
        receipt["transfer"]["loaded_shared_sha256"] = tensor_hash({key: state[key] for key in shared})
        receipt["transfer"]["loaded_head_sha256"] = tensor_hash({key: state[key] for key in head_keys})
        if receipt["transfer"]["loaded_shared_sha256"] != receipt["transfer"]["legal_shared_sha256"] or receipt["transfer"]["loaded_head_sha256"] != receipt["transfer"]["old_head_sha256"]:
            raise RuntimeError("The actual loaded shared values or preserved entire head differ")
        receipt["old_three_class_entire_head_preserved"] = True
        del state, merged, old_state, legal_state
        gc.collect()
        receipt["transferred_initialization_before_finetune"] = {name: evaluate(splits[name]) for name in ("valid", "test")}
        for module in (model.encoder, model.attn, model.norm, model.mamba.embeddings):
            module.requires_grad_(False)
        initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
        frozen_before = tensor_hash({name: value for name, value in model.named_parameters() if not value.requires_grad})
        receipt["architecture"] = {**old_meta["architecture"], "frozen": ["encoder", "attn", "norm", "mamba.embeddings_unused"],
            "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "all_parameters": sum(value.numel() for value in model.parameters()), "rule_residual": False}
        trainable = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        steps = math.ceil(len(splits["train"]) / args.batch_size) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
        generator = torch.Generator().manual_seed(args.seed)
        receipt.update(status="training", planned_optimizer_steps=steps, completed_optimizer_steps=0,
            train_rows_per_epoch=len(splits["train"]), actual_examples_with_optimizer_update=0,
            history=[], gradient_evidence={}, actual_train_pair_ids_per_epoch=[])
        save()
        print(f"Actual transfer fine-tuning starts: {steps} optimizer steps; all inputs complete; unchanged split 216/36/36", flush=True)
        training_started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            model.encoder.eval()
            order = torch.randperm(len(splits["train"]), generator=generator).tolist()
            losses, visited = [], []
            receipt["current_epoch"] = epoch
            receipt["current_epoch_completed_pair_ids"] = visited
            for start in range(0, len(order), args.batch_size):
                batch = [splits["train"][index] for index in order[start:start + args.batch_size]]
                receipt["current_batch_attempted_pair_ids"] = [row["pair_id"] for row in batch]
                receipt["current_batch_optimizer_update_completed"] = False
                left, right, targets = encode(batch)
                optimizer.zero_grad(set_to_none=True)
                logits, _, _ = model(left["input_ids"].to(device), left["attention_mask"].to(device), right["input_ids"].to(device), right["attention_mask"].to(device))
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite real training loss")
                loss.backward()
                for name, value in model.named_parameters():
                    if value.requires_grad and value.grad is not None:
                        norm = float(value.grad.detach().float().norm())
                        if not math.isfinite(norm):
                            raise RuntimeError("Non-finite real gradient: " + name)
                        observed = receipt["gradient_evidence"].setdefault(name, {"max_l2": 0.0, "nonzero_steps": 0})
                        observed["max_l2"] = max(observed["max_l2"], norm)
                        observed["nonzero_steps"] += int(norm > 0)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                receipt["completed_optimizer_steps"] += 1
                receipt["actual_examples_with_optimizer_update"] += len(batch)
                receipt["current_batch_optimizer_update_completed"] = True
                losses.append(float(loss.detach()))
                visited.extend(row["pair_id"] for row in batch)
                if receipt["completed_optimizer_steps"] % 36 == 0:
                    save()
                    print(f"epoch={epoch} step={receipt['completed_optimizer_steps']}/{steps} loss={losses[-1]:.6f}", flush=True)
            if sorted(visited) != sorted(row["pair_id"] for row in splits["train"]):
                raise RuntimeError("An epoch did not actually visit every fixed training row exactly once")
            receipt["actual_train_pair_ids_per_epoch"].append(visited)
            receipt["history"].append({"epoch": epoch, "actual_rows": len(visited), "batches": len(losses),
                "mean_cross_entropy": sum(losses) / len(losses), "first_batch_loss": losses[0], "last_batch_loss": losses[-1]})
            save()
        receipt["training_seconds"] = time.perf_counter() - training_started
        if receipt["completed_optimizer_steps"] != steps:
            raise RuntimeError("Actual optimizer count did not reach the explicit plan")
        receipt["parameter_changes"] = {name: {"l2_delta": float((dict(model.named_parameters())[name].detach().cpu() - before).float().norm()),
            "numel": before.numel()} for name, before in initial.items()}
        changed_ssm = [name for name, value in receipt["parameter_changes"].items() if name.startswith("mamba.") and value["l2_delta"] > 0 and any(key in name for key in ("A_log", ".D", "x_proj", "dt_proj"))]
        if not changed_ssm or not any(value["nonzero_steps"] > 0 for name, value in receipt["gradient_evidence"].items() if name.startswith("mamba.")):
            raise RuntimeError("No actual selective Mamba parameter update or nonzero gradient")
        receipt["changed_selective_state_space_parameters"] = changed_ssm
        receipt["frozen_parameters_unchanged"] = frozen_before == tensor_hash({name: value for name, value in model.named_parameters() if not value.requires_grad})
        if not receipt["frozen_parameters_unchanged"]:
            raise RuntimeError("A frozen model parameter changed")
        receipt["after_finetune"] = {name: evaluate(splits[name]) for name in ("valid", "test")}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / ".gitignore").write_text("*\n", encoding="ascii")
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, args.output / "model.pt")
        token_files = ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for name in token_files:
            shutil.copyfile(old / name, args.output / name)
        original_tokenizer_json = json.loads((old / "tokenizer.json").read_text(encoding="utf-8"))
        artifact_tokenizer_json = dict(original_tokenizer_json)
        artifact_tokenizer_json["truncation"] = None
        artifact_tokenizer_json["padding"] = None
        (args.output / "tokenizer.json").write_text(json.dumps(artifact_tokenizer_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt["tokenizer_files_exactly_copied_from_old_three"] = {name: file_hash(old / name) == file_hash(args.output / name) for name in token_files}
        receipt["tokenizer_artifact"] = {
            "source_persistent_truncation": original_tokenizer_json.get("truncation"),
            "source_persistent_padding": original_tokenizer_json.get("padding"),
            "artifact_persistent_truncation": artifact_tokenizer_json["truncation"],
            "artifact_persistent_padding": artifact_tokenizer_json["padding"],
            "source_tokenizer_json_sha256": file_hash(old / "tokenizer.json"),
            "artifact_tokenizer_json_sha256": file_hash(args.output / "tokenizer.json"),
            "vocabulary_bytes_identical": file_hash(old / "vocab.txt") == file_hash(args.output / "vocab.txt"),
            "semantic_fields_unchanged": {key: value for key, value in original_tokenizer_json.items() if key not in ("truncation", "padding")} ==
                {key: value for key, value in artifact_tokenizer_json.items() if key not in ("truncation", "padding")},
            "change": "Only the new artifact tokenizer.json is reserialized with persistent truncation and padding disabled; source files are untouched."}
        if not receipt["tokenizer_artifact"]["vocabulary_bytes_identical"] or not receipt["tokenizer_artifact"]["semantic_fields_unchanged"]:
            raise RuntimeError("Artifact tokenizer vocabulary or semantic fields changed")
        meta = {"architecture": receipt["architecture"], "supervision": receipt["supervision"], "training_scope": receipt["scope"],
            "num_labels": 3, "class_names": list(LABELS), "compatible_with_contract_entailment_runtime": True,
            "human_gold": False, "contract_law_quality_accepted": False, "product_default_unchanged": True,
            "optimizer_steps": receipt["completed_optimizer_steps"], "epochs": args.epochs, "seed": args.seed,
            "data_sha256": CANONICAL_DATA_SHA, "data_raw_sha256": RAW_DATA_SHA, "source_checkpoints": receipt["source_checkpoints"],
            "transfer": {key: value for key, value in receipt["transfer"].items() if key != "shared_keys"},
            "max_training_tokens_per_tower": max_length, "actual_input_lengths": lengths, "truncation": False,
            "tokenizer_artifact": receipt["tokenizer_artifact"],
            "existing_runtime_input_limits": receipt["existing_runtime_input_limits"],
            "valid": {key: value for key, value in receipt["after_finetune"]["valid"].items() if key != "predictions"},
            "test": {key: value for key, value in receipt["after_finetune"]["test"].items() if key != "predictions"},
            "training_evidence": str(args.evidence), "limitations": receipt["limitations"], "trained_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        (args.output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt["checkpoint"] = {"path": str(args.output / "model.pt"), "bytes": (args.output / "model.pt").stat().st_size,
            "sha256": file_hash(args.output / "model.pt")}
        model.to("cpu")
        del optimizer, scheduler, trainable, initial, loss, logits, left, right, targets
        gc.collect()
        torch.cuda.empty_cache()
        os.environ["ZLQL_ENTAILMENT_DEVICE"] = "cuda"
        reloaded, reloaded_tokenizer, loaded_meta = B.load_model(args.output)
        receipt["tokenizer_artifact"]["reloaded_persistent_truncation"] = reloaded_tokenizer.backend_tokenizer.truncation
        receipt["tokenizer_artifact"]["reloaded_persistent_padding"] = reloaded_tokenizer.backend_tokenizer.padding
        if reloaded_tokenizer.backend_tokenizer.truncation is not None or reloaded_tokenizer.backend_tokenizer.padding is not None:
            raise RuntimeError("Reloaded artifact unexpectedly contains persistent tokenizer truncation or padding")
        reloaded_tokenizer.backend_tokenizer.no_truncation()
        reloaded_tokenizer.backend_tokenizer.no_padding()
        model = reloaded
        tokenizer = reloaded_tokenizer
        receipt["strict_reload"] = loaded_meta["loaded_architecture"]
        receipt["strict_reload_evaluation"] = {name: evaluate(splits[name]) for name in ("valid", "test")}
        receipt["strict_reload_max_probability_delta"] = {}
        for name in ("valid", "test"):
            before_rows = receipt["after_finetune"][name]["predictions"]
            after_rows = receipt["strict_reload_evaluation"][name]["predictions"]
            max_delta = max(abs(left["probabilities"][label] - right["probabilities"][label])
                            for left, right in zip(before_rows, after_rows) for label in LABELS)
            receipt["strict_reload_max_probability_delta"][name] = max_delta
            if any((left["pair_id"], left["predicted"]) != (right["pair_id"], right["predicted"]) for left, right in zip(before_rows, after_rows)) or max_delta > 1e-6:
                raise RuntimeError("Reloaded complete-input predictions differ from the saved model")
        predictor = B.Predictor(reloaded, reloaded_tokenizer, device=device, meta=loaded_meta)
        receipt["runtime_predictor_smoke"] = []
        for label in LABELS:
            row = next(row for row in splits["test"] if row["label"] == label)
            prediction = predictor.predict(row["clause"], row["article"])
            receipt["runtime_predictor_smoke"].append({"pair_id": row["pair_id"], "synthetic_expected": label,
                "prediction": prediction, "actual_input_truncated": prediction["input_truncated"],
                "complete_input": not prediction["input_truncated"]})
        receipt["source_checkpoints_unchanged"] = all(file_hash(Path(value["path"]) / "model.pt") == value["sha256"] for value in receipt["source_checkpoints"].values())
        if not receipt["source_checkpoints_unchanged"]:
            raise RuntimeError("An original checkpoint changed")
        receipt.update(status="trained_strict_reloaded_candidate_only", peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), observed_test_improvement_is_independent_acceptance=False)
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["product_source_hashes_after"] = {name: file_hash(ROOT / name) for name in frozen}
        receipt["product_sources_unchanged"] = receipt["product_source_hashes_after"] == frozen
        if not receipt["product_sources_unchanged"]:
            receipt.update(status="failed", error="Frozen product sources changed during the run")
        if torch is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        receipt.update(finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), total_seconds=time.perf_counter() - started)
        save()
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "actual_steps": receipt.get("completed_optimizer_steps"),
            "checkpoint": receipt.get("checkpoint"), "error": receipt.get("error")}, ensure_ascii=False), flush=True)
    return 0 if receipt["status"] == "trained_strict_reloaded_candidate_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
