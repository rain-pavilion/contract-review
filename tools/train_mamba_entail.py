"""Actually train an opt-in small Mamba entailment branch on grouped synthetic data.

This is a lightweight engineering training run, not sufficient training or an
independent human-labelled legal benchmark. Legacy checkpoints and defaults are
never overwritten. Requires an idle GPU budget; no CPU or network fallback.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from mamba_entail_synthetic_data import build_dataset


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.20)
    args = parser.parse_args()
    output = ROOT / "zhilv_qiling_v2/models/bert_mamba_entail"
    evidence = ROOT / "tools/acceptance_results/mamba_entail_training"
    evidence.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    receipt_file = evidence / "run.json"
    if (output / "model.pt").exists():
        raise RuntimeError("New checkpoint already exists; inspect it rather than overwriting")
    receipt = {"status": "initializing", "scope": "lightweight_actual_training_not_sufficient_training_or_human_gold",
               "seed": args.seed, "command": sys.argv, "cpu_threads": 2, "cpu_interop_threads": 1,
               "product_default_replaced": False, "original_checkpoint_modified": False,
               "supervision": "synthetic_weak_supervision_not_human_gold", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    def save():
        receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    torch = None
    try:
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from zhilv_qiling_v2 import bert_entail as B, entailment as E
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("A free CUDA budget is required; refusing to compete with the CPU model")
        free, total = torch.cuda.mem_get_info()
        if free < 2500 * 1024**2:
            raise RuntimeError("Less than 2500 MiB GPU memory is free; no user application will be closed")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        receipt["resources"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "free_gpu_before_bytes": free, "total_gpu_bytes": total,
            "allocator_fraction_limit": args.gpu_memory_fraction, "no_user_applications_closed": True}
        data = build_dataset(args.seed)
        for row in data["rows"]:
            row["features"] = E.clause_features(row["clause"], row["article"], risk_tags=(), codes=())
        # Hash the exact saved training rows, including features and split IDs.
        canonical = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in data["rows"]) + "\n"
        (evidence / "pairs.jsonl").write_text(canonical, encoding="utf-8")
        receipt["data"] = {"path": str((evidence / "pairs.jsonl").relative_to(ROOT)),
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(), "text_dataset_sha256": data["sha256"],
            "counts": data["counts"], "group_split": data["group_split"], "limitation": data["limitation"],
            "source_and_template_group_overlap": False, "human_gold": False}
        tokenizer = AutoTokenizer.from_pretrained(str(B._weights_dir()), local_files_only=True)
        lengths = {field: max(len(tokenizer(row[field], truncation=False)["input_ids"]) for row in data["rows"])
                   for field in ("clause", "article")}
        if max(lengths.values()) > args.max_length:
            raise RuntimeError("Training sequence budget would truncate these synthetic examples")
        receipt["data"]["max_token_lengths"] = lengths
        receipt["max_training_tokens_per_tower"] = args.max_length
        mamba_config = {"hidden_size": 64, "num_hidden_layers": 1, "state_size": 8,
            "expand": 2, "conv_kernel": 4, "vocab_size": 1, "bos_token_id": 0, "eos_token_id": 0,
            "pad_token_id": 0, "use_cache": False, "use_mamba_kernels": False}
        model = B.build_model(enable_mamba=True, enable_rule_head=False, dropout=.10, mamba_config=mamba_config)
        if model.mamba is None:
            raise RuntimeError("Explicit Mamba branch was not constructed")
        legacy = B.OUTPUT_DIR / "model.pt"
        receipt["legacy_checkpoint_sha256_before"] = file_hash(legacy)
        legacy_state = torch.load(legacy, map_location="cpu", weights_only=True)
        initial_backbone = {key: value for key, value in legacy_state.items() if key.startswith(("encoder.", "attn.", "norm."))}
        model.load_state_dict(initial_backbone, strict=False)  # deliberate frozen-backbone initialization only
        del initial_backbone, legacy_state
        for module in (model.encoder, model.attn, model.norm, model.mamba.embeddings):
            module.requires_grad_(False)
        receipt["initialization"] = "existing BERT/attention/norm checkpoint frozen; fresh classification head and fresh Mamba/projections/gate; no rule residual"
        receipt["architecture"] = {"encoder_hidden_size": model.encoder.config.hidden_size,
            "mamba": True, "mamba_config": model.mamba.config.to_dict(), "dropout": .10,
            "projection": [model.encoder.config.hidden_size, 64, model.encoder.config.hidden_size],
            "mamba_gate": "sigmoid trainable scalar", "rule_residual": False,
            "frozen": ["encoder", "attn", "norm", "mamba.embeddings (unused with inputs_embeds)"],
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "all_parameters": sum(p.numel() for p in model.parameters())}
        initial_mamba = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith("mamba.")}
        initial_branch = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith("mamba")}
        receipt["mamba_core_sha256_before"] = tensor_hash(initial_mamba)
        receipt["mamba_full_branch_sha256_before"] = tensor_hash(initial_branch)
        model.to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
        steps = math.ceil(len(data["train"]) / args.batch_size) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(args.seed)
        receipt.update(status="training", optimizer="AdamW", learning_rate=args.lr,
                       weight_decay=.01, planned_optimizer_steps=steps, completed_optimizer_steps=0,
                       epochs=args.epochs, batch_size=args.batch_size, history=[], gradient_evidence={})
        save()
        print(f"Actual GPU training begins: {steps} steps, train={len(data['train'])}, grouped synthetic only", flush=True)
        def encode(rows):
            c = tokenizer([row["clause"] for row in rows], padding=True, truncation=False, return_tensors="pt")
            a = tokenizer([row["article"] for row in rows], padding=True, truncation=False, return_tensors="pt")
            return c, a, torch.tensor([B.LABEL2ID[row["label"]] for row in rows], device=device)
        training_started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for module in (model.encoder, model.attn, model.norm):
                module.eval()
            order = torch.randperm(len(data["train"]), generator=generator).tolist()
            losses = []
            for offset in range(0, len(order), args.batch_size):
                batch = [data["train"][i] for i in order[offset:offset + args.batch_size]]
                c, a, labels = encode(batch)
                optimizer.zero_grad(set_to_none=True)
                logits, _, _ = model(c["input_ids"].to(device), c["attention_mask"].to(device),
                                     a["input_ids"].to(device), a["attention_mask"].to(device))
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite actual training loss")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if name.startswith("mamba") and parameter.grad is not None:
                        norm = float(parameter.grad.detach().float().norm())
                        if not math.isfinite(norm):
                            raise RuntimeError("Non-finite Mamba gradient")
                        evidence_row = receipt["gradient_evidence"].setdefault(name, {"max_l2": 0., "nonzero_steps": 0})
                        evidence_row["max_l2"] = max(evidence_row["max_l2"], norm)
                        evidence_row["nonzero_steps"] += int(norm > 0)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                receipt["completed_optimizer_steps"] += 1
                losses.append(float(loss.detach()))
                if receipt["completed_optimizer_steps"] % 18 == 0:
                    print(f"step={receipt['completed_optimizer_steps']}/{steps} loss={losses[-1]:.6f}", flush=True)
                    save()
            receipt["history"].append({"epoch": epoch, "mean_cross_entropy": sum(losses) / len(losses),
                                       "first_batch_loss": losses[0], "last_batch_loss": losses[-1], "batches": len(losses)})
            save()
        receipt["training_seconds"] = time.perf_counter() - training_started
        final_mamba = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith("mamba.")}
        final_branch = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith("mamba")}
        receipt["mamba_core_sha256_after"] = tensor_hash(final_mamba)
        receipt["mamba_full_branch_sha256_after"] = tensor_hash(final_branch)
        receipt["parameter_changes"] = {name: {"l2_delta": float((final_branch[name] - before).float().norm()),
            "max_abs_delta": float((final_branch[name] - before).abs().max()), "numel": before.numel()}
            for name, before in initial_branch.items()}
        ssm_changed = [name for name, item in receipt["parameter_changes"].items()
                       if name.startswith("mamba.") and item["l2_delta"] > 0 and any(part in name for part in ("A_log", "x_proj", "dt_proj", ".D"))]
        if receipt["mamba_core_sha256_after"] == receipt["mamba_core_sha256_before"] or not ssm_changed:
            raise RuntimeError("Mamba selective state-space parameters did not actually change")
        receipt["changed_selective_state_space_parameters"] = ssm_changed
        receipt["weak_grouped_valid"] = B.evaluate(model, tokenizer, data["valid"], device=device, batch_size=args.batch_size)
        receipt["weak_grouped_test"] = B.evaluate(model, tokenizer, data["test"], device=device, batch_size=args.batch_size)
        receipt["metrics_are_independent_human_gold"] = False
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, output / "model.pt")
        tokenizer.save_pretrained(str(output))
        meta = {"architecture": receipt["architecture"], "supervision": receipt["supervision"],
                "training_scope": receipt["scope"], "seed": args.seed, "optimizer_steps": receipt["completed_optimizer_steps"],
                "data_sha256": receipt["data"]["sha256"], "valid": receipt["weak_grouped_valid"],
                "test": receipt["weak_grouped_test"], "max_training_tokens_per_tower": args.max_length,
                "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(), "default_backend_unchanged": True}
        (output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt["checkpoint"] = {"path": str((output / "model.pt").relative_to(ROOT)),
                                 "bytes": (output / "model.pt").stat().st_size, "sha256": file_hash(output / "model.pt")}
        del optimizer, scheduler, model, trainable, initial_mamba, final_mamba, initial_branch, final_branch
        gc.collect()
        torch.cuda.empty_cache()
        os.environ["ZLQL_ENTAILMENT_DEVICE"] = "cuda"  # this isolated training process only
        reloaded, reloaded_tokenizer, loaded_meta = B.load_model(output)
        predictor = B.Predictor(reloaded, reloaded_tokenizer, device=device, meta=loaded_meta)
        receipt["strict_reload"] = loaded_meta["loaded_architecture"]
        receipt["reload_predictions"] = []
        for label in B.LABELS:
            row = next(row for row in data["test"] if row["label"] == label)
            receipt["reload_predictions"].append({"pair_id": row["pair_id"], "synthetic_expected": label,
                "clause": row["clause"], "article": row["article"], "prediction": predictor.predict(row["clause"], row["article"])})
        receipt["legacy_checkpoint_sha256_after"] = file_hash(legacy)
        if receipt["legacy_checkpoint_sha256_before"] != receipt["legacy_checkpoint_sha256_after"]:
            raise RuntimeError("Legacy checkpoint unexpectedly changed")
        receipt.update(status="trained_strict_reloaded_predictions_completed", total_seconds=time.perf_counter() - started,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        del predictor, reloaded, reloaded_tokenizer
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc), total_seconds=time.perf_counter() - started)
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        receipt["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_file),
                          "steps": receipt.get("completed_optimizer_steps"), "history": receipt.get("history"),
                          "error": receipt.get("error")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
