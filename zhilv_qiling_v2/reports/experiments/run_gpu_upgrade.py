"""GPU 升级三件事的一键入口：BERT-CRF NER、BGE-M3 向量、双塔判别。

用法
----
    python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade ner
    python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade embed
    python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade entail
    python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade all

每个子命令都把"改前 / 改后"打在一起，方便直接抄进论文：
- `ner`   ：感知机 0.0179 / 规则 0.4204（读 `models/ner_perceptron.json`）vs BERT-CRF 实测
- `embed` ：LSA 密集通道 vs BGE-M3 密集通道的检索指标（读/写 `models/bge_m3/`）
- `entail`：8 维弱监督判别 vs 双塔 cross-attention 的 macro-F1
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ...config import MODEL_DIR, config_from_env, ensure_dirs
from ... import learned as L

PKG_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def report_ner_old() -> dict:
    """读旧感知机与规则通道的留出集指标，作为对照基线。"""
    payload = _load_json(MODEL_DIR / L.ModelRegistry.FILES["ner"])
    meta = payload.get("meta", {})
    valid = meta.get("valid", {})
    return {
        "model_macro_f1": valid.get("model_macro_f1", valid.get("macro_f1")),
        "rule_macro_f1": valid.get("rule_macro_f1"),
        "model_per_kind": valid.get("model_per_kind") or valid.get("per_kind"),
        "rule_per_kind": valid.get("rule_per_kind"),
        "windows": meta.get("windows"),
    }


def cmd_ner(args) -> int:
    from ... import bert_ner
    from ...train import collect_docs

    print("[NER] 载入语料…", flush=True)
    docs = collect_docs()
    print(f"[NER] 样本 {len(docs)} 份（有金标 {sum(1 for d in docs if d.gold_entities)} 份）", flush=True)
    print("[NER] 开始训练 BERT-CRF（GPU）…", flush=True)
    started = time.time()
    result = bert_ner.train(docs, config=None, epochs=args.epochs, batch_size=args.batch,
                            oversample_gold=args.oversample)
    elapsed = time.time() - started
    old = report_ner_old()

    print("\n=== NER 对照（留出集，span 级宏平均 F1）===")
    print(f"{'通道':<22}{'宏平均 F1':>12}")
    print(f"{'字符级感知机（旧）':<22}{str(old.get('model_macro_f1')):>12}")
    print(f"{'规则通道':<22}{str(old.get('rule_macro_f1')):>12}")
    print(f"{'BERT-CRF（新）':<22}{result['valid']['macro_f1']:>12}")
    print(f"\n逐类（新）：{result['valid']['per_kind']}")
    print(f"旧模型逐类：{old.get('model_per_kind')}")
    print(f"规则逐类：{old.get('rule_per_kind')}")
    print(f"训练窗口 {result['windows']} / epoch {result['epochs']} / 设备 {result['device']} / 耗时 {elapsed:.1f}s")
    _dump("bert_ner", {"new": result, "old": old, "elapsed_seconds": round(elapsed, 1)})
    return 0


def cmd_embed(args) -> int:
    from ... import bert_embed, learned
    from ...config import MODEL_DIR, V1_STATUTES, config_from_env
    from ...retrieve import Retriever, StatuteIndex, load_statutes
    from ...train import _load_issue_specs, evaluate_retrieval

    config = config_from_env()
    statutes = load_statutes(V1_STATUTES)
    print(f"[EMBED] 法条 {len(statutes)} 条，开始 BGE-M3 编码…", flush=True)
    started = time.time()
    meta = bert_embed.build_statute_vectors(statutes)
    elapsed = time.time() - started
    print(f"[EMBED] 完成：{meta['count']} 条 × {meta['dim']} 维，设备 {meta['device']}，耗时 {elapsed:.1f}s")

    print("\n=== 检索通道对照（v1 协议，强金标准）===", flush=True)
    specs, _template_map = _load_issue_specs()
    ranker = learned.RetrievalRanker.load(MODEL_DIR / "rerank_ltr.json")
    index = StatuteIndex(statutes, config)
    had_bge = bool(getattr(index, "bge_ready", False))
    print(f"[EMBED] 索引 bge_ready={had_bge}")
    retriever = Retriever(index, config, ranker=ranker)
    results: dict[str, Any] = {}
    for label, flag in (("lsa", False), ("bge", True)):
        index.bge_ready = flag
        metrics = evaluate_retrieval(retriever, index, statutes, specs, ranker, config)
        results[label] = metrics
        for name in ("rrf_only", "rrf_plus_ltr"):
            strong = metrics.get(name, {}).get("strong", {})
            print(f"  {label:<4} {name:<12} Recall@5={strong.get('recall@5')} MRR@5={strong.get('mrr@5')} n={strong.get('n')}")
    index.bge_ready = had_bge
    _dump("bge_m3", {"meta": meta, "lsa": results["lsa"], "bge": results["bge"],
                    "elapsed_seconds": round(elapsed, 1)})
    return 0


def cmd_entail(args) -> int:
    from ... import bert_entail
    from ...train import collect_docs

    print("[ENTAIL] 构造条款×条文三元组…", flush=True)
    docs = collect_docs()
    started = time.time()
    pairs = bert_entail.build_pairs(
        docs, per_clause=getattr(args, "per_clause", 3), limit_clauses=args.pairs
    )
    counts = {}
    for row in pairs:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"[ENTAIL] 三元组 {len(pairs)} 条 / 标签分布 {counts}，开始训练双塔…", flush=True)
    result = bert_entail.train(pairs, epochs=args.epochs, batch_size=args.batch)
    valid_rows = result.get("valid_rows") or []
    if valid_rows:
        rules = bert_entail.evaluate_rules(valid_rows)
        print(
            f"[ENTAIL] 同一验证集 n={rules['rows']}："
            f"规则 macro-F1 {rules['macro_f1']:.4f} / acc {rules['accuracy']:.4f}  "
            f"| 双塔 macro-F1 {result['valid']['macro_f1']:.4f} / acc {result['valid']['accuracy']:.4f}",
            flush=True,
        )
        result["rules_baseline"] = rules
        result.pop("valid_rows", None)
    elapsed = time.time() - started
    print("\n=== 判别器对照（macro-F1）===")
    print(f"8 维弱监督（旧）：见 models/entailment.json 的 meta.valid（若存在）")
    print(f"双塔 cross-attention（新）：{result['valid']['macro_f1']}  准确率 {result['valid']['accuracy']}")
    print(f"逐类：{result['valid']['per_label']}")
    print(f"三元组 {result['pairs']} / 设备 {result['device']} / 耗时 {elapsed:.1f}s")
    _dump("bert_entail", {"new": result, "elapsed_seconds": round(elapsed, 1)})
    return 0


def _dump(name: str, payload: dict) -> None:
    target = PKG_ROOT / "reports" / f"gpu_upgrade_{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写 {target}")


def main(argv=None) -> int:
    import argparse

    ensure_dirs()
    parser = argparse.ArgumentParser(prog="run_gpu_upgrade", description="GPU 升级三件事")
    parser.add_argument("task", choices=["ner", "embed", "entail", "all"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=3, help="金标文档过采样倍数")
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--per-clause", type=int, default=3, dest="per_clause")
    args = parser.parse_args(argv)

    tasks = ["ner", "embed", "entail"] if args.task == "all" else [args.task]
    for task in tasks:
        print(f"\n{'=' * 62}\n== {task}\n{'=' * 62}")
        code = {"ner": cmd_ner, "embed": cmd_embed, "entail": cmd_entail}[task](args)
        if code:
            return code
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
