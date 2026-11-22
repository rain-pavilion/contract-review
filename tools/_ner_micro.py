"""用统一口径（纯金标评测）重算三条通道的微平均/宏平均，含 PARTY 宽松口径。

跑法：
    python tools\\_ner_micro.py [--limit N]

做三件事：
1. 规则通道：直接拿 ner.rule_elements 的输出当预测，对金标算 P/R/F1；
2. BERT-CRF（现存权重）：bert_ner.evaluate（已改成纯金标 + 微平均）；
3. 逐类明细 + 混淆来源（哪一类漏、哪一类多）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import bert_ner, ner, train  # noqa: E402


def rule_predict(text: str) -> list[tuple[int, int, str]]:
    """规则通道的预测结果，转成 (start, end, kind) 三元组。

    直接复用 train.rule_silver_spans —— 它就是规则通道的产物（已做首尾 trim 与重叠去重），
    与训练银标同源，保证"规则"这条对照线可复现。注意它不产 SUBJECT（规则通道没有标的识别器）。
    """
    return list(train.rule_silver_spans(text))


def score(gold: set[tuple[int, int, str]], pred: set[tuple[int, int, str]]) -> dict[str, dict[str, float]]:
    kinds = ("PARTY", "AMOUNT", "DATE", "SUBJECT")
    stats = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in kinds}
    for kind in kinds:
        g = {item for item in gold if item[2] == kind}
        p = {item for item in pred if item[2] == kind}
        stats[kind]["tp"] = len(g & p)
        stats[kind]["fp"] = len(p - g)
        stats[kind]["fn"] = len(g - p)
    out: dict[str, dict[str, float]] = {}
    for kind, row in stats.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[kind] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), **row}
    out["micro"] = bert_ner.micro_stats(stats)
    f1s = [out[kind]["f1"] for kind in kinds]
    out["macro"] = {"f1": round(sum(f1s) / len(f1s), 4)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    docs = train.collect_docs()
    _, valid = train.split_docs(docs, 0.25, seed=20261104)

    rule_stats = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ("PARTY", "AMOUNT", "DATE", "SUBJECT")}
    counted = 0
    for doc in valid:
        if args.limit and counted >= args.limit:
            break
        if not doc.gold_entities:
            continue
        text = train.doc_text(doc)
        if len(text) < 40:
            continue
        gold = set(bert_ner._gold_spans(doc, text))
        if not gold:
            continue
        pred = set(rule_predict(text))
        counted += 1
        for kind in rule_stats:
            g = {item for item in gold if item[2] == kind}
            p = {item for item in pred if item[2] == kind}
            rule_stats[kind]["tp"] += len(g & p)
            rule_stats[kind]["fp"] += len(p - g)
            rule_stats[kind]["fn"] += len(g - p)

    per_kind = {}
    for kind, row in rule_stats.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per_kind[kind] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), **row}
    macro = sum(per_kind[k]["f1"] for k in per_kind) / len(per_kind)
    rule_report = {
        "docs": counted,
        "per_kind": per_kind,
        "micro": bert_ner.micro_stats(rule_stats),
        "macro": {"f1": round(macro, 4)},
    }

    print("=== 规则通道（纯金标评测）===")
    print(json.dumps(rule_report, ensure_ascii=False, indent=2))

    model_path = bert_ner.OUTPUT_DIR / "model.pt"
    if model_path.exists():
        model, tokenizer, _meta = bert_ner.load_model()
        bert_report = bert_ner.evaluate(model, tokenizer, valid, limit=args.limit)
        print("=== BERT-CRF（纯金标评测）===")
        print(json.dumps(bert_report, ensure_ascii=False, indent=2))
    else:
        print("[skip] 没有 bert_ner 权重")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
