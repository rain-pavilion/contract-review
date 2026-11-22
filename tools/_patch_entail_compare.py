"""让 entail 训练能"模型 vs 规则"同口径对比。

改动三处：
1. `train()` 返回值带上 `valid_rows` —— 否则外部没法在同一批验证样本上跑规则基线，
   只能拿模型和不同的数字比，那种比较没有意义。
2. 新增 `evaluate_rules(rows)` —— 用 `entailment.judge_pair`（rules backend）在同一批
   样本上算 accuracy / per_label / macro_f1，结构与 `evaluate()` 完全一致。
3. `run_gpu_upgrade.cmd_entail` 打印标签分布 + 两行对照。

中文文件必须用 Python 读写。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "zhilv_qiling_v2" / "bert_entail.py"
RUNNER = ROOT / "zhilv_qiling_v2" / "reports" / "experiments" / "run_gpu_upgrade.py"


def patch_module() -> int:
    text = TARGET.read_text(encoding="utf-8")
    original = text

    old = """    return {"pairs": len(rows), "history": history, "valid": metrics, "device": str(device)}"""
    new = """    return {
        "pairs": len(rows),
        "history": history,
        "valid": metrics,
        "valid_rows": valid_rows,
        "device": str(device),
    }"""
    assert old in text, "train 返回值未命中"
    text = text.replace(old, new, 1)

    anchor = """def load_model(dir_path: pathlib.Path | str | None = None):"""
    addition = '''def evaluate_rules(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """同一批样本上跑 rules 后端（8 维特征判别），口径与 evaluate() 完全一致。

    为什么要这个：训练出来的双塔如果还不如规则，必须当场看得见。
    旧版就是这么埋掉的 —— 模型 macro-F1 0.5771 看着像"有效果"，但它把
    「均不承担赔偿责任」判成 neutral 0.9995，而规则判 contradict 0.9983。
    """
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    for row in rows:
        verdict = E.judge_pair(
            row["clause"],
            {"article_no": "", "content": row["article"], "law_name": "", "risk_tags": []},
            codes=(),
        )
        if verdict.label in confusion.get(row["label"], {}):
            confusion[row["label"]][verdict.label] += 1
    per_label: dict[str, list[float]] = {}
    f1_values: list[float] = []
    total = sum(sum(row.values()) for row in confusion.values())
    hits = sum(confusion[label][label] for label in LABELS)
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = [round(precision, 4), round(recall, 4), round(f1, 4), tp + fn]
        f1_values.append(f1)
    return {
        "rows": total,
        "accuracy": round(hits / total, 4) if total else 0.0,
        "per_label": per_label,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "confusion": confusion,
    }


def load_model(dir_path: pathlib.Path | str | None = None):'''
    assert anchor in text, "load_model 锚点未命中"
    text = text.replace(anchor, addition, 1)

    if text == original:
        print("模块没有改动")
        return 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"模块已改写（{len(original.splitlines())} → {len(text.splitlines())} 行）")
    return 0


def patch_runner() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    original = text

    old = """    pairs = bert_entail.build_pairs(docs, limit_clauses=args.pairs)
    print(f"[ENTAIL] 三元组 {len(pairs)} 条，开始训练双塔…", flush=True)
    result = bert_entail.train(pairs, epochs=args.epochs, batch_size=args.batch)"""
    new = """    pairs = bert_entail.build_pairs(
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
        result.pop("valid_rows", None)"""
    assert old in text, "runner 训练段未命中"
    text = text.replace(old, new, 1)

    old = """    parser.add_argument("--pairs", type=int, default=300)"""
    new = """    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--per-clause", type=int, default=3, dest="per_clause")"""
    assert old in text, "runner 参数段未命中"
    text = text.replace(old, new, 1)

    if text == original:
        print("runner 没有改动")
        return 1
    RUNNER.write_text(text, encoding="utf-8")
    print(f"runner 已改写（{len(original.splitlines())} → {len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    code = patch_module()
    code |= patch_runner()
    sys.exit(code)
