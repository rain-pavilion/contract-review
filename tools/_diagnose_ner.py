"""节点① 诊断：当前 BERT-CRF 在留出集上错在哪。

只读诊断，不改模型。输出：
1) 逐类 P/R/F1（与 bert_ner.evaluate 同口径）；
2) 每类前若干个 FP / FN 样例（带上下文），用来判断是"标错了"还是"模型弱"；
3) 金标与会话银标的覆盖差异（为什么 SUBJECT 全 0）。
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import bert_ner, train  # noqa: E402


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


docs = train.collect_docs()
train_docs, valid_docs = train.split_docs(docs, ratio=0.25)
golden = [d for d in valid_docs if getattr(d, "gold_entities", None)]
print(f"全部 {len(docs)} 份 / 训练 {len(train_docs)} / 留出 {len(valid_docs)}（其中带金标 {len(golden)}）")

head("1. 留出集逐类指标")
model, tokenizer = bert_ner.load_model()[:2]
metrics = bert_ner.evaluate(model, tokenizer, valid_docs)
for kind, row in metrics["per_kind"].items():
    precision, recall, f1 = row  # per_kind 的值是 [P, R, F1]，不含 tp/fp/fn
    print(f"  {kind:8s} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
print(f"  macro-F1 = {metrics['macro_f1']:.4f}  文档数 {metrics['docs']}")

head("2. 错误样例（每类最多 6 个，看是标错还是模型弱）")
KINDS = ("PARTY", "AMOUNT", "DATE", "SUBJECT")
samples: dict[str, dict[str, list[str]]] = {k: {"fp": [], "fn": []} for k in KINDS}
for doc in valid_docs[:40]:
    try:
        full = bert_ner.doc_text_of(doc)
    except Exception:
        continue
    if len(full) < 40:
        continue
    gold = set(bert_ner._spans_of(doc, full))
    pred = {(s, e, k) for s, e, k, _ in bert_ner.predict(model, tokenizer, full)}
    for kind in KINDS:
        g = {i for i in gold if i[2] == kind}
        p = {i for i in pred if i[2] == kind}
        for s, e, _k in list(p - g)[:3]:
            ctx = full[max(0, s - 12):e + 12].replace("\n", " ")
            samples[kind]["fp"].append(f"「{full[s:e]}」 ctx: …{ctx}…")
        for s, e, _k in list(g - p)[:3]:
            ctx = full[max(0, s - 12):e + 12].replace("\n", " ")
            samples[kind]["fn"].append(f"「{full[s:e]}」 ctx: …{ctx}…")

for kind in KINDS:
    for tag in ("fp", "fn"):
        items = samples[kind][tag][:6]
        label = "误报 FP" if tag == "fp" else "漏报 FN"
        print(f"\n[{kind} {label}] 共收集 {len(samples[kind][tag])}")
        for item in items:
            print("   -", item)

head("3. 金标 vs 银标覆盖（解释 SUBJECT 全 0）")
for kind in KINDS:
    g = s = 0
    for doc in docs:
        try:
            full = bert_ner.doc_text_of(doc)
        except Exception:
            continue
        if len(full) < 40:
            continue
        g += sum(1 for i in bert_ner._spans_of(doc, full) if i[2] == kind)
        s += sum(1 for i in train.rule_silver_spans(full) if i[2] == kind)
    print(f"  {kind:8s} 金标 {g:4d} 个 / 银标 {s:4d} 个")
