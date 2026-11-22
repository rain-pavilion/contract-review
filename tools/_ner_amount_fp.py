"""诊断 AMOUNT 的误报来源：把 fp 的字符串与上下文打出来，决定是收紧规则还是补金标。"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import bert_ner as B, train


def main() -> None:
    docs = train.collect_docs()
    _, valid = train.split_docs(docs, ratio=0.25)
    model, tokenizer, _meta = B.load_model()[:3]
    report = B.evaluate(model, tokenizer, valid)
    print("report keys:", list(report.keys()))
    print("micro:", json.dumps(report.get("micro"), ensure_ascii=False))
    print("counts:", json.dumps(report.get("counts"), ensure_ascii=False))
    samples = report.get("samples") or {}
    print("sample kinds:", list(samples.keys())[:6])
    amount = samples.get("AMOUNT") or {}
    fps = amount.get("fp") or []
    print(f"AMOUNT fp = {len(fps)}")
    for item in fps[:20]:
        print("   ", json.dumps(item, ensure_ascii=False)[:170])


if __name__ == "__main__":
    main()
