"""把 AMOUNT 的误报逐个打出来（文本 + 上下文 + 货币单位 + 线索词），据此决定收紧规则。

注意两个 API 形状（踩过）：
- `_gold_spans(doc, full)` 返回 `[(start, end, kind)]`，不是带属性的对象；
- `predict(...)` 返回 `[(start, end, kind, score)]`，同样是元组。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import bert_ner as B, train

CUES = ("单价", "税率", "费率", "合计", "总计", "小计", "总价", "合同价", "价款", "金额",
        "押金", "保证金", "违约金", "上限", "不低于", "不超过", "预算", "最高")


def main() -> None:
    docs = train.collect_docs()
    _, valid = train.split_docs(docs, ratio=0.25)
    model, tokenizer, _ = B.load_model()[:3]
    total = with_unit = 0
    buckets: dict[str, int] = {}
    for doc in valid:
        text = train.doc_text(doc)
        gold = {(s, e) for s, e, k in B._gold_spans(doc, text) if k == "AMOUNT"}
        if not gold:
            continue
        for start, end, kind, _score in B.predict(model, tokenizer, text):
            if kind != "AMOUNT" or (start, end) in gold:
                continue
            piece = text[start:end]
            total += 1
            unit = any(u in piece for u in ("元", "圆", "人民币"))
            with_unit += 1 if unit else 0
            ctx = text[max(0, start - 14): min(len(text), end + 8)].replace("\n", " ")
            cue = next((c for c in CUES if c in ctx), "") or "（无线索词）"
            buckets[cue] = buckets.get(cue, 0) + 1
            if total <= 26:
                print(f"  fp#{total:02d} [{piece}] unit={int(unit)} cue={cue} | {ctx}")
    print(f"\nAMOUNT fp 合计 {total} ｜ 带货币单位 {with_unit} ｜ 无线索词 {total - with_unit}")
    print("线索词分布:", json.dumps(buckets, ensure_ascii=False))


if __name__ == "__main__":
    main()
