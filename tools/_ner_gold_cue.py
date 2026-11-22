"""量化：金标 AMOUNT 有多少带"合同级金额"线索词。

目的——判断能否给 AMOUNT 加一道"合同级线索闸"（只有带 总价/合同价/金额/价款/合计…
的金额才输出）。闸门能不能加，取决于**金标是否也几乎都带线索**：
金标带、误报不带，闸门就是净收益；金标大量不带，闸门就是砍召回。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import bert_ner as B, train

CUES = ("总价", "合同价", "合同金额", "金额", "价款", "合计", "总计", "小写", "人民币", "元整", "大写")


def main() -> None:
    docs = train.collect_docs()
    _, valid = train.split_docs(docs, ratio=0.25)
    hit = miss = 0
    buckets: dict[str, int] = {}
    missed: list[str] = []
    for doc in valid:
        text = train.doc_text(doc)
        for start, end, kind in B._gold_spans(doc, text):
            if kind != "AMOUNT":
                continue
            ctx = text[max(0, start - 20): min(len(text), end + 10)].replace("\n", " ")
            cue = next((c for c in CUES if c in ctx), "")
            if cue:
                hit += 1
                buckets[cue] = buckets.get(cue, 0) + 1
            else:
                miss += 1
                if len(missed) < 12:
                    missed.append(ctx)
    print(f"金标 AMOUNT {hit + miss} 个 ｜ 带线索 {hit} ｜ 无线索 {miss}")
    print("线索分布:", json.dumps(buckets, ensure_ascii=False))
    for item in missed:
        print("   无线索:", item[:90])


if __name__ == "__main__":
    main()
