"""CRF 标签链自检：统计金标里有多少非法转移（I-X 的前一位不是 B-X/I-X）。

负 loss 的唯一成因是"金标路径的分数 > log Z"，也就是金标里存在被
`illegal` 惩罚的转移。这个脚本把成因量化，不猜。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from zhilv_qiling_v2 import bert_ner, train


def main() -> int:
    docs = train.collect_docs()
    samples = bert_ner.build_dataset(docs[:40])
    print(f"[1] 窗口 {len(samples)}")
    tokenizer = AutoTokenizer.from_pretrained(str(bert_ner._weights_dir()))

    bad = 0
    checked = 0
    examples: list[str] = []
    for piece, spans in samples[:40]:
        encoded, labels = bert_ner._encode(tokenizer, piece, spans)
        ids = [value for value in labels]
        for index in range(1, len(ids)):
            prev, cur = ids[index - 1], ids[index]
            if cur < 0 or prev < 0:
                continue
            checked += 1
            label_cur = bert_ner.ID2LABEL[cur]
            label_prev = bert_ner.ID2LABEL[prev]
            if label_cur.startswith("I-"):
                kind = label_cur[2:]
                if label_prev not in (f"B-{kind}", f"I-{kind}"):
                    bad += 1
                    if len(examples) < 3:
                        examples.append(f"{label_prev} -> {label_cur} @ {index} …{piece[max(0, index-8):index+8]!r}")
    print(f"[2] 检查转移 {checked} 个，非法 {bad} 个")
    for row in examples:
        print("   ", row)

    model = bert_ner.build_model(len(bert_ner.LABELS))
    crf = model.crf
    emissions = torch.randn(1, 12, len(bert_ner.LABELS))
    tags = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]])
    mask = torch.ones(1, 12)
    gold = crf._scores(emissions, tags, mask)
    part = crf._partition(emissions, mask)
    print(f"[3] 随机参数下 gold={gold.item():.2f} logZ={part.item():.2f} loss={(part - gold).mean().item():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
