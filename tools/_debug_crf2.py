"""CRF 数值自检：用穷举法核对 `_partition`（log Z）与 `_scores`（gold 分）。

只有在 steps=3、num_tags=3 这种规模才能穷举 9 条路径，用来判定
"负 loss" 到底是标签问题还是分区函数算错。
"""

from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from zhilv_qiling_v2 import bert_ner


def brute_force(crf, emissions, mask):
    batch, steps, tags = emissions.shape
    total = []
    for row in range(batch):
        alpha = None
        for path in itertools.product(range(tags), repeat=steps):
            score = crf.start[path[0]] + emissions[row, 0, path[0]]
            for step in range(1, steps):
                if float(mask[row, step]) == 0.0:
                    continue
                score = score + crf.transitions[path[step], path[step - 1]] + emissions[row, step, path[step]]
            score = score + crf.end[path[-1]]
            alpha = score if alpha is None else torch.logaddexp(alpha, score)
        total.append(alpha)
    return torch.stack(total)


def main() -> int:
    torch.manual_seed(0)
    num_tags, steps, batch = 3, 3, 2
    crf = bert_ner.build_crf(num_tags)
    with torch.no_grad():
        crf.transitions.copy_(torch.randn(num_tags, num_tags))
        crf.start.copy_(torch.randn(num_tags))
        crf.end.copy_(torch.randn(num_tags))

    emissions = torch.randn(batch, steps, num_tags)
    mask = torch.ones(batch, steps)
    mask[1, 2] = 0  # 第二行最后一个位置是 padding

    mine = crf._partition(emissions, mask)
    ref = brute_force(crf, emissions, mask)
    print("[1] _partition :", [round(float(v), 4) for v in mine])
    print("[1] 穷举 logZ  :", [round(float(v), 4) for v in ref])
    print("[1] 最大误差  :", round(float((mine - ref).abs().max()), 6))

    tags = torch.tensor([[0, 1, 2], [0, 0, 0]])
    gold = crf._scores(emissions, tags, mask)
    print("[2] gold 分   :", [round(float(v), 4) for v in gold])
    print("[2] gold ≤ logZ ?", bool((gold <= ref + 1e-4).all()))
    print("[2] nll       :", round(float(crf.nll(emissions, tags, mask)), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
