"""让 `rerank_ltr.json` 带上 `normalize` 与融合权重，推理才能复现 0.6944。

根因：`train.train_rerank` 手写 JSON 时只写了 names/weights/bias/metrics，
把 `normalize`（列标准化开关）、`fusion_weights`、`rrf_k` 丢了；而
`RetrievalRanker.normalized` 读的正是 `meta["normalize"]` —— 一旦读不到就回落
逐条 `score()`（14/15 维量纲差两个数量级），+LTR 的强金标准 Recall@5 从 0.6944
掉到 0.2222。`bench` 复算时暴露的正是这个。
"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/train.py")
text = path.read_text(encoding="utf-8")

old = '''    weights": weights,
        "bias": 0.0,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": len(pairs),
            "queries": len(evaluation),
            "metrics": metrics,
        },
    }'''
new = '''    weights": weights,
        "bias": 0.0,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": len(pairs),
            "queries": len(evaluation),
            "metrics": metrics,
            # 下面三项必须在场：推理端靠 normalize 走"候选集内列标准化"，
            # 靠 fusion_weights / rrf_k 复现训练时的候选池。缺一项，重排结果就漂。
            "normalize": True,
            "fusion_weights": fusion["weights"],
            "rrf_k": fusion["rrf_k"],
            "l2": 0.02,
        },
    }'''
if old not in text:
    raise SystemExit("锚点不匹配，先看 train.py 的 payload 写法")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("train.py 已写入 normalize / fusion_weights / rrf_k")
