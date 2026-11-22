"""按行号补 `normalize` / `fusion_weights` / `rrf_k`（上一版字符串锚点没对上缩进）。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/train.py")
lines = path.read_text(encoding="utf-8").splitlines()

start = next(i for i, line in enumerate(lines) if line.startswith("def train_rerank"))
target = None
for index in range(start, len(lines)):
    if '"metrics": metrics,' in lines[index]:
        target = index
        break
if target is None:
    raise SystemExit("train_rerank 里没找到 metrics 行")

indent = " " * (len(lines[target]) - len(lines[target].lstrip()))
extra = [
    f'{indent}"normalize": True,',
    f'{indent}"fusion_weights": fusion["weights"],',
    f'{indent}"rrf_k": fusion["rrf_k"],',
    f'{indent}"l2": 0.02,',
]
if any('"normalize": True' in line for line in lines[start:target + 20]):
    print("已经有 normalize，跳过")
else:
    lines[target + 1 : target + 1] = extra
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("已插入 4 行 metadata（第", target + 2, "行起）")
