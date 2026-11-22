"""定位 rerank payload 写入块并补 metadata（前两次锚点都没对上）。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/train.py")
lines = path.read_text(encoding="utf-8").splitlines()

anchor = [i for i, line in enumerate(lines) if 'FILES["rerank"]' in line]
if not anchor:
    raise SystemExit("找不到 FILES[rerank] 写入行")
write_line = anchor[0]
print("写入行:", write_line + 1)

start = max(0, write_line - 30)
for index in range(start, write_line + 1):
    print(f"{index + 1:4}|{lines[index]}")

# 在 payload 的 meta 字典里补四个键
target = None
for index in range(start, write_line):
    if '"metrics": metrics,' in lines[index]:
        target = index
        break
if target is None:
    raise SystemExit("payload 的 meta 里没有 metrics 行")

indent = " " * (len(lines[target]) - len(lines[target].lstrip()))
extra = [
    f'{indent}"normalize": True,',
    f'{indent}"fusion_weights": fusion["weights"],',
    f'{indent}"rrf_k": fusion["rrf_k"],',
    f'{indent}"l2": 0.02,',
]
if all(key in lines[target + 1] or key in lines[target + 2] for key in ('"normalize"', '"rrf_k"')):
    print("已存在，跳过")
else:
    lines[target + 1 : target + 1] = extra
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("已插入 metadata")
