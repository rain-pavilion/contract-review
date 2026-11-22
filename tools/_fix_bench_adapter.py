"""把 bench 的适配器对准 evaluate_retrieval 的真实结构（recall@k 是字典）。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/bench.py")
text = path.read_text(encoding="utf-8")

old = '''            if not isinstance(row, dict) or "recall@1" not in row:
                continue
            recall = {
                f"@{k}": float(row.get(f"recall@{k}", 0.0)) for k in range(1, top_k + 1)
            }'''
new = '''            if not isinstance(row, dict) or "recall@k" not in row:
                continue
            raw_recall = row.get("recall@k") or {}
            recall = {
                f"@{k}": float(raw_recall.get(f"@{k}", 0.0)) for k in range(1, top_k + 1)
            }'''
if old in text:
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("适配器已修")
else:
    print("已是新写法，跳过")
