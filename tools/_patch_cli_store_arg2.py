"""补齐 batch 的 `--store`（上一版循环里多了个提前 break，只插进 review 一处）。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/cli.py")
text = path.read_text(encoding="utf-8")

pairs = [
    (
        '    review.add_argument("--no-dp", action="store_true", help="改用 v1 式贪心切分（消融）")',
        '    review.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")',
    ),
    (
        '    batch.add_argument("--csv", default=None, help="把逐份结果写成 CSV")',
        '    batch.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")',
    ),
    (
        '    explain.add_argument("--no-dp", action="store_true")',
        '    explain.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")',
    ),
]

for anchor, flag in pairs:
    if anchor not in text:
        print("跳过（锚点缺失）:", anchor[:36])
        continue
    if flag in text.split(anchor, 1)[1][:200]:
        print("已存在:", flag[:30])
        continue
    text = text.replace(anchor, anchor + "\n" + flag, 1)
    print("已插入:", flag[:30])

path.write_text(text, encoding="utf-8")
print("--store 总次数:", text.count('"--store"'))
