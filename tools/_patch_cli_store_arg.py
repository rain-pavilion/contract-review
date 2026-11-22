"""补上 `--store` 参数注册（上一步只加进了执行体，参数没加上）。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/cli.py")
text = path.read_text(encoding="utf-8")

pairs = [
    (
        '    review.add_argument("--no-dp", action="store_true", help="改用 v1 式贪心切分（消融）")',
        '    review.add_argument("--no-dp", action="store_true", help="改用 v1 式贪心切分（消融）")\n'
        '    review.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")',
    ),
    (
        '    batch.add_argument("--csv", default=None, help="把逐份结果写成 CSV")',
        '    batch.add_argument("--csv", default=None, help="把逐份结果写成 CSV")\n'
        '    batch.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")',
    ),
]

for old, new in pairs:
    if old not in text:
        raise SystemExit(f"锚点缺失：{old[:40]}")
    if '"--store"' in text:
        break
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("--store 出现次数:", text.count('"--store"'))
