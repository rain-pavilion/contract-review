"""查看 cli.py 里 review/batch 的参数注册与 store 代码是否插入成功。"""

import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
src = pathlib.Path("zhilv_qiling_v2/cli.py").read_text(encoding="utf-8")

for name in ("review = sub.add_parser", "batch = sub.add_parser"):
    start = src.find(name)
    print("=" * 20, name.split(" =")[0])
    print(src[start : start + 800])

print("review 已入库:", "已入库" in src)
print("batch 用 batch_to_store:", "batch_to_store" in src)
print("--store 出现次数:", src.count('"--store"'))
