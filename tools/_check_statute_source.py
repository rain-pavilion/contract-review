"""法条库来源取证：这份 jsonl 到底哪来的、文本能不能对上官方原文。

四步：
1) 看 v1 里两个相关脚本的头注释（merge_statute_tags / build_legal_index），判断"生成"还是"合并"；
2) 库内自检：条号连续性、content 长度分布、是否有完全重复条文；
3) 与项目内原始汇编 txt 逐条比对（同一文本的两个来源）；
4) 现场向官方源取原文比对（flk.npc.gov.cn / www.gov.cn），限时，失败如实记录。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUTES = ROOT / "data" / "法条结构化数据.jsonl"
COMPILED = ROOT / "data" / "法律法规条文汇编（房屋买卖租赁委托中介保管）.txt"


def head(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ------------------------------------------------------------- 1. 生成脚本
head("1. v1 里两个相关脚本的自我说明")
for name in ("merge_statute_tags.py", "build_legal_index.py"):
    path = ROOT / "scripts" / name
    if not path.exists():
        print(f"[{name}] 不存在")
        continue
    lines = path.read_text(encoding="utf-8").splitlines()[:16]
    print(f"\n--- scripts/{name} ---")
    for line in lines:
        print("   " + line)

# ------------------------------------------------------------- 2. 库内自检
head("2. 库内自检")
rows = [json.loads(line) for line in STATUTES.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"条数 {len(rows)}；条号唯一 {len({r['article_no'] for r in rows})}；uid 唯一 {len({r['article_uid'] for r in rows})}")
lengths = sorted(len(r["content"]) for r in rows)
print(f"条文长度 最短 {lengths[0]} / 中位 {lengths[len(lengths)//2]} / 最长 {lengths[-1]} / 均值 {sum(lengths)/len(lengths):.1f}")
dupes = len(rows) - len({r["content"] for r in rows})
print(f"完全重复条文 {dupes} 条")
no_tag = [r for r in rows if not r.get("risk_tags")]
print(f"无标签 {len(no_tag)} 条（占 {len(no_tag)/len(rows)*100:.1f}%）")
print(f"version_id 取值 {sorted({r.get('version_id','') for r in rows})[:3]}")
print(f"effective_from 取值 {sorted({r.get('effective_from','') for r in rows})[:3]}")

CHECKS = ("第七百零五条", "第五百七十七条", "第五百零九条")
print("\n抽样三条（完整正文）：")
by_no = {r["article_no"]: r for r in rows}
for no in CHECKS:
    row = by_no.get(no)
    if row:
        print(f"\n【{no}】标的标签 {row.get('risk_tags')}｜适用 {row.get('contract_types')}")
        print("   " + row["content"].replace("\n", " "))
    else:
        print(f"【{no}】库内缺失")

# --------------------------------------------------------- 3. 与本地汇编比对
head("3. 与项目内原始汇编 txt 比对（同一文本的另一来源）")
if COMPILED.exists():
    raw = COMPILED.read_text(encoding="utf-8", errors="replace")
    print(f"汇编文件 {COMPILED.name}：{len(raw)} 字符")
    for no in CHECKS:
        row = by_no.get(no)
        if not row:
            continue
        needle = row["content"].split("。")[0][:30]
        hit = needle in raw
        print(f"  {no}：库内首句「{needle}…」→ 汇编中{'找到' if hit else '未找到'}")
else:
    print("汇编文件不存在")

# --------------------------------------------------------- 4. 官方源现场核对
head("4. 官方源现场取原文（限时 15s/请求，失败如实记录）")
TARGETS = [
    ("国家法律法规数据库-民法典", "https://flk.npc.gov.cn/api/detail?id=ZmY4MDgwODE3OWE4YjI5YjAxN2E1YjE5YjM3ZDBiMzM%3D"),
    ("中国政府网-民法典全文", "https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm"),
]
for label, url in TARGETS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            print(f"  [{resp.status}] {label}  {len(body)} B")
            text = body.decode("utf-8", errors="replace")
            for no in CHECKS:
                snippet = by_no[no]["content"].split("。")[0][:16] if no in by_no else ""
                print(f"      含「{no}」原文首句：{snippet in text}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [ERR] {label}  {type(exc).__name__}: {str(exc)[:70]}")
