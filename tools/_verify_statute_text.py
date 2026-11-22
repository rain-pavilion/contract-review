"""法条文本真实性再验：去掉【小标题】前缀后做归一化比对。

三个比对面：
1) 库内 1260 条 与 项目内汇编 txt —— 统计能在汇编里找到多少条（同源证明）；
2) 库内 1260 条 与 官方在线全文 —— 逐个候选 URL 试，命中多少条；
3) 抽样打印，人工可读即可判断是不是真条文。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUTES = ROOT / "data" / "法条结构化数据.jsonl"
COMPILED = ROOT / "data" / "法律法规条文汇编（房屋买卖租赁委托中介保管）.txt"

HEAD_RE = re.compile(r"^【[^】]{1,12}】")


def strip_title(text: str) -> str:
    """去掉条文前的【小标题】与所有空白，只留纯文本用于比对。"""
    return re.sub(r"\s+", "", HEAD_RE.sub("", text.strip()))


rows = [json.loads(line) for line in STATUTES.read_text(encoding="utf-8").splitlines() if line.strip()]
by_no = {r["article_no"]: r for r in rows}
pure = {r["article_no"]: strip_title(r["content"]) for r in rows}

print("=" * 76)
print("1. 与项目内汇编 txt 比对（去【】前缀 + 去空白，取前 20 字做子串匹配）")
print("=" * 76)
raw = COMPILED.read_text(encoding="utf-8", errors="replace")
flat = re.sub(r"\s+", "", raw)
print(f"汇编 {COMPILED.name}：{len(raw)} 字符，归一化后 {len(flat)} 字符")

hit = [no for no, text in pure.items() if len(text) >= 20 and text[:20] in flat]
print(f"库内 {len(rows)} 条中，前 20 字能在汇编里找到：**{len(hit)} 条**（{len(hit)/len(rows)*100:.1f}%）")
for no in ("第七百零五条", "第五百七十七条", "第五百零九条"):
    text = pure.get(no, "")
    print(f"  {no}：{'命中' if text[:20] in flat else '未命中'}  「{text[:26]}…」")

print()
print("=" * 76)
print("2. 与官方在线全文比对（逐个候选，限时 12s）")
print("=" * 76)
CANDIDATES = [
    ("中国政府网 民法典全文", "https://www.gov.cn/guoqing/2020-12/24/content_5573005.htm"),
    ("中国人大网 民法典", "http://www.npc.gov.cn/npc/c2/c30834/202006/t20200602_306457.html"),
    ("国家法律法规数据库 搜索页", "https://flk.npc.gov.cn/xf.html"),
]
for label, url in CANDIDATES:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read()
        text = body.decode("utf-8", errors="replace")
        flat_online = re.sub(r"\s+", "", text)
        found = [no for no, t in pure.items() if len(t) >= 20 and t[:20] in flat_online]
        print(f"  [{resp.status}] {label}  {len(body)} B → 命中 {len(found)} 条")
        if found:
            print(f"       例：{found[:5]}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [ERR] {label}  {type(exc).__name__}: {str(exc)[:60]}")

print()
print("=" * 76)
print("3. 抽样三条，人工可直接判断（真条文还是编的）")
print("=" * 76)
for no in ("第一条", "第七百零五条", "第五百七十七条", "第一千二百六十条"):
    row = by_no.get(no)
    if row:
        print(f"\n【{no}】")
        print("  " + row["content"].replace("\n", " "))
