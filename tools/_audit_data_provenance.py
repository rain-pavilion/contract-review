"""样本数据真实性复核：法条库 / 真实政采合同 / 合成对抗样本 三类各给可核验证据。

不读任何缓存结论，全部现场重算：
1) 法条库：条数、法律名分布、条号连续性、字段完整率、抽样条文原文；
2) 真实合同：manifest 统计 + 现场 HTTP 复验（detail/pdf 状态码与字节数）+ PDF 实际页数
   与 meta 记录是否一致；
3) 合成样本：对抗集 manifest 的 issue_codes 是否真的出现在 docx 正文里（逐码给命中证据）。
"""

from __future__ import annotations

import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import parsers, retrieve  # noqa: E402
from zhilv_qiling_v2.config import V1_GENERATED, V1_GOV, V1_STATUTES, WORKSPACE  # noqa: E402

try:
    import pymupdf
except Exception:  # noqa: BLE001
    pymupdf = None


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1. 法条库
head("1. 法条库 data/法条结构化数据.jsonl")

statutes = retrieve.load_statutes(V1_STATUTES)
laws = {}
tagged = 0
typed = 0
urls = set()
for item in statutes:  # load_statutes 返回的是 dict 列表，不是 Statute 对象
    laws[item["law_name"]] = laws.get(item["law_name"], 0) + 1
    if item.get("risk_tags"):
        tagged += 1
    if item.get("contract_types"):
        typed += 1
    if item.get("source_url"):
        urls.add(item["source_url"].split("//")[-1].split("/")[0])

print(f"文件          {V1_STATUTES}  ({V1_STATUTES.stat().st_size/1024:.0f} KB)")
print(f"条数          {len(statutes)}")
print(f"法律名分布    {laws}")
print(f"带 risk_tags  {tagged} 条；带 contract_types {typed} 条")
print(f"来源域名      {sorted(urls)}")
first, last = statutes[0], statutes[-1]
print(f"首条 / 末条   {first['article_no']} / {last['article_no']}")
sample = next(s for s in statutes if s["article_no"] == "第七百零五条")
print(f"抽样 {sample['article_no']}（{sample['article_uid']}）：{sample['content'][:70]}…")
print(f"  标签 {sample.get('risk_tags')}  适用合同 {str(sample.get('contract_types'))[:40]}  效力 {sample.get('authority_level')}")
print(f"  来源 {sample.get('source_url')}")

# ------------------------------------------------------------ 2. 真实合同
head("2. 真实政采合同 data/gov_contracts")

rows = [json.loads(line) for line in (V1_GOV / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
pages = sum(r.get("page_count", 0) for r in rows)
scanned = sum(r.get("scanned_pages", 0) for r in rows)
print(f"份数          {len(rows)}   页数合计 {pages}   扫描页 {scanned} ({scanned/pages*100:.1f}%)")
print(f"来源分布      {sorted({r.get('source') for r in rows})}")
print(f"抓到的时间    {min(r.get('crawled_at','') for r in rows)[:10]} ~ {max(r.get('crawled_at','') for r in rows)[:10]}")
print(f"有金额/签订日 {sum(1 for r in rows if r.get('amount'))} / {sum(1 for r in rows if r.get('sign_date'))}")

print("\n现场复验（detail + pdf，仅取 3 份，超时 25s）：")
ok = fail = 0
page_match = []
for row in rows[:3]:
    for label, url in (("detail", row.get("detail_url")), ("pdf", row.get("pdf_url"))):
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            print(f"  [{resp.status_code}] {label:6s} {len(resp.content):>9d} B  {url[:88]}")
            ok += resp.status_code == 200
            fail += resp.status_code != 200
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR] {label:6s} {type(exc).__name__}: {str(exc)[:50]}  {url[:70]}")
            fail += 1
print(f"  小计 成功 {ok} / 失败 {fail}")

if pymupdf is not None:
    print("\nPDF 实际页数 vs manifest 记录（抽 2 份）：")
    for row in rows[:2]:
        url = row.get("pdf_url")
        if not url:
            continue
        try:
            blob = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"}).content
            doc = pymupdf.open(stream=blob, filetype="pdf")
            real, rec = doc.page_count, row.get("page_count")
            text_pages = sum(1 for p in doc if len(p.get_text().strip()) > 60)
            print(f"  {row['contract_id']}  实际 {real} 页 / 记录 {rec} 页 -> {'一致' if real == rec else '不一致'}；有文字层 {text_pages} 页")
            if real == rec:
                page_match.append(True)
            doc.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  {row.get('contract_id')} 下载/解析失败：{type(exc).__name__}: {str(exc)[:60]}")

# ------------------------------------------------------------ 3. 合成样本
head("3. 合成对抗样本（注入缺陷是否真的写在正文里）")

adv_dir = V1_GENERATED / "samr_multi_template_2025" / "adversarial"
manifest = json.loads((adv_dir / "manifest.json").read_text(encoding="utf-8"))
records = manifest.get("records", [])
print(f"对抗样本 {len(records)} 份；字段 {sorted(records[0].keys())}")

PROBES = {
    "zero_rent": ["0元", "0 元", "人民币0"],
    "date_reversed": ["2026年12月1日起至2026年11月30日止", "12月1日", "11月30日"],
    "negative_fee": ["-", "负"],
    "cash_only": ["现金"],
    "liability_exemption": ["不承担"],
}
for row in records[:3]:
    docx = adv_dir / row["docx"]
    parsed = parsers.parse_path(docx)
    text = parsed.blocks and "\n".join(b.text for b in parsed.blocks) or ""
    codes = row.get("issue_codes", [])
    print(f"\n{row['contract_id']}  {docx.name}  文本 {len(text)} 字  金标 {codes}")
    for code in codes:
        probes = PROBES.get(code, [])
        if not probes:
            print(f"  - {code:26s} （无探针，跳过）")
            continue
        hit = [p for p in probes if p in text]
        print(f"  - {code:26s} {'命中 ' + str(hit) if hit else '未在正文中找到探针词'}")

# ------------------------------------------------------ 4. 审查结果库
head("4. 本轮审查结果库 data/reviews.sqlite")
from zhilv_qiling_v2 import store as store_mod  # noqa: E402

db_path = WORKSPACE / "data" / "reviews.sqlite"
if db_path.exists():
    with store_mod.ReviewStore(db_path) as db:
        stats = db.stats()
        print(f"合同 {stats.contracts} / 风险 {stats.risks} / 引用 {stats.citations}")
        print(f"评分均值 {stats.score_mean}  区间 {stats.score_min}~{stats.score_max}  平均 {stats.elapsed_mean_ms} ms")
else:
    print("（不存在）")
