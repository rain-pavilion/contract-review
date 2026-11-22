"""法条知识库 + 合同比对检索的自证脚本。

回答三个问题，每一条都有可复现输出：
  1. 法条数据库建没建？—— 库的规模、字段、索引结构（可对到 data/法条结构化数据.jsonl 的原文）
  2. 合同能不能比对到条文？—— 拿一份真实合同跑完整链路，逐条风险列出命中条文
  3. 数据真不真？—— 库与语料的来源字段（article_uid / source_url / detail_url）都能回查

运行：
    python zhilv_qiling_v2/reports/experiments/demo_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))

from zhilv_qiling_v2.config import default_config  # noqa: E402
from zhilv_qiling_v2.pipeline import review_bytes, review_path  # noqa: E402
from zhilv_qiling_v2.retrieve import get_retriever, rewrite_query  # noqa: E402

STATUTES = WORKSPACE / "data" / "法条结构化数据.jsonl"
REAL_CONTRACT = WORKSPACE / "data" / "gov_contracts" / "6f840264fb25"
ADVERSARIAL = (WORKSPACE / "data" / "generated_contracts" / "samr_multi_template_2025"
               / "adversarial" / "ADV-GF20250801-0002.docx")

PROBES = [
    ("租赁期限没写上限", "本合同租赁期限由双方另行商定，未约定具体年限。", "房屋租赁"),
    ("押金远超租金", "承租人应向出租人支付押金人民币 600000 元，月租金 5000 元。", "房屋租赁"),
    ("受托人可自行处分", "受托人有权自行决定处分委托人的任何资产，无需另行取得同意。", "委托"),
]


def section(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def main() -> int:
    rows = [json.loads(line) for line in STATUTES.open(encoding="utf-8")]
    by_uid = {row["article_uid"]: row for row in rows}
    tags = sorted({tag for row in rows for tag in (row.get("risk_tags") or [])})

    section("一、法条数据库：规模与字段")
    print(f"文件            : {STATUTES.relative_to(WORKSPACE)}  ({STATUTES.stat().st_size / 1024:.0f} KB)")
    print(f"条文            : {len(rows)} 条（{rows[0]['law_name']} {rows[0]['article_no']}"
          f" ~ {rows[-1]['article_no']}）")
    print(f"带风险标签      : {sum(1 for r in rows if r.get('risk_tags'))} 条 / {len(tags)} 个标签")
    print(f"带合同类型      : {sum(1 for r in rows if r.get('contract_types'))} 条")
    print(f"字段            : {', '.join(sorted(rows[0].keys()))}")

    config = default_config()
    retriever = get_retriever(STATUTES, config)
    index = retriever.index
    print(f"内存索引        : {index.n} 条 / {index.total_len} token / 词表 {len(index.idf)}"
          f" / 平均条长 {index.avgdl:.1f}")
    print(f"融合与重排      : 权重 {retriever.weights} / rrf_k={retriever.rrf_k}"
          f" / LTR {len(retriever.ranker.weights)} 维")
    print("三路通道        : BM25 稀疏 + LSA 稠密 + 标签图 PPR（重排后追加标签扩展补位）")
    print("硬护栏          : 只接受索引里真实存在的条号，模型生成的条号在写入 Citation 前被丢弃")

    section("二、真实合同端到端：条款 → 风险 → 法条")
    targets = [p for p in (ADVERSARIAL, REAL_CONTRACT) if p.exists()]
    if not targets:
        print("语料不在本地，跳到第三部分")
    for path in targets:
        if path.is_dir():
            # 爬虫落库的真实合同按页存：文字层在 text/page_*.txt（扫描页为空串），按页序拼接
            pages = sorted(path.glob("text/page_*.txt"))
            text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in pages)
            report = review_bytes(f"{path.name}.txt", text.encode("utf-8"), config)
        else:
            report = review_path(path, config)
        print(f"\n合同   : {path.name}  评分 {report.score}  风险 {len(report.risks)} 条"
              f"  耗时 {report.stats.get('elapsed_seconds', 0):.2f} s")
        for risk in report.risks[:6]:
            laws = "；".join(f"{c.law_name}{c.article_no}" for c in risk.citations) or "（无引用，已标待补材料）"
            print(f"  - {risk.code:<24} {risk.severity:<6} 置信 {risk.confidence:.2f}  {laws}")
            if risk.citations:
                quote = risk.citations[0].quote.replace("\n", " ")[:70]
                print(f"      引文 : {quote}")
        if not report.risks:
            print("  （该合同未检出风险，属正向样本）")

    section("三、单条款探针：同一套检索链路的最近邻")
    print("说明：这里是「裸条款 + 合同类型」的最小输入，不含上下文；排序质量以 6.4 节基准为准"
          "（强金标准 Recall@5 = 0.6944、MRR@5 = 1.0000），不拿个案替基准背书。")
    for label, clause, contract_type in PROBES:
        plan = rewrite_query(clause, contract_type=contract_type)
        result = retriever.search(plan, top_k=3)
        print(f"\n【{label}】{contract_type}")
        print(f"  条款 : {clause}")
        for rank, hit in enumerate(result.hits, 1):
            if rank > 3:
                break
            row = by_uid.get(hit.article_uid, {})
            content = (row.get("content") or "").replace("\n", " ")
            origin = "标签扩展补位" if hit.channels.get("q_expansion", 0) > 0 else "文本命中"
            print(f"  {rank}. {hit.article_no}  score={hit.score:.4f}  [{origin}]")
            print(f"     {content[:86]}")

    section("四、核对方式")
    print("· 任一 article_no 都能在 data/法条结构化数据.jsonl 里搜到原文（含 source_url）")
    print("· 任一真实合同都能在 data/gov_contracts/manifest.jsonl 里对到 detail_url 与 pdf_url")
    print("· 上线后同一套数据在镜像内，判分逻辑不依赖外网")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
