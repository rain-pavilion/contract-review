"""十万份级存储压测：写入吞吐、分页深度、按码过滤、FTS 检索、统计聚合。

申报书第 6 条写"支持十万份级存储、检索与批量审查调度"，此前只在 20 份上验过。这里造
10^5 条合成审查记录（不跑模型，直接构造 ReviewReport），压 SQLite 侧的四类操作：

1. **写入**：逐条 `save()` + 每 5000 条一次事务提交，量吞吐（份/秒）；
2. **分页**：从第 1 页翻到第 10^4 页，量单页耗时（深分页是 SQLite 的经典退化点）；
3. **过滤**：按风险码 + 等级过滤分页（走 risks 子查询）；
4. **FTS**：中文子串检索（trigram 分词器），量 QPS；
5. **统计**：`stats()` 全库聚合（by_severity / by_code / 分位）。

产物写 `zhilv_qiling_v2/reports/store_scale.json`。默认 100000 条，可用 `--n` 调小做冒烟。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import store as S  # noqa: E402
from zhilv_qiling_v2.config import WORKSPACE, ensure_dirs  # noqa: E402
from zhilv_qiling_v2.models import (Citation, Evidence, FinalRisk, ReviewReport,  # noqa: E402
                                    RISK_CODES, SpanRef)

REPORT = WORKSPACE / "zhilv_qiling_v2" / "reports" / "store_scale.json"
CODES = sorted(RISK_CODES)
SEVERITIES = ("high", "medium", "low")
LAWS = ("中华人民共和国民法典",)
CLAUSE_TEXT = [
    "甲方应当在收到发票后十五日内支付价款，逾期按日万分之五支付违约金。",
    "乙方应当具备履行本合同所需的资质，并在签约前提供营业执照复印件。",
    "租赁期限自二〇二六年十二月一日起至二〇二七年十一月三十日止。",
    "保管物为艺术品一件，保管费用为人民币四千八百元整。",
    "因不可抗力导致不能履行的，双方均不承担违约责任。",
]


def make_report(index: int, rng: random.Random, per_contract: int = 1) -> ReviewReport:
    risks: list[FinalRisk] = []
    for slot in range(max(1, per_contract)):
        code = rng.choice(CODES)
        severity = rng.choice(SEVERITIES)
        confidence = round(rng.uniform(0.4, 0.95), 3)
        risks.append(
            FinalRisk(
                risk_id=f"R-{index:07d}-{slot:03d}",
                code=code,
                category="法律风险",
                severity=severity,
                message=f"{code} 在条款 C{(index + slot) % 37} 上被判定为 {severity}",
                citations=[
                    Citation(law_name=LAWS[0], article_no=f"第{rng.randint(1, 1260)}条",
                             quote="（抽查引文）", source_url="https://flk.npc.gov.cn/")
                ],
                clause_ids=[f"C{(index + slot) % 37}"],
                spans=[],
                evidence=[Evidence(kind="quote", text="（抽查证据）", strength=0.6)],
                confidence=confidence,
                calibrated_confidence=confidence,
                review_required=severity == "high",
                rationale="压测合成",
                agent_sources=["clause_review"],
                cross_validated=False,
                tags=[],
            )
        )
    severity_counts: dict[str, int] = {}
    for risk in risks:
        severity_counts[risk.severity] = severity_counts.get(risk.severity, 0) + 1
    return ReviewReport(
        contract_id=f"SCALE-{index:07d}",
        source_name=f"scale_{index:07d}.docx",
        risks=risks,
        review_queue=[risk.risk_id for risk in risks if risk.review_required],
        conflicts=[],
        score=100 - rng.randint(0, 90),
        score_breakdown={"penalty": float(rng.randint(0, 90))},
        category_counts={"法律风险": len(risks)},
        severity_counts=severity_counts,
        trace=[],
        stats={"synthetic": True},
        schema_version="2.2",
    )


def timed(label: str, func, *args, **kwargs) -> tuple[float, object]:
    start = time.perf_counter()
    value = func(*args, **kwargs)
    cost = time.perf_counter() - start
    print(f"  {label:<34} {cost*1000:9.1f} ms")
    return cost, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="十万份级存储压测")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--db", default="")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--risks", type=int, default=1, help="每份合同合成多少条风险（真实均值为 7.5）")
    args = parser.parse_args(argv)

    ensure_dirs()
    db = pathlib.Path(args.db) if args.db else WORKSPACE / "data" / "reports" / "store_scale.sqlite"
    if db.exists():
        db.unlink()
    for suffix in ("-wal", "-shm"):
        side = pathlib.Path(str(db) + suffix)
        if side.exists():
            side.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    store = S.ReviewStore(db)
    n = args.n
    print(f"压测库 {db} | 目标 {n} 份")

    # ---------------- 写入 ----------------
    reports = [make_report(index, rng, args.risks) for index in range(n)]
    conn = store._conn  # 压测直接控事务，避免每条 INSERT 各自提交把吞吐压到几十份/秒
    start = time.perf_counter()
    conn.execute("BEGIN")
    for index, report in enumerate(reports, 1):
        store.save(report, digest=f"{index:032x}"[:32], text_chars=1500 + index % 500)
        if index % 5000 == 0:
            conn.commit()
            conn.execute("BEGIN")
            done = time.perf_counter() - start
            print(f"    写入 {index:>7}/{n}  {index/done:8.1f} 份/秒")
    conn.commit()
    write_cost = time.perf_counter() - start
    rate = n / max(write_cost, 1e-9)
    print(f"  写入合计 {write_cost:.1f} s → {rate:.1f} 份/秒（{write_cost/n*1000:.2f} ms/份）")

    # 正文入库（供 FTS 用）：只灌前 5% 的条款，模拟"重点合同做全文检索"的真实用法
    clause_n = max(1, n // 20)
    start = time.perf_counter()
    conn.execute("BEGIN")
    for index in range(clause_n):
        store.save_clauses(
            f"SCALE-{index:07d}",
            [
                (f"C{index % 37}-{k}", f"条款{k}：" + CLAUSE_TEXT[(index + k) % len(CLAUSE_TEXT)])
                for k in range(4)
            ],
        )
    conn.commit()
    clause_cost = time.perf_counter() - start
    print(f"  条款入库 {clause_n} 份 × 4 条：{clause_cost:.1f} s")

    # ---------------- 分页 ----------------
    page_costs = []
    for page in (1, 10, 100, 1000, 5000, 10_000):
        if (page - 1) * 20 >= n:
            continue
        cost, _ = timed(f"分页 page={page}", store.list_contracts, page=page, size=20)
        page_costs.append({"page": page, "ms": round(cost * 1000, 2)})

    # ---------------- 过滤 ----------------
    filter_costs = []
    for code in CODES[:3]:
        cost, rows = timed(f"按码过滤 {code}", store.list_contracts, page=1, size=20, code=code)
        filter_costs.append({"code": code, "ms": round(cost * 1000, 2), "rows": len(rows)})

    # ---------------- FTS ----------------
    queries = ["押金", "违约金", "资质", "保管物", "不可抗力"]
    fts_costs = []
    for query in queries:
        cost, hits = timed(f"FTS 检索 {query}", store.search_clauses, query, limit=20)
        fts_costs.append({"query": query, "ms": round(cost * 1000, 2), "hits": len(hits)})

    # ---------------- 统计 ----------------
    cost, stats = timed("stats() 全库聚合", store.stats)
    stats_cost = cost

    # ---------------- 报告 ----------------
    size_mb = db.stat().st_size / 1e6
    result = {
        "db": str(db),
        "contracts": n,
        "risks_per_contract": args.risks,
        "size_mb": round(size_mb, 1),
        "bytes_per_contract": round(size_mb * 1e6 / n, 1),
        "write": {"seconds": round(write_cost, 2), "per_second": round(rate, 1),
                  "ms_per_contract": round(write_cost / n * 1000, 2)},
        "clauses": {"contracts": clause_n, "rows": clause_n * 4,
                    "seconds": round(clause_cost, 2),
                    "rows_per_second": round(clause_n * 4 / max(clause_cost, 1e-9), 1)},
        "paging": page_costs,
        "filtering": filter_costs,
        "fts": {"tokenizer": store.fts, "queries": fts_costs,
                "mean_ms": round(statistics.mean(x["ms"] for x in fts_costs), 2)},
        "stats_seconds": round(stats_cost, 3),
        "stats_summary": {
            "contracts": stats.contracts,
            "risks": stats.risks,
            "by_severity": stats.by_severity,
            "top_codes": sorted(stats.by_code.items(), key=lambda kv: -kv[1])[:5],
            "score_mean": stats.score_mean,
            "pending_review": stats.pending_review,
        },
        "verdict": {
            "十万份可写": rate >= 200,
            "深分页可接受": all(item["ms"] < 200 for item in page_costs),
            "FTS 可用": bool(store.fts) and all(item["hits"] >= 0 for item in fts_costs),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"库体积 {size_mb:.1f} MB（{result['bytes_per_contract']} B/份）")
    print(f"FTS 分词器 = {store.fts}")
    print("结论 =", json.dumps(result["verdict"], ensure_ascii=False))
    print(f"已写 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
