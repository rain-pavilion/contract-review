"""命令行入口：`python -m zhilv_qiling_v2.cli <子命令>`。

子命令一览：

======================  ==========================================================
`review`                审查单个文件或多个文件，输出 Markdown/JSON/纯分数
`batch`                 按目录批量审查，输出汇总与 CSV
`explain`               单份审查并打印节点轨迹（耗时、输入/输出摘要）
`retrieve`              条款文本 → 法条检索（可直接对比 RRF 与 LTR 排序）
`crawl`                 抓取政府采购合同（源可多选，可带 OCR）
`ocr`                   对 PDF/图片做 OCR，可输出 CER 对照
`train`                 训练五个组件并写报告
`models`                已训练权重清单
`doctor`                环境自检：依赖、模型、语料、法条索引
`schema`                打印风险码表与七分类映射（论文附录/前端下拉框用）
======================  ==========================================================

设计原则：所有子命令都支持 `--json` 输出机器可读结果，退出码 0/1/2 区分成功、
业务失败、参数错误；不给 traceback 直接砸在用户脸上（`--debug` 才显示）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import (
    MODEL_DIR,
    V1_GENERATED,
    V1_GOV,
    V1_STATUTES,
    WORKSPACE,
    capabilities,
    config_from_env,
    default_config,
)
from .version import VERSION_TAG


# ---------------------------------------------------------------------------
# 输出小工具
# ---------------------------------------------------------------------------

def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _print_report(report, *, verbose: bool = False) -> None:
    """把 ReviewReport 打成人类可读的 Markdown。"""
    print(f"# {report.source_name}")
    print()
    print(f"- 合同编号：`{report.contract_id}`")
    print(f"- 综合评分：**{report.score}**  {report.score_breakdown}")
    print(f"- 风险数：{len(report.risks)}  待人工复核：{report.review_queue}"
          f"  仲裁冲突：{len(report.conflicts)}")
    print(f"- 分类计数：{report.category_counts}")
    print(f"- 耗时：{report.stats.get('elapsed_seconds')} s"
          f"（条款 {report.stats.get('clauses')} 条，候选 {report.stats.get('candidates')}）")
    print()
    from .review_assessment import review_framework_markdown
    framework = report.stats.get("review_framework")
    print("\n".join(review_framework_markdown(framework)) if isinstance(framework, dict)
          else "未保存三层审查记录。")
    print()
    if not report.risks:
        print("本次未生成具体风险候选；不代表审查通过。")
        return
    print("| # | 风险码 | 分类 | 等级 | 置信 | 说明 | 法条 |")
    print("|---|--------|------|------|------|------|------|")
    for index, risk in enumerate(report.risks, start=1):
        citations = "；".join(
            f"{citation.law_name}{citation.article_no}" for citation in risk.citations
        ) or "—"
        message = risk.message.replace("|", "／")
        if not verbose and len(message) > 60:
            message = message[:60] + "…"
        print(f"| {index} | `{risk.code}` | {risk.category} | {risk.severity} | "
              f"{risk.calibrated_confidence:.2f} | {message} | {citations} |")
    if verbose:
        print()
        for index, risk in enumerate(report.risks, start=1):
            print(f"### {index}. {risk.code} — {risk.message}")
            if risk.rationale:
                print(f"- 理由：{risk.rationale}")
            for evidence in risk.evidence[:3]:
                print(f"- 证据（{evidence.kind}）：{evidence.text[:160]}")
            for citation in risk.citations:
                print(f"- 引用：《{citation.law_name}》{citation.article_no}"
                      f"「{citation.quote[:80]}」")
            print()


def _trace_table(report) -> None:
    print("| 序号 | 节点 | 轮次 | 状态 | 耗时 ms | 摘要 |")
    print("|------|------|------|------|---------|------|")
    for event in report.trace:
        print(f"| {event.seq} | {event.node} | {event.round} | {event.status} | "
              f"{event.elapsed_ms:.1f} | {event.summary or ''} |")


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    from .pipeline import prepare, review_path

    config = config_from_env()
    from .review_context import apply_review_options
    config = apply_review_options(config, {"contract_type": getattr(args, "contract_type", "auto"),
        "review_position": getattr(args, "review_position", "neutral")})
    if args.no_rerank:
        config.use_rerank = False
    if args.no_dp:
        config.use_dp_segmentation = False
    reports = []
    for item in args.paths:
        path = Path(item)
        if not path.exists():
            print(f"[跳过] 不存在：{path}", file=sys.stderr)
            continue
        started = time.time()
        report = review_path(path, config)
        report.stats.setdefault("wall_seconds", round(time.time() - started, 2))
        reports.append(report)
        if getattr(args, "store", False):
            from . import store as store_mod

            try:
                document = prepare(path.name, path=path, config=config)
                with store_mod.default_store(config) as db:
                    db.save(
                        report,
                        digest=report.contract_id,
                        text_chars=len(document.full_text),
                    )
                    db.save_clauses(
                        report.contract_id,
                        [(clause.clause_id, clause.text) for clause in document.clauses],
                    )
                print(f"[入库] {report.contract_id}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 —— 入库失败不该影响审查结果输出
                print(f"[入库失败] {type(exc).__name__}: {exc}", file=sys.stderr)
    if not reports:
        print("没有可审查的文件", file=sys.stderr)
        return 1
    if args.json:
        _print_json([report.to_dict() for report in reports])
        return 0
    for report in reports:
        _print_report(report, verbose=args.verbose)
        if args.trace:
            _trace_table(report)
        print()
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .pipeline import batch_review, dataset_files

    config = config_from_env()
    from .review_context import apply_review_options
    config = apply_review_options(config, {"contract_type": getattr(args, "contract_type", "auto"),
        "review_position": getattr(args, "review_position", "neutral")})
    root = Path(args.root) if args.root else None
    paths = dataset_files(root) if root else dataset_files()
    if not paths:
        print("没有找到样本", file=sys.stderr)
        return 1
    if args.limit:
        paths = paths[: args.limit]
    started = time.time()
    if getattr(args, "store", False):
        from . import store as store_mod

        with store_mod.default_store(config) as db:
            result = store_mod.batch_to_store(
                paths, db, config, limit=getattr(args, "limit", None)
            )
        print(f"入库 {result.processed} 份，跳过 {result.skipped} 份，失败 {len(result.failed)} 份")
        if args.json:
            _print_json(result.as_dict())
        return 0
    summary = batch_review(paths, config)
    elapsed = time.time() - started
    rows = [
        {
            "contract_id": report.contract_id,
            "source": report.source_name,
            "score": report.score,
            "risks": len(report.risks),
            "review_queue": report.review_queue,
            "codes": ",".join(sorted({risk.code for risk in report.risks})),
            "seconds": round(report.stats.get("elapsed_seconds", 0.0), 3),
        }
        for report in summary.reports
    ]
    scores = [row["score"] for row in rows]
    codes: dict[str, int] = {}
    for report in summary.reports:
        for risk in report.risks:
            codes[risk.code] = codes.get(risk.code, 0) + 1
    payload = {
        "count": len(rows),
        "failed": summary.failed,
        "elapsed_seconds": round(elapsed, 2),
        "score_mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "risk_histogram": dict(sorted(codes.items(), key=lambda pair: -pair[1])),
        "rows": rows,
    }
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["contract_id"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[batch] CSV → {out}")
    if args.json:
        _print_json(payload)
    else:
        print(f"样本 {payload['count']} 份，失败 {payload['failed']} 份，"
              f"总耗时 {payload['elapsed_seconds']} s")
        print(f"评分 均值 {payload['score_mean']} / 最小 {payload['score_min']}"
              f" / 最大 {payload['score_max']}")
        print("风险码分布：")
        for code, count in payload["risk_histogram"].items():
            print(f"  {code:<24} {count}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    args.verbose = True
    args.trace = True
    args.json = False
    return cmd_review(args)


def cmd_retrieve(args: argparse.Namespace) -> int:
    from .retrieve import get_retriever, rewrite_query

    config = config_from_env()
    text = args.text
    if args.file:
        candidate = Path(args.file)
        if not candidate.exists():
            print(f"文件不存在：{candidate}", file=sys.stderr)
            return 1
        text = candidate.read_text(encoding="utf-8", errors="ignore")
    if not text:
        print("需要 --text 或 --file", file=sys.stderr)
        return 2
    plan = rewrite_query(text, contract_type=args.type, codes=args.codes,
                         elements=None)
    retriever = get_retriever(V1_STATUTES, config)
    started = time.perf_counter()
    result = retriever.search(plan, top_k=args.top_k, use_rerank=not args.no_rerank)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = {
        "query": plan.raw,
        "rewritten": plan.rewritten,
        "slots": plan.slots,
        "tags": plan.tags,
        "elapsed_ms": round(elapsed_ms, 2),
        "hits": [
            {
                "rank": hit.rank,
                "score": round(hit.score, 6),
                "law": hit.law_name,
                "article": hit.article_no,
                "risk_tags": hit.risk_tags,
                "content": hit.content[:200],
            }
            for hit in result.hits
        ],
        "stats": result.stats,
    }
    if args.json:
        _print_json(payload)
        return 0
    print(f"改写：{plan.rewritten}")
    print(f"槽位：{plan.slots}")
    print(f"标签：{plan.tags}   耗时 {elapsed_ms:.2f} ms   候选 {result.stats.get('candidates')}")
    for hit in result.hits:
        print(f"{hit.rank:>2}. 《{hit.law_name}》{hit.article_no}  score={hit.score:.5f}"
              f"  tags={','.join(hit.risk_tags[:3])}")
        print(f"    {hit.content[:110]}…")
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    from .crawl import SOURCES, crawl

    names = args.sources or ["ccgp_beijing"]
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        print(f"未知源：{unknown}，可选：{sorted(SOURCES)}", file=sys.stderr)
        return 2
    result = crawl(limit=args.limit, pages=args.pages, sources=names,
                   use_ocr=args.use_ocr, token=args.token, rate=args.rate)
    if args.json:
        _print_json(result)
    else:
        print(f"合同 {result.get('contracts')} 份，页 {result.get('pages')} 页，"
              f"扫描页 {result.get('scanned')}，失败 {result.get('failed')}")
        for note in result.get("degraded", []):
            print(f"[降级] {note}")
        for contract in result.get("items", [])[:20]:
            print(f"  {contract.get('contract_id')}  {contract.get('title', '')[:40]}"
                  f"  {contract.get('amount', '')}  {contract.get('sign_date', '')}")
    return 0 if not result.get("failed") else 1


def cmd_ocr(args: argparse.Namespace) -> int:
    from .ocr import OcrBudget, evaluate_against_text_layer, get_engine

    engine = get_engine()
    if engine is None:
        print("OCR 引擎不可用（缺少 rapidocr/paddle/tesseract）", file=sys.stderr)
        return 1
    budget = OcrBudget(max_pages=args.max_pages, max_seconds=args.max_seconds, dpi=args.dpi)
    target = Path(args.path)
    if not target.exists():
        print(f"文件不存在：{target}", file=sys.stderr)
        return 1
    started = time.time()
    if target.suffix.lower() == ".pdf":
        result = engine.recognize_pdf(target, budget=budget)
        payload = result.to_dict()
        payload["elapsed_seconds"] = round(time.time() - started, 2)
        if args.cer:
            compared = evaluate_against_text_layer(target, result)
            payload["cer"] = compared
    else:
        page = engine.recognize_image(target)
        payload = {
            "pages": 1,
            "engine": getattr(engine, "engine_name", ""),
            "text": getattr(page, "text", ""),
            "confidence": round(float(getattr(page, "confidence", 0.0) or 0.0), 4),
            "elapsed_seconds": round(time.time() - started, 2),
        }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = payload.get("text") or "\n".join(
            page.get("text", "") for page in payload.get("pages_detail", []) if isinstance(page, dict)
        )
        out.write_text(text or json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ocr] 结果 → {out}")
    if args.json:
        _print_json(payload)
    else:
        print(f"引擎 {payload.get('engine')}  页数 {payload.get('pages')}  "
              f"耗时 {payload.get('elapsed_seconds')} s")
        if payload.get("stats"):
            print(f"统计 {payload['stats']}")
        if payload.get("cer"):
            print(f"CER {payload['cer']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import train as train_module

    argv: list[str] = []
    if args.only:
        argv += ["--only", args.only]
    if args.quick:
        argv += ["--quick"]
    return int(train_module.main(argv) or 0)


def cmd_models(args: argparse.Namespace) -> int:
    from .learned import registry

    reg = registry()
    payload: dict[str, Any] = {"model_dir": str(MODEL_DIR), "available": reg.available(), "files": {}}
    for name, filename in reg.FILES.items():
        path = MODEL_DIR / filename
        entry: dict[str, Any] = {"file": filename, "exists": path.exists()}
        if path.exists():
            entry["bytes"] = path.stat().st_size
            entry["modified"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
            )
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                entry["meta"] = loaded.get("meta", {})
            except (json.JSONDecodeError, OSError):
                entry["meta"] = {}
        payload["files"][name] = entry
    if args.json:
        _print_json(payload)
        return 0
    print(f"模型目录：{MODEL_DIR}")
    for name, entry in payload["files"].items():
        flag = "✓" if entry["exists"] else "✗"
        print(f"  {flag} {name:<12} {entry['file']:<22} {entry.get('bytes', 0):>8} B  "
              f"{entry.get('modified', '')}")
        meta = entry.get("meta") or {}
        keys = [key for key in ("windows", "pairs", "samples", "trained_at") if key in meta]
        if keys:
            print("      " + "  ".join(f"{key}={meta[key]}" for key in keys))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    caps = capabilities()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    for name in ("numpy", "pymupdf", "sklearn", "requests", "docx", "ocr_rapid"):
        checks.append((name, bool(getattr(caps, name, False)), "可选加速器" if name != "numpy" else "必需"))
    checks.append(("法条语料", V1_STATUTES.exists(), str(V1_STATUTES)))
    checks.append(("生成合同集", V1_GENERATED.exists(), str(V1_GENERATED)))
    checks.append(("政采合同集", V1_GOV.exists(), str(V1_GOV)))
    checks.append(("模型目录", MODEL_DIR.exists(), str(MODEL_DIR)))
    ok = all(passed for _name, passed, _detail in checks if _name in ("python>=3.10", "numpy", "法条语料"))
    if args.json:
        _print_json({"ok": ok, "checks": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in checks
        ], "capabilities": caps.to_dict()})
        return 0 if ok else 1
    print(f"智律契灵 v2 环境自检（{VERSION_TAG}）")
    for name, passed, detail in checks:
        print(f"  {'✓' if passed else '✗'} {name:<16} {detail}")
    print(f"\n结论：{'可用' if ok else '缺必需依赖'}")
    if not caps.llm:
        print("提示：未配置 ZLQL_LLM_ENDPOINT，LLM 通道整体降级为纯规则+训练权重（这是设计内的降级路径）。")
    return 0 if ok else 1


def cmd_store(args: argparse.Namespace) -> int:
    """审查结果库：统计、分页列表、单份风险、条款全文检索。"""
    from . import store as store_mod

    with store_mod.default_store(config_from_env()) as db:
        action = getattr(args, "store_action", "stats")
        if action == "stats":
            payload = db.stats().as_dict()
        elif action == "list":
            rows, total = db.list_contracts(
                page=args.page,
                size=args.size,
                code=args.code,
                severity=args.severity,
            )
            payload = {"total": total, "page": args.page, "size": args.size,
                       "items": [row.as_dict() for row in rows]}
        elif action == "risks":
            payload = {"contract_id": args.contract_id, "risks": db.get_risks(args.contract_id)}
        elif action == "search":
            payload = {"query": args.text, "hits": db.search_clauses(args.text, limit=args.limit)}
        else:  # pragma: no cover - argparse 已限定
            raise SystemExit(f"未知动作 {action}")
    if args.json:
        _print_json(payload)
    else:
        _print_store(payload, action)
    return 0


def _print_store(payload: dict, action: str) -> None:
    if action == "stats":
        print(f"合同 {payload['contracts']} 份 / 风险 {payload['risks']} 条 / 引用 {payload['citations']} 条")
        print(f"评分 均值 {payload['score_mean']} 区间 {payload['score_min']}~{payload['score_max']}")
        print(f"平均耗时 {payload['elapsed_mean_ms']} ms，待人工复核 {payload['pending_review']} 条")
        if payload.get("by_severity"):
            print("分级：" + " ".join(f"{k}={v}" for k, v in payload["by_severity"].items()))
        if payload.get("by_code"):
            top = list(payload["by_code"].items())[:8]
            print("高频码：" + " ".join(f"{k}={v}" for k, v in top))
    elif action == "list":
        print(f"共 {payload['total']} 份（第 {payload['page']} 页，每页 {payload['size']}）")
        for item in payload["items"]:
            print(f"  {item['contract_id']}  {item['score']:>3} 分  "
                  f"{item['risks']:>2} 条风险  高 {item['high']}  {item['source_name']}")
    elif action == "risks":
        print(f"{payload['contract_id']}：{len(payload['risks'])} 条风险")
        for risk in payload["risks"]:
            mark = "!" if risk["review_required"] else " "
            print(f" {mark}[{risk['severity']:<6}] {risk['code']:<24} {risk['confidence']:.2f}  "
                  f"{risk['message'][:40]}")
            if risk["articles"]:
                print(f"        引用：{'、'.join(risk['articles'])}")
    else:
        print(f"“{payload['query']}” 命中 {len(payload['hits'])} 条")
        for hit in payload["hits"]:
            print(f"  {hit.get('contract_id')} / {hit.get('clause_id')}  {str(hit.get('snippet'))[:60]}")


def cmd_bench(args: argparse.Namespace) -> int:
    """统一评测台：检索（LawBench 口径）/ 检出 / 延迟。"""
    from . import bench as bench_mod

    tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    result = bench_mod.run_bench(
        config_from_env(), tasks=tasks, limit=args.limit, save=not args.no_save
    )
    if args.json:
        _print_json(result)
    else:
        print(bench_mod.summarize(result))
        if result.get("path"):
            print(f"已写入 {result['path']}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from .models import RISK_CATEGORIES, RISK_CODES, code_category, code_label, code_severity
    from .rules import rule_catalog

    payload = {
        "categories": list(RISK_CATEGORIES),
        "codes": [
            {
                "code": code,
                "label": code_label(code),
                "category": code_category(code),
                "severity": code_severity(code),
            }
            for code in RISK_CODES
        ],
        "rules": rule_catalog(),
    }
    if args.json:
        _print_json(payload)
        return 0
    print(f"七分类：{'、'.join(payload['categories'])}")
    print(f"风险码 {len(payload['codes'])} 个：")
    for item in payload["codes"]:
        print(f"  {item['code']:<26} {item['category']:<8} {item['severity']:<7} {item['label']}")
    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zhilv-qiling", description=f"智律契灵 v2 命令行（{VERSION_TAG}）"
    )
    parser.add_argument("--version", action="version", version=f"智律契灵 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="审查单个/多个文件")
    review.add_argument("paths", nargs="+")
    review.add_argument("--json", action="store_true")
    review.add_argument("--verbose", action="store_true")
    review.add_argument("--trace", action="store_true", help="打印节点轨迹")
    review.add_argument("--no-rerank", action="store_true", help="关掉 LTR 重排（消融）")
    review.add_argument("--no-dp", action="store_true", help="改用 v1 式贪心切分（消融）")
    review.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")
    review.set_defaults(func=cmd_review)

    batch = sub.add_parser("batch", help="批量审查")
    batch.add_argument("--root", default=None)
    batch.add_argument("--limit", type=int, default=0)
    batch.add_argument("--csv", default=None, help="把逐份结果写成 CSV")
    batch.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")
    batch.add_argument("--json", action="store_true")
    batch.set_defaults(func=cmd_batch)

    explain = sub.add_parser("explain", help="单份审查 + 完整轨迹")
    explain.add_argument("paths", nargs="+")
    explain.add_argument("--json", action="store_true")
    explain.add_argument("--verbose", action="store_true")
    explain.add_argument("--trace", action="store_true")
    explain.add_argument("--no-rerank", action="store_true")
    explain.add_argument("--no-dp", action="store_true")
    explain.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")
    explain.set_defaults(func=cmd_explain)

    from .review_context import CONTRACT_TYPES, POSITION_LABELS
    for review_parser in (review, batch, explain):
        review_parser.add_argument("--contract-type", choices=("auto", *CONTRACT_TYPES), default="auto",
            help="本次合同类型；auto按文本推断，手动选择保留原推断记录")
        review_parser.add_argument("--review-position", choices=tuple(POSITION_LABELS), default="neutral",
            help="本次审查视角：neutral中立、party_a甲方、party_b乙方")

    retrieve = sub.add_parser("retrieve", help="法条检索")
    retrieve.add_argument("--text", default="")
    retrieve.add_argument("--file", default=None)
    retrieve.add_argument("--type", default="通用", help="合同类型（影响标签通道）")
    retrieve.add_argument("--codes", nargs="*", default=[])
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.add_argument("--no-rerank", action="store_true")
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(func=cmd_retrieve)

    crawl = sub.add_parser("crawl", help="抓取政府采购合同")
    crawl.add_argument("--sources", nargs="*", default=None)
    crawl.add_argument("--limit", type=int, default=5)
    crawl.add_argument("--pages", type=int, default=1)
    crawl.add_argument("--use-ocr", action="store_true")
    crawl.add_argument("--token", default=None)
    crawl.add_argument("--rate", type=float, default=1.0)
    crawl.add_argument("--json", action="store_true")
    crawl.set_defaults(func=cmd_crawl)

    ocr = sub.add_parser("ocr", help="OCR 单份 PDF/图片")
    ocr.add_argument("path")
    ocr.add_argument("--max-pages", type=int, default=30)
    ocr.add_argument("--max-seconds", type=float, default=300.0)
    ocr.add_argument("--dpi", type=int, default=200)
    ocr.add_argument("--cer", action="store_true", help="对有文字层的 PDF 计算 CER")
    ocr.add_argument("--out", default=None)
    ocr.add_argument("--json", action="store_true")
    ocr.set_defaults(func=cmd_ocr)

    train = sub.add_parser("train", help="训练五个组件")
    train.add_argument("--only", default="", help="boundary,ner,rerank,risk,calibration")
    train.add_argument("--quick", action="store_true")
    train.set_defaults(func=cmd_train)

    models = sub.add_parser("models", help="已训练权重清单")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    doctor = sub.add_parser("doctor", help="环境自检")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    store_parser = sub.add_parser("store", help="审查结果库：统计/列表/风险/条款检索")
    store_sub = store_parser.add_subparsers(dest="store_action", required=True)
    store_stats = store_sub.add_parser("stats", help="库统计")
    store_stats.set_defaults(func=cmd_store, store_action="stats")
    store_list = store_sub.add_parser("list", help="分页列出合同")
    store_list.add_argument("--page", type=int, default=1)
    store_list.add_argument("--size", type=int, default=20)
    store_list.add_argument("--code", default=None, help="按风险码过滤")
    store_list.add_argument("--severity", default=None, choices=("high", "medium", "low", "info"))
    store_list.set_defaults(func=cmd_store, store_action="list")
    store_risks = store_sub.add_parser("risks", help="单份合同的风险明细")
    store_risks.add_argument("contract_id")
    store_risks.set_defaults(func=cmd_store, store_action="risks")
    store_search = store_sub.add_parser("search", help="条款全文检索（找哪份合同写了这句话）")
    store_search.add_argument("text")
    store_search.add_argument("--limit", type=int, default=20)
    store_search.set_defaults(func=cmd_store, store_action="search")
    for _sp in (store_parser, store_stats, store_list, store_risks, store_search):
        _sp.add_argument("--json", action="store_true", help="输出 JSON")

    bench_parser = sub.add_parser("bench", help="统一评测台：retrieval,detection,latency")
    bench_parser.add_argument("--tasks", default="retrieval,detection,latency")
    bench_parser.add_argument("--limit", type=int, default=None)
    bench_parser.add_argument("--no-save", action="store_true", help="只打印不落盘")
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.set_defaults(func=cmd_bench)

    schema = sub.add_parser("schema", help="风险码表")
    schema.add_argument("--json", action="store_true")
    schema.set_defaults(func=cmd_schema)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，--debug 才显示 traceback
        import traceback

        if getattr(args, "debug", False):
            traceback.print_exc()
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
