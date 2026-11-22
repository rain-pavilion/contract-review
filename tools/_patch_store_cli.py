"""给 config 加 store_path，给 CLI 加 store / bench 两个子命令，给 api 加存储路由。

一次脚本做完，避免多次手改大文件（中文文件禁用 PowerShell 文本替换）。
"""

from __future__ import annotations

import pathlib

root = pathlib.Path(".")
cfg_path = root / "zhilv_qiling_v2" / "config.py"
cli_path = root / "zhilv_qiling_v2" / "cli.py"

# ---------- 1. config.StorePath ----------
cfg = cfg_path.read_text(encoding="utf-8")
anchor = '''    # --- 评测 ---
    seed: int = 20261104'''
addition = '''    # --- 存储与调度（申报书：十万份级存储/检索/批量审查）---
    store_path: str = ""
    """审查结果库路径；空则落 ``WORKSPACE/data/reviews.sqlite``。"""

    # --- 评测 ---
    seed: int = 20261104'''
assert anchor in cfg
cfg = cfg.replace(anchor, addition, 1)
cfg_path.write_text(cfg, encoding="utf-8")
print("config.py 加 store_path")

# ---------- 2. CLI ----------
cli = cli_path.read_text(encoding="utf-8")

# 2a. 命令实现：插在 cmd_schema 之前
anchor_fn = "def cmd_schema("
assert anchor_fn in cli
store_cmds = '''def cmd_store(args: argparse.Namespace) -> int:
    """审查结果库：统计、分页列表、单份风险、条款全文检索。"""
    from . import store as store_mod

    with store_mod.default_store(_config_from(args)) as db:
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
        _config_from(args), tasks=tasks, limit=args.limit, save=not args.no_save
    )
    if args.json:
        _print_json(result)
    else:
        print(bench_mod.summarize(result))
        if result.get("path"):
            print(f"已写入 {result['path']}")
    return 0


'''
cli = cli.replace(anchor_fn, store_cmds + anchor_fn, 1)

# 2b. 子命令注册：插在 schema 注册之前
anchor_parser = '    _add_common(parser_"schema"'
if anchor_parser not in cli:
    # 兜底：找 schema 的 add_parser 行
    import re

    match = re.search(r"^.*add_parser\(\s*\"schema\".*$", cli, re.M)
    assert match, "找不到 schema 的 add_parser"
    anchor_parser = match.group(0).strip()

store_parser = '''    store_parser = sub.add_parser("store", help="审查结果库：统计/列表/风险/条款检索")
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

'''
cli = cli.replace(anchor_parser, store_parser + anchor_parser, 1)
cli_path.write_text(cli, encoding="utf-8")
print("cli.py 加 store/bench")
