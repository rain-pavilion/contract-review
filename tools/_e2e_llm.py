"""端到端验证：本地小模型通道（OpenAI 兼容服务）接进五节点审查链路。

前提：另起一个进程跑 `python -m zhilv_qiling_v2.local_llm --serve --port 8620`，
并把 `ZLQL_LLM_ENDPOINT` 指向它。本脚本只负责"喂合同 → 看 trace / nodes / llm 字段"。

关注三件事：
1. `stats["nodes"]["llm"]` 是否为 True（通道真的被调用，而不是降级成 llm_off）；
2. `stats["llm"]` 里 accepted / rejected 的比例（引文护栏有没有挡住编造条款）；
3. `judge_llm` 是否产出 summary 且**没有改动分数**（节点⑤只读）。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import config as C  # noqa: E402
from zhilv_qiling_v2 import pipeline as P  # noqa: E402


def main(argv: list[str]) -> int:
    target = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("samples/tech-development-contract.txt")
    cfg = C.config_from_env()
    print(f"endpoint = {cfg.llm_endpoint or '(空)'}")
    print(f"model    = {cfg.llm_model or '(空)'}")
    print(f"enable_llm = {cfg.enable_llm}  max_clauses = {cfg.llm_max_clauses}")

    started = time.perf_counter()
    report = P.review_path(target, config=cfg)
    elapsed = time.perf_counter() - started

    stats = report.stats or {}
    nodes = stats.get("nodes", {})
    llm = stats.get("logic_llm", {})
    judge = stats.get("judge_llm", {})
    print()
    print(f"耗时 {elapsed:.2f} s | 评分 {report.score} | 风险 {len(report.risks)} 条 | 条款 {stats.get('clauses')} 条")
    print(f"nodes    = {json.dumps(nodes, ensure_ascii=False)}")
    print(f"llm      = {json.dumps(llm, ensure_ascii=False)}")
    print(f"judge_llm= {json.dumps(judge, ensure_ascii=False)[:400]}")
    print(f"degraded = {report.stats.get('degraded', [])}")
    print()
    print("风险明细：")
    for risk in report.risks[:10]:
        cites = ";".join(f"{c.law_name}{c.article_no}" for c in risk.citations[:2]) or "-"
        print(f"  {risk.code:<24} {risk.severity:<6} conf={risk.calibrated_confidence:.3f} "
              f"src={','.join(risk.agent_sources)} | {(risk.message or '')[:44]} | {cites}")
    print()
    print("trace：")
    for event in report.trace:
        print(f"  {event.seq:>2} {event.node:<22} {event.elapsed_ms:>8.1f} ms  {event.summary[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
