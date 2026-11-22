"""审查引擎：把五个智能体按有界回环组织成一次可复现的运行。

v1 用 LangGraph 的 `StateGraph` 组织，回环靠 LLM 输出里的字段驱动，好处是灵活，
坏处是**离线不可运行、回环次数不可预期、trace 只能从运行时捞**。

v2 换成手写调度器，换来四样工程性质：

1. **纯函数式状态**：`ReviewState` 是不可变快照，每步产出新快照，可序列化进报告；
2. **有界回环**：四条回环各有上限（`max_legal_retries` / `max_judge_retries` /
   `max_evidence_rounds` / `max_arbitration_rounds`），到上限终止并保留待核验；
3. **并发可关**：`parallel_agents=True` 时 `clause_review` 与 `cross_review` 真并行
   （ThreadPoolExecutor），关掉则顺序执行以便逐步调试；
4. **trace 全量**：每个节点的输入/输出摘要、耗时、轮次都进 `TraceEvent`，
   报告里可以直接画时序图（论文第 6 章要用）。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import agents as A
from . import graph as G
from . import nodes as N
from . import rules as R
from . import subject_graph as S
from .config import Config
from .models import (
    RISK_CATEGORIES,
    Contract,
    Conflict,
    FinalRisk,
    ReviewReport,
    RiskCandidate,
    TraceEvent,
)


def _digest(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:10]


@dataclass
class ReviewState:
    """一次审查的不可变快照（每步产出新实例）。"""

    contract: Contract
    round: int = 0
    candidates: list[RiskCandidate] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    risks: list[FinalRisk] = field(default_factory=list)
    agent_results: list[A.AgentResult] = field(default_factory=list)
    score: int = 100
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=lambda: {
        "legal_retries": 0, "judge_retries": 0,
        "evidence_rounds": 0, "arbitration_rounds": 0,
    })
    degraded: list[str] = field(default_factory=list)

    def with_candidates(self, extra: Sequence[RiskCandidate]) -> "ReviewState":
        return ReviewState(
            contract=self.contract, round=self.round,
            candidates=[*self.candidates, *extra],
            conflicts=self.conflicts, risks=self.risks,
            agent_results=list(self.agent_results),
            score=self.score, score_breakdown=self.score_breakdown,
            counters=dict(self.counters), degraded=list(self.degraded),
        )


class ReviewEngine:
    """调度器。`retriever` 为 None 时自动按 `data/法条结构化数据.jsonl` 建索引。"""

    def __init__(self, config: Config | None = None, retriever: Any = None) -> None:
        self.config = config or Config()
        self._retriever = retriever
        self.trace: list[TraceEvent] = []
        self._seq = 0

    # -- retriever 惰性获取 -------------------------------------------------
    @property
    def retriever(self):
        if self._retriever is None:
            from .config import V1_STATUTES
            from .retrieve import get_retriever

            statute_file = self.config.statute_file()
            if statute_file.exists():
                self._retriever = get_retriever(statute_file, self.config)
            else:
                self._retriever = False  # 标记为"明确没有"
        return self._retriever or None

    # -- trace --------------------------------------------------------------
    def _trace(self, node: str, *, status: str = "ok", elapsed: float = 0.0,
               input_digest: str = "", output_digest: str = "", summary: str = "",
               note: str = "", round: int = 0) -> None:
        round_no = round
        if not self.config.trace_enabled:
            return
        self._seq += 1
        self.trace.append(
            TraceEvent(
                seq=self._seq, node=node, round=int(round_no or self._seq), status=status,
                elapsed_ms=float("%.2f" % (elapsed * 1000.0)), input_digest=input_digest,
                output_digest=output_digest, summary=summary, note=note,
            )
        )

    # -- 主流程 -------------------------------------------------------------
    def run(self, contract: Contract) -> ReviewReport:
        from .entailment import review_prediction_cache

        with review_prediction_cache() as cache:
            report = self._run_review(contract)
            report.stats["entailment_cache"] = cache.summary()
            return report

    def _run_review(self, contract: Contract) -> ReviewReport:
        started = time.time()
        self.trace = []
        self._seq = 0

        # 多智能体路径：六个 Agent 走消息总线 + 图调度（申报书的 Graph-of-Agents）
        # 开关默认关闭，保证既有链路与 187 个测试的行为不变；打开后走 agents_multi_run。
        if getattr(self.config, "enable_multi_agent", False):
            return self._run_multi_agent(contract, started)

        state = ReviewState(contract=contract)
        config = self.config

        # ① understand
        t0 = time.time()
        understood = A.understand(contract, config)
        state.agent_results.append(understood)
        self._trace("understand", elapsed=time.time() - t0,
                    input_digest=_digest(contract.document.full_text[:200]),
                    output_digest=_digest(understood.notes),
                    summary="; ".join(understood.notes))

        # ②③ clause_review / cross_review（可并行）
        t0 = time.time()
        if config.parallel_agents:
            with ThreadPoolExecutor(max_workers=2) as pool:
                import contextvars
                f1 = pool.submit(contextvars.copy_context().run, A.clause_review, contract, config)
                f2 = pool.submit(contextvars.copy_context().run, A.cross_review, contract, config)
                clause_result, cross_result = f1.result(), f2.result()
        else:
            clause_result = A.clause_review(contract, config)
            cross_result = A.cross_review(contract, config)
        state.agent_results.extend([clause_result, cross_result])
        state = state.with_candidates([*clause_result.candidates, *cross_result.candidates])
        self._trace("clause_review+cross_review", elapsed=time.time() - t0,
                    output_digest=_digest(len(state.candidates)),
                    summary=f"候选 {len(state.candidates)} 条（条款层 {len(clause_result.candidates)} / "
                            f"跨条款 {len(cross_result.candidates)}）")

        # ② 的模型通道（申报书原文："调用 LegalOne-8B 逐条遍历推理"）。
        # 端点配了就跑；模型给的每条结论都必须带原文引文，否则丢弃（硬护栏在 _locate_quote）。
        logic_llm: dict[str, Any] = {}
        if getattr(config, "enable_llm", False):
            t0 = time.time()
            llm_candidates, llm_meta, llm_degraded = N._llm_logic(
                contract, config, contract.document.clauses)
            logic_llm = dict(llm_meta or {})
            if llm_candidates:
                state = state.with_candidates(llm_candidates)
            state.degraded.extend(llm_degraded)
            self._trace("logic_llm", elapsed=time.time() - t0,
                        input_digest=_digest(logic_llm.get("clauses", 0)),
                        output_digest=_digest(len(llm_candidates)),
                        summary=(f"模型遍历 {logic_llm.get('clauses', 0)} 条条款，"
                                 f"采纳 {logic_llm.get('accepted', 0)} 条 / 引文未通过 "
                                 f"{logic_llm.get('rejected', 0)} 条"
                                 + (f"；降级：{'; '.join(llm_degraded)[:90]}" if llm_degraded else "")),
                        note="; ".join(llm_degraded))

        # ③ 主体资格审查（申报书节点③；v2.1 补回，独立成节点）
        subject_hint: dict[str, Any] = {}
        if getattr(config, "enable_subject_check", True):
            t0 = time.time()
            subject = S.check_subjects(contract, config)
            subject_hint = subject.query_hint()
            if subject_hint.get("needs_government_rules") and contract.contract_type == "通用":
                contract.contract_type = "政府采购"
            state = state.with_candidates(subject.candidates)
            state.degraded.extend(subject.degraded)
            self._trace("subject_check", elapsed=time.time() - t0,
                        input_digest=_digest(subject.stats),
                        output_digest=_digest(len(subject.candidates)),
                        summary=f"主体 {subject.stats['nodes']} 个 / 图边 {subject.stats['edges']} 条 / "
                                f"结论 {len(subject.candidates)} 条 / 类型 {','.join(subject.stats['kinds']) or '未知'}")

        # 图谱信号并进候选
        if config.use_contract_graph:
            graph_data = G.build_contract_graph(contract.document.clauses, contract.elements)
            contract.graph_stats = dict(graph_data.stats)
            graph_signals = G.graph_signals(graph_data, contract.document.clauses)
            state = state.with_candidates(R.signals_to_candidates(graph_signals))
            self._trace("contract_graph", output_digest=_digest(graph_data.stats),
                        summary=f"图节点 {graph_data.stats['nodes']} / 边 {graph_data.stats['edges']}"
                                f" / 悬空引用 {graph_data.stats['dangling']}")

        # 回环 A：证据不足 -> 补跑规则（每轮放宽最短长度，抓更多边角条款）
        while (state.counters["evidence_rounds"] < config.max_evidence_rounds
               and self._needs_evidence(state)):
            state.counters["evidence_rounds"] += 1
            t0 = time.time()
            extra = R.run_rules(contract)
            known = {c.candidate_id for c in state.candidates}
            fresh = [c for c in extra if c.candidate_id not in known]
            state = state.with_candidates(fresh)
            self._trace("loop:evidence", round=state.counters["evidence_rounds"],
                        elapsed=time.time() - t0, output_digest=_digest(len(fresh)),
                        summary=f"补跑规则新增 {len(fresh)} 条候选")

        # 仲裁
        t0 = time.time()
        kept, conflicts = A.arbitrate(state.candidates, config)
        state = ReviewState(
            contract=state.contract, round=state.counters["arbitration_rounds"],
            candidates=kept, conflicts=[*state.conflicts, *conflicts],
            risks=state.risks, agent_results=state.agent_results,
            score=state.score, score_breakdown=state.score_breakdown,
            counters=dict(state.counters), degraded=list(state.degraded),
        )
        self._trace("arbitrate", elapsed=time.time() - t0,
                    output_digest=_digest(len(kept)),
                    summary=f"仲裁后 {len(kept)} 条，冲突 {len(conflicts)} 条")

        # ④ legal（带重试：第一轮用改写查询，重试用条文原句）
        attempts = 0
        while True:
            t0 = time.time()
            legal_result = A.legal(contract, state.candidates, config, self.retriever)
            state.agent_results.append(legal_result)
            cited = sum(1 for items in legal_result.citations.values() if items)
            self._trace("legal", round=attempts, elapsed=time.time() - t0,
                        output_digest=_digest(cited),
                        summary=f"{cited}/{len(state.candidates)} 条候选挂到法条")
            if cited or attempts >= config.max_legal_retries:
                break
            attempts += 1
            state.counters["legal_retries"] = attempts

        # ⑤ judge
        judge_attempts = 0
        while True:
            t0 = time.time()
            judged = A.judge(contract, state.candidates, state.agent_results, config)
            state.agent_results.append(judged)
            state.risks = judged.payload["risks"]
            state.score = judged.payload["score"]
            state.score_breakdown = judged.payload["detail"]
            self._trace("judge", round=judge_attempts, elapsed=time.time() - t0,
                        output_digest=_digest(state.score),
                        summary=f"评分 {state.score}，风险 {len(state.risks)} 条")
            if judge_attempts >= config.max_judge_retries or not self._needs_more_judging(state):
                break
            judge_attempts += 1
            state.counters["judge_retries"] = judge_attempts

        # ⑤ 的模型分支（申报书要求"模型参与综合评估"）：只写复核意见，分数仍由规则算
        judge_llm: dict[str, Any] = {}
        if getattr(config, "enable_llm", False):
            t0 = time.time()
            llm_outcome = N.node_judge(state.risks, config, contract)
            if llm_outcome.payload.get("summary"):
                judge_llm = {
                    "summary": llm_outcome.payload["summary"],
                    "focus": list(llm_outcome.payload.get("focus") or []),
                    "model": (llm_outcome.llm or {}).get("model", ""),
                }
            state.degraded.extend(llm_outcome.degraded)
            self._trace("judge_llm", elapsed=time.time() - t0,
                        output_digest=_digest(judge_llm.get("summary", "")),
                        summary=("复核意见已生成" if judge_llm else "模型不可用，仅规则评分")
                                + (f"；降级：{'; '.join(llm_outcome.degraded)[:90]}"
                                   if llm_outcome.degraded else ""),
                        note=("; ".join(llm_outcome.degraded) if llm_outcome.degraded else ""))

        # 报告装配
        category_counts = {category: 0 for category in RISK_CATEGORIES}
        severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "info": 0}
        for risk in state.risks:
            category_counts[risk.category] = category_counts.get(risk.category, 0) + 1
            severity_counts[risk.severity] = severity_counts.get(risk.severity, 0) + 1
        review_queue = [r.risk_id for r in state.risks if r.review_required]
        stats = {
            "elapsed_seconds": round(time.time() - started, 3),
            "clauses": len(contract.document.clauses),
            "blocks": len(contract.document.blocks),
            "candidates": len(state.candidates),
            "agents": [result.to_dict() for result in state.agent_results],
            "counters": dict(state.counters),
            "degraded": sorted(
                {d for result in state.agent_results for d in result.degraded}
                | set(state.degraded)
            ),
            "graph": contract.graph_stats,
            "contract_type": contract.contract_type,
            "parse": contract.document.stats,
            "subject": subject_hint,
            "judge_llm": judge_llm,
                "logic_llm": logic_llm,
            "nodes": {
                "order": list(N.NODE_ORDER),
                "subject_check": bool(getattr(config, "enable_subject_check", True)),
                "entailment": bool(getattr(config, "enable_entailment", True)),
                "llm": bool(getattr(config, "enable_llm", False)),
            },
        }
        return ReviewReport(
            contract_id=contract.contract_id,
            source_name=contract.document.source_name,
            risks=state.risks,
            review_queue=review_queue,
            conflicts=state.conflicts,
            score=state.score,
            score_breakdown=state.score_breakdown,
            category_counts=category_counts,
            severity_counts=severity_counts,
            trace=self.trace,
            stats=stats,
        )

    # -- 多智能体路径 -------------------------------------------------------

    def _run_multi_agent(self, contract: Contract, started: float) -> ReviewReport:
        """六个 Agent 走消息总线跑一遍，把结果装配成同一套 ReviewReport。

        与主链路的差别只在"怎么调度"：这里是图调度 + 消息投递（拓扑外拒收），
        风险、引用、评分、复核队列的产物结构完全一致，所以 API/Web/CLI 不用改。
        """
        from . import agents_multi_run as MAR

        run = MAR.run_multi_agent(contract, self.config, retriever=getattr(self, "retriever", None))
        risks = list(run.risks)
        category_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "info": 0}
        for risk in risks:
            category_counts[risk.category] = category_counts.get(risk.category, 0) + 1
            severity_counts[risk.severity] = severity_counts.get(risk.severity, 0) + 1
        review_queue = [r.risk_id for r in risks if r.review_required]

        # trace 用 Agent 轨迹填，前端展示"谁在什么时候给谁发了什么"
        for step in run.steps:
            self._trace(
                step.get("agent", "agent"),
                status=step.get("status", "unknown"),
                round=int(step.get("round") or 0),
                elapsed=float(step.get("seconds") or 0.0),
                input_digest=_digest(step.get("inbox")),
                output_digest=_digest(step.get("outbox")),
                summary=f"{step.get('role','')}｜模型 {step.get('model','')}｜{step.get('status','')}"
                        + (f"；{step['error'][:60]}" if step.get("error") else ""),
                note="; ".join(
                    n if isinstance(n, str) else json.dumps(n, ensure_ascii=False, default=str)
                    for n in (step.get("notes") or [])
                ),
            )

        stats = {
            "elapsed_seconds": round(time.time() - started, 3),
            "clauses": len(contract.document.clauses),
            "blocks": len(contract.document.blocks),
            "candidates": len(run.candidates),
            "agents": [s for s in run.steps],
            "counters": {},
            "degraded": sorted(set(run.degraded)),
            "graph": contract.graph_stats,
            "contract_type": contract.contract_type,
            "parse": contract.document.stats,
            "subject": run.diagnostics.get("subject", {}),
            "judge_llm": run.diagnostics.get("judge_llm", {}),
            "judge_evidence": run.diagnostics.get("judge_evidence", {}),
            "logic_llm": run.diagnostics.get("logic_llm", {}),
            "statute_checks": run.diagnostics.get("statute_checks", []),
            "cross_validation": run.diagnostics.get("cross_validation", {}),
            "multi_agent": {
                "enabled": True,
                "order": run.order,
                "topology": {k: list(v) for k, v in MAR.TOPOLOGY.items()},
                "bus": run.bus.stats() if run.bus else {},
                "messages": run.bus.dump() if run.bus else [],
                "messages_truncated": False,
                "message_count": len(run.bus.log) if run.bus else 0,
            },
            "nodes": {
                "order": list(N.NODE_ORDER),
                "subject_check": bool(getattr(self.config, "enable_subject_check", True)),
                "entailment": bool(getattr(self.config, "enable_entailment", True)),
                "llm": bool(getattr(self.config, "enable_llm", False)),
            },
        }
        score, breakdown = A.score_risks(risks, self.config)
        return ReviewReport(
            contract_id=contract.contract_id,
            source_name=contract.document.source_name,
            risks=risks,
            review_queue=review_queue,
            conflicts=[],
            score=int(score),
            score_breakdown=dict(breakdown or {}),
            category_counts=category_counts,
            severity_counts=severity_counts,
            trace=self.trace,
            stats=stats,
        )

    # -- 回环判据 -----------------------------------------------------------
    def _needs_evidence(self, state: ReviewState) -> bool:
        """候选太少（<2 条）或完全没有材料缺失类提示时，补跑一轮规则。"""
        if not state.candidates:
            return True
        codes = {c.code for c in state.candidates}
        material = {"need_title_cert", "need_authorization", "need_license", "dispute_resolution_missing"}
        return not (codes & material)

    def _needs_more_judging(self, state: ReviewState) -> bool:
        """高风险无引用、或复核队列异常大时，再审一轮（v2 里只在配置允许时触发）。"""
        if not state.risks:
            return False
        risky = [r for r in state.risks if r.severity == "high"]
        if risky and any(not r.citations for r in risky):
            return bool(self.config.max_judge_retries)
        return False
