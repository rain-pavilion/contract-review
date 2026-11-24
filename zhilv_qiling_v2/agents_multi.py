"""六核心节点：主脑调度 + 五个业务 Agent，带消息总线与调度器。

为什么需要这一层（不是包一层皮）：
申报书的立论是 Graph-of-Agents——节点之间通过"边"传递结构化消息，每个节点把子任务
派发给专门的模型或工具。此前 `nodes.py` 是单进程顺序调用五个函数，节点间共享同一份
Python 变量，没有 Agent 身份、没有消息协议、没有调度器。这一层补的正是这三样：

  Agent        有名字、有职责、有自己绑定的"执行体"（模型/规则/检索器）、有预算
  Message      结构化消息（type / sender / payload / trace_id / hop），可序列化落盘
  MessageBus   按拓扑图投递：只有存在边的两个 Agent 才能通信（这是 GoA 的"图"）
  Orchestrator 按依赖图调度、收集消息、处理回环（contradict 回灌），带终止条件

设计取舍：
1. **执行体可插拔**：每个 Agent 绑定一个 `executor`，可以是规则函数、BERT 模型、
   本地 LLM 服务、云端 OpenAI 兼容端点。申报书说"派发给专门的大模型"，
   这里让每个 Agent 独立配置 `model`，互不影响（NER 用 BERT，逻辑用 8B，检索用 BM25+BGE）。
2. **不重写节点逻辑**：`nodes.py` 的函数原样复用，Agent 只是给它们加上身份、消息、
   预算与可观测性。这样既有单测不会失效，多智能体路径可单独开关。
3. **消息可落盘**：`run()` 返回 `AgentTrace`，能导出成 JSON——答辩时可以直接展示
   "谁在什么时候给谁发了什么"，而不是嘴上说"多智能体"。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import entailment as E
from . import learned as L
from . import nodes as N
from . import rules as R
from . import subject_graph as S
from . import text as T
from .config import Config
from .models import Contract, Evidence, RiskCandidate

# ---------------------------------------------------------------- 消息与拓扑

# GoA 的"图"：谁能给谁发消息。只有这里列出的边才允许投递——
# 通信许可与执行依赖分开：同层职责分支各自接收主脑任务，编号不表示串行前置。
#
# 主脑（orchestrator）在入口之后、各节点之前：它决定这一份合同把哪些子任务派给谁、
# 用哪个档位的执行体（cheap 规则/1.7B，strong 8B）。logic 跑完会把"高风险条款"
# 回报给主脑（回边），主脑据此决定要不要再派一次强模型复核（reviewer）。
TOPOLOGY: dict[str, tuple[str, ...]] = {
    "intake": ("orchestrator",),                            # 入口只把解析结果交给主脑
    "orchestrator": ("ner", "logic", "subject", "statute", "reviewer", "judge"),
    "ner": ("subject", "statute", "judge"),                 # 结构化要素发往下游；逻辑分支独立读合同
    "logic": ("subject", "statute", "judge", "orchestrator"),  # 回报高风险 → 主脑再派活
    "subject": ("logic", "statute", "judge", "orchestrator"),
    "statute": ("logic", "subject", "judge", "orchestrator"),
    "reviewer": ("judge", "orchestrator"),
    "judge": (),
}

CORE_NODES: tuple[dict[str, Any], ...] = (
    {"id": "orchestrator", "number": 1, "label": "主脑规划与调度"},
    {"id": "ner", "number": 2, "label": "要素抽取"},
    {"id": "logic", "number": 3, "label": "逻辑审查"},
    {"id": "subject", "number": 4, "label": "主体审查"},
    {"id": "statute", "number": 5, "label": "法条匹配"},
    {"id": "judge", "number": 6, "label": "综合评估"},
)

# 职责分层，不是并发执行承诺；②③是同层分析分支，当前执行器按资源顺序调度。
CORE_STAGES: tuple[tuple[str, ...], ...] = (
    ("orchestrator",), ("ner", "logic"), ("subject",), ("statute",), ("judge",),
)

AGENT_SPECS: dict[str, dict[str, str]] = {
    "intake": {"role": "预处理", "executor": "layout+splitter+ner", "model": "bert-crf"},
    "orchestrator": {"role": "主脑规划与调度", "executor": "validated planner + model router", "model": "router"},
    "ner": {"role": "要素抽取", "executor": "bert-crf + rules", "model": "bert-base-chinese"},
    "logic": {"role": "逻辑错误检测", "executor": "rules + detectors + llm", "model": "legalone"},
    "subject": {"role": "主体资格审查", "executor": "knowledge graph + bm25", "model": "qualification-kg"},
    "statute": {"role": "法条匹配", "executor": "rrf retrieval + entailment", "model": "bert-dual-tower"},
    "reviewer": {"role": "强模型复核", "executor": "strong llm on focus clauses", "model": "legalone-8b"},
    "judge": {"role": "综合评估", "executor": "scoring + llm review", "model": "legalone"},
}

# 插件位：外部模块可以 `register_agent("xxx", spec, handler)` 往图里加节点，
# 加完还要在 TOPOLOGY 里补边（`link_agent("a", "b")`），否则投递会被拒。
PLUGIN_SPECS: dict[str, dict[str, str]] = {}
PLUGIN_HANDLERS: dict[str, Callable[["AgentContext"], dict[str, Any]]] = {}


def register_agent(
    name: str,
    spec: Mapping[str, str],
    handler: Callable[["AgentContext"], dict[str, Any]],
    *,
    upstream: Sequence[str] = (),
    downstream: Sequence[str] = (),
) -> None:
    """注册一个插件 Agent（主脑会按拓扑把它排进调度）。"""
    PLUGIN_SPECS[name] = {k: str(v) for k, v in spec.items()}
    PLUGIN_HANDLERS[name] = handler
    if upstream or downstream:
        TOPOLOGY[name] = tuple(dict.fromkeys(list(TOPOLOGY.get(name, ())) + list(downstream)))
        for up in upstream:
            TOPOLOGY[up] = tuple(dict.fromkeys(list(TOPOLOGY.get(up, ())) + [name]))
    else:
        TOPOLOGY.setdefault(name, ())


def agent_specs() -> dict[str, dict[str, str]]:
    """内置 + 插件，合成一张总表。"""
    merged = dict(AGENT_SPECS)
    merged.update(PLUGIN_SPECS)
    return merged



@dataclass
class Message:
    """Agent 之间传递的结构化消息。可序列化，便于答辩时展示通信轨迹。"""

    sender: str
    receiver: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    hop: int = 0
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "kind": self.kind,
            "payload": _jsonable(self.payload),
            "trace_id": self.trace_id,
            "hop": self.hop,
            "at": round(self.at, 4),
        }


def _jsonable(obj: Any) -> Any:
    """消息要落盘，把不可序列化的东西降级成可读字符串。"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, RiskCandidate):
        return {
            "candidate_id": obj.candidate_id,
            "code": obj.code,
            "severity": obj.severity,
            "confidence": round(float(obj.confidence), 4),
            "clause_ids": list(obj.clause_ids or []),
            "message": obj.message[:120],
            "evidence": len(obj.evidence),
        }
    if isinstance(obj, Evidence):
        return {"kind": obj.kind, "text": obj.text[:120], "strength": obj.strength}
    return repr(obj)[:200]


class MessageBus:
    """按 TOPOLOGY 投递。拓扑外的边直接拒投——这就是"图"的约束力所在。"""

    def __init__(self, trace_id: str | None = None, *, strict: bool = True):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.strict = strict
        self.log: list[Message] = []
        self.rejected: list[dict[str, Any]] = []

    def allowed(self, sender: str, receiver: str) -> bool:
        return receiver in TOPOLOGY.get(sender, ())

    def publish(self, message: Message) -> bool:
        message.trace_id = self.trace_id
        if self.strict and not self.allowed(message.sender, message.receiver):
            self.rejected.append(
                {"sender": message.sender, "receiver": message.receiver, "kind": message.kind}
            )
            return False
        self.log.append(message)
        return True

    def inbox(self, agent: str, kinds: Sequence[str] | None = None) -> list[Message]:
        items = [m for m in self.log if m.receiver == agent]
        if kinds:
            items = [m for m in items if m.kind in kinds]
        return items

    def stats(self) -> dict[str, Any]:
        per_edge: dict[str, int] = {}
        for m in self.log:
            key = f"{m.sender}->{m.receiver}"
            per_edge[key] = per_edge.get(key, 0) + 1
        return {
            "messages": len(self.log),
            "edges_used": per_edge,
            "rejected": len(self.rejected),
            "trace_id": self.trace_id,
        }

    def dump(self) -> list[dict[str, Any]]:
        return [m.as_dict() for m in self.log]


@dataclass
class AgentTrace:
    trace_id: str
    order: list[str]
    steps: list[dict[str, Any]]
    bus: MessageBus
    degraded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "order": self.order,
            "steps": self.steps,
            "bus": self.bus.stats(),
            "degraded": self.degraded,
        }


# ---------------------------------------------------------------- Agent

@dataclass
class Agent:
    """一个 Agent = 身份 + 职责 + 执行体 + 预算。"""

    name: str
    role: str
    executor: str
    model: str
    handle: Callable[["AgentContext"], dict[str, Any]]
    budget_seconds: float = 60.0

    def run(self, ctx: "AgentContext") -> dict[str, Any]:
        started = time.time()
        inbox = ctx.bus.inbox(self.name)
        try:
            out = self.handle(ctx) or {}
            status = out.get("status", "ok")
        except Exception as exc:  # 单个 Agent 失败不拖垮编排
            out = {"error": f"{type(exc).__name__}: {exc}"}
            status = "error"
            ctx.degraded.append(f"{self.name}:{type(exc).__name__}")
        elapsed = time.time() - started
        ctx.steps.append(
            {
                "agent": self.name,
                "role": self.role,
                "executor": self.executor,
                "model": out.get("model", self.model),
                "status": status,
                "round": int(ctx.shared.get("round", 1)),
                "seconds": round(elapsed, 3),
                "inbox": len(inbox),
                "outbox": len([m for m in ctx.bus.log if m.sender == self.name]),
                "notes": out.get("notes", [])[:4] if isinstance(out.get("notes"), list) else [],
                "error": out.get("error", ""),
            }
        )
        return out


@dataclass
class AgentContext:
    contract: Contract
    config: Config
    bus: MessageBus
    retriever: Any = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)   # 节点间共享的中间结果（不进消息）

    def emit(self, sender: str, receiver: str, kind: str, payload: dict[str, Any]) -> None:
        hop = 1 + max((m.hop for m in self.bus.log), default=-1)
        self.bus.publish(Message(sender=sender, receiver=receiver, kind=kind, payload=payload, hop=hop))
