"""合同结构图：把"图结构引导"落成可计算的东西。

v1 的 `kg.py` 建的是"条款-要素"共现图，只用来给检索加标签扩展。v2 把它做成
**审查用的结构图**，并额外提供三样能直接进报告的量：

1. `build_contract_graph`：节点 = 条款 / 当事人 / 金额 / 日期 / 标的，边 = 同条款提及、
   数值指向、条款交叉引用。用途：让"这条金额到底属于哪个义务"变成图上的可达性问题。
2. `graph_signals`：图上能直接读出的异常——孤立条款（无任何要素挂载）、
   引文悬空（引用指向不存在的条款）。
3. `obligation_chain`：按"谁 → 做什么 → 何时"抽义务链，供跨条款核对。

纯标准库，无第三方依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import text as T
from .models import (
    Clause,
    ContractElements,
    ContractGraphData,
    Element,
    ElementSignal,
    GraphEdge,
    GraphNode,
)


def _node_of_element(element: Element) -> GraphNode:
    kind = element.kind
    node_id = f"{kind}::{element.normalized or element.value}"
    return GraphNode(
        node_id=node_id,
        kind=kind,
        label=element.value[:20],
        clause_id=element.clause_id,
        attrs={
            "role": element.role,
            "confidence": element.confidence,
            "source": element.source,
        },
    )


def _edge_of_element(element: Element) -> list[GraphEdge]:
    if not element.clause_id:
        return []
    node_id = f"{element.kind}::{element.normalized or element.value}"
    relation = {
        "party": "appears_in",
        "amount": "states_amount",
        "date": "states_date",
        "subject": "defines_subject",
        "identifier": "identifies_by",
        "duration": "lasts",
    }.get(element.kind, "relates_to")
    return [GraphEdge(source=element.clause_id, target=node_id, kind=relation,
                      weight=float(element.confidence))]


def build_contract_graph(
    clauses: Sequence[Clause],
    elements: ContractElements,
) -> ContractGraphData:
    """建图：条款为核心，要素为叶子，条款交叉引用为边。"""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for clause in clauses:
        nodes.append(
            GraphNode(
                node_id=clause.clause_id,
                kind="clause",
                label=((clause.label or "") + " " + (clause.title or ""))[:24].strip(),
                clause_id=clause.clause_id,
                attrs={"level": clause.level, "length": len(clause.body or "")},
            )
        )
    seen: set[str] = set()
    for element in elements.all():
        node = _node_of_element(element)
        if node.node_id not in seen:
            nodes.append(node)
            seen.add(node.node_id)
        edges.extend(_edge_of_element(element))

    # 条款交叉引用（悬空的进 dangling 列表，不造"missing::"节点）
    label_index = {T.squash(c.label): c.clause_id for c in clauses if c.label}
    dangling: list[str] = []
    for clause in clauses:
        body = clause.body or clause.text or ""
        for match in re.finditer(r"第([一二三四五六七八九十百零〇\d]{1,6})条", body):
            token = T.squash("第%s条" % match.group(1))
            if token in T.squash(clause.label or ""):
                continue
            target = label_index.get(token)
            if target is None:
                if token not in dangling:
                    dangling.append(token)
                continue
            edges.append(GraphEdge(source=clause.clause_id, target=target,
                                   kind="references", weight=0.8))
    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "clauses": len(clauses),
        "parties": len(elements.parties),
        "amounts": len(elements.amounts),
        "dates": len(elements.dates),
        "dangling": len(dangling),
    }
    return ContractGraphData(nodes=nodes, edges=edges, dangling=dangling, stats=stats)


CRITICAL_ROLES = ("价款", "标的", "履行")


def graph_signals(graph: ContractGraphData, clauses: Sequence[Clause]) -> list[ElementSignal]:
    """图上直接可读的异常。

    "未挂载要素"只在**关键角色**条款上算信号（价款/标的/履行）。早先对所有 ≥60 字的条款都报，
    真实政采合同 20~40 条款里能报出 50 多条，把 `subject_unspecified` 变成噪声、评分被扣到 0；
    而保密/争议解决这类条款本来就不挂金额日期，报出来没有信息量。
    """
    out: list[ElementSignal] = []
    degree: dict[str, int] = {}
    for edge in graph.edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    for clause in clauses:
        if len(clause.body or "") < 60:
            continue
        roles = set(clause.metadata.get("roles") or ())
        if not roles & set(CRITICAL_ROLES):
            continue
        if degree.get(clause.clause_id, 0) == 0:
            out.append(
                ElementSignal(
                    kind="subject_unspecified",
                    value=clause.clause_id,
                    note=f"条款「{clause.title or clause.label}」未挂载任何要素，可能是空条款",
                    confidence=0.42,
                    severity="low",
                    clause_id=clause.clause_id,
                )
            )
    if graph.dangling:
        out.append(
            ElementSignal(
                kind="subject_unspecified",
                value="|".join(graph.dangling[:4]),
                note=f"图上有 {len(graph.dangling)} 处交叉引用指向不存在的条款："
                     + "、".join(graph.dangling[:4]),
                confidence=0.5,
                severity="medium",
            )
        )
    return out[:6]


@dataclass
class Obligation:
    actor: str
    action: str
    target: str = ""
    deadline: str = ""
    clause_id: str = ""
    confidence: float = 0.5


def obligation_chain(clauses: Sequence[Clause], elements: ContractElements) -> list[Obligation]:
    """抽"谁 → 做什么 → 何时"的义务链，供跨条款核对与报告展示。"""
    out: list[Obligation] = []
    parties = {p.normalized: p.value for p in elements.parties}
    for clause in clauses:
        body = clause.body or clause.text
        if not body:
            continue
        actor = ""
        for normalized, surface in parties.items():
            if normalized and normalized in body:
                actor = surface
                break
        for keyword in ("应当", "应于", "必须", "须于", "保证", "负责"):
            index = body.find(keyword)
            if index < 0:
                continue
            tail = body[index:index + 60]
            dates = T.all_dates(tail)
            out.append(
                Obligation(
                    actor=actor,
                    action=T.truncate(T.squash(tail), 40),
                    target=clause.title or clause.label,
                    deadline=dates[0].isoformat() if dates else "",
                    clause_id=clause.clause_id,
                    confidence=0.6 if actor else 0.45,
                )
            )
            break
    return out[:60]
