"""合同结构图：把"图结构引导"落成可计算的东西。

v1 的 `kg.py` 建的是"条款-要素"共现图，只用来给检索加标签扩展。v2 把它做成
**审查用的结构图**，并额外提供三样能直接进报告的量：

1. `build_contract_graph`：节点 = 条款 / 当事人 / 金额 / 日期 / 标的，边 = 同条款提及、
   数值指向、条款交叉引用。这些边只表示结构连接，共现或可达不证明属于同一义务。
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


_REFERENCE_NUMBERS = "一二三四五六七八九十百零〇"
_REFERENCE_RE = re.compile(rf"第([{_REFERENCE_NUMBERS}\d]{{1,6}})条")
_ATTACHMENT_ID = rf"(?:[A-Za-z\d]+|[{_REFERENCE_NUMBERS}]+)"
_ATTACHMENT_RE = re.compile(rf"附件\s*(?:\(\s*{_ATTACHMENT_ID}\s*\)|{_ATTACHMENT_ID})")
_COORDINATED_REFERENCE_RE = re.compile(r"\s*(?:、|，|,|及|和|与|以及|或)\s*")


def _source_scope(clause: Clause) -> tuple[str, str]:
    """Prefer a file identifier; a title is only a fallback for legacy metadata."""
    for key in ("source_id", "document_id", "source_title"):
        value = clause.metadata.get(key)
        if value:
            return key, str(value)
    return "", ""


def _source_name(value: str) -> str:
    return T.squash(T.full_to_half(value)).casefold()


def _explicit_reference_source(
    prefix: str,
    aliases: dict[str, tuple[str, set[tuple[str, str]]]],
) -> tuple[str, set[tuple[str, str]]] | None:
    """An explicit unknown source must not silently fall back to the current file."""
    # A line wrap is layout, not evidence that a source qualifier has ended.
    prefix = re.split(r"[。；;！？!?]", T.full_to_half(prefix))[-1].strip()
    prefix = re.sub(r"(?:中的|的|之|中)\s*$", "", prefix).rstrip()
    quoted = re.search(r'《([^《》]+)》$|「([^「」]+)」$|“([^“”]+)”$|"([^"]+)"$', prefix)
    if quoted:
        name = next(value for value in quoted.groups() if value is not None)
        leading = prefix[:quoted.start()].rstrip()
        attachments = list(_ATTACHMENT_RE.finditer(leading))
        if attachments and attachments[-1].end() == len(leading):
            # "附件A《技术协议》" is one qualifier. Looking up only the quoted
            # title can lose an attachment identifier or choose another file.
            name = attachments[-1].group() + name
        return name, aliases.get(_source_name(name), (name, set()))[1]
    normalized = _source_name(prefix)
    for alias in sorted(aliases, key=len, reverse=True):
        if normalized.endswith(alias):
            return aliases[alias]
    # Retain even an unprovided attachment, rather than linking its article number
    # to a same-numbered article of the current contract.
    attachments = list(_ATTACHMENT_RE.finditer(prefix))
    if attachments:
        name = prefix[attachments[-1].start():]
        if len(name) <= 50 and "第" not in name:
            return name, set()
    return None


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

    # Keep every candidate: neither a later source nor a repeated label may
    # overwrite an earlier clause. Unqualified references prefer their own file.
    label_index: dict[str, list[Clause]] = {}
    aliases: dict[str, tuple[str, set[tuple[str, str]]]] = {}
    for clause in clauses:
        if clause.label:
            label_index.setdefault(T.squash(clause.label), []).append(clause)
        title = str(clause.metadata.get("source_title") or "")
        names = [title, re.sub(r"\.(?:pdf|docx?|txt)$", "", title, flags=re.I)]
        attachment = _ATTACHMENT_RE.search(T.full_to_half(title))
        if attachment:
            names.append(attachment.group())
        for name in names:
            normalized = _source_name(name)
            if normalized:
                aliases.setdefault(normalized, (name, set()))[1].add(_source_scope(clause))

    # Keep dangling as strings for graph_signals and existing report consumers.
    dangling: list[str] = []
    reference_issues: list[dict[str, Any]] = []
    source_scoped_references = 0
    ambiguous_references = 0
    for clause in clauses:
        body = clause.body or clause.text or ""
        previous_end = 0
        previous_qualified: tuple[str, set[tuple[str, str]]] | None = None
        for match in _REFERENCE_RE.finditer(body):
            token = T.squash("第%s条" % match.group(1))
            candidates = label_index.get(token, [])
            qualified = _explicit_reference_source(body[:match.start()], aliases)
            if (qualified is None and previous_qualified is not None
                    and _COORDINATED_REFERENCE_RE.fullmatch(body[previous_end:match.start()])):
                # "附件A第一条、第二条" names one source for both articles.
                # Carry even an unresolved qualifier so the second reference
                # cannot fall back to an unrelated local article.
                qualified = previous_qualified
            previous_end = match.end()
            previous_qualified = qualified
            scoped = qualified is not None
            if qualified is not None:
                name, scopes = qualified
                candidates = [item for item in candidates if _source_scope(item) in scopes]
                reference = name + token
            else:
                local = [item for item in candidates if _source_scope(item) == _source_scope(clause)]
                if local:
                    candidates = local
                    scoped = True
                reference = token
            if len(candidates) != 1:
                reason = "ambiguous_reference" if candidates else "missing_reference"
                ambiguous_references += bool(candidates)
                if reference not in dangling:
                    dangling.append(reference)
                reference_issues.append({"clause_id": clause.clause_id,
                                         "reference": reference, "reason": reason,
                                         "candidate_clause_ids": [item.clause_id for item in candidates]})
                continue
            target = candidates[0].clause_id
            if target == clause.clause_id:
                continue
            source_scoped_references += scoped
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
        "ambiguous_references": ambiguous_references,
        "source_scoped_references": source_scoped_references,
        "reference_issues": reference_issues,
        "analysis_scope": "structural_only",
        "automatic_legal_confirmation_allowed": False,
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
                note=f"图上有 {len(graph.dangling)} 类交叉引用无法唯一定位到目标条款："
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
