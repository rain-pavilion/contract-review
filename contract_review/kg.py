"""Deterministic knowledge graphs built on top of M2/M3 outputs.

Two independent layers, both pure Python so they run offline and stay
testable without the vector index or any model:

- ``ContractGraph``  : clauses + extracted elements as nodes; clause tree,
  element location and cross-references (第X条/款/项) as edges. The graph
  queries surface structural defects the LLM reviewers cannot be trusted to
  notice consistently, above all dangling clause references.
- ``LegalTagGraph``  : the statute JSONL already carries ``risk_tags`` and
  ``contract_types``, which is a latent article-tag bipartite graph. Making
  it explicit lets retrieval expand along tags after the vector/BM25 recall,
  and gives a tag-only fallback when the Qdrant index is absent.

The graphs only supply *facts* (ElementSignal / extra candidate articles).
They never emit risk verdicts — that boundary stays with the review agents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agents.state import ElementSignal
from .legal_kb import RetrievedArticle
from .models import Contract

DEFAULT_STATUTE_FILE = Path("data/法条结构化数据.jsonl")


# --------------------------------------------------------------------------
# Chinese numerals (第X条 references; enough range for contract clauses)
# --------------------------------------------------------------------------

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def cn_to_int(text: str) -> int | None:
    """Convert a Chinese or arabic clause number ('二十一' / '21') to int."""

    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    section = 0
    seen_any = False
    for char in text:
        digit = _CN_DIGITS.get(char)
        if digit is not None:
            section += digit
            seen_any = True
            continue
        unit = {"十": 10, "百": 100, "千": 1000}.get(char)
        if unit is None:
            return None
        # “第一条” style never carries units, but “第十条” has a bare 十.
        total += (section or 1) * unit
        section = 0
        seen_any = True
    return total + section if seen_any else None


# --------------------------------------------------------------------------
# Layer A: contract structure graph
# --------------------------------------------------------------------------

# 第X条/款/项, Chinese or arabic numerals. 章/节 headings are not operative
# cross-references inside clause bodies, so they are intentionally excluded.
_CLAUSE_REF_RE = re.compile(
    r"第\s*(?P<num>[0-9]+|[零〇一二两三四五六七八九十百千]+)\s*"
    r"(?P<kind>条|款|项)"
)

# A reference immediately following a statute name cites a law, not another
# contract clause (“《民法典》第七百二十一条” / “民法典第七百二十一条”).
_LAW_NAME_RE = re.compile(r"(?:[《〉》]|[法典条例规定解释办法])\s*$")


@dataclass(slots=True)
class KGNode:
    node_id: str
    kind: str  # clause / party / amount / date / subject
    label: str
    clause_id: str = ""
    role: str = ""


@dataclass(slots=True)
class KGEdge:
    source: str
    target: str
    kind: str  # parent / located_in / references
    detail: str = ""


@dataclass(slots=True)
class DanglingReference:
    clause_id: str
    raw: str
    kind: str  # 条/款/项
    number: int


@dataclass(slots=True)
class ContractGraph:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    dangling: list[DanglingReference] = field(default_factory=list)
    resolved_refs: int = 0

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "resolved_references": self.resolved_refs,
            "dangling_references": len(self.dangling),
        }


def _label_number(label: str, kind_hint: str) -> int | None:
    """Best-effort number carried by a clause label.

    ``kind_hint`` (条/款/项) narrows matching when the label spells out the
    granularity; labels like “1.1” are left to the sibling fallback.
    """

    match = re.fullmatch(
        r"第?\s*([0-9]+|[零〇一二两三四五六七八九十百千]+)\s*"
        r"(条|款|项)?",
        label.strip(),
    )
    if not match:
        dotted = re.fullmatch(r"(\d+)(?:[.．](\d+))+", label.strip())
        if dotted:
            # Last segment is the most local number (1.2 -> 款-ish level).
            return int(label.split(".")[-1].split("．")[-1])
        return None
    if match.group(2) and match.group(2) != kind_hint:
        return None
    return cn_to_int(match.group(1))


def _is_statute_citation(text: str, start: int) -> bool:
    window = text[max(0, start - 12) : start]
    return bool(_LAW_NAME_RE.search(window))


def build_contract_graph(contract: Contract) -> ContractGraph:
    """Assemble the structure graph and run its deterministic queries."""

    graph = ContractGraph()
    clauses = contract.document.clauses

    clause_index = {clause.clause_id: clause for clause in clauses}
    # (granularity, number) -> clause, first occurrence wins; 款/项 also get
    # a sibling-only lookup when references are resolved below.
    numbered: dict[tuple[str, int], str] = {}
    for clause in clauses:
        for kind in ("条", "款", "项"):
            number = _label_number(clause.label, kind)
            if number is not None:
                numbered.setdefault((kind, number), clause.clause_id)

    for clause in clauses:
        graph.nodes.append(
            KGNode(
                node_id=clause.clause_id,
                kind="clause",
                label=clause.heading or clause.label,
                clause_id=clause.clause_id,
            )
        )
        if clause.parent_id and clause.parent_id in clause_index:
            graph.edges.append(
                KGEdge(clause.clause_id, clause.parent_id, "parent")
            )

    # Element nodes attach to their source clause.
    element_specs = [
        ("party", contract.elements.parties),
        ("amount", contract.elements.amounts),
        ("date", contract.elements.dates),
        ("subject", contract.elements.subjects),
    ]
    for kind, elements in element_specs:
        for element in elements:
            node_id = element.element_id
            graph.nodes.append(
                KGNode(
                    node_id=node_id,
                    kind=kind,
                    label=element.value,
                    clause_id=element.clause_id or "",
                    role=element.role,
                )
            )
            if element.clause_id and element.clause_id in clause_index:
                graph.edges.append(
                    KGEdge(node_id, element.clause_id, "located_in")
                )

    # Cross-reference edges; anything that cannot be resolved is a structural
    # defect (common after copy-pasting clauses between contract templates).
    for clause in clauses:
        siblings = [
            c.clause_id
            for c in clauses
            if c.parent_id == clause.parent_id and c.clause_id != clause.clause_id
        ]
        for match in _CLAUSE_REF_RE.finditer(clause.text):
            if _is_statute_citation(clause.text, match.start()):
                continue
            kind = match.group("kind")
            number = cn_to_int(match.group("num"))
            if number is None:
                continue
            target = numbered.get((kind, number))
            if kind in ("款", "项") and (
                not target
                or (target != clause.clause_id and target not in siblings)
            ):
                # 款/项 are local: a number resolved outside the same parent
                # is a different clause's subsection, i.e. still dangling.
                local = next(
                    (
                        cid
                        for cid in siblings
                        if _label_number(clause_index[cid].label, kind) == number
                    ),
                    None,
                )
                target = local
            if target and target != clause.clause_id:
                graph.edges.append(
                    KGEdge(
                        clause.clause_id,
                        target,
                        "references",
                        detail=match.group(0).replace(" ", ""),
                    )
                )
                graph.resolved_refs += 1
            elif not target:
                graph.dangling.append(
                    DanglingReference(
                        clause_id=clause.clause_id,
                        raw=match.group(0).replace(" ", ""),
                        kind=kind,
                        number=number,
                    )
                )

    return graph


def graph_signals(contract: Contract) -> tuple[list[ElementSignal], dict[str, int]]:
    """Structural facts from the contract graph.

    Returns (signals, graph stats). Signals join the same queue mechanism as
    the M3 mathematical signals — they never become risk verdicts on their
    own.
    """

    graph = build_contract_graph(contract)
    signals: list[ElementSignal] = []

    seen_dangling: set[tuple[str, str]] = set()
    for ref in graph.dangling:
        key = (ref.clause_id, ref.raw)
        if key in seen_dangling:
            continue
        seen_dangling.add(key)
        signals.append(
            ElementSignal(
                kind="reference",
                note=f"引用了不存在的{ref.raw}，疑似条款编号缺失或错引",
                clause_id=ref.clause_id,
                value=ref.raw,
            )
        )

    for element in contract.elements.amounts:
        if not element.role:
            signals.append(
                ElementSignal(
                    kind="amount",
                    note=f"金额“{element.value}”缺少用途/角色标注，无法关联付款义务",
                    clause_id=element.clause_id or "",
                    value=element.value,
                )
            )

    return signals, graph.stats()


# --------------------------------------------------------------------------
# Layer B: statute tag graph (article <-> risk_tag / contract_type)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Statute:
    article_uid: str
    law_name: str
    article_no: str
    content: str
    status: str
    risk_tags: list[str]
    contract_types: list[str]


class LegalTagGraph:
    """Bipartite index over the statute file with file-level caching."""

    _cache: dict[str, tuple[float, "LegalTagGraph"]] = {}

    def __init__(self, statutes: list[dict]) -> None:
        self._by_uid: dict[str, _Statute] = {}
        self._tag_index: dict[str, list[str]] = {}
        self._type_index: dict[str, list[str]] = {}
        self.expansion_count = 0
        for raw in statutes:
            if raw.get("status") and raw.get("status") != "effective":
                continue
            statute = _Statute(
                article_uid=str(raw.get("article_uid", "")),
                law_name=str(raw.get("law_name", "")),
                article_no=str(raw.get("article_no", "")),
                content=str(raw.get("content", "")),
                status=str(raw.get("status", "")),
                risk_tags=[str(t) for t in raw.get("risk_tags", []) if t],
                contract_types=[
                    str(t) for t in raw.get("contract_types", []) if t
                ],
            )
            if not statute.article_uid:
                continue
            self._by_uid[statute.article_uid] = statute
            for tag in statute.risk_tags:
                self._tag_index.setdefault(tag, []).append(statute.article_uid)
            for ctype in statute.contract_types:
                self._type_index.setdefault(ctype, []).append(
                    statute.article_uid
                )

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str = DEFAULT_STATUTE_FILE) -> "LegalTagGraph":
        path = Path(path)
        mtime = path.stat().st_mtime
        cache_key = str(path.resolve())
        cached = cls._cache.get(cache_key)
        if cached and cached[0] == mtime:
            graph = cached[1]
            graph.expansion_count = 0
            return graph
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        graph = cls(records)
        cls._cache[cache_key] = (mtime, graph)
        return graph

    @classmethod
    def load_default(cls) -> "LegalTagGraph | None":
        """Load the bundled statute file; None when it is unavailable."""

        try:
            return cls.load()
        except OSError:
            return None

    # -- queries -------------------------------------------------------------

    @property
    def article_count(self) -> int:
        return len(self._by_uid)

    def to_article(self, uid: str, score: float = 0.0) -> RetrievedArticle | None:
        statute = self._by_uid.get(uid)
        if statute is None:
            return None
        return RetrievedArticle(
            article_uid=statute.article_uid,
            law_name=statute.law_name,
            article_no=statute.article_no,
            content=statute.content,
            score=score,
            risk_tags=list(statute.risk_tags),
            contract_types=list(statute.contract_types),
        )

    def articles_for_tags(
        self, tags: list[str], limit: int = 6
    ) -> list[RetrievedArticle]:
        """Direct tag lookup — the no-index retrieval fallback path."""

        scored: dict[str, int] = {}
        for tag in tags:
            for uid in self._tag_index.get(tag, []):
                scored[uid] = scored.get(uid, 0) + 1
        ordered = sorted(scored, key=lambda uid: (-scored[uid], uid))[:limit]
        return [
            article
            for uid in ordered
            if (article := self.to_article(uid, score=float(scored[uid])))
            is not None
        ]

    def expand(
        self, seed_uids: list[str], limit: int = 2
    ) -> list[RetrievedArticle]:
        """Articles sharing risk_tags with the recalled seeds.

        Ranking is the number of shared tags, so provisions directly written
        for the same risk family surface before loosely related ones.
        """

        seeds = {uid for uid in seed_uids if uid in self._by_uid}
        score: dict[str, int] = {}
        for uid in seeds:
            for tag in self._by_uid[uid].risk_tags:
                for other in self._tag_index.get(tag, []):
                    if other not in seeds:
                        score[other] = score.get(other, 0) + 1
        ordered = sorted(score, key=lambda uid: (-score[uid], uid))[:limit]
        articles = [
            article
            for uid in ordered
            if (article := self.to_article(uid, score=float(score[uid])))
            is not None
        ]
        self.expansion_count += len(articles)
        return articles
