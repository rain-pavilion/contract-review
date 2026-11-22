"""资质法规知识图谱 + 检索（节点③「主体资格审查」的 RAG 部分）。

申报书对节点③写的是"以知识图谱与检索增强生成（RAG）核验主体资格，增强事实判断的准确性"。
早先的实现只有一张内置词表（`subject_graph.QUALIFICATION_RULES`，11 组关键词），既没有图谱
也没有检索，更没有法规依据 —— 判出 `need_license` 时说不清"依据哪条规定"。本模块补上这三件事：

1. **知识库**：`data/qualification_kg.jsonl`，一条 = 一个"行业 → 资质 → 依据法规"的三元组，
   带 `source_url`（官方发布页）与 `applies_to`（触发该行业的关键词）。
2. **图**：`SubjectGraph` 把「主体 → 行业 → 资质 → 法规」建成四类节点、三类边的有向图，
   与合同里抽出的主体节点（来自 `subject_graph.build_subject_graph`）对接。
3. **RAG**：`QualificationIndex` 用 BM25（字 unigram + bigram）+ 行业关键词加权召回，
   `retrieve(subject, contract_text, top_k)` 返回带分数的条目，节点③ 据此判"是否具备"。

设计上守两条：**不编条号**（`basis_article` 允许为空，只保证法规名与发布机关准确）、
**收录已废止的资质**（`status="abolished"`，如物业服务企业资质 2017 年取消），
否则同一套规则会把合规合同误报成缺资质。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import text as T
from .config import WORKSPACE

KG_FILE = WORKSPACE / "data" / "qualification_kg.jsonl"

# BM25 参数与全项目其他稀疏通道保持一致（retrieve.StatuteIndex 用同一组值）
K1 = 1.5
B = 0.72

INDUSTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "建筑工程": ("施工", "工程", "建筑", "土建", "装修", "装饰", "安装", "市政", "监理", "勘察", "设计"),
    "食品经营": ("食品", "餐饮", "食堂", "配餐", "食材"),
    "药品器械": ("药品", "医疗器械", "医药", "试剂", "耗材"),
    "消防服务": ("消防", "火灾报警", "灭火", "消防设施"),
    "印刷": ("印刷", "印务", "排版", "出版物"),
    "道路运输": ("运输", "货运", "客运", "物流", "配送"),
    "劳务派遣": ("劳务派遣", "派遣", "人力外包"),
    "检验检测": ("检测", "检验", "检定", "校准", "测试机构"),
    # 注意：这里**不放"保密"**。合同里的保密条款太常见，一放就会把普通技术开发合同
    # 误判成涉密项目（实测：技术开发合同因含保密条款被判"需涉密信息系统集成资质"）。
    "涉密集成": ("涉密", "涉密信息系统", "分级保护", "秘密载体", "国家秘密"),
    "信息技术": ("信息系统集成", "系统集成", "信息化建设", "软件开发", "运维服务", "数据中台"),
    "医疗服务": ("医疗", "医院", "诊所", "诊疗", "护理"),
    "物业服务": ("物业", "物业管理", "保洁", "绿化养护"),
    "保安服务": ("保安", "门卫", "守护", "武装押运"),
    "拍卖": ("拍卖", "竞买", "落槌"),
    "快递业务": ("快递", "快件", "末端网点"),
}


@dataclass(frozen=True)
class QualificationEntry:
    """一条资质要求。`status="abolished"` 表示该资质已取消，用于抑制误报。"""

    entry_id: str
    industry: str
    qualification: str
    authority: str
    basis_law: str
    basis_article: str
    requirement: str
    applies_to: tuple[str, ...] = ()
    source_url: str = ""
    status: str = "active"

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "industry": self.industry,
            "qualification": self.qualification,
            "authority": self.authority,
            "basis_law": self.basis_law,
            "basis_article": self.basis_article,
            "requirement": self.requirement,
            "applies_to": list(self.applies_to),
            "source_url": self.source_url,
            "status": self.status,
        }

    def citation_text(self) -> str:
        head = f"《{self.basis_law}》"
        if self.basis_article:
            head += f"{self.basis_article}"
        return f"{head}：{self.requirement}"


@dataclass
class QualificationHit:
    entry: QualificationEntry
    score: float
    matched: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"entry": self.entry.as_dict(), "score": round(self.score, 4), "matched": list(self.matched)}


@dataclass
class SubjectGraph:
    """「主体 → 行业 → 资质 → 法规」四类节点、三类边。"""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def add_node(self, node_id: str, kind: str, label: str, **attrs: Any) -> None:
        self.nodes.setdefault(node_id, {"node_id": node_id, "kind": kind, "label": label, **attrs})

    def add_edge(self, source: str, target: str, kind: str, **attrs: Any) -> None:
        self.edges.append({"source": source, "target": target, "kind": kind, **attrs})

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), **{f"n_{k}": v for k, v in kinds.items()}}

    def paths(self, subject_label: str, limit: int = 3) -> list[str]:
        """给人看的路径串，直接进 risk 的 evidence。"""
        out: list[str] = []
        for edge in self.edges:
            node = self.nodes.get(edge["source"])
            if not node or node.get("label") != subject_label:
                continue
            mid = self.nodes.get(edge["target"], {})
            out.append(f"{subject_label} -{edge['kind']}-> {mid.get('label', edge['target'])}")
            if len(out) >= limit:
                break
        return out


def load_entries(path: Path | None = None) -> list[QualificationEntry]:
    target = Path(path or KG_FILE)
    if not target.exists():
        return []
    out: list[QualificationEntry] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.append(
            QualificationEntry(
                entry_id=str(item.get("entry_id") or ""),
                industry=str(item.get("industry") or ""),
                qualification=str(item.get("qualification") or ""),
                authority=str(item.get("authority") or ""),
                basis_law=str(item.get("basis_law") or ""),
                basis_article=str(item.get("basis_article") or ""),
                requirement=str(item.get("requirement") or ""),
                applies_to=tuple(item.get("applies_to") or ()),
                source_url=str(item.get("source_url") or ""),
                status=str(item.get("status") or "active"),
            )
        )
    return out


def _entry_text(entry: QualificationEntry) -> str:
    return " ".join(
        part
        for part in (
            entry.industry,
            entry.qualification,
            entry.requirement,
            entry.basis_law,
            " ".join(entry.applies_to),
        )
        if part
    )


class QualificationIndex:
    """BM25 检索 + 行业关键词加权。语料只有几十条，建索引是毫秒级，无需缓存文件。"""

    def __init__(self, entries: Sequence[QualificationEntry] | None = None):
        self.entries: list[QualificationEntry] = list(entries) if entries is not None else load_entries()
        self._docs: list[list[str]] = [T.tokenize(_entry_text(entry)) for entry in self.entries]
        self._len: list[int] = [len(doc) for doc in self._docs]
        self._avgdl: float = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._df: dict[str, int] = {}
        for doc in self._docs:
            for token in set(doc):
                self._df[token] = self._df.get(token, 0) + 1
        total = max(len(self._docs), 1)
        self._idf: dict[str, float] = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for token, freq in self._df.items()
        }
        self._tf: list[dict[str, int]] = []
        for doc in self._docs:
            counter: dict[str, int] = {}
            for token in doc:
                counter[token] = counter.get(token, 0) + 1
            self._tf.append(counter)

    @property
    def size(self) -> int:
        return len(self.entries)

    def _score(self, index: int, tokens: Sequence[str]) -> tuple[float, tuple[str, ...]]:
        tf = self._tf[index]
        length = max(self._len[index], 1)
        score = 0.0
        matched: list[str] = []
        for token in tokens:
            freq = tf.get(token)
            if not freq:
                continue
            idf = self._idf.get(token, 0.0)
            denom = freq + K1 * (1 - B + B * length / max(self._avgdl, 1e-6))
            score += idf * freq * (K1 + 1) / max(denom, 1e-6)
            matched.append(token)
        return score, tuple(matched[:6])

    def retrieve(
        self,
        query: str,
        *,
        contract_text: str = "",
        top_k: int = 5,
        include_abolished: bool = True,
    ) -> list[QualificationHit]:
        """按查询串召回资质条目；`contract_text` 用于行业关键词加权（合同正文没提到的行业降权）。"""
        query = T.squash(query or "")
        if not query:
            return []
        tokens = T.tokenize(query)
        hits: list[QualificationHit] = []
        for index, entry in enumerate(self.entries):
            if entry.status != "active" and not include_abolished:
                continue
            score, matched = self._score(index, tokens)
            if score <= 0:
                continue
            # 行业加权：正文里出现该行业别名，说明合同确实落在这个行业，给 1.25 倍
            boost = 1.0
            if contract_text and entry.industry in INDUSTRY_ALIASES:
                if any(alias in contract_text for alias in INDUSTRY_ALIASES[entry.industry]):
                    boost = 1.25
            hits.append(QualificationHit(entry=entry, score=score * boost, matched=matched))
        hits.sort(key=lambda hit: -hit.score)
        return hits[:top_k]


def infer_industries(contract_text: str, *, limit: int = 4) -> list[tuple[str, int]]:
    """从合同正文推断行业（命中别名个数降序）。用于给 RAG 造查询串。"""
    counts: list[tuple[str, int]] = []
    for industry, aliases in INDUSTRY_ALIASES.items():
        hit = sum(contract_text.count(alias) for alias in aliases)
        if hit:
            counts.append((industry, hit))
    counts.sort(key=lambda item: (-item[1], item[0]))
    return counts[:limit]


def build_graph(entries: Sequence[QualificationEntry]) -> SubjectGraph:
    """把知识库变成图：行业节点、资质节点、法规节点，以及 industry→requires→qualification→basis→law。"""
    graph = SubjectGraph()
    for entry in entries:
        industry_id = f"ind:{entry.industry}"
        qual_id = f"qual:{entry.qualification}"
        law_id = f"law:{entry.basis_law}"
        graph.add_node(industry_id, "industry", entry.industry)
        graph.add_node(qual_id, "qualification", entry.qualification, status=entry.status)
        graph.add_node(law_id, "statute", entry.basis_law, authority=entry.authority, url=entry.source_url)
        graph.add_edge(industry_id, qual_id, "requires", status=entry.status)
        graph.add_edge(qual_id, law_id, "based_on", article=entry.basis_article)
    return graph


_CACHE: dict[str, Any] = {}


def get_index(path: Path | None = None) -> QualificationIndex:
    key = str(path or KG_FILE)
    cached = _CACHE.get(key)
    if cached is None:
        cached = QualificationIndex(load_entries(path))
        _CACHE[key] = cached
    return cached


def get_graph(path: Path | None = None) -> SubjectGraph:
    key = f"graph::{path or KG_FILE}"
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_graph(get_index(path).entries)
        _CACHE[key] = cached
    return cached


def describe(path: Path | None = None) -> dict[str, Any]:
    index = get_index(path)
    graph = get_graph(path)
    by_status: dict[str, int] = {}
    laws = set()
    for entry in index.entries:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
        laws.add(entry.basis_law)
    return {
        "file": str(path or KG_FILE),
        "entries": index.size,
        "industries": len({entry.industry for entry in index.entries}),
        "laws": len(laws),
        "by_status": by_status,
        "graph": graph.stats(),
    }


# 需要额外语境才成立的行业：光有"信息系统集成""运维"不足以要求资质，必须出现涉密语境。
# 涉密语境只认"涉密 / 分级保护 / 秘密载体 / 国家秘密"，**不认泛"保密"**——合同里的保密条款
# 与涉密项目是两件事（技术开发合同普遍有保密条款，用它触发涉密资质是误报）。
CONTEXT_GATED: dict[str, tuple[str, ...]] = {
    "涉密集成": ("涉密", "分级保护", "秘密载体", "国家秘密"),
    "医疗服务": ("医疗", "诊疗", "执业"),
    "拍卖": ("拍卖", "竞买"),
}


@dataclass
class Verdict:
    """`should_report` 为真表示"这份合同确实要求该资质且正文未体现具备"。"""

    should_report: bool
    entry: QualificationEntry | None = None
    reason: str = ""
    suppressed: tuple[str, ...] = ()
    hits: tuple[QualificationHit, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_report": self.should_report,
            "entry": self.entry.as_dict() if self.entry else None,
            "reason": self.reason,
            "suppressed": list(self.suppressed),
            "hits": [hit.as_dict() for hit in self.hits],
        }


def qualification_verdict(
    contract_text: str,
    *,
    index: QualificationIndex | None = None,
    top_k: int = 3,
) -> Verdict:
    """节点③ 的判据：该合同所在行业**是否真的**要求某项资质。

    四条规则，按顺序判：

    1. 行业不在知识库里 → 不报（宁缺勿滥，避免"任何合同都要资质"）。
    2. 命中条目标 `status="abolished"` → **抑制**（物业服务企业资质、计算机信息系统集成资质
       都已取消，继续报就是误报）。
    3. 命中条目属于 `CONTEXT_GATED` 且正文缺少语境词 → 抑制（普通信息系统集成不需要涉密资质）。
    4. 其余 active 条目 → 报，并把法规名与要求要点作为依据。

    `suppressed` 记录"考虑过但不报"的条目，写进 evidence 让人看得见判据，而不是黑箱。
    """
    index = index or get_index()
    flat = T.squash(contract_text or "")
    if not flat:
        return Verdict(False, None, "合同正文为空", ())
    industries = infer_industries(flat)
    if not industries:
        return Verdict(False, None, "正文未出现可识别的行业关键词", ())
    suppressed: list[str] = []
    collected: list[QualificationHit] = []
    for industry, count in industries:
        for hit in index.retrieve(industry, contract_text=flat, top_k=top_k):
            if hit.entry.industry != industry:
                continue
            collected.append(hit)
            entry = hit.entry
            if entry.status == "abolished":
                suppressed.append(f"{entry.industry}:{entry.qualification}（已取消，不报）")
                continue
            gate = CONTEXT_GATED.get(entry.industry)
            if gate and not any(word in flat for word in gate):
                suppressed.append(f"{entry.industry}:{entry.qualification}（缺语境词，不报）")
                continue
            # 正文已经提到该项资质的名称（"投标人应具备食品经营许可证"），说明合同自己写清了
            if entry.qualification.split("（")[0] in flat:
                suppressed.append(f"{entry.industry}:{entry.qualification}（正文已载明）")
                continue
            return Verdict(
                True,
                entry,
                f"合同涉及「{entry.industry}」业务（行业命中 {count} 次），"
                f"按{entry.citation_text()}应先取得{entry.qualification}",
                tuple(suppressed),
                tuple(collected),
            )
    return Verdict(False, None, "命中条目均已取消或缺语境，不认定缺资质", tuple(suppressed), tuple(collected))

