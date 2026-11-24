"""v2 数据契约。

与 v1 `models.py` + `agents/state.py` 的差异（升级点）
-----------------------------------------------------
1. **统一到一处**：v1 把数据类拆在 `models.py`（解析态）和 `agents/state.py`（审查态），
   两边各有一份 `to_dict`，字段漂移过（`confidence` 语义两处不同）。v2 全部收在这一个
   模块，`to_dict()` 统一由 `_asdict_shallow()` 生成，保证 JSON 可序列化且字段名稳定。
2. **可追溯性升级**：所有中间结论都带 `SpanRef`（block_id + 字符区间 + 页码 + bbox），
   v1 只到 block 粒度，报告里无法定位到"第 12 行第 30 字"。这是论文里"证据可溯源"的
   硬要求（三条硬护栏之一）。
3. **证据即结构**：`Evidence` 区分 `kind`（clause_text / element / graph_edge / statute /
   retrieval），带 `strength`，仲裁节点按 `strength` 加权，而不是 v1 的"有证据/没证据"
   二元判断。
4. **可重放**：`TraceEvent` 记录每个节点的输入摘要、输出摘要、耗时、参数与降级原因，
   支持离线重放与论文截图。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 序列化工具
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """递归把 dataclass / Path / set 转成可 JSON 序列化的结构。"""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if hasattr(value, "__fspath__"):
        return str(value)
    return value


class _Record:
    """所有数据契约的公共基类：`to_dict` / `to_json` / `digest`。"""

    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}  # type: ignore[call-overload]

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def digest(self, *, length: int = 12) -> str:
        """内容指纹：用于缓存键、去重、实验结果对比。"""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


def stable_id(prefix: str, *parts: Any, length: int = 10) -> str:
    """稳定 ID：同样的输入永远得到同样的 ID（v1 用 self-increment，跨进程不可复现）。"""
    raw = "\x1f".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:length]}"


# ---------------------------------------------------------------------------
# 定位：字符级溯源
# ---------------------------------------------------------------------------

@dataclass
class SpanRef(_Record):
    """一段可回溯的文本位置。

    `quote` 里放实际命中的原文片段，报告里直接展示；`block_id` 用来回查 SourceBlock，
    `char_start/char_end` 是**该 block 文本内**的偏移，不是全文偏移（全文偏移在
    `DocumentParse.offsets` 里查），这样 block 重组后仍然有效。
    """

    block_id: str
    quote: str = ""
    char_start: int = -1
    char_end: int = -1
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    clause_id: str | None = None

    def label(self) -> str:
        bit = f"p{self.page}" if self.page else "p?"
        if self.char_start >= 0:
            return f"{self.block_id}@{bit}[{self.char_start}:{self.char_end}]"
        return f"{self.block_id}@{bit}"


# ---------------------------------------------------------------------------
# 解析层
# ---------------------------------------------------------------------------

@dataclass
class SourceBlock(_Record):
    """解析后的最小文本单元（段落 / 表格行 / 图注 / 附件页）。

    v1 只有 `kind`（paragraph/table/cell）。v2 增加 `bbox`、`column`、`font_size`、
    `is_bold`：版面分析需要几何信息，标题识别也需要字号特征（v1 纯靠正则）。
    """

    block_id: str
    order: int
    kind: str
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    column: int = 0
    font_size: float | None = None
    is_bold: bool = False
    style: str = ""
    number_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    def excerpt(self, limit: int = 60) -> str:
        text = self.text.strip().replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Clause(_Record):
    """合同条款（三级结构：编/章 → 条 → 款）。"""

    clause_id: str
    label: str
    title: str
    level: int
    heading: str
    body: str
    text: str
    block_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)
    """条款级附加信息（角色标注、要素索引等，由各智能体写入）。"""

    @property
    def length(self) -> int:
        return len(self.text)

    def path(self) -> str:
        return f"{self.label} {self.title}".strip()

    def excerpt(self, limit: int = 80) -> str:
        text = self.body or self.text
        text = text.strip().replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class BoundaryDecision(_Record):
    """相邻两个 block 之间的边界判定。

    v1 用 `relation` 四分类 + 单一 `confidence`。v2 保留四分类以便与 v1 对照，但
    `confidence` 由三路证据（规则、统计分类器、版面几何）**融合**得到，且把每一路的
    分数都留在 `channel_scores` 里，方便做消融实验。
    """

    boundary_id: str
    left_block_id: str
    right_block_id: str
    relation: str
    confidence: float
    source: str = "rule"
    channel_scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    review_required: bool = False
    warnings: list[str] = field(default_factory=list)

    CONTINUATION = "CONTINUATION"
    SAME_CLAUSE = "SAME_CLAUSE"
    NEW_SUBCLAUSE = "NEW_SUBCLAUSE"
    NEW_CLAUSE = "NEW_CLAUSE"

    def is_split(self) -> bool:
        return self.relation in (self.NEW_CLAUSE, self.NEW_SUBCLAUSE)


@dataclass
class StructureIssue(_Record):
    code: str
    message: str
    severity: str = "medium"
    block_ids: list[str] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)
    refixable: bool = False
    fix_applied: str = ""


@dataclass
class DocumentParse(_Record):
    """一份文档的解析结果。"""

    source_name: str
    file_type: str
    full_text: str
    blocks: list[SourceBlock] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    boundaries: list[BoundaryDecision] = field(default_factory=list)
    issues: list[StructureIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def block_map(self) -> dict[str, SourceBlock]:
        return {b.block_id: b for b in self.blocks}

    def clause_map(self) -> dict[str, Clause]:
        return {c.clause_id: c for c in self.clauses}

    def locate(self, block_id: str, start: int, end: int) -> SpanRef:
        block = next((b for b in self.blocks if b.block_id == block_id), None)
        if block is None:
            return SpanRef(block_id=block_id, char_start=start, char_end=end)
        return SpanRef(
            block_id=block_id,
            quote=block.text[start:end],
            char_start=start,
            char_end=end,
            page=block.page,
            bbox=block.bbox,
        )


# ---------------------------------------------------------------------------
# 要素层
# ---------------------------------------------------------------------------

@dataclass
class Element(_Record):
    """结构化要素（当事人 / 金额 / 日期 / 标的 / 编号 / 期限）。"""

    element_id: str
    kind: str
    value: str
    normalized: str
    role: str = ""
    clause_id: str | None = None
    span: SpanRef | None = None
    confidence: float = 0.5
    source: str = "rule"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractElements(_Record):
    parties: list[Element] = field(default_factory=list)
    amounts: list[Element] = field(default_factory=list)
    dates: list[Element] = field(default_factory=list)
    subjects: list[Element] = field(default_factory=list)
    identifiers: list[Element] = field(default_factory=list)
    durations: list[Element] = field(default_factory=list)

    def all(self) -> list[Element]:
        return [*self.parties, *self.amounts, *self.dates,
                *self.subjects, *self.identifiers, *self.durations]

    def by_kind(self, kind: str) -> list[Element]:
        return [e for e in self.all() if e.kind == kind]

    def counts(self) -> dict[str, int]:
        return {
            "party": len(self.parties),
            "amount": len(self.amounts),
            "date": len(self.dates),
            "subject": len(self.subjects),
            "identifier": len(self.identifiers),
            "duration": len(self.durations),
        }


@dataclass
class ElementSignal(_Record):
    """要素层面的"数学信号"：由要素之间的一致性推出来的可疑点。"""

    kind: str
    note: str
    severity: str = "medium"
    value: str = ""
    clause_id: str | None = None
    spans: list[SpanRef] = field(default_factory=list)
    confidence: float = 0.6


@dataclass
class Contract(_Record):
    contract_id: str
    document: DocumentParse
    elements: ContractElements = field(default_factory=ContractElements)
    signals: list[ElementSignal] = field(default_factory=list)
    graph_stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"
    contract_type: str = "通用"
    """合同类型（房屋租赁/委托/中介/保管/商品房预售…），由标题与角色词推断。"""
    review_context: dict[str, Any] = field(default_factory=dict)

    def clause_map(self) -> dict[str, Clause]:
        return self.document.clause_map()


# ---------------------------------------------------------------------------
# 图谱层
# ---------------------------------------------------------------------------

@dataclass
class GraphNode(_Record):
    node_id: str
    kind: str                 # clause / element / statute / risk_tag / party
    label: str
    clause_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge(_Record):
    source: str
    target: str
    kind: str                 # contains / references / defines / cites / implies
    weight: float = 1.0
    detail: str = ""


@dataclass
class ContractGraphData(_Record):
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    dangling: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def out_edges(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source == node_id]

    def in_edges(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target == node_id]

    def to_adjacency(self) -> dict[str, list[tuple[str, float]]]:
        adj: dict[str, list[tuple[str, float]]] = {}
        for edge in self.edges:
            adj.setdefault(edge.source, []).append((edge.target, edge.weight))
        return adj


@dataclass
class Statute(_Record):
    """法条。字段对齐 v1 的 `法条结构化数据.jsonl`。"""

    article_uid: str
    law_name: str
    article_no: str
    content: str
    authority_level: str = "法律"
    status: str = "有效"
    risk_tags: list[str] = field(default_factory=list)
    contract_types: list[str] = field(default_factory=list)
    source_url: str = ""

    def citation_quote(self, limit: int = 160) -> str:
        text = " ".join(self.content.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# 检索层
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit(_Record):
    article_uid: str
    law_name: str
    article_no: str
    content: str
    score: float
    channels: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    authority_level: str = "法律"
    risk_tags: list[str] = field(default_factory=list)
    source_url: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_citation(self, limit: int = 160) -> "Citation":
        text = " ".join(self.content.split())
        quote = text if len(text) <= limit else text[: limit - 1] + "…"
        return Citation(
            law_name=self.law_name,
            article_no=self.article_no,
            quote=quote,
            source_url=self.source_url,
            article_uid=self.article_uid,
            provenance=dict(self.provenance),
        )


@dataclass
class QueryPlan(_Record):
    """检索查询计划：改写后的一等公民，便于消融。"""

    raw: str
    rewritten: str = ""
    templates: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    contract_types: list[str] = field(default_factory=list)
    weight_by_template: dict[str, float] = field(default_factory=dict)
    slots: dict[str, list[str]] = field(default_factory=dict)
    """四元组槽位：(行为要件, 主体, 标的, 法律后果)，检索消融时可单独关掉某一槽。"""


# ---------------------------------------------------------------------------
# 审查层
# ---------------------------------------------------------------------------

@dataclass
class Evidence(_Record):
    """证据。`strength` 参与仲裁加权，`kind` 决定在报告里的渲染方式。"""

    kind: str
    text: str
    strength: float = 0.5
    span: SpanRef | None = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCandidate(_Record):
    """审查过程中产生的候选风险（尚未定级）。"""

    candidate_id: str
    code: str
    message: str
    severity: str = "medium"
    clause_ids: list[str] = field(default_factory=list)
    spans: list[SpanRef] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "rule"
    agent_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        fact = str((self.extra or {}).get("fact_key") or "")
        if fact:
            return f"{self.code}|fact:{fact}"
        anchor = sorted(self.clause_ids)[0] if self.clause_ids else ""
        issue = str((self.extra or {}).get("issue_code") or "")
        # 同一风险码下的不同缺陷类型（如保密缺失: insecure_delivery / public_disclosure）
        # 必须各自保留，否则仲裁会把它们并成一条、丢掉缺陷标签。
        return f"{self.code}|{anchor}|{issue}" if issue else f"{self.code}|{anchor}"

    def evidence_append(
        self,
        kind: str,
        text: str,
        strength: float = 0.6,
        source: str = "",
    ) -> None:
        """追加一条证据（同文本不重复追加）。

        节点之间互相传递结论时用这个入口：例如节点④ 判出"条款与所引条文相反"，
        就把它作为一条证据挂回候选，而不是直接改风险码。
        """
        clean = (text or "").strip()
        if not clean:
            return
        if any(item.text == clean for item in self.evidence):
            return
        self.evidence.append(
            Evidence(
                kind=kind,
                text=clean,
                strength=max(0.0, min(1.0, float(strength))),
                span=None,
                source=source or self.source,
            )
        )

    def merge_from(self, other: "RiskCandidate") -> None:
        """融合另一条候选：保留更高置信度、合并证据与来源。"""
        if other.confidence > self.confidence:
            self.confidence = other.confidence
            self.message = other.message
            self.severity = other.severity
        for src in other.agent_sources or [other.source]:
            if src and src not in self.agent_sources:
                self.agent_sources.append(src)
        for ev in other.evidence:
            if all((ev.kind, ev.source, ev.text, ev.extra) != (e.kind, e.source, e.text, e.extra) for e in self.evidence):
                self.evidence.append(ev)
        for cid in other.clause_ids:
            if cid not in self.clause_ids:
                self.clause_ids.append(cid)
        for span in other.spans:
            if all(s.label() != span.label() for s in self.spans):
                self.spans.append(span)
        for tag in other.tags:
            if tag not in self.tags:
                self.tags.append(tag)
        # Safety states are monotone across aliases: an earlier false/supported
        # value must not erase a later unresolved evidence assessment.
        for key, value in (other.extra or {}).items():
            if key in {"manual_required", "semantic_pending"}:
                self.extra[key] = bool(self.extra.get(key)) or bool(value)
            elif key == "review_flags":
                prior = self.extra.get(key) or []
                prior = prior if isinstance(prior, list) else [prior]
                incoming = value if isinstance(value, list) else [value]
                self.extra[key] = list(dict.fromkeys(str(v) for v in prior + incoming if v))
            elif key == "claim_grounding_status":
                rank = {"supported": 0, "unknown": 1, "unsupported": 2}
                old = self.extra.get(key)
                self.extra[key] = max((old, value), key=lambda v: rank.get(v, 1)) if old else value
            else:
                self.extra.setdefault(key, value)


@dataclass
class Citation(_Record):
    law_name: str
    article_no: str
    quote: str
    source_url: str = ""
    article_uid: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"《{self.law_name}》{self.article_no}"


@dataclass
class Conflict(_Record):
    type: str
    description: str
    involved_ids: list[str] = field(default_factory=list)
    resolution: str = ""
    material: bool = False


@dataclass
class FinalRisk(_Record):
    """最终风险条目（报告里的一条）。"""

    risk_id: str
    code: str
    category: str
    severity: str
    message: str
    citations: list[Citation] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)
    spans: list[SpanRef] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    calibrated_confidence: float = 0.5
    review_required: bool = False
    rationale: str = ""
    agent_sources: list[str] = field(default_factory=list)
    cross_validated: bool = False
    tags: list[str] = field(default_factory=list)

    SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}

    def sort_key(self) -> tuple[int, float]:
        return (-self.SEVERITY_ORDER.get(self.severity, 0), -self.calibrated_confidence)


RISK_CATEGORIES: tuple[str, ...] = (
    "法律风险",
    "商业不合理",
    "权利义务失衡",
    "履约风险",
    "事实不足",
    "需补充材料",
    "未发现实质风险",
)

#: 原有 27 个风险代码保持含义，另含通用跨条款不一致代码。
RISK_CODES: dict[str, tuple[str, str, str]] = {
    "zero_rent": ("零租金或未约定租金", "商业不合理", "high"),
    "abnormal_price": ("价格明显异常", "商业不合理", "medium"),
    "fee_mismatch": ("费用与总价不一致", "履约风险", "high"),
    "date_reversed": ("日期倒置", "履约风险", "high"),
    "date_missing": ("关键日期缺失", "需补充材料", "medium"),
    "amount_missing": ("合同金额缺失", "需补充材料", "medium"),
    "amount_inconsistent": ("金额大小写不一致", "履约风险", "high"),
    "party_missing": ("当事人缺失", "需补充材料", "high"),
    "missing_party_info": ("当事人信息不完整", "需补充材料", "medium"),
    "subject_unspecified": ("标的约定不明", "事实不足", "medium"),
    "vague_delivery": ("交付时间/方式模糊", "履约风险", "medium"),
    "unclear_standard": ("验收标准不明", "事实不足", "medium"),
    "liability_exemption": ("免责条款过宽", "法律风险", "high"),
    "one_sided_breach": ("违约责任单向", "权利义务失衡", "high"),
    "waiver_of_rights": ("放弃法定权利", "法律风险", "high"),
    "unilateral_modify": ("单方变更权", "权利义务失衡", "medium"),
    "illegal_terms": ("涉嫌违法条款", "法律风险", "high"),
    "unverified_property": ("标的物权属未核验", "法律风险", "medium"),
    "need_title_cert": ("需补充权属证明", "需补充材料", "medium"),
    "need_authorization": ("需补充授权文件", "需补充材料", "medium"),
    "need_license": ("需补充资质许可", "需补充材料", "medium"),
    "dispute_resolution_missing": ("争议解决方式缺失", "需补充材料", "medium"),
    "jurisdiction_ambiguous": ("管辖约定不明", "法律风险", "low"),
    "dangling_reference": ("条款交叉引用悬空", "履约风险", "medium"),
    "cross_clause_inconsistency": ("跨条款约定不一致", "履约风险", "medium"),
    "duplicate_clause": ("条款重复约定", "商业不合理", "low"),
    "confidentiality_missing": ("保密条款缺失", "需补充材料", "low"),
    "no_substantive_risk": ("未发现实质风险", "未发现实质风险", "info"),
}

CATEGORY_TO_CODES: dict[str, list[str]] = {}
for _code, (_label, _cat, _sev) in RISK_CODES.items():
    CATEGORY_TO_CODES.setdefault(_cat, []).append(_code)


def code_label(code: str) -> str:
    return RISK_CODES.get(code, (code, "法律风险", "medium"))[0]


def code_category(code: str) -> str:
    return RISK_CODES.get(code, (code, "法律风险", "medium"))[1]


def code_severity(code: str) -> str:
    return RISK_CODES.get(code, (code, "法律风险", "medium"))[2]


# ---------------------------------------------------------------------------
# 运行轨迹与报告
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent(_Record):
    """一个节点的执行记录，供重放与论文制图。"""

    seq: int
    node: str
    round: int = 0
    status: str = "ok"          # ok / degraded / skipped / error
    elapsed_ms: float = 0.0
    input_digest: str = ""
    output_digest: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class ReviewReport(_Record):
    contract_id: str
    source_name: str
    risks: list[FinalRisk] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    score: float = 100.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"

    def sorted_risks(self) -> list[FinalRisk]:
        return sorted(self.risks, key=lambda r: r.sort_key())

    def to_markdown(self) -> str:
        lines = [
            f"# 合同审查报告 · {self.source_name}",
            "",
            f"- 合同 ID：`{self.contract_id}`",
            f"- 综合评分：**{self.score:.1f} / 100**",
        ]
        if self.severity_counts:
            parts = "、".join(f"{k} {v}" for k, v in sorted(self.severity_counts.items()))
            lines.append(f"- 风险分布：{parts}")
        if self.stats:
            lines.append(
                f"- 条款 {self.stats.get('clauses', 0)} 条 · "
                f"要素 {self.stats.get('elements', 0)} 个 · "
                f"耗时 {self.stats.get('elapsed_seconds', 0):.2f}s · "
                f"LLM {'可用' if self.stats.get('llm_available') else '不可用（已降级）'}"
            )
        from .review_assessment import review_framework_markdown
        framework = self.stats.get("review_framework")
        if isinstance(framework, dict):
            lines += ["", *review_framework_markdown(framework), ""]
        else:
            lines += ["", "未保存三层审查记录。", ""]
        context_review = self.stats.get("contract_context_review")
        if isinstance(context_review, dict):
            from .context_review_report import context_review_markdown
            lines += ["", *context_review_markdown(context_review), ""]
        lines += ["", "## 评估结果关联风险明细", ""]
        if not self.risks:
            lines.append("本次未生成具体风险候选；不代表审查通过。")
        for index, risk in enumerate(self.sorted_risks(), start=1):
            head = (
                f"### {index}. [{risk.severity}] {'审查执行提示' if risk.code == 'no_substantive_risk' else code_label(risk.code)}"
                f"（{risk.category}）"
            )
            lines += [head, "", risk.message, ""]
            if risk.citations:
                lines.append("**法条依据**")
                for cite in risk.citations:
                    lines.append(f"- {cite.label()}：{cite.quote}")
            if risk.spans:
                lines.append("**定位**：" + "；".join(s.label() for s in risk.spans))
            lines.append(
                f"**置信度** {risk.confidence:.2f}"
                f"（校准后 {risk.calibrated_confidence:.2f}）"
                f" · 来源 {'+'.join(risk.agent_sources) or risk.code}"
            )
            if risk.rationale:
                lines.append(f"**理由**：{risk.rationale}")
            lines.append("")
        if self.review_queue:
            lines += ["## 人工复核队列", ""]
            for item in self.review_queue:
                lines.append(f"- `{item}` 待人工复核" if isinstance(item, str)
                             else f"- `{item.get('id')}` {item.get('reason')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def dedupe_candidates(candidates: Iterable[RiskCandidate]) -> list[RiskCandidate]:
    """按 dedupe_key 合并，保留最长证据链。"""
    table: dict[str, RiskCandidate] = {}
    for cand in candidates:
        key = cand.dedupe_key
        if key in table:
            table[key].merge_from(cand)
        else:
            table[key] = cand
    return list(table.values())


def spans_to_text(spans: Sequence[SpanRef]) -> str:
    return "".join(s.quote for s in spans if s.quote)
