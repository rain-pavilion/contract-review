# -*- coding: utf-8 -*-
"""节点③ 主体资格审查：法律知识图谱 + 规则核验的可落地实现。

申报书对这一节点的要求是"集成法律知识图谱与检索增强生成（RAG）技术，对合同主体的
资格、权限与信用等事项进行核验"。真做一张覆盖全国企业资质与信用的图谱不现实（也没有
公开可离线使用的数据源），所以这里把它拆成三件**能验证**的事，并用一张小图把结果串起来：

```
主体 ──has_type──→ 主体类型（企业/机关/事业单位/个人）
   │                   │
   │                   └──requires──→ 资质（按标的与合同类型推导）
   ├──holds──→ 资质凭证（文本中出现的许可证/资质表述）
   ├──authorized_by──→ 法定代表人 / 授权委托书（签约权限线索）
   ├──credit──→ 失信/被执行/经营异常（信用线索）
   └──located_in──→ 地区（与统一社会信用代码前缀比对）
```

图上每条边都带 `path` 字段（人类可读），核验结论的 `evidence` 直接引用这条路径——
这就是申报书要的"可溯源"，也是答辩时能指着讲的东西。

核验结论仍然只输出 RiskCandidate，风险码沿用现有 27 码：
* 资质缺失 → `need_license`
* 权限不明 → `need_authorization`
* 主体不明/类型与角色冲突 → `subject_unspecified`
* 存在失信线索 → `illegal_terms`（法律风险类，message 明确写"需核验信用"）

所有判定都要求"有文本证据"，没有证据就不出结论——这是全项目的硬护栏。
"""
from __future__ import annotations

import json
import re
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import legal_kg as LK
from . import text as T
from .config import Config
from .models import (Citation, Clause, Contract, Evidence, RiskCandidate, SpanRef,
                     code_severity, stable_id)

# ---------------------------------------------------------------- 词表与规则

#: 主体名称后缀 → (主体类型, 置信度)。顺序即优先级。
ORG_SUFFIX: tuple[tuple[str, str, float], ...] = (
    ("有限责任公司", "企业", 0.95),
    ("股份有限公司", "企业", 0.95),
    ("有限公司", "企业", 0.93),
    ("集团公司", "企业", 0.9),
    ("分公司", "企业", 0.88),
    ("公司", "企业", 0.85),
    ("合伙企业", "企业", 0.85),
    ("事务所", "企业", 0.85),
    ("厂", "企业", 0.7),
    ("研究院", "事业单位", 0.8),
    ("设计院", "事业单位", 0.8),
    ("大学", "事业单位", 0.9),
    ("学院", "事业单位", 0.88),
    ("学校", "事业单位", 0.85),
    ("医院", "事业单位", 0.88),
    ("中心", "事业单位", 0.6),
    ("委员会", "机关", 0.9),
    ("管理局", "机关", 0.92),
    ("财政局", "机关", 0.92),
    ("局", "机关", 0.8),
    ("厅", "机关", 0.8),
    ("部", "机关", 0.6),
    ("办公室", "机关", 0.85),
    ("人民政府", "机关", 0.95),
    ("银行", "企业", 0.9),
)

#: 标的关键词 → 需要的资质（(关键词组, 资质名, 依据说明)）。
#: 只列政采与常见采购标的，其余按"通用经营资质"处理。
QUALIFICATION_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("建筑", "施工", "装修", "装饰", "改造", "工程"), "建筑业企业资质", "《建筑法》第26条"),
    (("食品", "餐饮", "食堂", "配餐"), "食品经营许可证", "《食品安全法》第35条"),
    (("药品", "医疗器械", "医用"), "药品/医疗器械经营许可证", "《药品管理法》《医疗器械监督管理条例》"),
    (("物业", "保洁", "绿化养护"), "物业服务企业资质或相应许可", "地方物业服务管理规定"),
    (("消防", "安防", "监控"), "消防设施工程专业承包资质", "《消防法》"),
    (("印刷", "出版", "广告"), "印刷经营许可证", "《印刷业管理条例》"),
    (("运输", "物流", "货运"), "道路运输经营许可证", "《道路运输条例》"),
    # 只保留强线索：泛化到"信息/数据/系统/平台"会让每份政采数据类合同都命中（实测 fp=10）
    (("涉密", "等级保护", "信息系统集成", "信息安全服务", "CMMI", "网络安全"), "信息系统集成或信息安全服务资质", "采购需求文件"),
    (("劳务", "派遣"), "劳务派遣经营许可证", "《劳动合同法》第57条"),
    (("检测", "检验", "认证"), "检验检测机构资质认定（CMA）", "《检验检测机构资质认定管理办法》"),
    (("医疗", "诊疗", "体检"), "医疗机构执业许可证", "《医疗机构管理条例》"),
)

#: 签约权限的**语境**词：出现才值得检查法定代表人是否交代（避免"有主体就报"）
AUTH_CONTEXT: tuple[str, ...] = (
    "代理", "授权", "委托", "代表人",
)

#: 签约权限线索（出现任意一个即认为权限链条有交代）
AUTH_TOKENS: tuple[str, ...] = (
    "法定代表人", "法定代表", "授权委托", "授权代表", "委托代理人", "代理人",
    "授权书", "委托书", "负责人", "经办人",
)

#: 资质凭证线索（出现即认为"holds"边成立）
LICENSE_TOKENS: tuple[str, ...] = (
    "许可证", "资质", "资格证书", "认证证书", "营业执照", "执业许可", "备案证明",
)

#: 信用负面线索
CREDIT_TOKENS: tuple[str, ...] = (
    "失信", "被执行人", "经营异常", "重大税收违法", "黑名单", "严重违法", "信用中国",
)

#: 统一社会信用代码前缀（省级行政区划码前两位）→ 地区关键词
PROVINCE_CODE: dict[str, str] = {
    "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
    "21": "辽宁", "22": "吉林", "23": "黑龙江", "31": "上海", "32": "江苏",
    "33": "浙江", "34": "安徽", "35": "福建", "36": "江西", "37": "山东",
    "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
    "46": "海南", "50": "重庆", "51": "四川", "52": "贵州", "53": "云南",
    "54": "西藏", "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏",
    "65": "新疆",
}

#: 角色语义：谁是出钱方、谁是收钱方
PAYER_ROLES = {"buyer", "custodian", "depositor", "client", "principal"}
PAYEE_ROLES = {"seller", "third", "agent", "contractor", "supplier"}


@dataclass
class SubjectNode:
    node_id: str
    name: str
    role: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectEdge:
    source: str
    target: str
    kind: str
    weight: float = 1.0
    detail: str = ""
    path: str = ""


@dataclass
class SubjectResult:
    candidates: list[RiskCandidate] = field(default_factory=list)
    nodes: list[SubjectNode] = field(default_factory=list)
    edges: list[SubjectEdge] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def query_hint(self) -> dict[str, Any]:
        """传给节点④的结构化信息：主体类型决定检索偏向的合同类型与关键词。"""
        kinds = {node.kind for node in self.nodes if node.kind != "未知"}
        licenses = {
            edge.target.replace("Q:", "")
            for edge in self.edges
            if edge.kind in ("requires", "claimed_holds", "verified_fact")
        }
        return {
            "subject_kinds": sorted(kinds),
            "licenses": sorted(licenses),
            "needs_government_rules": "机关" in kinds,
        }

    def paths(self) -> list[str]:
        return [edge.path for edge in self.edges if edge.path]


# ---------------------------------------------------------------- 图构建


def _classify_kind(name: str) -> tuple[str, float]:
    """按名称后缀判主体类型；判不出时用"2~4 个汉字"猜自然人。"""
    clean = T.squash(name or "")
    for suffix, kind, confidence in ORG_SUFFIX:
        if clean.endswith(suffix) or suffix in clean[-8:]:
            return kind, confidence
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", clean):
        return "个人", 0.6
    if clean:
        return "未知", 0.3
    return "未知", 0.0


def _first_clause_with(clauses: Sequence[Clause], *tokens: str) -> Clause | None:
    for clause in clauses:
        body = clause.text or ""
        if any(token in body for token in tokens):
            return clause
    return None


def _subject_span(name: str, clauses: Sequence[Clause]) -> SpanRef | None:
    for clause in clauses:
        index = (clause.text or "").find(name)
        if index >= 0:
            return SpanRef(
                block_id=clause.block_ids[0] if clause.block_ids else "",
                quote=name,
                char_start=index,
                char_end=index + len(name),
                clause_id=clause.clause_id,
            )
    return None


def _make(
    code: str,
    message: str,
    *,
    clause: Clause | None,
    evidence_text: str,
    path: str,
    confidence: float,
    severity: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RiskCandidate:
    """与 rules._candidate 同构，但 evidence 里额外塞一条"图上路径"。"""
    from . import rules as R  # 局部导入，避免循环

    candidate = R._candidate(
        code,
        message,
        clause,
        evidence_text,
        confidence=confidence,
        severity=severity,
        source="subject_graph",
        extra=extra,
    )
    candidate.evidence.append(
        Evidence(
            kind="graph_path",
            text=T.truncate(path, 160),
            strength=min(0.95, confidence + 0.1),
            span=None,
            source="subject_graph",
            extra={"path": path},
        )
    )
    return candidate


_IDENTITY_LABEL = r"(?:统一社会信用代码|社会信用代码|工商注册号|注册号|身份证号码|身份证号|证件号码|纳税人识别号)"
_IDENTITY_FIELD = re.compile(_IDENTITY_LABEL + r"\s*(?:为|是)?[\s:：=|｜]*([A-Za-z0-9]{8,30})(?![A-Za-z0-9])")
_ROLE_NAMES = {
    "buyer": ("甲方", "采购人", "委托方", "发包人", "买受人", "需方", "买方", "出租人"),
    "seller": ("乙方", "供应商", "受托方", "承包人", "出卖人", "供货方", "供方", "卖方"),
    "third": ("丙方", "丁方"),
    "depositor": ("寄存人",), "custodian": ("保管人",),
    "lender": ("贷款人",), "borrower": ("借款人",),
    "tenant": ("承租人",), "transferor": ("转让方",), "transferee": ("受让方",),
}
_ROLE_LOOKUP = {name: role for role, names in _ROLE_NAMES.items() for name in names}
_ROLE_PATTERN = "|".join(re.escape(name) for name in sorted(_ROLE_LOOKUP, key=len, reverse=True))


def _subject_key(value: str) -> str:
    return re.sub(r"[\s（）()]", "", str(value or "")).casefold()


def _raw_subject_name(node: SubjectNode) -> str:
    return str(node.attrs.get("source_name") or node.name)


def _subject_identity_bindings(contract: Contract, nodes: Sequence[SubjectNode]) -> dict[str, dict[str, Any]]:
    """Bind IDs from explicit name/role identity fields, never mere proximity.

    A generic ID on the next line or elsewhere in the contract is not proof.
    Evidence records are deliberately not an input to this binding operation.
    """
    collected = {node.node_id: [] for node in nodes}
    ambiguous_ids: set[str] = set()
    by_role = {role: [node for node in nodes if node.role == role] for role in _ROLE_NAMES}
    sources = [(clause.clause_id, clause.text or "") for clause in contract.document.clauses]
    if not sources:
        sources = [("", contract.document.full_text or "")]
    for clause_id, source in sources:
        for unit in re.split(r"[\r\n；;。！？!?]+", source):
            if not unit.strip():
                continue
            for field in _IDENTITY_FIELD.finditer(unit):
                identity = field.group(1).upper()
                before = unit[:field.start()]
                normalized_before = _subject_key(before)
                named: list[tuple[SubjectNode, str]] = []
                for node in nodes:
                    name = _subject_key(_raw_subject_name(node))
                    if not name:
                        continue
                    for occurrence in re.finditer(re.escape(name), normalized_before):
                        prefix = normalized_before[:occurrence.start()]
                        gap = normalized_before[occurrence.end():]
                        if prefix and (prefix[-1].isalnum() or prefix[-1] == "_"):
                            continue
                        if not re.fullmatch(r"[,，:：|｜]*(?:的)?[,，:：|｜]*", gap):
                            continue
                        role_header = re.search(r"(" + _ROLE_PATTERN + r")[:：]$", prefix)
                        if role_header and node.role != _ROLE_LOOKUP[role_header.group(1)]:
                            continue
                        named.append((node, "named_identifier_field"))
                        break
                role_field = re.search(r"(" + _ROLE_PATTERN + r")(?:的)?\s*$", before)
                role_nodes = by_role.get(_ROLE_LOOKUP[role_field.group(1)], []) if role_field else []
                choices = {node.node_id: (node, basis) for node, basis in named}
                for node in role_nodes:
                    choices.setdefault(node.node_id, (node, "role_identifier_field"))
                if len(choices) != 1:
                    ambiguous_ids.update(choices)
                    continue
                node, basis = next(iter(choices.values()))
                if re.match(r"\s*(?:[/、]|或|以及|和|与)\s*([A-Za-z0-9]{8,30})(?![A-Za-z0-9])", unit[field.end():]):
                    ambiguous_ids.add(node.node_id)
                collected[node.node_id].append({"subject_id": identity, "clause_id": clause_id,
                    "quote": unit.strip(), "binding": basis, "field": field.group(0)})
    owners: dict[str, set[str]] = {}
    for node in nodes:
        for entry in collected[node.node_id]:
            owners.setdefault(entry["subject_id"], set()).add(_subject_key(_raw_subject_name(node)))
    bindings = {}
    for node in nodes:
        evidence = collected[node.node_id]
        identifiers = sorted({entry["subject_id"] for entry in evidence})
        shared = any(len(owners[identity]) > 1 for identity in identifiers)
        ambiguous = node.node_id in ambiguous_ids or len(identifiers) > 1 or shared
        bindings[node.node_id] = {"status": "ambiguous" if ambiguous else "bound" if identifiers else "unbound",
            "identifiers": identifiers,
            "reason": "identifier_shared_by_different_subjects" if shared else
                      "ambiguous_subject_identifier_binding" if ambiguous else
                      "explicit_subject_identifier_field" if identifiers else "no_explicit_subject_identifier_binding",
            "evidence": evidence, "method": "explicit_subject_identifier_fields_only"}
    return bindings


def _nodes_in_clause_scope(nodes: Sequence[SubjectNode], clauses: Sequence[Clause]) -> list[SubjectNode]:
    selected: set[str] = set()
    for clause in clauses:
        text = _subject_key(clause.text)
        mentions = [(node, match.start(), match.end()) for node in nodes
            if _subject_key(_raw_subject_name(node))
            for match in re.finditer(re.escape(_subject_key(_raw_subject_name(node))), text)]
        for node, start, end in mentions:
            # A short company name inside another known party's longer name is
            # not a separate reference to that short-named party.
            if any(other_start <= start and other_end >= end and other_end - other_start > end - start
                   for _, other_start, other_end in mentions):
                continue
            header = re.search(r"(" + _ROLE_PATTERN + r")[:：]$", text[:start])
            if not header or node.role == _ROLE_LOOKUP[header.group(1)]:
                selected.add(node.node_id)
        for match in re.finditer(_ROLE_PATTERN, text):
            if any(start <= match.start() and end >= match.end() for _, start, end in mentions):
                continue
            role = _ROLE_LOOKUP[match.group(0)]
            suffix = text[match.end():]
            # An explicit role header naming a different known party is an
            # identity conflict, not a reference to the normal role holder.
            if suffix.startswith((":", "：")):
                named = [node for node in nodes if suffix[1:].startswith(_subject_key(_raw_subject_name(node)))]
                if named and not any(node.role == role for node in named):
                    continue
            selected.update(node.node_id for node in nodes if node.role == role)
    return [node for node in nodes if node.node_id in selected]


_ACTIVITY_PATTERNS = (
    r"(?:建筑|装饰|装修|消防设施|安防)(?:工程|施工)|(?:承担|负责|实施|提供).{0,10}(?:施工|装修|建筑工程)",
    r"餐饮服务|配餐服务|食堂.{0,6}(?:经营|承包)|食品(?:采购|经营|销售|供应)|(?:提供|销售|经营|供应).{0,8}食品",
    r"(?:药品|医疗器械)(?:采购|经营|销售|供应)|(?:销售|经营|供应).{0,6}(?:药品|医疗器械)",
    r"(?:印刷|出版)(?:服务|业务|经营|制作|加工|采购)|(?:提供|负责|承接|从事|开展|进行|承担).{0,10}(?:印刷|出版)",
    r"道路运输经营|货物运输服务|货运服务|运输服务(?:采购|合同)|(?:从事|经营).{0,6}(?:货运|运输业务)",
    r"涉密.{0,8}(?:系统集成|信息服务)|(?:系统集成|信息服务).{0,8}涉密",
    r"劳务派遣|(?:提供|开展|从事).{0,8}(?:检验|检测|认证)服务|检验检测服务",
    r"(?:医疗|诊疗|体检)服务|(?:提供|开展|从事).{0,6}诊疗",
    r"物业服务|保洁服务|绿化养护服务",
)


def _is_placeholder_subject(name: str) -> bool:
    from .ner_constraints import is_role_definition_fragment
    if is_role_definition_fragment(name):
        return True
    clean = T.squash(name)
    return bool(re.fullmatch(r"(?:本合同(?:中的?)?)?(?:" + _ROLE_PATTERN +
        r")(?:系指|是指|指|为)(?:采购人|中标人|中标供应商|投标人|采购代理机构|供应商)", clean))


def _canonical_subjects(subjects: Sequence, full_text: str) -> list[tuple[Any, str, list[str]]]:
    """Repair only an incomplete company suffix backed by a full original name.

    General prefix matching is unsafe: subsidiaries and similarly named legal
    entities remain separate, as do identical names in different party roles.
    """
    from .ner_constraints import GENERIC_ROLES
    canonical = []
    # Only an exact name with one unambiguous explicit role can supply the
    # role of a generic mention. Do not merge buyer/seller of the same name.
    explicit = {}
    for element in subjects:
        name = T.squash(element.value or "")
        role = (element.role or "other").lower()
        if not _is_placeholder_subject(element.value) and role not in GENERIC_ROLES:
            explicit.setdefault(name, set()).add(role)
    original = _subject_key(full_text)
    suffixes = ("有限责任公司", "股份有限公司", "有限公司")
    for element in subjects:
        element = copy(element)
        element.extra = dict(element.extra or {})
        name = T.squash(element.value or "")
        if _is_placeholder_subject(name):
            continue
        role = (element.role or "other").lower()
        original_role = role
        role_choices = explicit.get(name, set())
        if role in GENERIC_ROLES and len(role_choices) == 1:
            role = next(iter(role_choices))
            element = copy(element)
            element.role = role
            element.extra = dict(element.extra or {}, canonical_role_binding={
                "from": original_role, "to": role, "basis": "exact_name_unique_explicit_role"})
        matches = set()
        for other in subjects:
            complete = T.squash(other.value or "")
            if (other.role or "other").lower() != role or _subject_key(complete) not in original:
                continue
            for suffix in suffixes:
                if complete.endswith(suffix):
                    stem = complete[:-len(suffix)]
                    if len(stem) >= 4 and name in {stem + suffix[:i] for i in range(1, len(suffix))}:
                        matches.add(complete)
        resolved = next(iter(matches)) if len(matches) == 1 else name
        mention = {"element_id": element.element_id, "value": element.value,
            "role": original_role, "resolved_role": role, "resolved_name": resolved,
            "source": element.source, "span": element.span.to_dict() if element.span else None,
            "role_binding": element.extra.get("canonical_role_binding")}
        existing = next((row for row in canonical
                         if row[1] == resolved and (row[0].role or "other").lower() == role), None)
        if existing:
            existing[0].extra["canonical_subject_mentions"].append(mention)
            raw_alias = element.value
            if raw_alias != resolved and raw_alias not in existing[2]:
                existing[2].append(raw_alias)
        else:
            element.extra["canonical_subject_mentions"] = [mention]
            canonical.append((element, resolved, [element.value] if element.value != resolved else []))
    return canonical


def _subject_units(clauses: Sequence[Clause]) -> list[Clause]:
    """Keep claims within one sentence; a party header cannot cover the whole file."""
    units = []
    for clause in clauses:
        for text in re.split(r"[。！？；;，,\n]+", clause.text or ""):
            if text.strip():
                units.append(Clause(clause.clause_id, clause.label, clause.title, clause.level,
                    clause.heading, text, text, clause.block_ids, page_start=clause.page_start,
                    page_end=clause.page_end))
    return units


def _activity_requirement(unit: Clause, nodes: Sequence[SubjectNode]):
    text = unit.text
    # An enterprise name, legal citation, generic product test or certification
    # clause does not establish the regulated activity performed under a contract.
    for node in sorted(nodes, key=lambda value: len(value.name), reverse=True):
        for name in [node.name, *node.attrs.get("aliases", [])]:
            text = text.replace(name, "主体")
    text = re.sub(r"《[^》]+》", "", text)
    if not any(re.search(pattern, text) for pattern in _ACTIVITY_PATTERNS):
        return None
    if re.search(r"(?:法律法规|相关规定|适用法律|行业分类|案例示例|仅供参考)", text) and not re.search(
            r"(?:甲方|乙方|供应商|承包人|受托方|主体).{0,16}(?:提供|承担|负责|从事|经营|开展)", text):
        return None
    verdict = LK.qualification_verdict(text)
    # In particular, never recreate a requirement that the knowledge base has
    # suppressed because it was abolished or its applicability gate did not pass.
    return verdict.entry if verdict.should_report else None


def _performing_subjects(unit: Clause, nodes: Sequence[SubjectNode]) -> list[SubjectNode]:
    explicit = _nodes_in_clause_scope(nodes, [unit])
    actors = []
    for node in explicit:
        labels = [_raw_subject_name(node), *_ROLE_NAMES.get(node.role, ())]
        if any(re.search(re.escape(label) + r"(?:应|须|需|将|承诺|保证|必须|负责|承担|：|:|\s|由){0,6}"
                         r"(?:提供|承担|负责|实施|从事|经营|开展|承接|具备|取得|持有)", unit.text)
               for label in labels):
            actors.append(node)
    if actors:
        return actors
    providers = [node for node in explicit if node.role in PAYEE_ROLES]
    if len(providers) == 1:
        return providers
    providers = [node for node in nodes if node.role in PAYEE_ROLES]
    return providers if len(providers) == 1 else []


def build_subject_graph(contract: Contract) -> tuple[list[SubjectNode], list[SubjectEdge]]:
    """把当事人、资质要求、凭证线索、权限线索、信用线索连成一张小图。"""
    clauses = list(contract.document.clauses)
    full_text = "\n".join(clause.text for clause in clauses) or contract.document.full_text
    subjects = list(contract.elements.parties)

    nodes: list[SubjectNode] = []
    edges: list[SubjectEdge] = []
    seen_subjects: set[str] = set()

    for element, name, aliases in _canonical_subjects(subjects, full_text):
        if not name:
            continue
        kind, kind_conf = _classify_kind(name)
        role = (element.role or "other").lower()
        node_id = f"S:{role}:{name}"
        if node_id in seen_subjects:
            continue
        seen_subjects.add(node_id)
        nodes.append(
            SubjectNode(
                node_id=node_id,
                name=name,
                role=role,
                kind=kind,
                attrs={
                    "kind_conf": kind_conf,
                    "confidence": round(float(element.confidence or 0.6), 3),
                    "source": element.source,
                    "source_name": name,
                    "extracted_name": element.value,
                    "aliases": aliases,
                    "name_repair": "source_name_normalization_or_same_role_suffix_completion" if aliases else "",
                    "source_mentions": element.extra.get("canonical_subject_mentions", []),
                },
            )
        )
        edges.append(
            SubjectEdge(
                source=node_id,
                target=f"K:{kind}",
                kind="has_type",
                weight=kind_conf,
                detail=f"名称后缀判定为{kind}",
                path=f"「{name}」-has_type→「{kind}」",
            )
        )
    edge_keys = set()
    for unit in _subject_units(clauses):
        entry = _activity_requirement(unit, nodes)
        if entry is not None:
            for node in _performing_subjects(unit, nodes):
                key = (node.node_id, "requires", entry.qualification)
                if key not in edge_keys:
                    edge_keys.add(key)
                    node.attrs.setdefault("qualification_scope", []).append({"clause_id": unit.clause_id,
                        "qualification": entry.qualification, "activity_text": unit.text,
                        "basis_law": entry.basis_law, "basis_article": entry.basis_article,
                        "source_url": entry.source_url})
                    edges.append(SubjectEdge(node.node_id, "Q:" + entry.qualification, "requires", 0.7,
                        entry.citation_text(), f"「{node.name}」-requires→「{entry.qualification}」（{entry.citation_text()}；条款{unit.clause_id}）"))
        claim_nodes = _nodes_in_clause_scope(nodes, [unit])
        if len(claim_nodes) > 1 and not re.search(r"双方|各方|甲乙双方", unit.text):
            claim_nodes = []
        for node in claim_nodes:
            mentions = []
            if any(token in unit.text for token in LICENSE_TOKENS):
                mentions.append(("claimed_holds", "Q:凭证", "资质凭证（合同表述，非真实性核验）"))
            if any(token in unit.text for token in AUTH_TOKENS):
                mentions.append(("claimed_authorized_by", "A:授权", "签约权限（合同表述，非实际授权书）"))
            mentions.extend(("credit_mention", "C:" + token, token) for token in CREDIT_TOKENS if token in unit.text)
            for kind, target, label in mentions:
                key = (node.node_id, kind, target)
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                edges.append(SubjectEdge(node.node_id, target, kind, 0.6,
                    f"关联条款{unit.clause_id}：{unit.text[:160]}", f"「{node.name}」-{kind}→「{label}」（条款{unit.clause_id}）"))
    bindings = _subject_identity_bindings(contract, nodes)
    for node in nodes:
        binding = bindings[node.node_id]
        node.attrs["identity_binding"] = binding
        node.attrs["subject_identifiers"] = binding["identifiers"]
        # Identity-derived hints must not borrow another subject's identifier.
        for identity in binding["identifiers"] if binding["status"] == "bound" else []:
            digits = re.sub(r"\D", "", identity)
            if len(digits) == 18:
                province = PROVINCE_CODE.get(digits[:2])
                if province and province not in (full_text or ""):
                    edges.append(
                        SubjectEdge(
                            source=node.node_id,
                            target=f"R:{province}",
                            kind="located_in",
                            weight=0.5,
                            detail=f"统一社会信用代码前缀 {digits[:2]} 对应{province}，正文未出现该地名",
                            path=f"「{node.name}」-located_in→「{province}」",
                        )
                    )

    return nodes, edges


# ---------------------------------------------------------------- 核验


def _evidence_availability(contract: Contract, context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Availability is a declaration about a file, never a verified fact."""
    if context is None:
        stats = getattr(contract.document, "stats", {})
        context = stats.get("contract_package", {}) if isinstance(stats, dict) else {}
    if not isinstance(context, dict):
        return []
    raw = context.get("missing_documents", [])
    if not isinstance(raw, list):
        return []
    def strings(value):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip())) if isinstance(value, (list, tuple)) else []
    gaps = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        availability = str(row.get("availability") or "missing")
        document_id = str(row.get("document_id") or stable_id("missing_document", row.get("title", ""), length=20))
        private = availability == "unavailable_private"
        blocked_scope = row.get("blocked_fact_scope")
        if isinstance(blocked_scope, (list, tuple)):
            blocked_scope = "；".join(strings(blocked_scope))
        gaps.append({"document_id": document_id, "title": str(row.get("title") or document_id),
            "availability": availability, "reason_source": str(row.get("reason_source") or "not_supplied"),
            "reason": str(row.get("reason") or ("用户说明该材料涉及隐私、无法提供" if private else "材料尚未提供或尚未核验")),
            "fact_types": [kind for kind in strings(row.get("fact_types"))
                           if kind in ("registration", "license", "authorization", "credit")],
            "subject_names": strings(row.get("subject_names")),
            "subject_roles": strings(row.get("subject_roles")), "clause_ids": strings(row.get("clause_ids")),
            "blocked_fact_scope": str(blocked_scope or "缺少该材料支持的事实继续待核验"),
            "status": "pending", "fact_status": "unknown", "independently_verified": False,
            "auto_retry": False if private else True, "retry_trigger": "new_evidence" if private else "available_source",
            "legal_truth_established": False})
    return gaps


def _gap_applies(gap: dict[str, Any], node: SubjectNode, fact_type: str = "",
                 clause_ids: Sequence[str] = ()) -> bool:
    # No name/role selector means an unbound document gap, not a file missing
    # for every party. Scope must be declared explicitly by the package importer.
    names, roles = gap.get("subject_names", []), gap.get("subject_roles", [])
    if not (names or roles):
        return False
    if names and _subject_key(node.name) not in {_subject_key(name) for name in names}:
        return False
    if roles and node.role not in roles:
        return False
    if fact_type and fact_type not in gap.get("fact_types", []):
        return False
    scoped = set(gap.get("clause_ids", []))
    return not scoped or bool(clause_ids and set(clause_ids) <= scoped)


def _apply_evidence_gaps(results: Sequence[dict[str, Any]], gaps: Sequence[dict[str, Any]],
                         nodes: Sequence[SubjectNode]) -> None:
    node_map = {node.node_id: node for node in nodes}
    for result in results:
        node = node_map.get(result.get("subject_node_id"))
        malformed = {"invalid_verification_request", "duplicate_verification_request_id", "request_clause_scope_missing",
                     "request_clause_not_found", "request_fact_type_unresolved"}
        if (node is None or result.get("status") == "completed" or not result.get("fact_type")
                or malformed.intersection(result.get("reasons", []))):
            continue
        matched = [dict(gap) for gap in gaps if gap["availability"] == "unavailable_private"
            and _gap_applies(gap, node, result.get("fact_type", ""), result.get("clause_ids", []))]
        if not matched:
            continue
        result["unavailable_evidence"] = matched
        result["reasons"] = list(dict.fromkeys([*result.get("reasons", []), "unavailable_private"]))
        result.update(reason_code="unavailable_private", reason_source="user_statement"
            if all(gap["reason_source"] == "user_statement" for gap in matched) else "declared_availability",
            auto_retry=False, retry_trigger="new_evidence")


def _resolve_verification_requests(
    requests: Sequence[dict[str, Any]], nodes: Sequence[SubjectNode],
    clauses: Sequence[Clause], fact_checks: Sequence[dict[str, Any]],
    *, contract_id: str = "",
) -> list[dict[str, Any]]:
    """Consume statute requests only when subject, scope and evidence are bound."""
    from .subject_facts import FACT_TYPES, _scope_applicability
    clause_map = {clause.clause_id: clause for clause in clauses}
    checks_by_node = {check["subject_node_id"]: check for check in fact_checks}
    explicit_counts: dict[str, int] = {}
    for request in requests:
        if isinstance(request, dict) and request.get("request_id"):
            key = str(request["request_id"])
            explicit_counts[key] = explicit_counts.get(key, 0) + 1
    derived_counts: dict[str, int] = {}
    results = []
    for index, raw in enumerate(requests):
        request = dict(raw) if isinstance(raw, dict) else {}
        raw_clause_ids = request.get("clause_ids", [])
        clause_ids = list(dict.fromkeys(str(value) for value in raw_clause_ids)) if isinstance(raw_clause_ids, (list, tuple)) else []
        canonical = {key: request.get(key) for key in ("candidate_id", "article_uid", "risk_code",
            "subject_name", "subject_id", "subject_node_id", "fact_type")}
        canonical["clause_ids"] = sorted(clause_ids)
        request_id = str(request.get("request_id") or stable_id("subject_request",
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str), length=20))
        if not request.get("request_id"):
            occurrence = derived_counts.get(request_id, 0)
            derived_counts[request_id] = occurrence + 1
            if occurrence:
                request_id = stable_id("subject_request_duplicate", request_id, occurrence, length=20)
        result = {"request_id": request_id, "candidate_id": str(request.get("candidate_id") or ""),
            "clause_ids": clause_ids, "article_uid": str(request.get("article_uid") or ""),
            "risk_code": str(request.get("risk_code") or ""),
            "subject_name": "", "subject_node_id": "", "subject_id": "", "fact_type": "",
            "status": "pending", "fact_status": "unknown", "reason": "", "fact_ids": [], "records": [],
            "independently_verified": False}
        reasons = []
        if not isinstance(raw, dict):
            reasons.append("invalid_verification_request")
        if explicit_counts.get(request_id, 0) > 1:
            reasons.append("duplicate_verification_request_id")
        if not clause_ids:
            reasons.append("request_clause_scope_missing")
        if any(cid not in clause_map for cid in clause_ids):
            reasons.append("request_clause_not_found")
        scope = [clause_map[cid] for cid in clause_ids if cid in clause_map]
        scope_text = "\n".join(clause.text for clause in scope)
        fact_type = str(request.get("fact_type") or "")
        if not fact_type:
            fact_type = {"need_license": "license", "need_authorization": "authorization"}.get(result["risk_code"], "")
            if result["risk_code"] == "illegal_terms" and any(token in scope_text for token in CREDIT_TOKENS):
                fact_type = "credit"
            if result["risk_code"] == "subject_unspecified" and re.search(_IDENTITY_LABEL + r"|营业执照|登记|注册", scope_text):
                fact_type = "registration"
        if fact_type not in FACT_TYPES:
            reasons.append("request_fact_type_unresolved")
            fact_type = ""
        result["fact_type"] = fact_type
        possible = _nodes_in_clause_scope(nodes, scope)
        if request.get("subject_node_id"):
            possible = [node for node in possible if node.node_id == request["subject_node_id"]]
        if request.get("subject_name"):
            possible = [node for node in possible if _subject_key(_raw_subject_name(node)) == _subject_key(request["subject_name"])]
        if request.get("subject_id"):
            selected_id = str(request["subject_id"]).strip().upper()
            possible = [node for node in possible if node.attrs["identity_binding"]["status"] == "bound"
                        and node.attrs["identity_binding"]["identifiers"] == [selected_id]]
        result["candidate_subject_node_ids"] = [node.node_id for node in possible]
        if len(possible) != 1:
            reasons.append("request_subject_ambiguous" if possible else "request_subject_not_bound_in_clause_scope")
        else:
            node = possible[0]
            checked = checks_by_node[node.node_id]
            binding = checked["identity_binding"]
            result.update(subject_name=checked["subject_name"], subject_node_id=node.node_id,
                subject_role=node.role,
                subject_id=binding["identifiers"][0] if binding["status"] == "bound" else "",
                subject_identifiers=binding["identifiers"], identity_binding=binding)
            if binding["status"] != "bound":
                reasons.append(binding["reason"])
            relevant = [record for record in checked["records"] if record["fact_type"] == fact_type]
            result["records"] = relevant
            status = checked["checks"].get(fact_type, "unknown")
            if fact_type in ("license", "authorization"):
                # Fact validity and applicability answer different questions.
                # A food license cannot discharge a construction-license request
                # merely because both are filed under the same subject/type.
                scoped, applicable, scope_reasons = [], [], []
                for record in relevant:
                    reason = ""
                    review = record.get("applicability")
                    if review is None:
                        reason = "evidence_scope_applicability_missing"
                    else:
                        try:
                            review = _scope_applicability(review)
                        except (TypeError, ValueError):
                            reason = "evidence_scope_applicability_invalid"
                        else:
                            if review["contract_id"] != contract_id:
                                reason = "evidence_scope_different_contract"
                            elif not clause_ids or not set(clause_ids) <= set(review["clause_ids"]):
                                reason = "evidence_scope_clause_not_bound"
                            elif review["verdict"] != "applicable":
                                reason = ("evidence_scope_not_applicable" if review["verdict"] == "not_applicable"
                                          else "evidence_scope_needs_manual")
                    row = dict(record, request_scope_status="pending" if reason else "human_attested_applicable",
                               request_scope_reason=reason)
                    scoped.append(row)
                    if reason:
                        scope_reasons.append(reason)
                    else:
                        applicable.append(row)
                result["records"] = scoped
                result["scope_status"] = "human_attested_applicable" if applicable else "pending"
                result["scope_review_reasons"] = list(dict.fromkeys(scope_reasons))
                result["scope_independently_verified"] = False
                if relevant and not applicable:
                    reasons.extend(scope_reasons)
                relevant = applicable
                values = {record["effective_status"] for record in relevant} - {"unknown"}
                status = ("conflicting_evidence" if len(values) > 1 else next(iter(values)) if values else "unknown")
            result["fact_status"] = status
            if status in ("unknown", "conflicting_evidence"):
                details = sorted({reason for record in relevant for reason in record.get("review_reasons", [])})
                reasons.append("conflicting_evidence" if status == "conflicting_evidence" else
                               "subject_evidence_unresolved:" + ",".join(details) if details else "no_effective_reviewed_evidence")
            else:
                result["fact_ids"] = [record["fact_id"] for record in relevant if record["effective_status"] == status]
        result["reasons"] = list(dict.fromkeys(reasons))
        result["reason"] = result["reasons"][0] if reasons else "bound_reviewed_local_evidence"
        if not reasons:
            result["status"] = "completed"
        results.append(result)
    return results


def check_subjects(contract: Contract, config: Config | None = None, *,
                   focus_clause_ids: Sequence[str] = (),
                   verification_requests: Sequence[dict[str, Any]] = (),
                   package_context: dict[str, Any] | None = None) -> SubjectResult:
    """节点③ 主入口：核验资格、权限与信用，输出候选 + 图。"""
    config = config or Config()
    focus = set(focus_clause_ids)
    clauses = sorted(contract.document.clauses, key=lambda c: c.clause_id not in focus)
    nodes, edges = build_subject_graph(contract)
    result = SubjectResult(nodes=nodes, edges=edges)
    evidence_gaps = _evidence_availability(contract, package_context)

    full_text = "\n".join(clause.text for clause in clauses) or contract.document.full_text or ""
    has_license_token = any(token in full_text for token in LICENSE_TOKENS)
    has_auth_token = any(token in full_text for token in AUTH_TOKENS)
    credit_hits = [token for token in CREDIT_TOKENS if token in full_text]

    org_nodes = [node for node in nodes if node.kind in ("企业", "事业单位", "机关")]
    person_nodes = [node for node in nodes if node.kind == "个人"]

    # 1) 资质：有 requires 边、但没有任何 holds 线索
    requires_edges = [edge for edge in edges if edge.kind == "requires"]
    for node in nodes:
        required = [edge for edge in requires_edges if edge.source == node.node_id]
        if not required or any(edge.source == node.node_id and edge.kind == "claimed_holds" for edge in edges):
            continue
        license_names = sorted({edge.target.replace("Q:", "") for edge in required})
        scopes = node.attrs.get("qualification_scope", [])
        scope_ids = {row["clause_id"] for row in scopes}
        clause = next((row for row in clauses if row.clause_id in scope_ids), None)
        verdict = LK.qualification_verdict(" ".join(row["activity_text"] for row in scopes))
        basis_entry = verdict.entry if verdict.should_report else None
        candidate = _make(
            "need_license",
            f"{node.name}承担的合同活动涉及{('、'.join(license_names[:2]))}，但未见与该主体绑定的资质或许可表述，"
            "需结合实际业务及来源材料核验适用资质，不能据此认定其没有资质",
            clause=clause,
            evidence_text=(clause.text[:60] if clause else license_names[0]),
            path=" ; ".join(edge.path for edge in required[:2]),
            confidence=0.58,
            extra={
                "issue_code": "need_license",
                "licenses": license_names,
                "subject_name": node.name, "subject_node_id": node.node_id, "subject_role": node.role,
                "basis_law": (basis_entry.basis_law if basis_entry else ""),
                "basis_article": (basis_entry.basis_article if basis_entry else ""),
                "source_url": (basis_entry.source_url if basis_entry else ""),
            },
        )
        if basis_entry is not None:
            # 把法规依据作为一条独立证据挂上去：引用可点开官方页面，答辩时能指着讲
            candidate.evidence.append(
                Evidence(
                    kind="statute_basis",
                    text=basis_entry.citation_text(),
                    strength=0.8,
                    source=basis_entry.source_url,
                )
            )
        result.candidates.append(candidate)

    # 2) 权限：只在正文确实涉及代理/授权/签署语境时，才检查法定代表人线索是否缺失。
    # 原实现"有机构主体就查"，对每份模板合同都报（实测 fp=8），改为先看语境。
    for node in org_nodes:
        scoped = [clause for clause in _subject_units(clauses)
                  if _nodes_in_clause_scope(nodes, [clause]) == [node]]
        auth_context = any(any(token in clause.text for token in AUTH_CONTEXT)
                           and any(token in clause.text for token in ("签约", "签署", "签订", "签字", "签章", "盖章"))
                           for clause in scoped)
        if not auth_context or any(edge.source == node.node_id and edge.kind == "claimed_authorized_by" for edge in edges):
            continue
        clause = _first_clause_with(scoped, "签章", "签字", "盖章", "代表", "签署", "授权", "委托")
        result.candidates.append(
            _make(
                "need_authorization",
                f"{node.name}的签约权限无交代：关联表述中未见法定代表人或授权委托线索，"
                "需核验签署人权限",
                clause=clause,
                evidence_text=(clause.text[:60] if clause else node.name),
                path=f"「{node.name}」-authorized_by→「（空）」",
                confidence=0.58,
                extra={"issue_code": "need_authorization", "subject_name": node.name,
                       "subject_node_id": node.node_id, "subject_role": node.role},
            )
        )

    # 3) 角色与主体类型冲突：机关出现在收款方，或个人作为大额货物/服务供应方
    amounts = sorted(
        {float(a.normalized) for a in contract.elements.amounts if _is_number(a.normalized)},
        reverse=True,
    )
    top_amount = amounts[0] if amounts else 0.0
    for node in nodes:
        if node.kind == "机关" and node.role in PAYEE_ROLES:
            clause = _first_clause_with(clauses, "价款", "支付", "收款", "结算")
            result.candidates.append(
                _make(
                    "subject_unspecified",
                    f"{node.name}为机关，却在合同中处于收款方角色，需核实主体身份与收款账户",
                    clause=clause,
                    evidence_text=node.name,
                    path=f"「{node.name}」-has_type→「机关」-role→「{node.role}」",
                    confidence=0.55,
                    extra={"issue_code": "subject_unspecified"},
                )
            )
        if node.kind == "个人" and node.role in PAYEE_ROLES and top_amount >= 100_000:
            clause = _first_clause_with(clauses, "价款", "标的", "服务", "货物")
            result.candidates.append(
                _make(
                    "need_license",
                    f"{node.name}为自然人，作为标的额 {top_amount:,.2f} 元的供应方，"
                    "需结合具体标的核实经营资格与履约能力，不能仅由金额推定其无权签约",
                    clause=clause,
                    evidence_text=node.name,
                    path=f"「{node.name}」-has_type→「个人」-amount→{top_amount:.2f}",
                    confidence=0.6,
                    extra={"issue_code": "need_license", "amount": top_amount},
                )
            )

    # 4) 信用：正文出现失信类表述
    for node in nodes:
        node_credit = [edge for edge in edges if edge.source == node.node_id and edge.kind == "credit_mention"]
        if not node_credit:
            continue
        tokens = [edge.target.removeprefix("C:") for edge in node_credit]
        scoped = [clause for clause in _subject_units(clauses)
                  if node in _nodes_in_clause_scope(nodes, [clause])]
        clause = _first_clause_with(scoped, *tokens)
        result.candidates.append(
            _make(
                "illegal_terms",
                f"{node.name}的关联表述出现「{'、'.join(tokens[:2])}」，需核对是查询要求、否定表述还是实际信用记录，不能直接认定失信",
                clause=clause,
                evidence_text=(clause.text[:60] if clause else tokens[0]),
                path=" ; ".join(edge.path for edge in node_credit)[:200],
                confidence=0.7,
                extra={"issue_code": "illegal_terms", "credit": tokens[:3], "subject_name": node.name,
                       "subject_node_id": node.node_id, "subject_role": node.role},
            )
        )

    # 5) 主体不明：一个当事人都没抽到，但合同里出现了"甲方/乙方"
    if not nodes and ("甲方" in full_text or "乙方" in full_text):
        clause = clauses[0] if clauses else None
        result.candidates.append(
            _make(
                "subject_unspecified",
                "合同出现甲方/乙方表述，但未能确定任何一方主体名称，需人工确认当事人",
                clause=clause,
                evidence_text=(clause.text[:60] if clause else "甲方/乙方"),
                path="「（未识别）」-has_type→「未知」",
                confidence=0.6,
                extra={"issue_code": "subject_unspecified"},
            )
        )

    # 外部事实只来自明确导入并附核验来源的证据，不来自合同中的自述。
    from . import subject_facts as SF
    records, fact_degraded = SF.load_facts(getattr(config, "subject_facts_path", ""))
    result.degraded.extend(fact_degraded)
    fact_checks = []
    for node in nodes:
        binding = node.attrs["identity_binding"]
        checked = SF.resolve_subject(_raw_subject_name(node), records,
            subject_ids=binding["identifiers"], identity_ambiguous=binding["status"] == "ambiguous",
            contract_id=contract.contract_id,
            max_age_days=int(getattr(config, "subject_facts_max_age_days", 30)))
        checked.update(subject_node_id=node.node_id, subject_role=node.role,
            clause_ids=[clause.clause_id for clause in clauses if node in _nodes_in_clause_scope(nodes, [clause])],
            identity_binding=binding)
        checked["evidence_gaps"] = [dict(gap) for gap in evidence_gaps
            if gap.get("fact_types") and _gap_applies(gap, node, clause_ids=gap.get("clause_ids", []))
            and (not gap.get("clause_ids") or set(gap["clause_ids"]) <= {clause.clause_id for clause in clauses})]
        fact_checks.append(checked)
        for record in checked["records"]:
            status = record["effective_status"]
            path = f"「{node.name}」-reviewed_evidence→「{record['fact_type']}:{status}」"
            result.edges.append(SubjectEdge(node.node_id, "F:" + record["fact_id"],
                "reviewed_fact", 0.85 if status != "unknown" else 0.3,
                record["source"]["url"], path))
            negative = status in ("invalid", "inactive", "adverse")
            if not negative:
                continue
            code = {"license": "need_license", "authorization": "need_authorization",
                    "credit": "illegal_terms", "registration": "subject_unspecified"}[record["fact_type"]]
            # A same-named counterparty's first mention is not this subject's
            # source anchor. Prefer its explicitly bound identity field.
            scoped_ids = set(checked["clause_ids"])
            identity_ids = {entry.get("clause_id") for entry in binding.get("evidence", [])
                if entry.get("subject_id") == record.get("subject_id")}
            clause = next((entry for entry in clauses
                if entry.clause_id in scoped_ids and entry.clause_id in identity_ids), None)
            if clause is None:
                scoped = [entry for entry in clauses if entry.clause_id in scoped_ids]
                clause = _first_clause_with(scoped, _raw_subject_name(node)) or (scoped[0] if scoped else None)
            fact_key = stable_id("reviewed_subject_fact", node.node_id, record.get("subject_id", ""),
                record["fact_id"], record["fact_type"], clause.clause_id if clause else "", length=32)
            candidate = _make(code,
                f"{node.name}的{record['fact_type']}导入核验记录为{status}；"
                "请核对来源、适用范围与时效后人工确认，不视为系统直接查询官方数据的结论",
                clause=clause, evidence_text=node.name, path=path, confidence=0.8,
                extra={"issue_code": code, "subject_name": node.name,
                       "subject_node_id": node.node_id, "subject_role": node.role,
                       "subject_id": record.get("subject_id", ""), "fact_key": fact_key,
                       "fact_id": record["fact_id"], "fact_type": record["fact_type"],
                       "verification_status": "supplied_record_claim",
                       "source_actor_kind": record.get("source_actor_kind", "legacy_unspecified"),
                       "independently_verified": False})
            candidate.candidate_id = stable_id("rc_subject_fact", fact_key, length=24)
            candidate.evidence.append(Evidence("reviewed_subject_fact",
                f"{record['source']['publisher']}；核验日期{record['source']['verified_at']}；"
                f"范围{record['scope'] or '未注明'}；记录{record['fact_id']}",
                0.85, source=record["source"]["url"], extra=record))
            result.candidates.append(candidate)

    verification_results = _resolve_verification_requests(verification_requests, nodes, clauses, fact_checks,
        contract_id=contract.contract_id)
    _apply_evidence_gaps(verification_results, evidence_gaps, nodes)
    pending_requests = [request["request_id"] for request in verification_results if request["status"] != "completed"]
    for request in verification_results:
        if request["status"] != "completed":
            result.degraded.append(f"subject_verification_pending:{request['request_id']}:{request['reason']}")

    result.stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "requires": len(requires_edges),
        "has_license_token": has_license_token,
        "has_auth_token": has_auth_token,
        "credit_hits": credit_hits,
        "kinds": sorted({node.kind for node in nodes}),
        "candidates": len(result.candidates),
        "focus_clause_ids": sorted(focus),
        "verification_requests": list(verification_requests),
        "verification_results": verification_results,
        "verification_complete": not pending_requests,
        "pending_verification_request_ids": pending_requests,
        "fact_checks": fact_checks,
        "excluded_subject_mentions": [{"source_name": element.value, "role": element.role,
            "element_id": element.element_id, "source": element.source,
            "span": element.span.to_dict() if element.span else None,
            "reason": "role_definition_without_entity_name"} for element in contract.elements.parties
            if _is_placeholder_subject(element.value or "")],
        "evidence_gaps": evidence_gaps,
        "evidence_availability_is_user_declaration": any(gap["reason_source"] == "user_statement" for gap in evidence_gaps),
        "fact_source": "reviewed_local_evidence" if records else "not_configured_or_empty",
        "contract_claims_independently_verified": False,
    }
    if not nodes:
        result.degraded.append("no_subject_extracted")
    return result


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def run_subject_check(contract: Contract, config: Config | None = None) -> list[RiskCandidate]:
    """与 rules.run_rules 同形的入口，便于在引擎里平铺调用。"""
    return check_subjects(contract, config).candidates
