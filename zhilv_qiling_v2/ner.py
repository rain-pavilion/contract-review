"""规则和可训练NER双通道抽取，按真实原文位置、类型与角色仲裁。

同值的不同位置保留，通道在同一位置且角色兼容时才可相互佐证。
金额、日期和边界约束产生可重放回执；抽取冲突不等于合同违法。
独立人工实体级F1尚需人工金标验收。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import text as T
from .config import Config
from .learned import NerTagger
from .models import (
    Clause,
    ContractElements,
    DocumentParse,
    Element,
    ElementSignal,
    SpanRef,
    stable_id,
)

# ---------------------------------------------------------------------------
# 词表与正则
# ---------------------------------------------------------------------------

PARTY_ROLES: tuple[str, ...] = (
    "甲方", "乙方", "丙方", "丁方", "委托方", "受托方", "采购人", "供应商",
    "发包人", "承包人", "出卖人", "买受人", "出租人", "承租人", "寄存人", "保管人",
    "贷款人", "借款人", "供方", "需方", "买方", "卖方", "转让方", "受让方",
)

_PARTY_LINE_RE = re.compile(
    rf"(?:{'|'.join(PARTY_ROLES)})"
    r"(?:[（(][^）)]{0,12}[）)])?\s*(?:（以下简称[^）)]*）)?\s*[：:]\s*"
    r"([^\n，,；;。]{2,40})"
)

# 人名/机构名截断标记：出现这些字说明"当事人名"已经结束，后面是正文
_PERSON_STOP = frozenset("将于系为，,、（( ：:应向根据鉴于负责按照签章盖章地址电话联系人以下简称")
_PARTY_TAIL_CUT: tuple[str, ...] = (
    "将", "于", "应", "须", "向", "根据", "鉴于", "负责", "按照", "同意", "有权",
    "签章", "签字", "盖章", "地址", "电话", "联系人", "以下简称", "约定", "拟",
    "保管", "交付", "数量", "金额", "价款", "履行", "期限", "提供", "完成",
)


def _party_pattern(role: str) -> re.Pattern[str]:
    """按角色逐个建正则：同句里"甲方：A 将物交由 乙方：B"两方都要能抓到。"""
    return re.compile(
        role
        + r"(?:[（(][^）)]{0,12}[）)])?\s*(?:（以下简称[^）)]*）)?\s*[：:]\s*"
        + r"([^\n，,；;。]{2,40})"
    )


_PARTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (role, _party_pattern(role)) for role in PARTY_ROLES
)

# 日期模式（extract_dates / extract_durations 共用，后者要跳过日期内部的"年/月/日"）
# 分隔符里要含下划线且允许出现在"年/月"标志两侧：真实政采合同用 Word 填空下划线写成
# "2026年_10月29日"，不放开这些位置整片漏抽（真实合同测试集里签订日就漏在这一条上）。
_DATE_PATTERN = re.compile(
    r"(?:\d{4}\s*[-/.年]\s*[_＿\s]*\d{1,2}\s*[-/.月]\s*[_＿\s]*\d{1,2}\s*[_＿\s]*[日号]?"
    r"|[〇零一二三四五六七八九]{2,4}\s*年\s*[_＿\s]*[一二三四五六七八九十]{1,3}\s*月"
    r"(?:[_＿\s]*[一二三四五六七八九十〇零]{1,3}\s*[日号])?)"
)


def _trim_party(raw: str) -> tuple[str, float]:
    """把"当事人：XXX将标的物交由……"截成真正的当事人名称，并给出置信度。

    这是规则通道最容易虚高的一处：不截断的话 `source` 字段会带上一整句正文。
    """
    cleaned = raw.strip()
    if not cleaned:
        return "", 0.0
    company = _COMPANY_RE.match(cleaned)
    if company:
        return company.group(0), 0.9
    for length in (4, 3, 2):
        head = cleaned[:length]
        if len(head) < length or not all("\u4e00" <= ch <= "\u9fff" for ch in head):
            continue
        if len(cleaned) == length or cleaned[length] in _PERSON_STOP:
            return head, 0.72
    cut = len(cleaned)
    for keyword in _PARTY_TAIL_CUT:
        index = cleaned.find(keyword)
        if 1 <= index < cut:
            cut = index
    # 值里若又出现"乙方（""丙方："，说明抓过头了，在第二个角色标签处截断
    for role, _ in _PARTY_PATTERNS:
        for sep in ("：", ":", "（", "("):
            index = cleaned.find(role + sep)
            if 1 <= index < cut:
                cut = index
    trimmed = cleaned[:cut].strip()
    return trimmed, (0.72 if 2 <= len(trimmed) <= 40 else 0.0)
_COMPANY_RE = re.compile(
    r"[\u4e00-\u9fff（）()A-Za-z0-9·\-]{2,40}?"
    r"(?:有限责任公司|股份有限公司|有限公司|集团公司|公司|中心|研究院|设计院|大学|学院|"
    r"学校|医院|银行|事务所|委员会|管理局|财政局|发展中心|集团|厂|站|所|社|部|局|厅|院)"
)
_PERSON_RE = re.compile(r"[\u4e00-\u9fff]{2,4}(?=[，,。；;\s]|$)")

CREDIT_CODE_RE = re.compile(r"\b([0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10})\b")
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
CONTRACT_NO_RE = re.compile(r"[（(]?\d{4}[）)]?[\-－]?[A-Za-z0-9]{0,6}[\-－]?\d{3,}")
AMOUNT_RE = re.compile(
    r"(?:人民币|RMB|￥|¥)?\s*[0-9][0-9,，]*(?:\.[0-9]{1,2})?\s*(?:万元|万圆|亿元|元|圆)"
)
AMOUNT_CN_RE = re.compile(
    r"(?:人民币)?[壹贰叁肆伍陆柒捌玖拾佰仟万亿零〇]{2,}(?:元整|元|圆整|圆)"
)
DURATION_RE = re.compile(r"(\d{1,4}|[一二三四五六七八九十百]{1,4})\s*(?:个)?(?:日历天|工作日|天|日|个月|月|年)")
SUBJECT_KEYS: tuple[str, ...] = (
    "项目名称", "标的名称", "货物名称", "服务名称", "工程名称", "合同名称",
    "标的", "采购内容", "服务内容", "项目内容", "品名",
    "保管物", "保管标的", "租赁物", "委托事项", "服务项目", "工程内容", "采购标的",
)

_ROLE_HINTS: dict[str, str] = {
    "甲方": "buyer", "采购人": "buyer", "委托方": "buyer", "发包人": "buyer",
    "买受人": "buyer", "需方": "buyer", "买方": "buyer", "出租人": "buyer",
    "乙方": "seller", "供应商": "seller", "受托方": "seller", "承包人": "seller",
    "出卖人": "seller", "供货方": "seller", "供方": "seller", "卖方": "seller",
    "丙方": "third", "丁方": "third", "保管人": "custodian", "寄存人": "depositor",
}

_AMOUNT_ROLE_HINTS: tuple[tuple[str, str], ...] = (
    ("合同总价", "total"), ("合同金额", "total"), ("总价", "total"), ("总金额", "total"),
    ("价款", "total"), ("单价", "unit"), ("定金", "deposit"), ("预付款", "prepay"),
    ("违约金", "penalty"), ("保证金", "bond"), ("保管费", "fee"), ("服务费", "fee"),
    ("尾款", "balance"), ("进度款", "progress"), ("首付款", "prepay"),
)

_DATE_ROLE_HINTS: tuple[tuple[str, str], ...] = (
    ("签订", "sign"), ("签署", "sign"), ("生效", "effective"), ("起始", "start"),
    ("开始", "start"), ("届满", "end"), ("终止", "end"), ("结束", "end"),
    ("交付", "delivery"), ("验收", "acceptance"), ("付款", "payment"),
)


# ---------------------------------------------------------------------------
# 规则通道
# ---------------------------------------------------------------------------

@dataclass
class RuleHit:
    kind: str
    value: str
    normalized: str
    role: str
    confidence: float
    clause_id: str | None
    span: SpanRef
    extra: dict[str, Any] = field(default_factory=dict)


def _find_clause_for_block(document: DocumentParse, block_id: str) -> str | None:
    for clause in document.clauses:
        if block_id in clause.block_ids:
            return clause.clause_id
    return None


def _nearest_role(text: str, position: int, hints: Sequence[tuple[str, str]], default: str) -> str:
    best: tuple[int, str] = (10**9, default)
    for keyword, role in hints:
        index = text.rfind(keyword, 0, position)
        if index >= 0 and position - index < best[0]:
            best = (position - index, role)
    return best[1]


def _role_from_block(text: str, hints: Sequence[tuple[str, str]], default: str = "other") -> str:
    """块内任意位置出现角色关键词时的兜底判定。

    合同字段表里"签订日期"与"2026年11月4日"经常落在两个相邻单元格/块里，
    单看日期所在块的**前方**文本找不到线索，于是退化成 `other`。
    """
    for keyword, role in hints:
        if keyword in text:
            return role
    return default


def extract_parties(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for block in document.blocks:
        text = block.text
        for role_key, pattern in _PARTY_PATTERNS:
            for match in pattern.finditer(text):
                cleaned, confidence = _trim_party(match.group(1))
                if len(cleaned) < 2 or confidence <= 0:
                    continue
                # 排除"甲方"后面直接跟动词的情况（"甲方应当……"）
                if cleaned.endswith(("应", "须", "负责", "按照", "有权", "同意")):
                    continue
                start = match.start(1)
                hits.append(
                    RuleHit(
                        kind="party",
                        value=cleaned,
                        normalized=T.normalize_person(cleaned),
                        role=_ROLE_HINTS.get(role_key, "unknown"),
                        confidence=confidence,
                        clause_id=_find_clause_for_block(document, block.block_id),
                        span=document.locate(block.block_id, start, start + len(cleaned)),
                        extra={"role_label": role_key},
                    )
                )
        # 没有"甲方："前缀，但整行就是一个机构名（当事人栏表格）
        if block.kind == "table_row":
            cells = block.metadata.get("cells") or []
            row_text = "".join(cells)
            for cell in cells:
                name = cell.strip()
                if not (_COMPANY_RE.fullmatch(name) and 4 <= len(name) <= 40):
                    continue
                prev_role = next((r for r in PARTY_ROLES if r in row_text), "")
                offset = block.text.find(name)
                if offset < 0:
                    offset = 0
                hits.append(
                    RuleHit(
                        kind="party",
                        value=name,
                        normalized=T.normalize_person(name),
                        role=_ROLE_HINTS.get(prev_role, "unknown"),
                        confidence=0.66,
                        clause_id=_find_clause_for_block(document, block.block_id),
                        span=document.locate(block.block_id, offset, offset + len(name)),
                        extra={"from": "table_cell"},
                    )
                )
    return hits


def extract_amounts(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for block in document.blocks:
        text = block.text
        for pattern, source in ((AMOUNT_RE, "digit"), (AMOUNT_CN_RE, "cn")):
            for match in pattern.finditer(text):
                raw = match.group(0).strip()
                value = T.money_to_float(raw)
                if value is None:
                    continue
                negative = match.start() > 0 and text[match.start() - 1] in "-－负"
                if negative:
                    value = -value
                role = _nearest_role(text, match.start(), _AMOUNT_ROLE_HINTS, "other")
                confidence = 0.92 if source == "digit" else 0.8
                if role == "other":
                    confidence -= 0.1
                hits.append(
                    RuleHit(
                        kind="amount",
                        value=raw,
                        normalized=f"{value:.2f}",
                        role=role,
                        confidence=confidence,
                        clause_id=_find_clause_for_block(document, block.block_id),
                        span=document.locate(block.block_id, match.start(), match.end()),
                        extra={
                            "numeric": value,
                            "notation": source,
                            "zero": value == 0,
                            "negative": negative,
                        },
                    )
                )
    return hits


def extract_dates(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for block in document.blocks:
        text = block.text
        flat = T.full_to_half(text)
        seen: set[tuple] = set()
        for match in _DATE_PATTERN.finditer(flat):
            raw = match.group(0)
            parsed = T.parse_date(raw)
            if parsed is None:
                continue
            iso = parsed.isoformat()
            role = _nearest_role(text, match.start(), _DATE_ROLE_HINTS, "other")
            if role == "other":
                role = _role_from_block(text, _DATE_ROLE_HINTS, "other")
            occurrence = (iso, role, match.start(), match.end())
            if occurrence in seen:
                continue
            seen.add(occurrence)
            hits.append(
                RuleHit(
                    kind="date",
                    value=raw.strip(),
                    normalized=iso,
                    role=role,
                    confidence=0.9 if role != "other" else 0.78,
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=document.locate(block.block_id, match.start(), match.end()),
                )
            )
    return hits


def extract_subjects(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for block in document.blocks:
        text = block.text
        for key in SUBJECT_KEYS:
            index = text.find(key)
            if index < 0:
                continue
            tail = text[index + len(key):]
            match = re.match(r"\s*[：:]\s*([^\n，,；;。]{2,60})", tail)
            if not match:
                continue
            raw = match.group(1).strip()
            if len(raw) < 2:
                continue
            start = index + len(key) + match.start(1)
            hits.append(
                RuleHit(
                    kind="subject",
                    value=raw,
                    normalized=T.squash(raw)[:60],
                    role=key,
                    confidence=0.82,
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=document.locate(block.block_id, start, start + len(raw)),
                )
            )
            break
    return hits


def extract_identifiers(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    patterns: tuple[tuple[re.Pattern[str], str, float], ...] = (
        (CREDIT_CODE_RE, "credit_code", 0.95),
        (PHONE_RE, "phone", 0.82),
        (CONTRACT_NO_RE, "contract_no", 0.6),
    )
    for block in document.blocks:
        for pattern, role, confidence in patterns:
            for match in pattern.finditer(block.text):
                raw = match.group(1) if match.groups() else match.group(0)
                if role == "contract_no" and len(raw) < 8:
                    continue
                hits.append(
                    RuleHit(
                        kind="identifier",
                        value=raw,
                        normalized=T.squash(raw),
                        role=role,
                        confidence=confidence,
                        clause_id=_find_clause_for_block(document, block.block_id),
                        span=document.locate(block.block_id, match.start(), match.end()),
                    )
                )
    return hits


def extract_durations(document: DocumentParse) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for block in document.blocks:
        flat = T.full_to_half(block.text)
        # 日期里的"2026年 / 11月 / 4日"不是履约期限，先算出日期区间再跳过重叠
        date_spans = [(m.start(), m.end()) for m in _DATE_PATTERN.finditer(flat)]
        for match in DURATION_RE.finditer(flat):
            if any(match.start() < end and match.end() > start for start, end in date_spans):
                continue
            raw = match.group(0).strip()
            hits.append(
                RuleHit(
                    kind="duration",
                    value=raw,
                    normalized=raw.replace(" ", ""),
                    role="term",
                    confidence=0.7,
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=document.locate(block.block_id, match.start(), match.end()),
                )
            )
    return hits


def rule_elements(document: DocumentParse) -> list[RuleHit]:
    return [
        *extract_parties(document),
        *extract_amounts(document),
        *extract_dates(document),
        *extract_subjects(document),
        *extract_identifiers(document),
        *extract_durations(document),
    ]


# ---------------------------------------------------------------------------
# 模型通道
# ---------------------------------------------------------------------------

_MODEL_KIND_MAP = {"PARTY": "party", "AMOUNT": "amount", "DATE": "date", "SUBJECT": "subject"}


def model_elements(
    document: DocumentParse,
    tagger: NerTagger | None,
    *,
    min_confidence: float = 0.0,
) -> list[Element]:
    """用感知机在**长文本块**上抽要素（短元信息行交给规则更准）。"""
    if tagger is None:
        return []
    out: list[Element] = []
    for block in document.blocks:
        if block.kind in ("table_row",) or len(block.text) < 12:
            continue
        for start, end, entity, score in tagger.decode(block.text):
            kind = _MODEL_KIND_MAP.get(entity)
            if not kind:
                continue
            raw_surface = block.text[start:end]
            start += len(raw_surface) - len(raw_surface.lstrip())
            end -= len(raw_surface) - len(raw_surface.rstrip())
            surface = block.text[start:end]
            if len(surface) < 2:
                continue
            normalized = _normalize_surface(kind, surface)
            if not normalized:
                continue
            confidence = 0.5 + 0.45 * max(0.0, min(1.0, abs(score) / 6.0))
            if confidence < min_confidence:
                continue
            out.append(
                Element(
                    element_id=stable_id("el", "model", kind, normalized, block.block_id),
                    kind=kind,
                    value=surface,
                    normalized=normalized,
                    role="model",
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=document.locate(block.block_id, start, end),
                    confidence=round(confidence, 3),
                    source="model",
                )
            )
    return out


def _normalize_surface(kind: str, surface: str) -> str:
    if kind == "amount":
        value = T.money_to_float(surface)
        return f"{value:.2f}" if value else ""
    if kind == "date":
        return T.parse_date_text(surface)
    if kind == "party":
        cleaned = T.normalize_person(surface)
        return cleaned if len(cleaned) >= 2 else ""
    return T.squash(surface)[:60]


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------

_KIND_BUCKET = {
    "party": "parties",
    "amount": "amounts",
    "date": "dates",
    "subject": "subjects",
    "identifier": "identifiers",
    "duration": "durations",
}


def fuse_elements(
    rule_hits: Sequence[RuleHit],
    model_hits: Sequence[Element],
) -> ContractElements:
    """Only identical occurrences with compatible roles corroborate each other."""
    from .ner_constraints import location_key, compatible_roles, GENERIC_ROLES
    bucket: dict[tuple, Element] = {}

    for hit in rule_hits:
        key = (hit.kind, hit.normalized, location_key(hit), hit.role)
        element = Element(
            element_id=stable_id("el", *key),
            kind=hit.kind,
            value=hit.value,
            normalized=hit.normalized,
            role=hit.role,
            clause_id=hit.clause_id,
            span=hit.span,
            confidence=hit.confidence,
            source="rule",
            extra=dict(hit.extra),
        )
        existing = bucket.get(key)
        if existing is None or element.confidence > existing.confidence:
            bucket[key] = element

    for element in model_hits:
        key = (element.kind, element.normalized, location_key(element), element.role)
        same_location = [value for value in bucket.values() if value.kind == element.kind
                         and location_key(value) == location_key(element)]
        existing = next((value for value in same_location if value.normalized == element.normalized
                         and compatible_roles(value.role, element.role)), None)
        if existing is None:
            for other in same_location:
                for item in (other, element):
                    item.extra["manual_required"] = True
                    item.extra["channel_disagreement"] = True
                    item.extra.setdefault("review_flags", []).append("ner_same_span_channel_conflict")
            bucket[key] = element
            continue
        # 双通道命中：置信度加性提升（上限 0.99），并标注来源
        merged_conf = min(0.99, max(existing.confidence, element.confidence) + 0.08)
        existing.confidence = round(merged_conf, 3)
        existing.source = "rule+model"
        existing.extra["corroboration_scope"] = "same_source_span_and_compatible_role"
        existing.extra["channel_sources"] = ["rule", element.source]
        if existing.role in GENERIC_ROLES and element.role not in GENERIC_ROLES:
            existing.role = element.role
        if not existing.span and element.span:
            existing.span = element.span

    out = ContractElements()
    for element in bucket.values():
        kind = element.kind
        attr = _KIND_BUCKET.get(kind)
        if not attr:
            continue
        getattr(out, attr).append(element)
    for attr in ("parties", "amounts", "dates", "subjects", "identifiers", "durations"):
        getattr(out, attr).sort(key=lambda e: -e.confidence)
    return out


# ---------------------------------------------------------------------------
# 要素一致性信号
# ---------------------------------------------------------------------------

def element_signals(elements: ContractElements, document: DocumentParse) -> list[ElementSignal]:
    """由要素之间的一致性推出的可疑点（纯确定性推理，可解释、可写进报告）。"""
    signals: list[ElementSignal] = []

    # 1) 当事人缺失 / 只有一方
    sides = {p.role for p in elements.parties}
    if not elements.parties:
        signals.append(
            ElementSignal(kind="party_missing", note="未抽取到任何当事人名称",
                          severity="high", confidence=0.8)
        )
    elif "buyer" not in sides or "seller" not in sides:
        signals.append(
            ElementSignal(
                kind="missing_party_info",
                note=f"当事人角色不完整：已识别 {sorted(sides)}，缺少{'买方' if 'buyer' not in sides else '卖方'}",
                severity="medium", confidence=0.65,
            )
        )

    # 2) 金额缺失 / 大小写不一致 / 分项之和与总价不符
    totals = [a for a in elements.amounts if a.role == "total"]
    if not elements.amounts:
        signals.append(
            ElementSignal(kind="amount_missing", note="全文未识别到任何金额",
                          severity="medium", confidence=0.7)
        )
    # Amount notation comparisons must belong to the same local obligation.
    # An unscoped sum over total/unit/stage prices is not a valid proof of an
    # inconsistent contract; explicit composition is handled by fact_slots.
    groups: dict[tuple, list[Element]] = {}
    for amount in elements.amounts:
        if amount.extra.get("numeric_comparison_eligible") is False:
            continue
        if amount.role not in ("other", "unknown", "model", ""):
            groups.setdefault((amount.clause_id, amount.role), []).append(amount)
    block_map = document.block_map()
    for values in groups.values():
        digits = [a for a in values if a.extra.get("notation") == "digit"]
        chinese = [a for a in values if a.extra.get("notation") == "cn"]
        if len(digits) != 1 or len(chinese) != 1 or digits[0].normalized == chinese[0].normalized:
            continue
        left, right = digits[0], chinese[0]
        if not left.span or not right.span or left.span.block_id != right.span.block_id:
            continue
        block = block_map.get(left.span.block_id)
        if not block or abs(left.span.char_start-right.span.char_start)>80:
            continue
        if "大写" not in block.text[max(0,right.span.char_start-8):right.span.char_start]:
            continue
        signals.append(ElementSignal(kind="amount_inconsistent",
            note=f"同一金额角色的大小写声明不同：数字 {left.normalized} vs 大写 {right.normalized}",
            severity="high", value=left.normalized, confidence=0.6))

    # 3) 日期倒置
    dates = sorted({d.normalized for d in elements.dates})
    if len(dates) >= 2:
        by_role: dict[str, str] = {}
        for date in elements.dates:
            by_role.setdefault(date.role, date.normalized)
        start = by_role.get("start") or by_role.get("sign")
        end = by_role.get("end")
        if start and end and end < start:
            signals.append(
                ElementSignal(
                    kind="date_reversed",
                    note=f"结束日期 {end} 早于开始/签订日期 {start}",
                    severity="high", value=f"{start}->{end}", confidence=0.85,
                )
            )
    if not elements.dates:
        signals.append(
            ElementSignal(kind="date_missing", note="全文未识别到关键日期",
                          severity="medium", confidence=0.6)
        )

    # 4) 买卖双方同名（模板套错 / 名称填反）
    roles: dict[str, set[str]] = {}
    for party in elements.parties:
        if not party.extra.get("channel_disagreement"):
            roles.setdefault(party.normalized, set()).add(party.role)
    for name, named_roles in sorted(roles.items()):
        if {"buyer", "seller"}.issubset(named_roles):
            signals.append(
                ElementSignal(kind="party_duplicated", note=f"当事人 {name} 同时出现在双方位置",
                              severity="medium", value=name, confidence=0.55)
            )
            break

    # 5) 标的缺失
    if not elements.subjects:
        signals.append(
            ElementSignal(kind="subject_unspecified", note="未识别到标的/项目名称",
                          severity="medium", confidence=0.6)
        )

    return signals


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_BERT_CACHE: dict[str, Any] = {}


def _bert_load():
    """惰性加载 BERT-CRF；没有 GPU 产物时返回 (None, None)。"""
    if "loaded" in _BERT_CACHE:
        return _BERT_CACHE.get("model"), _BERT_CACHE.get("tokenizer")
    model = tokenizer = None
    try:
        from . import bert_ner

        if bert_ner.available():
            model, tokenizer, _meta = bert_ner.load_model()
    except Exception:
        model = tokenizer = None
    _BERT_CACHE["loaded"] = True
    _BERT_CACHE["model"] = model
    _BERT_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def bert_elements(document: DocumentParse, *, min_confidence: float = 0.0) -> list[Element]:
    """用 BERT-CRF 抽要素（节点①的模型通道，需 GPU 训练产物）。"""
    model, tokenizer = _bert_load()
    if model is None or tokenizer is None:
        return []
    from . import bert_ner

    out: list[Element] = []
    for block in document.blocks:
        if len(block.text.strip()) < 8:
            continue
        try:
            spans = bert_ner.predict(model, tokenizer, block.text)
        except Exception:
            continue
        for start, end, entity, score in spans:
            kind = _MODEL_KIND_MAP.get(entity)
            if not kind:
                continue
            raw_surface = block.text[start:end]
            start += len(raw_surface) - len(raw_surface.lstrip())
            end -= len(raw_surface) - len(raw_surface.rstrip())
            surface = block.text[start:end]
            if len(surface) < 2:
                continue
            normalized = _normalize_surface(kind, surface)
            if not normalized:
                continue
            # 模型会把0.05%里的 0.05 当金额（实测在干净样本上造出 fee_mismatch 误报）。
            # 百分号/千分号紧跟其后，或数值不足 1 又没有货币单位时，一律不当金额。
            if kind == "amount":
                tail = block.text[end:end + 1]
                has_unit = any(unit in surface for unit in ("元", "圆", "万", "亿"))
                if tail in ("%", "％", "‰"):
                    continue
                try:
                    magnitude = abs(float(normalized))
                except (TypeError, ValueError):
                    magnitude = 0.0
                if magnitude < 1.0 and not has_unit:
                    continue
            confidence = 0.72 + 0.25 * max(0.0, min(1.0, float(score)))
            if confidence < min_confidence:
                continue
            # party 的角色表是 dict{关键词: 角色}，金额/日期是 (关键词, 角色) 元组表，
            # 两种结构不同，先统一成按位置就近匹配的形态
            if kind == "party":
                hints = tuple(_ROLE_HINTS.items())
                role = _nearest_role(block.text, start, hints, "other")
            elif kind == "amount":
                role = _nearest_role(block.text, start, _AMOUNT_ROLE_HINTS, "other")
            elif kind == "date":
                role = _nearest_role(block.text, start, _DATE_ROLE_HINTS, "other")
            else:
                role = "other"
            out.append(
                Element(
                    element_id=stable_id("el", "bert", kind, normalized, block.block_id),
                    kind=kind,
                    value=surface,
                    normalized=normalized,
                    role=role,
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=document.locate(block.block_id, start, end),
                    confidence=confidence,
                    source="bert_ner",
                )
            )
    return out


def extract_elements(
    document: DocumentParse,
    config: Config,
    tagger: NerTagger | None = None,
) -> tuple[ContractElements, list[ElementSignal], dict[str, Any]]:
    rule_hits = rule_elements(document) if config.ner_enable_rules else []
    model_hits: list[Element] = []
    bert_hits: list[Element] = []
    if config.ner_enable_model:
        if getattr(config, "ner_use_bert", True):
            bert_hits = bert_elements(document, min_confidence=config.ner_min_confidence)
        # BERT 通道有产出就不再叠加感知机（后者宏 F1 只有 0.0179，只作无 GPU 时的兜底）
        model_hits = bert_hits or model_elements(
            document, tagger, min_confidence=config.ner_min_confidence
        )

    from .ner_constraints import constrain_hits, VERSION
    rule_hits, rejected_rules = constrain_hits(document, rule_hits)
    model_hits, rejected_model = constrain_hits(document, model_hits)
    elements = fuse_elements(rule_hits, model_hits)
    signals = element_signals(elements, document)

    stats = {
        "rule_hits": len(rule_hits),
        "model_hits": len(model_hits),
        "fused": len(elements.all()),
        "by_kind": elements.counts(),
        "model_available": tagger is not None or bool(bert_hits),
        "model_channel": "bert_ner" if bert_hits else ("perceptron" if model_hits else "off"),
        "signals": len(signals),
        "constraint_rejections": rejected_rules + rejected_model,
        "constraint_version": VERSION,
        "channel_conflicts": [element.element_id for element in elements.all()
                              if element.extra.get("channel_disagreement")],
        "independent_entity_f1": None,
    }
    return elements, signals, stats
