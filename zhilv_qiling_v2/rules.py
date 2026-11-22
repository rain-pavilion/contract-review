"""确定性风险规则引擎：27 个风险码的可解释检测器。

v1 把"发现风险"几乎全押在 LLM 上：模型一起来什么都好，模型一停就只剩要素层的
几条硬规则。v2 把风险检测拆成**三层**，任何一层单独都能出结论：

1. 要素层（`ner.element_signals`）：金额/日期/当事人的算术与顺序矛盾；
2. 条款层（本模块）：单条条款内的"坏写法"——免责过宽、单向违约、放弃权利、
   单方变更、交付模糊、验收无标准、违法条款、零租金、异常价格；
3. 文档层（本模块）：跨条款的结构缺陷——争议解决缺失、管辖不明、交叉引用悬空、
   条款重复、保密缺失、权属/授权/资质材料缺失。

每条规定产出 `RiskCandidate`，带 `evidence`（原文片段 + 条款号）与 `confidence`。
LLM 通道（`agents.clause_review`）只在**规则没覆盖到的语义组合**上补充，
所以离线也能给出可写进报告的完整风险清单——这是 v2 相对 v1 最实质的工程改进。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import text as T
from .models import (
    Clause,
    Contract,
    ContractElements,
    ElementSignal,
    Evidence,
    RiskCandidate,
    code_category,
    code_label,
    code_severity,
    stable_id,
)

#: 模糊措辞（交付/验收/时间约定里出现即视为不明确）
VAGUE_WORDS: tuple[str, ...] = (
    "及时", "尽快", "适当", "视情况", "另行约定", "另行协商", "根据实际情况",
    "合理期限", "必要时", "酌情", "原则上", "尽量", "大概", "左右", "等等",
)
#: 单方权利措辞
UNILATERAL_WORDS: tuple[str, ...] = (
    "单方变更", "单方面变更", "单方解除", "单方面解除", "无需通知", "无须通知",
    "最终解释权", "自行决定", "有权随时", "随时调整", "不予通知",
)
#: 免责措辞
EXEMPTION_PATTERNS: tuple[str, ...] = (
    "不承担任何责任", "不承担任何法律", "概不负责", "免除全部责任", "免除其全部责任",
    "无论何种原因", "任何情况下均不承担", "不负任何责任", "不承担赔偿责任",
)
#: 放弃权利措辞
WAIVER_PATTERNS: tuple[str, ...] = (
    "放弃", "不得主张", "无权要求", "无权主张", "视为放弃", "不得提出异议",
    "不得追究", "承诺不追究", "永不追究",
)
#: 争议解决相关词
DISPUTE_WORDS: tuple[str, ...] = ("争议", "纠纷", "争执")
RESOLUTION_WORDS: tuple[str, ...] = ("仲裁", "人民法院", "法院", "诉讼", "调解")
#: 保密相关词
CONFIDENTIAL_WORDS: tuple[str, ...] = ("保密", "商业秘密", "不得披露", "保密义务")
#: 权属/授权/资质
TITLE_WORDS: tuple[str, ...] = ("权属", "产权", "不动产权证书", "房产证", "所有权证", "处分权")
AUTH_WORDS: tuple[str, ...] = ("授权委托书", "授权文件", "书面授权", "委托书")
LICENSE_WORDS: tuple[str, ...] = ("许可证", "资质", "许可证明", "备案证明", "预售许可")


@dataclass
class RuleContext:
    """规则需要的上下文：条款、全文、要素、邻接关系。"""

    contract: Contract
    clauses: list[Clause]
    full_text: str
    elements: ContractElements
    signals: list[ElementSignal]
    clause_map: dict[str, Clause] = field(default_factory=dict)

    @classmethod
    def build(cls, contract: Contract) -> "RuleContext":
        clauses = contract.document.clauses
        return cls(
            contract=contract,
            clauses=clauses,
            full_text=contract.document.full_text,
            elements=contract.elements,
            signals=contract.signals,
            clause_map={c.clause_id: c for c in clauses},
        )


def _candidate(
    code: str,
    message: str,
    clause: Clause | None,
    evidence_text: str = "",
    *,
    confidence: float = 0.6,
    severity: str | None = None,
    source: str = "rule",
    extra: dict[str, Any] | None = None,
) -> RiskCandidate:
    label = code_label(code)
    category = code_category(code)
    default_severity = code_severity(code)
    spans = []
    if clause is not None and evidence_text:
        index = clause.text.find(evidence_text[:24])
        spans.append(
            contract_span(clause, index if index >= 0 else 0, evidence_text)
        )
    evidence = []
    if evidence_text:
        evidence.append(
            Evidence(
                kind="text",
                text=T.truncate(T.squash(evidence_text), 120),
                strength=0.7,
                span=spans[0] if spans else None,
                source=source,
            )
        )
    return RiskCandidate(
        candidate_id=stable_id("rc", code, clause.clause_id if clause else "doc", evidence_text[:32]),
        code=code,
        message=message or label,
        severity=severity or default_severity,
        clause_ids=[clause.clause_id] if clause else [],
        spans=spans,
        evidence=evidence,
        confidence=round(confidence, 3),
        source=source,
        tags=sorted({str(v) for k, v in (extra or {}).items() if k == "issue_code"}),
        extra=dict(extra or {}),
    )


def contract_span(clause: Clause, start: int, fragment: str):
    from .models import SpanRef

    block_id = clause.block_ids[0] if clause.block_ids else clause.clause_id
    return SpanRef(
        block_id=block_id,
        quote=fragment[:80],
        char_start=max(0, start),
        char_end=max(0, start) + len(fragment),
        clause_id=clause.clause_id,
    )


# ---------------------------------------------------------------------------
# 条款层规则
# ---------------------------------------------------------------------------

ClauseRule = Callable[[Clause, RuleContext], list[RiskCandidate]]


def _norm_amounts(elements: ContractElements) -> list[float]:
    out: list[float] = []
    for element in elements.amounts:
        try:
            out.append(float(element.normalized))
        except (TypeError, ValueError):
            continue
    return out


def _numeric_comparison_allowed(elements: ContractElements, clause_id: str, value: float) -> bool:
    """Honor source-quality exclusions without deleting the original candidate."""
    for element in elements.amounts:
        if element.clause_id != clause_id or (element.extra or {}).get("numeric_comparison_eligible") is not False:
            continue
        try:
            if abs(float(element.normalized) - value) < 0.005:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _bound_money_fields(clause: Clause, elements: ContractElements, labels: Sequence[str]) -> list[dict[str, Any]]:
    """Extract only money directly attached to the named field in this clause.

    Unlabelled table cells and isolated OCR capitalized fragments are deliberately
    not used to infer a deposit, unit price or contract total.
    """
    label = "|".join(re.escape(item) for item in sorted(labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?P<label>{label})\s*(?:[（(][^）)\n]{{0,16}}[）)]\s*)?"
        r"(?:金额\s*)?(?:约定\s*)?(?:为|是|：|:|=)?\s*"
        r"(?P<money>(?:人民币|￥|¥)?\s*[+-]?\d[\d,，]*(?:\.\d+)?\s*(?:亿元|万元|元|圆))")
    found = []
    for match in pattern.finditer(T.full_to_half(clause.text)):
        value = T.money_to_float(match.group("money"))
        if value is not None and _numeric_comparison_allowed(elements, clause.clause_id, value):
            found.append({"value": value, "label": match.group("label"), "quote": clause.text[match.start():match.end()],
                          "clause_id": clause.clause_id, "start": match.start(), "end": match.end()})
    return found


def _same_document_source(left: Clause, right: Clause) -> bool:
    lhs = left.metadata.get("source_id") or left.metadata.get("document_id")
    rhs = right.metadata.get("source_id") or right.metadata.get("document_id")
    return lhs == rhs


def _bound_contract_total(clause: Clause, ctx: RuleContext) -> dict[str, Any] | None:
    labels = ("合同总金额", "合同总价", "合同总额", "合同价款", "合同金额")
    local = _bound_money_fields(clause, ctx.elements, labels)
    candidates = local or [row for other in ctx.clauses if _same_document_source(clause, other)
                           for row in _bound_money_fields(other, ctx.elements, labels)]
    positive = [row for row in candidates if row["value"] > 0]
    # Original/amended totals, multiple objects or other ambiguity cannot be
    # resolved by taking the largest amount in the whole contract.
    if len({row["value"] for row in positive}) != 1:
        return None
    return positive[0]


def rule_zero_rent(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    if not any(keyword in clause.text for keyword in ("租金", "费用", "报酬", "价款", "保管费")):
        return []
    if re.search(r"(租金|费用|报酬|价款|保管费)[^。；;]{0,12}(为零|(?<![\d.])0+(?:\.0+)?\s*元|零元|免费|无偿)", clause.text):
        return [_candidate("zero_rent", "租金/费用约定为零或无偿", clause,
                           _clip_evidence(clause.text, ("租金", "费用", "报酬")), confidence=0.85)]
    for value in _norm_amounts(ctx.elements):
        if value == 0:
            continue
    return []


def rule_abnormal_price(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    if not any(keyword in clause.text for keyword in ("押金", "定金", "保证金", "违约金")):
        return []
    deposits = _bound_money_fields(clause, ctx.elements, ("押金", "定金", "保证金"))
    total_binding = _bound_contract_total(clause, ctx) if deposits else None
    if not total_binding:
        return []
    total = total_binding["value"]
    for deposit in deposits:
        value = deposit["value"]
        if value > total * 0.5 and total >= 10000:
            candidate = _candidate(
                "abnormal_price", f"押金/保证金 {value:.0f} 元超过合同总额 {total:.0f} 元的一半",
                clause, deposit["quote"], confidence=0.62,
                extra={"amount_binding": {"deposit": deposit, "contract_total": total_binding}})
            if total_binding["clause_id"] not in candidate.clause_ids:
                candidate.clause_ids.append(total_binding["clause_id"])
            candidate.evidence_append("amount_binding", total_binding["quote"], 0.7, "rule:bound_contract_total")
            return [candidate]
    return []


def _retained_delivery_date(clause: Clause, ctx: RuleContext) -> dict[str, Any] | None:
    """An optional future amendment is not proof that the existing date is absent."""
    text = re.sub(r"\s+", "", clause.text)
    conditional = bool(re.search(r"(?:如|若|如果)[^。；;]{0,45}(?:交货|交付)时间[^。；;]{0,15}(?:需要|需)[^。；;]{0,6}(?:变更|调整)", text))
    if not conditional or "书面" not in text or not any(w in text for w in ("确认", "确定")):
        return None
    if re.search(r"原[^。；;]{0,15}(?:取消|不再适用|作废)", text):
        return None
    package_id = clause.metadata.get("package_id")
    group_id = clause.metadata.get("article_group_id")
    if not package_id or not group_id or clause.metadata.get("source_role") != "supplemental_agreement":
        return None
    retained = [other for other in ctx.clauses if _same_document_source(clause, other)
                and other.metadata.get("article_group_id") == group_id
                and re.search(r"(?:交货|交付)时间[^。；;]{0,70}(?:仍按|按照|依照)原合同(?:执行|约定)",
                              re.sub(r"\s+", "", other.text))]
    if not retained:
        return None
    dated = []
    for original in ctx.clauses:
        if original.metadata.get("package_id") != package_id or original.metadata.get("source_role") != "original_contract":
            continue
        for match in re.finditer(r"(?:交货|交付)时间\s*[:：]?[^。；;\n]{0,90}", original.text):
            dates = T.all_dates(match.group())
            if dates:
                dated.append({"clause_id": original.clause_id, "source_id": original.metadata.get("source_id"),
                              "quote": match.group(), "dates": [value.isoformat() for value in dates]})
    if not dated:
        return None
    return {"reason": "本条仅约定有条件的书面变更，同条其他子款保留原合同交期，原合同存在明确日期。",
            "retention_clause_ids": [item.clause_id for item in retained], "original_delivery_evidence": dated,
            "meaning": "排除没有任何交期的断言；不确认实际交付、变更已经发生或OCR逐字准确。"}


def rule_vague_delivery(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    if not any(keyword in clause.text for keyword in ("交付", "交货", "完成时间", "履行期限", "工期")):
        return []
    hits = [word for word in VAGUE_WORDS if word in clause.text]
    has_date = bool(T.all_dates(clause.text)) or bool(re.search(r"\d+\s*(日|天|月|年|个工作日)", clause.text))
    if hits and not has_date:
        retained = _retained_delivery_date(clause, ctx)
        if retained:
            clause.metadata.setdefault("rule_binding_audit", {})["vague_delivery"] = retained
            return []
        return [
            _candidate(
                "vague_delivery",
                f"交付/履行时间使用模糊措辞（{'、'.join(hits[:3])}）且未给出具体日期或天数",
                clause, _clip_evidence(clause.text, ("交付", "履行期限", "工期")),
                confidence=0.68,
            )
        ]
    return []


def rule_unclear_standard(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    if "验收" not in clause.text and "质量标准" not in clause.text:
        return []
    hits = [word for word in VAGUE_WORDS if word in clause.text]
    weak = [word for word in ("满意", "认可", "同意后", "另行确定", "由甲方决定") if word in clause.text]
    if hits or weak:
        return [
            _candidate(
                "unclear_standard",
                f"验收标准缺少可检验依据（{'、'.join((hits + weak)[:3])}）",
                clause, _clip_evidence(clause.text, ("验收", "质量标准")),
                confidence=0.62,
            )
        ]
    return []


def rule_liability_exemption(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    hits = [pattern for pattern in EXEMPTION_PATTERNS if pattern in clause.text]
    if hits:
        return [
            _candidate(
                "liability_exemption",
                f"存在过宽免责表述（{hits[0]}），可能构成《民法典》第五百零六条项下无效免责",
                clause, _clip_evidence(clause.text, hits), confidence=0.82,
            )
        ]
    return []


def rule_one_sided_breach(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    text = clause.text
    if "违约" not in text and "责任" not in text:
        return []
    one_sided = re.search(
        r"(甲方|乙方|出租人|承租人|委托人|受托人)[^。；;]{0,20}(不承担|无需承担|不追究)[^。；;]{0,10}(违约|赔偿|责任)",
        text,
    )
    if one_sided:
        return [
            _candidate(
                "one_sided_breach",
                f"仅对单方免除违约责任：{one_sided.group(0)[:36]}",
                clause, one_sided.group(0), confidence=0.78,
            )
        ]
    if re.search(r"(甲方|出租人|委托人|采购人)[^。；;]{0,8}违约[^。；;]{0,16}(不适用|不承担|免除)", text):
        return [
            _candidate("one_sided_breach", "违约条款对一方不适用，责任分配失衡",
                       clause, _clip_evidence(text, ("违约",)), confidence=0.7)
        ]
    return []


def rule_waiver_of_rights(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    hits = [pattern for pattern in WAIVER_PATTERNS if pattern in clause.text]
    if hits and any(keyword in clause.text for keyword in ("权利", "索赔", "追究", "主张", "异议", "诉讼", "仲裁")):
        return [
            _candidate(
                "waiver_of_rights",
                f"约定放弃法定权利（{hits[0]}），可能因排除对方主要权利而无效",
                clause, _clip_evidence(clause.text, hits), confidence=0.75,
            )
        ]
    return []


def rule_unilateral_modify(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    hits = [word for word in UNILATERAL_WORDS if word in clause.text]
    if hits:
        return [
            _candidate(
                "unilateral_modify",
                f"保留单方变更/解释权（{'、'.join(hits[:2])}），对方缺乏对价保护",
                clause, _clip_evidence(clause.text, hits), confidence=0.72,
            )
        ]
    return []


def rule_illegal_terms(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    patterns: tuple[tuple[str, str], ...] = (
        (r"(押金|定金|保证金)[^。；;]{0,12}(一律|全部|均)?不予退还", "押金/定金一律不退，涉嫌不当得利"),
        (r"(不得|禁止)[^。；;]{0,10}(起诉|诉讼|申请仲裁|向法院)", "限制对方诉权，违反强制性规定"),
        (r"(人身|伤亡|工伤)[^。；;]{0,12}(免责|不承担|概不负责)", "人身损害免责条款无效"),
        (r"(超期|逾期)[^。；;]{0,10}(视为|即)[^。；;]{0,6}(放弃|解除)", "逾期即视为放弃权利，条款显失公平"),
        (r"最终解释权", "以格式条款保留最终解释权"),
    )
    for pattern, message in patterns:
        match = re.search(pattern, clause.text)
        if match:
            return [
                _candidate("illegal_terms", message, clause, match.group(0), confidence=0.8)
            ]
    return []


def rule_unverified_property(clause: Clause, ctx: RuleContext) -> list[RiskCandidate]:
    if not any(keyword in clause.text for keyword in ("出租", "转让", "出售", "买卖", "租赁")):
        return []
    negative = re.search(r"(无权属证明|未取得[^。；;]{0,8}(权属|产权)|无产权证|无法提供[^。；;]{0,6}(权属|产权))", clause.text)
    if negative:
        return [
            _candidate(
                "unverified_property",
                "标的物权属未核验即约定出租/转让，存在无权处分风险",
                clause, negative.group(0), confidence=0.85,
            )
        ]
    return []


CLAUSE_RULES: tuple[ClauseRule, ...] = (
    rule_zero_rent,
    rule_abnormal_price,
    rule_vague_delivery,
    rule_unclear_standard,
    rule_liability_exemption,
    rule_one_sided_breach,
    rule_waiver_of_rights,
    rule_unilateral_modify,
    rule_illegal_terms,
    rule_unverified_property,
)


def _clip_evidence(text: str, keywords: Sequence[str], width: int = 48) -> str:
    for keyword in keywords:
        index = text.find(keyword)
        if index >= 0:
            start = max(0, index - 6)
            return text[start:start + width]
    return T.truncate(text, width)


# ---------------------------------------------------------------------------
# 文档层规则
# ---------------------------------------------------------------------------

def rule_missing_materials(ctx: RuleContext) -> list[RiskCandidate]:
    """权属证明 / 授权文件 / 资质许可 —— 都是"材料缺失"型风险，不需要 LLM。"""
    out: list[RiskCandidate] = []
    text = ctx.full_text
    is_house = any(keyword in text for keyword in ("房屋", "商品房", "住宅", "写字楼", "租赁房屋"))
    is_agency = any(keyword in text for keyword in ("代理", "委托", "中介", "受托"))
    is_licensed = any(keyword in text for keyword in ("预售", "经营许可", "资质", "施工", "医疗", "教育"))
    if is_house and not any(word in text for word in TITLE_WORDS):
        out.append(
            _candidate(
                "need_title_cert",
                "房屋类合同未见权属证明（不动产权证书/房产证）核验条款",
                None, confidence=0.6, source="rule:document",
            )
        )
    if is_agency and not any(word in text for word in AUTH_WORDS):
        out.append(
            _candidate(
                "need_authorization",
                "代理/委托关系未见授权委托书或授权范围约定",
                None, confidence=0.58, source="rule:document",
            )
        )
    if is_licensed and not any(word in text for word in LICENSE_WORDS):
        out.append(
            _candidate(
                "need_license",
                "经营/预售类合同未见资质或许可证明要求",
                None, confidence=0.55, source="rule:document",
            )
        )
    return out


def rule_dispute_resolution(ctx: RuleContext) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    has_dispute = any(word in ctx.full_text for word in DISPUTE_WORDS)
    has_resolution = any(word in ctx.full_text for word in RESOLUTION_WORDS)
    if has_dispute and not has_resolution:
        out.append(
            _candidate(
                "dispute_resolution_missing",
                "出现争议相关表述但未约定仲裁或诉讼等解决方式",
                None, confidence=0.66, source="rule:document",
            )
        )
    elif has_dispute and has_resolution:
        body = ctx.full_text
        if not re.search(r"(仲裁委员会|人民法院|法院管辖|有管辖权的法院)", body):
            out.append(
                _candidate(
                    "jurisdiction_ambiguous",
                    "争议解决条款未指向具体仲裁机构或法院，管辖可能不明",
                    None, confidence=0.55, source="rule:document",
                )
            )
    elif not has_dispute:
        out.append(
            _candidate(
                "dispute_resolution_missing",
                "全文未见争议解决条款",
                None, confidence=0.6, source="rule:document",
            )
        )
    return out


def rule_dangling_reference(ctx: RuleContext) -> list[RiskCandidate]:
    """条款交叉引用悬空：引用了"第X条"但该条不存在。"""
    out: list[RiskCandidate] = []
    labels = {T.squash(c.label) for c in ctx.clauses if c.label}
    numbers = {T.squash(c.heading[:14]) for c in ctx.clauses}
    for clause in ctx.clauses:
        for match in re.finditer(r"第([一二三四五六七八九十百零〇\d]{1,6})条", clause.text):
            token = "第%s条" % match.group(1)
            if token in T.squash(clause.label or ""):
                continue
            if any(token in number for number in numbers):
                continue
            if token in labels:
                continue
            out.append(
                _candidate(
                    "dangling_reference",
                    f"引用 {token} 但该条不存在或编号不一致",
                    clause, match.group(0), confidence=0.5, source="rule:document",
                )
            )
            break
    return out


def rule_duplicate_clause(ctx: RuleContext) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    seen: list[Clause] = []
    for clause in ctx.clauses:
        body = T.squash(clause.body or clause.text)
        if len(body) < 30:
            continue
        for earlier in seen:
            earlier_body = T.squash(earlier.body or earlier.text)
            if T.jaccard(body, earlier_body) > 0.82:
                out.append(
                    _candidate(
                        "duplicate_clause",
                        f"与 {earlier.label or earlier.title} 内容高度重复",
                        clause, T.truncate(body, 40), confidence=0.6, source="rule:document",
                    )
                )
                break
        seen.append(clause)
    return out


def rule_confidentiality(ctx: RuleContext) -> list[RiskCandidate]:
    text = ctx.full_text
    sensitive = any(keyword in text for keyword in ("个人信息", "数据", "客户信息", "商业秘密", "技术资料"))
    if sensitive and not any(word in text for word in CONFIDENTIAL_WORDS):
        return [
            _candidate(
                "confidentiality_missing",
                "合同涉及数据/个人信息/技术资料但未约定保密义务",
                None, confidence=0.58, source="rule:document",
            )
        ]
    return []


DOCUMENT_RULES: tuple[Callable[[RuleContext], list[RiskCandidate]], ...] = (
    rule_missing_materials,
    rule_dispute_resolution,
    rule_dangling_reference,
    rule_duplicate_clause,
    rule_confidentiality,
)


# ---------------------------------------------------------------------------
# 要素层信号 → 候选
# ---------------------------------------------------------------------------

_SIGNAL_CODE_MAP: dict[str, tuple[str, str]] = {
    "party_missing": ("party_missing", "未识别到任何当事人名称"),
    "missing_party_info": ("missing_party_info", "当事人角色不完整"),
    "amount_missing": ("amount_missing", "全文未识别到金额"),
    "amount_inconsistent": ("amount_inconsistent", "金额大小写不一致"),
    "fee_mismatch": ("fee_mismatch", "分项金额之和超过总价"),
    "date_reversed": ("date_reversed", "履约结束日期早于开始/签订日期"),
    "date_missing": ("date_missing", "未识别到关键日期"),
    "party_duplicated": ("party_missing", "买卖双方名称相同"),
    "subject_unspecified": ("subject_unspecified", "未识别到标的或项目名称"),
}


def signals_to_candidates(signals: Iterable[ElementSignal]) -> list[RiskCandidate]:
    from .source_relations import extraction_gap_metadata

    out: list[RiskCandidate] = []
    for signal in signals:
        code, fallback = _SIGNAL_CODE_MAP.get(signal.kind, ("", ""))
        if not code:
            continue
        candidate = RiskCandidate(
            candidate_id=stable_id("rc", code, signal.kind, signal.value or signal.note),
            code=code,
            message=signal.note or fallback,
            severity=signal.severity,
            clause_ids=[signal.clause_id] if signal.clause_id else [],
            spans=list(signal.spans),
            evidence=[
                Evidence(kind="signal", text=signal.note, strength=signal.confidence,
                         source="ner")
            ],
            confidence=round(signal.confidence, 3),
            source="element",
        )
        # A span-less extraction notice stays visible for manual review but is
        # never a scored defect: not extracting an element does not prove the
        # contract omits it.
        gap = extraction_gap_metadata(signal)
        if gap:
            candidate.extra.update(dict(gap))
            candidate.evidence.append(Evidence(
                kind="extraction_gap", source="element_extraction",
                text="抽取未识别到该要素，不等于合同正文未约定；需人工核对原文。",
                strength=0, extra=dict(gap)))
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_rules(contract: Contract) -> list[RiskCandidate]:
    """跑全部规则；返回未去重的候选列表（去重交给 `fusion`）。"""
    ctx = RuleContext.build(contract)
    out: list[RiskCandidate] = []
    out.extend(signals_to_candidates(ctx.signals))
    for clause in ctx.clauses:
        for rule in CLAUSE_RULES:
            try:
                out.extend(rule(clause, ctx))
            except Exception:  # 单条规则异常不能毁掉整份审查
                continue
    for doc_rule in DOCUMENT_RULES:
        try:
            out.extend(doc_rule(ctx))
        except Exception:
            continue
    return out


def rule_catalog() -> dict[str, dict[str, Any]]:
    """规则清单（给 Web UI 与论文附录用）。"""
    out: dict[str, dict[str, Any]] = {}
    for rule in CLAUSE_RULES:
        code = rule.__name__.replace("rule_", "")
        out[rule.__name__] = {"scope": "clause", "doc": (rule.__doc__ or "").strip()[:120]}
    for rule in DOCUMENT_RULES:
        out[rule.__name__] = {"scope": "document", "doc": (rule.__doc__ or "").strip()[:120]}
    return out
