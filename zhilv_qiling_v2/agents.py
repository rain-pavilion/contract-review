"""五个审查智能体 + 仲裁与融合（v2 的 v1 `agents/` 重写版）。

v1 用 LangGraph 把五个智能体串成图，回环依赖 LLM 输出，**模型一停整条链就断**。
v2 的改法：每个智能体都是"确定性主干 + 可选 LLM 增强"，主干不依赖任何外部服务，
LLM 只在能提升语义覆盖的地方补一刀；每条结论都带 `source` 与 `evidence`，可逐条追溯。

五个智能体（与研究报告的五节点一一对应）：

| 智能体 | 职责 | 确定性主干 | LLM 增强点 |
| --- | --- | --- | --- |
| `understand` | 条款角色标注、履行链抽取 | 关键词 + 图结构 | 条款角色纠错 |
| `clause_review` | 单条款坏写法 | `rules.CLAUSE_RULES` | 隐含风险、语义矛盾 |
| `cross_review` | 跨条款一致性 | 要素/条款对账 | 需要理解的跨条矛盾 |
| `legal` | 法条适用 | 三路检索 + 标签图 | 一致性判定 |
| `judge` | 综合评分与分类 | 加权罚分 + 校准 | 说明文生成 |

回环（与 v1 同名的四个上限）：`max_legal_retries`、`max_judge_retries`、
`max_evidence_rounds`、`max_arbitration_rounds`，每次回环都写 `TraceEvent`。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import detectors as D
from . import entailment as E
from . import rules as R
from . import text as T
from .config import Config
from .models import (
    RISK_CATEGORIES,
    Citation,
    Clause,
    Conflict,
    Contract,
    Evidence,
    FinalRisk,
    RetrievalHit,
    RiskCandidate,
    ReviewReport,
    TraceEvent,
    code_category,
    code_label,
    code_severity,
    dedupe_candidates,
    stable_id,
)

# ---------------------------------------------------------------------------
# 条款角色（understand 主干）
# ---------------------------------------------------------------------------

CLAUSE_ROLES: dict[str, tuple[str, ...]] = {
    "主体": ("甲方", "乙方", "丙方", "当事人", "名称", "住所", "法定代表人", "统一社会信用代码"),
    "标的": ("标的", "项目名称", "服务内容", "货物", "房屋", "保管物", "工程", "数据"),
    "价款": ("价款", "金额", "租金", "费用", "报酬", "支付", "结算", "发票", "税"),
    "履行": ("交付", "履行", "期限", "工期", "验收", "进度", "地点", "方式"),
    "担保": ("押金", "定金", "保证金", "担保", "抵押", "质押", "保证"),
    "违约责任": ("违约", "赔偿", "违约金", "损失", "逾期", "罚"),
    "解除变更": ("解除", "终止", "变更", "转让", "续订", "撤销"),
    "保密": ("保密", "商业秘密", "个人信息", "披露"),
    "争议解决": ("争议", "仲裁", "诉讼", "管辖", "法院"),
    "其他": ("附件", "通知", "送达", "生效", "份数", "签章"),
}


def classify_clause(clause: Clause) -> list[str]:
    text = clause.text
    scores: list[tuple[int, str]] = []
    for role, keywords in CLAUSE_ROLES.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits:
            scores.append((hits, role))
    if not scores:
        return ["其他"]
    scores.sort(reverse=True)
    top = scores[0][0]
    return [role for score, role in scores if score >= max(1, top - 1)][:3]


@dataclass
class AgentResult:
    """单个智能体的产物。"""

    agent: str
    candidates: list[RiskCandidate] = field(default_factory=list)
    citations: dict[str, list[Citation]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    hours: float = 0.0
    elapsed: float = 0.0
    degraded: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "candidates": len(self.candidates),
            "citations": {k: len(v) for k, v in self.citations.items()},
            "notes": self.notes,
            "elapsed": round(self.elapsed, 3),
            "degraded": self.degraded,
        }


# ---------------------------------------------------------------------------
# ① understand
# ---------------------------------------------------------------------------

def understand(contract: Contract, config: Config) -> AgentResult:
    """条款角色标注 + 履行链：给后续智能体提供"这条在讲什么"。"""
    started = time.time()
    roles: dict[str, list[str]] = {}
    chain: list[dict[str, Any]] = []
    for clause in contract.document.clauses:
        assigned = classify_clause(clause)
        roles[clause.clause_id] = assigned
        clause.metadata["roles"] = assigned
        if any(role in assigned for role in ("履行", "价款")):
            chain.append(
                {
                    "clause_id": clause.clause_id,
                    "label": clause.label,
                    "roles": assigned,
                    "dates": [d.isoformat() for d in T.all_dates(clause.text)],
                    "amounts": [m for m in _amount_like(clause.text)],
                }
            )
    missing_roles = [role for role in CLAUSE_ROLES if role not in {r for v in roles.values() for r in v}]
    notes = [f"条款角色覆盖 {len(set(r for v in roles.values() for r in v))}/{len(CLAUSE_ROLES)}"]
    if missing_roles:
        notes.append("未见角色：" + "、".join(missing_roles))
    degraded: list[str] = []
    if config.enable_llm:
        degraded.append("llm:role-correction-rule-only")
    return AgentResult(
        agent="understand",
        notes=notes,
        elapsed=time.time() - started,
        degraded=degraded,
        payload={"roles": roles, "chain": chain, "missing_roles": missing_roles},
    )


def _amount_like(text: str) -> list[str]:
    from .ner import AMOUNT_RE

    return [m.group(0) for m in AMOUNT_RE.finditer(T.full_to_half(text))][:4]


# ---------------------------------------------------------------------------
# ② clause_review
# ---------------------------------------------------------------------------

def clause_review(contract: Contract, config: Config) -> AgentResult:
    """单条款审查：规则主干（10 条规则）+ 可选 LLM 语义补充。"""
    started = time.time()
    ctx = R.RuleContext.build(contract)
    candidates: list[RiskCandidate] = []
    for clause in ctx.clauses:
        for rule in R.CLAUSE_RULES:
            try:
                candidates.extend(rule(clause, ctx))
            except Exception:
                continue
    degraded: list[str] = []
    if config.enable_llm:
        degraded.append("llm:clause-semantic-rule-only")
    return AgentResult(
        agent="clause_review",
        candidates=candidates,
        notes=[f"条款层规则命中 {len(candidates)} 条"],
        elapsed=time.time() - started,
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# ③ cross_review
# ---------------------------------------------------------------------------

def cross_review(contract: Contract, config: Config) -> AgentResult:
    """跨条款一致性：日期顺序、金额对账、当事人名称一致、义务重复。"""
    started = time.time()
    ctx = R.RuleContext.build(contract)
    candidates: list[RiskCandidate] = []
    candidates.extend(R.rule_duplicate_clause(ctx))
    candidates.extend(R.rule_dangling_reference(ctx))
    candidates.extend(R.rule_missing_materials(ctx))
    candidates.extend(_party_name_consistency(contract))
    candidates.extend(_date_order_across_clauses(contract))
    candidates.extend(D.run_detectors(contract))
    degraded: list[str] = []
    if config.enable_llm:
        degraded.append("llm:cross-semantic-rule-only")
    return AgentResult(
        agent="cross_review",
        candidates=candidates,
        notes=[f"跨条款核对命中 {len(candidates)} 条"],
        elapsed=time.time() - started,
        degraded=degraded,
    )


def _party_name_consistency(contract: Contract) -> list[RiskCandidate]:
    """正文里的当事人名称与签署栏是否一致（不一致常意味着模板被改坏）。"""
    out: list[RiskCandidate] = []
    names = {p.normalized for p in contract.elements.parties if len(p.normalized) >= 2}
    if not names:
        return out
    full = contract.document.full_text
    sign_block = full[-600:] if len(full) > 600 else full
    missing = [name for name in names if name not in sign_block and len(sign_block) > 60]
    if missing and len(missing) == len(names):
        out.append(
            R._candidate(
                "missing_party_info",
                "签署栏未见正文中的当事人名称，签署主体可能不完整",
                None,
                confidence=0.5,
                source="agent:cross_review",
            )
        )
    return out


def _date_order_across_clauses(contract: Contract) -> list[RiskCandidate]:
    """把各条款的日期按角色串成时间线，检查“签订晚于履行”这类倒置。"""
    out: list[RiskCandidate] = []
    dates: list[tuple[str, Any, str]] = []
    for clause in contract.document.clauses:
        roles = clause.metadata.get("roles") or classify_clause(clause)
        role = "、".join(roles)
        for value in T.all_dates(clause.text):
            dates.append((role, value, clause.clause_id))
    if len(dates) < 2:
        return out
    sign = [value for role, value, _ in dates if "签订" in role]
    end = [value for role, value, _ in dates
           if any(tag in role for tag in ("履行", "交付"))]
    if sign and end and min(sign) > max(end):
        out.append(
            R._candidate(
                "date_reversed",
                f"签订日期 {min(sign).isoformat()} 晚于履行期限 {max(end).isoformat()}，时间线倒置",
                None,
                confidence=0.7,
                source="agent:cross_review",
            )
        )
    return out


# ---------------------------------------------------------------------------
# ④ legal（法条适用）
# ---------------------------------------------------------------------------

LEGAL_TAG_HINT: dict[str, str] = {
    "zero_rent": "租金 支付 数额",
    "abnormal_price": "价格 价款 公平",
    "fee_mismatch": "价款 支付 履行",
    "date_reversed": "履行期限 约定 届满",
    "date_missing": "履行期限 约定",
    "amount_missing": "价款 条款 主要条款",
    "amount_inconsistent": "价款 数额 解释",
    "party_missing": "当事人 名称 合同成立",
    "missing_party_info": "当事人 名称 住所",
    "subject_unspecified": "标的 质量 数量 条款",
    "vague_delivery": "履行期限 交付 约定不明",
    "unclear_standard": "质量要求 检验 验收",
    "liability_exemption": "免责条款 格式条款 无效",
    "one_sided_breach": "违约责任 违约金 损失赔偿",
    "waiver_of_rights": "格式条款 排除对方主要权利 无效",
    "unilateral_modify": "合同变更 协商一致",
    "illegal_terms": "违反强制性规定 无效 公序良俗",
    "unverified_property": "无权处分 权利瑕疵 担保",
    "need_title_cert": "出卖人 权利瑕疵 权属",
    "need_authorization": "代理 授权 越权",
    "need_license": "许可 资质 批准",
    "dispute_resolution_missing": "争议解决 仲裁 诉讼",
    "jurisdiction_ambiguous": "管辖 协议管辖 法院",
    "dangling_reference": "合同解释 条款 参照",
    "duplicate_clause": "合同解释 条款 重复",
    "confidentiality_missing": "保密义务 商业秘密",
    "no_substantive_risk": "合同效力 履行",
}


def legal(
    contract: Contract,
    candidates: Sequence[RiskCandidate],
    config: Config,
    retriever: Any = None,
) -> AgentResult:
    """给每条候选风险找本地法条，并执行"引用必须可溯源"的硬护栏。"""
    started = time.time()
    degraded: list[str] = []
    citations: dict[str, list[Citation]] = {}
    if retriever is None:
        degraded.append("legal:no-index")
        return AgentResult(agent="legal", citations=citations, degraded=degraded,
                           elapsed=time.time() - started)

    from .retrieve import rewrite_query

    seen_codes: dict[str, list[Citation]] = {}
    for candidate in candidates:
        # A different clause with the same risk code needs its own semantic
        # check. Code-only caching previously reused unrelated citations.
        clause_text = ""
        if candidate.clause_ids:
            clause = contract.clause_map().get(candidate.clause_ids[0])
            if clause is not None:
                clause_text = clause.text
        hint = LEGAL_TAG_HINT.get(candidate.code, "")
        query_text = " ".join(part for part in (hint, T.truncate(clause_text, 100)) if part)
        plan = rewrite_query(query_text, contract_type=contract.contract_type,
                             elements=contract.elements, codes=[candidate.code])
        result = retriever.search(plan, top_k=config.top_k)
        hits: list[RetrievalHit] = result.hits
        # 节点④ 第二半：判"一致/冲突"（申报书要求），冲突的条文不进引用、只留证据
        entailment_notes: list[str] = []
        if getattr(config, "enable_entailment", True) and hits:
            verdicts = E.annotate(clause_text or query_text, hits[:3], codes=[candidate.code])
            candidate.extra["statute_checks"] = [asdict(v) for v in verdicts]
            pending = [v for v in verdicts if v.manual_required or v.semantic_pending or v.disagreement]
            if pending:
                candidate.extra["manual_required"] = True
                for verdict in pending:
                    candidate.evidence.append(Evidence(kind="entailment_pending", text=verdict.evidence_text(),
                        source="legal", strength=0.0, extra=asdict(verdict)))
            conflicts = [v for v in verdicts if v.is_conflict]
            if conflicts:
                candidate.evidence_append(
                    "entailment_conflict",
                    "、".join(v.evidence_text() for v in conflicts[:2]),
                    0.72,
                )
                candidate.confidence = min(0.99, candidate.confidence + 0.05)
                entailment_notes.append(f"{candidate.code}:{len(conflicts)}处冲突")
                keep_uids = {v.article_uid for v in verdicts if not v.is_conflict}
                hits = [hit for hit in hits if hit.article_uid in keep_uids] or hits[:1]
        # 硬护栏：只接受索引里真实存在、且能回指到原文的引用
        verified: list[Citation] = []
        for hit in hits:
            if not hit.article_uid or not hit.content:
                continue
            verified.append(hit.to_citation())
            if len(verified) >= config.max_citations_per_risk:
                break
        if not verified:
            degraded.append(f"legal:no-citation:{candidate.code}")
        seen_codes[candidate.code] = verified
        citations[candidate.candidate_id] = verified
    if config.enable_llm:
        degraded.append("llm:consistency-check-rule-only")
    notes = [f"为 {len(seen_codes)} 个风险码匹配法条"]
    if entailment_notes:
        notes.append("条款—条文冲突：" + "；".join(entailment_notes[:4]))
    return AgentResult(
        agent="legal",
        citations=citations,
        notes=notes,
        elapsed=time.time() - started,
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# ⑤ judge
# ---------------------------------------------------------------------------

def score_risks(risks: Sequence[FinalRisk], config: Config) -> tuple[int, dict[str, Any]]:
    """0~100 分：满分基线减去按等级加权的罚分（等级权重来自 config）。

    **同类饱和**：同一风险码重复出现时罚分递减（第 2 条 ×0.6、第 3 条 ×0.4、第 4 条起 ×0.25）。
    实测真实政采合同里 `subject_unspecified`（空条款提示）能报到 55 次，按全额罚分直接把分数
    扣到 0，但"合同里有几十个短条款没挂要素"并不等于"这份合同糟透了"。饱和之后同类噪声不会
    压过真正的高危条目，评分才有区分度。
    """
    saturation = (1.0, 0.6, 0.4, 0.25)
    seen: dict[str, int] = {}
    penalty = 0.0
    detail: dict[str, float] = {}
    from .judge_evidence import risk_basis
    excluded: list[dict[str, Any]] = []
    for risk in sorted(risks, key=lambda item: -item.confidence):
        if risk.code == "no_substantive_risk":
            continue
        basis = risk_basis(risk)
        if not basis['score_eligible']:
            excluded.append(basis)
            continue
        weight = config.severity_weights.get(risk.severity, 6.0)
        index = seen.get(risk.code, 0)
        factor = saturation[index] if index < len(saturation) else saturation[-1]
        seen[risk.code] = index + 1
        contribution = weight * float(risk.confidence) * factor
        penalty += contribution
        detail[risk.code] = round(detail.get(risk.code, 0.0) + contribution, 3)
    score = max(0, min(100, round(100 - penalty)))
    return score, {"penalty": round(penalty, 3), "by_code": detail,
                   "excluded_unsubstantiated": excluded,
                   "score_scope": "supported_text_and_deterministic_candidates_only",
                   "not_a_clearance_or_accuracy": True}


def judge(
    contract: Contract,
    candidates: Sequence[RiskCandidate],
    agent_results: Sequence[AgentResult],
    config: Config,
) -> AgentResult:
    """综合评估：分类、复核需求、是否需要更多证据。"""
    started = time.time()
    risks = fuse(candidates, agent_results, contract, config)
    score, detail = score_risks(risks, config)
    by_category: dict[str, int] = {category: 0 for category in RISK_CATEGORIES}
    for risk in risks:
        by_category[risk.category] = by_category.get(risk.category, 0) + 1
    need_review = [r.risk_id for r in risks if r.review_required]
    notes = [f"评分 {score}，风险 {len(risks)} 条，待复核 {len(need_review)} 条"]
    return AgentResult(
        agent="judge",
        notes=notes,
        elapsed=time.time() - started,
        payload={
            "score": score,
            "detail": detail,
            "risks": risks,
            "by_category": by_category,
            "need_review": need_review,
        },
    )


# ---------------------------------------------------------------------------
# 仲裁 + 融合
# ---------------------------------------------------------------------------

def arbitrate(
    candidates: Sequence[RiskCandidate],
    config: Config,
) -> tuple[list[RiskCandidate], list[Conflict]]:
    """同一风险码的多个候选之间做证据加权仲裁。

    与 v1 的区别：v1 的仲裁是"再叫一次 LLM 让它裁决"，v2 先用**证据强度 + 规则/模型
    来源先验**做确定性裁决，只有分值接近（差值 < `arbitration_material_gap`）时才
    标记为需要人工/LLM 复核，写进 `Conflict`。
    """
    buckets: dict[str, list[RiskCandidate]] = {}
    for candidate in candidates:
        buckets.setdefault(candidate.dedupe_key or candidate.code, []).append(candidate)
    kept: list[RiskCandidate] = []
    conflicts: list[Conflict] = []
    for key, group in buckets.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group.sort(key=lambda c: (-_evidence_strength(c), -c.confidence))
        best = group[0]
        runner = group[1]
        gap = _evidence_strength(best) - _evidence_strength(runner)
        best.confidence = min(0.99, max(c.confidence for c in group) + 0.02)
        best.evidence.extend(
            evidence for c in group[1:] for evidence in c.evidence[:1]
        )
        for other in group[1:]:
            best.merge_from(other)
        kept.append(best)
        if gap < config.arbitration_material_gap:
            conflicts.append(
                Conflict(
                    type="evidence_tie",
                    description=(
                        f"风险码 {best.code} 有 {len(group)} 条候选，证据强度差 "
                        f"{gap:.3f} < {config.arbitration_material_gap}"
                    ),
                    involved_ids=[c.candidate_id for c in group[:3]],
                    resolution="保留证据更强的一条，标记为需复核",
                    material=False,
                )
            )
            best.confidence = min(0.99, best.confidence - 0.05)
    return kept, conflicts


def _evidence_strength(candidate: RiskCandidate) -> float:
    base = candidate.confidence
    source_bonus = {
        "rule": 0.10,
        "element": 0.12,
        "agent:cross_review": 0.06,
        "agent:arbitrate": 0.0,
        "llm": 0.04,
    }.get(candidate.source, 0.0)
    evidence = sum(e.strength for e in candidate.evidence) / max(1, len(candidate.evidence))
    return round(base * 0.6 + evidence * 0.3 + source_bonus, 4)


def _candidate_requires_manual_review(candidate: RiskCandidate) -> bool:
    """Consume pending states even for high-confidence, cited candidates."""
    records = [candidate.extra or {}] + [ev.extra or {} for ev in candidate.evidence]
    records += [check for check in candidate.extra.get("statute_checks", []) if isinstance(check, dict)]
    for record in records:
        if record.get("manual_required") or record.get("semantic_pending") or record.get("review_flags"):
            return True
        if record.get("cross_validation_status") == "pending_manual" or record.get("semantic_status") == "pending_manual":
            return True
        status = record.get("claim_grounding_status")
        if status is not None and status != "supported":
            return True
    return False


def fuse(
    candidates: Sequence[RiskCandidate],
    agent_results: Sequence[AgentResult],
    contract: Contract,
    config: Config,
) -> list[FinalRisk]:
    """去重 → 校准 → 挂引用 → 出 `FinalRisk`。"""
    citations: dict[str, list[Citation]] = {}
    for result in agent_results:
        for candidate_id, items in result.citations.items():
            citations.setdefault(candidate_id, []).extend(items)
    source_map: dict[str, set[str]] = {}
    for result in agent_results:
        for candidate in result.candidates:
            source_map.setdefault(candidate.code, set()).add(result.agent)

    from .risk_identity import normalize_fact_candidates
    deduped = dedupe_candidates(normalize_fact_candidates(list(candidates), contract))
    risks: list[FinalRisk] = []
    calibrator = getattr(config, "_calibrator", None)
    for candidate in deduped:
        confidence = candidate.confidence
        if calibrator is not None:
            try:
                confidence = calibrator.apply(confidence)
            except Exception:
                pass
        label = code_label(candidate.code)
        category = code_category(candidate.code)
        severity = candidate.severity or code_severity(candidate.code)
        found = []
        seen_citations = set()
        for alias in [candidate.candidate_id] + list(candidate.extra.get("alias_candidate_ids", [])):
            for citation in citations.get(alias, []):
                key = (citation.article_uid, citation.law_name, citation.article_no)
                if key not in seen_citations:
                    seen_citations.add(key)
                    found.append(citation)
        review_required = (confidence < config.review_threshold or not found
                           or _candidate_requires_manual_review(candidate))
        risks.append(
            FinalRisk(
                risk_id=stable_id("rk", candidate.code, candidate.candidate_id),
                code=candidate.code,
                severity=severity,
                message=candidate.message or label,
                citations=found,
                clause_ids=list(candidate.clause_ids),
                spans=list(candidate.spans),
                confidence=round(float(confidence), 3),
                review_required=review_required,
                rationale=_rationale(candidate, found),
                agent_sources=sorted(source_map.get(candidate.code, {candidate.source})),
                category=candidate.extra.get("category") or category,
                cross_validated=len(source_map.get(candidate.code, ())) > 1,
                tags=_risk_tags(candidate),
                evidence=list(candidate.evidence),
                calibrated_confidence=round(float(confidence), 3),
            )
        )
    if not risks:
        risks.append(
            FinalRisk(
                risk_id=stable_id("rk", "no_substantive_risk", contract.contract_id),
                code="no_substantive_risk",
                severity="info",
                message="本次未生成具体风险候选；不代表审查通过，未决事项仍需核对",
                confidence=0.6,
                rationale="本次已执行的检查没有形成具体候选；不证明未执行检查、缺失事实或完整语义审查已完成",
                agent_sources=["judge"],
                category="审查执行提示",
                review_required=True,
            )
        )
    risks.sort(key=lambda r: r.sort_key())
    return risks


def _risk_tags(candidate: RiskCandidate) -> list[str]:
    """把候选的标签与"数据集口径缺陷码"一起带到最终风险上，便于评测对齐。"""
    tags = list(candidate.tags or [])
    issue_code = candidate.extra.get("issue_code") if candidate.extra else None
    if issue_code and issue_code not in tags:
        tags.append(str(issue_code))
    return tags


def _rationale(candidate: RiskCandidate, citations: Sequence[Citation]) -> str:
    parts = [candidate.message]
    if candidate.evidence:
        parts.append("证据：" + candidate.evidence[0].text[:60])
    if citations:
        parts.append("依据：" + "、".join(f"{c.law_name}{c.article_no}" for c in citations[:2]))
    else:
        parts.append("依据：本地知识库未匹配到直接条文，建议人工补充")
    return "；".join(parts)
