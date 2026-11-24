"""Read-only workflow / question / assessment projection of a review payload.

The catalog defines questions. Findings and bounded negative results require
their own records. Neither a missing finding nor a successful execution receipt
certifies legal truth, external facts, or the completeness of legal review.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from typing import Any

from .review_explanations import build_review_explanations
from .review_points import review_point_catalog


_STATUS_LABELS = {
    "issue_found": "发现问题线索", "needs_manual": "待人工核验", "not_checked": "未执行审查",
    "no_issue_detected": "限定范围内未检出问题", "not_applicable": "限定范围内不适用",
}
_CORE_NODES = [
    {"id": "orchestrator", "number": 1, "label": "主脑规划与调度"},
    {"id": "ner", "number": 2, "label": "要素抽取"},
    {"id": "logic", "number": 3, "label": "逻辑审查"},
    {"id": "subject", "number": 4, "label": "主体审查"},
    {"id": "statute", "number": 5, "label": "法条匹配"},
    {"id": "judge", "number": 6, "label": "综合评估"},
]
_CORE_STAGES = [["orchestrator"], ["ner", "logic"], ["subject"], ["statute"], ["judge"]]
_FACT_POINTS = {"registration": "subject.identity", "identity": "subject.identity",
    "authorization": "subject.authorization", "license": "subject.qualification", "qualification": "subject.qualification",
    "title": "subject.title", "ownership": "subject.title"}
_LIMIT = "仅反映本次所给文本、证据与执行记录；未检出问题不代表事实真实、法律结论成立或审查完整。"
_NODE_ALIASES = {"understand": {"intake"}, "clause_review+cross_review": {"logic"},
    "logic_llm": {"logic"}, "subject_check": {"subject"}, "legal": {"statute"},
    "judge_llm": {"judge"}, "arbitrate": {"judge"}}


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value if isinstance(item, (str, int)) and str(item)] if isinstance(value, list) else []


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None and str(value)))


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _pending(row: dict) -> list[str]:
    reasons = _strings(row.get("manual_reasons")) + _strings(row.get("pending_reasons"))
    for field in ("review_flags", "validation_errors", "upstream_review_reasons", "semantic_pending_reasons"):
        reasons += _strings(row.get(field))
    if row.get("manual_required") is True or row.get("review_required") is True:
        reasons.append("human_review_requested")
    if row.get("pending") is True or row.get("pending_manual") is True:
        reasons.append("review_item_pending")
    if row.get("status") in ("manual_review", "needs_manual", "pending", "pending_manual", "failed", "error"):
        reasons.append("review_item_pending")
    return _unique(reasons)


def _explanations(payload: dict) -> list[dict]:
    """Keep both saved references and uncertainty visible in the actual ledger."""
    supplied = _rows(_dict(payload.get("review_explanations")).get("items"))
    generated = _rows(build_review_explanations(payload).get("items"))
    # Multiple representations are intentionally retained here: point assembly
    # deduplicates identifiers and evidence without clearing either pending axis.
    return generated + supplied


def _evidence(rows: list[dict], clause_ids: list[str], clauses: dict[str, str]) -> list[dict]:
    result, seen = [], set()
    for source in rows:
        cid, quote = str(source.get("clause_id") or ""), str(source.get("quote") or "")
        key = (cid, quote)
        if key in seen:
            continue
        seen.add(key)
        item = copy.deepcopy(source)
        item.update(clause_id=cid, quote=quote,
            located=bool(_compact(quote) and _compact(quote) in _compact(clauses.get(cid))),
            source_available=cid in clauses)
        result.append(item)
    cited = {item["clause_id"] for item in result}
    for cid in clause_ids:
        if cid not in cited:
            result.append({"clause_id": cid, "quote": "", "located": False,
                "source_available": cid in clauses, "role": "context_reference"})
    return result


def _candidate_ids(risk: dict) -> set[str]:
    result = set(_strings(risk.get("candidate_ids")))
    for evidence in _rows(risk.get("evidence")):
        extra = _dict(evidence.get("extra"))
        result.update(_strings(extra.get("candidate_ids")))
        if extra.get("candidate_id"):
            result.add(str(extra["candidate_id"]))
    return result


def _mapped_points(item: dict, risk_points: dict[str, str], candidate_points: dict[str, set[str]],
                   known: set[str]) -> set[str]:
    points = set(_strings(item.get("point_ids")))
    if item.get("point_id"):
        points.add(str(item["point_id"]))
    for key in ("risk_id", "item_id"):
        if str(item.get(key) or "") in risk_points:
            points.add(risk_points[str(item[key])])
    for candidate in _strings(item.get("candidate_ids")) + [str(item.get("candidate_id") or "")]:
        points.update(candidate_points.get(candidate, set()))
    fact_types = _strings(item.get("fact_types")) + [str(item.get("fact_type") or "")]
    for fact_type in fact_types:
        if fact_type in _FACT_POINTS:
            points.add(_FACT_POINTS[fact_type])
    # Older subject feedback projections retained a readable title rather than
    # fact_type. Only specific fact labels are mapped; general work stays global.
    if str(item.get("kind") or "").startswith("subject_fact"):
        title = str(item.get("title") or "")
        for label, point in (("签约授权", "subject.authorization"), ("经营资质", "subject.qualification"),
                             ("主体登记", "subject.identity")):
            if label in title:
                points.add(point)
    return points & known


def _validated_check(check: dict, clauses: dict[str, str]) -> dict:
    """Validate a receipt's shape, execution and binding before using its claim."""
    reasons = _pending(check)
    status = check.get("status")
    shape_valid = (isinstance(check.get("check_id"), str) and bool(check["check_id"].strip())
        and isinstance(status, str) and status in _STATUS_LABELS
        and isinstance(check.get("rule_version"), str) and bool(check["rule_version"].strip())
        and isinstance(check.get("scope"), str) and bool(check["scope"].strip())
        and isinstance(check.get("clause_ids"), list)
        and all(isinstance(cid, str) and bool(cid.strip()) for cid in check["clause_ids"])
        and isinstance(check.get("evidence"), list)
        and all(isinstance(item, dict) and isinstance(item.get("clause_id"), str)
                and isinstance(item.get("quote"), str) for item in check["evidence"])
        and isinstance(check.get("pending_reasons"), list)
        and all(isinstance(reason, str) for reason in check["pending_reasons"])
        and isinstance(check.get("manual_required"), bool)
        and check.get("legal_truth_established") is False)
    if not shape_valid:
        reasons.append("review_point_receipt_invalid")
    executed = check.get("execution_status") in ("executed", "completed")
    unperformed = status == "not_checked" and bool(check.get("check_id"))
    if not executed and not unperformed:
        reasons.append("review_point_execution_unconfirmed")
    cids = _strings(check.get("clause_ids"))
    evidence = _evidence(_rows(check.get("evidence")), cids, clauses)
    quoted = [item for item in evidence if item.get("quote")]
    bound = bool(quoted and all(item["located"] and item["clause_id"] in cids for item in quoted)
                 and cids and all(cid in clauses for cid in cids)
                 and set(cids) <= {item["clause_id"] for item in quoted})
    if not bound and not unperformed:
        reasons.append("contract_quote_missing_or_unlocated")
    if check.get("scope_complete") is not True and not unperformed:
        reasons.append("review_point_scope_incomplete")
    # An inapplicability claim still needs a concrete, attributable explanation.
    if status == "not_applicable" and not str(check.get("actual_issue") or "").strip():
        reasons.append("review_point_applicability_unexplained")
    return {"check": check, "valid": bool(shape_valid and executed and not unperformed), "unperformed": unperformed,
        "reasons": _unique(reasons),
        "evidence": evidence, "clause_ids": cids,
        "execution_status": "executed" if shape_valid and executed else "not_recorded" if unperformed else "unconfirmed"}


def _assessment(point: dict, risks: list[dict], explanations: list[dict], checks: list[dict],
                clauses: dict[str, str], contract_id: str) -> dict:
    receipts = [_validated_check(check, clauses) for check in checks]
    valid_checks = [item["check"] for item in receipts if item["valid"]]
    risk_ids = _unique(risk.get("risk_id") for risk in risks)
    explanation_ids = _unique(item.get("item_id") for item in explanations)
    clause_ids = _unique([cid for risk in risks for cid in _strings(risk.get("clause_ids"))]
        + [str(item.get("clause_id") or "") for row in explanations for item in _rows(row.get("contract_evidence"))]
        + [cid for row in receipts for cid in row["clause_ids"]])
    raw_evidence = [span for risk in risks for span in _rows(risk.get("spans"))]
    raw_evidence += [item for row in explanations for item in _rows(row.get("contract_evidence"))]
    raw_evidence += [item for row in receipts for item in row["evidence"]]
    evidence = _evidence(raw_evidence, clause_ids, clauses)
    reasons = [reason for row in risks + explanations for reason in _pending(row)]
    reasons += [reason for row in receipts for reason in row["reasons"]]
    if risks and (not any(item["located"] for item in evidence)
                  or any(item.get("quote") and not item["located"] for item in evidence)):
        reasons.append("contract_quote_missing_or_unlocated")
    finding_checks = [check for check in valid_checks if check.get("status") == "issue_found"]
    if risks or finding_checks:
        status = "issue_found"
    elif receipts and all(item["unperformed"] for item in receipts) and not explanations:
        status = "not_checked"
    elif reasons or any(check.get("status") == "needs_manual" for check in valid_checks):
        status = "needs_manual"
    elif valid_checks:
        statuses = {check["status"] for check in valid_checks}
        if len(statuses) != 1:
            status = "needs_manual"
            reasons.append("review_point_result_disagreement")
        else:
            status = valid_checks[0]["status"]
    elif explanations:
        # A relation or fact record does not execute this review question.
        status = "needs_manual"
        reasons.append("review_point_check_not_recorded")
    else:
        status = "not_checked"
    actual = _unique([risk.get("message") for risk in risks]
        + [check.get("actual_issue") for check in valid_checks])
    if not actual:
        actual = [{"not_checked": "尚无针对本审查点的有效执行记录，不能据无风险条目推断已完成。",
            "needs_manual": "现有材料或执行记录仍有未决内容，尚不能形成该审查点的完整判断。"}.get(status,
            "现有记录提示需要结合原文及上下文复核。")]
    explicit_impacts = _unique(check.get("impact") for check in valid_checks if check.get("impact"))
    if explicit_impacts:
        impact, impact_basis = "；".join(explicit_impacts), "explicit_check"
    elif risks or finding_checks or status == "needs_manual":
        impact, impact_basis = point["impact_template"], "point_template"
    else:
        impact, impact_basis = "未对实际影响作出判断。", "not_assessed"
    recommendations = _unique(check.get("recommendation") for check in valid_checks)
    recommendation = "；".join(recommendations) or (point["recommendation_template"] if status != "not_checked"
        else "按审查标准取得必要材料并执行检查，保留范围和证据记录。")
    scopes = _unique(check.get("scope") for check in valid_checks)
    scope = "；".join(scopes) if scopes else point["scope"]
    if scopes:
        scope += " 仅为该范围的文本检查，不代表完整审查或最终法律结论。"
    reasons = _unique(reasons)
    unperformed_execution = ("executed" if any(item["execution_status"] == "executed" for item in receipts) else "not_recorded")
    execution = (unperformed_execution if status == "not_checked" else
                 "executed" if risks or valid_checks else "unconfirmed" if receipts else
                 "recorded" if explanations else "not_recorded")
    return {"result_id": "assessment_" + hashlib.sha256((contract_id + "|" + point["point_id"]).encode()).hexdigest()[:20],
        "point_id": point["point_id"], "status": status, "status_label": _STATUS_LABELS[status],
        "question": point["question"], "criterion": point["criterion"], "actual_issue": "；".join(actual),
        "impact": impact, "impact_basis": impact_basis, "recommendation": recommendation,
        "risk_ids": risk_ids, "explanation_item_ids": explanation_ids,
        "check_ids": _unique(check.get("check_id") for check in valid_checks),
        "clause_ids": clause_ids, "evidence": evidence, "pending_reasons": reasons,
        "execution_status": execution,
        "source_status": "provided_text_located" if evidence and any(item["located"] for item in evidence)
            else "evidence_unlocated" if evidence else "not_recorded",
        "scope": scope, "manual_required": bool(reasons or status == "needs_manual"),
        "legal_truth_established": False}


def _execution(events: list[dict], agents: list[str], aliases=()) -> tuple[str, list[str]]:
    expected = set(agents)
    observed = [row for row in events if str(row.get("node") or row.get("agent") or "") in expected | set(aliases)]
    names = _unique(row.get("node") or row.get("agent") for row in observed)
    if not observed:
        return "not_recorded", names
    statuses = {str(row.get("status") or "unknown") for row in observed}
    if statuses & {"failed", "error", "blocked"}:
        return "failed", names
    if statuses & {"degraded", "partial", "needs_manual", "pending", "pending_manual"}:
        return "needs_manual", names
    if statuses <= {"ok", "succeeded", "completed"}:
        covered = set(names)
        for name in names:
            covered.update(_NODE_ALIASES.get(name, set()))
        return ("completed" if expected <= covered else "partial"), names
    if statuses <= {"skipped", "disabled"}:
        return "skipped", names
    return "unknown", names


def _workflow(payload: dict, explanations: list[dict], assessments: list[dict], unmapped: list[str]) -> dict:
    stats = _dict(payload.get("stats"))
    subject = _dict(_dict(stats.get("subject")).get("stats")) or _dict(_dict(payload.get("subject_checks")).get("stats"))
    cross = _dict(stats.get("cross_validation"))
    supervisor = _dict(stats.get("supervisor"))
    events = list(_rows(payload.get("trace")))
    traced_nodes = {str(row.get("node") or row.get("agent") or "") for row in events}
    for record in _rows(stats.get("agents")):
        name = str(record.get("node") or record.get("agent") or "")
        if _strings(record.get("degraded")):
            events.append({**record, "status": "degraded"})
        elif record.get("status") or name not in traced_nodes:
            # Legacy AgentResult summaries lack a status. They cannot erase an
            # explicit trace status, and alone cannot certify successful work.
            events.append(record)
    pending_ids = {str(item.get("item_id") or "explanation-" + str(index))
                   for index, item in enumerate(explanations) if _pending(item)}
    pending_ids.update(_strings(payload.get("review_queue")))
    pending_ids.update(str(item.get("risk_id") or item.get("item_id") or "queue-" + str(index))
                       for index, item in enumerate(_rows(payload.get("review_queue"))))
    verification_reasons = []
    for field in ("pending_feedback_item_ids", "pending_work_item_ids", "pending_subject_verification_request_ids",
                  "pending_review_assignment_ids", "deferred_next_cycle_work_item_ids"):
        values = _strings(cross.get(field))
        pending_ids.update(values)
        if values:
            verification_reasons.append(field)
    for field in ("feedback_verification_complete", "work_items_complete"):
        if field in cross and cross[field] is not True:
            verification_reasons.append(field + "_unresolved")
    if cross.get("converged") is False:
        verification_reasons.append("cross_validation_not_converged")
    if cross.get("manual_review_requested") is True:
        verification_reasons.append("cross_validation_manual_review_requested")
    pending_ids.update(_strings(subject.get("pending_verification_request_ids")))
    if subject.get("verification_complete") is False:
        verification_reasons.append("subject_verification_incomplete")
    if any(_pending(item) and (str(item.get("kind") or "").startswith("subject") or item.get("kind") == "document_evidence_gap")
           for item in explanations):
        verification_reasons.append("subject_evidence_pending")
    if supervisor.get("manual_review_requested") is True:
        verification_reasons.append("supervisor_manual_review_requested")
    if _rows(supervisor.get("pending_reviews")):
        verification_reasons.append("expert_review_incomplete")
    for assessment in assessments:
        if assessment["manual_required"]:
            refs = assessment["risk_ids"] + assessment["explanation_item_ids"]
            pending_ids.update(refs or ["assessment:" + assessment["point_id"]])
    if unmapped:
        pending_ids.update(unmapped)
        verification_reasons.append("unmapped_risk_requires_review")
    stages = []
    definitions = [
        ("context", "审查背景与主脑规划", ["orchestrator"], ()),
        ("parse", "文本解析与条款定位", ["intake"], ("understand",)),
        ("analysis", "要素与逻辑分析", ["ner", "logic"], ("clause_review+cross_review", "logic_llm")),
        ("verification", "主体、依据与交叉核验", ["subject", "statute"], ("subject_check", "legal")),
        ("assessment", "综合评估与结果归集", ["judge"], ("judge_llm", "arbitrate")),
    ]
    trace_reasons = []
    for stage_id, label, agents, aliases in definitions:
        status, observed = _execution(events, agents, aliases)
        reasons = []
        if status in ("failed", "needs_manual", "partial", "unknown"):
            reasons.append(stage_id + "_execution_" + status)
            trace_reasons.extend(reasons)
        if stage_id == "context" and status == "not_recorded" and _dict(stats.get("review_context")):
            status = "recorded"
        if stage_id == "verification" and verification_reasons:
            if status != "failed":
                status = "needs_manual"
            reasons += verification_reasons
        stages.append({"id": stage_id, "label": label, "agents": agents, "observed_agents": observed,
            "status": status, "gate": {"status": "blocked" if reasons else "open" if status in ("completed", "recorded") else "unknown",
                "reasons": _unique(reasons)}})
    pending_ids.update("workflow:" + reason for reason in verification_reasons + trace_reasons)
    manual = bool(pending_ids)
    stages.append({"id": "human_review", "label": "人工复核与最终意见", "agents": ["human"],
        "observed_agents": [], "status": "required" if manual else "not_recorded",
        "gate": {"status": "pending" if manual else "unknown", "reasons": ["human_review_not_recorded"] if manual else []}})
    any_execution = any(row["execution_status"] != "not_recorded" for row in assessments) or bool(events)
    status = ("needs_manual" if manual else "partially_checked" if any_execution else "not_recorded")
    return {"status": status, "stages": stages, "manual_required": manual, "pending_count": len(pending_ids),
        "pending_item_ids": sorted(pending_ids), "core_nodes": copy.deepcopy(_CORE_NODES),
        "core_stages": copy.deepcopy(_CORE_STAGES),
        "execution_mode": _dict(stats.get("multi_agent")).get("execution_mode", "not_recorded"),
        "scope": "职责分层不代表各节点实际已执行，也不承诺并发；状态取自本报告执行与待办记录。"}


def build_review_framework(payload: dict[str, Any]) -> dict[str, Any]:
    """Build three linked layers without mutating the report or inventing a pass."""
    payload = _dict(payload)
    points = review_point_catalog()
    known = {point["point_id"] for point in points}
    code_points = {code: point["point_id"] for point in points for code in point["risk_codes"]}
    risks = [risk for risk in _rows(payload.get("risks")) if risk.get("code") != "no_substantive_risk"]
    explanations = _explanations(payload)
    risk_points, candidate_points = {}, {}
    for risk in risks:
        point = code_points.get(str(risk.get("code") or ""))
        if point:
            if risk.get("risk_id"):
                risk_points[str(risk["risk_id"])] = point
            for candidate in _candidate_ids(risk):
                candidate_points.setdefault(candidate, set()).add(point)
    explanation_points = [(item, _mapped_points(item, risk_points, candidate_points, known)) for item in explanations]
    clauses = {str(row["clause_id"]): str(row.get("text") or row.get("body") or "")
        for row in _rows(_dict(payload.get("document")).get("clauses")) if row.get("clause_id")}
    checks = _rows(_dict(payload.get("stats")).get("review_point_checks"))
    assessments = [_assessment(point,
        [risk for risk in risks if code_points.get(str(risk.get("code") or "")) == point["point_id"]],
        [item for item, mapping in explanation_points if point["point_id"] in mapping],
        [check for check in checks if check.get("point_id") == point["point_id"]],
        clauses, str(payload.get("contract_id") or "")) for point in points]
    unmapped = _unique(risk.get("risk_id") or "unknown-risk-" + str(index)
        for index, risk in enumerate(risks) if str(risk.get("code") or "") not in code_points)
    counts = Counter(row["status"] for row in assessments)
    workflow = _workflow(payload, explanations, assessments, unmapped)
    return {"schema_version": "contract-review-three-layers-v1",
        "context": copy.deepcopy(_dict(_dict(payload.get("stats")).get("review_context"))),
        "workflow": workflow, "review_points": points, "assessments": assessments,
        "summary": {"total": len(points), "by_status": {status: counts[status] for status in _STATUS_LABELS},
            "manual_required": sum(row["manual_required"] for row in assessments),
            "pending_count": workflow["pending_count"], "unmapped_risk_ids": unmapped,
            "scope": _LIMIT, "legal_truth_established": False},
        "scope": _LIMIT, "legal_truth_established": False}


def review_framework_markdown(framework: dict[str, Any]) -> list[str]:
    """Render the same three layers for text reports, including bounded scope."""
    framework = _dict(framework)
    workflow = _dict(framework.get("workflow"))
    lines = ["## 工作流 / 审查点 / 评估结果", "", str(framework.get("scope") or _LIMIT), "",
        "### 工作流", "", f"- 流程状态：{workflow.get('status', 'not_recorded')}；待处理记录：{workflow.get('pending_count', 0)}。"]
    for stage in _rows(workflow.get("stages")):
        lines.append(f"- {stage.get('label', stage.get('id', ''))}：{stage.get('status', 'not_recorded')}")
    lines += ["", "### 审查点与评估结果", ""]
    for row in _rows(framework.get("assessments")):
        lines += [f"#### {row.get('question', row.get('point_id', ''))}", "",
            f"- 审查标准：{row.get('criterion', '')}",
            f"- 评估结果：{row.get('status_label', row.get('status', 'not_checked'))}",
            f"- 实际问题／执行情况：{row.get('actual_issue', '')}",
            f"- 可能影响：{row.get('impact', '')}",
            f"- 处理建议：{row.get('recommendation', '')}",
            f"- 适用范围：{row.get('scope', '')}"]
        if row.get("risk_ids"):
            lines.append("- 关联风险：" + "、".join(_strings(row.get("risk_ids"))))
        if row.get("explanation_item_ids"):
            lines.append("- 解释记录：" + "、".join(_strings(row.get("explanation_item_ids"))))
        for evidence in _rows(row.get("evidence")):
            if evidence.get("quote"):
                lines.append(f"- 证据〔{evidence.get('clause_id', '')}；{'原文可定位' if evidence.get('located') else '引文待核对'}〕：{evidence['quote']}")
        if row.get("pending_reasons"):
            lines.append("- 待核验事项：" + "、".join(_strings(row.get("pending_reasons"))))
        lines.append("")
    return lines
