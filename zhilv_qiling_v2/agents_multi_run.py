"""多智能体层（下）：六个 Agent 的具体实现与调度器。

上半部分（`agents_multi.py`）给了 Message / MessageBus / Agent / AgentContext 这套骨架，
这里把五个功能节点包成有身份的 Agent，并加一个真正的调度器——它按拓扑发消息、
处理回边、收集结果，最后把整条通信轨迹导成 JSON。

要展示"多智能体"而不是"五个函数顺序调用"，差别就在三处：
  1. 每个 Agent 只从**消息队列**里取输入，不去读别人的中间变量；
  2. 拓扑外的投递会被 `MessageBus` 拒收（`strict=True`），图有约束力；
  3. 轨迹可导出（`MultiAgentRun.trace()`），谁在什么时候给谁发了什么，一目了然。
"""

from __future__ import annotations

import json
import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import agents as A
from . import nodes as N
from . import text as T
from .agents_multi import (
    AGENT_SPECS,
    Agent,
    AgentContext,
    AgentTrace,
    Message,
    MessageBus,
    PLUGIN_HANDLERS,
    TOPOLOGY,
    _jsonable,
    agent_specs,
)
from .config import Config
from .models import Contract, Citation, FinalRisk, RiskCandidate, Evidence, SpanRef, dedupe_candidates, stable_id
from .review_work_items import ReviewWorkLedger, initial_work_items, feedback_work_items, fingerprint as work_fingerprint
from .feedback_dependencies import IncrementalDispatch, dependency_graph
from .feedback_cycle import FeedbackCycle

# 每个 Agent 可以绑不同模型：申报书说"派发给专门的大模型或工具"，
# 这里让路由是声明式的——改这张表就能换模型，不用改调用代码。
MODEL_ROUTES: dict[str, dict[str, str]] = {
    "orchestrator": {"kind": "dispatch", "model": "router", "note": "按风险密度决定派谁、用哪档执行体"},
    "ner": {"kind": "local", "model": "bert-base-chinese", "note": "BERT-CRF 9 类 BIO"},
    "logic": {"kind": "llm", "model": "legalone", "note": "规则+检测器为主，模型为可选分支"},
    "subject": {"kind": "local", "model": "qualification-kg", "note": "15 条资质图谱 + BM25"},
    "statute": {"kind": "local", "model": "bert-dual-tower", "note": "RRF 检索 + 双塔判别"},
    "reviewer": {"kind": "llm", "model": "legalone-8b", "note": "只对高风险条款跑强模型（贵、慢）"},
    "judge": {"kind": "hybrid", "model": "legalone", "note": "确定性打分 + 模型复核意见"},
}


def _cfg_for(agent: str, config: Config) -> Config:
    """按路由给某个 Agent 生成本地配置副本（模型端点、温度、预算可各自不同）。"""
    clone = Config(**{**config.__dict__}) if hasattr(config, "__dict__") else config
    route = MODEL_ROUTES.get(agent, {})
    if route.get("kind") in ("llm", "hybrid"):
        clone.enable_llm = bool(getattr(config, "enable_llm", False))
    else:
        # 非 LLM 节点的 Agent 不去调大模型，避免"每个节点都在等模型"
        clone.enable_llm = False
    return clone


# ---------------------------------------------------------------- 各 Agent 的 handle


def _handle_intake(ctx: AgentContext) -> dict[str, Any]:
    """入口 Agent：把合同切分结果交给主脑，由主脑决定派活。"""
    doc = ctx.contract.document
    clauses = list(doc.clauses)
    brief = {
        "contract_id": ctx.contract.contract_id,
        "contract_type": ctx.contract.contract_type,
        "clauses": len(clauses),
        "chars": len(doc.full_text or ""),
        "clause_ids": [c.clause_id for c in clauses[:80]],
    }
    for peer in TOPOLOGY["intake"]:
        ctx.emit("intake", peer, "contract.parsed", brief)
    ctx.shared["clauses"] = clauses
    return {"notes": [f"切分 {len(clauses)} 条款，交主脑调度"]}


def _handle_orchestrator(ctx: AgentContext) -> dict[str, Any]:
    """主脑：第一轮派活（分档位），第二轮按 logic 回报的高风险条款派强模型复核。

    这是申报书里"各节点将子任务派发给专门的大模型或工具"的落点——
    派谁、用什么档位，由这里按合同特征决定，而不是写死在代码里。
    """
    config = ctx.config
    clauses = list(ctx.shared.get("clauses") or [])
    notes: list[str] = []

    # —— 第二轮：logic 已回报缺陷，挑选需要强模型复核的条款 ——
    defects_msgs = ctx.bus.inbox("orchestrator", kinds=["defects.detected"])
    if defects_msgs:
        top_k = max(1, int(getattr(config, "llm_review_top_k", 3) or 3))
        severity_rank = {"high": 3, "medium": 2, "low": 1}
        scored: list[tuple[float, str]] = []
        for msg in defects_msgs:
            for item in (msg.payload.get("defects") or []):
                rank = severity_rank.get(str(item.get("severity", "")), 1)
                score = rank * 100 + float(item.get("confidence") or 0)
                for cid in (item.get("clause_ids") or []):
                    scored.append((score, str(cid)))
        scored.sort(reverse=True)
        picked: list[str] = []
        for _score, cid in scored:
            if cid and cid not in picked:
                picked.append(cid)
            if len(picked) >= top_k:
                break

        strong = (getattr(config, "llm_strong_endpoint", "") or "").strip()
        if not strong:
            notes.append(f"未配置强模型端点，跳过 {len(picked)} 条复核")
            return {"notes": notes}
        if not picked:
            notes.append("无高风险条款，无需强模型复核")
            return {"notes": notes}

        by_id = {c.clause_id: c for c in clauses}
        chosen = [by_id[cid] for cid in picked if cid in by_id]
        if not chosen:
            notes.append("高风险条款不在切分结果内，跳过复核")
            return {"notes": notes}
        ctx.emit(
            "orchestrator",
            "reviewer",
            "review.assign",
            {
                "clause_ids": [c.clause_id for c in chosen],
                "endpoint": strong,
                "model": getattr(config, "llm_strong_model", "") or "legalone-8b",
                "reason": f"按严重度+置信取前 {len(chosen)} 条",
            },
        )
        notes.append(f"派发 {len(chosen)} 条高风险条款给强模型复核")
        return {"notes": notes}

    # —— 第一轮：按合同规模决定各节点的档位 ——
    chars = len(ctx.contract.document.full_text or "")
    plan = {
        "ner": {"tier": "local", "model": "bert-base-chinese"},
        "logic": {
            "tier": "llm" if getattr(config, "enable_llm", False) else "rules",
            "model": getattr(config, "llm_model", "") or "rules",
        },
        "subject": {"tier": "local", "model": "qualification-kg"},
        "statute": {"tier": "local", "model": "bert-dual-tower"},
    }
    focus = "长合同" if chars > 8000 else ("中合同" if chars > 3000 else "短合同")
    for peer in ("ner", "logic", "subject", "statute"):
        ctx.emit("orchestrator", peer, "task.assign", {"node": peer, **plan.get(peer, {})})
    ctx.shared["plan"] = plan
    notes.append(f"{focus}（{chars} 字）｜{len(clauses)} 条款，{len(plan)} 个节点已派活")
    notes.append("档位：" + "，".join(f"{k}={v['tier']}" for k, v in plan.items()))
    return {"notes": notes}


def _handle_reviewer(ctx: AgentContext) -> dict[str, Any]:
    """强模型复核：只跑主脑指派的那几条高风险条款（8B 全量跑不动）。"""
    assignments = ctx.bus.inbox("reviewer", kinds=["review.assign"])
    if not assignments:
        return {"notes": ["未收到复核指派"]}
    task = assignments[-1].payload
    endpoint = str(task.get("endpoint") or "").strip()
    model = str(task.get("model") or "legalone-8b")
    wanted = list(task.get("clause_ids") or [])
    clauses = [c for c in (ctx.shared.get("clauses") or []) if c.clause_id in wanted]
    if not clauses:
        return {"notes": ["指派条款不在切分结果内"]}

    # 单独造一份只指向强模型端点的配置，避免动到主配置
    clone = Config(**{**ctx.config.__dict__}) if hasattr(ctx.config, "__dict__") else ctx.config
    clone.enable_llm = True
    clone.llm_endpoint = endpoint
    clone.llm_model = model
    clone.llm_max_clauses = len(clauses)
    # 强模型慢：沿用 cheap 档的 90 s 会把每条都判成超时（实测 8B 4bit 256 token 约 110 s）
    clone.llm_timeout = float(getattr(ctx.config, "llm_strong_timeout", 600.0) or 600.0)

    started = time.time()
    outcome = N.node_logic(ctx.contract, clone, clauses=clauses)
    elapsed = time.time() - started
    # 只要模型产出的结论：规则与检测器在 cheap 档已经跑过一遍，不能重复计数
    candidates = [c for c in outcome.candidates if "llm" in str(getattr(c, "source", ""))]
    ctx.shared["review"] = candidates
    payload = {
        "clauses": len(clauses),
        "candidates": len(candidates),
        "model": model,
        "seconds": round(elapsed, 1),
        "codes": [c.code for c in candidates][:8],
    }
    ctx.emit("reviewer", "judge", "review.done", payload)
    notes = [f"{model} 复核 {len(clauses)} 条 → {len(candidates)} 条结论（{elapsed:.0f}s）"]
    if outcome.degraded:
        notes.append("降级：" + "；".join(outcome.degraded[:2]))
    return {"notes": notes, "candidates": candidates}


def _handle_ner(ctx: AgentContext) -> dict[str, Any]:
    outcome = N.node_ner(ctx.contract, _cfg_for("ner", ctx.config))
    ctx.shared["ner"] = outcome
    # ContractElements 是分组容器（parties/amounts/dates/subjects/...），不是可迭代列表
    els = ctx.contract.elements
    elements: list[dict[str, Any]] = []
    for kind in ("parties", "amounts", "dates", "subjects"):
        for el in list(getattr(els, kind, []) or [])[:12]:
            elements.append(
                {
                    "kind": str(getattr(el, "kind", kind[:-1])),
                    "value": str(getattr(el, "value", "")),
                    "source": str(getattr(el, "source", "")),
                    "clause_id": str(getattr(el, "clause_id", "") or ""),
                }
            )
    payload = {
        "elements": elements[:40],
        "parties": [e["value"] for e in elements if e["kind"] == "party"][:6],
        "amounts": [e["value"] for e in elements if e["kind"] == "amount"][:6],
        "dates": [e["value"] for e in elements if e["kind"] == "date"][:6],
    }
    # 要素沿三条边同时下发——这正是"要素经边同时送③④"的实现
    for peer in TOPOLOGY["ner"]:
        ctx.emit("ner", peer, "elements.extracted", payload)
    return {"notes": [outcome.summary()] + outcome.degraded[:2]}


def _latest(ctx: AgentContext, agent: str, kind: str) -> dict[str, Any]:
    messages = ctx.bus.inbox(agent, kinds=[kind])
    return messages[-1].payload if messages else {}


def _candidate_from_payload(item: dict[str, Any]) -> RiskCandidate:
    spans = [SpanRef(**v) for v in item.get("spans", [])]
    evidence = []
    for value in item.get("evidence", []):
        value = dict(value)
        if isinstance(value.get("span"), dict):
            value["span"] = SpanRef(**value["span"])
        evidence.append(Evidence(**value))
    clean = {k: v for k, v in item.items() if k not in ("spans", "evidence")}
    return RiskCandidate(**clean, spans=spans, evidence=evidence)


def _register_outcome(ctx: AgentContext, outcome: N.NodeOutcome) -> None:
    ctx.degraded.extend(f"{outcome.node}:{reason}" for reason in outcome.degraded)


def _subject_fact_feedback(ctx: AgentContext, subject: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind node③'s supplied records/unknowns to the clauses node② must revisit."""
    result: list[dict[str, Any]] = []
    clauses = ctx.contract.document.clauses
    candidate_rows = list(subject.get("candidates") or []) + list(_latest(ctx, "logic", "statute.matched").get("risk_checks") or [])
    identities_by_candidate: dict[str, set[tuple[str, ...]]] = {}
    for row in candidate_rows:
        extra = row.get("extra") if isinstance(row, dict) else None
        if not isinstance(extra, dict) or not row.get("candidate_id"):
            continue
        identity = tuple(str(extra.get(field) or "") for field in
            ("subject_node_id", "subject_name", "subject_role", "subject_id", "fact_id", "fact_type"))
        if any(identity):
            identities_by_candidate.setdefault(str(row["candidate_id"]), set()).add(identity)
    ambiguous_candidate_ids = {cid for cid, identities in identities_by_candidate.items() if len(identities) > 1}

    def private_gaps(checked: dict[str, Any], fact_type: str, cid: str,
                     request: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        request = request or {}
        gaps = request.get("unavailable_evidence") or checked.get("evidence_gaps", [])
        return [gap for gap in gaps if isinstance(gap, dict)
                and gap.get("availability") == "unavailable_private"
                and fact_type in gap.get("fact_types", [])
                and (not gap.get("clause_ids") or cid in gap["clause_ids"])]

    def targets(checked: dict[str, Any], record: dict[str, Any], fact_type: str,
                cid: str, request: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Name/clause co-occurrence is never a subject identity binding."""
        name = str(checked.get("subject_name") or record.get("subject_name") or "")
        node_id = str(checked.get("subject_node_id") or "")
        binding = checked.get("identity_binding") if isinstance(checked.get("identity_binding"), dict) else {}
        identifiers = binding.get("identifiers") or checked.get("subject_identifiers") or []
        identity_bound = (binding.get("status", checked.get("identity_binding_status")) == "bound"
            and isinstance(identifiers, list) and len(identifiers) == 1
            and bool(record.get("subject_id")) and identifiers[0] == record.get("subject_id")
            and record.get("identity_match") == "name_and_identifier"
            and record.get("subject_name") == name)
        selected: set[str] = set()
        errors: list[str] = []
        for row in candidate_rows:
            extra = row.get("extra") if isinstance(row, dict) else None
            if (not isinstance(extra, dict) or not row.get("candidate_id")
                    or cid not in (row.get("clause_ids") or []) or extra.get("subject_name") != name):
                continue
            candidate_id = str(row["candidate_id"])
            if candidate_id in ambiguous_candidate_ids:
                errors.append("ambiguous_candidate_identity:" + candidate_id)
                continue
            if extra.get("subject_node_id") and node_id and extra["subject_node_id"] != node_id:
                continue
            if extra.get("fact_id") and extra["fact_id"] != record.get("fact_id"):
                continue
            if extra.get("fact_type") and extra["fact_type"] != fact_type:
                continue
            exact_node = bool(node_id and extra.get("subject_node_id") == node_id)
            exact_fact = bool(record.get("fact_id") and extra.get("fact_id") == record["fact_id"]
                and extra.get("fact_type") == fact_type and identity_bound)
            if exact_node or exact_fact:
                selected.add(candidate_id)
            else:
                errors.append("subject_candidate_identity_not_bound:" + candidate_id)
        # A resolved upstream request names one exact obligation. Preserve that
        # ID without expanding it to other candidates sharing a clause/name.
        explicit = str(request.get("candidate_id") or "")
        if explicit:
            if explicit in ambiguous_candidate_ids:
                errors.append("ambiguous_candidate_identity:" + explicit)
            else:
                selected.add(explicit)
        return sorted(selected), list(dict.fromkeys(errors))

    def append(checked: dict[str, Any], record: dict[str, Any], fact_type: str,
               clause_ids: list[str], *, request: dict[str, Any] | None = None) -> None:
        request = request or {}
        name = str(checked.get("subject_name") or record.get("subject_name") or "")
        statuses = checked.get("checks") or {}
        status = str(record.get("effective_status") or record.get("status") or "unknown")
        aggregate = request.get("fact_status") or statuses.get(fact_type)
        if aggregate in ("unknown", "conflicting_evidence"):
            status = str(aggregate)
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        source_id = stable_id("subject_source", json.dumps(source, sort_keys=True, ensure_ascii=False), length=24) if source else ""
        source_complete = bool(record.get("fact_id") and all(isinstance(source.get(key), str) and source[key].strip()
            for key in ("url", "publisher", "verified_at", "reviewer", "verification_method"))
            and str(source.get("url", "")).startswith(("http://", "https://")))
        fact_id = str(record.get("fact_id") or "missing-evidence:" + stable_id("unknown_subject_check", name, fact_type,
            checked.get("subject_node_id", ""), request.get("request_id", ""), length=24))
        fact_text = json.dumps({"fact_id": fact_id, "subject_name": name, "subject_id": record.get("subject_id", ""),
            "fact_type": fact_type, "effective_status": status, "scope": record.get("scope", ""),
            "valid_from": record.get("valid_from", ""), "valid_until": record.get("valid_until", ""),
            "review_reason": record.get("review_reason") or request.get("reason") or
                ("no_matching_evidence" if not record else ""),
            "evidence_ref": record.get("evidence_ref", ""), "identity_match": record.get("identity_match", "")},
            sort_keys=True, ensure_ascii=False)
        for cid in clause_ids or [""]:
            candidate_ids, candidate_binding_errors = targets(checked, record, fact_type, cid, request)
            item = {"feedback_kind": "subject_fact", "fact_id": fact_id, "source_id": source_id,
                "subject_name": name, "subject_node_id": checked.get("subject_node_id", ""),
                "subject_role": checked.get("subject_role", ""), "subject_id": record.get("subject_id", ""),
                "identity_binding": checked.get("identity_binding", {}),
                "fact_type": fact_type, "effective_status": status, "source": source,
                "record_provided": bool(record), "check_states": statuses,
                "source_complete": source_complete, "fact_evidence_text": fact_text,
                "source_evidence_text": json.dumps(source, sort_keys=True, ensure_ascii=False) if source else "",
                "clause_id": cid, "clause_ids": [cid] if cid else [], "candidate_ids": candidate_ids,
                "candidate_binding_status": "pending_manual" if candidate_binding_errors else "bound" if candidate_ids else "no_candidate",
                "candidate_binding_errors": candidate_binding_errors,
                "request_id": request.get("request_id", ""), "upstream_pending": bool(request and request.get("status") != "completed"),
                "provenance_mode": checked.get("source_mode", "reviewed_local_evidence" if source else "no_matching_evidence"),
                "independently_verified": False}
            # Availability is a scheduling condition, never a factual verdict.
            gaps = private_gaps(checked, fact_type, cid, request)
            if status in ("unknown", "conflicting_evidence") and gaps:
                item.update(unavailable_evidence=gaps, auto_retry=False, retry_trigger="new_evidence")
            item["feedback_item_id"] = stable_id("subject_feedback", json.dumps(item, sort_keys=True, ensure_ascii=False), length=24)
            result.append(item)

    for checked in subject.get("fact_checks") or []:
        if not isinstance(checked, dict):
            continue
        name = str(checked.get("subject_name") or "")
        clause_ids = list(dict.fromkeys(str(cid) for cid in checked.get("clause_ids", []) if cid))
        if not clause_ids:
            clause_ids = [clause.clause_id for clause in clauses if name and name in clause.text]
        records = [record for record in checked.get("records", []) if isinstance(record, dict)]
        types = sorted(set((checked.get("checks") or {}).keys()) | {str(record.get("fact_type")) for record in records if record.get("fact_type")})
        for kind in types:
            relevant = [record for record in records if record.get("fact_type") == kind]
            for record in relevant:
                append(checked, record, kind, clause_ids)
        missing = [kind for kind in types if not any(record.get("fact_type") == kind for record in records)]
        if missing:
            # A registry's default four unknowns are one unresolved subject
            # check unless a specific private file prevents a particular type
            # in a particular clause. Never freeze public registration/credit
            # work merely because the same party's authorization is private.
            for cid in clause_ids or [""]:
                private_types = [kind for kind in missing if private_gaps(checked, kind, cid)]
                for kind in private_types:
                    unknown = dict(checked, checks={kind: "unknown"})
                    append(unknown, {}, kind, [cid] if cid else [])
                remaining = [kind for kind in missing if kind not in private_types]
                if remaining:
                    unknown = dict(checked, checks={kind: "unknown" for kind in remaining})
                    append(unknown, {}, "unresolved_subject_checks", [cid] if cid else [])
    # Every statute→subject request is a separate obligation. Unbound requests
    # deliberately have no matching review window and remain pending.
    for request in subject.get("verification_results") or []:
        if not isinstance(request, dict):
            continue
        checked = {key: request.get(key, "") for key in ("subject_name", "subject_node_id", "subject_role")}
        records = [record for record in request.get("records", []) if isinstance(record, dict)]
        for record in records or [{}]:
            append(checked, record, str(request.get("fact_type") or "unknown"),
                list(dict.fromkeys(str(cid) for cid in request.get("clause_ids", []) if cid)), request=request)
    return list({item["feedback_item_id"]: item for item in result}.values())


def _statute_feedback_item(check: dict[str, Any]) -> dict[str, Any]:
    """Give the complete supplied evidence a stable ID before node② bounds it.

    Include source text and every check field, not merely candidate/article IDs.
    An upstream stale ID cannot make changed evidence reuse an earlier receipt.
    """
    item = copy.deepcopy(check)
    item.pop("feedback_item_id", None)
    quote = str(item.get("article_quote") or "")
    item["article_quote"] = quote
    item["article_quote_truncated"] = bool(item.get("article_quote_truncated") or len(quote) > 1000)
    item["feedback_item_id"] = stable_id("statute_feedback",
        json.dumps(item, sort_keys=True, ensure_ascii=False), length=24)
    return item


def _live_feedback_request(ctx: AgentContext):
    statute = _latest(ctx, "logic", "statute.matched")
    subject = _latest(ctx, "logic", "subject.checked")
    checks = list(statute.get("risk_checks") or [])
    subject_feedback = _subject_fact_feedback(ctx, subject)
    feedback_key = hashlib.sha256(json.dumps({"checks": checks,
        "facts": subject.get("fact_checks", []), "subject_feedback": subject_feedback,
        "verification_results": subject.get("verification_results", []),
        "verification_complete": subject.get("verification_complete"),
        "pending_verification_request_ids": subject.get("pending_verification_request_ids", [])},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    focus = {cid for check in checks if check.get("label") == "contradict" or check.get("disagreement") or check.get("manual_required")
             for cid in check.get("clause_ids", [])}
    relevant = [_statute_feedback_item(check) for check in checks
                if focus.intersection(check.get("clause_ids", []))]
    focus.update(cid for item in subject_feedback for cid in item.get("clause_ids", []))
    return feedback_key, checks, subject, focus, relevant, subject_feedback


def _feedback_request(ctx: AgentContext):
    live = _live_feedback_request(ctx)
    cycle = ctx.shared.get("feedback_cycle")
    if cycle is not None:
        cycle.observe(live)
        return cycle.request()
    return live


def _feedback_pending(ctx: AgentContext) -> dict[str, Any]:
    key, _, subject, focus, relevant, facts = _feedback_request(ctx)
    required = bool((focus and relevant) or facts or subject.get("pending_verification_request_ids")
                    or subject.get("verification_complete") is False)
    active_pending = required and key not in ctx.shared.get("processed_feedback", set())
    latest = ctx.shared.get("feedback_latest", {}).get(key, {})
    verified = set(latest.get("verified_feedback_item_ids", []))
    # A new fingerprint has no response snapshot yet. Its pending ledger must
    # still include *all* current obligations, including statute checks. Older
    # fingerprint acknowledgements cannot complete these changed inputs.
    expected_ids = [item["feedback_item_id"] for item in [*relevant, *facts]]
    item_ids = list(dict.fromkeys(list(latest.get("uncompleted_feedback_item_ids", [])) +
        list(latest.get("unmatched_feedback_item_ids", [])) +
        [item_id for item_id in expected_ids if item_id not in verified])) if active_pending else []
    cycle = ctx.shared.get("feedback_cycle")
    deferred_ids = [item["work_item_id"] for item in cycle.deferred_items()] if cycle is not None else []
    item_ids += deferred_ids
    pending = bool(active_pending or deferred_ids)
    required = bool(required or deferred_ids)
    return {"required": required, "pending": pending, "fingerprint": key if required else None,
            "item_ids": item_ids, "pending_subject_request_ids": list(subject.get("pending_verification_request_ids", []))}


def _review_work_state(ctx: AgentContext) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Observe existing receipts without manufacturing any legacy acknowledgement."""
    key, _, subject, _, statutes, facts = _feedback_request(ctx)
    logic = ctx.shared.get("logic")
    initial = dict(getattr(logic, "llm", {}) or {})
    latest = ctx.shared.get("feedback_latest", {}).get(key, {})
    items = initial_work_items(ctx.contract.contract_id, initial)
    items += feedback_work_items(ctx.contract.contract_id, statutes, facts, subject, latest,
        _current_feedback_consumption(ctx),
        {clause.clause_id: clause.text for clause in ctx.contract.document.clauses})
    if getattr(ctx.config, "llm_feedback_window_policy", "all_matching_windows") == "cover_all_context_v1":
        for item in items:
            provided = item.get("provided_feedback", {})
            if (item.get("kind") == "subject_fact"
                    and (not provided.get("source_complete") or not provided.get("source_id"))):
                # Match node2's manual-only plan. Missing external sources
                # cannot be obtained by repeating the unchanged model request.
                # Keep every pending/source/semantic flag and any real receipt.
                item["dispatchable"] = False
                item["dispatch_policy_reason"] = "external_subject_source_required"
                item["dispatch_policy"] = "cover_all_context_v1"
    ledger = ctx.shared.setdefault("review_work_ledger", ReviewWorkLedger())
    scheduler = ctx.shared.setdefault("incremental_dispatch", IncrementalDispatch())
    graph = dependency_graph(items)
    cycle = ctx.shared.get("feedback_cycle")
    if cycle is not None:
        cycle.bind_graph(items, graph)
    scheduler.bind_prior_processing(items, graph)
    scheduler.record_processed(items, graph)
    if cycle is not None:
        cycle.update_status(items)
        items += cycle.deferred_items()
    ledger.observe(items, int(ctx.shared.get("round", 1)))
    ctx.shared["feedback_dependency_graph"] = graph
    return items, graph


def _current_feedback_consumption(ctx: AgentContext) -> list[dict[str, Any]]:
    """Project only the latest fingerprint's bound receipts for human review.

    Protocol completion and semantic disagreement are deliberately separate:
    rejected/contradictory opinions can finish the protocol yet require a human.
    This projection never changes node② coverage, risk confidence or scores.
    """
    fingerprint, _, _, _, statutes, facts = _feedback_request(ctx)
    inputs = [*statutes, *facts]
    latest = ctx.shared.get("feedback_latest", {}).get(fingerprint, {})
    latest = latest if isinstance(latest, dict) else {}
    calls = [call for call in latest.get("calls", []) if isinstance(call, dict)]
    global_verified = set(latest.get("verified_feedback_item_ids", []))
    global_pending = set(latest.get("uncompleted_feedback_item_ids", [])) | set(latest.get("unmatched_feedback_item_ids", []))
    attempts = getattr(ctx.shared.get("logic"), "payload", {}).get("feedback_attempts", [])
    round_no = next((attempt.get("round") for attempt in reversed(attempts)
                     if attempt.get("fingerprint") == fingerprint), None)
    counts: dict[str, int] = {}
    for item in inputs:
        item_id = str(item.get("feedback_item_id") or "")
        counts[item_id] = counts.get(item_id, 0) + 1
    records = []
    for item in inputs:
        item_id = str(item.get("feedback_item_id") or "")
        subject_fact = item.get("feedback_kind") == "subject_fact"
        target_ids = item.get("candidate_ids", []) if subject_fact else [item.get("candidate_id")]
        candidate_ids = sorted({str(value) for value in target_ids if value})
        errors: list[str] = []
        observations: list[dict[str, Any]] = []
        if not latest:
            errors.append("current_fingerprint_not_attempted")
        if not item_id or counts.get(item_id) != 1:
            errors.append("duplicate_or_missing_feedback_input_id")
        bound_calls = [call for call in calls if item_id in call.get("feedback_item_ids", [])]
        if not bound_calls:
            errors.append("no_bound_feedback_response")
        root = "subject_fact_verifications" if subject_fact else "verifications"
        binding_fields = ("fact_id", "source_id", "subject_name", "fact_type", "clause_id") if subject_fact else ("candidate_id", "article_uid")
        for call in bound_calls:
            call_errors = [str(error.get("reason") or "feedback_validation_error")
                for error in call.get("verification_errors", []) if isinstance(error, dict)
                and (error.get("feedback_item_id") == item_id or not error.get("feedback_item_id"))]
            rows = [row for row in call.get(root, []) if isinstance(row, dict)
                    and row.get("feedback_item_id") == item_id]
            if len(rows) != 1:
                call_errors.append("missing_or_duplicate_bound_verification")
            for row in rows:
                row_errors = list(call_errors)
                if any(row.get(field) != item.get(field) for field in binding_fields):
                    row_errors.append("feedback_binding_mismatch")
                if row.get("verdict") not in N.FEEDBACK_VERDICTS:
                    row_errors.append("invalid_feedback_verdict")
                if call.get("status") in ("error", "partial", "invalid_schema"):
                    row_errors.append("feedback_call_" + str(call["status"]))
                if item_id not in call.get("verified_feedback_item_ids", []):
                    row_errors.append("feedback_response_not_acknowledged")
                observations.append({"job_id": call.get("job_id"), "window_id": call.get("window_id"),
                    "call_status": call.get("status"), "verdict": row.get("verdict"),
                    "reason": row.get("reason", ""),
                    "clause_quote_mode": str(call.get("clause_quote_mode") or "unspecified"),
                    "clause_quote_mode_note": "合同引文来自原文片段选择，相关性仍需复核" if call.get("clause_quote_mode") == "native_source_extract_selection" else "",
                    "quotes": {field: row.get(field, "") for field in
                        (("clause_quote", "fact_quote", "source_quote") if subject_fact else ("clause_quote", "article_quote"))},
                    "validation_errors": list(dict.fromkeys(row_errors))})
                errors.extend(row_errors)
            errors.extend(call_errors)
        if item_id not in global_verified or item_id in global_pending:
            errors.append("feedback_item_pending")
        if subject_fact:
            if not item.get("source_complete") or not item.get("source_id"):
                errors.append("subject_fact_source_incomplete")
            if item.get("effective_status") in (None, "", "unknown", "conflicting_evidence"):
                errors.append("subject_fact_status_unresolved")
            if item.get("upstream_pending"):
                errors.append("subject_fact_upstream_request_pending")
        if item.get("article_quote_truncated") or item.get("fact_evidence_text_truncated") or item.get("source_evidence_text_truncated"):
            errors.append("feedback_evidence_truncated")
        errors = list(dict.fromkeys(errors))
        verdicts = sorted({str(row["verdict"]) for row in observations if row.get("verdict") in N.FEEDBACK_VERDICTS})
        disagreement = len(verdicts) > 1
        protocol_complete = bool(observations and not errors)
        candidate_binding_errors = list(item.get("candidate_binding_errors") or [])
        upstream_review_reasons = []
        if not subject_fact:
            if (item.get("input_truncated") or item.get("token_coverage_complete") is False
                    or item.get("pair_coverage_complete") is False
                    or (item.get("coverage_complete") is False and not item.get("semantic_pending"))):
                upstream_review_reasons.append("entailment_input_coverage_incomplete")
            if item.get("semantic_pending") or item.get("decision_scope") == "local_window_only":
                upstream_review_reasons.append("entailment_local_window_only")
            if item.get("window_disagreement"):
                upstream_review_reasons.append("entailment_window_disagreement")
            if item.get("disagreement"):
                upstream_review_reasons.append("neural_rule_disagreement")
            if item.get("manual_required") and not upstream_review_reasons:
                upstream_review_reasons.append("upstream_statute_manual_required")
        pending_manual = bool(not protocol_complete or disagreement or candidate_binding_errors
            or upstream_review_reasons or any(verdict != "confirmed" for verdict in verdicts))
        record = {"feedback_fingerprint": fingerprint, "feedback_item_id": item_id,
            "feedback_kind": "subject_fact" if subject_fact else "statute", "current": True,
            "round": round_no, "candidate_ids": candidate_ids, "clause_ids": list(item.get("clause_ids") or []),
            "protocol_complete": protocol_complete, "verification_status": "complete" if protocol_complete else "pending",
            "semantic_status": "pending_manual" if pending_manual else "provided_relation_confirmed",
            "pending_manual": pending_manual, "disagreement": disagreement,
            "upstream_review_reasons": upstream_review_reasons,
            "verdicts": verdicts, "observations": observations, "validation_errors": errors,
            "candidate_binding_status": item.get("candidate_binding_status", "bound" if candidate_ids else "no_candidate"),
            "candidate_binding_errors": candidate_binding_errors,
            "meaning": "模型只复核给定关系；协议完成不证明法律结论或外部来源真实，不改变评分"}
        record.update({field: item[field] for field in
            ("candidate_id", "article_uid", "fact_id", "source_id", "subject_name", "fact_type", "request_id",
             "input_truncated", "coverage_complete", "token_coverage_complete", "pair_coverage_complete",
             "semantic_pending", "window_disagreement", "decision_scope", "window_audit") if field in item})
        cycle = ctx.shared.get("feedback_cycle")
        if cycle is not None:
            record.update(active_snapshot_only=True, feedback_cycle_id=cycle.cycle_id)
            if cycle.deferred_affects(item):
                record["newer_input_pending"] = True
                record["pending_manual"] = True
                record["semantic_status"] = "pending_manual"
                record["upstream_review_reasons"].append("newer_evidence_deferred_next_cycle")
        records.append(record)
    return records


def _attach_current_feedback(ctx: AgentContext, candidates: Sequence[RiskCandidate]) -> list[dict[str, Any]]:
    """Attach audit-only evidence by explicit candidate/alias identity, never clause proximity."""
    records = _current_feedback_consumption(ctx)
    for candidate in candidates:
        candidate.evidence = [evidence for evidence in candidate.evidence
            if not (evidence.kind == "model_feedback" and evidence.source == "node2_feedback")]
        candidate.extra.pop("model_feedback_current", None)
        aliases = {candidate.candidate_id, *candidate.extra.get("alias_candidate_ids", [])}
        matched = [record for record in records if aliases.intersection(record["candidate_ids"])]
        if not matched:
            continue
        candidate.extra["model_feedback_current"] = copy.deepcopy(matched)
        for record in matched:
            verdicts = "/".join(record["verdicts"]) or "未取得有效逐项结论"
            status = "协议完成" if record["protocol_complete"] else "协议待核验"
            manual = "；人工待复核" if record["pending_manual"] else ""
            binding = ("法条" + str(record.get("article_uid", ""))) if record["feedback_kind"] == "statute" else (
                "主体" + str(record.get("subject_name", "")) + "／事实" + str(record.get("fact_id", "")) + "／来源" + str(record.get("source_id", "")))
            details = []
            for observation in record["observations"]:
                quotes = "；".join(f"{field}={quote}" for field, quote in observation["quotes"].items() if quote)
                quote_mode = "合同引文模式=" + observation["clause_quote_mode"]
                if observation["clause_quote_mode_note"]:
                    quote_mode += "（" + observation["clause_quote_mode_note"] + "）"
                details.append(f"{observation['verdict']}：{observation['reason']}；{quote_mode}" + ("；" + quotes if quotes else ""))
            if record["validation_errors"]:
                details.append("校验／待核验原因：" + ", ".join(record["validation_errors"]))
            if record.get("upstream_review_reasons"):
                details.append("上游关系仍待人工核验：" + ", ".join(record["upstream_review_reasons"]))
            if record["candidate_binding_errors"]:
                details.append("候选身份待核验：" + ", ".join(record["candidate_binding_errors"]))
            text = (f"当前模型反馈 {record['feedback_item_id']}（指纹{record['feedback_fingerprint'][:12]}，"
                f"{binding}）：{verdicts}；{status}{manual}"
                + ("；跨窗口意见不一致" if record["disagreement"] else "") + "。" + "；".join(details)
                + "。仅核对给定关系，不证明法律结论或来源真实，不改变评分。")
            candidate.evidence.append(Evidence("model_feedback", text, 0.0, source="node2_feedback",
                extra=copy.deepcopy(record)))
    logic = ctx.shared.get("logic")
    if logic is not None:
        logic.payload["feedback_consumption_current"] = copy.deepcopy(records)
        logic.llm["feedback_consumption_current"] = copy.deepcopy(records)
    return records


def _handle_logic(ctx: AgentContext) -> dict[str, Any]:
    previous = ctx.shared.get("logic")
    feedback_key, checks, subject, focus, relevant_checks, subject_feedback = _feedback_request(ctx)
    if previous is None:
        outcome = N.node_logic(ctx.contract, _cfg_for("logic", ctx.config))
    else:
        # 原全文结果保留；仅对新增逐项证据涉及的条款补跑模型，避免每轮全文重复调用。
        outcome = copy.deepcopy(previous)
        items, graph = _review_work_state(ctx)
        dispatch = ctx.shared["incremental_dispatch"].plan(items, graph, int(ctx.shared.get("round", 1)))
        reuse_only = bool(dispatch["reused_work_item_ids"] and not dispatch["dispatch_work_item_ids"])
        ctx.shared["last_incremental_dispatch"] = dispatch
        if feedback_key not in ctx.shared.setdefault("processed_feedback", set()) and not reuse_only:
            if focus and getattr(ctx.config, "enable_llm", False):
                clauses = [c for c in ctx.contract.document.clauses if c.clause_id in focus]
                # 同一 focus 上的支持/中性结果也要核对；其他条款仍保留在法条审计，
                # 但不应被记成本次局部模型回环未送达的反馈。
                fresh = N.node_logic(ctx.contract, _cfg_for("logic", ctx.config), clauses=clauses,
                                     verification_feedback=relevant_checks,
                                     subject_fact_feedback=subject_feedback)
                outcome.candidates = dedupe_candidates(outcome.candidates + fresh.candidates)
                attempt_index = len(outcome.payload.get("feedback_runs", [])) + 1
                attempt_binding = {"attempt_id": stable_id("fb_attempt", ctx.contract.contract_id,
                        feedback_key, ctx.shared.get("round", 1), attempt_index, length=24),
                    "attempt_index": attempt_index, "fingerprint": feedback_key,
                    "round": ctx.shared.get("round", 1),
                    "run_sha256": hashlib.sha256(json.dumps(fresh.llm, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()}
                outcome.payload.setdefault("feedback_run_bindings", []).append(attempt_binding)
                ctx.shared.setdefault("feedback_latest_binding", {})[feedback_key] = copy.deepcopy(attempt_binding)
                outcome.payload.setdefault("feedback_runs", []).append(fresh.llm)
                ctx.shared.setdefault("feedback_latest", {})[feedback_key] = copy.deepcopy(fresh.llm)
                outcome.payload.setdefault("feedback_attempts", []).append({
                    "fingerprint": feedback_key, "round": ctx.shared.get("round", 1),
                    "attempt_id": attempt_binding["attempt_id"], "attempt_index": attempt_index,
                    "complete": fresh.llm.get("complete") is True and fresh.llm.get("feedback_complete") is True})
                outcome.degraded.extend(fresh.degraded)
                if (fresh.llm.get("complete") is True and fresh.llm.get("feedback_complete") is True
                        and not fresh.llm.get("uncompleted_feedback_item_ids")
                        and not fresh.llm.get("unmatched_feedback_item_ids")
                        and subject.get("verification_complete") is not False
                        and not subject.get("pending_verification_request_ids")):
                    ctx.shared["processed_feedback"].add(feedback_key)
            elif not _feedback_pending(ctx)["required"]:
                ctx.shared["processed_feedback"].add(feedback_key)
        elif reuse_only:
            # Keep latest raw calls, pending IDs and fingerprints untouched.
            # An unchanged needs_manual response may avoid another identical
            # model request, but it never becomes a completed verification.
            outcome.payload.setdefault("incremental_feedback_reuse", []).append(copy.deepcopy(dispatch))
    candidates = list(outcome.candidates)
    for candidate in candidates:
        matching = [c for c in checks if c.get("candidate_id") == candidate.candidate_id]
        for check in matching:
            text = f"候选{candidate.candidate_id} / 条款{','.join(check.get('clause_ids', []))} / 法条{check.get('article_uid')}: {check.get('label')}"
            candidate.evidence_append("cross_validation", text, min(0.8, float(check.get("score", 0.5))), "statute_node")
        if matching:
            candidate.extra["statute_checks"] = matching
            candidate.extra["cross_validation_status"] = "pending_manual" if any(c.get("label") == "contradict" or c.get("disagreement") or c.get("manual_required") for c in matching) else "relation_checked"
        for fact in subject.get("fact_checks", []):
            if fact.get("subject_name") and any(fact["subject_name"] in (ctx.contract.clause_map().get(cid).text if cid in ctx.contract.clause_map() else "") for cid in candidate.clause_ids):
                candidate.extra.setdefault("subject_checks", {})[fact["subject_name"]] = fact["checks"]
    outcome.payload["cross_validation"] = {"round": int(ctx.shared.get("round", 1)),
        "matched_candidates": sum(bool(c.extra.get("statute_checks")) for c in candidates),
        "feedback_fingerprint": feedback_key, "feedback_state": _feedback_pending(ctx)}
    _register_outcome(ctx, outcome)
    ctx.shared["logic"] = outcome
    _review_work_state(ctx)
    focus = list(dict.fromkeys(cid for c in candidates for cid in c.clause_ids))
    payload = {"defects": [{"candidate_id": c.candidate_id, "code": c.code,
        "severity": c.severity, "clause_ids": list(c.clause_ids), "confidence": float(c.confidence)} for c in candidates],
        "candidates": [c.to_dict() for c in candidates], "count": len(candidates),
        "focus_clause_ids": focus, "round": ctx.shared.get("round", 1)}
    for peer in TOPOLOGY["logic"]:
        ctx.emit("logic", peer, "defects.detected", payload)
    return {"notes": [outcome.summary(), outcome.payload["cross_validation"]], "candidates": candidates}


def _handle_subject(ctx: AgentContext) -> dict[str, Any]:
    logic = _latest(ctx, "subject", "defects.detected")
    statute = _latest(ctx, "subject", "statute.matched")
    focus = list(logic.get("focus_clause_ids") or [])
    requests = [check for check in statute.get("risk_checks", [])
                if check.get("risk_code") in ("need_license", "need_authorization", "illegal_terms", "subject_unspecified")]
    outcome = N.node_subject(ctx.contract, ctx.config, focus_clause_ids=focus,
                             verification_requests=requests)
    outcome.payload["focus_from_node2"] = focus
    _register_outcome(ctx, outcome)
    ctx.shared["subject"] = outcome
    hint = dict(outcome.payload.get("hint") or {})
    payload = {"conclusions": len(outcome.candidates), "hint": hint,
        "candidates": [c.to_dict() for c in outcome.candidates],
        "fact_checks": outcome.payload.get("stats", {}).get("fact_checks", []),
        "verification_requests": requests, "focus_from_node2": focus,
        "verification_results": outcome.payload.get("stats", {}).get("verification_results", []),
        "verification_complete": outcome.payload.get("stats", {}).get("verification_complete"),
        "pending_verification_request_ids": outcome.payload.get("stats", {}).get("pending_verification_request_ids", []),
        "round": ctx.shared.get("round", 1)}
    for peer in TOPOLOGY["subject"]:
        ctx.emit("subject", peer, "subject.checked", payload)
    return {"notes": [outcome.summary()], "candidates": list(outcome.candidates)}


def _handle_statute(ctx: AgentContext) -> dict[str, Any]:
    # 从显式边消息重建候选，不隐式读取其它节点共享对象。
    logic = _latest(ctx, "statute", "defects.detected")
    subject = _latest(ctx, "statute", "subject.checked")
    if not logic or not subject:
        ctx.degraded.append("statute:upstream_missing")
        return {"error": "upstream_missing"}
    candidates = [_candidate_from_payload(item) for item in
                  list(logic.get("candidates") or []) + list(subject.get("candidates") or [])]
    hint = dict(subject.get("hint") or {})
    if hint.get("needs_government_rules"):
        hint.setdefault("contract_type", "政府采购")
    outcome = N.node_statute(ctx.contract, candidates, ctx.config, retriever=ctx.retriever, hint=hint)
    _register_outcome(ctx, outcome)
    ctx.shared["statute"] = outcome
    checks = list(outcome.payload.get("risk_checks") or [])
    counts = (outcome.payload.get("entailment") or {}).get("counts", {})
    payload = {"citations": len(outcome.citations), "entailment_counts": counts,
        "conflicts": sum(c.get("label") == "contradict" for c in checks),
        "risk_checks": checks, "round": ctx.shared.get("round", 1),
        "articles": [f"{c.law_name}{c.article_no}" for c in outcome.citations]}
    for peer in TOPOLOGY["statute"]:
        ctx.emit("statute", peer, "statute.matched", payload)
    conflicts = [c for c in checks if c.get("label") == "contradict"]
    if conflicts:
        ctx.emit("statute", "logic", "statute.conflict", {"conflicts": len(conflicts), "items": conflicts})
    return {"notes": [outcome.summary()], "citations": list(outcome.citations)}


def _handle_judge(ctx: AgentContext) -> dict[str, Any]:
    logic = ctx.shared.get("logic")
    subject = ctx.shared.get("subject")
    statute = ctx.shared.get("statute")
    logic_cands = list(getattr(logic, "candidates", []) or [])
    subject_cands = list(getattr(subject, "candidates", []) or [])
    review_cands = list(ctx.shared.get("review") or [])
    per_candidate = dict(((getattr(statute, "payload", {}) or {}).get("per_candidate") or {}))
    # Subject candidates are recreated on each round, so bind the current
    # receipts immediately before final fusion. Older fingerprints stay in stats.
    _attach_current_feedback(ctx, logic_cands + subject_cands + review_cands)
    # fuse 要的是 AgentResult 列表（带 agent 名，用于 agent_sources 与引用归并）
    agent_results = [
        A.AgentResult(agent="node2_logic", candidates=logic_cands),
        A.AgentResult(agent="node3_subject", candidates=subject_cands),
        A.AgentResult(agent="node4_statute", candidates=[], citations=per_candidate),
    ]
    if review_cands:
        agent_results.append(A.AgentResult(agent="reviewer_strong", candidates=review_cands))
    fused = A.fuse(
        logic_cands + subject_cands + review_cands,
        agent_results,
        ctx.contract,
        _cfg_for("judge", ctx.config),
    )
    unresolved = {str(item.get("candidate_id")) for item in
                  getattr(statute, "payload", {}).get("risk_checks", [])
                  if item.get("label") == "contradict" or item.get("disagreement") or item.get("manual_required")}
    pending_ids = {stable_id("rk", c.code, c.candidate_id) for c in logic_cands + subject_cands
                   if c.candidate_id in unresolved or c.extra.get("cross_validation_status") == "pending_manual"}
    for risk in fused:
        if risk.risk_id in pending_ids:
            risk.review_required = True
            risk.rationale += "；存在逐项法条关系冲突、神经模型与规则判别分歧或输入覆盖不完整，保留原始结果与限制；回环稳定不代表这些待核验原因已消除，请人工核对。"
        if any(evidence.kind == "model_feedback" and evidence.source == "node2_feedback"
               and evidence.extra.get("current") and evidence.extra.get("pending_manual") for evidence in risk.evidence):
            risk.review_required = True
            risk.rationale += "；当前模型逐项反馈存在不支持、待核验、校验失败或意见分歧，请查看本项模型反馈证据后人工复核；未自动删除风险或调整评分。"
    outcome = N.node_judge(fused, _cfg_for("judge", ctx.config), ctx.contract)
    _register_outcome(ctx, outcome)
    ctx.shared["judge"] = outcome
    ctx.shared["fused"] = fused
    payload = {
        "score": outcome.payload.get("score"),
        "breakdown": outcome.payload.get("breakdown"),
        "summary": outcome.payload.get("summary"),
        "focus": outcome.payload.get("focus"),
        "evidence_basis": outcome.payload.get("evidence_basis"),
    }
    # ⑤ 是终点（TOPOLOGY["judge"] 为空），没有下游可发；保留 payload 供轨迹展示
    ctx.shared["judge_payload"] = payload
    return {"notes": [outcome.summary()] + outcome.degraded[:2], "score": outcome.payload.get("score", 100)}


HANDLERS: dict[str, Callable[[AgentContext], dict[str, Any]]] = {
    "intake": _handle_intake,
    "orchestrator": _handle_orchestrator,
    "ner": _handle_ner,
    "logic": _handle_logic,
    "subject": _handle_subject,
    "statute": _handle_statute,
    "reviewer": _handle_reviewer,
    "judge": _handle_judge,
}


def build_agents(config: Config | None = None) -> dict[str, Agent]:
    """按 agent_specs()（内置 + 插件）+ MODEL_ROUTES 造 Agent 实例。"""
    from .agents_multi import agent_specs

    agents: dict[str, Agent] = {}
    for name, spec in agent_specs().items():
        route = MODEL_ROUTES.get(name, {})
        handler = HANDLERS.get(name) or PLUGIN_HANDLERS.get(name)
        if handler is None:
            continue
        agents[name] = Agent(
            name=name,
            role=spec["role"],
            executor=spec["executor"],
            model=route.get("model", spec.get("model", "n/a")),
            handle=handler,
            budget_seconds=float(getattr(config or Config(), "llm_timeout", 90.0)),
        )
    return agents


@dataclass
class MultiAgentRun:
    """一次多智能体审查的结果 + 完整通信轨迹。"""

    contract_id: str
    order: list[str]
    score: int
    risks: list[FinalRisk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    candidates: list[RiskCandidate] = field(default_factory=list)
    bus: MessageBus | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def trace(self) -> AgentTrace:
        return AgentTrace(
            trace_id=self.bus.trace_id if self.bus else "",
            order=self.order,
            steps=self.steps,
            bus=self.bus or MessageBus(),
            degraded=self.degraded,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "order": self.order,
            "score": self.score,
            "risks": len(self.risks),
            "citations": len(self.citations),
            "messages": len(self.bus.log) if self.bus else 0,
            "edges": self.bus.stats()["edges_used"] if self.bus else {},
            "degraded": self.degraded,
            "elapsed": round(self.elapsed, 3),
            "diagnostics": self.diagnostics,
            "steps": [
                {k: s.get(k) for k in ("agent", "role", "model", "status", "seconds", "inbox", "outbox")}
                for s in self.steps
            ],
        }

    def dump_trace(self) -> str:
        """完整通信轨迹（JSON 字符串）：给答辩演示与问题复现用。

        除了 steps，还带 messages —— 只有边计数看不出“谁给谁发了什么”，
        而拓扑是否真的约束住通信，恰恰要看消息本身。
        """
        payload = self.trace().as_dict()
        payload["messages"] = self.bus.dump() if self.bus else []
        return json.dumps(payload, ensure_ascii=False, indent=2)


# 调度顺序 = 拓扑的拓扑排序结果（固定写出来，便于复现与文档引用）
# 主脑出现两次：开工前派活，logic 回报缺陷后再派一次强模型复核。
SCHEDULE: tuple[str, ...] = (
    "intake", "orchestrator", "ner", "logic", "reviewer", "subject", "statute", "judge",
)


def run_multi_agent(
    contract: Contract,
    config: Config | None = None,
    *,
    retriever: Any = None,
    strict_topology: bool = True,
    feedback_cycle_state: dict[str, Any] | None = None,
) -> MultiAgentRun:
    """按图调度 Agent 跑一遍，返回结果与通信轨迹。

    与 `nodes.run_five_nodes` 的区别：那边是函数顺序调用，这边每个 Agent 只读自己的
    收件箱、只按拓扑发消息；`strict_topology=True` 时拓扑外的投递会被拒收并计数。
    """
    config = config or Config()
    clause_sources = {clause.clause_id: clause.text for clause in contract.document.clauses}
    if feedback_cycle_state is not None:
        if not getattr(config, "freeze_feedback_cycle", False):
            raise ValueError("feedback_cycle_continuation_requires_opt_in")
        FeedbackCycle.validate_scope(feedback_cycle_state, contract_id=contract.contract_id,
                                     clause_sources=clause_sources)
    started = time.time()

    if retriever is None:
        try:
            from .config import V1_STATUTES
            from .retrieve import get_retriever

            retriever = get_retriever(config.statute_file(), config)
        except Exception:
            retriever = None

    bus = MessageBus(strict=strict_topology)
    ctx = AgentContext(contract=contract, config=config, bus=bus, retriever=retriever)
    agents = build_agents(config)

    def _step(name: str) -> None:
        agent = agents.get(name)
        if agent is not None:
            agent.run(ctx)

    ctx.shared["round"] = 1
    _step("intake")
    _step("orchestrator")
    _step("ner")
    _step("logic")
    if bus.inbox("orchestrator", kinds=["defects.detected"]):
        _step("orchestrator")
        _step("reviewer")
    _step("subject")
    _step("statute")
    if getattr(config, "freeze_feedback_cycle", False):
        live_request = _live_feedback_request(ctx)
        ctx.shared["feedback_cycle"] = (FeedbackCycle.advance(feedback_cycle_state,
            contract_id=contract.contract_id, clause_sources=clause_sources, request=live_request)
            if feedback_cycle_state is not None else
            FeedbackCycle(contract.contract_id, clause_sources, live_request))
    _review_work_state(ctx)

    def fingerprint() -> str:
        logic = ctx.shared.get("logic")
        subject = ctx.shared.get("subject")
        statute = ctx.shared.get("statute")
        state = {"candidates": sorted((c.candidate_id, c.code, tuple(c.clause_ids))
            for c in list(getattr(logic, "candidates", [])) + list(getattr(subject, "candidates", []))),
            "facts": getattr(subject, "payload", {}).get("stats", {}).get("fact_checks", []),
            "statutes": getattr(statute, "payload", {}).get("risk_checks", []),
            "feedback": _feedback_pending(ctx)}
        return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    rounds = [{"round": 1, "fingerprint": fingerprint(), "changed": True}]
    maximum = max(1, min(8, int(getattr(config, "review_max_rounds", 3))))
    termination = "round_limit"
    for round_no in range(2, maximum + 1):
        ctx.shared["round"] = round_no
        before = fingerprint()
        work_before, _ = _review_work_state(ctx)
        def work_state_key(items: list[dict[str, Any]]) -> str:
            return work_fingerprint(sorted((item["work_item_id"], item["input_fingerprint"],
                item["source_status"], item["semantic_status"], item["pending_reasons"])
                for item in items))
        before_work_key = work_state_key(work_before)
        for agent in ("logic", "subject", "statute"):
            _step(agent)
        after = fingerprint()
        changed = before != after
        feedback_state = _feedback_pending(ctx)
        work_after, _ = _review_work_state(ctx)
        pending_work_ids = [item["work_item_id"] for item in work_after if item["pending"]]
        work_changed = before_work_key != work_state_key(work_after)
        dispatch = ctx.shared.get("last_incremental_dispatch", {})
        reuse_only = bool(dispatch.get("reused_work_item_ids") and not dispatch.get("dispatch_work_item_ids"))
        rounds.append({"round": round_no, "fingerprint": after, "changed": changed,
                       "feedback_pending": feedback_state["pending"],
                       "pending_feedback_item_ids": feedback_state["item_ids"],
                       "pending_subject_verification_request_ids": feedback_state["pending_subject_request_ids"],
                       "work_state_changed": work_changed, "pending_work_item_ids": pending_work_ids,
                       "incremental_dispatch": copy.deepcopy(dispatch)})
        if not changed and not feedback_state["pending"] and not pending_work_ids:
            termination = "stable"
            break
        no_dispatchable_work = not any(item.get("dispatchable") for item in work_after)
        if ((feedback_state["pending"] or pending_work_ids) and not work_changed
                and (reuse_only or no_dispatchable_work)):
            termination = "stalled_with_pending"
            break
    final_feedback = _feedback_pending(ctx)
    final_items, _ = _review_work_state(ctx)
    work_ledger = ctx.shared["review_work_ledger"].snapshot()
    ctx.shared["cross_validation"] = {"rounds": rounds, "max_rounds": maximum,
        "termination": termination, "converged": termination == "stable",
        "feedback_verification_complete": not final_feedback["pending"],
        "pending_feedback_fingerprints": [final_feedback["fingerprint"]] if final_feedback["pending"] else [],
        "pending_feedback_item_ids": final_feedback["item_ids"],
        "pending_subject_verification_request_ids": final_feedback["pending_subject_request_ids"],
        "pending_work_item_ids": work_ledger["pending_work_item_ids"],
        "work_items_complete": not work_ledger["pending_work_item_ids"],
        "meaning": "结构化结论稳定不代表法律判断正确；冲突及未知事实仍需人工复核"}
    if termination == "round_limit":
        ctx.degraded.append("cross_validation_round_limit")
    if final_feedback["pending"]:
        ctx.degraded.append("cross_validation_feedback_pending")
    if termination == "stalled_with_pending":
        ctx.degraded.append("cross_validation_stalled_with_pending")
    _step("judge")

    fused = list(ctx.shared.get("fused") or [])
    judge = ctx.shared.get("judge")
    score = int(getattr(judge, "payload", {}).get("score", 100))
    citations = list(getattr(ctx.shared.get("statute"), "citations", []) or [])
    candidates = (
        list(getattr(ctx.shared.get("logic"), "candidates", []) or [])
        + list(getattr(ctx.shared.get("subject"), "candidates", []) or [])
        + list(ctx.shared.get("review") or [])
    )
    logic = ctx.shared.get("logic")
    logic_llm = dict(getattr(logic, "llm", {}))
    feedback_runs = list(getattr(logic, "payload", {}).get("feedback_runs", []))
    logic_llm["feedback_runs"] = feedback_runs
    logic_llm["review_work_items"] = work_ledger
    logic_llm["feedback_dependency_graph"] = copy.deepcopy(ctx.shared["feedback_dependency_graph"])
    logic_llm["incremental_dispatch"] = ctx.shared["incremental_dispatch"].snapshot()
    logic_llm["feedback_all_attempts_complete"] = all(
        run.get("complete", False) and run.get("feedback_complete", False)
        for run in feedback_runs)
    current = ctx.shared.get("feedback_latest", {}).get(final_feedback["fingerprint"])
    logic_llm["feedback_latest_runs"] = [current] if current is not None else []
    logic_llm["feedback_attempts"] = getattr(logic, "payload", {}).get("feedback_attempts", [])
    logic_llm["feedback_run_bindings"] = getattr(logic, "payload", {}).get("feedback_run_bindings", [])
    current_key, _, current_subject, current_focus, current_statutes, current_facts = _feedback_request(ctx)
    current_binding = ctx.shared.get("feedback_latest_binding", {}).get(final_feedback["fingerprint"])
    logic_llm["feedback_current_request"] = {
        "required": final_feedback["required"], "fingerprint": final_feedback["fingerprint"],
        "attempt_id": current_binding.get("attempt_id") if current_binding is not None else None,
        "attempt_index": current_binding.get("attempt_index") if current_binding is not None else None,
        "focus_clause_ids": sorted(current_focus), "items": copy.deepcopy([*current_statutes, *current_facts]),
        "subject_verification_complete": current_subject.get("verification_complete"),
        "pending_subject_request_ids": list(current_subject.get("pending_verification_request_ids", []))}
    logic_llm["feedback_runs_complete"] = bool(not final_feedback["pending"] and all(
        run.get("complete", False) and run.get("feedback_complete", False)
        for run in logic_llm["feedback_latest_runs"]))
    cycle = ctx.shared.get("feedback_cycle")
    if cycle is not None:
        cycle_state = cycle.snapshot()
        cycle_state["termination"] = termination
        # This is an additional diagnostic, never a replacement for pending gates.
        logic_llm["feedback_cycle"] = cycle_state
        ctx.shared["cross_validation"]["feedback_cycle_id"] = cycle.cycle_id
        ctx.shared["cross_validation"]["deferred_next_cycle_work_item_ids"] = cycle_state["deferred_pending_work_item_ids"]
        ctx.shared["cross_validation"]["active_cycle_attempt_complete"] = cycle_state["active_attempt_complete"]
        ctx.shared["cross_validation"]["feedback_verification_complete"] = bool(
            not final_feedback["pending"] and cycle_state["active_attempt_complete"]
            and not cycle_state["deferred_pending_work_item_ids"])
        logic_llm["feedback_runs_complete"] = bool(logic_llm["feedback_runs_complete"]
            and ctx.shared["cross_validation"]["feedback_verification_complete"])

    return MultiAgentRun(
        contract_id=contract.contract_id,
        order=[s["agent"] for s in ctx.steps],
        score=score,
        risks=fused,
        citations=citations,
        candidates=candidates,
        bus=bus,
        steps=ctx.steps,
        degraded=ctx.degraded,
        elapsed=time.time() - started,
        diagnostics={"logic_llm": logic_llm,
                     "judge_evidence": getattr(ctx.shared.get("judge"), "payload", {}),
                     "subject": getattr(ctx.shared.get("subject"), "payload", {}),
                     "judge_llm": getattr(ctx.shared.get("judge"), "llm", {}),
                     "statute_checks": getattr(ctx.shared.get("statute"), "payload", {}).get("risk_checks", []),
                     "cross_validation": ctx.shared.get("cross_validation", {}),
                     "review_work_items": work_ledger,
                     "feedback_dependency_graph": copy.deepcopy(ctx.shared["feedback_dependency_graph"]),
                     "incremental_dispatch": ctx.shared["incremental_dispatch"].snapshot()},
    )
