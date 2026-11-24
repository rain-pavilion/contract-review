"""Read-only, item-level explanations of a saved review and its evidence.

This projection never changes risks, scores, feedback or pending ledgers. A
located quotation and a successful model receipt are evidence about execution
and supplied text; neither certifies legal truth or an external company fact.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any


_REASONS = {
    "feedback_item_pending": "该反馈尚未完成核验",
    "feedback_verification_needs_manual": "模型明确要求人工复核",
    "verification_needs_manual": "模型明确要求人工复核",
    "neural_rule_disagreement": "神经模型与规则的判断不一致",
    "entailment_local_window_only": "仅核对了局部窗口，尚不能确认完整上下文",
    "entailment_window_disagreement": "不同原文窗口的判断不一致",
    "entailment_input_coverage_incomplete": "条款与法条的输入覆盖不完整",
    "subject_fact_source_incomplete": "主体证据来源不完整",
    "subject_fact_status_unresolved": "主体事实仍为未知或存在冲突",
    "external_subject_evidence_pending": "尚缺独立主体事实证据",
    "subject_verification_request_pending": "主体核验请求仍未完成",
    "newer_evidence_deferred_next_cycle": "新证据尚未进入本轮核验",
    "bound_processing_receipt_missing_or_invalid": "没有与本项完整绑定的成功调用记录",
    "no_bound_feedback_response": "没有对应本项的模型反馈",
    "current_fingerprint_not_attempted": "当前证据版本尚未完成模型核验",
    "feedback_evidence_truncated": "核验所需的证据被截断",
    "claim_grounding_pending": "原文尚不足以支持该项完整主张",
    "contract_quote_missing_or_unlocated": "缺少可定位到对应条款的合同引文",
    "article_quote_missing_or_unlocated": "缺少可定位到对应法条的引文",
    "human_review_requested": "机器报告要求人工复核该风险",
    "model_relation_not_confirmed": "模型未一致确认给定关系",
    "source_authenticity_not_verified": "证据来源及主体事实未获独立认证",
    "source_status_unknown": "主体事实为待核验，导入材料不改变此状态",
    "protocol_not_complete": "本项反馈协议尚未完整通过",
    "upstream_statute_manual_required": "法条适用关系仍需人工确认",
    "subject_fact_upstream_request_pending": "主体资格或授权所需的上游证据尚未补齐",
    "subject_fact_verification_needs_manual": "模型要求人工核对主体证据",
    "missing_verification": "本项没有得到完整的模型回复",
    "initial_lineage_not_available": "缺少该次初审的原文及续审记录",
    "initial_source_selection_invalid": "模型选取的原文依据未通过来源校验",
    "initial_max_attempts_exhausted": "该原文范围已到尝试上限，仍需人工补核",
    "initial_response_invalid": "该原文范围的模型答复不完整或格式无效",
    "feedback_verification_incomplete": "逐项反馈核验尚未完成",
    "review_work_state_unresolved": "逐项工作台账的执行、语义或来源状态缺失、未知或不一致",
    "review_state_unknown": "核验状态缺失或使用未知取值，需人工确认",
    "decision_scope_unresolved": "模型判别范围未知或不足以覆盖当前完整输入",
    "unavailable_private": "用户说明相关文件涉及隐私、无法提供；对应事实仍待核验",
    "document_evidence_missing": "合同包所需材料尚未提供或尚未核验",
    "private_evidence_unavailable_wait_for_new_evidence": "用户说明相关隐私文件无法提供，本项等待新证据并继续待核验",
    "ocr_numeric_field_unverified": "关联日期或金额来自尚未人工逐字核对的OCR，不能直接认定为已证实的数值冲突",
    "contract_source_ocr_not_human_checked": "合同依据来自未经逐字核对的OCR，应先对照对应扫描原件",
    "statute_source_ocr_not_human_checked": "法条依据来自未经逐字核对的OCR，应先核对官方原件中的对应文字",
}


def _rows(value: Any) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _located(quote: Any, source: Any) -> bool:
    return bool(_compact(quote) and _compact(quote) in _compact(source))


def _manual(row: dict, reasons: list[str]) -> dict:
    row["manual_reasons"] = _unique(reasons)
    row["reason_details"] = [{"code": reason, "text": _reason_text(reason)}
                             for reason in row["manual_reasons"]]
    row["manual_required"] = bool(reasons)
    row["legal_truth_established"] = False
    steps = _manual_steps(row["manual_reasons"])
    if steps:
        row["next_action"] = "；".join(steps) + "。"
    row["next_actions"] = steps or [row.get("next_action", "核对原文和依据后记录人工意见。")]
    return row


def _manual_steps(reasons: list[str]) -> list[str]:
    """Instructions for a future reviewer, never a claim of human completion."""
    codes = set(reasons)
    steps = []
    private_unavailable = bool(codes.intersection({"unavailable_private", "private_evidence_unavailable_wait_for_new_evidence"}))
    if {"ocr_numeric_field_unverified", "contract_source_ocr_not_human_checked", "statute_source_ocr_not_human_checked"} & codes:
        steps.append("打开关联原始PDF及标注页码，逐字核对手写日期、金额、单位、小数点和签署位置，并区分原合同与补充协议；OCR读数未确认前保持待核验，不将识别异常当成已经证实的合同冲突")
    if any(any(token in code for token in ("local_window", "coverage", "window_disagreement", "decision_scope"))
           for code in codes):
        steps.append("打开关联条款的完整上下文及原合同附件，核对是否为同一主体、同一标的、同一履约阶段和已触发的条件，再判断能否共同履行")
    if any(any(token in code for token in ("claim", "grounding", "absence", "unsupported", "literal_support"))
           for code in codes):
        steps.append("逐句对照风险主张与合同引文，检查引文是否支持完整主张；涉及缺少约定、仅限或全部等判断时查阅全文，依据不足的部分继续待核验")
    if any(any(token in code for token in ("neural_rule", "rule_disagreement", "model_relation_not_confirmed"))
           for code in codes):
        steps.append("并列核对规则判断、神经模型判断及各自理由，结合条款事实检查法条适用条件、例外和约定优先关系，记录采纳或不采纳的依据")
    if private_unavailable:
        steps.append("保留用户关于隐私材料无法提供的说明，逐项标注受影响的事实范围和结论限制；现有材料不足的事项保持待核验，取得新的脱敏证明、可核对来源或实际授权后再开启该项核验")
    elif "document_evidence_missing" in codes:
        steps.append("查看缺件名称、对应事实范围和现有来源，取得可提供的新材料后补核；没有证据的事项继续待核验")
    if not private_unavailable and any(any(token in code for token in ("external_subject", "subject_fact", "subject_verification", "identity",
                                          "source_authenticity", "source_status_unknown"))
           for code in codes):
        steps.append("补充对应主体的登记、经营资质或签约授权记录，核对名称、统一社会信用代码、来源机构、核验日期、有效期和适用范围；授权还需对应本合同，信用事项需有相应查询记录")
    if any(any(token in code for token in ("receipt", "no_bound_feedback", "fingerprint_not_attempted",
                                          "binding_mismatch", "missing_verification", "review_work_state",
                                          "protocol_not_complete", "feedback_call", "initial_lineage"))
           for code in codes):
        steps.append("核对本项真实调用的 call_id、job_id、window_id、source_sha256 与当前证据版本是否一致；回执或回复缺失时恢复或重跑该任务，保留原失败记录")
    if any(any(token in code for token in ("truncat", "evidence_truncated", "output_incomplete", "initial_response_invalid",
                                          "initial_max_attempts", "more_findings", "initial_output"))
           for code in codes):
        steps.append("取回完整原文和模型输出，检查截断位置及尚未覆盖的条款；必要时分段续审或调整输出预算，在完整输出和覆盖校验完成前保持 pending")
    if any(any(token in code for token in ("quote_missing", "quote_not", "source_binding", "source_selection_invalid"))
           for code in codes):
        steps.append("按条款编号和法条编号定位引用的连续原文，核对是否来自本项实际输入，补齐或纠正错绑引文后再判断其相关性")
    if "newer_evidence_deferred_next_cycle" in codes or "deferred_to_next_cycle" in codes:
        steps.append("核对新增证据与本周期快照的差异，在下一周期审查新版本；旧版本反馈不能替代新证据核验")
    if any(any(token in code for token in ("needs_manual", "human_review_requested")) for code in codes):
        steps.append("由复核人核对原文和法律依据，逐项填写采纳、否定或仍待补充的意见及理由；人工意见与机器状态分别保留")
    return _unique(steps)


def _reason_text(reason: str) -> str:
    if reason in _REASONS:
        return _REASONS[reason]
    if "ground" in reason or "claim" in reason:
        return "合同原文是否足以支持该项主张仍须人工核对"
    if "subject" in reason or "identity" in reason:
        return "主体身份、资质或授权证据仍未完整核对"
    if "window" in reason or "coverage" in reason:
        return "相关条款与完整上下文尚未核验完毕"
    if "quote" in reason or "source" in reason:
        return "引用原文或来源绑定未通过检查"
    if "receipt" in reason or "call" in reason or "binding" in reason:
        return "本项调用与证据的对应关系尚未完整核对"
    if "initial" in reason or "output" in reason:
        return "模型初审输出仍有未完成或待修正内容"
    return "本项核验仍有未决内容，需结合原文和调用记录确认"


def _receipt_bound(call: dict) -> bool:
    fields = ("call_id", "job_id", "window_id", "source_sha256")
    return any(receipt.get("status") in ("succeeded", "degraded")
               and (receipt.get("status") != "degraded" or receipt.get("reason") in
                    ("json_retry", "response_format_retry"))
               and all(call.get(key) and receipt.get(key) == call.get(key) for key in fields)
               for receipt in _rows(call.get("request_receipts")))


def _raw_pending_reasons(record: dict) -> list[str]:
    """Conserve raw uncertainty even when a derived summary says confirmed.

    Older archives may lack optional flags; when a flag is present, only its
    explicit resolved value clears that particular axis. These checks do not
    repair or overwrite the original flags or their evidence.
    """
    reasons = []
    for key, reason in (
        ("semantic_pending", "entailment_local_window_only"),
        ("window_disagreement", "entailment_window_disagreement"),
        ("disagreement", "neural_rule_disagreement"),
        ("input_truncated", "entailment_input_coverage_incomplete"),
        ("article_quote_truncated", "feedback_evidence_truncated"),
        ("fact_evidence_text_truncated", "feedback_evidence_truncated"),
        ("source_evidence_text_truncated", "feedback_evidence_truncated"),
        ("upstream_claim_grounding_pending", "claim_grounding_pending"),
        ("upstream_manual_required", "upstream_statute_manual_required"),
        ("manual_required", "upstream_statute_manual_required"),
        ("upstream_pending", "subject_fact_upstream_request_pending"),
        ("newer_input_pending", "newer_evidence_deferred_next_cycle"),
    ):
        if key in record and record[key] is not False:
            reasons.append(reason)
    for key in ("coverage_complete", "token_coverage_complete", "pair_coverage_complete", "input_valid"):
        if key in record and record[key] is not True:
            reasons.append("entailment_input_coverage_incomplete")
    if "source_complete" in record and record["source_complete"] is not True:
        reasons.append("subject_fact_source_incomplete")
    if "decision_scope" in record and record["decision_scope"] != "whole_input":
        reasons.append("entailment_local_window_only" if record["decision_scope"] == "local_window_only"
                       else "decision_scope_unresolved")
    if "claim_grounding_status" in record and record["claim_grounding_status"] not in (None, "supported"):
        reasons.append("claim_grounding_pending")
    for key in ("manual_reasons", "claim_grounding_manual_reasons", "semantic_pending_reasons", "review_flags"):
        values = record.get(key)
        if isinstance(values, list):
            reasons.extend(str(value) for value in values if value)
        elif values:
            reasons.append("review_state_unknown")
    return reasons


def build_review_explanations(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a full API/archive payload; safe for legacy and partial reports.

    Feed this after attaching ``document`` and ``subject_checks``. ``items``
    contains risks, individual feedback obligations, external subject facts and
    unrepresented initial-review work. ``sources`` preserves the source text
    referred to by item IDs. Human risk feedback remains a separate observation.
    """
    stats = _dict(payload.get("stats"))
    logic = _dict(stats.get("logic_llm"))
    clause_rows = {str(row["clause_id"]): row
                   for row in _rows(_dict(payload.get("document")).get("clauses")) if row.get("clause_id")}
    clauses = {cid: str(row.get("text") or row.get("body") or "") for cid, row in clause_rows.items()}
    works = _rows(_dict(logic.get("review_work_items") or stats.get("review_work_items")).get("items"))
    by_feedback = {str(row["legacy_feedback_item_id"]): row for row in works if row.get("legacy_feedback_item_id")}
    records = _rows(logic.get("feedback_consumption_current"))
    # An explicitly empty latest set means no current result. Historical calls
    # must not be presented as current successful execution.
    latest = _rows(logic.get("feedback_latest_runs"))
    calls = [call for run in latest for call in _rows(run.get("calls"))]
    articles: dict[str, dict] = {}
    items: list[dict] = []

    def evidence(clause_ids, quotes=()):
        result = []
        for row in quotes:
            cid, quote = str(row.get("clause_id") or ""), str(row.get("quote") or "")
            result.append({"clause_id": cid, "quote": quote,
                "located": _located(quote, clauses.get(cid)), "role": "cited_quote"})
        cited = {row["clause_id"] for row in result}
        for cid in _unique(clause_ids):
            if cid not in cited:
                result.append({"clause_id": cid, "quote": "", "located": False,
                               "role": "context_reference", "source_available": cid in clauses})
        for row in result:
            metadata = _dict(_dict(clause_rows.get(row["clause_id"])).get("metadata"))
            for key in ("source_id", "document_id", "source_title", "source_role", "source_page_start",
                        "source_page_end", "original_sha256", "extraction_quality"):
                if key in metadata:
                    row[key] = copy.deepcopy(metadata[key])
        return result

    def article(row):
        uid = str(row.get("article_uid") or "")
        if uid:
            existing = articles.setdefault(uid, {"article_uid": uid, "law_name": row.get("law_name", ""),
                "article_no": row.get("article_no", ""), "text": "", "source_url": row.get("source_url", "")})
            text = str(row.get("article_quote") or row.get("quote") or "")
            if len(text) > len(existing["text"]):
                existing["text"] = text
        return uid

    for work in works:
        article(_dict(work.get("provided_feedback")))
    for risk in _rows(payload.get("risks")):
        for citation in _rows(risk.get("citations")):
            article(citation)
        extras = [_dict(row.get("extra")) for row in _rows(risk.get("evidence"))]
        cids = _unique([extra.get("candidate_id") for extra in extras] +
                       [cid for extra in extras for cid in extra.get("candidate_ids", [])])
        relevant = [row for row in records if set(cids).intersection(row.get("candidate_ids", []))]
        quoted = evidence(risk.get("clause_ids", []), _rows(risk.get("spans")))
        reasons = ["human_review_requested"] if risk.get("review_required") else []
        numeric_risk = risk.get("code") in {"date_reversed", "date_missing", "fee_mismatch", "amount_missing", "amount_inconsistent"}
        numeric_risk = numeric_risk or bool(re.search(r"日期|金额|总价|价款|百分比|手写", str(risk.get("message") or "")))
        if numeric_risk and any(_dict(row.get("extraction_quality")).get("status") == "machine_ocr_unverified"
                                and _dict(row.get("extraction_quality")).get("human_verified") is not True for row in quoted):
            reasons.append("ocr_numeric_field_unverified")
        if not any(row["located"] for row in quoted):
            reasons.append("contract_quote_missing_or_unlocated")
        grounding = [extra for extra in extras if "claim_grounding_status" in extra]
        if any(extra.get("manual_required") or extra.get("claim_grounding_status") in
               ("unsupported", "pending", "partial", "needs_manual") for extra in grounding):
            reasons.append("claim_grounding_pending")
        for record in relevant:
            if record.get("pending_manual") or record.get("semantic_status") != "provided_relation_confirmed":
                reasons.extend(record.get("validation_errors", []))
                reasons.extend(record.get("upstream_review_reasons", []))
                reasons.append("feedback_item_pending")
        row = {"item_id": str(risk.get("risk_id") or ""), "kind": "risk", "risk_id": risk.get("risk_id"),
            "title": risk.get("message", "风险主张"), "candidate_ids": cids,
            "contract_evidence": quoted, "article_ids": _unique(c.get("article_uid") for c in _rows(risk.get("citations"))),
            "feedback_item_ids": _unique(r.get("feedback_item_id") for r in relevant),
            "evidence_status": "contract_quote_located" if any(q["located"] for q in quoted) else "missing_contract_quote",
            "claim_support": [str(g.get("claim_grounding_status")) for g in grounding],
            "model_verdicts": [], "calls": [],
            "human_feedback": copy.deepcopy(_dict(payload.get("feedback")).get(str(risk.get("risk_id")))),
            "status": "manual_review" if reasons else "text_evidence_located",
            "status_label": "待人工确认" if reasons else "原文可定位",
            "explanation": "展示机器风险主张与对应原文。引文可定位仅说明文本来源，仍须核对是否足以支持主张及法律适用。",
            "next_action": "结合关联条款核对风险主张，核对法条适用后在人工复核中记录意见。"}
        items.append(_manual(row, reasons))

    # Keep obligations present in the work ledger even if there is no model
    # response, rather than silently dropping those needing the most attention.
    current = {str(row.get("feedback_item_id")): row for row in records if row.get("feedback_item_id")}
    for fid, work in by_feedback.items():
        current.setdefault(fid, {"feedback_item_id": fid, "pending_manual": True,
            "semantic_status": work.get("semantic_status"), "protocol_complete": work.get("protocol_complete"),
            "clause_ids": work.get("clause_ids", [])})
    for fid, record in current.items():
        work = by_feedback.get(fid, {})
        supplied = _dict(work.get("provided_feedback"))
        kind = "subject_fact_feedback" if record.get("feedback_kind") == "subject_fact" or work.get("kind") == "subject_fact" else "statute_feedback"
        bound_calls = [call for call in calls if fid in call.get("feedback_item_ids", [])]
        receipts = []
        for call in bound_calls:
            response_rows = [row for row in _rows(call.get("subject_fact_verifications" if kind == "subject_fact_feedback" else "verifications"))
                             if row.get("feedback_item_id") == fid]
            binding_fields = ("fact_id", "source_id", "subject_name", "fact_type", "clause_id") if kind == "subject_fact_feedback" else ("candidate_id", "article_uid")
            expected = supplied or record
            response_bound = bool(len(response_rows) == 1 and all(expected.get(key) is not None
                and response_rows[0].get(key) == expected.get(key) for key in binding_fields)
                and fid in call.get("verified_feedback_item_ids", [])
                and call.get("status") == "ok"
                and not any(error.get("feedback_item_id") in (fid, "", None)
                            for error in _rows(call.get("verification_errors"))))
            if response_bound:
                quote_fields = ("clause_quote", "fact_quote", "source_quote") if kind == "subject_fact_feedback" else ("clause_quote", "article_quote")
                matches = [observation for observation in _rows(record.get("observations"))
                    if observation.get("job_id") == call.get("job_id")
                    and observation.get("window_id") == call.get("window_id")
                    and observation.get("verdict") == response_rows[0].get("verdict")
                    and all(_dict(observation.get("quotes")).get(key, "") == response_rows[0].get(key, "")
                            for key in quote_fields)]
                response_bound = bool(len(matches) == 1)
            receipts.append({key: call.get(key) for key in ("call_id", "window_id", "job_id", "source_sha256", "status")}
                | {"receipt_bound": _receipt_bound(call), "response_bound": response_bound})
        reasons = list(record.get("validation_errors", [])) + list(record.get("upstream_review_reasons", []))
        reasons += list(record.get("candidate_binding_errors", [])) + list(work.get("pending_reasons", []))
        for original in (record, supplied, _dict(record.get("window_audit")), _dict(supplied.get("window_audit"))):
            reasons.extend(_raw_pending_reasons(original))
        if record.get("protocol_complete") is not True:
            reasons.append("protocol_not_complete")
        if record.get("pending_manual") is not False or record.get("semantic_status") != "provided_relation_confirmed" or work.get("pending") is not False:
            reasons.append("feedback_item_pending")
        if (work.get("semantic_status") != "provided_relation_confirmed"
                or work.get("source_status") != "bound_provided_evidence"
                or work.get("execution_status") != "executed"
                or work.get("actual_processed") is not True
                or work.get("protocol_complete") is not True):
            reasons.append("review_work_state_unresolved")
        if any(fid in run.get(key, []) for run in latest for key in
               ("uncompleted_feedback_item_ids", "unmatched_feedback_item_ids")):
            reasons.append("feedback_item_pending")
        if fid in _dict(stats.get("cross_validation")).get("pending_feedback_item_ids", []):
            reasons.append("feedback_item_pending")
        if not receipts or not all(r["receipt_bound"] and r["response_bound"] for r in receipts):
            reasons.append("bound_processing_receipt_missing_or_invalid")
        reasons.extend(supplied.get("manual_reasons", []))
        if supplied.get("manual_required") or supplied.get("upstream_manual_required"):
            reasons.append("upstream_statute_manual_required")
        if supplied.get("disagreement") or record.get("disagreement"):
            reasons.append("neural_rule_disagreement")
        observations = _rows(record.get("observations"))
        verdicts = _unique(record.get("verdicts", []) + [o.get("verdict") for o in observations])
        if verdicts != ["confirmed"]:
            reasons.append("model_relation_not_confirmed")
        clause_ids = record.get("clause_ids") or supplied.get("clause_ids") or []
        quotes = []
        for observation in observations:
            quote = _dict(observation.get("quotes")).get("clause_quote", "")
            matching = [cid for cid in clause_ids if _located(quote, clauses.get(cid))]
            quotes.append({"clause_id": matching[0] if len(matching) == 1 else "", "quote": quote})
        quoted = evidence(clause_ids, quotes)
        if not quotes or not all(row["located"] for row in quoted if row["role"] == "cited_quote"):
            reasons.append("contract_quote_missing_or_unlocated")
        uid = str(record.get("article_uid") or supplied.get("article_uid") or "")
        article_quotes = [str(_dict(o.get("quotes")).get("article_quote") or "") for o in observations]
        if kind == "statute_feedback" and (not article_quotes or not all(
                _located(q, supplied.get("article_quote")) for q in article_quotes)):
            reasons.append("article_quote_missing_or_unlocated")
        if kind == "subject_fact_feedback" and supplied.get("independently_verified") is not True:
            reasons.append("source_authenticity_not_verified")
        if kind == "subject_fact_feedback" and supplied.get("effective_status") in (None, "", "unknown", "conflicting_evidence"):
            reasons.append("source_status_unknown")
        if kind == "subject_fact_feedback" and supplied.get("auto_retry") is False and any(
                gap.get("availability") == "unavailable_private" for gap in _rows(supplied.get("unavailable_evidence"))):
            reasons.append("unavailable_private")
        law = articles.get(uid, {})
        law_title = " ".join(str(v) for v in (law.get("law_name"), law.get("article_no")) if v)
        row = {"item_id": fid, "kind": kind, "title": str(supplied.get("subject_name") or record.get("subject_name") or "主体身份待补充")
                if kind == "subject_fact_feedback" else law_title or str(supplied.get("law_name") or uid),
            "work_item_id": work.get("work_item_id"), "candidate_ids": record.get("candidate_ids", []),
            "contract_evidence": quoted, "article_ids": [uid] if uid else [], "article_quotes": article_quotes,
            "model_verdicts": verdicts, "model_reasons": [str(o.get("reason") or "") for o in observations],
            "calls": receipts, "protocol_complete": record.get("protocol_complete") is True,
            "source": copy.deepcopy(supplied.get("source", {})), "fact_id": supplied.get("fact_id"),
            "fact_evidence_text": supplied.get("fact_evidence_text", ""),
            "source_evidence_text": supplied.get("source_evidence_text", ""),
            "independently_verified": supplied.get("independently_verified") is True,
            "auto_retry": supplied.get("auto_retry", True), "retry_trigger": supplied.get("retry_trigger", "evidence_review"),
            "unavailable_evidence": copy.deepcopy(supplied.get("unavailable_evidence", [])),
            "evidence_status": "bound_provided_evidence" if not reasons else "evidence_pending",
            "status": "manual_review" if reasons else "provided_relation_checked",
            "status_label": "待人工核验" if reasons else "给定关系有证据支持",
            "explanation": "模型反馈、原文引文和调用绑定逐项列出。模型确认仅涉及给定材料之间的关系，不代表法律结论正确或主体事实真实。",
            "next_action": "补充当前缺失的证据，并核对原文与法条的适用关系；人工意见单独记录。" if reasons else
                           "可追溯查看本项原文和调用；最终法律意见仍由复核人员确认。"}
        items.append(_manual(row, reasons))

    subject = _dict(_dict(stats.get("subject")).get("stats")) or _dict(_dict(payload.get("subject_checks")).get("stats"))
    for fact in _rows(subject.get("fact_checks")):
        for fact_type, status in _dict(fact.get("checks")).items():
            identity = str(fact.get("subject_node_id") or fact.get("subject_name") or "") + ":" + fact_type
            reasons = []
            if status in (None, "", "unknown", "conflicting_evidence"):
                reasons.append("source_status_unknown")
            if fact.get("independently_verified") is not True:
                reasons.append("source_authenticity_not_verified")
            gaps = [gap for gap in _rows(fact.get("evidence_gaps"))
                    if fact_type in gap.get("fact_types", []) and status in (None, "", "unknown", "conflicting_evidence")]
            private_gaps = [gap for gap in gaps if gap.get("availability") == "unavailable_private"]
            if private_gaps:
                reasons.append("unavailable_private")
            items.append(_manual({"item_id": "fact_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                "kind": "subject_fact", "title": str(fact.get("subject_name") or "身份待补充") + " · " +
                    {"registration": "主体登记", "authorization": "签约授权", "license": "经营资质", "credit": "信用记录"}.get(fact_type, fact_type),
                "fact_type": fact_type, "fact_status": status, "subject_name": fact.get("subject_name", ""),
                "contract_evidence": evidence(fact.get("clause_ids", [])), "article_ids": [], "calls": [],
                "source_records": copy.deepcopy(fact.get("records", [])), "model_verdicts": [],
                "evidence_status": "external_evidence_pending" if reasons else "provided_external_evidence",
                "status": "manual_review" if reasons else "provided_fact_record",
                "status_label": "主体事实待核验" if reasons else "已有事实来源记录",
                "independently_verified": fact.get("independently_verified") is True,
                "unavailable_evidence": copy.deepcopy(private_gaps),
                "evidence_gaps": copy.deepcopy(gaps),
                "auto_retry": False if private_gaps else True,
                "retry_trigger": "new_evidence" if private_gaps else "evidence_review",
                "explanation": "按主体身份绑定的登记、资质、授权或信用记录；合同自述和材料上传不能证明主体事实。",
                "next_action": "核验主体统一社会信用代码、来源机构、日期、有效期和许可范围，并补充来源记录。"}, reasons))

    for gap in _rows(subject.get("evidence_gaps")):
        private = gap.get("availability") == "unavailable_private"
        reasons = ["unavailable_private" if private else "document_evidence_missing"]
        identity = str(gap.get("document_id") or gap.get("title") or "unknown_document")
        items.append(_manual({"item_id": "gap_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            "kind": "document_evidence_gap", "title": str(gap.get("title") or identity),
            "document_id": gap.get("document_id"), "availability": gap.get("availability"),
            "reason_source": gap.get("reason_source"), "availability_reason": gap.get("reason"),
            "blocked_fact_scope": gap.get("blocked_fact_scope"),
            "subject_names": copy.deepcopy(gap.get("subject_names", [])),
            "subject_roles": copy.deepcopy(gap.get("subject_roles", [])),
            "fact_types": copy.deepcopy(gap.get("fact_types", [])),
            "contract_evidence": evidence(gap.get("clause_ids", [])), "article_ids": [], "calls": [],
            "model_verdicts": [], "source_records": [], "fact_status": "unknown",
            "status": "manual_review", "status_label": "隐私缺件已注明，事实待核验" if private else "材料待补核",
            "evidence_status": "unavailable_private" if private else "external_evidence_pending",
            "independently_verified": False, "auto_retry": False if private else True,
            "retry_trigger": "new_evidence" if private else "available_source",
            "explanation": "材料可用性说明与事实证据分开记录。用户说明无法提供隐私文件，不代表相关义务已履行、授权真实或主体核验通过。",
            "next_action": "保存缺件说明，现有证据不足的结论保持待核验。"}, reasons))

    for work in works:
        if work.get("legacy_feedback_item_id") or work.get("kind") == "subject_fact":
            continue
        reasons = list(work.get("pending_reasons", []))
        if work.get("pending") and not reasons:
            reasons.append("feedback_item_pending")
        items.append(_manual({"item_id": work.get("work_item_id"), "kind": work.get("kind", "review_work"),
            "title": "初审覆盖" if work.get("kind") == "initial_review" else "待办核验任务",
            "contract_evidence": evidence(work.get("clause_ids", [])), "article_ids": [], "calls": [],
            "receipt_call_ids": copy.deepcopy(work.get("receipt_call_ids", [])), "model_verdicts": [],
            "status": "manual_review" if reasons else "execution_recorded",
            "status_label": "范围待补核" if reasons else "已有执行记录",
            "evidence_status": work.get("source_status", "unrecorded"),
            "explanation": "初审范围和执行凭证记录；覆盖完整并不证明法律问题已穷尽。",
            "next_action": "人工核对未完成的原文范围，必要时补充证据后重新审查。"}, reasons))
    supervisor = _dict(stats.get("supervisor"))
    if supervisor.get("manual_review_requested"):
        decisions = [row for row in _rows(supervisor.get("decisions")) if row.get("action") == "manual"]
        decision = decisions[-1] if decisions else {}
        identity = str(payload.get("contract_id") or "") + str(decision.get("decision_id") or "manual")
        items.append(_manual({
            "item_id": "supervisor_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            "kind": "supervisor_manual_review", "title": "主脑请求人工复核",
            "contract_evidence": evidence(decision.get("clause_ids", [])), "article_ids": [],
            "calls": [], "model_verdicts": [], "status": "manual_review", "status_label": "主脑待人工处理",
            "explanation": str(decision.get("reason") or "主脑请求人工核对，本次自动流程未关闭该事项。"),
            "next_action": "人工核对相关合同、主体材料及法律依据，并记录处理意见。",
        }, ["human_review_requested"]))
    for review in _rows(supervisor.get("pending_reviews")):
        identity = str(payload.get("contract_id") or "") + json.dumps(review, sort_keys=True, ensure_ascii=False)
        items.append(_manual({
            "item_id": "expert_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            "kind": "expert_review", "title": "专家模型复核未完成",
            "contract_evidence": evidence(review.get("clause_ids", [])), "article_ids": [],
            "calls": [], "model_verdicts": [], "status": "manual_review", "status_label": "专家复核待补核",
            "model_id": review.get("model_id"), "attempts": review.get("attempts", 0),
            "explanation": "已指派专家模型，但尚无完整执行凭证；未将失败或部分输出记为完成。",
            "next_action": "检查模型服务及完整调用结果，必要时重试或由人工复核。",
        }, ["expert_review_incomplete"]))
    counts = Counter(item["kind"] for item in items)
    from .review_task_summary import build_task_summary
    return {"version": "review-explanations-v1", "contract_id": payload.get("contract_id"),
        "summary": {"total": len(items), "by_kind": dict(counts),
                    "manual_required": sum(item["manual_required"] for item in items),
                    "provided_relation_checked": sum(item["status"] == "provided_relation_checked" for item in items)},
        "items": items, "sources": {"clauses": clauses, "articles": articles},
        "task_summary": build_task_summary(items, works, contract_id=str(payload.get("contract_id") or "")),
        "scope": "只投影本报告当前证据与待办；不修改机器评分、原始状态或人工意见。",
        "legal_truth_established": False}
