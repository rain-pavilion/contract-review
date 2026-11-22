"""Display-only grouping of repeated external-evidence work.

The original items, source anchors, protocol IDs and pending state stay intact.
Groups are navigation units, never new receipts or factual confirmations.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


_FACT_NAMES = {"registration": "主体登记", "authorization": "签约授权",
               "license": "经营资质", "credit": "信用记录"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity(item: dict, work: dict) -> dict | None:
    supplied = work.get("provided_feedback")
    if (item.get("kind") != "subject_fact_feedback" or item.get("manual_required") is not True
            or not isinstance(supplied, dict) or not supplied.get("subject_node_id")
            or not supplied.get("fact_id") or not supplied.get("fact_type")
            or supplied.get("candidate_ids") or supplied.get("request_id")
            or item.get("candidate_ids") or work.get("pending") is not True):
        return None
    # A candidate-specific or explicit upstream request is a separate obligation.
    # Only source-pending subject work is grouped, not model conclusions.
    if work.get("source_status") != "external_evidence_pending":
        return None
    return {
        "subject": {key: supplied.get(key) for key in
                    ("subject_node_id", "subject_id", "subject_name", "subject_role", "identity_binding")},
        "fact": {key: supplied.get(key) for key in ("fact_id", "fact_type", "check_states")},
        "source": {key: supplied.get(key) for key in
                   ("source_id", "source", "source_evidence_text", "record_provided", "provenance_mode")},
        # The exact fact text carries record scope, validity dates and identity.
        # Clause IDs here are occurrence anchors, but any explicit applicability
        # scope or evidence-gap scope must remain part of the grouping identity.
        "scope": {key: supplied.get(key) for key in
                  ("scope", "applicable_scope", "applicability_scope", "valid_from", "valid_until",
                   "fact_evidence_text", "unavailable_evidence")},
        "state": {"item": {key: item.get(key) for key in
                             ("status", "manual_required", "evidence_status", "claim_support")},
                  "work": {key: work.get(key) for key in
                             ("execution_status", "source_status", "semantic_status", "pending",
                              "protocol_complete", "dispatchable", "actual_processed")},
                  "fact": {key: supplied.get(key) for key in
                            ("effective_status", "source_complete", "upstream_pending",
                             "independently_verified", "auto_retry", "retry_trigger")}},
    }


def build_task_summary(items: list[dict], works: list[dict], *, contract_id: str = "") -> dict:
    """Return small group references; never rewrite an input item or ledger."""
    by_feedback = {str(work.get("legacy_feedback_item_id")): work for work in works
                   if isinstance(work, dict) and work.get("legacy_feedback_item_id")}
    grouped: dict[str, dict] = {}
    for index, item in enumerate(items):
        item_id = str(item.get("item_id") or "")
        work = by_feedback.get(item_id, {})
        identity = _identity(item, work)
        key = _json([contract_id, identity]) if identity else _json([contract_id, "single", index, item_id])
        supplied = work.get("provided_feedback", {}) if identity else {}
        fact_type = str(supplied.get("fact_type") or "")
        fact_label = _FACT_NAMES.get(fact_type, fact_type)
        if fact_type == "unresolved_subject_checks":
            fact_label = "、".join(_FACT_NAMES.get(kind, kind) for kind in
                                  sorted((supplied.get("check_states") or {}).keys())) or "主体资料"
        title = ((str(supplied.get("subject_name") or "主体") + " · " + fact_label + "待核验")
                 if identity else str(item.get("title") or "审查事项"))
        group = grouped.setdefault(key, {
            "group_id": "display_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24],
            "kind": "subject_evidence_task" if identity else "original_item",
            "title": title, "item_ids": [], "work_item_ids": [], "clause_ids": [],
            "raw_item_count": 0, "manual_item_count": 0, "manual_required": False,
            "status": "manual_review" if identity else item.get("status", "unrecorded"),
            "legal_truth_established": False,
        })
        group["item_ids"].append(item_id)
        group["raw_item_count"] += 1
        group["manual_item_count"] += int(item.get("manual_required") is True)
        group["manual_required"] = group["manual_required"] or item.get("manual_required") is True
        if work.get("work_item_id") and work["work_item_id"] not in group["work_item_ids"]:
            group["work_item_ids"].append(work["work_item_id"])
        for evidence in item.get("contract_evidence", []):
            cid = evidence.get("clause_id") if isinstance(evidence, dict) else None
            if cid and cid not in group["clause_ids"]:
                group["clause_ids"].append(cid)
    groups = list(grouped.values())
    return {"version": "display-task-groups-v1", "groups": groups,
            "raw_item_count": len(items),
            "raw_manual_item_count": sum(item.get("manual_required") is True for item in items),
            "display_group_count": len(groups),
            "manual_group_count": sum(group["manual_required"] for group in groups),
            "grouped_subject_item_count": sum(group["raw_item_count"] for group in groups
                                               if group["kind"] == "subject_evidence_task"),
            "subject_evidence_group_count": sum(group["kind"] == "subject_evidence_task" for group in groups),
            "scope": "仅按同主体、事实、来源、适用范围和状态归组展示；原始事项、条款锚点、待办状态及逐项批注不变。",
            "legal_truth_established": False}
