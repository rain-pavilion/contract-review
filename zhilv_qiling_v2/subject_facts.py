"""Local evidence registry for subject facts supplied by a human or trusted adapter.

The registry never fetches user-controlled URLs and never upgrades contract prose
to verified facts. A supplied review attestation is provenance, not a guarantee by
this software that an official record is authentic. Positive evidence expires.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

LOCK = threading.RLock()
FACT_TYPES = {"registration", "license", "authorization", "credit"}
STATUSES = {"valid", "invalid", "active", "inactive", "adverse", "clear", "unknown"}
TYPE_STATUSES = {"registration": {"active", "inactive", "unknown"},
                 "license": {"valid", "invalid", "unknown"},
                 "authorization": {"valid", "invalid", "unknown"},
                 "credit": {"clear", "adverse", "unknown"}}
ACTOR_KINDS = {"human", "adapter", "legacy_unspecified"}
MAX_RECORDS = 20000
MAX_BYTES = 10 * 1024 * 1024


def _name(value: Any) -> str:
    return re.sub(r"[\s（）()]", "", str(value or "")).casefold()


def _day(value: Any, field: str, required: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not required:
        return ""
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field}必须是YYYY-MM-DD日期") from exc


def _scope_applicability(value: Any) -> dict[str, Any]:
    """Validate a supplied human scope judgement, not legal correctness.

    A license's free-text scope does not prove applicability to a contract.
    This attestation explicitly names the contract and the applicable clauses.
    """
    if not isinstance(value, dict) or value.get("actor_kind") != "human":
        raise ValueError("applicability必须是明确标记actor_kind=human的人工适用性判断")
    verdict = value.get("verdict")
    if verdict not in ("applicable", "not_applicable", "needs_manual"):
        raise ValueError("applicability.verdict必须是applicable、not_applicable或needs_manual")
    clean = {"actor_kind": "human", "verdict": verdict}
    for key, limit in (("contract_id", 128), ("reviewer", 100), ("basis", 2000)):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > limit:
            raise ValueError(f"applicability.{key}必须填写且不超过{limit}字")
        clean[key] = text.strip()
    clause_ids = value.get("clause_ids")
    if (not isinstance(clause_ids, list) or not 1 <= len(clause_ids) <= 1000
            or any(not isinstance(cid, str) or not cid.strip() or len(cid) > 128 for cid in clause_ids)):
        raise ValueError("applicability.clause_ids须为1至1000个明确适用的条款ID")
    clean["clause_ids"] = sorted({cid.strip() for cid in clause_ids})
    reviewed_at = _day(value.get("reviewed_at"), "applicability.reviewed_at", True)
    if reviewed_at > date.today().isoformat():
        raise ValueError("适用性核对日期不能在未来")
    clean.update(reviewed_at=reviewed_at, independently_verified=False)
    return clean


def validate_fact(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("主体证据必须是JSON对象")
    subject_name = str(record.get("subject_name") or "").strip()
    fact_type = str(record.get("fact_type") or "")
    status = str(record.get("status") or "unknown")
    if not subject_name or len(subject_name) > 200 or fact_type not in FACT_TYPES or status not in STATUSES:
        raise ValueError("主体名称、fact_type或status不合法")
    if status not in TYPE_STATUSES[fact_type]:
        raise ValueError("status与fact_type不对应")
    source = record.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("source必须包含来源与人工核验记录")
    explicit_actor = "actor_kind" in source
    actor_kind = str(source.get("actor_kind") or "").strip() if explicit_actor else "legacy_unspecified"
    if actor_kind not in ACTOR_KINDS:
        raise ValueError("source.actor_kind必须是human、adapter或legacy_unspecified")
    if actor_kind == "adapter" and status != "unknown":
        raise ValueError("自动来源绑定适配器只能导入unknown；不能凭actor标记认定正负业务事实")
    if str(record.get("evidence_ref") or "").startswith("material:") and status != "unknown":
        raise ValueError("原件哈希仅证明存储完整性，材料绑定事实必须保持unknown")
    url = str(source.get("url") or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("source.url必须是无账户信息的HTTP(S)来源地址；系统不会自动访问它")
    publisher = str(source.get("publisher") or "").strip()
    reviewer = str(source.get("reviewer") or "").strip()
    method = str(source.get("verification_method") or "").strip()
    if not publisher or not reviewer or not method:
        raise ValueError("需填写来源机构、核验人及核验方法；不能把合同自述作为已核验事实")
    verified_at = _day(source.get("verified_at"), "source.verified_at", True)
    if verified_at > date.today().isoformat():
        raise ValueError("核验日期不能在未来")
    valid_until = _day(record.get("valid_until"), "valid_until")
    valid_from = _day(record.get("valid_from"), "valid_from")
    if valid_from and valid_until and valid_from > valid_until:
        raise ValueError("有效期起止倒置")
    identifier = str(record.get("subject_id") or "").strip().upper()
    if identifier and not re.fullmatch(r"[A-Z0-9]{8,30}", identifier):
        raise ValueError("subject_id只能包含8至30位字母数字")
    scope = str(record.get("scope") or "").strip()[:500]
    contract_id = str(record.get("contract_id") or "").strip()[:128]
    if fact_type == "authorization" and status == "valid" and not contract_id:
        raise ValueError("有效授权证据必须绑定contract_id，避免将其他合同授权套用")
    clean = {
        "subject_name": subject_name, "subject_id": identifier,
        "fact_type": fact_type, "status": status, "scope": scope,
        "contract_id": contract_id, "valid_from": valid_from, "valid_until": valid_until,
        "evidence_ref": str(record.get("evidence_ref") or "")[:500],
        "source": {"url": url[:2000], "publisher": publisher[:200],
                   "reviewer": reviewer[:100], "verification_method": method[:500],
                   "verified_at": verified_at},
        "authenticity": "human_attested_not_independently_verified_by_software",
    }
    # Legacy canonical hashing includes the historical authenticity literal.
    # Keep that digest basis for actor-less old records, while displaying truthful
    # derived metadata below. New explicit actor kinds are part of their hash.
    display_authenticity = {
        "human": "human_claimed_attestation_not_independently_verified_by_software",
        "adapter": "automated_source_binding_not_independently_verified_by_software",
        "legacy_unspecified": "supplied_provenance_not_independently_verified_by_software",
    }[actor_kind]
    if explicit_actor:
        clean["source"]["actor_kind"] = actor_kind
        clean["authenticity"] = display_authenticity
    if "applicability" in record:
        clean["applicability"] = _scope_applicability(record["applicability"])
    digest = hashlib.sha256(json.dumps(clean, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if record.get("record_sha256") and record["record_sha256"] != digest:
        raise ValueError("主体事实记录校验和不一致")
    clean["fact_id"] = digest[:24]
    clean["record_sha256"] = digest
    clean["authenticity"] = display_authenticity
    clean["source_actor_kind"] = actor_kind
    clean["verification_level"] = {
        "human": "human_attestation_claim",
        "adapter": "automated_source_binding_claim",
        "legacy_unspecified": "unclassified_supplied_provenance",
    }[actor_kind]
    clean["independently_verified"] = False
    clean["record_hash_schema"] = ("canonical-v3-scope-applicability" if "applicability" in clean else
                                  "canonical-v2-actor-kind" if explicit_actor else "canonical-v1-legacy")
    return clean


def load_facts(path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path:
        return [], ["subject_fact_source_not_configured"]
    source = Path(path)
    if not source.exists():
        return [], ["subject_fact_source_missing"]
    try:
        if source.stat().st_size > MAX_BYTES:
            return [], ["subject_fact_source_too_large"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or len(records) > MAX_RECORDS:
            return [], ["subject_fact_source_invalid"]
        clean = [validate_fact(item) for item in records]
        return clean, []
    except (OSError, ValueError, TypeError) as exc:
        return [], [f"subject_fact_source_invalid:{type(exc).__name__}"]


def import_facts(records: Iterable[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    """Import into a server-configured path only; do not accept a path from HTTP input."""
    new = [validate_fact(item) for item in records]
    if not new or len(new) > 1000:
        raise ValueError("每次导入1至1000条主体事实")
    target = Path(path)
    with LOCK:
        old, errors = load_facts(target)
        if errors and target.exists():
            raise ValueError("现有主体事实库无法读取，未覆盖原文件")
        merged = {item["fact_id"]: item for item in old + new}
        if len(merged) > MAX_RECORDS:
            raise ValueError("主体事实库记录数超限")
        payload = {"schema_version": "1.0", "updated_at": datetime.now(timezone.utc).isoformat(),
                   "records": list(merged.values())}
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(raw.encode("utf-8")) > MAX_BYTES:
            raise ValueError("主体事实库文件过大")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            temporary.write_text(raw, encoding="utf-8")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {"imported": len(new), "total": len(merged),
            "fact_ids": [x["fact_id"] for x in new],
            "authenticity": "supplied_provenance_not_independently_verified_by_software",
            "source_actor_kinds": sorted({x["source_actor_kind"] for x in new}),
            "independently_verified": False}


def resolve_subject(subject_name: str, records: list[dict[str, Any]], *, full_text: str = "",
                    contract_id: str = "", max_age_days: int = 30,
                    today: date | None = None, subject_ids: Iterable[str] = (),
                    identity_ambiguous: bool = False) -> dict[str, Any]:
    """Resolve evidence against identifiers already bound to this subject.

    ``full_text`` remains a compatibility argument, but is deliberately never
    searched for an evidence-supplied ID. The caller must derive ``subject_ids``
    from explicit subject/identifier fields in the contract, independently of
    these evidence records. Name-only or ambiguous identities remain unknown.
    """
    today = today or date.today()
    actual_ids = sorted({str(value).strip().upper() for value in subject_ids
                         if re.fullmatch(r"[A-Za-z0-9]{8,30}", str(value).strip())})
    ambiguous = bool(identity_ambiguous or len(actual_ids) > 1)
    matched = []
    for item in records:
        if _name(item["subject_name"]) != _name(subject_name):
            continue
        row = dict(item)
        reasons = []
        identity = str(item.get("subject_id") or "").upper()
        identity_reason = ("subject_identifier_ambiguous" if ambiguous else
                           "subject_identifier_not_bound" if not actual_ids else
                           "evidence_identifier_not_supplied" if not identity else
                           "subject_identifier_mismatch" if identity != actual_ids[0] else "")
        if identity_reason:
            reasons.append(identity_reason)
        if item.get("contract_id") and item["contract_id"] != contract_id:
            reasons.append("different_contract")
        if item.get("valid_from") and item["valid_from"] > today.isoformat():
            reasons.append("not_yet_effective")
        if item.get("valid_until") and item["valid_until"] < today.isoformat():
            reasons.append("expired")
        checked = date.fromisoformat(item["source"]["verified_at"])
        if (today - checked).days > max(0, int(max_age_days)):
            reasons.append("source_stale")
        if item["status"] in ("valid", "active", "clear") and item["fact_type"] in ("license", "authorization") and not item.get("valid_until"):
            reasons.append("validity_end_not_supplied")
        row.update(effective_status="unknown" if reasons else item["status"],
                   review_reason=reasons[-1] if reasons else "", review_reasons=reasons,
                   identity_review_reason=identity_reason,
                   identity_match="unresolved" if identity_reason else "name_and_identifier")
        matched.append(row)
    statuses = {}
    for kind in sorted(FACT_TYPES):
        relevant = [r for r in matched if r["fact_type"] == kind]
        values = {r["effective_status"] for r in relevant}
        statuses[kind] = "conflicting_evidence" if len(values - {"unknown"}) > 1 else (
            next(iter(values - {"unknown"})) if values - {"unknown"} else "unknown")
    return {"subject_name": subject_name, "checks": statuses, "records": matched,
            "subject_identifiers": actual_ids,
            "identity_binding_status": "ambiguous" if ambiguous else "bound" if actual_ids else "unbound",
            "source_mode": "supplied_local_evidence" if matched else "no_matching_evidence",
            "independently_verified": False}
