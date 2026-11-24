"""Source-bound, bounded contract combinations; never automatic legal clearance.

The temporal rule joins three literal premises, not arbitrary named graph nodes.
All offsets are Unicode character offsets in the exact supplied clause text.
``original_sha256`` is provenance of the original file; ``source_sha256`` binds
the quoted clause text and must not be confused with the original file hash.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
import re
from typing import Any, Mapping

from .fact_slots import extract_fact_slots
from .proposition_graph import build_proposition_graph


VERSION = "contract-context-review-v1"
_TEMPORAL = "acceptance-before-required-test-v1"
_PROVENANCE = ("source_id", "document_id", "source_title", "source_role",
               "source_page_start", "original_sha256", "extraction_quality")
_OBJECT = re.compile(r"本设备|本项目|本服务|本合同|(?:项目|设备|标的)[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+")
_STAGE = re.compile(r"(最终|初步|初次|初|终|阶段性)?验收")
_PAYMENT = re.compile(r"验收(?:合格|通过)(?:之?后|之后)[^。；;\n]{0,28}(?:支付|付清|付款)[^。；;\n]{0,12}尾款")
_PAYMENT_NEGATIVE = re.compile(
    r"(?:禁止|严禁|不(?:得|应|再|予|用|必|会|需|须|能|可)?|无[须需]|免于|拒绝|暂停|暂缓|停止)"
    r"(?:再|予以)?(?:向(?:甲方|乙方|买方|卖方))?(?:予以)?(?:支付|付清|付款)|"
    r"(?:并非|不是|非|不作为|不构成)[^。；;\n]{0,10}(?:支付|付清|付款)[^。；;\n]{0,12}(?:前提|条件)")
_DEEMED = re.compile(r"(?:未|没有)[^。；;\n]{0,20}异议[^。；;\n]{0,12}(?:视为|视同|默认为)[^。；;\n]{0,12}验收(?:合格|通过)")
_TEST = re.compile(r"验收[^。；;\n]{0,28}(?:包含|包括|进行|完成)[^。；;\n]{0,12}(?:性能|功能|压力|安全|质量)?测试")
_NUMBER = r"(?:\d{1,5}|[零〇一二三四五六七八九十百两]{1,6})"
_TIME = re.compile(r"(?P<trigger>收货|收到货物|交货|交付|通知|收到通知|签署|生效)(?:之日)?"
                   r"(?P<basis>次日起|之日起|日起|之后|后|起)(?:第)?\s*(?P<number>" + _NUMBER +
                   r")\s*(?:个)?(?P<unit>工作日|自然日|日|天)")
_QUALIFIER = re.compile(r"除非|除[^。；;\n]{0,40}外|但是|但|另有约定|不得|无需|无须|不应|不必|"
                        r"并非|不视为|不适用|不能视为|不构成|不包含|不包括|不予|不要求|不需|不触发|不影响")
_PRIORITY = re.compile(r"优先|以[^。；;\n]{0,40}为准|修订|修正|变更|替代|补充协议")


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _id(prefix, value):
    return prefix + _hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:24]


def _unique(items):
    return list(dict.fromkeys(items))


def _spans(value):
    """Yield even partially malformed span dictionaries, so missing keys fail."""
    if isinstance(value, Mapping):
        if any(key in value for key in ("quote", "source_sha256", "start", "end")):
            yield value
        else:
            for item in value.values():
                yield from _spans(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _spans(item)


def _valid_span(span, records):
    if not isinstance(span, Mapping):
        return False
    record = records.get(span.get("clause_id"))
    if not record:
        return False
    start, end = span.get("start"), span.get("end")
    return (type(start) is int and type(end) is int and 0 <= start < end <= len(record["text"])
            and span.get("source_sha256") == record["sha256"]
            and span.get("quote") == record["text"][start:end])


def _evidence(record, span=None):
    span = span or {"clause_id": record["clause_id"], "start": 0, "end": len(record["text"]),
                    "quote": record["text"], "source_sha256": record["sha256"]}
    return {**deepcopy(dict(span)), **deepcopy(record["provenance"]), "located": True,
            "offset_scope": "clause_text"}


def _evidence_list(spans, records):
    evidence, seen = [], set()
    for span in spans:
        if not _valid_span(span, records):
            return None
        key = (span["clause_id"], span["start"], span["end"], span["source_sha256"])
        if key not in seen:
            evidence.append(_evidence(records[span["clause_id"]], span))
            seen.add(key)
    return evidence


def _source_pending(records, document):
    reasons = []
    for record in records:
        quality = record["provenance"].get("extraction_quality") or {}
        if isinstance(quality, Mapping) and quality.get("human_verified") is not True:
            status = str(quality.get("status", "")).lower()
            if "ocr" in status or quality.get("requires_human_review") is True:
                reasons.append("来源“%s”的OCR文本尚未人工核对，须先核对原件。" % record["provenance"]["source_title"])
    warnings = _get(document, "warnings", []) or []
    stats = _get(document, "stats", {}) or {}
    if stats.get("needs_ocr_pages") or any(re.search(r"OCR|扫描|解析|识别", str(w), re.I) for w in warnings):
        reasons.append("解析/OCR存在提示，须核对所给全文与原件的一致性。")
    return _unique(reasons)


def _receipt(contract_id, rule, point_id, *, finding=None, evidence=(), reasons=(), scope_complete=False):
    row = {"check_id": _id("context_check_", [contract_id, rule, finding and finding["finding_id"]]),
           "point_id": point_id, "status": "needs_manual" if reasons else "not_checked",
           "execution_status": "completed", "scope_complete": bool(scope_complete),
           "scope": "provided_whole_contract_source_bound_literal_combinations_only",
           "clause_ids": _unique(e["clause_id"] for e in evidence), "evidence": list(evidence),
           "actual_issue": "", "impact": "", "recommendation": "", "severity": "medium",
           "manual_required": True, "pending_reasons": list(reasons), "rule_version": rule,
           "legal_truth_established": False}
    if finding:
        for key in ("point_id", "status", "evidence", "actual_issue", "impact", "recommendation",
                    "severity", "manual_required", "pending_reasons"):
            row[key] = deepcopy(finding[key])
        row["clause_ids"] = _unique(e["clause_id"] for e in row["evidence"])
        row["finding_id"] = finding["finding_id"]
        row["scope_complete"] = finding["status"] == "issue_found"
    return row


def _finding(kind, title, point_id, evidence, claims, assumptions, pending, actual_issue, impact, recommendation):
    identity = [kind, [(e["clause_id"], e["start"], e["end"], e["source_sha256"]) for e in evidence]]
    steps = []
    for index, claim in enumerate(claims):
        final = index == len(claims) - 1
        premises = ([s["step_id"] for s in steps] + ["evidence_" + str(i + 1) for i in range(len(evidence))]
                    if final else ["evidence_" + str(index + 1)] if index < len(evidence)
                    else [s["step_id"] for s in steps])
        steps.append({"step_id": "step_" + str(index + 1), "claim": claim,
                      "premise_ids": premises})
    quoted = deepcopy(evidence)
    for index, item in enumerate(quoted):
        item["evidence_id"] = "evidence_" + str(index + 1)
    return {"finding_id": _id("context_finding_", identity), "kind": kind, "title": title,
            "point_id": point_id, "status": "needs_manual" if pending else "issue_found",
            "actual_issue": actual_issue, "impact": impact, "recommendation": recommendation,
            "evidence": quoted, "reasoning_steps": steps, "assumptions": list(assumptions),
            "pending_reasons": _unique(pending), "manual_required": True, "severity": "medium",
            "legal_truth_established": False, "automatic_legal_confirmation_allowed": False}


def _prepare(contract):
    document = _get(contract, "document")
    clauses = list(_get(document, "clauses", []) or [])
    full_text = _get(document, "full_text", "")
    full_text = full_text if isinstance(full_text, str) else ""
    counts = Counter(_get(clause, "clause_id") for clause in clauses
                     if isinstance(_get(clause, "clause_id"), str))
    records, violations, documents, slots = {}, [], {}, []

    def violation(kind, ids, message):
        row = {"kind": kind, "clause_ids": list(ids), "message": message}
        row["constraint_id"] = _id("constraint_", row)
        if row not in violations:
            violations.append(row)

    for index, clause in enumerate(clauses):
        clause_id = _get(clause, "clause_id")
        text = _get(clause, "text") or _get(clause, "body", "")
        metadata = _get(clause, "metadata", {}) or {}
        source_id = metadata.get("source_id") or metadata.get("document_id") or "provided-document"
        document_id = metadata.get("document_id") or source_id
        provenance = {key: deepcopy(metadata.get(key)) for key in _PROVENANCE}
        provenance.update(source_id=source_id, document_id=document_id,
                          source_title=metadata.get("source_title") or _get(document, "source_name", "所给合同"),
                          source_role=metadata.get("source_role") or "provided_document",
                          source_page_start=metadata.get("source_page_start", _get(clause, "page_start")),
                          extraction_quality=deepcopy(metadata.get("extraction_quality") or {}))
        key = (str(source_id), str(document_id))
        source_doc = documents.setdefault(key, {**deepcopy(provenance), "clause_ids": [], "_texts": []})
        source_doc["clause_ids"].append(clause_id)
        source_doc["_texts"].append(text if isinstance(text, str) else "")
        if not isinstance(clause_id, str) or not clause_id:
            violation("clause_id_missing", [], "第%d个条款缺少有效的唯一条款标识。" % (index + 1))
            continue
        if counts[clause_id] != 1:
            violation("duplicate_clause_id", [clause_id], "重复条款标识使引文和图节点绑定不唯一。")
            continue
        if not isinstance(text, str) or not text.strip():
            violation("source_binding_invalid", [clause_id], "条款缺少可核对的原文。")
            continue
        record = {"clause_id": clause_id, "text": text, "sha256": _hash(text), "provenance": provenance}
        inconsistent = [field for field in ("source_title", "source_role", "original_sha256")
                        if source_doc.get(field) is not None and provenance.get(field) is not None
                        and source_doc[field] != provenance[field]]
        if inconsistent:
            violation("source_metadata_conflict", source_doc["clause_ids"], "同一来源标识的元数据不一致：" + "、".join(inconsistent))
        start = full_text.find(text)
        if start < 0:
            violation("source_text_mismatch", [clause_id], "条款原文不能在所给合同全文中精确定位。")
            continue
        try:
            extracted = extract_fact_slots(clause_id, text)
            source = extracted["source"]
            valid = (extracted.get("clause_id") == clause_id and source.get("text") == text
                     and source.get("sha256") == record["sha256"] and source.get("start_offset") == 0
                     and source.get("end_offset") == len(text)
                     and all(isinstance(extracted.get(k), list) for k in ("amounts", "obligations", "sentences"))
                     and all(_valid_span(span, {clause_id: record}) for span in _spans(
                         {k: extracted[k] for k in ("amounts", "obligations", "sentences")})))
        except (ValueError, KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            violation("source_binding_invalid", [clause_id], "要素的原文、偏移或SHA-256绑定失效；该条款不参与组合。")
            continue
        records[clause_id] = record
        slots.append(extracted)
    conflicting = {cid for v in violations if v["kind"] == "source_metadata_conflict" for cid in v["clause_ids"]}
    records = {cid: row for cid, row in records.items() if cid not in conflicting}
    slots = [row for row in slots if row["clause_id"] in records]
    # Coverage is a property of the final, validated source set. Excluding a
    # damaged exception must never strengthen the other literal premises.
    coverage = bytearray(len(full_text))
    for record in records.values():
        text = record["text"]
        start = full_text.find(text)
        while start >= 0:
            coverage[start:start + len(text)] = b"1" * len(text)
            start = full_text.find(text, start + len(text))
    if any(not coverage[index] and not char.isspace() for index, char in enumerate(full_text)):
        violation("full_text_coverage_gap", [], "所给全文存在未被有效唯一条款覆盖的非空文本，完整上下文尚待核对。")
    source_documents = []
    for row in documents.values():
        texts = row.pop("_texts")
        row.update(text_sha256=_hash("\n".join(texts)), text_hash_scope="joined_supplied_clause_text",
                   clause_count=len(row["clause_ids"]))
        source_documents.append(row)
    return document, clauses, records, slots, source_documents, violations, violation


def _graph_findings(graph, records, slots, document, global_pending, violation):
    propositions = graph.get("propositions", [])
    prop_counts = Counter(row.get("proposition_id") for row in propositions)
    nodes = {}
    for node in propositions:
        node_id = node.get("proposition_id")
        if (not node_id or prop_counts[node_id] != 1 or not _valid_span(node.get("source_span"), records)
                or not _valid_span(node.get("sentence_span"), records)
                or any(not _valid_span(span, records) for span in _spans(node))):
            violation("proposition_binding_invalid", [], "命题图节点缺少唯一、有效的原文绑定。")
        else:
            nodes[node_id] = node
    valid_edges = set()
    for edge in graph.get("edges", []):
        good = (edge.get("to") in nodes and (not edge.get("resolved") or edge.get("from") in nodes)
                and _valid_span(edge.get("condition_span"), records))
        if not good:
            violation("edge_endpoint_invalid", [], "命题图关系端点或前置条件的来源绑定无效，不能用于组合发现。")
        elif edge.get("resolved"):
            valid_edges.add((edge["from"], edge["to"]))
    amounts = {row["slot_id"]: row for doc in slots for row in doc["amounts"]}
    findings = []
    for relation in graph.get("relations", []):
        kind = relation.get("kind")
        claims, spans = [], []
        if kind == "literal_prerequisite_cycle":
            ids = relation.get("proposition_ids", [])
            if (len(ids) < 2 or any(node not in nodes for node in ids)
                    or any((ids[i], ids[(i + 1) % len(ids)]) not in valid_edges for i in range(len(ids)))):
                violation("relation_binding_invalid", [], "前置条件环的节点或关系不能由有效原文及边共同支持。")
                continue
            spans = [nodes[node]["sentence_span"] for node in ids]
            claims = ["原文明确将本项履行绑定于另一项已定位履行事件之后。" for _ in ids]
            claims.append("已定位的显式前置条件边构成文本上的环；条件实际成就及法律僵局仍未确认。")
            title, point = "显式履行前置条件形成循环线索", "consistency.references"
            actual = "已绑定来源的显式前置条件形成循环，须核对是否需要先行履行安排。"
            impact = "可能产生履行启动次序争议或互相等待，尚不能据此认定法律上的履行僵局。"
            recommendation = "结合主体、标的、阶段、条件和例外核对该环，必要时明确先行履行或独立触发机制。"
        elif kind in ("explicit_advance_tail_partition", "same_role_literal_value_difference"):
            ids = relation.get("slot_ids", [])
            if not ids or any(slot_id not in amounts for slot_id in ids):
                violation("relation_binding_invalid", [], "金额关系的要素端点无有效来源绑定。")
                continue
            if kind == "explicit_advance_tail_partition" and relation.get("arithmetic_consistent") is not False:
                continue
            spans = relation.get("source_spans", [])
            if len(spans) != len(ids) or any(span != amounts[slot_id]["source_span"] for span, slot_id in zip(spans, ids)):
                violation("relation_binding_invalid", [], "金额关系的引文不对应其金额要素。")
                continue
            title, point = "显式金额分项计算不一致" if kind == "explicit_advance_tail_partition" else "同角色字面金额不一致线索", "commercial.amount"
            if kind == "explicit_advance_tail_partition":
                kind = "explicit_partition_mismatch"
                actual = "原文明示的预付款与尾款合计%s元，与合同总价%s元不一致。" % (relation.get("component_sum_yuan"), relation.get("total_yuan"))
                claims = ["原文明示合同总价及由预付款和尾款组成的关系。",
                          "仅对原文明示分项求和，未将合同总价重复计入分项。", actual]
            else:
                actual = "同主体、同字面标的范围与同金额角色出现不同字面金额；仍须核对是否确属同一义务。"
                claims = ["逐项保留两个金额的主体、标的、角色及原文来源。", actual]
            impact = "可能导致价款、支付或结算口径争议；字面数值差异不自动等于法律冲突。"
            recommendation = "核对原件及相关附件的适用范围、金额角色、计价口径和修订优先关系。"
        else:
            continue
        evidence = _evidence_list(spans, records)
        if not evidence:
            violation("relation_binding_invalid", [], "组合关系缺少有效、精确的原文证据。")
            continue
        source_records = list({e["clause_id"]: records[e["clause_id"]] for e in evidence}.values())
        scopes = {match.group() for record in source_records for match in _OBJECT.finditer(record["text"])}
        qualifications = [record for record in records.values()
                          if _PRIORITY.search(record["text"])
                          or ((_QUALIFIER.search(record["text"]) or _PAYMENT_NEGATIVE.search(record["text"]))
                              and re.search(r"验收|测试|尾款|付款|支付|金额|价款", record["text"])
                              and (not _OBJECT.search(record["text"])
                                   or any(match.group() in scopes for match in _OBJECT.finditer(record["text"]))))]
        additional = [record for record in qualifications
                      if record["clause_id"] not in {item["clause_id"] for item in source_records}]
        for record in additional:
            evidence.append(_evidence(record))
        source_records.extend(additional)
        pending = [*global_pending, *_source_pending(source_records, document)]
        if additional:
            pending.append("所给合同其他条款包含优先适用、修订或相关否定/例外，须联合核对图关系的适用前提，不自动解释其效力。")
        if any(_QUALIFIER.search(record["text"]) or _PRIORITY.search(record["text"])
               or _PAYMENT_NEGATIVE.search(record["text"]) for record in source_records):
            pending.append("相关原文包含否定、例外或优先/修订约定，须核对其对组合的影响。")
        source_keys = {(r["provenance"]["source_id"], r["provenance"]["document_id"]) for r in source_records}
        if len(source_keys) > 1:
            pending.append("跨来源图关系仍须核对同名主体、标的和阶段是否确指同一义务，不能仅凭节点名称确认。")
        findings.append(_finding(kind, title, point, evidence, claims,
            ["所引原文属于本次提供合同范围；外部事实和最终法律效力未由本检查确认。"],
            pending, actual, impact, recommendation))
    return findings


def _number(value):
    if value.isascii() and value.isdigit():
        return int(value)
    digits = {char: i for i, char in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if value in digits:
        return digits[value]
    match = re.fullmatch(r"([一二三四五六七八九两]?)十([一二三四五六七八九]?)", value)
    if match:
        return digits.get(match[1], 1) * 10 + digits.get(match[2], 0)
    return None


def _temporal_candidates(records):
    candidates = {"payment": [], "deemed": [], "test": []}
    for record in records.values():
        text = record["text"]
        for kind, pattern in (("payment", _PAYMENT), ("deemed", _DEEMED), ("test", _TEST)):
            for match in pattern.finditer(text):
                # Scope and stage are local to the asserted premise sentence.
                # Earlier clauses never silently supply either property.
                left = max(text.rfind(mark, 0, match.start()) for mark in ("。", "；", ";", "\n")) + 1
                endings = [end for mark in ("。", "；", ";", "\n") if (end := text.find(mark, match.end())) >= 0]
                right = min(endings) + 1 if endings else len(text)
                sentence = text[left:right]
                objects = _unique(m.group() for m in _OBJECT.finditer(sentence))
                stages = _unique(m.group(1) for m in _STAGE.finditer(sentence))
                stage = stages[0] if len(stages) == 1 else None
                stage = {"终": "最终", "初": "初步", "初次": "初步"}.get(stage, stage)
                time, earliest_explicit = None, False
                if kind == "deemed":
                    # The last time immediately preceding the absence of an
                    # objection belongs to this default, not a prior action.
                    before = list(_TIME.finditer(sentence[:match.start() - left]))
                    time = before[-1] if before else None
                elif kind == "test":
                    earliest = re.search(r"最早(?:在|于)?", sentence[match.end() - left:])
                    if earliest:
                        tail = sentence[match.end() - left + earliest.end():]
                        time = _TIME.search(tail)
                        earliest_explicit = bool(time and re.match(r"(?:才能|才可|方可|可以)?完成", tail[time.end():]))
                timing = None
                if time:
                    timing = {"trigger": time["trigger"], "basis": "next_day" if time["basis"] == "次日起" else "after_event",
                              "days": _number(time["number"]), "unit": "business" if time["unit"] == "工作日" else "calendar"}
                span = {"clause_id": record["clause_id"], "start": left, "end": right,
                        "quote": sentence, "source_sha256": record["sha256"]}
                candidates[kind].append({"record": record, "scope": objects[0] if len(objects) == 1 else None,
                                         "stage": stage, "timing": timing, "sentence": sentence,
                                         "evidence": _evidence(record, span),
                                         "earliest_explicit": earliest_explicit,
                                         "payment_negative": bool(_PAYMENT_NEGATIVE.search(sentence)) if kind == "payment" else False,
                                         "necessary_explicit": bool(re.search(r"(?:必须|应当|须|应|需)(?:要)?(?:包含|包括|进行|完成)", sentence))})
    return candidates


def _temporal_findings(candidates, records, document, global_pending):
    findings = []
    priority_records = [record for record in records.values() if _PRIORITY.search(record["text"])]
    if not all(candidates.values()):
        return findings
    seen = set()
    for payment, deemed, test in product(candidates["payment"], candidates["deemed"], candidates["test"]):
        rows = (payment, deemed, test)
        source_records = list({r["record"]["clause_id"]: r["record"] for r in rows}.values())
        identity = tuple((r["record"]["clause_id"], r["evidence"]["start"], r["evidence"]["end"]) for r in rows)
        if identity in seen:
            continue
        seen.add(identity)
        pending = list(global_pending)
        scopes, stages = [r["scope"] for r in rows], [r["stage"] for r in rows]
        if None in scopes or len(set(scopes)) != 1 or scopes[0] == "本合同":
            pending.append("三处约定的标的适用范围缺失、不一致或仅为合同级范围，须先确认是否属于同一标的。")
        if None in stages or len(set(stages)) != 1 or stages[0] != "最终":
            pending.append("初验、终验或其他验收阶段未能明确一致，不能互相替代。")
        if any(_QUALIFIER.search(r["record"]["text"]) for r in rows):
            pending.append("相关原文存在否定或例外，必须核对是否限制默认验收、必要测试或尾款触发。")
        if payment["payment_negative"]:
            pending.append("付款谓词存在否定、禁止或非付款前提表述，不能将其认定为验收触发尾款支付。")
        if not test["necessary_explicit"]:
            pending.append("测试是否属于最终验收必须完成的必要前提未被明确表述，不能将可选测试当成必要测试。")
        if priority_records:
            pending.append("所给合同存在优先适用、修订或变更约定，须先确认三处条款是否共同适用。")
            source_records = list({r["clause_id"]: r for r in [*source_records, *priority_records]}.values())
        other_exceptions = [record for record in records.values()
                            if (_QUALIFIER.search(record["text"]) or _PAYMENT_NEGATIVE.search(record["text"]))
                            and re.search(r"验收|测试|尾款", record["text"])
                            and (not _OBJECT.search(record["text"])
                                 or any(m.group() in scopes for m in _OBJECT.finditer(record["text"])))]
        if other_exceptions:
            pending.append("相关标的的其他条款存在验收、测试或尾款例外，须先确认其对三处约定的限制。")
            source_records = list({r["clause_id"]: r for r in [*source_records, *other_exceptions]}.values())
        pending.extend(_source_pending(source_records, document))
        deadline, earliest = deemed["timing"], test["timing"]
        comparable = bool(deadline and earliest and deadline["days"] is not None and earliest["days"] is not None
                          and deadline["trigger"] == earliest["trigger"] == "收货"
                          and deadline["basis"] == earliest["basis"] == "after_event"
                          and deadline["unit"] == earliest["unit"] and test["earliest_explicit"])
        if not comparable:
            pending.append("默认验收期限与测试最早完成时间的起算事件、日历单位或确定日数未能一致绑定，不能直接比较。")
        if comparable and deadline["days"] >= earliest["days"]:
            # This one inequality does not clear the complete acceptance or
            # payment arrangements, so no general negative receipt is issued.
            continue
        evidence = [deepcopy(row["evidence"]) for row in rows]
        for record in source_records:
            for match in re.finditer(r"[^。；;\n]+[。；;]?", record["text"]):
                if _QUALIFIER.search(match.group()) or _PRIORITY.search(match.group()) or _PAYMENT_NEGATIVE.search(match.group()):
                    span = {"clause_id": record["clause_id"], "start": match.start(), "end": match.end(),
                            "quote": match.group(), "source_sha256": record["sha256"]}
                    if not any(e["clause_id"] == span["clause_id"] and e["start"] == span["start"]
                               and e["end"] == span["end"] for e in evidence):
                        evidence.append(_evidence(record, span))
        if payment["payment_negative"]:
            actual = "付款原文包含否定、禁止或非付款前提表述；虽存在验收与测试时间组合线索，尚不能认定验收提前触发尾款支付。"
        elif comparable:
            actual = ("若三处约定共同适用于同一设备最终验收，收货后%s日的默认验收早于必要性能测试最早在第%s日完成，"
                      "可能在测试前触发最终验收及尾款付款。") % (deadline["days"], earliest["days"])
        else:
            actual = "已定位最终验收付款、默认验收和必要测试的三处组合线索；时间或适用前提未核清，尚不能确认测试前触发验收及付款。"
        claims = ["付款原文含否定或禁止表述，验收是否触发尾款支付尚未成立。" if payment["payment_negative"]
                  else "付款条款文本将尾款支付绑定于验收合格后，付款触发仍须按原文核对。",
                  "默认验收约定以规定期限内未提出书面异议为前提，并非已确认该前提实际成就。",
                  "必要测试及其最早完成时间单独保留，未将测试默认为已完成。",
                  actual]
        title = "验收时序与付款否定约定须联合核对" if payment["payment_negative"] else "默认最终验收可能早于必要测试并触发尾款"
        findings.append(_finding("acceptance_before_required_test", title,
            "performance.acceptance", evidence, claims,
            ["三处明确的同名标的及最终验收确指本合同包中的同一事项。",
             "未提出书面异议仅为合同文字条件，其实际成就尚未核实。",
             "日数比较仅覆盖明确的同一起算与同一日历单位，不确认效力或实际付款义务。"],
            pending, actual, "可能在性能检验尚不能完成时提前形成验收与付款争议，影响质量异议及尾款安排。",
            "核对全部附件的适用范围、验收阶段、起算和日历口径；明确必要测试完成前默认验收不触发最终确认及尾款。"))
    return findings


def _completeness(callback, contract, result):
    if callback is None:
        return {"status": "not_evaluated", "execution_status": "not_run", "manual_required": True,
                "pending_reasons": ["尚未接入用户提供的信息齐全性检查算法，本模块不推定材料齐全。"],
                "legal_truth_established": False}
    try:
        value = callback(contract, context=deepcopy(result))
        if not isinstance(value, dict) or value.get("status") not in {"completed", "needs_manual", "not_evaluated", "error"}:
            raise ValueError("completeness callback must return a dict with an explicit supported status")
        json.dumps(value, ensure_ascii=False, allow_nan=False)

        def legal_confirmation(item):
            if isinstance(item, Mapping):
                return any((key in {"legal_truth_established", "automatic_legal_confirmation_allowed"} and val is True)
                           or legal_confirmation(val) for key, val in item.items())
            return isinstance(item, (list, tuple)) and any(legal_confirmation(val) for val in item)

        if legal_confirmation(value):
            raise ValueError("completeness callback cannot establish legal truth")
        value = deepcopy(value)
        value.update(execution_status="completed", legal_truth_established=False)
        return value
    except Exception as exc:
        return {"status": "error", "execution_status": "failed", "manual_required": True,
                "pending_reasons": ["信息齐全性回调执行失败或返回契约无效，须另行核对。"],
                "error_type": type(exc).__name__, "legal_truth_established": False}


def evaluate_contract_context(contract, config=None, *, completeness_check=None) -> dict[str, Any]:
    """Evaluate all supplied clauses without models, credentials or side effects.

    ``completeness_check`` is an optional future extension with the explicit
    signature ``callback(contract, *, context=<deep-copied result-so-far>)``.
    It must return a JSON dict with ``status`` equal to ``completed``,
    ``needs_manual``, ``not_evaluated`` or ``error``. It cannot establish legal
    truth. Its exceptions and invalid return values become an isolated error
    receipt; they never turn a missing algorithm into an assertion of coverage.
    ``config`` is reserved for engine interface compatibility; this rule set
    does not inspect credentials or use configuration to bypass source gates.
    """
    document, clauses, records, slots, source_documents, violations, violation = _prepare(contract)
    coverage_pending = (["所给全文与有效条款的覆盖关系尚待核实，不能确认已包含全部影响组合的约定。"]
                        if any(v["kind"] == "full_text_coverage_gap" for v in violations) else [])
    graph = build_proposition_graph(slots, scope_complete=not violations)
    findings = _graph_findings(graph, records, slots, document, coverage_pending, violation)
    candidates = _temporal_candidates(records)
    findings.extend(_temporal_findings(candidates, records, document, coverage_pending))
    # The graph remains source-bound when handed to later display/export code.
    def enrich(value):
        if isinstance(value, dict):
            if _valid_span(value, records):
                value.update(_evidence(records[value["clause_id"]], value))
            else:
                for nested in value.values():
                    enrich(nested)
        elif isinstance(value, list):
            for nested in value:
                enrich(nested)
    enrich(graph)
    contract_id = _get(contract, "contract_id", "provided-contract")
    checks = [_receipt(contract_id, _TEMPORAL if f["kind"] == "acceptance_before_required_test" else VERSION,
                       f["point_id"], finding=f) for f in findings]
    if not any(f["kind"] == "acceptance_before_required_test" for f in findings):
        available = _unique(r["record"]["clause_id"] for rows in candidates.values() for r in rows)
        evidence = [_evidence(records[cid]) for cid in available]
        missing = [label for kind, label in (("payment", "验收后尾款支付"), ("deemed", "默认验收期限"),
                                            ("test", "必要测试及最早完成时间")) if not candidates[kind]]
        reasons = (["缺少可共同核对的原文前提：" + "、".join(missing) + "；未形成三证据组合不能视为通过。"]
                   if available and missing else [])
        if available:
            reasons.extend(_source_pending([records[cid] for cid in available], document))
            if any(_PRIORITY.search(record["text"]) for record in records.values()):
                reasons.append("优先适用或修订约定尚待核对；本次未生成早触发发现不构成完整安排通过。")
            reasons.extend(coverage_pending)
        check = _receipt(contract_id, _TEMPORAL, "performance.acceptance", evidence=evidence, reasons=reasons)
        if not available:
            check["pending_reasons"] = ["未定位到本有界三证据规则所需的可执行原文；不构成验收或付款安排通过。"]
        elif not reasons:
            check["pending_reasons"] = ["未形成默认期限早于必要测试的可确认组合；本有界结果不构成完整验收或付款安排通过。"]
        checks.append(check)
    if violations:
        checks.append(_receipt(contract_id, VERSION + "-constraints", "consistency.references",
                               reasons=[v["message"] for v in violations]))
    constraint_checks = []
    for kind, kinds in (("unique_clause_ids", {"duplicate_clause_id", "clause_id_missing"}),
                        ("source_quote_bindings", {"source_binding_invalid", "source_text_mismatch", "full_text_coverage_gap", "source_metadata_conflict", "proposition_binding_invalid", "relation_binding_invalid"}),
                        ("edge_endpoints", {"edge_endpoint_invalid"})):
        constraint_checks.append({"kind": kind, "status": "invalid" if any(v["kind"] in kinds for v in violations) else "valid"})
    result = {"schema_version": VERSION, "execution_status": "completed", "source_documents": source_documents,
              "proposition_graph": graph,
              "constraints": {"status": "invalid" if violations else "valid", "scope_complete": not violations,
                              "checks": constraint_checks, "violations": violations},
              "findings": findings, "checks": checks,
              "summary": {"clause_count": len(clauses), "source_count": len(source_documents),
                          "constraint_violation_count": len(violations), "finding_count": len(findings),
                          "pending_count": sum(c["manual_required"] is True for c in checks)},
              "legal_truth_established": False, "automatic_legal_confirmation_allowed": False}
    result["completeness"] = _completeness(completeness_check, contract, result)
    if result["completeness"].get("status") != "completed" or result["completeness"].get("manual_required") is True:
        result["summary"]["pending_count"] += 1
    return result
