"""Conservative claim-to-evidence support checks, separate from model output.

supported means a narrow literal proposition is supported by the cited text;
unsupported means this citation does not support the assertion, not that its
opposite is proved.  unknown is an explicit abstention.  No result establishes
legal truth, official subject authenticity, complete contract coverage or risk
confirmation.  Original findings, messages and citations are never rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .fact_slots import extract_fact_slots
from .initial_review import source_sha256
from .proposition_graph import build_proposition_graph

VERSION = "claim-text-support-v1"
_TOPICS = {
    "fee_mismatch": r"总价|价款|费用|金额|元|租金|款项|预付款|尾款",
    "amount_inconsistent": r"大写|小写|金额|总价|价款|元",
    "one_sided_breach": r"违约|赔偿|责任|免责|违约金",
    "liability_exemption": r"责任|免责|赔偿|承担",
    "vague_delivery": r"交付|交货|运输|送达|期限|履行时间",
    "unclear_standard": r"验收|标准|质量|条件|付款|节点|价款",
    "subject_unspecified": r"标的|规格|型号|技术参数|配置|服务范围|货物名称",
    "missing_party_info": r"当事人|买方|卖方|甲方|乙方|组织机构|统一社会信用|代表|签字|名称",
    "need_license": r"资质|许可|经营范围|资格|执照",
    "need_authorization": r"授权|代理|代表|委托|签字",
}
_ABSENCE = re.compile(r"未(?:明确|提供|载明|约定|说明|列明|写明)|缺少|缺失|没有|仅(?:约定|约束|提及|表述)|只(?:约束|约定)")


def _id(value):
    return "claim_" + hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


def assess_claim(finding: Mapping[str, Any], source_segments: Sequence[Mapping[str, Any]], *,
                 contract_type: str = "通用", scope_complete: bool = False,
                 slot_documents: Sequence[Mapping[str, Any]] | None = None,
                 proposition_graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assess the exact cited sources while retaining broader-scope unknowns.

    ``scope_complete`` is a caller-supplied scope assertion, not inferred from
    one window's transport completion.  Even True does not prove absence of an
    omitted attachment or establish the truth of a legal conclusion.
    """
    original = copy.deepcopy(dict(finding)) if isinstance(finding, Mapping) else copy.deepcopy(finding)
    segments = copy.deepcopy(list(source_segments))
    reasons, unsupported, support, bindings, invalid = [], [], [], [], []
    valid_segments = []
    for index, segment in enumerate(segments):
        if (not isinstance(segment, Mapping) or not isinstance(segment.get("clause_id"), str) or not segment.get("clause_id")
                or not isinstance(segment.get("text"), str) or type(segment.get("start")) is not int
                or segment["start"] < 0 or type(segment.get("end")) is not int
                or segment["end"] - segment["start"] != len(segment["text"])):
            invalid.append(index)
        else:
            valid_segments.append(segment)
    if invalid:
        unsupported.append("source_segment_binding_invalid")
    if not isinstance(finding, Mapping):
        finding = {}
    message, code = finding.get("message", ""), finding.get("code", "")
    if not isinstance(message, str) or not message.strip():
        unsupported.append("claim_message_missing")
        message = ""
    if not isinstance(code, str):
        code = ""
    related = finding.get("related_quotes", [])
    rows = [{"clause_id": finding.get("clause_id"), "quote": finding.get("quote")}]
    if isinstance(related, list):
        rows.extend(related)
    else:
        unsupported.append("related_evidence_invalid")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("quote"), str) or not row["quote"].strip():
            unsupported.append("quoted_evidence_missing")
            continue
        found = []
        for segment_index, segment in enumerate(valid_segments):
            if segment["clause_id"] != row.get("clause_id"):
                continue
            cursor = 0
            while (offset := segment["text"].find(row["quote"], cursor)) >= 0:
                found.append({"segment_index": segment_index, "clause_id": segment["clause_id"],
                    "start": segment["start"] + offset, "end": segment["start"] + offset + len(row["quote"]),
                    "quote": row["quote"], "segment_sha256": hashlib.sha256(segment["text"].encode("utf-8")).hexdigest()})
                cursor = offset + 1
        if not found:
            unsupported.append("citation_not_exactly_source_bound")
        bindings.append({"citation_index": row_index, "matches": found})
    cited_texts = [row.get("quote", "") for row in rows if isinstance(row, Mapping) and isinstance(row.get("quote"), str)]
    cited_text = "\n".join(cited_texts)
    window_text = "\n".join(segment["text"] for segment in valid_segments)
    documents = list(slot_documents) if slot_documents is not None else [extract_fact_slots(
        segment["clause_id"], segment["text"], start_offset=segment["start"]) for segment in valid_segments]
    graph = dict(proposition_graph) if proposition_graph is not None else build_proposition_graph(documents, scope_complete=scope_complete)
    cited_documents = []
    for binding in bindings:
        if binding["matches"]:
            match = binding["matches"][0]
            cited_documents.append(extract_fact_slots(match["clause_id"], match["quote"], start_offset=match["start"]))
    cited_amounts = [amount for document in cited_documents for amount in document["amounts"]]
    full_amounts = [amount for document in documents for amount in document.get("amounts", [])]
    claimed_amounts = extract_fact_slots("claim_message", message)["amounts"] if message else []
    if code in _TOPICS and not re.search(_TOPICS[code], cited_text):
        unsupported.append("cited_text_has_no_claim_topic")
    claimed_actors = set(re.findall(r"甲方|乙方|丙方|买方|卖方|出租方|承租方", message))
    cited_actors = set(re.findall(r"甲方|乙方|丙方|买方|卖方|出租方|承租方", cited_text))
    if claimed_actors - cited_actors:
        unsupported.append("claimed_actor_not_in_cited_evidence")
    account_primary = cited_texts[0] if cited_texts else ""
    account_dominant = bool("账号" in account_primary and len(re.findall(r"\d", account_primary)) / max(1, len(account_primary)) >= .25)
    if account_dominant and code in _TOPICS:
        unsupported.append("account_extract_does_not_support_substantive_claim")
    if re.search(r"租金|承租|出租", message) and not re.search(r"租金|租赁|承租|出租", window_text):
        if re.search(r"采购|买卖", contract_type) or re.search(r"采购|买方|卖方|货物名称", window_text):
            unsupported.append("unfounded_lease_assumption_in_purchase_scope")
        else:
            reasons.append("contract_type_for_rent_claim_unknown")
    if _ABSENCE.search(message):
        reasons.append("absence_or_only_claim_not_proved_by_cited_text")
        if not scope_complete:
            reasons.append("local_absence_cannot_establish_whole_contract_absence")
    if re.search(r"违反|违法|无效|非法|不符合法律", message):
        reasons.append("legal_conclusion_not_verified_by_text_support_check")
    if claimed_amounts:
        source_values = {amount["value_yuan"] for amount in cited_amounts if amount["value_yuan"] is not None}
        if any(amount["value_yuan"] not in source_values for amount in claimed_amounts if amount["value_yuan"] is not None):
            unsupported.append("claimed_monetary_value_not_in_cited_evidence")
    amount_roles = {amount["amount_role"] for amount in full_amounts}
    if {"contract_total", "tail"}.issubset(amount_roles) and re.search(
            r"总价.{0,40}尾款.{0,35}(?:相加|合计|不一致|冲突|超过)|尾款.{0,40}总价.{0,35}(?:相加|合计|不一致|冲突|超过)", message):
        if not any(relation["kind"] == "explicit_advance_tail_partition" and not relation["arithmetic_consistent"] for relation in graph.get("relations", [])):
            unsupported.append("contract_total_and_tail_are_not_additive_or_same_role")
    if code == "one_sided_breach" and re.search(r"仅|只", message) and "违约" not in window_text:
        unsupported.append("asserted_one_sided_breach_terms_absent_from_supplied_scope")
    if code == "dangling_reference":
        reasons.append("attachment_or_reference_absence_not_established")
    if code == "amount_inconsistent":
        comparisons = []
        for document in cited_documents:
            for sentence in document["sentences"]:
                if "大写" not in sentence["span"]["quote"] or "小写" not in sentence["span"]["quote"]:
                    continue
                amounts = [row for row in document["amounts"] if row["sentence_span"] == sentence["span"]]
                chinese = [row for row in amounts if row["writing"] == "chinese"]
                arabic = [row for row in amounts if row["writing"] == "arabic"]
                if len(chinese) == len(arabic) == 1 and chinese[0]["value_yuan"] is not None and arabic[0]["value_yuan"] is not None:
                    comparisons.append((chinese[0]["value_yuan"], arabic[0]["value_yuan"]))
        if len(comparisons) == 1:
            left, right = comparisons[0]
            if left == right:
                unsupported.append("explicit_uppercase_lowercase_values_are_equal")
            else:
                support.append("explicit_uppercase_lowercase_literal_value_difference")
        else:
            reasons.append("no_unambiguous_explicit_uppercase_lowercase_pair")
    if (code == "liability_exemption" and re.search(r"不承担(?:任何)?(?:赔偿)?责任|免除.{0,8}责任", cited_text)
            and re.search(r"不承担.{0,8}责任|免除.{0,8}责任|免责", message) and not _ABSENCE.search(message)):
        support.append("literal_exemption_wording_present")

    def cited(span):
        return any(match["clause_id"] == span["clause_id"] and match["start"] <= span["start"]
                   and span["end"] <= match["end"] for binding in bindings for match in binding["matches"])

    supporting_relations = []
    for relation in graph.get("relations", []):
        if (code in ("fee_mismatch", "cross_clause_inconsistency") and relation["kind"] == "same_role_literal_value_difference"
                and all(cited(span) for span in relation.get("source_spans", []))):
            support.append("source_bound_same_role_literal_value_difference")
            supporting_relations.append(relation)
        if (code == "fee_mismatch" and relation["kind"] == "explicit_advance_tail_partition"
                and not relation["arithmetic_consistent"] and all(cited(span) for span in relation.get("source_spans", []))):
            support.append("source_bound_explicit_partition_arithmetic_difference")
            supporting_relations.append(relation)
    if not support and not unsupported:
        reasons.append("no_narrow_literal_support_rule_establishes_claim")
    status = "unsupported" if unsupported else "unknown" if reasons or not support else "supported"
    result = {"version": VERSION, "claim_grounding_status": status,
        "manual_required": status != "supported" or bool(supporting_relations and graph.get("manual_required")),
        "automatic_confirmation_allowed": False, "legal_truth_established": False,
        "scope": "supplied_scope_complete_asserted" if scope_complete else "supplied_window_only",
        "scope_complete_asserted_by_caller": scope_complete is True,
        "reasons": list(dict.fromkeys([*unsupported, *reasons])), "literal_support_reasons": list(dict.fromkeys(support)),
        "finding_unchanged": original, "source_binding_matches": bindings,
        "window_source_sha256": source_sha256(segments) if not invalid else None,
        "cited_fact_slots": cited_documents, "supporting_literal_relations": supporting_relations,
        "account_dominant_primary_quote": account_dominant,
        "meaning": "Textual support only: unsupported does not prove the opposite; unknown is abstention; no legal or factual confirmation"}
    result["assessment_id"] = _id({"finding": original, "source_sha256": result["window_source_sha256"], "version": VERSION})
    return result
