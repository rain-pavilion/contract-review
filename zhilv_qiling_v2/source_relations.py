"""Source-bound literal relations and extraction-coverage gates.

Two narrow, rule-based safeguards used by the review nodes:

* extraction_gap_metadata recognizes a span-less extraction notice (e.g.
  "未识别到标的/项目名称").  Such a notice is kept for manual review but can
  never be scored as a contract defect: an extraction miss does not prove the
  contract is silent.
* amount_relation_candidates compares exactly comparable monetary slots of the
  same payment role, subject, object, stage, condition set and absolute date.
  It emits a literal-difference candidate with both exact source spans.  It
  never aggregates contract totals with instalments, never guesses an actor or
  item, never treats a condition as satisfied and never asserts a legal
  conflict.

Everything here is deterministic and reads only the supplied clause text.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import itertools
import json
import re
from typing import Any

from .models import Evidence, RiskCandidate, SpanRef

_GAP_KINDS = {"party_missing", "missing_party_info", "amount_missing", "date_missing",
              "subject_unspecified"}
_EXTRACTION_NOTE = re.compile(r"未(?:抽取|识别)到|当事人角色不完整.*已识别|未挂载任何要素")
_ROLE_NAMES = {
    "contract_total": "合同总价", "advance": "预付款", "first_payment": "首付款",
    "first_instalment": "首期款", "second_instalment": "第二期款", "third_instalment": "第三期款",
    "tail": "尾款", "retention": "质保金", "deposit": "保证金", "rent": "租金",
}
_PHASE = re.compile(r"第[一二三四五六七八九十百0-9]+(?:期|阶段|批)|首批|末批|最终|初步")
_DUE = re.compile(r"(?:20\d{2}年)?\d{1,2}月(?:\d{1,2}[日号])?"
                  r"|(?:[一二三四五六七八九十百\d]+)(?:个工作日|工作日|日|天|个月)内")
_ABSOLUTE_DUE = re.compile(r"(?:20\d{2}|\d{2})年|\d{1,2}月\d{1,2}[日号]|20\d{2}[年/-]\d{1,2}")
_BASIS = re.compile(r"每(?:月|年|季度|日|套|台|件)|按(?:月|年|季度)|/(?:月|年|套|台|件)")
_AMENDMENT = re.compile(r"原(?:定|约定|合同|价)|改为|调整为|变更为|修订|补充协议|更正|替代|取代|"
                        r"不再适用|以[^。；;\n]{1,40}为准|另行约定|折后|优惠|减免|扣减")
_RANGE = re.compile(r"(?:元|万)[^。；;\n]{0,4}(?:至|到|~|～|—)|介于|区间|范围内")
_OBJECTS = re.compile(r"本合同|本项目|本设备|本服务|(?:项目|设备|标的)[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+")
_SAME_ITEM = re.compile(r"同一笔|该笔|上述一笔|仅(?:约定|有)?一笔|只有一笔|仅一笔")
_PARTIES = re.compile(r"甲方|乙方|丙方|买方|卖方|出租方|承租方")
_PAY_TO = re.compile(r"(?:甲方|乙方|买方|卖方)(?:应当|应|须|需)?向(?:甲方|乙方|买方|卖方)[^，。；;\n]{0,12}?(?:支付|付清|付款)")
_CLAUSE_LABEL = re.compile(r"^\s*(第[一二三四五六七八九十百0-9]+条)")


def _id(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


def extraction_gap_metadata(signal: Any) -> dict[str, Any]:
    """Recognize extraction-coverage notices, not every span-less or missing code."""
    if (getattr(signal, "kind", None) not in _GAP_KINDS or getattr(signal, "spans", None)
            or not _EXTRACTION_NOTE.search(str(getattr(signal, "note", "")))):
        return {}
    return {"claim_grounding_status": "extraction_gap", "evidence_kind": "extraction_gap",
            "manual_required": True, "score_eligible": False,
            "fact_key": "extraction_gap:" + str(signal.kind) + ":" + str(signal.clause_id or "document"),
            "reasons": ["extraction_absence_does_not_prove_contract_absence"],
            "legal_truth_established": False,
            "support_scope": "extraction_coverage_only"}


def mark_extraction_gap(candidate: RiskCandidate, signal: Any) -> RiskCandidate:
    """Staged adapter for tests; production stamps metadata at candidate creation."""
    result = copy.deepcopy(candidate)
    metadata = extraction_gap_metadata(signal)
    if metadata:
        result.extra.update(metadata)
        result.evidence.append(Evidence(kind="extraction_gap", source="element_extraction",
            text="抽取未识别到该要素，不等于合同正文未约定；需人工核对原文。", strength=0,
            extra=copy.deepcopy(metadata)))
    return result


_ACTOR_PREFIX = re.compile(r"^(?:甲方|乙方|买方|卖方|出租方|承租方)")


def _effective_conditions(row: dict) -> list[dict]:
    """Drop a verb phrase mis-parsed as a second condition.

    The third clause states the condition "本合同最终验收后的同一笔尾款"; the
    sixth clause sentence "向乙方支付本合同最终验收后的同一笔尾款" also yields
    an actor-prefixed pseudo-condition "乙方支付本合同最终验收后".  It
    duplicates the acceptance condition already present in the same row, so
    keeping it would split one payment role into two incomparable nodes.  The
    rule is narrow: actor-prefixed, carries a payment verb, and its
    non-payment events exactly match another condition with the same timing.
    """
    conditions = list(row.get("conditions") or [])
    kept = []
    for condition in conditions:
        events = set(condition.get("events") or [])
        quote = str((condition.get("span") or {}).get("quote") or "")
        if ("payment" in events and _ACTOR_PREFIX.match(quote) and events - {"payment"}
                and any(set(other.get("events") or []) == events - {"payment"}
                        and other.get("timing") == condition.get("timing")
                        for other in conditions if other is not condition)):
            continue
        kept.append(condition)
    return kept


def _normalized_conditions(row: dict) -> tuple:
    """Event/timing/polarity set: duplicate spans of the same condition collapse."""
    return tuple(sorted({(tuple(sorted(str(event) for event in (item.get("events") or []))),
                          str(item.get("timing") or ""), bool(item.get("negated")))
                         for item in _effective_conditions(row)}))


def _due_parts(row: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split absolute dates from relative deadlines; only absolute dates identify a node."""
    text = str((row.get("sentence_span") or {}).get("quote") or "")
    values = list(dict.fromkeys(_DUE.findall(text)))
    absolute = tuple(value for value in values if _ABSOLUTE_DUE.search(value))
    relative = tuple(value for value in values if not _ABSOLUTE_DUE.search(value))
    return absolute, relative


def _clause_label(text: str, clause_id: str) -> str:
    match = _CLAUSE_LABEL.search(text or "")
    return match.group(1) if match else clause_id


def _local_scope(row: dict, document: dict) -> tuple[tuple | None, list[str], dict[str, Any]]:
    """Resolve one amount sentence into a comparable literal scope.

    Returns (scope, reasons, detail).  The scope is comparable only when the
    payment role, explicit object, stage, rate basis, normalized condition set
    and absolute date are unambiguous.  Relative deadlines (e.g. 五日内) are
    kept in detail for the message but do not split one payment role into two
    nodes: they are a performance deadline, not a different instalment.
    """
    reasons: list[str] = []
    detail: dict[str, Any] = {}
    text = row["sentence_span"]["quote"]
    if row["amount_role"] not in _ROLE_NAMES:
        reasons.append("amount_role_not_specific")
    if row["value_yuan"] is None:
        reasons.append("amount_value_not_parsed")
    if len(set(_OBJECTS.findall(text))) > 1:
        reasons.append("multiple_explicit_objects")
    if row["value_kind"] != "literal" or row["polarity"] != "positive":
        reasons.append("qualified_or_negative_amount")
    if row["exceptions"]:
        reasons.append("exception_scope_not_flattened")
    if _AMENDMENT.search(text):
        reasons.append("amendment_or_price_adjustment_not_flattened")
    if _RANGE.search(text):
        reasons.append("amount_range_not_an_exact_value")
    phases = tuple(dict.fromkeys(_PHASE.findall(text)))
    absolute_dues, relative_dues = _due_parts(row)
    bases = tuple(dict.fromkeys(_BASIS.findall(text)))
    if len(phases) > 1 or len(absolute_dues) > 1 or len(bases) > 1:
        reasons.append("multiple_local_stages_dates_or_bases")
    detail["relative_dues"] = relative_dues
    detail["absolute_dues"] = absolute_dues
    if any(item.get("unparsed_condition_markers") and item["span"] == row["sentence_span"]
           for item in document["sentences"]):
        reasons.append("condition_grammar_unresolved")
    # A nearby previous role cannot be blindly assigned to a new, unlabeled fee.
    same_role_anchor = [item for item in document["amounts"]
                        if item["role_span"] == row["role_span"]
                        and item["source_span"]["start"] < row["source_span"]["start"]]
    if same_role_anchor:
        previous = max(same_role_anchor, key=lambda item: item["source_span"]["end"])
        offset = document["source"]["start_offset"]
        between = document["source"]["text"][previous["source_span"]["end"]-offset:row["source_span"]["start"]-offset]
        if not (re.search(r"大写|小写", between) and not re.search(r"另|及|和|费|罚|利息|赔偿", between)):
            reasons.append("nearest_role_does_not_bind_additional_amount")
    if reasons:
        return None, reasons, detail
    return (row["amount_role"], phases, absolute_dues, bases,
            _normalized_conditions(row)), [], detail


def _phrase_reference(row: dict, document: dict) -> dict | None:
    """Locate an explicit same-item phrase (同一笔/该笔/仅一笔) inside the amount sentence."""
    label = _ROLE_NAMES.get(row["amount_role"])
    if not label or row["amount_role"] == "contract_total":
        return None
    quote = row["sentence_span"]["quote"]
    match = _SAME_ITEM.search(quote)
    if not match or label not in quote:
        return None
    tail = quote[match.start():]
    if label not in tail[:len(label) + 6]:
        return None
    marker = tail[:tail.find(label) + len(label)]
    if not marker or marker not in quote:
        return None
    source = document["source"]
    index = source["text"].find(marker)
    if index < 0:
        return None
    return {"clause_id": document["clause_id"], "start": index, "end": index + len(marker),
            "quote": marker, "source_sha256": source["sha256"]}


def _item_reference(row: dict, documents: list[dict]) -> dict | None:
    """Resolve only an explicit single-item declaration or same-item wording.

    No payer is inherited, and ordinary repeated role names are not references.
    """
    role = row["amount_role"]
    label = _ROLE_NAMES.get(role)
    if not label or role == "contract_total":
        return None
    pattern = re.compile(r"本合同(?:约定的)?(?:同一笔|同一项|仅有一笔|仅约定一笔|只有一笔|仅一笔)" + re.escape(label))
    declarations = []
    for document in documents:
        for match in pattern.finditer(document["source"]["text"]):
            declarations.append({"clause_id": document["clause_id"], "start": match.start(),
                                 "end": match.end(), "quote": match.group(),
                                 "source_sha256": document["source"]["sha256"]})
    own = [span for span in declarations if span["clause_id"] == row["role_span"]["clause_id"]
           and span["start"] <= row["role_span"]["start"] < span["end"]]
    if own:
        return {"key": "explicit_contract_single_item:" + role, "spans": own,
                "subject_inherited": False, "basis": "explicit_same_or_single_item_phrase"}
    if len(declarations) != 1:
        return None
    document = next(item for item in documents if item["clause_id"] == row["source_span"]["clause_id"])
    role_start = row["role_span"]["start"]
    prefix = document["source"]["text"][:role_start]
    reference = re.search(r"(?:该笔|该项|上述|同一笔|同一项)$", prefix)
    if reference is None:
        return None
    span = {"clause_id": document["clause_id"], "start": reference.start(),
            "end": row["role_span"]["end"],
            "quote": document["source"]["text"][reference.start():row["role_span"]["end"]],
            "source_sha256": document["source"]["sha256"]}
    return {"key": "explicit_contract_single_item:" + role,
            "spans": [declarations[0], span], "subject_inherited": False,
            "basis": "unique_explicit_declaration_and_anaphoric_phrase"}


def amount_relation_candidates(clauses: list[dict[str, str]], *, contract_id: str,
                               scope_complete: bool = False) -> dict[str, Any]:
    """Turn strictly comparable, exact source slots into review candidates.

    Inputs are actual clause IDs and unmodified clause text. Nothing reads a
    benchmark answer, guesses actors/objects, or sums unrelated monetary values.
    """
    from .fact_slots import extract_fact_slots
    from .proposition_graph import build_proposition_graph

    documents = [extract_fact_slots(row["clause_id"], row["text"]) for row in clauses]
    clause_text = {row["clause_id"]: row["text"] for row in clauses}
    graph = build_proposition_graph(documents, scope_complete=scope_complete)
    invalid = set(graph["invalid_source_clause_ids"])
    groups: dict[tuple, list[dict]] = {}
    diagnostics: list[dict] = []
    observations: list[dict] = []
    details: dict[str, dict] = {}
    for document in documents:
        if document["clause_id"] in invalid:
            continue
        for row in document["amounts"]:
            scope, reasons, detail = _local_scope(row, document)
            if scope is None:
                diagnostics.append({"slot_id": row["slot_id"], "source_span": row["source_span"],
                                    "reasons": reasons})
                continue
            groups.setdefault(scope, []).append(row)
            details[row["slot_id"]] = detail
    candidates: list[RiskCandidate] = []
    seen: set[str] = set()
    for scope, rows in groups.items():
        # This enumerates only one exact role/stage/absolute-date/basis/condition bucket.
        for left, right in itertools.combinations(rows, 2):
            if left["slot_id"] == right["slot_id"] or Decimal(left["value_yuan"]) == Decimal(right["value_yuan"]):
                continue
            bindings = [(row["subject"], row["recipient"], row["object_scope"]["name"]) for row in (left, right)]
            pair = [left["slot_id"], right["slot_id"]]
            if any(a is not None and b is not None and a != b for a, b in zip(*bindings)):
                diagnostics.append({"slot_ids": pair, "reasons": ["explicit_subject_recipient_or_object_differs"]})
                continue
            source_spans = [copy.deepcopy(row["source_span"]) for row in (left, right)]
            references = [_item_reference(row, documents) for row in (left, right)]
            explicit_same_item = all(references) and references[0]["key"] == references[1]["key"]
            if not explicit_same_item:
                phrase_spans = []
                for row in (left, right):
                    document = next(item for item in documents
                                    if item["clause_id"] == row["source_span"]["clause_id"])
                    phrase_spans.append(_phrase_reference(row, document))
                if all(phrase_spans) and phrase_spans[0]["quote"] == phrase_spans[1]["quote"]:
                    explicit_same_item = True
                    references = [{"key": "explicit_same_item_phrase:" + scope[0],
                                   "spans": [phrase_spans[0], phrase_spans[1]],
                                   "subject_inherited": False,
                                   "basis": "same_literal_same_item_phrase_in_both_sentences"}]
            if not explicit_same_item and (any(binding[0] is None or binding[2] is None for binding in bindings)
                                           or bindings[0] != bindings[1]):
                observations.append({"kind": "same_role_amounts_scope_pending", "status": "unknown",
                    "slot_ids": pair, "source_spans": source_spans,
                    "reason": "subject_recipient_or_object_not_jointly_bound",
                    "manual_required": True, "score_eligible": False, "legal_conflict_established": False})
                continue
            fact_key = _id("amount_relation_", [contract_id, sorted(pair), scope])
            if fact_key in seen:
                continue
            seen.add(fact_key)
            role = _ROLE_NAMES[scope[0]]
            left_label = _clause_label(clause_text.get(left["source_span"]["clause_id"], ""),
                                       left["source_span"]["clause_id"])
            right_label = _clause_label(clause_text.get(right["source_span"]["clause_id"], ""),
                                        right["source_span"]["clause_id"])
            description = ("原文明示同指的" + role if explicit_same_item
                           else "同一显式绑定的" + str(bindings[0][2]) + role)
            due_left = details.get(left["slot_id"], {}).get("relative_dues", ())
            due_right = details.get(right["slot_id"], {}).get("relative_dues", ())
            if due_left == due_right:
                due_note = ""
            elif due_left and due_right:
                due_note = ("两处付款期限表述不同（" + "/".join(due_left) + "，" + "/".join(due_right)
                            + "），需核对是否同一履行节点；")
            elif due_right:
                due_note = ("一处未写付款期限、另一处为「" + "/".join(due_right)
                            + "」，需核对是否同一履行节点；")
            else:
                due_note = ("一处为「" + "/".join(due_left) + "」、另一处未写付款期限，需核对是否同一履行节点；")
            condition_note = ("两处条件均对应同一事件，但不代表条件已经满足；" if scope[4] else "")
            clause_parties = {row["source_span"]["clause_id"]: sorted(set(_PARTIES.findall(
                clause_text.get(row["source_span"]["clause_id"], ""))))
                for row in (left, right)}
            pay_to = [_PAY_TO.search(clause_text.get(row["source_span"]["clause_id"], "")) is not None
                      for row in (left, right)]
            if all(pay_to):
                recipient_note = "两处所在条款均有甲方向乙方支付的明示；"
            elif left["recipient"] is None and right["recipient"] is None:
                recipient_note = "收款方未在两处金额句内分别重述，需核对主体绑定；"
            else:
                recipient_note = ""
            message = (description + "出现不同金额记载："
                       + left_label + "「" + left["source_span"]["quote"] + "」与"
                       + right_label + "「" + right["source_span"]["quote"] + "」；"
                       + condition_note + due_note + recipient_note
                       + "需核对具体适用条款及是否存在有效变更。")
            extra = {"fact_key": fact_key, "issue_code": "same_role_amount_difference",
                     "claim_grounding_status": "supported" if scope_complete else "unknown",
                     "manual_required": True, "legal_conflict_established": False,
                     "source_relation": {"kind": "same_role_literal_value_difference", "slot_ids": pair,
                        "source_spans": source_spans, "role": scope[0], "subject": bindings[0][0],
                        "recipient": bindings[0][1], "object_scope": bindings[0][2],
                        "clause_parties": clause_parties,
                        "explicit_item_references": references if explicit_same_item else [],
                        "phase": scope[1], "absolute_due": scope[2], "basis": scope[3],
                        "conditions": scope[4], "relative_due": [list(due_left), list(due_right)],
                        "conditions_assumed_true": False, "scope_complete": scope_complete}}
            # Amount spans bind the exact values; the enclosing sentence spans bind
            # the same-item phrase and clause context used by the message.
            span_inputs = [*source_spans,
                           copy.deepcopy(left["sentence_span"]), copy.deepcopy(right["sentence_span"])]
            spans = [SpanRef(block_id=span["clause_id"], clause_id=span["clause_id"],
                             char_start=span["start"], char_end=span["end"], quote=span["quote"])
                     for span in span_inputs]
            candidates.append(RiskCandidate(candidate_id=fact_key, code="cross_clause_inconsistency",
                message=message, severity="medium",
                clause_ids=list(dict.fromkeys(span.clause_id for span in spans)),
                spans=spans, confidence=.7, source="literal_amount_relation",
                evidence=[Evidence(kind="literal_relation", text=message, source="literal_amount_relation",
                                   span=spans[0], strength=.7, extra=copy.deepcopy(extra))], extra=extra))
    return {"candidates": candidates, "observations": observations, "diagnostics": diagnostics,
            "slot_documents": documents, "proposition_graph": graph,
            "scope_complete": scope_complete, "legal_truth_established": False,
            "amount_aggregation_performed": False}
