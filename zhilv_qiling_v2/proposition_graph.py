"""Conservative combinations of exact literal slots, not a legal reasoner.

Conditions are prerequisite edges, never assumed true.  Exceptions, negation,
unbound subjects and unexamined cross-window scope prevent an unconditional
conclusion.  Contract totals are not payment instalments and are never added to
their own components.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

VERSION = "source-bound-proposition-graph-v1"


def _id(value):
    return "relation_" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def _span_valid(span, document):
    if not isinstance(span, dict):
        return False
    source = document["source"]
    start, end = span.get("start"), span.get("end")
    return (type(start) is int and type(end) is int and source["start_offset"] <= start < end <= source["end_offset"]
        and span.get("clause_id") == document["clause_id"] and span.get("source_sha256") == source["sha256"]
        and source["text"][start-source["start_offset"]:end-source["start_offset"]] == span.get("quote"))


def _source_spans(value):
    if isinstance(value, dict):
        if {"clause_id", "start", "end", "quote", "source_sha256"}.issubset(value):
            yield value
        else:
            for item in value.values():
                yield from _source_spans(item)
    elif isinstance(value, list):
        for item in value:
            yield from _source_spans(item)


def _context(row):
    return (row.get("subject"), row.get("recipient"), row.get("object_scope", {}).get("name"),
        row.get("stage"), tuple(item["span"]["quote"] for item in row.get("conditions", [])),
        tuple(item["quote"] for item in row.get("exceptions", [])), row.get("polarity"))


def build_proposition_graph(slot_documents: Sequence[Mapping[str, Any]], *, scope_complete: bool = False,
                            unresolved_cross_window_relations: Sequence[str] = ()) -> dict[str, Any]:
    """Return literal relations with explicit unknowns and immutable source IDs."""
    documents = list(slot_documents)
    amounts, propositions, invalid_sources = [], [], []
    for document in documents:
        try:
            source = document["source"]
            good = (hashlib.sha256(source["text"].encode("utf-8")).hexdigest() == source["sha256"]
                and source["end_offset"] - source["start_offset"] == len(source["text"])
                and all(_span_valid(span, document) for span in _source_spans(
                    {key: document.get(key) for key in ("amounts", "obligations", "sentences")})))
        except (KeyError, TypeError, AttributeError):
            good = False
        if not good:
            invalid_sources.append(str(document.get("clause_id", "unknown")))
            continue
        amounts.extend(document.get("amounts", []))
        propositions.extend(row for row in document.get("obligations", []) if row.get("mention_role") == "operative")
    relations, unknowns, edges = [], [], []
    for document in documents:
        if str(document.get("clause_id")) in invalid_sources:
            continue
        for sentence in document.get("sentences", []):
            if sentence.get("unparsed_condition_markers"):
                unknowns.append({"kind": "unparsed_condition", "source_span": sentence["span"],
                    "reason": "temporal_or_conditional_marker_outside_supported_grammar"})
    non_additive_pairs = []
    for index, left in enumerate(amounts):
        for right in amounts[index+1:]:
            if left["slot_id"] == right["slot_id"]:
                continue
            if {left["amount_role"], right["amount_role"]} == {"contract_total", "tail"}:
                non_additive_pairs.append({"slot_ids": [left["slot_id"], right["slot_id"]],
                    "reason": "contract_total_and_tail_are_distinct_non_additive_roles"})
            if left["amount_role"] != right["amount_role"] or left["amount_role"] == "unknown":
                continue
            if left["value_yuan"] == right["value_yuan"]:
                continue
            context = _context(left)
            if (_context(left) != _context(right) or context[0] is None or context[2] is None
                    or left["polarity"] != "positive" or left["conditions"] or left["exceptions"]
                    or left["value_kind"] != "literal" or right["value_kind"] != "literal"):
                unknowns.append({"kind": "amount_comparison", "slot_ids": [left["slot_id"], right["slot_id"]],
                    "reason": "amount_subject_scope_stage_condition_exception_or_polarity_not_jointly_resolved"})
                continue
            row = {"kind": "same_role_literal_value_difference", "status": "supported",
                "slot_ids": [left["slot_id"], right["slot_id"]],
                "values_yuan": [left["value_yuan"], right["value_yuan"]],
                "source_spans": [left["source_span"], right["source_span"]],
                "legal_conflict_established": False, "manual_required": True}
            row["relation_id"] = _id(row)
            relations.append(row)
    # Only an explicit partition sentence permits this narrow calculation.
    # No generic sum of all amounts or role guessing is performed.
    for document in documents:
        if str(document.get("clause_id")) in invalid_sources:
            continue
        for sentence in document.get("sentences", []):
            quote = sentence["span"]["quote"]
            if not ("由" in quote and "组成" in quote and "预付款" in quote and "尾款" in quote):
                continue
            members = [row for row in document.get("amounts", []) if row["sentence_span"] == sentence["span"] and row["writing"] == "arabic"]
            groups = {role: [row for row in members if row["amount_role"] == role] for role in ("contract_total", "advance", "tail")}
            if (any(len(value) != 1 for value in groups.values()) or len(members) != 3
                    or any(row["subject"] is None or row["object_scope"]["name"] is None or row["conditions"]
                           or row["exceptions"] or row["polarity"] != "positive" or row["value_kind"] != "literal" for row in members)
                    or len({(row["subject"], row["recipient"], row["object_scope"]["name"]) for row in members}) != 1):
                unknowns.append({"kind": "explicit_partition", "source_span": sentence["span"],
                    "reason": "partition_members_or_subject_scope_qualifiers_unresolved"})
                continue
            total, advance, tail = (groups[role][0] for role in ("contract_total", "advance", "tail"))
            component_sum = Decimal(advance["value_yuan"]) + Decimal(tail["value_yuan"])
            row = {"kind": "explicit_advance_tail_partition", "status": "supported", "manual_required": True,
                "slot_ids": [total["slot_id"], advance["slot_id"], tail["slot_id"]],
                "total_yuan": total["value_yuan"], "component_sum_yuan": format(component_sum, "f"),
                "arithmetic_consistent": component_sum == Decimal(total["value_yuan"]),
                "source_spans": [row["source_span"] for row in (total, advance, tail)], "legal_conflict_established": False}
            row["relation_id"] = _id(row)
            relations.append(row)
    # Resolve only explicit actor/object/stage/event conditions.  Missing actor
    # or phase is not silently supplied by another clause.
    def payment_role(proposition):
        roles = {row["amount_role"] for row in amounts if row["sentence_span"] == proposition["sentence_span"]
                 and row["amount_role"] != "unknown"}
        return next(iter(roles)) if len(roles) == 1 else "unknown"

    def signature(proposition):
        return (proposition["event"], proposition["subject"], proposition["object_scope"]["name"],
                proposition["stage"], payment_role(proposition) if proposition["event"] == "payment" else "not_payment")

    for target in propositions:
        if target["exceptions"] or target["polarity"] == "negative":
            unknowns.append({"kind": "qualified_obligation", "proposition_id": target["proposition_id"],
                "reason": "exception_or_negation_retained_not_flattened", "source_span": target["sentence_span"]})
        for condition in target["conditions"]:
            quote = condition["span"]["quote"]
            actors = set(re.findall(r"甲方|乙方|丙方|买方|卖方", quote))
            scopes = set(re.findall(r"本合同|本项目|本设备|本服务|(?:项目|设备|标的)[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+", quote))
            stages = re.findall(r"最终|初步|首批|第[一二三四五六七八九十0-9]+批", quote)
            role = "tail" if "尾款" in quote else "advance" if "预付款" in quote else "unknown"
            condition_signature = (condition["events"][0] if len(condition["events"]) == 1 else None,
                next(iter(actors)) if len(actors) == 1 else None, next(iter(scopes)) if len(scopes) == 1 else None,
                stages[-1] if stages else "unspecified", role if condition["events"] == ["payment"] else "not_payment")
            matching = [item for item in propositions if signature(item) == condition_signature]
            resolved = (len(matching) == 1 and condition_signature[0] is not None
                and condition_signature[1] is not None and condition_signature[2] is not None
                and condition["timing"] == "after" and not condition["negated"]
                and target["polarity"] == "positive" and not target["exceptions"]
                and matching[0]["polarity"] == "positive" and not matching[0]["exceptions"])
            edge = {"kind": "explicit_prerequisite", "to": target["proposition_id"],
                "from": matching[0]["proposition_id"] if resolved else None,
                "condition_span": condition["span"], "resolved": resolved,
                "condition_assumed_true": False, "legal_conclusion": None}
            edges.append(edge)
            if not resolved:
                unknowns.append({"kind": "condition_link", "source_span": condition["span"],
                    "reason": "condition_actor_scope_stage_polarity_or_target_unresolved"})
    adjacency = {row["proposition_id"]: [] for row in propositions}
    for edge in edges:
        if edge["resolved"]:
            adjacency[edge["from"]].append(edge["to"])
    cycles = set()

    def visit(start, current, path):
        for other in adjacency.get(current, []):
            if other == start and len(path) > 1:
                cycle = tuple(path)
                cycles.add(min(cycle[index:] + cycle[:index] for index in range(len(cycle))))
            elif other not in path:
                visit(start, other, [*path, other])

    for proposition_id in adjacency:
        visit(proposition_id, proposition_id, [proposition_id])
    for cycle in sorted(cycles):
        row = {"kind": "literal_prerequisite_cycle", "status": "supported", "proposition_ids": list(cycle),
            "manual_required": True, "conditions_satisfied": None, "legal_deadlock_established": False}
        row["relation_id"] = _id(row)
        relations.append(row)
    reasons = (["source_binding_invalid"] if invalid_sources else []) + (["unexamined_cross_window_scope"] if not scope_complete else [])
    if unknowns:
        reasons.append("conditional_or_literal_composition_unknowns")
    if unresolved_cross_window_relations:
        reasons.append("unresolved_cross_window_relations")
    status = "unknown" if reasons else "supported" if relations else "unknown"
    return {"version": VERSION, "proposition_count": len(propositions), "propositions": propositions,
        "amount_slot_ids": [row["slot_id"] for row in amounts], "edges": edges, "relations": relations,
        "non_additive_amount_role_pairs": non_additive_pairs, "unknowns": unknowns,
        "invalid_source_clause_ids": invalid_sources, "scope_complete": scope_complete is True,
        "unresolved_cross_window_relations": list(unresolved_cross_window_relations),
        "manual_required": status != "supported" or bool(relations), "manual_reasons": reasons,
        "status": status,
        "automatic_legal_confirmation_allowed": False, "legal_truth_established": False,
        "amount_aggregation_policy": "No sum except explicit, fully bound advance-plus-tail partition; total is excluded from components"}
