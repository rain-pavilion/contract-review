"""Conservative fact identities and source anchors across detection channels.

A risk code alone is never a fact identity. Date findings merge only when both
date values and an unambiguous source-clause scope agree. Ambiguous unanchored
findings retain their original identities rather than merging distinct events.
"""
from __future__ import annotations

import copy
from typing import Sequence

from . import text as T
from .models import Contract, RiskCandidate, SpanRef


def _iso_dates(value: str) -> set[str]:
    return {v.isoformat() for v in T.all_dates(value or "")}


def _source_spans(contract: Contract, clause_ids: list[str], values: set[str]) -> list[SpanRef]:
    spans = []
    clause_map = contract.clause_map()
    block_map = contract.document.block_map()
    for cid in clause_ids:
        clause = clause_map[cid]
        blocks = [(bid, block_map[bid].text, block_map[bid].page)
                  for bid in clause.block_ids if bid in block_map]
        if not blocks:
            blocks = [(clause.block_ids[0] if clause.block_ids else cid, clause.text, clause.page_start)]
        for bid, text, page in blocks:
            for pattern in (T._DATE_RE, T._CN_DATE_RE):
                for match in pattern.finditer(text):
                    parsed = T.parse_date(match.group(0))
                    if parsed and parsed.isoformat() in values:
                        spans.append(SpanRef(bid, match.group(0), match.start(), match.end(),
                                             page=page, clause_id=cid))
    # An entity's span may locate a date in a parsed block more precisely than
    # clause offsets; source-text lookup above is preferred when available.
    if not spans:
        for element in contract.elements.dates:
            if element.clause_id in clause_ids and _iso_dates(element.normalized or element.value) & values and element.span:
                span = copy.deepcopy(element.span)
                span.clause_id = element.clause_id
                spans.append(span)
    seen = set()
    unique = []
    for span in spans:
        key = (span.block_id, span.char_start, span.char_end)
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _date_identity(candidate: RiskCandidate, contract: Contract) -> str:
    values = _iso_dates(candidate.message)
    if len(values) != 2:
        values |= {v for span in candidate.spans for v in _iso_dates(span.quote)}
    if len(values) != 2:
        return ""
    clause_map = contract.clause_map()
    existing = [cid for cid in candidate.clause_ids if cid in clause_map]
    locations = {value: set() for value in values}
    for element in contract.elements.dates:
        if element.clause_id in clause_map:
            for value in _iso_dates(element.normalized or element.value) & values:
                locations[value].add(element.clause_id)
    for cid, clause in clause_map.items():
        for value in _iso_dates(clause.text) & values:
            locations[value].add(cid)
    common = set.intersection(*(locations[value] for value in values))
    if existing:
        # Existing scope must contain both source dates; otherwise do not invent
        # a new anchor that could collapse unrelated milestones.
        if not all(locations[value].intersection(existing) for value in values):
            return ""
        selected = existing
    elif len(common) == 1:
        selected = list(common)
    elif not common and all(len(locations[value]) == 1 for value in values):
        selected = list(set.union(*(locations[value] for value in values)))
    else:
        return ""
    order = {clause.clause_id: i for i, clause in enumerate(contract.document.clauses)}
    selected.sort(key=lambda cid: order[cid])
    candidate.clause_ids = selected
    for span in _source_spans(contract, selected, values):
        if not any((s.block_id, s.char_start, s.char_end) == (span.block_id, span.char_start, span.char_end) for s in candidate.spans):
            candidate.spans.append(span)
    identity = "date_reversed:" + ",".join(sorted(values)) + ":" + ",".join(selected)
    candidate.extra["fact_key"] = identity
    candidate.extra["date_values"] = sorted(values)
    return identity


def normalize_fact_candidates(candidates: Sequence[RiskCandidate], contract: Contract) -> list[RiskCandidate]:
    output: list[RiskCandidate] = []
    by_fact: dict[str, RiskCandidate] = {}
    for candidate in candidates:
        fact = _date_identity(candidate, contract) if candidate.code == "date_reversed" else ""
        if not fact or fact not in by_fact:
            output.append(candidate)
            if fact:
                by_fact[fact] = candidate
            continue
        kept = by_fact[fact]
        aliases = set(kept.extra.get("alias_candidate_ids", [])) | set(candidate.extra.get("alias_candidate_ids", []))
        aliases.update((kept.candidate_id, candidate.candidate_id))
        kept.merge_from(candidate)
        kept.extra["alias_candidate_ids"] = sorted(aliases)
    clauses = contract.clause_map()
    block_clauses: dict[str, list[str]] = {}
    for cid, clause in clauses.items():
        for block_id in clause.block_ids:
            block_clauses.setdefault(block_id, []).append(cid)
    for candidate in output:
        sources = []
        source_ids = list(candidate.clause_ids)
        for span in [*candidate.spans, *(evidence.span for evidence in candidate.evidence if evidence.span)]:
            if span.clause_id in clauses:
                source_ids.append(span.clause_id)
            source_ids.extend(block_clauses.get(span.block_id, []))
        for cid in dict.fromkeys(source_ids):
            clause = clauses.get(cid)
            metadata = clause.metadata if clause else {}
            if metadata.get("extraction_quality", {}).get("status") == "machine_ocr_unverified":
                sources.append({"clause_id": cid, **{key: metadata.get(key) for key in
                    ("source_id", "source_title", "source_page_start", "source_page_end", "original_sha256")}})
        if sources:
            candidate.extra.update(manual_required=True, source_text_requires_review=True,
                                   source_review_sources=sources, automatic_confirmation_allowed=False)
            flags = candidate.extra.setdefault("review_flags", [])
            if "contract_source_ocr_not_human_checked" not in flags:
                flags.append("contract_source_ocr_not_human_checked")
            if not any(ev.kind == "source_text_quality" for ev in candidate.evidence):
                candidate.evidence_append("source_text_quality",
                    "候选依据来自尚未逐字核对的扫描件OCR；日期、金额及表格关系需对照原件，不能据此确认合同事实矛盾。",
                    0.0, "source_quality_gate")
                candidate.evidence[-1].extra.update(manual_required=True, source_review_sources=sources)
    return output
