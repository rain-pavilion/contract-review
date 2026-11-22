"""Token-exact planning and conservative aggregation for local NLI windows.

This module deliberately has no torch/tokenizers dependency. Full token coverage
is a transport property; agreement between local predictions is not a proof of
the semantic relation between the two complete source documents.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


LABELS = ("entail", "contradict", "neutral")


@dataclass(frozen=True)
class TokenWindow:
    window_id: str
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    char_start: int | None = None
    char_end: int | None = None

    def audit(self) -> dict[str, Any]:
        raw = json.dumps(self.token_ids, separators=(",", ":")).encode("ascii")
        return {"window_id": self.window_id, "token_start": self.token_start,
                "token_end": self.token_end, "token_count": len(self.token_ids),
                "char_start": self.char_start, "char_end": self.char_end,
                "offsets_available": self.char_start is not None,
                "token_ids_sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True)
class WindowPair:
    pair_id: str
    clause: TokenWindow
    article: TokenWindow


@dataclass(frozen=True)
class WindowPlan:
    clause_windows: tuple[TokenWindow, ...]
    article_windows: tuple[TokenWindow, ...]
    scheduled_pairs: tuple[WindowPair, ...]
    clause_token_count: int
    article_token_count: int
    max_pairs: int

    @property
    def planned_pair_count(self) -> int:
        return len(self.clause_windows) * len(self.article_windows)


def build_windows(token_ids: Sequence[int], capacity: int, *, prefix: str,
                  overlap: int = 32,
                  offsets: Sequence[Sequence[int]] | None = None) -> tuple[TokenWindow, ...]:
    """Slice original token IDs, including the last token, without re-tokenizing."""
    if capacity < 1 or overlap < 0 or overlap >= capacity:
        raise ValueError("invalid_window_capacity_or_overlap")
    ids = tuple(int(value) for value in token_ids)
    if offsets is not None and len(offsets) != len(ids):
        raise ValueError("token_offset_count_mismatch")
    windows: list[TokenWindow] = []
    start = 0
    while True:
        end = min(len(ids), start + capacity)
        char_start = char_end = None
        if offsets is not None:
            char_start = int(offsets[start][0]) if end > start else 0
            char_end = int(offsets[end - 1][1]) if end > start else 0
            if char_start < 0 or char_end < char_start:
                raise ValueError("invalid_token_offsets")
        windows.append(TokenWindow(f"{prefix}{len(windows):04d}", start, end,
                                   ids[start:end], char_start, char_end))
        if end >= len(ids):
            break
        start = end - overlap
    return tuple(windows)


def _edge_first_indices(size: int) -> Iterable[int]:
    """Include the tail early if the explicit execution budget is insufficient."""
    low, high = 0, size - 1
    while low <= high:
        yield low
        if low != high:
            yield high
        low, high = low + 1, high - 1


def make_plan(clause_ids: Sequence[int], article_ids: Sequence[int], *,
              clause_capacity: int, article_capacity: int, max_pairs: int = 64,
              overlap: int = 32,
              clause_offsets: Sequence[Sequence[int]] | None = None,
              article_offsets: Sequence[Sequence[int]] | None = None) -> WindowPlan:
    if max_pairs < 1:
        raise ValueError("max_window_pairs_must_be_positive")
    clauses = build_windows(clause_ids, clause_capacity, prefix="c", overlap=min(overlap, clause_capacity - 1),
                            offsets=clause_offsets)
    articles = build_windows(article_ids, article_capacity, prefix="a", overlap=min(overlap, article_capacity - 1),
                             offsets=article_offsets)
    chosen: list[WindowPair] = []
    seen: set[tuple[int, int]] = set()

    def add(ci: int, ai: int) -> bool:
        if (ci, ai) not in seen:
            seen.add((ci, ai))
            chosen.append(WindowPair(f"{clauses[ci].window_id}:{articles[ai].window_id}",
                                     clauses[ci], articles[ai]))
        return len(chosen) >= max_pairs

    # Guarantee both tails are attempted when the budget permits two pairs,
    # including unequal window counts where modulo diagonals miss one tail.
    if add(0, 0):
        return WindowPlan(clauses, articles, tuple(chosen), len(clause_ids), len(article_ids), max_pairs)
    if add(len(clauses) - 1, len(articles) - 1):
        return WindowPlan(clauses, articles, tuple(chosen), len(clause_ids), len(article_ids), max_pairs)
    # Cover source windows before filling the remaining Cartesian comparisons.
    for index in _edge_first_indices(max(len(clauses), len(articles))):
        if add(index % len(clauses), index % len(articles)):
            return WindowPlan(clauses, articles, tuple(chosen), len(clause_ids), len(article_ids), max_pairs)
    for ci in _edge_first_indices(len(clauses)):
        for ai in _edge_first_indices(len(articles)):
            if add(ci, ai):
                return WindowPlan(clauses, articles, tuple(chosen), len(clause_ids), len(article_ids), max_pairs)
    return WindowPlan(clauses, articles, tuple(chosen), len(clause_ids), len(article_ids), max_pairs)


def _coverage(windows: Iterable[TokenWindow], total: int) -> dict[str, Any]:
    spans = sorted(set((window.token_start, window.token_end) for window in windows))
    merged: list[list[int]] = []
    for start, end in spans:
        if start < 0 or end < start or end > total:
            raise ValueError("window_range_outside_source")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    missing: list[list[int]] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            missing.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < total:
        missing.append([cursor, total])
    return {"total_tokens": total, "covered_tokens": sum(end - start for start, end in merged),
            "covered_ranges": merged, "missing_ranges": missing,
            "tail_covered": total == 0 or bool(merged and merged[-1][1] == total),
            "complete": not missing}


def summarize(plan: WindowPlan, predictions: Sequence[dict[str, Any]], *,
              failures: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Retain local predictions and abstain from a global multi-window claim.

    The representative prediction is an *actual* local result (contradiction
    first, then neutral, then entailment). Its score is never a global score.
    Callers must honor ``semantic_pending`` and ``decision_scope``.
    """
    input_valid = plan.clause_token_count > 0 and plan.article_token_count > 0
    invalid_reasons = (["empty_clause_tokens"] if plan.clause_token_count == 0 else []) + (
        ["empty_article_tokens"] if plan.article_token_count == 0 else [])
    if not input_valid and predictions:
        raise ValueError("empty_token_input_cannot_have_model_predictions")
    by_id = {pair.pair_id: pair for pair in plan.scheduled_pairs}
    executed: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        pair_id = str(prediction.get("pair_id", ""))
        if pair_id not in by_id or pair_id in executed:
            raise ValueError("unknown_or_duplicate_window_prediction")
        if prediction.get("label") not in LABELS:
            raise ValueError("invalid_window_prediction_label")
        executed[pair_id] = dict(prediction)
    for failure in failures:
        if failure.get("pair_id") not in by_id or failure.get("pair_id") in executed:
            raise ValueError("invalid_failed_window_pair")
    actual_pairs = [by_id[pair_id] for pair_id in executed]
    clause_coverage = _coverage((pair.clause for pair in actual_pairs), plan.clause_token_count)
    article_coverage = _coverage((pair.article for pair in actual_pairs), plan.article_token_count)
    token_complete = input_valid and clause_coverage["complete"] and article_coverage["complete"]
    pair_complete = len(executed) == plan.planned_pair_count
    multi_window = plan.planned_pair_count > 1
    labels = sorted({str(item["label"]) for item in executed.values()})
    disagreement = len(labels) > 1
    reasons = ["invalid_empty_token_input"] if not input_valid else []
    if len(plan.scheduled_pairs) < plan.planned_pair_count:
        reasons.append("window_pair_budget_exhausted")
    if failures:
        reasons.append("window_prediction_failed")
    if not token_complete:
        reasons.append("token_coverage_incomplete")
    if not pair_complete:
        reasons.append("window_pair_coverage_incomplete")
    if disagreement:
        reasons.append("window_label_disagreement")
    if multi_window:
        reasons.append("multi_window_relation_not_validated")
    pending = not input_valid or multi_window or not pair_complete or not token_complete
    priority = {"contradict": 2, "neutral": 1, "entail": 0}
    selected = max(executed.values(), key=lambda item: (priority[item["label"]], float(item["score"])), default=None)
    # Avoid constructing an unbounded Cartesian list for pathological inputs.
    unexecuted = None
    if plan.planned_pair_count <= 4096:
        unexecuted = [f"{c.window_id}:{a.window_id}" for c in plan.clause_windows
                      for a in plan.article_windows if f"{c.window_id}:{a.window_id}" not in executed]
    return {
        "schema_version": 1, "policy": "token_exact_cartesian_local_nli_v1",
        "input_valid": input_valid, "invalid_input_reasons": invalid_reasons,
        "decision_scope": "invalid_input" if not input_valid else "local_window_only" if pending else "whole_input",
        "semantic_pending": pending, "manual_required": pending,
        "semantic_pending_reasons": reasons, "window_disagreement": disagreement,
        "local_labels": labels, "selected_pair_id": selected["pair_id"] if selected else None,
        "selected_prediction": selected,
        "clause_windows": [window.audit() for window in plan.clause_windows],
        "article_windows": [window.audit() for window in plan.article_windows],
        "clause_coverage": clause_coverage, "article_coverage": article_coverage,
        "token_coverage_complete": token_complete, "pair_coverage_complete": pair_complete,
        "planned_pairs": plan.planned_pair_count, "max_pairs": plan.max_pairs,
        "scheduled_pair_ids": list(by_id), "executed_pairs": len(executed),
        "executed_pair_ids": list(executed), "unexecuted_pair_ids": unexecuted,
        "unexecuted_pair_id_rule": None if unexecuted is not None else
            "all clause_window_id:article_window_id combinations except executed_pair_ids",
        "failed_pairs": [dict(failure) for failure in failures],
        "window_predictions": list(executed.values()),
    }
