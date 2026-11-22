"""Model-facing projection of transport diagnostics; original evidence stays put."""
from __future__ import annotations
import copy
import hashlib
import json


def select_feedback_windows(items, windows):
    """Deduplicate only when one existing window contains all matching sources.

    This is a transport policy, not a new source window or a legal completion
    receipt. Offsets and literal text must both match; partial windows without
    a complete superset retain all their original calls.
    """
    selected, audit = {}, []
    for item in items:
        item_id = item['feedback_item_id']
        clause_ids = set(item.get('clause_ids', []))
        matching = [w for w in windows if any(s.clause_id in clause_ids for s in w.segments)]
        def contains(outer, inner):
            return all(any(a.clause_id == b.clause_id and a.start <= b.start and a.end >= b.end
                           and a.text[b.start-a.start:b.end-a.start] == b.text
                           for a in outer.segments) for b in inner.segments)
        supersets = [w for w in matching if all(contains(w, other) for other in matching)]
        chosen = ([min(supersets, key=lambda w: (sum(len(s.text) for s in w.segments),
                                               w.kind != 'text', w.window_id))]
                  if supersets else matching)
        selected[item_id] = {w.window_id for w in chosen}
        audit.append({'feedback_item_id': item_id,
            'matching_window_ids': [w.window_id for w in matching],
            'selected_window_ids': [w.window_id for w in chosen],
            'omitted_duplicate_context_window_ids': [w.window_id for w in matching if w not in chosen],
            'source_context_fully_preserved': bool(matching),
            'omitted_windows_executed': False,
            'meaning': 'Text containment avoids repeated transport; it does not acknowledge omitted cross-window checks.'})
    return selected, audit


def project_feedback(item):
    original = copy.deepcopy(item)
    projected = copy.deepcopy(item)
    audit = projected.get("window_audit")
    receipt = None
    if isinstance(audit, dict) and len(json.dumps(audit, ensure_ascii=False)) > 512:
        digest = hashlib.sha256(json.dumps(audit, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        # These are semantic/coverage flags, not the voluminous per-window
        # tensor/attention/timing receipts. No source quote is shortened here.
        keep = ("schema_version", "policy", "input_valid", "decision_scope", "semantic_pending", "manual_required",
                "window_disagreement", "local_labels", "token_coverage_complete", "pair_coverage_complete",
                "planned_pairs", "executed_pairs", "max_pairs", "clause_source_sha256", "article_source_sha256")
        compact = {key:copy.deepcopy(audit[key]) for key in keep if key in audit}
        compact.update(full_window_audit_sha256=digest, projection="coverage_and_semantic_flags_only",
                       full_window_receipts_retained_outside_model_prompt=True)
        projected["window_audit"] = compact
        receipt = {"version":"feedback-diagnostics-projection-v1", "feedback_item_id":item.get("feedback_item_id"),
                   "full_window_audit_sha256":digest, "original_feedback":original,
                   "source_quotes_changed_by_projection":False,
                   "meaning":"Only transport diagnostics are projected; source/semantic pending remains authoritative."}
    return projected,receipt
