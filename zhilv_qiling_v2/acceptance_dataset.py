"""Strict human-review gates and a project-owned contract/statute ranking protocol.

Software can verify declarations, hashes and separation, not a person's identity
or the legal correctness of a label. No function creates a human answer.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

SCHEMA = "zlql-human-gold-v1"
RANKING_PROTOCOL = "zlql-contract-statute-ranking-v1"
ROOT = Path(__file__).resolve().parents[1]


class DatasetValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8") if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)


def _legacy():
    # Reuse the established schema/leakage checker without modifying its source.
    tools_path = str(ROOT / "tools")
    before = list(sys.path)
    try:
        sys.path.insert(0, tools_path)
        return importlib.import_module("acceptance_annotations")
    finally:
        sys.path[:] = before


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp_required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    if parsed > datetime.now(timezone.utc):
        raise ValueError("future_manual_signature")
    return parsed


def _registry(registry: dict) -> dict[str, dict]:
    people = registry.get("reviewers", []) if isinstance(registry, dict) else []
    by_id = {row.get("reviewer_id"): row for row in people if isinstance(row, dict)}
    if not people or len(by_id) != len(people) or not all(isinstance(key, str) and key.strip() for key in by_id):
        raise DatasetValidationError(["nonempty_unique_operator_registered_reviewers_required"])
    for person in by_id.values():
        if person.get("human_declared") is not True or person.get("role") not in ("reviewer", "adjudicator"):
            raise DatasetValidationError(["explicit_human_declaration_and_role_required"])
        if not person.get("registered_by"):
            raise DatasetValidationError(["operator_registration_attestation_required"])
        _time(person.get("registered_at"))
    return by_id


def _annotations(annotations: Any, text: str) -> list[str]:
    errors = []
    if not isinstance(annotations, list):
        return ["annotation_list_required"]
    seen = set()
    for ann in annotations:
        if not isinstance(ann, dict):
            errors.append("annotation_object_required")
            continue
        if ann.get("type") not in ("entity", "risk", "law_reference") or not isinstance(ann.get("label"), str) or not ann["label"].strip():
            errors.append("annotation_type_and_label_required")
        start, end = ann.get("start"), ann.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            errors.append("annotation_source_span_invalid")
        elif ann.get("evidence") != text[start:end]:
            errors.append("annotation_source_evidence_mismatch")
        elif not ann["evidence"].strip():
            errors.append("annotation_evidence_cannot_be_whitespace")
        if "relevance" in ann and (type(ann["relevance"]) is not int or not 0 <= ann["relevance"] <= 3):
            errors.append("ranking_relevance_must_be_integer_0_to_3")
        key = digest(ann)
        if key in seen:
            errors.append("duplicate_annotation")
        seen.add(key)
    return errors


def _canonical_annotations(annotations: list[dict]) -> list[dict]:
    return sorted(deepcopy(annotations), key=lambda row: (row.get("start", -1), row.get("end", -1), digest(row)))


def _source_keys(row):
    source = row.get("source") or {}
    keys = set()
    for namespace, fields in (("group", ("group_id",)), ("source", ("source_id", "document_id", "contract_id")),
                              ("template", ("template_family", "template_group", "template_id"))):
        for field in fields:
            value = source.get(field) or row.get(field)
            if isinstance(value, str) and value.strip():
                keys.add((namespace, value.strip().casefold()))
    if source.get("origin") and source.get("path"):
        keys.add(("origin_path", str(source["origin"]).strip()+"|"+str(source["path"]).replace("\\", "/")))
    return keys


def _near_duplicate(left, right):
    a,b = _legacy().normalized(left), _legacy().normalized(right)
    if min(len(a),len(b)) < 80:
        return False
    aa={a[i:i+5] for i in range(len(a)-4)}
    bb={b[i:i+5] for i in range(len(b)-4)}
    return bool(aa and bb) and len(aa&bb)/len(aa|bb) >= .90


def _base(rows: list[dict]) -> dict[str, dict]:
    if not rows:
        raise DatasetValidationError(["empty_candidate_dataset"])
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise DatasetValidationError(["nonempty_nonwhitespace_source_text_required"])
    report = _legacy().validate(rows)
    errors = list(report["errors"])
    if any(row.get("status") != "unlabeled" or row.get("annotations") or row.get("reviews") for row in rows):
        errors.append("original_readonly_unlabeled_candidates_required")
    if errors:
        raise DatasetValidationError(errors)
    return {row["id"]: row for row in rows}


def import_independent_reviews(base_rows: list[dict], review_a: list[dict], review_b: list[dict],
                               registry: dict) -> dict:
    """Import two completed blind human files; never auto-adjudicate agreement."""
    base = _base(base_rows)
    people = _registry(registry)
    errors, imported = [], {}
    for slot, rows in (("A", review_a), ("B", review_b)):
        ids = [row.get("id") for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(base):
            errors.append(f"{slot}:exact_unique_candidate_ids_required")
            continue
        for row in rows:
            bid = row["id"]
            original = base[bid]
            prefix = f"{slot}:{bid}:"
            if any(row.get(key) != original.get(key) for key in ("schema_version", "text", "text_sha256", "source", "synthetic_candidate", "split")):
                errors.append(prefix + "source_or_predeclared_split_changed")
            assignment = row.get("blind_assignment") or {}
            person = people.get(assignment.get("reviewer_id"), {})
            declaration = row.get("review_declaration") or {}
            if assignment.get("slot") != slot or person.get("role") != "reviewer":
                errors.append(prefix + "registered_reviewer_and_matching_slot_required")
            if (declaration.get("method") != "manual" or declaration.get("review_complete") is not True
                    or declaration.get("used_system_predictions") is not False or declaration.get("saw_other_review") is not False):
                errors.append(prefix + "completed_blind_manual_declaration_required")
            if row.get("status") != "in_review" or row.get("reviews"):
                errors.append(prefix + "single_independent_review_state_required")
            errors.extend(prefix + error for error in _annotations(row.get("annotations"), original["text"]))
            if row.get("annotations") == [] and declaration.get("empty_annotation_confirmed") is not True:
                errors.append(prefix + "explicit_intentional_empty_annotation_confirmation_required")
            try:
                started, ended = _time(assignment.get("started_at")), _time(assignment.get("completed_at"))
                if ended < started:
                    errors.append(prefix + "review_completed_before_started")
            except (TypeError, ValueError):
                errors.append(prefix + "valid_completed_review_timestamps_required")
            imported[(slot, bid)] = {"reviewer_id": assignment.get("reviewer_id"),
                "reviewed_at": assignment.get("completed_at"), "started_at": assignment.get("started_at"),
                "method": "manual", "used_system_predictions": False, "saw_other_review": False,
                "decision": "independent_complete", "slot": slot,
                "empty_annotation_confirmed": declaration.get("empty_annotation_confirmed") is True,
                "annotations": deepcopy(row.get("annotations", [])),
                "annotations_sha256": digest(row.get("annotations", [])), "input_record_sha256": digest(row)}
    if errors:
        raise DatasetValidationError(errors)
    merged, conflicts = [], []
    for bid, original in base.items():
        a, b = imported[("A", bid)], imported[("B", bid)]
        if a["reviewer_id"] == b["reviewer_id"]:
            errors.append(bid + ":two_distinct_reviewers_required")
        agree = _canonical_annotations(a["annotations"]) == _canonical_annotations(b["annotations"])
        row = deepcopy(original)
        row.update(status="in_review", reviews=[a, b], annotations=[],
                   review_audit={"schema_version": "zlql-independent-import-v1", "registry_sha256": digest(registry),
                                 "immutable_source_sha256": digest({key: original.get(key) for key in ("id", "text", "text_sha256", "source", "split", "synthetic_candidate")}),
                                 "two_reviews_agree": agree, "adjudication_required": True})
        merged.append(row)
        conflicts.append({"id": bid, "text_sha256": original["text_sha256"], "annotations_agree": agree,
                          "review_A_sha256": a["annotations_sha256"], "review_B_sha256": b["annotations_sha256"],
                          "status": "awaiting_manual_adjudication"})
    if errors:
        raise DatasetValidationError(errors)
    return {"rows": merged, "conflicts": conflicts, "gold_ready": False, "human_answers_generated": False,
            "identity_assurance": "operator declarations only; software cannot establish human identity or legal correctness"}


def adjudicate(imported_rows: list[dict], decisions: list[dict], registry: dict) -> list[dict]:
    people = _registry(registry)
    by_id = {row.get("id"): row for row in decisions}
    if len(by_id) != len(decisions) or set(by_id) != {row["id"] for row in imported_rows}:
        raise DatasetValidationError(["exact_unique_manual_adjudication_ids_required"])
    result, errors = [], []
    for source in imported_rows:
        row = deepcopy(source)
        decision = by_id[row["id"]]
        prefix = row["id"] + ":"
        reviews = row.get("reviews", [])
        person = people.get(decision.get("adjudicator_id"), {})
        if len(reviews) != 2 or person.get("role") != "adjudicator" or decision.get("adjudicator_id") in {r.get("reviewer_id") for r in reviews}:
            errors.append(prefix + "third_registered_manual_adjudicator_required")
        if (decision.get("text_sha256") != row.get("text_sha256") or decision.get("method") != "manual"
                or decision.get("used_system_predictions") is not False or decision.get("reviewed_both_independent_annotations") is not True
                or not isinstance(decision.get("rationale"), str) or not decision["rationale"].strip()):
            errors.append(prefix + "manual_adjudication_source_declarations_and_rationale_required")
        errors.extend(prefix + error for error in _annotations(decision.get("annotations"), row["text"]))
        if decision.get("annotations") == [] and decision.get("empty_annotation_confirmed") is not True:
            errors.append(prefix + "intentional_empty_adjudication_confirmation_required")
        try:
            ended = _time(decision.get("adjudicated_at"))
            if any(ended < _time(review.get("reviewed_at")) for review in reviews):
                errors.append(prefix + "adjudication_precedes_independent_reviews")
        except (TypeError, ValueError):
            errors.append(prefix + "valid_adjudication_timestamp_required")
        row.update(status="adjudicated", annotations=deepcopy(decision.get("annotations", [])),
                   adjudication={**deepcopy(decision), "annotations_sha256": digest(decision.get("annotations", []))})
        result.append(row)
    if errors:
        raise DatasetValidationError(errors)
    return result


def assign_group_splits(rows: list[dict], seed: int = 20261118) -> list[dict]:
    """Assign before blind distribution; never choose the split from outcomes."""
    if any(row.get("status") != "unlabeled" or row.get("annotations") or row.get("reviews") for row in rows):
        raise DatasetValidationError(["split_assignment_requires_unlabeled_sources"])
    # Union declared source/template families before assigning any split.
    owner={}
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for index,row in enumerate(rows):
        for key in _source_keys(row):
            if key in owner:
                parent[find(index)]=find(owner[key])
            owner[key]=index
        for previous,other in enumerate(rows[:index]):
            if (_legacy().normalized(row.get("text", "")) == _legacy().normalized(other.get("text", ""))
                    or _near_duplicate(row.get("text", ""),other.get("text", ""))):
                parent[find(index)]=find(previous)
    row_groups={index:str(find(index)) for index in range(len(rows))}
    groups = sorted(set(row_groups.values()))
    if len(groups) < 3:
        raise DatasetValidationError(["at_least_three_source_groups_required"])
    random.Random(seed).shuffle(groups)
    count = max(1, round(len(groups) * .2))
    assignment = {group: "test" if i < count else "dev" if i < 2 * count else "train" for i, group in enumerate(groups)}
    result = deepcopy(rows)
    for index,row in enumerate(result):
        row["split"] = assignment[row_groups[index]]
    return result


def validate_gold(rows: list[dict], training_inventory: Path | str | None, registry: dict | None) -> dict:
    errors = []
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        return {"status": "rejected", "gold_ready": False, "records": len(rows),
                "errors": ["nonempty_nonwhitespace_source_text_required"], "human_answers_generated": False,
                "identity_assurance": "operator-declared people; no automated proof of identity or legal correctness"}
    try:
        people = _registry(registry or {})
    except (DatasetValidationError, TypeError, ValueError) as exc:
        people = {}
        errors.extend(getattr(exc, "errors", ["invalid_operator_registry"]))
    if not training_inventory:
        errors.append("complete_training_development_inventory_required")
    legacy = _legacy().validate(rows, training_inventory, require_gold=True)
    errors.extend(legacy["errors"])
    if not rows:
        errors.append("nonempty_gold_required")
    inventory = read_jsonl(training_inventory) if training_inventory else []
    used_groups = {r.get("source", {}).get("group_id") for r in inventory}
    used_texts = {digest(_legacy().normalized(r.get("text", ""))) for r in inventory}
    used_keys=set().union(*(_source_keys(row) for row in inventory)) if inventory else set()
    split_keys={}
    seen_split_texts=[]
    for row in rows:
        prefix = str(row.get("id", "?")) + ":"
        text = row.get("text", "")
        if (row.get("synthetic_candidate") is not False or row.get("source", {}).get("origin_kind") not in
                ("public_real_contract", "permissioned_real_contract")):
            errors.append(prefix + "formal_independent_gold_requires_explicit_real_contract_provenance")
        errors.extend(prefix + error for error in _annotations(row.get("annotations"), text))
        if row.get("split") in ("dev", "test") and (row.get("source", {}).get("group_id") in used_groups or digest(_legacy().normalized(text)) in used_texts):
            errors.append(prefix + "evaluation_source_overlaps_training_development_inventory")
        keys=_source_keys(row)
        if row.get("split") in ("dev","test"):
            if not any(kind=="source" for kind,value in keys) or not any(kind=="template" for kind,value in keys):
                errors.append(prefix+"evaluation_source_id_and_template_family_required")
            if keys & used_keys:
                errors.append(prefix+"evaluation_source_or_template_overlaps_training_inventory")
            if any(_near_duplicate(text,other.get("text","")) for other in inventory):
                errors.append(prefix+"near_duplicate_training_text_requires_manual_family_review")
        for key in keys:
            if key in split_keys and split_keys[key]!=row.get("split"):
                errors.append(prefix+"source_or_template_family_crosses_dataset_splits")
            split_keys[key]=row.get("split")
        if any(split != row.get("split") and _near_duplicate(text,other_text)
               for split,other_text in seen_split_texts):
            errors.append(prefix+"near_duplicate_text_crosses_dataset_splits")
        seen_split_texts.append((row.get("split"),text))
        reviews = row.get("reviews", [])
        if len(reviews) != 2 or {r.get("slot") for r in reviews} != {"A", "B"}:
            errors.append(prefix + "exactly_two_independent_review_slots_required")
        for review in reviews:
            if review.get("decision") != "independent_complete":
                errors.append(prefix + "independent_review_completion_decision_required")
            if people.get(review.get("reviewer_id"), {}).get("role") != "reviewer" or review.get("saw_other_review") is not False:
                errors.append(prefix + "review_identity_or_independence_declaration_invalid")
            errors.extend(prefix + "review:" + error for error in _annotations(review.get("annotations"), text))
            if review.get("annotations") == [] and review.get("empty_annotation_confirmed") is not True:
                errors.append(prefix + "empty_review_not_explicitly_confirmed")
            try:
                if _time(review.get("reviewed_at")) < _time(review.get("started_at")):
                    errors.append(prefix + "review_timestamps_out_of_order")
            except (TypeError, ValueError):
                errors.append(prefix + "valid_manual_review_timestamps_required")
        decision = row.get("adjudication") or {}
        if decision.get("text_sha256") != row.get("text_sha256"):
            errors.append(prefix + "adjudication_source_text_hash_mismatch")
        if decision.get("annotations") == [] and decision.get("empty_annotation_confirmed") is not True:
            errors.append(prefix + "empty_adjudication_not_explicitly_confirmed")
        if _canonical_annotations(decision.get("annotations",[])) != _canonical_annotations(row.get("annotations",[])):
            errors.append(prefix+"retained_manual_adjudication_differs_from_gold")
        if decision.get("annotations_sha256") != digest(decision.get("annotations",[])):
            errors.append(prefix+"retained_adjudication_annotation_hash_mismatch")
        if (people.get(decision.get("adjudicator_id"), {}).get("role") != "adjudicator"
                or decision.get("adjudicator_id") in {r.get("reviewer_id") for r in reviews}
                or decision.get("method") != "manual" or decision.get("used_system_predictions") is not False
                or decision.get("reviewed_both_independent_annotations") is not True):
            errors.append(prefix + "independent_manual_adjudication_required")
        if (row.get("review_audit") or {}).get("registry_sha256") != digest(registry or {}):
            errors.append(prefix + "operator_registry_changed_after_import")
        if (row.get("review_audit") or {}).get("immutable_source_sha256") != digest({key: row.get(key) for key in ("id", "text", "text_sha256", "source", "split", "synthetic_candidate")}):
            errors.append(prefix + "source_or_split_changed_after_blind_review_import")
        try:
            if any(_time(decision.get("adjudicated_at")) < _time(review.get("reviewed_at")) for review in reviews):
                errors.append(prefix + "adjudication_precedes_reviews")
        except (TypeError, ValueError):
            errors.append(prefix + "valid_manual_adjudication_timestamp_required")
    return {"status": "passed" if not errors else "rejected", "gold_ready": bool(rows) and not errors,
            "records": len(rows), "errors": sorted(set(errors)), "human_answers_generated": False,
            "identity_assurance": "operator-declared people; no automated proof of identity or legal correctness"}


def freeze_gold(rows: list[dict], training_inventory: Path | str, registry: dict, output: Path) -> dict:
    report = validate_gold(rows, training_inventory, registry)
    if not report["gold_ready"]:
        raise DatasetValidationError(report["errors"])
    if output.exists():
        raise DatasetValidationError(["immutable_gold_output_already_exists"])
    output.mkdir(parents=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    # Keep the signed byte stream identical on Windows and Linux (no CRLF rewrite).
    (output / "gold.jsonl").write_bytes(raw.encode("utf-8"))
    real = all(not row.get("synthetic_candidate") and row.get("source", {}).get("origin_kind") in
               ("public_real_contract", "permissioned_real_contract") for row in rows)
    manifest = {**report, "schema_version": SCHEMA, "immutable": True, "gold_sha256": digest(raw),
                "registry_sha256": digest(registry), "training_inventory_sha256": digest(Path(training_inventory).read_bytes()),
                "independently_sourced_real_contracts": real,
                "synthetic_candidates_present": any(row.get("synthetic_candidate") for row in rows),
                "frozen_at": datetime.now(timezone.utc).isoformat()}
    write_new_json(output / "manifest.json", manifest)
    return manifest


def ranking_metrics(gold: list[dict], predictions: list[dict], corpus: list[dict], *, ks=(1, 3, 5),
                    judgment_scope: str = "exhaustive_corpus", fixture_only: bool = False) -> dict:
    """Standard ranking metrics for a declared project protocol, never ProvBench."""
    if judgment_scope not in ("exhaustive_corpus", "adjudicated_pool") or not gold or not corpus:
        raise DatasetValidationError(["valid_nonempty_project_ranking_protocol_required"])
    corpus_ids = [row["provision_id"] for row in corpus]
    gids, pids = [row["id"] for row in gold], [row["id"] for row in predictions]
    if len(set(corpus_ids)) != len(corpus_ids) or len(set(gids)) != len(gids) or len(set(pids)) != len(pids) or set(gids) != set(pids):
        raise DatasetValidationError(["unique_exact_ranking_ids_required"])
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise DatasetValidationError(["positive_integer_ranking_cutoffs_required"])
    if len(set(ks)) != len(ks):
        raise DatasetValidationError(["unique_ranking_cutoffs_required"])
    by_id = {row["id"]: row for row in predictions}
    totals = {f"{name}@{k}": 0. for k in ks for name in ("Recall", "Precision", "nDCG")}
    totals["MRR"] = totals["MAP"] = 0.
    for row in gold:
        judgments = row.get("judgments", [])
        qrels = {item["provision_id"]: item["relevance"] for item in judgments}
        rank = by_id[row["id"]].get("ranked_provision_ids", [])
        if (len(qrels) != len(judgments) or not set(qrels) <= set(corpus_ids)
                or any(type(v) is not int or not 0 <= v <= 3 for v in qrels.values())
                or len(rank) != len(set(rank)) or not set(rank) <= set(corpus_ids)
                or len(rank) < min(max(ks), len(corpus_ids))):
            raise DatasetValidationError([row["id"] + ":invalid_judgments_or_ranking"])
        if judgment_scope == "exhaustive_corpus" and set(qrels) != set(corpus_ids):
            raise DatasetValidationError([row["id"] + ":exhaustive_manual_corpus_judgments_required"])
        if not set(rank) <= set(qrels):
            raise DatasetValidationError([row["id"] + ":unjudged_predictions_cannot_be_assumed_irrelevant"])
        if set(rank) != set(qrels):
            raise DatasetValidationError([row["id"] + ":full_declared_judgment_universe_ranking_required_for_MRR_MAP"])
        relevant = {pid for pid, grade in qrels.items() if grade > 0}
        if not relevant:
            raise DatasetValidationError([row["id"] + ":query_has_no_adjudicated_relevant_statute"])
        hits = [1 if pid in relevant else 0 for pid in rank]
        first = next((i for i, hit in enumerate(hits, 1) if hit), None)
        totals["MRR"] += 1 / first if first else 0.
        totals["MAP"] += sum(sum(hits[:i]) / i for i, hit in enumerate(hits, 1) if hit) / len(relevant)
        for k in ks:
            cutoff = min(k, len(qrels))
            count = sum(hits[:cutoff])
            dcg = sum((2 ** qrels[pid] - 1) / math.log2(i + 2) for i, pid in enumerate(rank[:cutoff]))
            ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(qrels.values(), reverse=True)[:cutoff]))
            totals[f"Recall@{k}"] += count / len(relevant)
            totals[f"Precision@{k}"] += count / cutoff
            totals[f"nDCG@{k}"] += dcg / ideal if ideal else 0.
    return {"protocol": RANKING_PROTOCOL, "official_benchmark": False, "official_ProvBench_score": False,
            "fixture_only": fixture_only, "formal_score": False if fixture_only else None,
            "judgment_scope": judgment_scope, "corpus_wide_recall_claim": judgment_scope == "exhaustive_corpus",
            "queries": len(gold), "metrics": {key: value / len(gold) for key, value in totals.items()},
            "pool_note": "Pooled judgments measure only the declared adjudicated pool; unjudged predictions are rejected."}


def validate_ranking_dataset(rows: list[dict], corpus: list[dict], training_inventory: Path | str | None,
                             registry: dict | None, *, judgment_scope="exhaustive_corpus") -> dict:
    """Bind each qrel to frozen two-reviewer adjudication and versioned statute text."""
    report = validate_gold(rows, training_inventory, registry)
    errors = list(report["errors"])
    if judgment_scope not in ("exhaustive_corpus", "adjudicated_pool"):
        errors.append("declared_ranking_judgment_scope_required")
    ids = [article.get("provision_id") for article in corpus]
    if not corpus or len(set(ids)) != len(ids) or not all(isinstance(pid, str) and pid for pid in ids):
        errors.append("nonempty_unique_statute_corpus_required")
    by_id = {article.get("provision_id"): article for article in corpus}
    for article in corpus:
        content, source = article.get("content"), article.get("source") or {}
        if not isinstance(content, str) or not content.strip() or article.get("content_sha256") != digest(content):
            errors.append("statute_snapshot_text_hash_required")
        if not all(source.get(key) for key in ("origin", "license_or_permission", "collected_at", "jurisdiction", "version_id", "effective_as_of")):
            errors.append("statute_source_permission_jurisdiction_version_and_asof_required")
    qrels = []
    for row in rows:
        if row.get("split") != "test":
            errors.append(str(row.get("id")) + ":ranking_quality_evaluation_requires_predeclared_test_split")
        judgments = []
        for ann in row.get("annotations", []):
            if ann.get("type") != "law_reference":
                continue
            article = by_id.get(ann.get("label"))
            if article is None:
                errors.append(row["id"] + ":unknown_adjudicated_statute_id")
                continue
            content = article.get("content", "")
            start, end = ann.get("article_start"), ann.get("article_end")
            if (type(ann.get("relevance")) is not int or not 0 <= ann["relevance"] <= 3
                    or not isinstance(ann.get("rationale"),str) or not ann["rationale"].strip()
                    or not isinstance(ann.get("article_evidence"),str) or not ann["article_evidence"].strip()
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(content) or ann.get("article_evidence") != content[start:end]
                    or ann.get("article_content_sha256") != digest(content)):
                errors.append(row["id"] + ":manual_relevance_rationale_and_exact_versioned_article_evidence_required")
            judgments.append({"provision_id": ann["label"], "relevance": ann.get("relevance")})
        if len({j["provision_id"] for j in judgments}) != len(judgments):
            errors.append(row["id"] + ":duplicate_manual_query_statute_judgment")
        if judgment_scope == "exhaustive_corpus" and {j["provision_id"] for j in judgments} != set(ids):
            errors.append(row["id"] + ":exhaustive_corpus_manual_judgments_required")
        if not any(type(j["relevance"]) is int and j["relevance"] > 0 for j in judgments):
            errors.append(row["id"] + ":at_least_one_manually_relevant_statute_required")
        qrels.append({"id": row["id"], "text_sha256": row["text_sha256"], "judgments": judgments})
    return {**report, "status": "passed" if not errors else "rejected", "gold_ready": not errors,
            "errors": sorted(set(errors)), "protocol": RANKING_PROTOCOL, "official_ProvBench_score": False,
            "judgment_scope": judgment_scope, "qrels": qrels, "corpus_sha256": digest(corpus),
            "legal_currency_assurance": "Versioned source snapshots only; software does not prove current legal applicability."}


def evaluate_ranking(rows: list[dict], corpus: list[dict], predictions: list[dict], training_inventory: Path | str,
                      registry: dict, *, judgment_scope="exhaustive_corpus", ks=(1, 3, 5)) -> dict:
    validation = validate_ranking_dataset(rows, corpus, training_inventory, registry, judgment_scope=judgment_scope)
    if not validation["gold_ready"]:
        raise DatasetValidationError(validation["errors"])
    result = ranking_metrics(validation["qrels"], predictions, corpus, ks=ks, judgment_scope=judgment_scope)
    result.update(formal_score=True, scope="project_owned_protocol_with_operator_attested_human_gold",
                  validation=validation, query_gold_sha256=digest(rows), predictions_sha256=digest(predictions),
                  synthetic_queries_present=any(row.get("synthetic_candidate") for row in rows))
    return result
