"""Prepare blank reviews, import human decisions, freeze gold, evaluate project ranking.

This CLI never generates an annotation or an operator's identity/declaration.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import acceptance_dataset as A


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(candidates, out, seed=20261118):
    rows = A.read_jsonl(candidates)
    A._base(rows)
    if all(row["split"] == "unassigned" for row in rows):
        rows = A.assign_group_splits(rows, seed)
    elif any(row["split"] == "unassigned" for row in rows):
        raise A.DatasetValidationError(["mixed_assigned_and_unassigned_splits_require_manual_resolution"])
    if out.exists():
        raise A.DatasetValidationError(["new_blank_package_directory_required"])
    out.mkdir(parents=True)
    new_jsonl(out / "candidates.jsonl", rows)
    for slot in ("A", "B"):
        reviews = deepcopy(rows)
        for row in reviews:
            row.update(status="in_review", blind_assignment={"slot": slot, "reviewer_id": None,
                       "started_at": None, "completed_at": None},
                       review_declaration={"method": None, "review_complete": False,
                       "used_system_predictions": None, "saw_other_review": None,
                       "empty_annotation_confirmed": False})
        new_jsonl(out / f"review_{slot}.jsonl", reviews)
    decisions = [{"id": row["id"], "text_sha256": row["text_sha256"], "adjudicator_id": None,
                  "method": None, "used_system_predictions": None,
                  "reviewed_both_independent_annotations": False, "rationale": "",
                  "adjudicated_at": None, "empty_annotation_confirmed": False, "annotations": []}
                 for row in rows]
    new_jsonl(out / "adjudication_blank.jsonl", decisions)
    A.write_new_json(out / "operator_registry_blank.json", {"reviewers": [], "required_fields_per_person":
        ["reviewer_id", "role (reviewer or adjudicator)", "human_declared", "registered_by", "registered_at (timezone ISO8601)"]})
    report = {"status": "pending_human_annotation", "records": len(rows), "seed": seed,
              "candidate_sha256": A.digest(Path(candidates).read_bytes()), "human_answers_generated": False,
              "identities_generated": False, "gold_ready": False, "formal_score": None,
              "source_metadata_copied_not_authenticated": True}
    A.write_new_json(out / "preparation.json", report)
    return report


def audit_package(package):
    """Inventory existing CSV package. Completion counts never constitute gold."""
    manifest = read_json(package / "manifest.json")
    files = {}
    for slot in ("A", "B", "仲裁"):
        for task in ("clause_law", "entity_fact", "cross_clause_loop"):
            path = package / f"{slot}_{task}.csv"
            if not path.exists():
                files[path.name] = {"exists": False, "completed_rows": 0}
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            complete = sum(str(row.get("review_complete", "")).strip().casefold() in ("true", "1", "yes", "是")
                           and row.get("annotation_status") in ("completed", "adjudicated") for row in rows)
            files[path.name] = {"exists": True, "rows": len(rows), "completed_rows": complete,
                                "sha256": A.digest(path.read_bytes())}
    return {"status": "pending_human_annotation", "contracts": len(manifest.get("records", [])),
            "manifest_sha256": A.digest((package / "manifest.json").read_bytes()), "files": files,
            "gold_ready": False, "formal_score": None, "human_answers_generated": False,
            "blockers": ["CSV completion is not reviewed gold: exact source offsets, blind identities, third-person adjudication and leakage clearance are required",
                         "existing source package reports unresolved prior training exposure and source permission; no independent acceptance clearance is issued"],
            "scope": "local artifact inventory only; no real identity, source authenticity or legal label certification"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    q = commands.add_parser("prepare")
    q.add_argument("--candidates", type=Path, required=True)
    q.add_argument("--seed", type=int, default=20261118)
    q.add_argument("--out", type=Path, required=True)
    q = commands.add_parser("import_reviews")
    for name in ("candidates", "review-a", "review-b", "registry", "out"):
        q.add_argument("--" + name, type=Path, required=True)
    q = commands.add_parser("adjudicate")
    for name in ("data", "decisions", "registry", "out"):
        q.add_argument("--" + name, type=Path, required=True)
    for action in ("validate", "freeze", "ranking"):
        q = commands.add_parser(action)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--registry", type=Path)
        q.add_argument("--training-inventory", type=Path)
        q.add_argument("--out", type=Path, required=True)
        if action == "ranking":
            q.add_argument("--corpus", type=Path, required=True)
            q.add_argument("--predictions", type=Path, required=True)
            q.add_argument("--judgment-scope", choices=("exhaustive_corpus", "adjudicated_pool"), default="exhaustive_corpus")
    q = commands.add_parser("audit_package")
    q.add_argument("--package", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.candidates, args.out, args.seed)
        elif args.action == "audit_package":
            result = audit_package(args.package)
            A.write_new_json(args.out, result)
        elif args.action == "import_reviews":
            result = A.import_independent_reviews(A.read_jsonl(args.candidates), A.read_jsonl(args.review_a),
                                                  A.read_jsonl(args.review_b), read_json(args.registry))
            args.out.mkdir(parents=True, exist_ok=False)
            new_jsonl(args.out / "imported.jsonl", result.pop("rows"))
            A.write_new_json(args.out / "import_receipt.json", result)
        elif args.action == "adjudicate":
            rows = A.adjudicate(A.read_jsonl(args.data), A.read_jsonl(args.decisions), read_json(args.registry))
            new_jsonl(args.out, rows)
            result = {"status": "human_decisions_imported_not_yet_frozen", "records": len(rows), "gold_ready": False}
        else:
            rows = A.read_jsonl(args.data)
            registry = read_json(args.registry) if args.registry else {}
            if args.action == "validate":
                result = A.validate_gold(rows, args.training_inventory, registry)
                A.write_new_json(args.out, result)
            elif args.action == "freeze":
                result = A.freeze_gold(rows, args.training_inventory, registry, args.out)
            else:
                result = A.evaluate_ranking(rows, read_json(args.corpus), A.read_jsonl(args.predictions),
                                            args.training_inventory, registry, judgment_scope=args.judgment_scope)
                A.write_new_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result.get("gold_ready") is False else 0
    except (A.DatasetValidationError, ValueError, KeyError, TypeError, OSError) as exc:
        result = {"status": "rejected", "gold_ready": False, "formal_score": None,
                  "human_answers_generated": False, "errors": getattr(exc, "errors", [str(exc)])}
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
