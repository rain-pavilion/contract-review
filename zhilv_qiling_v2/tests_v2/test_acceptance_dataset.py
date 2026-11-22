"""Mechanical gate tests only. All people, contracts and reviews here are fixtures.

No test records a real human review or creates a deliverable legal gold dataset.
"""
from copy import deepcopy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from zhilv_qiling_v2 import acceptance_dataset as A


class _HumanGoldFixtures:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = {"reviewers": [
            {"reviewer_id": rid, "role": role, "human_declared": True,
             "registered_by": "AUTOMATED_TEST_FIXTURE_NOT_A_REAL_PERSON",
             "registered_at": "2026-01-01T00:00:00+00:00"}
            for rid, role in (("TEST_A", "reviewer"), ("TEST_B", "reviewer"), ("TEST_C", "adjudicator"))]}
        self.base = [self.candidate()]
        self.inventory = self.root / "inventory.jsonl"
        inv = self.candidate("training-fixture", "训练夹具：商品交付后进行验收。", "train")
        self.inventory.write_text(json.dumps(inv, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def candidate(rid="heldout-fixture", text="测试夹具：甲方应于交付后十日支付价款。", split="test"):
        return {"schema_version": A.SCHEMA, "id": rid, "text": text,
                "text_sha256": A.digest(text), "synthetic_candidate": False,
                "source": {"origin": "AUTOMATED_TEST_FIXTURE_ONLY", "origin_kind": "public_real_contract",
                           "source_id": rid, "group_id": rid, "template_family": rid,
                           "license_or_permission": "mechanical test only, no real legal dataset",
                           "collected_at": "2026-01-01T00:00:00+00:00"},
                "status": "unlabeled", "split": split, "annotations": [], "reviews": []}

    def review(self, slot, annotations=None):
        row = deepcopy(self.base[0])
        row.update(status="in_review", annotations=deepcopy(annotations or []),
                   blind_assignment={"slot": slot, "reviewer_id": "TEST_" + slot,
                                     "started_at": "2026-01-02T00:00:00+00:00",
                                     "completed_at": "2026-01-02T01:00:00+00:00"},
                   review_declaration={"method": "manual", "review_complete": True,
                                       "used_system_predictions": False, "saw_other_review": False,
                                       "empty_annotation_confirmed": True})
        return row

    def decision(self, annotations=None):
        return {"id": self.base[0]["id"], "text_sha256": self.base[0]["text_sha256"],
                "adjudicator_id": "TEST_C", "method": "manual", "used_system_predictions": False,
                "reviewed_both_independent_annotations": True, "rationale": "AUTOMATED FIXTURE, NOT HUMAN WORK",
                "adjudicated_at": "2026-01-03T00:00:00+00:00", "empty_annotation_confirmed": True,
                "annotations": deepcopy(annotations or [])}

    def imported(self, ann=None):
        return A.import_independent_reviews(self.base, [self.review("A", ann)], [self.review("B", ann)], self.registry)

    def gold(self, ann=None):
        return A.adjudicate(self.imported(ann)["rows"], [self.decision(ann)], self.registry)


class HumanGoldGateTests(_HumanGoldFixtures, unittest.TestCase):
    def test_complete_declared_workflow_freezes_once_and_binds_hashes(self):
        rows = self.gold()
        result = A.freeze_gold(rows, self.inventory, self.registry, self.root / "frozen")
        self.assertTrue(result["gold_ready"])
        self.assertFalse(result["human_answers_generated"])
        self.assertIn("operator-declared", result["identity_assurance"])
        self.assertEqual(result["gold_sha256"], A.digest((self.root / "frozen/gold.jsonl").read_bytes()))
        with self.assertRaises(A.DatasetValidationError):
            A.freeze_gold(rows, self.inventory, self.registry, self.root / "frozen")

    def test_empty_unreviewed_package_is_not_gold(self):
        self.assertFalse(A.validate_gold(self.base, self.inventory, self.registry)["gold_ready"])
        with self.assertRaises(A.DatasetValidationError):
            A.import_independent_reviews(self.base, self.base, self.base, self.registry)

    def test_same_reviewer_cannot_fill_both_blind_slots(self):
        b = self.review("B")
        b["blind_assignment"]["reviewer_id"] = "TEST_A"
        with self.assertRaisesRegex(A.DatasetValidationError, "two_distinct"):
            A.import_independent_reviews(self.base, [self.review("A")], [b], self.registry)

    def test_predictions_or_shared_review_are_rejected(self):
        for field in ("used_system_predictions", "saw_other_review"):
            a = self.review("A")
            a["review_declaration"][field] = True
            with self.subTest(field=field), self.assertRaises(A.DatasetValidationError):
                A.import_independent_reviews(self.base, [a], [self.review("B")], self.registry)

    def test_blank_is_not_an_intentionally_empty_review(self):
        a = self.review("A")
        a["review_declaration"].pop("empty_annotation_confirmed")
        with self.assertRaisesRegex(A.DatasetValidationError, "intentional_empty"):
            A.import_independent_reviews(self.base, [a], [self.review("B")], self.registry)

    def test_offsets_must_match_source_exactly(self):
        ann = [{"type": "risk", "label": "TEST", "start": 0, "end": 2, "evidence": "错误"}]
        with self.assertRaisesRegex(A.DatasetValidationError, "evidence_mismatch"):
            self.imported(ann)

    def test_split_or_source_change_after_assignment_rejected(self):
        a = self.review("A")
        a["split"] = "train"
        with self.assertRaisesRegex(A.DatasetValidationError, "source_or_predeclared_split_changed"):
            A.import_independent_reviews(self.base, [a], [self.review("B")], self.registry)

    def test_adjudicator_is_third_person_and_after_both_reviews(self):
        for field, value in (("adjudicator_id", "TEST_A"), ("adjudicated_at", "2026-01-01T00:00:00+00:00")):
            decision = self.decision()
            decision[field] = value
            with self.subTest(field=field), self.assertRaises(A.DatasetValidationError):
                A.adjudicate(self.imported()["rows"], [decision], self.registry)

    def test_retained_adjudication_cannot_diverge_from_gold(self):
        rows = self.gold()
        rows[0]["annotations"] = [{"type": "risk", "label": "TAMPERED", "start": 0, "end": 2, "evidence": rows[0]["text"][:2]}]
        rows[0]["adjudication"]["annotations_sha256"] = A.digest(rows[0]["annotations"])
        self.assertIn("retained_manual_adjudication_differs_from_gold", ";".join(A.validate_gold(rows, self.inventory, self.registry)["errors"]))

    def test_empty_adjudication_confirmation_rechecked_at_freeze(self):
        rows = self.gold()
        rows[0]["adjudication"].pop("empty_annotation_confirmed")
        self.assertIn("empty_adjudication_not_explicitly_confirmed", ";".join(A.validate_gold(rows, self.inventory, self.registry)["errors"]))

    def test_source_and_template_leakage_detected_independently(self):
        for key in ("source_id", "template_family"):
            other = self.candidate("different", "不同内容，不能靠文字去重发现。", "train")
            other["source"][key] = self.base[0]["source"][key]
            self.inventory.write_text(json.dumps(other) + "\n", encoding="utf-8")
            with self.subTest(key=key):
                self.assertIn("evaluation_source_or_template_overlaps_training_inventory", ";".join(A.validate_gold(self.gold(), self.inventory, self.registry)["errors"]))

    def test_long_near_duplicate_requires_manual_family_review(self):
        text = "".join(chr(0x4e00 + n) for n in range(180))
        self.base = [self.candidate(text=text)]
        other = self.candidate("other", text[:-1] + "。", "train")
        self.inventory.write_text(json.dumps(other) + "\n", encoding="utf-8")
        self.assertIn("near_duplicate_training_text", ";".join(A.validate_gold(self.gold(), self.inventory, self.registry)["errors"]))

    def test_internal_train_test_near_duplicates_rejected_with_changed_source_keys(self):
        text = "".join(chr(0x4e00 + n) for n in range(180))
        self.base = [self.candidate(text=text)]
        test = self.gold()
        self.base = [self.candidate("distinct-source", text[:-1] + "。", "train")]
        rows = test + self.gold()
        self.assertIn("near_duplicate_text_crosses_dataset_splits", ";".join(A.validate_gold(rows, self.inventory, self.registry)["errors"]))

    def test_preparation_unions_near_duplicate_templates_before_split(self):
        text = "".join(chr(0x4e00 + n) for n in range(180))
        rows = [self.candidate("a", text, "unassigned"), self.candidate("b", text[:-1] + "。", "unassigned")]
        rows += [self.candidate("other" + str(i), "不同夹具" + str(i), "unassigned") for i in range(4)]
        result = A.assign_group_splits(rows)
        self.assertEqual(result[0]["split"], result[1]["split"])

    def test_synthetic_or_unknown_origin_cannot_be_formal_real_gold(self):
        for mode in ("synthetic", "missing", "unknown"):
            self.base = [self.candidate()]
            if mode == "synthetic":
                self.base[0]["synthetic_candidate"] = True
            elif mode == "missing":
                self.base[0].pop("synthetic_candidate")
            else:
                self.base[0]["source"]["origin_kind"] = "unknown"
            with self.subTest(mode=mode):
                self.assertFalse(A.validate_gold(self.gold(), self.inventory, self.registry)["gold_ready"])

    def test_split_assignment_keeps_connected_sources_and_templates_together(self):
        rows = [self.candidate(str(i), "不同文本" + str(i), "unassigned") for i in range(7)]
        rows[1]["source"]["source_id"] = rows[0]["source"]["source_id"]
        rows[2]["source"]["template_family"] = rows[1]["source"]["template_family"]
        result = A.assign_group_splits(rows)
        self.assertEqual(len({r["split"] for r in result[:3]}), 1)
        self.assertEqual(rows[0]["split"], "unassigned")
        self.assertEqual({r["split"] for r in result}, {"train", "dev", "test"})

    def test_incomplete_training_inventory_never_clears_leakage(self):
        self.assertFalse(A.validate_gold(self.gold(), None, self.registry)["gold_ready"])
        self.inventory.write_text("", encoding="utf-8")
        self.assertFalse(A.validate_gold(self.gold(), self.inventory, self.registry)["gold_ready"])

    def test_empty_or_whitespace_source_cannot_enter_import_or_final_gates(self):
        for text in ("", "  \n\t"):
            self.base = [self.candidate(text=text)]
            with self.subTest(text=repr(text)), self.assertRaisesRegex(A.DatasetValidationError, "nonwhitespace"):
                self.imported()
            self.assertFalse(A.validate_gold(self.base, self.inventory, self.registry)["gold_ready"])

    def test_completed_review_state_rechecked_even_if_hashes_unchanged(self):
        rows = self.gold()
        rows[0]["reviews"][0]["decision"] = "not_completed"
        self.assertIn("independent_review_completion_decision_required", ";".join(A.validate_gold(rows, self.inventory, self.registry)["errors"]))

    def test_whitespace_quote_is_not_annotation_evidence(self):
        self.base = [self.candidate(text="测试 夹具")]
        ann = [{"type": "law_reference", "label": "FIXTURE", "start": 2, "end": 3, "evidence": " "}]
        with self.assertRaisesRegex(A.DatasetValidationError, "cannot_be_whitespace"):
            self.imported(ann)


class ProjectRankingTests(_HumanGoldFixtures, unittest.TestCase):
    def ranking_fixture(self):
        corpus = [{"provision_id": pid, "content": "法条夹具" + pid,
                   "content_sha256": A.digest("法条夹具" + pid),
                   "source": {key: "AUTOMATED_FIXTURE" for key in ("origin", "license_or_permission", "collected_at", "jurisdiction", "version_id", "effective_as_of")}}
                  for pid in ("L1", "L2", "L3")]
        annotations = [{"type": "law_reference", "label": art["provision_id"], "start": 0, "end": 2,
                        "evidence": self.base[0]["text"][:2], "relevance": grade, "rationale": "AUTOMATED_FIXTURE",
                        "article_start": 0, "article_end": 2, "article_evidence": art["content"][:2],
                        "article_content_sha256": art["content_sha256"]}
                       for art, grade in zip(corpus, (3, 0, 1))]
        return self.gold(annotations), corpus

    def test_ranking_metrics_known_small_example(self):
        gold = [{"id": "q", "judgments": [{"provision_id": "L1", "relevance": 1}, {"provision_id": "L2", "relevance": 0}]}]
        predictions = [{"id": "q", "ranked_provision_ids": ["L2", "L1"]}]
        result = A.ranking_metrics(gold, predictions, [{"provision_id": "L1"}, {"provision_id": "L2"}], fixture_only=True)
        self.assertEqual(result["metrics"]["MRR"], .5)
        self.assertEqual(result["metrics"]["MAP"], .5)
        self.assertEqual(result["metrics"]["Recall@1"], 0)
        self.assertEqual(result["metrics"]["Recall@3"], 1)
        self.assertFalse(result["official_ProvBench_score"])
        self.assertFalse(result["formal_score"])

    def test_duplicate_cutoffs_cannot_double_ranking_metrics(self):
        gold = [{"id": "q", "judgments": [{"provision_id": "L", "relevance": 1}]}]
        with self.assertRaisesRegex(A.DatasetValidationError,"unique_ranking_cutoffs"):
            A.ranking_metrics(gold,[{"id":"q","ranked_provision_ids":["L"]}],
                              [{"provision_id":"L"}],ks=(1,1),fixture_only=True)

    def test_formal_entry_rechecks_human_gold_and_statute_version(self):
        rows, corpus = self.ranking_fixture()
        predictions = [{"id": rows[0]["id"], "ranked_provision_ids": ["L1", "L3", "L2"]}]
        result = A.evaluate_ranking(rows, corpus, predictions, self.inventory, self.registry)
        self.assertTrue(result["formal_score"])
        self.assertFalse(result["official_ProvBench_score"])
        corpus[0]["content"] += "篡改"
        with self.assertRaises(A.DatasetValidationError):
            A.evaluate_ranking(rows, corpus, predictions, self.inventory, self.registry)

    def test_no_blank_gold_or_unjudged_negative_shortcut(self):
        rows, corpus = self.ranking_fixture()
        with self.assertRaises(A.DatasetValidationError):
            A.evaluate_ranking(self.base, corpus, [], self.inventory, self.registry)
        qrels = [{"id": "q", "judgments": [{"provision_id": "L1", "relevance": 1}]}]
        with self.assertRaisesRegex(A.DatasetValidationError, "unjudged"):
            A.ranking_metrics(qrels, [{"id": "q", "ranked_provision_ids": ["L1", "L2", "L3"]}], corpus, judgment_scope="adjudicated_pool")

    def test_truncated_ranking_cannot_claim_unqualified_MAP(self):
        corpus = [{"provision_id": "L" + str(i)} for i in range(6)]
        qrels = [{"id": "q", "judgments": [{"provision_id": art["provision_id"], "relevance": 1} for art in corpus]}]
        with self.assertRaisesRegex(A.DatasetValidationError, "full_declared"):
            A.ranking_metrics(qrels, [{"id": "q", "ranked_provision_ids": [art["provision_id"] for art in corpus[:5]]}], corpus)

    def test_scope_and_adjudicated_empty_are_rejected(self):
        rows, corpus = self.ranking_fixture()
        self.assertFalse(A.validate_ranking_dataset(rows, corpus, self.inventory, self.registry, judgment_scope="other")["gold_ready"])
        self.assertFalse(A.validate_ranking_dataset(self.gold(), corpus, self.inventory, self.registry)["gold_ready"])

    def test_whitespace_statute_snapshot_cannot_be_ranked_as_evidence(self):
        rows, corpus = self.ranking_fixture()
        corpus[0].update(content="  ",content_sha256=A.digest("  "))
        self.assertFalse(A.validate_ranking_dataset(rows, corpus, self.inventory, self.registry)["gold_ready"])


class WorkflowCliTests(_HumanGoldFixtures, unittest.TestCase):
    def test_preparation_never_fills_people_or_answers(self):
        from tools.selected_solutions_20261118 import human_acceptance as C
        path = self.root / "candidates.jsonl"
        C.new_jsonl(path, [self.candidate(str(i), "夹具" + str(i), "unassigned") for i in range(5)])
        result = C.prepare(path, self.root / "blank")
        self.assertFalse(result["gold_ready"])
        self.assertFalse(result["human_answers_generated"])
        self.assertEqual(C.read_json(self.root / "blank/operator_registry_blank.json")["reviewers"], [])
        rows = A.read_jsonl(self.root / "blank/review_A.jsonl")
        self.assertTrue(all(r["annotations"] == [] and r["blind_assignment"]["reviewer_id"] is None for r in rows))
        with self.assertRaises(A.DatasetValidationError):
            A.import_independent_reviews(A.read_jsonl(self.root / "blank/candidates.jsonl"), rows,
                A.read_jsonl(self.root / "blank/review_B.jsonl"), {})
        with self.assertRaises(A.DatasetValidationError):
            C.prepare(path, self.root / "blank")

    def test_csv_completion_inventory_is_never_gold(self):
        from tools.selected_solutions_20261118 import human_acceptance as C
        package = self.root / "package"
        package.mkdir()
        A.write_new_json(package / "manifest.json", {"records": [{"contract_id": "FIXTURE"}]})
        for slot in ("A", "B", "仲裁"):
            for task in ("clause_law", "entity_fact", "cross_clause_loop"):
                with (package / f"{slot}_{task}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("review_complete", "annotation_status"))
                    writer.writeheader()
                    writer.writerow({"review_complete": "true", "annotation_status": "completed"})
        result = C.audit_package(package)
        self.assertEqual(sum(f["completed_rows"] for f in result["files"].values()), 9)
        self.assertFalse(result["gold_ready"])
        self.assertIsNone(result["formal_score"])

    def test_cli_validation_persists_rejection_and_never_overwrites(self):
        from tools.selected_solutions_20261118 import human_acceptance as C
        path, report = self.root / "candidates.jsonl", self.root / "validation.json"
        C.new_jsonl(path, self.base)
        with redirect_stdout(io.StringIO()):
            code = C.main(["validate", "--data", str(path), "--out", str(report)])
        self.assertEqual(code, 2)
        original = report.read_bytes()
        self.assertFalse(C.read_json(report)["gold_ready"])
        with redirect_stdout(io.StringIO()):
            code = C.main(["validate", "--data", str(path), "--out", str(report)])
        self.assertEqual(code, 2)
        self.assertEqual(report.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
