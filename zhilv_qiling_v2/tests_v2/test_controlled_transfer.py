"""Lightweight transfer checks: no model construction, inference or training.

The retained real run uses repeatedly inspected synthetic weak labels.  These
tests establish fixed-plan and receipt consistency, never legal accuracy.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import controlled_transfer as C


RUN_DIRECTORY = C.ROOT / "tools/selected_solutions_20261118/transfer_ablation_real_v1"


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


@lru_cache(maxsize=1)
def review_saved_run() -> dict:
    """Rehash retained files and independently recalculate reported metrics."""
    run_path, plan_path = RUN_DIRECTORY / "run.json", RUN_DIRECTORY / "plan.json"
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    expected_ids = {init + "__" + profile for init in C.CHECKPOINTS for profile in C.PROFILES}
    arms = run.get("arms", [])
    checks = {
        "four_arm_completed_receipt": run.get("status") == "four_arm_controlled_weak_diagnostic_completed"
            and len(arms) == 4 and {arm["id"] for arm in arms} == expected_ids,
        "plan_file_hash_matches": C.file_sha(plan_path) == run.get("plan_sha256"),
        "fixed_data_file_hash_matches": C.file_sha(Path(plan["data_path"])) == plan.get("target_data_raw_sha256") == C.RAW_DATA_SHA,
        "fixed_step_count_and_batch_size": plan.get("optimizer_steps_per_arm") == 12 and plan.get("batch_size") == 2
            and all(arm.get("completed_optimizer_steps") == 12 for arm in arms),
        "same_fresh_target_head_hash": len({arm.get("target_head_initial_sha256") for arm in arms}) == 1
            and all(len(arm.get("target_head_initial_sha256", "")) == 64 for arm in arms)
            and plan.get("fresh_identical_target_head") is True,
        "no_five_to_three_head_mapping": plan.get("five_to_three_label_mapping") is False
            and all(arm.get("source_five_class_head_loaded") is False and arm.get("label_mapping") is None for arm in arms),
        "same_training_order_and_steps": all([row["step"] for row in arm.get("history", [])] == list(range(1, 13))
            and [pid for row in arm["history"] for pid in row["pair_ids"]] == plan["training_order"] for arm in arms),
        "freeze_checks_reported_true": all(arm.get("frozen_parameters_unchanged") is True for arm in arms),
        "trainable_parameters_follow_freeze_profiles": all(arm.get("trainable_parameters")
            and all(C.trainable_parameter(name, arm["freeze_profile"]) for name in arm["trainable_parameters"])
            and not any(name.startswith("encoder.") for name in arm["trainable_parameters"]) for arm in arms),
        "nonzero_updates_and_gradient_receipts": all(any(value > 0 for value in arm.get("parameter_l2_deltas", {}).values())
            and set(arm["parameter_l2_deltas"]) == set(arm["trainable_parameters"]) == set(arm["gradient_nonzero_steps"])
            and all(type(value) is int and 0 <= value <= 12 for value in arm["gradient_nonzero_steps"].values())
            and any(value > 0 for value in arm["gradient_nonzero_steps"].values()) for arm in arms),
        "save_roundtrip_assertions_reported_true": all(arm.get("save_roundtrip_exact") is True for arm in arms),
        "human_gold_and_quality_acceptance_not_claimed": run.get("human_gold") is False and run.get("quality_acceptance") is False
            and plan.get("human_gold") is False and plan.get("quality_acceptance") is False,
        "product_default_change_not_claimed": run.get("product_default_changed") is False and plan.get("project_default_changed") is False,
        "default_weight_unchanged_assertion_reported_true": run.get("default_checkpoint_sha256_unchanged") is True,
    }
    file_checks, metrics, comparisons = [], [], []
    for arm in arms:
        path = RUN_DIRECTORY / arm["trainable_state_file"]
        actual = C.file_sha(path)
        file_checks.append({"kind": "saved_trainable_state", "arm_id": arm["id"], "path": str(path),
            "bytes": path.stat().st_size, "actual_sha256": actual, "recorded_sha256": arm["trainable_state_sha256"],
            "matches": actual == arm["trainable_state_sha256"]})
        for phase in ("before", "after"):
            for split in ("valid", "test"):
                evaluation = arm[phase][split]
                predictions = evaluation["predictions"]
                confusion = {a: {b: 0 for b in C.LABELS} for a in C.LABELS}
                for row in predictions:
                    confusion[row["weak_expected"]][row["predicted"]] += 1
                f1 = []
                for label in C.LABELS:
                    tp = confusion[label][label]
                    fp = sum(confusion[other][label] for other in C.LABELS if other != label)
                    fn = sum(confusion[label][other] for other in C.LABELS if other != label)
                    f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
                accuracy = sum(confusion[label][label] for label in C.LABELS) / len(predictions)
                macro_f1 = sum(f1) / len(f1)
                metrics.append({"arm_id": arm["id"], "phase": phase, "split": split,
                    "accuracy_recalculated": accuracy, "macro_f1_recalculated": macro_f1,
                    "consistent": confusion == evaluation["confusion"] and _close(accuracy, evaluation["accuracy"])
                        and _close(macro_f1, evaluation["macro_f1"]) and evaluation["rows"] == len(predictions)
                        and [row["pair_id"] for row in predictions] == plan["target_pair_ids"][split]
                        and evaluation.get("human_gold") is False})
    for name, recorded in run["source_checkpoints"].items():
        actual = C.file_sha(Path(recorded["path"]) / "model.pt")
        expected = C.CHECKPOINTS[name][1]
        file_checks.append({"kind": "source_checkpoint_weights", "initialization": name,
            "path": str(Path(recorded["path"]) / "model.pt"), "actual_sha256": actual,
            "recorded_sha256": recorded["weights_sha256"], "pinned_sha256": expected,
            "matches": actual == recorded["weights_sha256"] == expected})
    indexed = {arm["id"]: arm for arm in arms}
    for entry in run["comparisons"]:
        profile = entry["freeze_profile"]
        base, legal = indexed["base__" + profile], indexed["legal_increment__" + profile]
        accuracy_delta = legal["after"]["test"]["accuracy"] - base["after"]["test"]["accuracy"]
        f1_delta = legal["after"]["test"]["macro_f1"] - base["after"]["test"]["macro_f1"]
        comparisons.append({"freeze_profile": profile, "weak_test_accuracy_delta_recalculated": accuracy_delta,
            "weak_test_macro_f1_delta_recalculated": f1_delta,
            "consistent": _close(accuracy_delta, entry["weak_test_accuracy_delta"])
                and _close(f1_delta, entry["weak_test_macro_f1_delta"])
                and entry.get("negative_or_zero_results_retained") is True and entry.get("independent_quality_conclusion") is None})
    checks.update(retained_file_hashes_match=all(row["matches"] for row in file_checks),
        all_evaluation_metrics_recalculated=all(row["consistent"] for row in metrics),
        all_comparison_deltas_recalculated=len(comparisons) == 2 and all(row["consistent"] for row in comparisons))
    raw_frozen_hashes_present = all(arm.get("frozen_before_sha256") and arm.get("frozen_after_sha256") for arm in arms)
    return {"reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_saved_run_and_file_hash_review_no_model_execution",
        "run_path": str(run_path), "run_sha256": C.file_sha(run_path), "plan_path": str(plan_path),
        "production_module_modified_by_this_review": False, "training_or_inference_executed_by_this_review": False,
        "receipt_consistency_checks_passed": all(checks.values()), "checks": checks,
        "file_checks": file_checks, "recalculated_evaluations": metrics, "recalculated_comparisons": comparisons,
        "fresh_head_sha256": arms[0]["target_head_initial_sha256"],
        "independent_frozen_tensor_hash_reconstruction_available": bool(raw_frozen_hashes_present),
        "recorded_implementation_sha256": run.get("implementation_sha256"),
        "current_implementation_sha256": C.file_sha(Path(C.__file__)),
        "human_gold": False, "independent_legal_quality_acceptance": False,
        "weak_test_improvement_observed": any(row["weak_test_accuracy_delta_recalculated"] > 0
            or row["weak_test_macro_f1_delta_recalculated"] > 0 for row in comparisons),
        "audit_limitations": [
            "The retained run records frozen_parameters_unchanged assertions, not before/after frozen tensor hashes; those historical tensor hashes cannot be independently reconstructed from this receipt.",
            "This review rehashes existing files and recalculates metrics from retained predictions; it does not rerun the models or independently certify historical execution.",
            "All target labels are repeatedly inspected synthetic weak supervision, one seed and 24 training rows. Equal/positive/negative results cannot establish independent real-contract legal quality.",
            "The source implementation may have changed after the recorded run; both hashes are retained rather than falsely equated."],
        "conclusion": "All four retained arms completed the fixed diagnostic with identical head initialization; both transfer-minus-base weak test deltas are zero. No real legal-quality improvement is established."}


class ControlledTransferPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pairs = C.ROOT / "tools/acceptance_results/mamba_entail_training/pairs.jsonl"
        cls.rows = C.read_jsonl(pairs) if pairs.is_file() else None

    def require_retained_pairs(self):
        if self.rows is None:
            self.skipTest("Pinned local training-pair evidence is not distributed in this checkout")

    def mocked_plan(self, rows):
        with patch.object(C, "file_sha", return_value=C.RAW_DATA_SHA), patch.object(C, "read_jsonl", return_value=rows):
            return C.build_ablation_plan(Path("FIXTURE_ONLY.jsonl"))

    def test_fixed_plan_balanced_and_reproducible_without_quality_claim(self):
        self.require_retained_pairs()
        plan = C.build_ablation_plan()
        self.assertEqual(plan, C.build_ablation_plan())
        by_id = {row["pair_id"]: row for row in self.rows}
        for split, count in (("train", 8), ("valid", 4), ("test", 4)):
            self.assertEqual(Counter(by_id[pid]["label"] for pid in plan["target_pair_ids"][split]),
                             Counter({label: count for label in C.LABELS}))
        self.assertEqual(len(plan["arms"]), 4)
        self.assertEqual(plan["optimizer_steps_per_arm"] * plan["batch_size"], len(plan["training_order"]))
        self.assertFalse(plan["human_gold"])
        self.assertFalse(plan["quality_acceptance"])
        self.assertFalse(plan["five_to_three_label_mapping"])

    def test_plan_has_source_group_and_template_isolation(self):
        self.require_retained_pairs()
        for field in ("source_id", "group_id", "template_family"):
            groups = {split: {row[field] for row in self.rows if row["split"] == split} for split in ("train", "valid", "test")}
            for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
                self.assertFalse(groups[first] & groups[second])

    def test_deterministic_selection_is_not_input_row_order_or_global_rng_dependent(self):
        self.require_retained_pairs()
        before = random.getstate()
        normal = self.mocked_plan(copy.deepcopy(self.rows))
        reverse = self.mocked_plan(list(reversed(copy.deepcopy(self.rows))))
        self.assertEqual(normal, reverse)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(set(normal["training_order"]), set(normal["target_pair_ids"]["train"]))
        self.assertFalse(set(normal["training_order"]) & set(normal["target_pair_ids"]["test"]))

    def test_changed_data_hash_is_rejected_before_rows_are_loaded(self):
        with patch.object(C, "file_sha", return_value="changed"), patch.object(C, "read_jsonl") as read:
            with self.assertRaisesRegex(ValueError, "original_fixed_weak_target_data_hash_changed"):
                C.build_ablation_plan(Path("FIXTURE_ONLY.jsonl"))
            read.assert_not_called()

    def test_wrong_count_or_unmarked_supervision_is_rejected(self):
        self.require_retained_pairs()
        wrong = copy.deepcopy(self.rows)
        wrong[0]["supervision"] = "not_declared_weak_supervision"
        for rows in (self.rows[:-1], wrong):
            with self.subTest(records=len(rows)), self.assertRaisesRegex(ValueError, "original_288_synthetic_weak_pairs_required"):
                self.mocked_plan(rows)

    def test_each_source_group_or_template_cross_split_leak_is_rejected(self):
        self.require_retained_pairs()
        for field in ("source_id", "group_id", "template_family"):
            rows = copy.deepcopy(self.rows)
            training = next(row for row in rows if row["split"] == "train")
            testing = next(row for row in rows if row["split"] == "test")
            testing[field] = training[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "original_target_group_leakage"):
                self.mocked_plan(rows)

    def test_insufficient_class_is_rejected_instead_of_replacing_with_another_label(self):
        self.require_retained_pairs()
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["split"] == "train" and row["label"] == "entail":
                row["label"] = "neutral"
        with self.assertRaisesRegex(ValueError, "insufficient_predeclared_balanced_target_rows"):
            self.mocked_plan(rows)

    def test_shared_transfer_excludes_all_old_prediction_heads(self):
        for name in ("encoder.layer.0.weight", "mamba.layers.0.weight", "mamba_gate", "mamba_input_projection.weight", "attn.in_proj_weight", "norm.weight"):
            self.assertTrue(C.is_shared(name), name)
        for name in ("head.0.weight", "head.3.bias", "rule_head.weight", "other.weight"):
            self.assertFalse(C.is_shared(name), name)

    def test_freeze_profiles_keep_encoder_frozen_and_select_intended_parameters(self):
        names = ("encoder.layer.0.weight", "mamba.layers.0.weight", "mamba_gate", "mamba_input_projection.weight",
                 "mamba_output_projection.bias", "attn.in_proj_weight", "norm.weight", "head.0.weight", "rule_head.weight", "other.weight")
        self.assertEqual({name for name in names if C.trainable_parameter(name, "head_only")}, {"head.0.weight"})
        self.assertEqual({name for name in names if C.trainable_parameter(name, C.PROFILES[1])},
                         set(names) - {"encoder.layer.0.weight", "rule_head.weight", "other.weight"})
        with self.assertRaisesRegex(ValueError, "unknown_fixed_freeze_profile"):
            C.trainable_parameter("head.0.weight", "not_a_profile")

    def test_tensor_receipt_hash_is_state_order_invariant_and_value_sensitive(self):
        class FakeTensor:
            def __init__(self, data): self.data = data
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return self.data
        first = {"a": FakeTensor(b"one"), "b": FakeTensor(b"two")}
        self.assertEqual(C._tensor_sha(first), C._tensor_sha(dict(reversed(list(first.items())))))
        self.assertNotEqual(C._tensor_sha(first), C._tensor_sha({**first, "b": FakeTensor(b"changed")}))

    def test_existing_output_refused_before_plan_or_checkpoints_are_read(self):
        with TemporaryDirectory() as directory, patch.object(C, "build_ablation_plan") as plan:
            with self.assertRaisesRegex(ValueError, "fresh_ablation_directory_required"):
                C.run_ablation(Path(directory))
            plan.assert_not_called()

    def test_checkpoint_pin_failure_leaves_no_output_and_never_imports_models(self):
        with TemporaryDirectory() as directory, patch.object(C, "build_ablation_plan", return_value={}), patch.object(C, "file_sha", return_value="wrong"):
            output = Path(directory) / "fresh"
            with self.assertRaisesRegex(ValueError, "pinned_source_checkpoint_changed:base"):
                C.run_ablation(output)
            self.assertFalse(output.exists())

    def fake_runner(self, directory, token_count=1):
        root = Path(directory)
        checkpoints = {}
        source_paths = []
        for name, labels in (("base", 3), ("legal_increment", 5)):
            path = root / name
            path.mkdir()
            (path / "model.pt").write_bytes(("AUTOMATED_NO_MODEL_FIXTURE_" + name).encode())
            (path / "meta.json").write_text(json.dumps({"architecture": {"mamba_config": {}}, "num_labels": labels}), encoding="utf-8")
            checkpoints[name] = (name, C.file_sha(path / "model.pt"))
            source_paths.append(path / "model.pt")
        rows = [{"pair_id": str(i), "clause": "fixture", "article": "fixture", "label": C.LABELS[i % 3]} for i in range(24)]
        plan = {"data_path": "NEVER_OPENED_FIXTURE.jsonl", "target_pair_ids": {"train": [row["pair_id"] for row in rows],
                "valid": ["0", "1", "2"], "test": ["3", "4", "5"]}, "seed": 20261118,
                "arms": [{"id": "fixture", "initialization": "base", "freeze_profile": "head_only"}], "limitations": ["AUTOMATED NO MODEL FIXTURE"]}
        seeds = []
        fake_torch = ModuleType("torch")
        fake_torch.set_num_threads = lambda value: None
        fake_torch.set_num_interop_threads = lambda value: None
        fake_torch.random = SimpleNamespace(default_generator=SimpleNamespace(manual_seed=seeds.append))
        fake_torch.__version__ = "NO_TRAINING_FIXTURE"
        fake_transformers = ModuleType("transformers")
        class FakeTokenizer:
            backend_tokenizer = SimpleNamespace(no_truncation=lambda: None, no_padding=lambda: None)
            def __call__(self, value, **kwargs):
                if kwargs.get("truncation") is not False:
                    raise AssertionError("Unexpected truncation")
                return {"input_ids": [1] * token_count}
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
        fake_backend = ModuleType("zhilv_qiling_v2.bert_entail")
        fake_backend.build_model = unittest.mock.Mock(side_effect=RuntimeError("INTENTIONAL_FAILURE_BEFORE_ANY_MODEL_BUILD"))
        return root, checkpoints, rows, plan, fake_torch, fake_transformers, fake_backend, source_paths

    def test_failure_before_any_model_build_is_recorded_and_sources_unchanged(self):
        import zhilv_qiling_v2 as package
        with TemporaryDirectory() as directory:
            root, checkpoints, rows, plan, torch, transformers, backend, sources = self.fake_runner(directory)
            before = {str(path): C.file_sha(path) for path in sources}
            with patch.object(C, "ROOT", root), patch.object(C, "CHECKPOINTS", checkpoints), patch.object(C, "build_ablation_plan", return_value=plan), \
                 patch.object(C, "read_jsonl", return_value=rows), patch.dict(sys.modules, {"torch": torch, "transformers": transformers,
                    "zhilv_qiling_v2.bert_entail": backend}), patch.object(package, "bert_entail", backend, create=True), patch.dict(C.os.environ, {}):
                with self.assertRaisesRegex(RuntimeError, "INTENTIONAL_FAILURE_BEFORE_ANY_MODEL_BUILD"):
                    C.run_ablation(root / "output")
            receipt = json.loads((root / "output/run.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "interrupted_or_failed_diagnostic")
            self.assertEqual(receipt["arms"], [])
            self.assertEqual(before, {str(path): C.file_sha(path) for path in sources})

    def test_overlength_input_is_rejected_before_any_model_build(self):
        import zhilv_qiling_v2 as package
        with TemporaryDirectory() as directory:
            root, checkpoints, rows, plan, torch, transformers, backend, _ = self.fake_runner(directory, token_count=385)
            with patch.object(C, "ROOT", root), patch.object(C, "CHECKPOINTS", checkpoints), patch.object(C, "build_ablation_plan", return_value=plan), \
                 patch.object(C, "read_jsonl", return_value=rows), patch.dict(sys.modules, {"torch": torch, "transformers": transformers,
                    "zhilv_qiling_v2.bert_entail": backend}), patch.object(package, "bert_entail", backend, create=True), patch.dict(C.os.environ, {}):
                with self.assertRaisesRegex(ValueError, "complete_input_exceeds_fixed_384_budget_no_truncation"):
                    C.run_ablation(root / "output")
            backend.build_model.assert_not_called()
            self.assertFalse((root / "output/run.json").exists())


class RetainedTransferReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (RUN_DIRECTORY / "run.json").is_file():
            raise unittest.SkipTest("Retained local real-run artifact is not distributed in this checkout")
        cls.review = review_saved_run()

    def test_retained_receipt_plan_metric_and_file_consistency(self):
        self.assertTrue(self.review["receipt_consistency_checks_passed"], self.review["checks"])
        self.assertFalse(self.review["training_or_inference_executed_by_this_review"])

    def test_zero_deltas_and_weak_scope_are_retained(self):
        self.assertFalse(self.review["weak_test_improvement_observed"])
        for row in self.review["recalculated_comparisons"]:
            self.assertEqual(row["weak_test_accuracy_delta_recalculated"], 0.)
            self.assertEqual(row["weak_test_macro_f1_delta_recalculated"], 0.)
        self.assertFalse(self.review["human_gold"])
        self.assertFalse(self.review["independent_legal_quality_acceptance"])

    def test_missing_historical_frozen_tensor_hashes_are_disclosed(self):
        self.assertFalse(self.review["independent_frozen_tensor_hash_reconstruction_available"])
        self.assertTrue(self.review["audit_limitations"])


if __name__ == "__main__":
    unittest.main()
