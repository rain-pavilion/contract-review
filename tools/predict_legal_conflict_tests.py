"""Standard-library CLI guards only; never import torch or execute a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import predict_legal_conflict as P


def valid_meta():
    return {"task": P.TASK, "num_labels": 5, "class_names": list(P.CLASS_NAMES),
        "compatible_with_contract_entailment_runtime": False, "contract_law_quality_accepted": False,
        "human_gold": False, "product_default_unchanged": True, "auxiliary_legal_dataset": True,
        "source_manifest_sha256": P.SOURCE_MANIFEST_SHA256,
        "architecture": {"mamba": True, "rule_residual": False, "encoder_hidden_size": 768,
                         "dropout": .1, "mamba_config": {"hidden_size": 64, "state_size": 8, "num_hidden_layers": 1}}}


def valid_input():
    return {"task": P.TASK, "items": [{"id": "released-test-1", "query": "给定下位法规原文。", "reference": "给定上位法原文。"}]}


class LegalConflictPredictionGuards(unittest.TestCase):
    def test_import_is_lightweight(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_accepts_original_task_schema(self):
        self.assertEqual(P.validate_input(valid_input())[0]["id"], "released-test-1")
        self.assertEqual(P.validate_meta(valid_meta())["num_labels"], 5)

    def test_contract_task_or_contract_fields_are_rejected(self):
        for mutation in ("task", "fields", "extra"):
            value = valid_input()
            if mutation == "task": value["task"] = "contract_law_entailment"
            elif mutation == "fields": value["items"][0] = {"id": "x", "clause": "合同", "article": "法条"}
            else: value["contract_acceptance"] = True
            with self.assertRaises(ValueError): P.validate_input(value)

    def test_blank_duplicate_and_unbounded_input_rejected(self):
        for mutation in ("blank", "duplicate", "long", "missing", "empty", "too_many"):
            value = valid_input()
            if mutation == "blank": value["items"][0]["query"] = "  "
            elif mutation == "duplicate": value["items"] *= 2
            elif mutation == "long": value["items"][0]["reference"] = "法" * 20001
            elif mutation == "missing": del value["items"][0]["reference"]
            elif mutation == "empty": value["items"] = []
            else: value["items"] *= 129
            with self.assertRaises(ValueError): P.validate_input(value)

    def test_native_three_class_or_reordered_metadata_rejected(self):
        for mutation in ("three", "bool", "missing", "order", "task"):
            value = valid_meta()
            if mutation == "three": value.update(num_labels=3, class_names=["entail", "contradict", "neutral"])
            elif mutation == "bool": value["num_labels"] = True
            elif mutation == "missing": del value["class_names"]
            elif mutation == "order": value["class_names"].reverse()
            else: value["task"] = "contract_law_entailment"
            with self.assertRaises(ValueError): P.validate_meta(value)

    def test_boundary_or_architecture_metadata_cannot_be_omitted(self):
        for key in ("compatible_with_contract_entailment_runtime", "contract_law_quality_accepted", "human_gold",
                    "product_default_unchanged", "auxiliary_legal_dataset", "source_manifest_sha256", "architecture"):
            value = valid_meta(); del value[key]
            with self.assertRaises(ValueError): P.validate_meta(value)

    def test_rule_head_or_missing_mamba_config_rejected(self):
        for mutation in ("residual", "mamba", "config"):
            value = valid_meta()
            if mutation == "residual": value["architecture"]["rule_residual"] = True
            elif mutation == "mamba": value["architecture"]["mamba"] = False
            else: del value["architecture"]["mamba_config"]
            with self.assertRaises(ValueError): P.validate_meta(value)

    def test_probabilities_are_finite_normalized_and_native_five_class(self):
        value = P.normalized_probabilities([.1, .2, .3, .15, .25000001])
        self.assertAlmostEqual(sum(value), 1, places=14)
        for bad in ([1, 0, 0], [float("nan"), .25, .25, .25, .25],
                    [float("inf"), 0, 0, 0, 0], [-.1, .2, .3, .3, .3],
                    [0, 0, 0, 0, 0], [.1] * 5, [True, 0, 0, 0, 0]):
            with self.assertRaises(ValueError): P.normalized_probabilities(bad)

    def test_duplicate_json_keys_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for source in ('{"task":"x","task":"y"}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(ValueError): P.read_json(path)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError): P.write_new_json(path, {"x": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")

    def test_write_disallows_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            with self.assertRaises(ValueError): P.write_new_json(path, {"x": float("nan")})
            self.assertFalse(path.exists())

    def test_checkpoint_hashes_are_enforced_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "meta.json").write_text(json.dumps(valid_meta()), encoding="utf-8")
            (base / "model.pt").write_bytes(b"synthetic guard bytes, not a model")
            pins = {name: P.file_sha256(base / name) for name in ("meta.json", "model.pt")}
            with patch.object(P, "PINNED_FILES", pins):
                path, meta, _ = P.inspect_checkpoint(base)
                self.assertEqual(path, base.resolve())
                self.assertEqual(meta["task"], P.TASK)
                (base / "model.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"): P.inspect_checkpoint(base)
        self.assertNotIn("torch", sys.modules)

    def test_missing_meta_fails_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): P.inspect_checkpoint(directory)

    def test_main_rejects_invalid_input_without_predicting(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "input.json"; out = base / "out.json"
            source.write_text('{"task":"contract_law_entailment","items":[]}', encoding="utf-8")
            with patch.object(P, "predict", side_effect=AssertionError("must not call model")):
                self.assertEqual(P.main(["--input-json", str(source), "--out", str(out)]), 2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
