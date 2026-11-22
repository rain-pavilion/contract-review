"""CPU-only contracts for the optional, weakly supervised Mamba checkpoint.

These tests never load model weights, initialize CUDA, or present synthetic
labels as independent human annotation.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from tools.mamba_entail_synthetic_data import build_dataset
from zhilv_qiling_v2 import bert_entail as B


class GroupedSyntheticDataTest(unittest.TestCase):
    def test_counts_and_three_labels_in_each_split(self):
        data = build_dataset()
        self.assertEqual(len(data["rows"]), 288)
        for split, total in (("train", 216), ("valid", 36), ("test", 36)):
            self.assertEqual(len(data[split]), total)
            self.assertEqual(data["counts"][split], {label: total // 3 for label in ("entail", "contradict", "neutral")})

    def test_source_template_and_group_are_disjoint(self):
        data = build_dataset()
        for field in ("source_id", "template_family", "group_id"):
            groups = {split: {row[field] for row in data[split]} for split in ("train", "valid", "test")}
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
                self.assertFalse(groups[left] & groups[right], (field, left, right))

    def test_reproducible_hash_and_distinct_pairs(self):
        first = build_dataset()
        second = build_dataset()
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["sha256"], "d13984bee93eb01d3400260c90b557196b537662f2dfbe67647e776cf5541f01")
        self.assertEqual(len({(row["clause"], row["article"]) for row in first["rows"]}), 288)
        self.assertNotEqual(first["group_split"], build_dataset(seed=1)["group_split"])

    def test_all_rows_explicitly_mark_synthetic_and_not_human_gold(self):
        data = build_dataset()
        self.assertIn("Not independent human gold", data["limitation"])
        for row in data["rows"]:
            self.assertTrue(row["article"].startswith("合成规范："))
            self.assertEqual(row["supervision"], "synthetic_weak_supervision_not_human_gold")


class StrictCheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        os.environ.pop("ZLQL_ENTAILMENT_MODEL_DIR", None)
        self.temp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp.name)
        (self.path / "model.pt").write_bytes(b"existence sentinel; never deserialized")
        self.model = Mock()
        self.model.mamba.config.to_dict.return_value = {"hidden_size": 64, "state_size": 8}
        self.tokenizer = object()
        self.auto_tokenizer = Mock()
        self.auto_tokenizer.from_pretrained.return_value = self.tokenizer
        self.transformers = types.SimpleNamespace(AutoTokenizer=self.auto_tokenizer)

    def tearDown(self):
        self.temp.cleanup()
        self.env.stop()

    def load(self, state, meta=None):
        if meta is not None:
            (self.path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        torch = Mock()
        torch.load.return_value = state
        with patch.dict(sys.modules, {"transformers": self.transformers}), \
             patch.object(B, "_torch", return_value=torch), \
             patch.object(B, "build_model", return_value=self.model) as builder, \
             patch.object(B, "inference_device", return_value="cpu"):
            result = B.load_model(self.path)
        return result, builder, torch

    def test_legacy_checkpoint_does_not_get_random_mamba_branch(self):
        state = {"encoder.weight": object(), "rule_head.weight": object()}
        (model, tokenizer, meta), builder, torch = self.load(state)
        builder.assert_called_once_with(enable_mamba=False, enable_rule_head=True, mamba_config=None, dropout=.15)
        self.model.load_state_dict.assert_called_once_with(state, strict=True)
        self.model.to.assert_called_once_with("cpu")
        torch.load.assert_called_once_with(self.path / "model.pt", map_location="cpu", weights_only=True)
        self.assertIs(model, self.model)
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual(meta["loaded_architecture"], {"mamba": False, "rule_residual": True, "strict_checkpoint_load": True, "mamba_config": None})
        self.assertEqual(meta["loaded_checkpoint"]["directory"], str(self.path.resolve()))
        self.assertEqual(meta["loaded_checkpoint"]["selection"], "explicit_argument")

    def test_saved_small_mamba_architecture_is_reconstructed_strictly(self):
        config = {"hidden_size": 64, "num_hidden_layers": 1, "state_size": 8, "vocab_size": 1}
        state = {"encoder.weight": object(), "mamba.layers.0.mixer.A_log": object()}
        (_, _, meta), builder, _ = self.load(state, {"architecture": {"mamba": True, "mamba_config": config, "dropout": .1}})
        builder.assert_called_once_with(enable_mamba=True, enable_rule_head=False, mamba_config=config, dropout=.1)
        self.model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertTrue(meta["loaded_architecture"]["mamba"])
        self.assertEqual(meta["loaded_architecture"]["mamba_config"]["hidden_size"], 64)

    def test_metadata_cannot_claim_mamba_without_parameters(self):
        with self.assertRaisesRegex(RuntimeError, "parameters are missing"):
            self.load({"encoder.weight": object()}, {"architecture": {"mamba": True}})
        self.model.load_state_dict.assert_not_called()

    def test_strict_state_mismatch_is_not_swallowed(self):
        self.model.load_state_dict.side_effect = RuntimeError("Missing key in state_dict")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load({"mamba.layers.0.mixer.A_log": object()})
        self.model.to.assert_not_called()

    def test_default_checkpoint_location_remains_legacy_bert(self):
        self.assertEqual(B.OUTPUT_DIR.name, "bert_entail")
        with patch.object(B, "load_model", return_value=(self.model, self.tokenizer, {})) as loader, \
             patch.object(B, "Predictor", return_value="sentinel"), \
             patch.object(B, "_PREDICTOR", {}):
            self.assertEqual(B.load_predictor(), "sentinel")
        loader.assert_called_once_with(None)

    def test_missing_explicit_checkpoint_does_not_substitute_legacy(self):
        with patch.object(B, "_torch", return_value=Mock()), \
             patch.dict(sys.modules, {"transformers": self.transformers}), \
             patch.object(B, "build_model") as builder:
            self.assertEqual(B.load_model(self.path / "absent"), (None, None, {}))
        builder.assert_not_called()
        self.auto_tokenizer.from_pretrained.assert_not_called()

    def test_environment_selects_exact_directory_and_available_does_not_fallback(self):
        missing = self.path / "explicitly-missing"
        os.environ["ZLQL_ENTAILMENT_MODEL_DIR"] = str(missing)
        self.assertEqual(B.selected_model_dir(), missing.resolve())
        with patch.object(B, "_weights_dir", return_value=self.path):
            self.assertFalse(B.available())
        with patch.object(B, "load_model", return_value=(None, None, {})) as loader, patch.object(B, "_PREDICTOR", {}):
            with self.assertRaisesRegex(FileNotFoundError, "explicitly-missing"):
                B.load_predictor()
        loader.assert_called_once_with(None)

    def test_explicit_argument_overrides_environment(self):
        os.environ["ZLQL_ENTAILMENT_MODEL_DIR"] = str(self.path / "environment")
        self.assertEqual(B.selected_model_dir(self.path), self.path.resolve())

    def test_successful_predictors_cached_per_resolved_path(self):
        first, second = object(), object()
        with patch.object(B, "load_model", return_value=(self.model, self.tokenizer, {})) as loader, \
             patch.object(B, "Predictor", side_effect=[first, second]) as factory, \
             patch.object(B, "_PREDICTOR", {}):
            self.assertIs(B.load_predictor(self.path), first)
            self.assertIs(B.load_predictor(self.path / "."), first)
            self.assertIs(B.load_predictor(self.path / "another"), second)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(factory.call_count, 2)

    def test_missing_checkpoint_failure_is_not_cached(self):
        with patch.object(B, "load_model", side_effect=[(None, None, {}), (self.model, self.tokenizer, {})]) as loader, \
             patch.object(B, "Predictor", return_value="recovered"), patch.object(B, "_PREDICTOR", {}) as cache:
            with self.assertRaises(FileNotFoundError):
                B.load_predictor(self.path)
            self.assertEqual(cache, {})
            self.assertEqual(B.load_predictor(self.path), "recovered")
        self.assertEqual(loader.call_count, 2)

    def test_predictor_initialization_failure_is_not_cached(self):
        with patch.object(B, "load_model", return_value=(self.model, self.tokenizer, {})), \
             patch.object(B, "Predictor", side_effect=[RuntimeError("device unavailable"), "recovered"]), \
             patch.object(B, "_PREDICTOR", {}) as cache:
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                B.load_predictor(self.path)
            self.assertEqual(cache, {})
            self.assertEqual(B.load_predictor(self.path), "recovered")

    def test_weak_supervision_and_architecture_visible_in_existing_runtime_metadata(self):
        from zhilv_qiling_v2 import entailment
        state = {"mamba.layers.0.mixer.A_log": object()}
        (_, _, meta), _, _ = self.load(state, {"architecture": {"mamba": True},
            "supervision": "synthetic_weak_supervision_not_human_gold",
            "training_scope": "lightweight_actual_training_not_sufficient_training_or_human_gold"})
        predictor = types.SimpleNamespace(meta=meta)
        with patch.object(entailment, "_BERT", {"predictor": predictor, "error": "", "reason": "", "load_seconds": 0}), \
             patch.object(entailment, "backend", return_value="bert"):
            runtime = entailment.runtime_status()
        self.assertEqual(runtime["bert_metadata"]["loaded_checkpoint"]["directory"], str(self.path.resolve()))
        self.assertTrue(runtime["bert_metadata"]["loaded_checkpoint"]["weak_supervision"])
        self.assertFalse(runtime["bert_metadata"]["loaded_checkpoint"]["human_gold"])
        self.assertTrue(runtime["bert_metadata"]["loaded_architecture"]["mamba"])


if __name__ == "__main__":
    unittest.main()
