"""Fake-only token-window regressions; no checkpoint, endpoint, or quality claim."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

from zhilv_qiling_v2 import bert_entail as B
from zhilv_qiling_v2 import entailment as E
from zhilv_qiling_v2 import entailment_windows as W
from zhilv_qiling_v2 import model_audit as A


def prediction(pair_id: str, label: str = "entail", score: float = .8) -> dict:
    other = (1 - score) / 2
    return {"pair_id": pair_id, "label": label, "score": score,
            "probabilities": {name: score if name == label else other for name in W.LABELS},
            "cosine": .4, "article_tokens_at_peak": ["FAKE_TEST_TOKEN"]}


class FakeTokenizer:
    def __init__(self, *, offsets: bool = True):
        self.offsets = offsets

    def __call__(self, text, *, add_special_tokens=True, truncation=False,
                 return_offsets_mapping=False, **kwargs):
        if return_offsets_mapping and not self.offsets:
            raise NotImplementedError("slow tokenizer has no character offsets")
        ids = [ord(char) for char in text]
        result = {"input_ids": [101] + ids + [102] if add_special_tokens else ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)


def fake_predictor(*, max_pairs=64, offsets=True):
    model = Mock()
    model.to.return_value = model
    return B.Predictor(model=model, tokenizer=FakeTokenizer(offsets=offsets),
                       device="fake_cpu", max_window_pairs=max_pairs)


class ShapeOnlyTensor:
    """Shape/transport double, never a numerical inference implementation."""
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def to(self, device):
        return self


class ActualTokenizerShapeContract(FakeTokenizer):
    """Match installed HF prepare_for_model's explicit batch-axis behavior."""
    def __init__(self):
        super().__init__()
        self.prepare_calls = []

    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        if kwargs.get("return_tensors") == "pt":
            ids = encoded["input_ids"]
            return {"input_ids": ShapeOnlyTensor([ids]),
                    "attention_mask": ShapeOnlyTensor([[1] * len(ids)])}
        return encoded

    def prepare_for_model(self, ids, *, prepend_batch_axis=False, **kwargs):
        self.prepare_calls.append({"ids": list(ids), "prepend_batch_axis": prepend_batch_axis, **kwargs})
        values = [101, *ids, 102]
        mask = [1] * len(values)
        return {"input_ids": ShapeOnlyTensor([values] if prepend_batch_axis else values),
                "attention_mask": ShapeOnlyTensor([mask] if prepend_batch_axis else mask)}


class ForwardBoundaryReached(RuntimeError):
    pass


class BatchStrictForwardProbe:
    """Reproduce BERT's batch/sequence unpack before stopping fake execution."""
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, cids, cmask, aids, amask, **kwargs):
        # These are the same two dimensions required by actual BertModel.forward.
        c_batch, c_seq = cids.shape
        a_batch, a_seq = aids.shape
        if (c_batch, a_batch) != (1, 1) or cmask.shape != cids.shape or amask.shape != aids.shape:
            raise ValueError("invalid_batch_or_attention_mask_shape")
        self.calls.append((cids.values, cmask.values, aids.values, amask.values))
        raise ForwardBoundaryReached("transport_reached_fake_forward_boundary")


class ShapeOnlyTorch:
    float32 = "fake_float32"

    @staticmethod
    def tensor(values, **kwargs):
        return ShapeOnlyTensor(values)

    @staticmethod
    def no_grad():
        return nullcontext()


class WindowPlanningTest(unittest.TestCase):
    def test_id_windows_overlap_and_include_exact_tail(self):
        ids = list(range(17))
        windows = W.build_windows(ids, 5, prefix="c", overlap=2,
                                  offsets=[(i * 2, i * 2 + 1) for i in ids])
        self.assertEqual([(w.token_start, w.token_end) for w in windows],
                         [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17)])
        self.assertEqual(windows[-1].token_ids, tuple(ids[-5:]))
        self.assertEqual(windows[-1].char_end, 33)
        self.assertTrue(all(w.token_ids == tuple(ids[w.token_start:w.token_end]) for w in windows))

    def test_invalid_scopes_fail_closed(self):
        for capacity, overlap in [(0, 0), (5, 5), (5, -1)]:
            with self.subTest(capacity=capacity, overlap=overlap), self.assertRaises(ValueError):
                W.build_windows([1], capacity, prefix="c", overlap=overlap)
        with self.assertRaisesRegex(ValueError, "offset_count"):
            W.build_windows([1, 2], 4, prefix="c", overlap=0, offsets=[(0, 1)])

    def test_cartesian_plan_has_each_pair_exactly_once(self):
        plan = W.make_plan(list(range(10)), list(range(13)), clause_capacity=5,
                           article_capacity=5, overlap=0)
        self.assertEqual(plan.planned_pair_count, 6)
        self.assertEqual(len(plan.scheduled_pairs), 6)
        self.assertEqual(len({p.pair_id for p in plan.scheduled_pairs}), 6)
        audit = W.summarize(plan, [prediction(p.pair_id) for p in plan.scheduled_pairs])
        self.assertTrue(audit["token_coverage_complete"])
        self.assertTrue(audit["pair_coverage_complete"])
        self.assertTrue(audit["semantic_pending"])
        self.assertEqual(audit["unexecuted_pair_ids"], [])

    def test_full_token_coverage_does_not_imply_complete_pairs_or_semantics(self):
        plan = W.make_plan(list(range(10)), list(range(10)), clause_capacity=5,
                           article_capacity=5, overlap=0, max_pairs=2)
        audit = W.summarize(plan, [prediction(p.pair_id) for p in plan.scheduled_pairs])
        self.assertTrue(audit["token_coverage_complete"])
        self.assertFalse(audit["pair_coverage_complete"])
        self.assertEqual(audit["planned_pairs"], 4)
        self.assertEqual(len(audit["unexecuted_pair_ids"]), 2)
        self.assertIn("window_pair_budget_exhausted", audit["semantic_pending_reasons"])
        self.assertTrue(audit["manual_required"])

    def test_insufficient_budget_includes_tail_and_exposes_middle_gaps(self):
        plan = W.make_plan(list(range(30)), [1], clause_capacity=5,
                           article_capacity=5, overlap=0, max_pairs=2)
        audit = W.summarize(plan, [prediction(p.pair_id) for p in plan.scheduled_pairs])
        self.assertTrue(audit["clause_coverage"]["tail_covered"])
        self.assertFalse(audit["token_coverage_complete"])
        self.assertEqual(audit["clause_coverage"]["missing_ranges"], [[5, 25]])
        self.assertEqual(audit["clause_coverage"]["covered_tokens"], 10)

    def test_two_pair_budget_attempts_both_tails_with_unequal_window_counts(self):
        plan = W.make_plan(list(range(20)), list(range(35)), clause_capacity=5,
                           article_capacity=5, overlap=0, max_pairs=2)
        self.assertEqual([pair.pair_id for pair in plan.scheduled_pairs],
                         ["c0000:a0000", "c0003:a0006"])
        audit = W.summarize(plan, [prediction(pair.pair_id) for pair in plan.scheduled_pairs])
        self.assertTrue(audit["clause_coverage"]["tail_covered"])
        self.assertTrue(audit["article_coverage"]["tail_covered"])
        self.assertFalse(audit["token_coverage_complete"])
        self.assertTrue(audit["semantic_pending"])

    def test_unanimous_local_entail_is_still_semantically_pending(self):
        plan = W.make_plan(list(range(10)), [1], clause_capacity=5,
                           article_capacity=5, overlap=0)
        audit = W.summarize(plan, [prediction(p.pair_id) for p in plan.scheduled_pairs])
        self.assertEqual(audit["selected_prediction"]["label"], "entail")
        self.assertFalse(audit["window_disagreement"])
        self.assertEqual(audit["decision_scope"], "local_window_only")
        self.assertTrue(audit["semantic_pending"])
        self.assertTrue(audit["manual_required"])

    def test_disagreement_preserves_a_real_local_contradiction_without_vote(self):
        plan = W.make_plan(list(range(20)), [1], clause_capacity=5,
                           article_capacity=5, overlap=0)
        outputs = [prediction(p.pair_id, "entail", .99) for p in plan.scheduled_pairs]
        outputs[-1] = prediction(plan.scheduled_pairs[-1].pair_id, "contradict", .6)
        audit = W.summarize(plan, outputs)
        self.assertTrue(audit["window_disagreement"])
        self.assertEqual(audit["selected_prediction"], outputs[-1])
        self.assertEqual(audit["local_labels"], ["contradict", "entail"])
        self.assertTrue(audit["semantic_pending"])

    def test_unknown_or_duplicate_results_cannot_fake_coverage(self):
        plan = W.make_plan([1], [2], clause_capacity=5, article_capacity=5, overlap=0)
        output = prediction(plan.scheduled_pairs[0].pair_id)
        with self.assertRaisesRegex(ValueError, "unknown_or_duplicate"):
            W.summarize(plan, [output, output])
        with self.assertRaisesRegex(ValueError, "unknown_or_duplicate"):
            W.summarize(plan, [prediction("not-scheduled")])

    def test_missing_character_offsets_are_explicit_not_guessed(self):
        window = W.build_windows([5, 6], 3, prefix="c", overlap=0)[0].audit()
        self.assertIsNone(window["char_start"])
        self.assertIsNone(window["char_end"])
        self.assertFalse(window["offsets_available"])

    def test_very_large_pair_plan_does_not_materialize_all_missing_ids(self):
        plan = W.make_plan(list(range(500)), list(range(500)), clause_capacity=5,
                           article_capacity=5, overlap=0, max_pairs=2)
        audit = W.summarize(plan, [prediction(p.pair_id) for p in plan.scheduled_pairs])
        self.assertEqual(audit["planned_pairs"], 10000)
        self.assertIsNone(audit["unexecuted_pair_ids"])
        self.assertIn("except executed_pair_ids", audit["unexecuted_pair_id_rule"])

    def test_empty_source_is_invalid_and_cannot_claim_model_output(self):
        for clause_ids, article_ids in (([], [1]), ([1], []), ([], [])):
            with self.subTest(clause=clause_ids, article=article_ids):
                plan = W.make_plan(clause_ids, article_ids, clause_capacity=5,
                                   article_capacity=5, overlap=0)
                audit = W.summarize(plan, [])
                self.assertFalse(audit["input_valid"])
                self.assertFalse(audit["token_coverage_complete"])
                self.assertTrue(audit["semantic_pending"])
                self.assertTrue(audit["manual_required"])
                self.assertEqual(audit["decision_scope"], "invalid_input")
                self.assertIsNone(audit["selected_prediction"])
                with self.assertRaisesRegex(ValueError, "empty_token_input"):
                    W.summarize(plan, [prediction(plan.scheduled_pairs[0].pair_id)])


class PredictorWindowTest(unittest.TestCase):
    def test_shape_double_reproduces_installed_prepare_for_model_default_1d(self):
        tokenizer = ActualTokenizerShapeContract()
        encoded = tokenizer.prepare_for_model([7, 8], return_tensors="pt")
        self.assertEqual(encoded["input_ids"].shape, (4,))
        self.assertEqual(encoded["attention_mask"].shape, (4,))

    def test_run_window_itself_adds_required_batch_axis_without_mocking_encoding_layer(self):
        tokenizer = ActualTokenizerShapeContract()
        model = BatchStrictForwardProbe()
        predictor = B.Predictor(model=model, tokenizer=tokenizer, device="fake_cpu")
        with patch.object(B, "_torch", return_value=ShapeOnlyTorch()), \
             self.assertRaises(ForwardBoundaryReached):
            predictor._run_window((7, 8, 9), (11, 12), {name: 0. for name in E.FEATURES})
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([row["prepend_batch_axis"] for row in tokenizer.prepare_calls], [True, True])
        self.assertTrue(all(row["truncation"] is False for row in tokenizer.prepare_calls))
        self.assertEqual(model.calls[0], ([[101, 7, 8, 9, 102]], [[1] * 5],
                                          [[101, 11, 12, 102]], [[1] * 4]))

    def test_run_window_single_input_still_uses_original_batched_tokenizer_call(self):
        tokenizer = ActualTokenizerShapeContract()
        model = BatchStrictForwardProbe()
        predictor = B.Predictor(model=model, tokenizer=tokenizer, device="fake_cpu")
        with patch.object(B, "_torch", return_value=ShapeOnlyTorch()), \
             self.assertRaises(ForwardBoundaryReached):
            predictor._run_window((ord("甲"),), (ord("法"),), {name: 0. for name in E.FEATURES},
                                  full_clause="甲", full_article="法")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(tokenizer.prepare_calls, [])
        self.assertEqual(model.calls[0][0], [[101, ord("甲"), 102]])
        self.assertEqual(model.calls[0][2], [[101, ord("法"), 102]])

    def test_single_window_keeps_original_full_text_encoding_path_and_score(self):
        predictor = fake_predictor()
        run = Mock(return_value=prediction("unused", "neutral", .91))
        with patch.object(predictor, "_run_window", run), patch.object(A, "record") as receipt:
            output = predictor.predict("甲方支付价款。", "当事人应当履行合同义务。")
        self.assertEqual(output["label"], "neutral")
        self.assertEqual(output["score"], .91)
        self.assertEqual(run.call_args.kwargs["full_clause"], "甲方支付价款。")
        self.assertEqual(run.call_args.kwargs["full_article"], "当事人应当履行合同义务。")
        self.assertEqual(output["clause_tokens"], len("甲方支付价款。") + 2)
        self.assertFalse(output["semantic_pending"])
        self.assertFalse(output["manual_required"])
        self.assertTrue(output["coverage_complete"])
        self.assertEqual(output["decision_scope"], "whole_input")
        self.assertEqual([call.args[3] for call in receipt.call_args_list], ["started", "succeeded"])

    def test_long_inputs_execute_every_original_token_window_pair(self):
        predictor = fake_predictor()
        clause = "".join(chr(0x4e00 + i) for i in range(500))
        article = "".join(chr(0x6000 + i) for i in range(600))
        seen = []

        def run(cids, aids, features, **kwargs):
            self.assertLessEqual(len(cids) + 2, B.MAX_CLAUSE)
            self.assertLessEqual(len(aids) + 2, B.MAX_ARTICLE)
            self.assertIsNone(kwargs["full_clause"])
            self.assertIsNone(kwargs["full_article"])
            seen.append((tuple(cids), tuple(aids), features))
            return prediction("unused")

        with patch.object(predictor, "_run_window", side_effect=run), patch.object(A, "record"):
            output = predictor.predict(clause, article)
        audit = output["window_audit"]
        self.assertEqual(len(seen), audit["planned_pairs"])
        self.assertGreater(len(seen), 1)
        self.assertTrue(output["token_coverage_complete"])
        self.assertTrue(output["pair_coverage_complete"])
        self.assertFalse(output["input_truncated"])
        self.assertFalse(output["coverage_complete"])
        self.assertTrue(output["semantic_pending"])
        for (cids, aids, features), result in zip(seen, audit["window_predictions"]):
            cs, ars = result["clause_scope"], result["article_scope"]
            ctext = clause[cs["token_start"]:cs["token_end"]]
            atext = article[ars["token_start"]:ars["token_end"]]
            self.assertEqual(cids, tuple(map(ord, ctext)))
            self.assertEqual(aids, tuple(map(ord, atext)))
            self.assertEqual(features, E.clause_features(ctext, atext, risk_tags=(), codes=()))
        self.assertTrue(any(cids[-1] == ord(clause[-1]) for cids, _, _ in seen))
        self.assertTrue(any(aids[-1] == ord(article[-1]) for _, aids, _ in seen))

    def test_slow_tokenizer_retains_token_ranges_and_marks_offsets_unavailable(self):
        predictor = fake_predictor(offsets=False)
        with patch.object(predictor, "_run_window", return_value=prediction("unused")), patch.object(A, "record"):
            output = predictor.predict("甲" * 400, "法律条文")
        self.assertTrue(output["token_coverage_complete"])
        for result in output["window_audit"]["window_predictions"]:
            self.assertIsNone(result["clause_scope"]["char_start"])
            self.assertEqual(result["feature_scope"]["clause"], "decoded_original_token_ids")

    def test_budget_shortfall_is_explicit_and_never_rewritten_as_completion(self):
        predictor = fake_predictor(max_pairs=2)
        with patch.object(predictor, "_run_window", return_value=prediction("unused")), patch.object(A, "record"):
            output = predictor.predict("甲" * 1500, "条文")
        self.assertEqual(output["window_audit"]["executed_pairs"], 2)
        self.assertTrue(output["window_audit"]["clause_coverage"]["tail_covered"])
        self.assertFalse(output["token_coverage_complete"])
        self.assertFalse(output["pair_coverage_complete"])
        self.assertTrue(output["input_truncated"])
        self.assertTrue(output["semantic_pending"])
        self.assertTrue(output["manual_required"])

    def test_midway_failure_retains_actual_outputs_and_unexecuted_ids(self):
        predictor = fake_predictor()
        run = Mock(side_effect=[prediction("unused", "contradict"), RuntimeError("PRIVATE_EXCEPTION_BODY")])
        with patch.object(predictor, "_run_window", run), patch.object(A, "record") as receipt:
            output = predictor.predict("甲" * 600, "条文")
        audit = output["window_audit"]
        self.assertEqual(output["label"], "contradict")
        self.assertEqual(audit["executed_pairs"], 1)
        self.assertEqual(len(audit["failed_pairs"]), 1)
        self.assertFalse(output["coverage_complete"])
        self.assertTrue(output["semantic_pending"])
        self.assertTrue(audit["unexecuted_pair_ids"])
        self.assertNotIn("PRIVATE_EXCEPTION_BODY", json.dumps(audit))
        self.assertEqual([call.args[3] for call in receipt.call_args_list],
                         ["started", "succeeded", "started", "failed"])

    def test_first_window_failure_retains_auditable_exception_without_private_error(self):
        predictor = fake_predictor()
        with patch.object(predictor, "_run_window", side_effect=RuntimeError("PRIVATE_EXCEPTION_BODY")), \
             patch.object(A, "record"), self.assertRaises(B.WindowInferenceError) as caught:
            predictor.predict("甲" * 400, "条文")
        audit = caught.exception.window_audit
        self.assertEqual(audit["executed_pairs"], 0)
        self.assertEqual(len(audit["failed_pairs"]), 1)
        self.assertTrue(audit["semantic_pending"])
        self.assertNotIn("PRIVATE_EXCEPTION_BODY", str(caught.exception) + json.dumps(audit))

    def test_invalid_environment_budget_cannot_silently_skip_inference(self):
        predictor = fake_predictor(max_pairs=None)
        with patch.dict(os.environ, {"ZLQL_ENTAILMENT_MAX_WINDOW_PAIRS": "0"}), \
             patch.object(predictor, "_run_window") as run, self.assertRaises(ValueError):
            predictor.predict("甲", "条文")
        run.assert_not_called()

    def test_empty_tokens_skip_model_and_preserve_invalid_pending_audit(self):
        for clause, article in (("", "法条"), ("条款", ""), ("", "")):
            predictor = fake_predictor()
            with self.subTest(clause=clause, article=article), \
                 patch.object(predictor, "_run_window") as run, patch.object(A, "record") as receipt, \
                 self.assertRaises(B.WindowInferenceError) as caught:
                predictor.predict(clause, article)
            run.assert_not_called()
            audit = caught.exception.window_audit
            self.assertFalse(audit["input_valid"])
            self.assertTrue(audit["manual_required"])
            self.assertTrue(audit["semantic_pending"])
            self.assertEqual(audit["executed_pairs"], 0)
            self.assertIsNone(audit["selected_prediction"])
            self.assertEqual(caught.exception.reason_code, "bert_empty_token_input")
            self.assertEqual(receipt.call_args.args[2:4], ("validate_input", "skipped"))


class VerdictWindowTest(unittest.TestCase):
    def judge(self, output):
        predictor = Mock()
        predictor.predict.return_value = output
        classifier = Mock()
        classifier.predict.return_value = (output["label"], output["score"], .5)
        with patch.object(E, "backend", return_value="bert"), \
             patch.object(E, "_bert_predictor", return_value=predictor), \
             patch.object(E, "load_classifier", return_value=classifier), patch.object(A, "record"):
            return E.judge_pair("甲方条款", {"content": "法条全文", "article_no": "一"})

    def test_legacy_one_window_output_remains_compatible(self):
        verdict = self.judge(prediction("test", "contradict", .9))
        self.assertEqual(verdict.label, "contradict")
        self.assertTrue(verdict.is_conflict)
        self.assertFalse(verdict.semantic_pending)
        self.assertFalse(verdict.manual_required)

    def test_mult_window_contradiction_is_real_local_signal_not_confirmed_global_conflict(self):
        verdict = self.judge({**prediction("test", "contradict", .9),
                              "token_coverage_complete": True, "pair_coverage_complete": True,
                              "coverage_complete": False, "semantic_pending": True,
                              "manual_required": True, "window_disagreement": True,
                              "decision_scope": "local_window_only",
                              "window_audit": {"planned_pairs": 4, "executed_pairs": 4}})
        self.assertEqual(verdict.label, "contradict")
        self.assertEqual(verdict.neural_label, "contradict")
        self.assertFalse(verdict.is_conflict)
        self.assertTrue(verdict.token_coverage_complete)
        self.assertTrue(verdict.pair_coverage_complete)
        self.assertFalse(verdict.coverage_complete)
        self.assertTrue(verdict.manual_required)
        self.assertTrue(verdict.window_disagreement)
        self.assertIn("不能证明全文关系", verdict.note)
        self.assertIn("局部窗口信号", verdict.evidence_text())

    def test_neural_failure_rules_fallback_is_still_pending(self):
        predictor = Mock()
        predictor.predict.side_effect = B.WindowInferenceError({"executed_pairs": 0,
                                                               "token_coverage_complete": False,
                                                               "pair_coverage_complete": False})
        classifier = Mock()
        classifier.predict.return_value = ("contradict", .95, .8)
        with patch.object(E, "backend", return_value="bert"), \
             patch.object(E, "_bert_predictor", return_value=predictor), \
             patch.object(E, "load_classifier", return_value=classifier), patch.object(A, "record"):
            verdict = E.judge_pair("合同违约应当赔偿", {"content": "违约应当赔偿", "article_no": "一"})
        self.assertEqual(verdict.source, "rules")
        self.assertTrue(verdict.semantic_pending)
        self.assertTrue(verdict.manual_required)
        self.assertFalse(verdict.coverage_complete)
        self.assertFalse(verdict.is_conflict)
        self.assertEqual(verdict.decision_scope, "rules_fallback_pending")

    def test_any_manual_or_disagreement_flag_blocks_confirmed_conflict(self):
        for flag in ("manual_required", "disagreement", "window_disagreement", "semantic_pending"):
            with self.subTest(flag=flag):
                verdict = E.EntailmentVerdict("one", "一", "contradict", .99, **{flag: True})
                self.assertFalse(verdict.is_conflict)
        self.assertTrue(E.EntailmentVerdict("one", "一", "contradict", .99).is_conflict)

    def test_single_window_neural_rule_disagreement_is_not_confirmed_conflict(self):
        predictor = Mock()
        predictor.predict.return_value = prediction("one", "contradict", .99)
        classifier = Mock()
        classifier.predict.return_value = ("neutral", .8, .5)
        with patch.object(E, "backend", return_value="bert"), \
             patch.object(E, "_bert_predictor", return_value=predictor), \
             patch.object(E, "load_classifier", return_value=classifier), patch.object(A, "record"):
            verdict = E.judge_pair("合同违约应当赔偿", {"content": "违约应当赔偿", "article_no": "一"})
        self.assertEqual(verdict.label, "contradict")
        self.assertTrue(verdict.disagreement)
        self.assertTrue(verdict.manual_required)
        self.assertFalse(verdict.is_conflict)

    def test_invalid_input_has_no_fabricated_neural_label_and_cannot_be_confirmed(self):
        predictor = Mock()
        predictor.predict.side_effect = B.WindowInferenceError(
            {"input_valid": False, "token_coverage_complete": False,
             "pair_coverage_complete": False, "executed_pairs": 0}, "bert_empty_token_input")
        classifier = Mock()
        classifier.predict.return_value = ("contradict", .95, .8)
        with patch.object(E, "backend", return_value="bert"), \
             patch.object(E, "_bert_predictor", return_value=predictor), \
             patch.object(E, "load_classifier", return_value=classifier), patch.object(A, "record"):
            verdict = E.judge_pair("", {"content": "法条", "article_no": "一"})
        self.assertEqual(verdict.source, "rules")
        self.assertEqual(verdict.neural_label, "")
        self.assertEqual(verdict.neural_score, 0.)
        self.assertEqual(verdict.decision_scope, "invalid_input")
        self.assertTrue(verdict.semantic_pending)
        self.assertTrue(verdict.manual_required)
        self.assertFalse(verdict.coverage_complete)
        self.assertFalse(verdict.is_conflict)


if __name__ == "__main__":
    unittest.main()
