"""Same-review neural reuse preserves provenance and all pending semantics."""
from __future__ import annotations

import contextvars
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from zhilv_qiling_v2 import entailment as E, model_audit as A
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import Contract, DocumentParse, ReviewReport
from zhilv_qiling_v2.tests_v2.test_entailment_windows import fake_predictor, prediction


ARTICLE = {"article_uid": "law-test", "article_no": "测试条", "content": "仅作协议测试的规范文本。"}
CLAUSE = "仅作协议测试的合同文本。"


def output(**extra):
    return {"label": "entail", "score": .9,
            "probabilities": {"entail": .9, "neutral": .05, "contradict": .05},
            "window_audit": {"call_id": "original-forward", "executed_pairs": 1,
                             "planned_pairs": 1, "predictions": [{"label": "entail"}]}, **extra}


class EntailmentRunCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"ZLQL_MODEL_AUDIT_PATH": str(Path(self.tmp.name) / "audit.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        self.predictor = SimpleNamespace(predict=Mock(side_effect=lambda *args: output()),
                                         device="test-cpu", max_window_pairs=64, meta={"checkpoint": "test"})
        self.rule = Mock()
        self.rule.predict.return_value = ("entail", .9, .5)
        for patcher in (patch.object(E, "backend", return_value="bert"),
                        patch.object(E, "_bert_predictor", side_effect=lambda: self.predictor),
                        patch.object(E, "load_classifier", return_value=self.rule)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def judge(self, clause=CLAUSE, article=None):
        return E.judge_pair(clause, ARTICLE if article is None else article)

    def test_real_window_path_records_one_forward_and_separate_reuse(self):
        self.predictor = fake_predictor()
        with patch.object(self.predictor, "_run_window", return_value=prediction("test")) as forward:
            with A.review_audit("one-review"), E.review_prediction_cache() as cache:
                first, second = self.judge(), self.judge()
                stats = cache.summary()
        events = A.recent_events(trace_id="one-review")
        forwarded = [e for e in events if e["operation"] == "predict_window" and e["status"] == "succeeded"]
        pair_calls = [e for e in events if e["operation"] == "predict_pair"]
        reused = [e for e in events if e["operation"] == "prediction_cache_reuse"]
        self.assertEqual(forward.call_count, 1)
        self.assertEqual((len(forwarded), len(pair_calls), len(reused)), (1, 1, 1))
        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(reused[0]["status"], "reused")
        self.assertEqual(reused[0]["details"]["call_id"], forwarded[0]["details"]["call_id"])
        self.assertEqual(reused[0]["details"]["executed_pairs"], 0)
        self.assertEqual((stats["prediction_calls"], stats["cache_reuses"]), (1, 1))
        self.assertNotIn(CLAUSE, str(events))

    def test_exact_clause_and_article_content_changes_do_not_reuse(self):
        with E.review_prediction_cache():
            self.judge()
            self.judge(CLAUSE + " ")
            self.judge(article={**ARTICLE, "content": ARTICLE["content"] + " "})
            self.judge()
        self.assertEqual(self.predictor.predict.call_count, 3)

    def test_predictor_identity_and_inference_options_invalidate(self):
        original = self.predictor
        with E.review_prediction_cache():
            self.judge()
            self.predictor = copy.copy(original)
            self.judge()
            self.predictor.max_window_pairs = 2
            self.judge()
            self.predictor.meta = {"checkpoint": "changed"}
            self.judge()
            with patch.dict(os.environ, {"ZLQL_ENTAILMENT_MAX_WINDOW_PAIRS": "3"}):
                self.judge()
        self.assertEqual(original.predict.call_count, 5)

    def test_new_scopes_and_calls_outside_scope_never_share(self):
        for _ in range(2):
            with E.review_prediction_cache():
                self.judge()
                self.judge()
        self.judge()
        self.judge()
        self.assertEqual(self.predictor.predict.call_count, 4)

    def test_nested_scopes_restore_the_outer_cache(self):
        with E.review_prediction_cache() as outer:
            self.judge()
            with E.review_prediction_cache() as inner:
                self.judge()
            self.assertEqual(inner.entries, {})
            self.judge()
            self.assertEqual(outer.cache_reuses, 1)
        self.assertEqual(outer.entries, {})
        self.assertEqual(self.predictor.predict.call_count, 2)

    def test_returned_nested_data_cannot_mutate_cached_result(self):
        original = output()
        self.predictor.predict.side_effect = lambda *args: original
        with E.review_prediction_cache():
            first = self.judge()
            first.window_audit["predictions"][0]["label"] = "mutated-first"
            self.assertEqual(original["window_audit"]["predictions"][0]["label"], "entail")
            second = self.judge()
            self.assertEqual(second.window_audit["predictions"][0]["label"], "entail")
            second.window_audit["predictions"][0]["label"] = "mutated-reuse"
            third = self.judge()
            self.assertEqual(third.window_audit["predictions"][0]["label"], "entail")
        self.assertEqual(self.predictor.predict.call_count, 1)

    def test_uncacheable_metadata_does_not_suppress_neural_inference(self):
        cyclic = {}
        cyclic["cycle"] = cyclic
        self.predictor.meta = cyclic
        with E.review_prediction_cache():
            first, second = self.judge(), self.judge()
        self.assertEqual(first.source, "bert_entail")
        self.assertEqual(second.source, "bert_entail")
        self.assertEqual(self.predictor.predict.call_count, 2)

    def test_exception_is_retryable_and_success_can_then_be_reused(self):
        self.predictor.predict.side_effect = [RuntimeError("test failure"), output()]
        with E.review_prediction_cache() as cache:
            failed, succeeded, reused = self.judge(), self.judge(), self.judge()
            self.assertEqual(cache.prediction_failures, 1)
        self.assertTrue(failed.manual_required)
        self.assertTrue(failed.semantic_pending)
        self.assertEqual(asdict(succeeded), asdict(reused))
        self.assertEqual(self.predictor.predict.call_count, 2)

    def test_partial_failed_and_malformed_outputs_are_not_cached(self):
        partial = output(window_audit={"failed_pairs": 1, "executed_pairs": 1, "planned_pairs": 2},
                         pair_coverage_complete=False, semantic_pending=True)
        self.predictor.predict.side_effect = [partial, partial, {"label": "entail"}, output()]
        with E.review_prediction_cache() as cache:
            for _ in range(4):
                self.judge()
            self.assertEqual(cache.uncacheable_results, 3)
        self.assertEqual(self.predictor.predict.call_count, 4)

    def test_pending_and_disagreements_remain_pending_after_reuse(self):
        pending = output(label="contradict", semantic_pending=True, manual_required=True,
                         coverage_complete=False, token_coverage_complete=True,
                         pair_coverage_complete=True, window_disagreement=True,
                         decision_scope="local_window_only")
        self.predictor.predict.side_effect = lambda *args: copy.deepcopy(pending)
        with E.review_prediction_cache():
            first, reused = self.judge(), self.judge()
        self.assertEqual(asdict(first), asdict(reused))
        self.assertTrue(reused.semantic_pending)
        self.assertTrue(reused.manual_required)
        self.assertTrue(reused.disagreement)
        self.assertTrue(reused.window_disagreement)
        self.assertFalse(reused.is_conflict)
        self.assertEqual(self.predictor.predict.call_count, 1)

    def test_rule_and_article_metadata_are_recomputed_per_use(self):
        with E.review_prediction_cache():
            first = self.judge()
            self.rule.predict.return_value = ("contradict", .8, .3)
            second = self.judge(article={**ARTICLE, "article_uid": "another-occurrence", "article_no": "第二条"})
        self.assertFalse(first.disagreement)
        self.assertTrue(second.disagreement)
        self.assertEqual(second.article_uid, "another-occurrence")
        self.assertEqual(second.article_no, "第二条")
        self.assertEqual(self.predictor.predict.call_count, 1)

    def test_copied_concurrent_contexts_share_only_the_current_review(self):
        with E.review_prediction_cache() as cache, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(contextvars.copy_context().run, self.judge) for _ in range(4)]
            verdicts = [future.result() for future in futures]
            self.assertEqual(cache.cache_reuses, 3)
        self.assertEqual(self.predictor.predict.call_count, 1)
        self.assertTrue(all(v.label == "entail" for v in verdicts))

    def test_engine_runs_have_fresh_cache_stats_and_exception_cleanup(self):
        engine = ReviewEngine()
        def review(contract):
            self.judge()
            self.judge()
            if contract.contract_id == "fail":
                raise RuntimeError("review failed")
            return ReviewReport(contract.contract_id, contract.document.source_name)
        def fixture(contract_id):
            return Contract(contract_id, DocumentParse("cache-fixture.txt", "txt", CLAUSE))
        with patch.object(engine, "_run_review", side_effect=review):
            with self.assertRaisesRegex(RuntimeError, "review failed"):
                engine.run(fixture("fail"))
            first = engine.run(fixture("contract-one"))
            second = engine.run(fixture("contract-two"))
        self.assertEqual(self.predictor.predict.call_count, 3)
        self.assertEqual(first.stats["entailment_cache"]["cache_reuses"], 1)
        self.assertEqual(second.stats["entailment_cache"]["prediction_calls"], 1)
        self.assertFalse(second.stats["entailment_cache"]["changes_legal_status"])
        self.assertIsNone(E._PREDICTION_CACHE.get())


if __name__ == "__main__":
    unittest.main()
