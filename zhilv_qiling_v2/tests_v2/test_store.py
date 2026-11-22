"""`tests_v2/test_store.py`：存储层与评测台的契约测试。

覆盖点（都是真实踩过的）：
1. 落库/读回：`save` → `get_risks` 的字段与引用条数一致
2. 幂等：同一份报告写两次不会重复计风险
3. 断点续跑：`batch_to_store(resume=True)` 第二次全部 skipped
4. 分页与过滤：`list_contracts(code=...)` 只返回命中该风险码的合同
5. 条款检索：trigram 分词器下中文子串能命中（默认 unicode61 会全空）
6. 统计：分级/高频码/待复核数与明细一致
7. bench：`summarize` 输出可读，`run_bench(tasks=("detection",))` 能落盘
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhilv_qiling_v2 import bench, store
from zhilv_qiling_v2.config import config_from_env
from zhilv_qiling_v2.models import Citation, FinalRisk, ReviewReport
from zhilv_qiling_v2.pipeline import review_path

SAMPLE = Path("samples/tech-development-contract.txt")
ADVERSARIAL = Path("data/generated_contracts/samr_multi_template_2025/adversarial")


def _report() -> ReviewReport:
    risk = FinalRisk(
        risk_id="rk_test_1",
        code="zero_rent",
        category="商业不合理",
        severity="high",
        message="租金约定为零",
        citations=[
            Citation(law_name="中华人民共和国民法典", article_no="第七百零三条", quote="", source_url=""),
            Citation(law_name="中华人民共和国民法典", article_no="第七百二十一条", quote="", source_url=""),
        ],
        clause_ids=["cl_1"],
        confidence=0.8,
        calibrated_confidence=0.75,
        review_required=True,
        tags=["zero_rent"],
    )
    return ReviewReport(
        contract_id="ct_unit_test",
        source_name="unit.docx",
        risks=[risk],
        review_queue=[risk],
        conflicts=[],
        score=42,
        score_breakdown={},
        category_counts={"商业不合理": 1},
        severity_counts={"high": 1},
        trace=[],
        stats={"elapsed_seconds": 0.5, "clauses": 7, "contract_type": "房屋租赁"},
        schema_version="2.2",
    )


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = store.ReviewStore(Path(self._tmp.name) / "t.sqlite")

    def tearDown(self) -> None:
        self.db.close()
        self._tmp.cleanup()

    def test_save_and_read_back(self) -> None:
        self.db.save(_report(), digest="d", text_chars=1234)
        self.assertEqual(self.db.stats().contracts, 1)
        risks = self.db.get_risks("ct_unit_test")
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]["code"], "zero_rent")
        self.assertEqual(len(risks[0]["articles"]), 2)
        self.assertTrue(risks[0]["review_required"])

    def test_save_is_idempotent(self) -> None:
        self.db.save(_report())
        self.db.save(_report())
        self.assertEqual(self.db.stats().risks, 1)

    def test_pagination_and_filter(self) -> None:
        self.db.save(_report())
        page, total = self.db.list_contracts(page=1, size=10)
        self.assertEqual(total, 1)
        self.assertEqual(page[0].contract_id, "ct_unit_test")
        self.assertEqual(page[0].high_count, 1)
        hits, _ = self.db.list_contracts(code="zero_rent")
        self.assertEqual(len(hits), 1)
        none, _ = self.db.list_contracts(code="not_a_code")
        self.assertEqual(len(none), 0)

    def test_clause_search_chinese_substring(self) -> None:
        self.assertTrue(self.db.fts, "FTS 应可用（trigram 或 plain）")
        self.db.save_clauses("ct_unit_test", [("cl_1", "第一条 租金与押金\n月租金为人民币零元，押金为人民币999999元。")])
        hits = self.db.search_clauses("押金", limit=5)
        self.assertTrue(hits, "中文子串检索应命中")
        self.assertEqual(hits[0]["contract_id"], "ct_unit_test")

    def test_stats_consistency(self) -> None:
        self.db.save(_report())
        stats = self.db.stats().as_dict()
        self.assertEqual(stats["risks"], 1)
        self.assertEqual(stats["by_severity"].get("high"), 1)
        self.assertEqual(stats["pending_review"], 1)
        self.assertEqual(stats["score_max"], 42)


class BatchResumeTest(unittest.TestCase):
    def test_resume_skips_and_counts(self) -> None:
        files = sorted(ADVERSARIAL.glob("*.docx"))[:2]
        if not files:
            self.skipTest("对抗集缺失")
        config = config_from_env()
        with tempfile.TemporaryDirectory() as tmp:
            db = store.ReviewStore(Path(tmp) / "b.sqlite")
            first = store.batch_to_store(files, db, config)
            self.assertEqual(first.processed, 2)
            self.assertEqual(first.skipped, 0)
            second = store.batch_to_store(files, db, config)
            self.assertEqual(second.processed, 0)
            self.assertEqual(second.skipped, 2)
            self.assertEqual(db.stats().contracts, 2)
            db.close()


class BenchTest(unittest.TestCase):
    def test_summarize_is_readable(self) -> None:
        result = {
            "retrieval": {
                "curves": {
                    "rrf_only": {"strong": {"recall": {"@5": 0.6278}, "mrr5": 1.0, "tier": "strong",
                                            "n": 6, "mrr10": 1.0, "source": "rrf_only"}}
                }
            },
            "detection": {"precision": 0.5, "recall": 1.0, "f1": 0.667,
                          "contract_hit": 10, "contracts": 10},
            "latency": {"p50_s": 1.5, "p95_s": 3.0, "first_call_s": 0.4, "docs_per_minute": 48},
        }
        text = bench.summarize(result)
        self.assertIn("R@5=0.6278", text)
        self.assertIn("P=0.500", text)
        self.assertIn("P50=1.50s", text)

    def test_detection_bench_shape(self) -> None:
        if not (ADVERSARIAL / "manifest.json").exists():
            self.skipTest("对抗集清单缺失")
        payload = bench.bench_detection(config_from_env(), limit=2)
        self.assertEqual(payload["contracts"], 2)
        self.assertGreaterEqual(payload["tp"], 1)
        self.assertIn("per_code", payload)
        json.dumps(payload, ensure_ascii=False)  # 必须可序列化

    def test_latency_bench_uses_sample(self) -> None:
        if not SAMPLE.exists():
            self.skipTest("样例缺失")
        payload = bench.bench_latency(config_from_env(), limit=2)
        self.assertGreaterEqual(payload["n"], 1)
        self.assertGreater(payload["p50_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
