"""New source corpora must not inherit unvalidated old-corpus ranking."""
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.learned import RetrievalRanker
from zhilv_qiling_v2.retrieve import Retriever, StatuteIndex, rewrite_query


class PackageRetrievalProfileTests(unittest.TestCase):
    def rows(self, packaged=True):
        rows = [
            {"article_uid": "fixture_a", "law_name": "测试规范甲", "article_no": "甲项",
             "content": "设备维护服务中，密钥轮换周期应在交付文档中明确。", "risk_tags": ["irrelevant_tag"]},
            {"article_uid": "fixture_b", "law_name": "测试规范乙", "article_no": "乙项",
             "content": "运输货物应当采用防潮包装并记录温度。", "risk_tags": ["transport"]},
            {"article_uid": "fixture_c", "law_name": "测试规范丙", "article_no": "丙项",
             "content": "验收记录应保存原始检测数据和设备序列号。", "risk_tags": []},
        ]
        if packaged:
            for row in rows:
                row.update(package_id="fixture_package", source_id="source_" + row["article_uid"])
        return rows

    def test_package_uses_raw_query_and_actual_bm25_order_for_unrelated_topics(self):
        cfg = Config(use_lsa=False, use_query_rewrite=True, use_rerank=True)
        index = StatuteIndex(self.rows(), cfg)
        ranker = RetrievalRanker(["q_bm25"], [-1000.0])
        with patch("zhilv_qiling_v2.retrieve.trained_fusion", side_effect=AssertionError("must not load old fusion")):
            retriever = Retriever(index, cfg, ranker)
        for query in ("设备维护的密钥轮换周期", "货物运输防潮包装温度", "验收保存检测数据设备序列号"):
            plan = rewrite_query(query)
            plan.rewritten = "无关改写文字"
            plan.tags = ["irrelevant_tag"]
            with patch.object(index, "dense", side_effect=AssertionError("must not use dense")), patch.object(index, "tag_channel", side_effect=AssertionError("must not use risk tags")):
                result = retriever.search(plan, top_k=3, use_rerank=True)
            expected = [index.statutes[i]["article_uid"] for i, _ in index.bm25(query, limit=3)]
            self.assertEqual([hit.article_uid for hit in result.hits], expected)
            self.assertEqual(result.stats["query"], query)
            self.assertEqual(result.stats["retrieval_profile"], "source_package_raw_bm25_v1")
            self.assertFalse(result.stats["rerank"])
            self.assertEqual(result.stats["channels"]["dense"], 0)
            self.assertEqual(result.stats["channels"]["tag"], 0)
            self.assertFalse(result.stats["query_rewrite_applied"])
        self.assertTrue(cfg.use_rerank)
        self.assertTrue(cfg.use_query_rewrite)

    def test_legacy_and_mixed_corpora_do_not_receive_package_profile(self):
        cfg = Config(use_lsa=False)
        for rows in (self.rows(False), self.rows() + self.rows(False)[:1]):
            index = StatuteIndex(rows, cfg)
            with patch("zhilv_qiling_v2.retrieve.trained_fusion", return_value={"weights": {"sparse": 2}, "rrf_k": 17}) as fusion:
                retriever = Retriever(index, cfg)
            fusion.assert_called_once()
            self.assertEqual(retriever.source_package_id, "")
            self.assertEqual(retriever.weights, {"sparse": 2})
            self.assertEqual(retriever.rrf_k, 17)

    def test_different_package_ids_do_not_share_a_profile(self):
        rows = self.rows()
        rows[1]["package_id"] = "different_package"
        retriever = Retriever(StatuteIndex(rows, Config(use_lsa=False)), Config(use_lsa=False))
        self.assertFalse(retriever.source_package_id)

    def test_disabled_lsa_does_not_fit_or_supply_dense_results(self):
        cfg = Config(use_lsa=False, use_bge=False)
        index = StatuteIndex(self.rows(False), cfg)
        self.assertFalse(index.lsa_ready)
        self.assertIsNone(index.lsa_matrix)
        self.assertEqual(index.dense("密钥轮换周期"), [])


if __name__ == "__main__":
    unittest.main()
