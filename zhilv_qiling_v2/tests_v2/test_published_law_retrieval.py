"""Published multidomain snapshots must keep source and version provenance."""
import unittest
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.retrieve import Retriever, StatuteIndex, rewrite_query


class PublishedLawRetrievalTests(unittest.TestCase):
    def test_mixed_published_domains_use_raw_query_and_keep_version_provenance(self):
        rows = [
            {"article_uid": "build-v1-a1", "law_name": "工程规范", "article_no": "第一条",
             "content": "建设工程价款支付和竣工验收", "retrieval_profile": "published_raw_bm25_v1",
             "version_id": "build-v1", "document_id": "build", "source_sha256": "ab" * 32,
             "effective_from": "2024-01-01", "jurisdiction": "全国", "validity": "effective",
             "snapshot_id": "test-snapshot"},
            {"article_uid": "finance-v2-a2", "law_name": "金融规范", "article_no": "第二条",
             "content": "借款利率和融资担保", "retrieval_profile": "published_raw_bm25_v1",
             "version_id": "finance-v2", "document_id": "finance", "source_sha256": "cd" * 32},
        ]
        cfg = Config(use_lsa=False, use_query_rewrite=True, use_rerank=True)
        retriever = Retriever(StatuteIndex(rows, cfg), cfg)
        plan = rewrite_query("建设工程价款支付")
        plan.rewritten = "与工程无关的借款利率"
        result = retriever.search(plan, top_k=1)
        self.assertEqual(result.stats["query"], "建设工程价款支付")
        self.assertEqual(result.stats["retrieval_profile"], "published_raw_bm25_v1")
        self.assertFalse(result.stats["rerank"])
        self.assertEqual(result.hits[0].article_uid, "build-v1-a1")
        self.assertEqual(result.hits[0].provenance["version_id"], "build-v1")
        self.assertEqual(result.hits[0].provenance["snapshot_id"], "test-snapshot")


if __name__ == "__main__":
    unittest.main()
