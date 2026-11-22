# -*- coding: utf-8 -*-
"""资质法规知识图谱与 RAG 的单元测试（节点③ 的"图谱 + RAG"部分）。"""
from __future__ import annotations

import unittest

from zhilv_qiling_v2 import legal_kg as K


class LegalKgTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = K.get_index()

    def test_entries_loaded(self) -> None:
        self.assertGreaterEqual(self.index.size, 15)
        industries = {entry.industry for entry in self.index.entries}
        self.assertIn("食品经营", industries)
        self.assertIn("物业服务", industries)

    def test_abolished_entries_present(self) -> None:
        """已取消的资质必须在库里，否则同一套规则会把合规合同误报成缺资质。"""
        abolished = [entry for entry in self.index.entries if entry.status == "abolished"]
        names = " ".join(entry.qualification for entry in abolished)
        self.assertGreaterEqual(len(abolished), 2)
        self.assertIn("物业服务", names)
        self.assertIn("系统集成", names)

    def test_retrieve_ranks_target_industry_first(self) -> None:
        hits = self.index.retrieve("食品经营 食堂 配送 资质")
        self.assertTrue(hits)
        self.assertEqual(hits[0].entry.industry, "食品经营")
        self.assertGreater(hits[0].score, 0)

    def test_graph_shape(self) -> None:
        graph = K.get_graph()
        stats = graph.stats()
        self.assertGreaterEqual(stats["n_industry"], 15)
        self.assertGreaterEqual(stats["n_qualification"], 15)
        self.assertGreaterEqual(stats["n_statute"], 15)
        self.assertEqual(stats["edges"], stats["n_qualification"] * 2)

    def test_verdict_reports_food(self) -> None:
        verdict = K.qualification_verdict("食堂食材配送服务，乙方每日配送蔬菜至甲方食堂")
        self.assertTrue(verdict.should_report)
        self.assertIsNotNone(verdict.entry)
        self.assertEqual(verdict.entry.industry, "食品经营")
        self.assertIn("食品经营许可", verdict.reason)

    def test_verdict_suppresses_abolished(self) -> None:
        """物业服务企业资质 2017 年取消，报出来就是误报。"""
        verdict = K.qualification_verdict("物业管理服务合同，乙方提供保洁与绿化养护")
        self.assertFalse(verdict.should_report)
        self.assertTrue(any("已取消" in item for item in verdict.suppressed))

    def test_verdict_suppresses_integration_without_secret_context(self) -> None:
        verdict = K.qualification_verdict("信息系统集成与运维服务，乙方负责机房维护与软件开发")
        self.assertFalse(verdict.should_report)

    def test_verdict_reports_secret_integration(self) -> None:
        verdict = K.qualification_verdict("涉密信息系统集成项目，乙方承担涉密网络建设")
        self.assertTrue(verdict.should_report)
        self.assertEqual(verdict.entry.industry, "涉密集成")

    def test_confidentiality_clause_is_not_secret_project(self) -> None:
        """合同里的保密条款不等于涉密项目 —— 这是实测踩到的误报。"""
        verdict = K.qualification_verdict("技术开发合同，双方约定保密义务，乙方开发数据中台")
        self.assertFalse(verdict.should_report)

    def test_verdict_requires_industry_keyword(self) -> None:
        verdict = K.qualification_verdict("甲乙双方就有关事宜达成一致，签订本合同。")
        self.assertFalse(verdict.should_report)
        self.assertTrue(verdict.reason)

    def test_entry_citation_text(self) -> None:
        entry = next(item for item in self.index.entries if item.entry_id == "QK-001")
        text = entry.citation_text()
        self.assertIn("建筑业企业资质管理规定", text)
        self.assertTrue(text.startswith("《"))

    def test_infer_industries_orders_by_hits(self) -> None:
        found = K.infer_industries("食品 食品 食品 物业")
        self.assertEqual(found[0][0], "食品经营")


if __name__ == "__main__":
    unittest.main()
