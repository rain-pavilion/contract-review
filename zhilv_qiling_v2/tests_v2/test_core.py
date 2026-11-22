"""v2 核心层单元测试：文本归一化、模型契约、版面、切分。

只测"契约与不变量"，不测具体阈值的绝对值 —— 阈值会随语料变化，测试应该
在重构后仍然有意义。运行：

    python -m unittest zhilv_qiling_v2.tests_v2.test_core -v
"""

from __future__ import annotations

import unittest

from zhilv_qiling_v2 import text as T
from zhilv_qiling_v2 import layout as L
from zhilv_qiling_v2.config import default_config
from zhilv_qiling_v2.models import (
    RISK_CATEGORIES,
    RISK_CODES,
    Clause,
    SourceBlock,
    code_category,
    code_label,
    code_severity,
    dedupe_candidates,
    stable_id,
)
from zhilv_qiling_v2.splitter import (
    detect_heading,
    dp_segment,
    greedy_segment,
    rule_boundary,
    split_document,
)


class TextUtilsTest(unittest.TestCase):
    """归一化与数字/金额/日期解析。"""

    def test_full_to_half_keeps_cjk(self):
        self.assertEqual(T.full_to_half("ＡＢＣ１２３，中文"), "ABC123,中文")

    def test_cn_to_int_variants(self):
        self.assertEqual(T.cn_to_int("二〇二五"), 2025)
        self.assertEqual(T.cn_to_int("壹拾贰"), 12)
        self.assertEqual(T.cn_to_int("一百零五"), 105)
        self.assertEqual(T.cn_to_int("十"), 10)

    def test_money_cn_uppercase(self):
        # v1 缺大写金额解析能力，这是 v2 新增的
        self.assertAlmostEqual(T.money_cn_to_float("人民币壹拾万零贰仟叁佰元整"), 102300.0)

    def test_money_to_float_keeps_zero(self):
        # 零金额是真实缺陷信号（"月租金为人民币0元"），不能被当成"解析失败"丢掉
        self.assertEqual(T.money_to_float("0元"), 0.0)
        self.assertAlmostEqual(T.money_to_float("12万元"), 120000.0)
        self.assertAlmostEqual(T.money_to_float("￥120,000.00"), 120000.0)
        self.assertIsNone(T.money_to_float("没有金额"))

    def test_parse_date_chinese(self):
        self.assertEqual(T.parse_date_text("二〇二五年三月一日"), "2025-03-01")
        self.assertEqual(T.parse_date_text("2026年12月1日"), "2026-12-01")

    def test_detect_label_variants(self):
        self.assertEqual(T.detect_label("第一条 委托事项")["level"], 2)
        self.assertEqual(T.detect_label("第一章 总则")["level"], 1)
        self.assertEqual(T.detect_label("1.2 乙方应当")["label"], "1.2")
        # "2025年" 不是编号
        self.assertFalse(T.detect_label("2025年3月1日签订")["label"])

    def test_tokenize_mixes_cjk_and_latin(self):
        tokens = T.tokenize("甲方ABC公司2026")
        self.assertIn("甲", tokens)
        self.assertIn("甲方", tokens)
        self.assertTrue(any(token.isdigit() for token in tokens))


class ModelsContractTest(unittest.TestCase):
    """模型层的"字段契约"必须稳定，否则下游 agents/engine 会静默出错。"""

    def test_stable_id_is_deterministic(self):
        self.assertEqual(stable_id("ct", "a.txt", 10), stable_id("ct", "a.txt", 10))
        self.assertNotEqual(stable_id("ct", "a.txt", 10), stable_id("ct", "a.txt", 11))
        self.assertTrue(stable_id("ct", "a.txt").startswith("ct_"))

    def test_risk_code_table_complete(self):
        for code in RISK_CODES:
            self.assertIn(code_category(code), RISK_CATEGORIES)
            # info 用于 "未发现实质风险" 这类非风险结论，是合法等级
            self.assertIn(code_severity(code), ("high", "medium", "low", "info"))
            self.assertTrue(code_label(code))

    def test_clause_metadata_default_is_independent(self):
        first = Clause(clause_id="c1", label="1", title="t", level=2,
                       heading="第一条 委托事项", body="正文", text="第一条 委托事项\n正文")
        second = Clause(clause_id="c2", label="2", title="t", level=2,
                        heading="第二条 价款", body="正文", text="第二条 价款\n正文")
        first.metadata["roles"] = ["主体"]
        self.assertEqual(second.metadata, {})

    def test_dedupe_candidates_keeps_higher_confidence(self):
        from zhilv_qiling_v2.models import RiskCandidate

        low = RiskCandidate(candidate_id="a", code="zero_rent", severity="medium",
                            message="低", confidence=0.4)
        high = RiskCandidate(candidate_id="b", code="zero_rent", severity="medium",
                             message="高", confidence=0.9)
        merged = dedupe_candidates([low, high])
        self.assertEqual(len(merged), 1)
        self.assertGreaterEqual(merged[0].confidence, 0.9)


class LayoutTest(unittest.TestCase):
    """版面处理：页眉页脚剥离、断行合并、表格行识别。"""

    def _block(self, order: int, text: str, page: int = 1, bold: bool = False,
               size: float = 12.0) -> SourceBlock:
        return SourceBlock(block_id=f"b{order}", order=order, kind="paragraph",
                           text=text, page=page, font_size=size, is_bold=bold)

    def test_strip_running_heads_removes_page_number(self):
        blocks = [self._block(index, f"第{index}条 内容{index}。", page=index)
                  for index in range(1, 5)]
        blocks += [self._block(9, "— 3 —", page=3)]
        cleaned, stats = L.strip_running_heads(blocks)
        self.assertTrue(all("— 3 —" not in block.text for block in cleaned))
        self.assertTrue(stats)  # 返回的统计是列表/字典均可，只要有内容

    def test_merge_wrapped_lines_keeps_meta_lines(self):
        blocks = [self._block(0, "项目名称：智慧园区数据中台建设项目"),
                  self._block(1, "合同编号：2026-001")]
        merged = L.merge_wrapped_lines(blocks)
        self.assertEqual(len(merged), 2)

    def test_mark_table_rows(self):
        # 三列以上且各列短、含数字列 → 判为表格行
        blocks = [self._block(0, "合同金额  1200000元  2026-03-01"),
                  self._block(1, "这是一段普通的叙述性合同正文，提到了若干义务。")]
        marked = L.mark_table_rows(blocks)
        self.assertTrue(any(block.kind == "table_row" for block in marked))


class SplitterTest(unittest.TestCase):
    """规则/DP/贪心三种切分都必须满足"区块不丢不重"。"""

    def _parse(self, text: str, source: str = "t.txt"):
        blocks = [
            SourceBlock(block_id=f"b{index}", order=index, kind="paragraph", text=line)
            for index, line in enumerate(text.strip().splitlines())
            if line.strip()
        ]
        return split_document(blocks, default_config(), source_name=source)

    def test_rule_boundary_recognizes_article(self):
        prev = SourceBlock(block_id="b0", order=0, kind="paragraph", text="第一条 委托事项")
        cur = SourceBlock(block_id="b1", order=1, kind="paragraph", text="1.1 甲方委托乙方开发。")
        kind, confidence, _evidence = rule_boundary(prev, cur)
        self.assertIn(kind, ("NEW_CLAUSE", "NEW_SUBCLAUSE"))
        self.assertGreaterEqual(confidence, 0.8)

    def test_split_keeps_all_blocks(self):
        text = "\n".join([
            "技术开发（委托）合同",
            "甲方：星海科技有限公司",
            "第一条 委托事项",
            "1.1 甲方委托乙方开发数据中台。",
            "1.2 乙方应按期交付。",
            "第二条 价款",
            "2.1 合同总金额为人民币1200000元。",
        ])
        document = self._parse(text)
        covered = {block_id for clause in document.clauses for block_id in clause.block_ids}
        self.assertEqual(covered, {block.block_id for block in document.blocks})
        self.assertFalse(any(issue.kind == "block_uncovered" for issue in document.issues))

    def test_dp_and_greedy_agree_on_clause_count_order(self):
        from zhilv_qiling_v2.splitter import BoundaryDecision, BoundaryEvidence

        text = "\n".join([
            "保管合同",
            "第一条 保管物", "1.1 甲方将艺术品交由乙方保管。",
            "第二条 保管费", "2.1 保管费为每年4800元。",
            "第三条 交付", "3.1 保管期满后乙方应返还保管物。",
        ])
        blocks = [
            SourceBlock(block_id=f"b{index}", order=index, kind="paragraph", text=line)
            for index, line in enumerate(text.splitlines())
        ]
        config = default_config()
        # 概率来自"该块是否是标题"（真实管线里由规则+分类器融合给出），
        # 均匀 0.5 不该切分 —— 那是 DP 的代价项在起作用，也算一个不变量。
        flat_segments, _flat_stats = dp_segment(blocks, [0.5] * len(blocks), config)
        self.assertEqual(len(flat_segments), 1)
        probs = [
            0.95 if detect_heading(block).label else 0.05 for block in blocks
        ]
        dp_segments, _dp_stats = dp_segment(blocks, probs, config)
        decisions = [
            BoundaryEvidence(
                index=index,
                decision=BoundaryDecision(
                    boundary_id=f"bd{index}",
                    left_block_id=blocks[index - 1].block_id,
                    right_block_id=blocks[index].block_id,
                    relation=(BoundaryDecision.NEW_CLAUSE if probs[index] > 0.5
                              else BoundaryDecision.SAME_CLAUSE),
                    confidence=max(probs[index], 1 - probs[index]),
                ),
                split_prob=probs[index],
            )
            for index in range(1, len(blocks))
        ]
        greedy_segments, _greedy_stats = greedy_segment(blocks, decisions, config)
        self.assertGreaterEqual(len(dp_segments), 3)
        self.assertGreaterEqual(len(greedy_segments), 3)
        self.assertEqual(
            [segment.blocks[0].order for segment in dp_segments],
            sorted(segment.blocks[0].order for segment in dp_segments),
        )

    def test_cover_clause_marked(self):
        document = self._parse("\n".join([
            "技术开发（委托）合同",
            "甲方：星海科技有限公司",
            "第一条 委托事项",
            "1.1 甲方委托乙方开发数据中台。",
        ]))
        covers = [clause for clause in document.clauses if clause.source == "cover"]
        self.assertTrue(covers)
        self.assertIn("合同", covers[0].title)


class RetrievalProtocolInvariantTest(unittest.TestCase):
    """检索通道权重与 LTR 特征表必须一致（训练与推理共用同一份名字表）。"""

    def test_rerank_feature_names_match_vector(self):
        from zhilv_qiling_v2.features import features_to_vector
        from zhilv_qiling_v2.learned import RERANK_FEATURES, default_ranker

        self.assertEqual(len(default_ranker().weights), len(RERANK_FEATURES))
        vector = features_to_vector({name: 1.0 for name in RERANK_FEATURES},
                                    names=RERANK_FEATURES)
        self.assertEqual(len(vector), len(RERANK_FEATURES))

    def test_boundary_feature_names_unique(self):
        from zhilv_qiling_v2.features import BOUNDARY_FEATURES

        self.assertEqual(len(BOUNDARY_FEATURES), len(set(BOUNDARY_FEATURES)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
