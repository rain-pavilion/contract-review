"""v2 审查链路测试：要素抽取 → 规则 → 检测器 → 五智能体 → 引擎 → 管线。

测试原则：断言"必须被检出/必须被溯源"的行为，不断言具体分值。分值会随权重
训练波动，行为（检出、去重、引用可溯源、回环有上限）才是契约。

    python -m unittest zhilv_qiling_v2.tests_v2.test_review -v
"""

from __future__ import annotations

import unittest

from zhilv_qiling_v2 import pipeline
from zhilv_qiling_v2.agents import arbitrate, fuse, understand
from zhilv_qiling_v2.config import default_config
from zhilv_qiling_v2.detectors import run_detectors
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import Clause, ContractElements, RiskCandidate
from zhilv_qiling_v2.rules import RuleContext, rule_zero_rent, run_rules

ZERO_RENT_CONTRACT = """房屋租赁合同
出租人（甲方）：王某某
承租人（乙方）：李某某
第一条 房屋基本情况
1.1 房屋位于某市某区某路 1 号，建筑面积 80 平方米。
第二条 租金
2.1 月租金为人民币0元，押金为人民币999999元。
第三条 租赁期限
3.1 租赁期限自2026年12月1日起至2026年11月30日止。
第四条 交付
4.1 甲方应及时交付房屋，验收标准由甲方自行确定。
"""

CLEAN_CONTRACT = """技术开发（委托）合同
委托方（甲方）：星海科技产业园管理有限公司
受托方（乙方）：云图数据技术有限公司
第一条 项目内容
1.1 甲方委托乙方开发智慧园区数据中台，交付物为源代码与部署文档。
第二条 合同价款
2.1 合同总金额为人民币1200000元（大写：壹佰贰拾万元整）。
2.2 甲方于验收合格后15个工作日内支付。
第三条 履行期限
3.1 项目自2026年3月1日起至2026年9月30日止，分三个阶段实施。
第四条 验收
4.1 验收标准为附件一《验收细则》所列全部指标。
第五条 违约责任
5.1 乙方逾期交付的，每逾期一日按合同总金额的0.05%支付违约金。
第六条 争议解决
6.1 双方协商不成的，提交合同签订地人民法院管辖。
第七条 其他
7.1 本合同自双方签字盖章之日起生效。
"""


class ElementAndRuleTest(unittest.TestCase):
    """要素抽取与规则引擎必须能识别"零租金 + 期限倒置"这类硬缺陷。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = default_config()
        cls.document = pipeline.prepare("zero.txt", ZERO_RENT_CONTRACT.encode("utf-8"),
                                        config=cls.config)
        cls.contract = pipeline.build_contract(cls.document, config=cls.config)

    def test_zero_amount_is_kept(self):
        amounts = [element.normalized for element in self.contract.elements.amounts]
        self.assertTrue(any(str(value).startswith("0") for value in amounts),
                        f"零金额被丢弃了：{amounts}")

    def test_parties_extracted(self):
        parties = {element.value for element in self.contract.elements.parties}
        self.assertTrue(any("王" in party for party in parties), parties)
        self.assertTrue(any("李" in party for party in parties), parties)

    def test_rules_flag_zero_rent_and_date_reversed(self):
        codes = {candidate.code for candidate in run_rules(self.contract)}
        self.assertIn("zero_rent", codes)

    def test_zero_rent_does_not_match_the_zero_suffix_of_positive_prices(self):
        context = RuleContext(contract=self.contract, clauses=[], full_text='',
                              elements=ContractElements(), signals=[])
        for value in ('10', '10000', '120.00', '0.01'):
            with self.subTest(value=value):
                text = f'合同价款为人民币{value}元。'
                clause = Clause('price', '1', '价款', 1, '', text, text)
                self.assertEqual(rule_zero_rent(clause, context), [])
        for value in ('0', '0.00'):
            with self.subTest(value=value):
                text = f'合同价款为人民币{value}元。'
                clause = Clause('price', '1', '价款', 1, '', text, text)
                self.assertTrue(rule_zero_rent(clause, context))

    def test_detectors_flag_date_reversed(self):
        codes = {candidate.extra.get("issue_code") or candidate.code
                 for candidate in run_detectors(self.contract)}
        self.assertTrue({"date_reversed", "zero_deposit", "zero_rent"} & codes, codes)

    def test_engine_produces_trace_and_score(self):
        engine = ReviewEngine(self.config)
        report = engine.run(self.contract)
        self.assertLess(report.score, 100)
        self.assertGreaterEqual(len(report.trace), 5)
        nodes = " ".join(event.node for event in report.trace)
        for expected in ("understand", "clause_review", "cross_review", "legal", "judge"):
            self.assertIn(expected, nodes)
        for risk in report.risks:
            self.assertIn(risk.severity, ("high", "medium", "low", "info"))
            for citation in risk.citations:
                # 硬护栏：引用必须来自本地知识库（有 law_name 与 article_no）
                self.assertTrue(citation.law_name)
                self.assertTrue(citation.article_no)


class FusionInvariantTest(unittest.TestCase):
    """融合层不变量：标签不丢、去重后同码同锚只有一条、置信度不越界。"""

    def test_dedupe_key_keeps_issue_code(self):
        left = RiskCandidate(candidate_id="a", code="confidentiality_missing",
                             severity="medium", message="不安全保管", confidence=0.6,
                             extra={"issue_code": "insecure_delivery"})
        right = RiskCandidate(candidate_id="b", code="confidentiality_missing",
                              severity="medium", message="公开披露", confidence=0.6,
                              extra={"issue_code": "public_disclosure"})
        self.assertNotEqual(left.dedupe_key, right.dedupe_key)

    def test_merge_from_carries_extra(self):
        left = RiskCandidate(candidate_id="a", code="zero_rent", severity="medium",
                             message="m", confidence=0.5)
        right = RiskCandidate(candidate_id="b", code="zero_rent", severity="medium",
                              message="m", confidence=0.6,
                              extra={"issue_code": "zero_rent"})
        left.merge_from(right)
        self.assertEqual(left.extra.get("issue_code"), "zero_rent")
        self.assertGreaterEqual(left.confidence, 0.6)

    def test_arbitrate_merges_same_issue_code(self):
        config = default_config()
        candidates = [
            RiskCandidate(candidate_id="a", code="zero_rent", severity="high",
                          message="来自规则", confidence=0.8,
                          tags=["zero_rent"], extra={"issue_code": "zero_rent"}),
            RiskCandidate(candidate_id="b", code="zero_rent", severity="high",
                          message="来自检测器", confidence=0.79,
                          tags=["zero_rent"], extra={"issue_code": "zero_rent"}),
        ]
        resolved, conflicts = arbitrate(candidates, config)
        self.assertEqual(len(resolved), 1)
        self.assertIn("zero_rent", resolved[0].tags)
        self.assertIsInstance(conflicts, list)

    def test_arbitrate_keeps_distinct_issue_codes(self):
        config = default_config()
        candidates = [
            RiskCandidate(candidate_id="a", code="confidentiality_missing",
                          severity="medium", message="U 盘交付", confidence=0.7,
                          tags=["insecure_delivery"], extra={"issue_code": "insecure_delivery"}),
            RiskCandidate(candidate_id="b", code="confidentiality_missing",
                          severity="medium", message="可公开披露", confidence=0.7,
                          tags=["public_disclosure"], extra={"issue_code": "public_disclosure"}),
        ]
        resolved, _conflicts = arbitrate(candidates, config)
        self.assertEqual(len(resolved), 2)

    def test_fuse_never_exceeds_one(self):
        config = default_config()
        document = pipeline.prepare("clean.txt", CLEAN_CONTRACT.encode("utf-8"),
                                    config=config)
        contract = pipeline.build_contract(document, config=config)
        heavy = [
            RiskCandidate(candidate_id=f"c{index}", code="zero_rent", severity="low",
                          message="m", confidence=0.99,
                          extra={"issue_code": "zero_rent"})
            for index in range(3)
        ]
        risks = fuse(heavy, [], contract, config)
        for risk in risks:
            self.assertLessEqual(risk.confidence, 1.0)
            self.assertGreaterEqual(risk.confidence, 0.0)


class PipelineContractTest(unittest.TestCase):
    """端到端管线：干净合同不该被报成高危；结果可 JSON 化。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = default_config()
        cls.clean = pipeline.review_bytes("clean.txt", CLEAN_CONTRACT.encode("utf-8"),
                                          cls.config)
        cls.dirty = pipeline.review_bytes("zero.txt", ZERO_RENT_CONTRACT.encode("utf-8"),
                                          cls.config)
        cls.clean_document = pipeline.prepare("clean.txt", CLEAN_CONTRACT.encode("utf-8"),
                                              config=cls.config)

    def test_clean_scores_higher_than_dirty(self):
        self.assertGreaterEqual(self.clean.score, self.dirty.score)

    def test_clean_has_no_high_risk(self):
        self.assertFalse([risk for risk in self.clean.risks if risk.severity == "high"])

    def test_review_report_serializable(self):
        payload = self.dirty.to_dict()
        self.assertIn("risks", payload)
        self.assertIn("trace", payload)
        self.assertEqual(payload["contract_id"], self.dirty.contract_id)

    def test_understanding_annotates_roles(self):
        contract = pipeline.build_contract(self.clean_document, config=self.config)
        result = understand(contract, self.config)
        self.assertEqual(result.agent, "understand")
        roles = [clause.metadata.get("roles") for clause in contract.document.clauses]
        self.assertTrue(any(roles))

    def test_batch_aggregates(self):
        summary = pipeline.BatchSummary(
            reports=[self.clean, self.dirty], failed=[], elapsed=0.5
        )
        self.assertEqual(len(summary.reports), 2)
        self.assertEqual(summary.failed, [])

    def test_elements_have_clause_links(self):
        contract = pipeline.build_contract(self.clean_document, config=self.config)
        for element in list(contract.elements.parties) + list(contract.elements.amounts):
            self.assertTrue(element.element_id)
            self.assertTrue(element.kind)


if __name__ == "__main__":
    unittest.main(verbosity=2)
