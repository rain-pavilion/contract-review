# -*- coding: utf-8 -*-
"""申报书五节点的测试：节点③主体资格审查、节点④一致/冲突判别、大模型通道降级。

约定：这些用例只断言"行为契约"（有没有结论、有没有图路径、模型不可用时是否降级），
不断言具体分值——分值随权重波动，行为才是契约。

    python -m unittest zhilv_qiling_v2.tests_v2.test_nodes -v
"""

from __future__ import annotations

import unittest

from zhilv_qiling_v2 import entailment as E
from zhilv_qiling_v2 import llm_client as L
from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2 import pipeline
from zhilv_qiling_v2 import subject_graph as S
from zhilv_qiling_v2.config import default_config
from zhilv_qiling_v2.engine import ReviewEngine

# 离线正向夹具明确包含涉密业务以及签约人的代理语境；普通信息系统
# 集成/等级保护或“委托乙方提供服务”不能单独触发对应缺失结论。
LICENSE_MISSING_CONTRACT = """信息系统集成与等级保护测评服务采购合同
采购人（甲方）：某市大数据管理局
供应商（乙方）：云图数据技术有限公司
第一条 服务内容
1.1 甲方委托乙方承担涉密信息系统集成与网络安全等级保护测评，含机房设备巡检与应急响应。
第二条 合同价款
2.1 合同总金额为人民币 980000 元（大写：玖拾捌万元整）。
2.2 甲方分两期支付。
第三条 履行期限
3.1 服务期自 2026 年 1 月 1 日起至 2026 年 12 月 31 日止。
第四条 其他
4.1 本合同自双方签字盖章之日起生效。
4.2 乙方委托项目经理签署本合同，未说明其签约权限。
"""

CREDIT_RISK_CONTRACT = """设备采购合同
采购人（甲方）：某区财政局
供应商（乙方）：某某商贸有限公司
第一条 标的
1.1 采购办公设备一批。
第二条 价款
2.1 合同金额为人民币 200000 元。
第三条 主体材料
3.1 乙方曾被列入失信被执行人名单，现提供信用修复证明。
"""

CONFLICT_CLAUSE = "乙方对因设备故障造成的任何损失均不承担任何赔偿责任，无论是否因其故意或过失。"
CONFLICT_ARTICLE = "因当事人一方的违约行为，损害对方人身权益、财产权益的，受损害方有权请求其承担违约责任。"


class LlmClientTest(unittest.TestCase):
    """大模型客户端必须"可缺席、可解析、可降级"。"""

    def test_chat_url_variants(self):
        self.assertEqual(
            L.chat_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        self.assertEqual(
            L.chat_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions",
        )
        self.assertEqual(
            L.chat_url("https://api.example.com"),
            "https://api.example.com/v1/chat/completions",
        )
        self.assertEqual(L.chat_url(""), "")

    def test_extract_json_from_code_fence(self):
        payload = L._extract_json('好的，结果如下：\n```json\n{"ok": true, "n": 2}\n```\n以上。')
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["n"], 2)

    def test_extract_json_raises_when_absent(self):
        with self.assertRaises(L.LLMClientError):
            L._extract_json("这里没有 JSON")

    def test_available_false_without_endpoint(self):
        self.assertFalse(L.available(L.LLMConfig()))
        self.assertFalse(L.available(None))

    def test_request_json_raises_without_endpoint(self):
        with self.assertRaises(L.LLMClientError):
            L.request_json(L.LLMConfig(), [{"role": "user", "content": "hi"}])


class SubjectGraphTest(unittest.TestCase):
    """节点③：主体类型判定、图上路径、资格/权限/信用三类结论。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = default_config()
        cls.document = pipeline.prepare(
            "ops.txt", LICENSE_MISSING_CONTRACT.encode("utf-8"), config=cls.config
        )
        cls.contract = pipeline.build_contract(cls.document, config=cls.config)
        cls.result = S.check_subjects(cls.contract, cls.config)

    def test_kind_classification(self):
        self.assertEqual(S._classify_kind("云图数据技术有限公司")[0], "企业")
        self.assertEqual(S._classify_kind("某市大数据管理局")[0], "机关")
        self.assertEqual(S._classify_kind("吴磊伟")[0], "个人")

    def test_graph_has_has_type_edges(self):
        kinds = {edge.kind for edge in self.result.edges}
        self.assertIn("has_type", kinds)
        self.assertTrue(self.result.paths())
        self.assertIn("has_type", self.result.paths()[0])

    def test_license_missing_detected(self):
        codes = {candidate.code for candidate in self.result.candidates}
        self.assertIn("need_license", codes, self.result.stats)
        requirements = [edge for edge in self.result.edges if edge.kind == "requires"]
        self.assertTrue(requirements)
        self.assertTrue(all(edge.source.startswith("S:seller:") for edge in requirements))

    def test_authorization_missing_detected(self):
        codes = {candidate.code for candidate in self.result.candidates}
        self.assertIn("need_authorization", codes, self.result.stats)

    def test_ordinary_it_business_delegation_is_not_secret_activity_or_signature_authorization(self):
        ordinary = LICENSE_MISSING_CONTRACT.replace("涉密信息系统集成", "政务信息系统集成").replace(
            "4.2 乙方委托项目经理签署本合同，未说明其签约权限。", "")
        document = pipeline.prepare("ordinary-ops.txt", ordinary.encode("utf-8"), config=self.config)
        result = S.check_subjects(pipeline.build_contract(document, config=self.config), self.config)
        self.assertNotIn("need_license", {candidate.code for candidate in result.candidates})
        self.assertNotIn("need_authorization", {candidate.code for candidate in result.candidates})

    def test_evidence_carries_graph_path(self):
        kinds = {ev.kind for candidate in self.result.candidates for ev in candidate.evidence}
        self.assertIn("graph_path", kinds)
        for candidate in self.result.candidates:
            self.assertTrue(candidate.evidence, candidate.code)

    def test_query_hint_has_no_prefix(self):
        hint = self.result.query_hint()
        for license_name in hint["licenses"]:
            self.assertNotIn("Q:", license_name)

    def test_credit_risk_detected(self):
        document = pipeline.prepare(
            "credit.txt", CREDIT_RISK_CONTRACT.encode("utf-8"), config=self.config
        )
        contract = pipeline.build_contract(document, config=self.config)
        result = S.check_subjects(contract, self.config)
        self.assertTrue(result.stats["credit_hits"])
        self.assertIn("illegal_terms", {candidate.code for candidate in result.candidates})

    def test_no_party_yields_unspecified(self):
        document = pipeline.prepare(
            "noparty.txt", "合同\n甲方应按约定支付。\n乙方应按时交付。\n".encode("utf-8"),
            config=self.config,
        )
        contract = pipeline.build_contract(document, config=self.config)
        result = S.check_subjects(contract, self.config)
        self.assertIn("subject_unspecified", {candidate.code for candidate in result.candidates})


class EntailmentTest(unittest.TestCase):
    def setUp(self):
        # These fixtures test the deterministic relation classifier, not an
        # optional on-disk neural checkpoint's empirical predictions.
        from unittest.mock import patch
        patcher = patch.object(E, "backend", return_value="rules")
        patcher.start()
        self.addCleanup(patcher.stop)
    """节点④第二半：条款与条文的一致/冲突判别。"""

    def test_features_shape(self):
        feats = E.clause_features("甲方应于验收后支付价款", "当事人应当按照约定支付价款。")
        self.assertEqual(set(feats), set(E.FEATURES))
        for value in feats.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_weak_label_three_classes(self):
        self.assertEqual(
            E.weak_label({"e_negation_flip": 1.0, "e_char_overlap": 0.3, "e_tag_match": 0.0,
                          "e_modality_gap": 0.0, "e_action_match": 1.0}),
            "contradict",
        )
        self.assertEqual(
            E.weak_label({"e_negation_flip": 0.0, "e_char_overlap": 0.4, "e_tag_match": 1.0,
                          "e_modality_gap": 0.0, "e_action_match": 1.0}),
            "entail",
        )
        self.assertEqual(
            E.weak_label({"e_negation_flip": 0.0, "e_char_overlap": 0.0, "e_tag_match": 0.0,
                          "e_modality_gap": 0.0, "e_action_match": 0.0}),
            "neutral",
        )

    def test_conflict_judgement(self):
        verdict = E.judge_pair(CONFLICT_CLAUSE, {"article_no": "第五百七十七条", "content": CONFLICT_ARTICLE})
        self.assertEqual(verdict.label, "contradict", verdict.features)
        self.assertGreaterEqual(verdict.score, 0.5)
        self.assertTrue(verdict.is_conflict)

    def test_unrelated_article_is_neutral(self):
        verdict = E.judge_pair("本合同的争议提交人民法院管辖", {"article_no": "第九百条", "content": "保管人应当妥善保管保管物。"})
        self.assertEqual(verdict.label, "neutral")

    def test_summarize_counts(self):
        verdicts = [
            E.judge_pair(CONFLICT_CLAUSE, {"article_no": "A", "content": CONFLICT_ARTICLE}),
            E.judge_pair(CONFLICT_CLAUSE, {"article_no": "B", "content": "与主题无关的条文内容。"}),
        ]
        summary = E.summarize(verdicts)
        self.assertEqual(summary["pairs"], 2)
        self.assertIn("counts", summary)


class FiveNodeRunTest(unittest.TestCase):
    """五节点串联：顺序正确、模型缺席时全部降级且不抛异常。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = default_config()
        cls.config.enable_llm = False
        cls.document = pipeline.prepare(
            "ops.txt", LICENSE_MISSING_CONTRACT.encode("utf-8"), config=cls.config
        )
        cls.contract = pipeline.build_contract(cls.document, config=cls.config)

    def test_node_order(self):
        run = N.run_five_nodes(self.contract, self.config)
        self.assertEqual([outcome.node for outcome in run.outcomes], list(N.NODE_ORDER))

    def test_llm_absent_degrades_without_raising(self):
        run = N.run_five_nodes(self.contract, self.config)
        degraded = run.degraded()
        self.assertTrue(any("llm_off" in item for item in degraded), degraded)
        self.assertIsInstance(run.score, int)

    def test_node2_collects_rule_and_detector(self):
        run = N.run_five_nodes(self.contract, self.config)
        logic = run.by_node("node2_logic")
        self.assertIsNotNone(logic)
        self.assertGreater(logic.payload["rule"] + logic.payload["detector"], 0)

    def test_node3_and_node4_payloads(self):
        run = N.run_five_nodes(self.contract, self.config)
        subject = run.by_node("node3_subject")
        statute = run.by_node("node4_statute")
        self.assertGreater(subject.payload["stats"]["nodes"], 0)
        # 检索索引可用时给出 entailment 统计；不可用时节点必须显式降级
        self.assertTrue(
            "entailment" in statute.payload or "retriever_unavailable" in statute.degraded,
            (statute.payload, statute.degraded),
        )

    def test_engine_trace_contains_subject_check(self):
        engine = ReviewEngine(self.config)
        report = engine.run(self.contract)
        names = " ".join(event.node for event in report.trace)
        for expected in ("understand", "clause_review", "cross_review",
                         "subject_check", "legal", "judge"):
            self.assertIn(expected, names)
        self.assertTrue(report.stats["nodes"]["subject_check"])


if __name__ == "__main__":
    unittest.main()
