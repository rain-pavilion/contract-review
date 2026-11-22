"""多智能体编排的契约测试。

测的是"图结构引导"这件事本身：拓扑允许什么、拒收什么、回环是否真的重跑、
轨迹能不能序列化成前端/答辩能看的形状。不测模型质量（那是另一套口径）。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi as M
from zhilv_qiling_v2 import agents_multi_run as R
from zhilv_qiling_v2 import pipeline
from zhilv_qiling_v2.config import WORKSPACE, Config
from zhilv_qiling_v2.engine import ReviewEngine

SAMPLE = WORKSPACE / "samples" / "tech-development-contract.txt"


def _contract(config: Config | None = None):
    config = config or Config()
    doc = pipeline.prepare(SAMPLE.name, SAMPLE.read_bytes(), config=config)
    return pipeline.build_contract(doc, config)


class TopologyTest(unittest.TestCase):
    """拓扑是白名单：不在边表里的投递必须被拒收并计数。"""

    def test_allowed_edges(self):
        bus = M.MessageBus(strict=True)
        self.assertTrue(bus.allowed("intake", "orchestrator"))   # 入口只交主脑
        self.assertTrue(bus.allowed("orchestrator", "ner"))      # 主脑派活
        self.assertTrue(bus.allowed("statute", "logic"))         # 回边
        self.assertTrue(bus.allowed("ner", "judge"))
        self.assertTrue(bus.allowed("logic", "orchestrator"))    # 回报高风险 → 主脑再派
        self.assertTrue(bus.allowed("reviewer", "judge"))        # 强模型复核进终审
        self.assertFalse(bus.allowed("judge", "logic"))          # judge 是汇点
        self.assertFalse(bus.allowed("ner", "intake"))           # 不倒灌
        self.assertFalse(bus.allowed("intake", "ner"))           # 入口不越级派活

    def test_publish_rejects_out_of_topology(self):
        bus = M.MessageBus(strict=True)
        bus.publish(M.Message(sender="judge", receiver="logic", kind="should.be.rejected", payload={"x": 1}))
        stats = bus.stats()
        self.assertEqual(stats["messages"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(bus.dump(), [])

    def test_non_strict_keeps_message(self):
        bus = M.MessageBus(strict=False)
        bus.publish(M.Message(sender="judge", receiver="logic", kind="loose", payload={"x": 1}))
        self.assertEqual(bus.stats()["messages"], 1)
        self.assertEqual(bus.stats()["rejected"], 0)

    def test_inbox_filters_by_kind(self):
        # 只测收件箱的 kind 过滤，不测拓扑：logic->ner 在真实拓扑里是会被拒的边
        # （logic 只能发给 subject/statute/judge），所以这里放开 strict。
        bus = M.MessageBus(strict=False)
        bus.publish(M.Message(sender="intake", receiver="ner", kind="contract.parsed", payload={"clauses": 3}))
        bus.publish(M.Message(sender="logic", receiver="ner", kind="defects.detected", payload={"defects": []}))
        self.assertEqual(len(bus.inbox("ner")), 2)
        self.assertEqual(len(bus.inbox("ner", kinds=("contract.parsed",))), 1)
        self.assertEqual(len(bus.inbox("judge")), 0)

    def test_every_agent_is_reachable(self):
        self.assertEqual(set(M.TOPOLOGY), set(R.SCHEDULE))
        for agent in R.SCHEDULE:
            self.assertIn(agent, M.AGENT_SPECS, f"{agent} 缺 Agent 规格")
            spec = M.AGENT_SPECS[agent]
            self.assertTrue(spec.get("role"))
            self.assertTrue(spec.get("model"))


class RunTest(unittest.TestCase):
    """端到端：六个 Agent 跑完、消息流经既定边、回环真的触发。"""

    @classmethod
    def setUpClass(cls):
        cls.contract = _contract()
        from zhilv_qiling_v2 import entailment
        # Topology assertions are independent of local optional neural weights.
        with patch.object(entailment, "backend", return_value="rules"):
            cls.out = R.run_multi_agent(cls.contract, Config())

    def test_all_steps_ok(self):
        bad = [s for s in self.out.steps if s["status"] != "ok"]
        self.assertEqual(bad, [])
        self.assertEqual(len([s for s in self.out.steps if s["agent"] == "intake"]), 1)

    def test_no_rejected_messages(self):
        stats = self.out.bus.stats()
        self.assertEqual(stats["rejected"], 0)
        self.assertGreater(stats["messages"], 10)

    def test_loop_back_reruns_logic(self):
        logic_steps = [s for s in self.out.steps if s["agent"] == "logic"]
        self.assertGreaterEqual(len(logic_steps), 2, "statute 冲突后应重跑 logic")
        self.assertGreater(logic_steps[-1]["inbox"], logic_steps[0]["inbox"])
        edges = self.out.bus.stats()["edges_used"]
        self.assertTrue(any(e.startswith("statute->logic") for e in edges))

    def test_products_shape(self):
        self.assertIsInstance(self.out.score, int)
        self.assertTrue(0 <= self.out.score <= 100)
        self.assertTrue(self.out.risks)
        for risk in self.out.risks:
            self.assertTrue(risk.code)
            self.assertTrue(risk.message)
        self.assertTrue(self.out.citations)

    def test_dump_is_json_serializable(self):
        payload = json.loads(self.out.dump_trace())
        self.assertEqual(payload["order"][:4], ["intake", "orchestrator", "ner", "logic"])
        self.assertIn("judge", payload["order"])
        self.assertTrue(payload["messages"])
        kinds = {m["kind"] for m in payload["messages"]}
        self.assertIn("contract.parsed", kinds)
        self.assertIn("statute.conflict", kinds)

    def test_summary_counts(self):
        summary = self.out.summary()
        self.assertEqual(summary["order"][:4], ["intake", "orchestrator", "ner", "logic"])
        self.assertIn("judge", summary["order"])
        self.assertEqual(summary["risks"], len(self.out.risks))
        self.assertGreater(summary["messages"], 10)
        self.assertEqual(len(summary["steps"]), len(self.out.steps))


class EngineSwitchTest(unittest.TestCase):
    """开关打开后走多智能体，产物结构与单进程链路一致。"""

    def test_default_is_single_pipeline(self):
        self.assertFalse(Config().enable_multi_agent)

    def test_env_switch(self):
        import os

        os.environ["ZLQL_MULTI_AGENT"] = "1"
        try:
            from zhilv_qiling_v2.config import config_from_env

            self.assertTrue(config_from_env().enable_multi_agent)
        finally:
            os.environ.pop("ZLQL_MULTI_AGENT", None)

    def test_multi_agent_path_same_report_shape(self):
        config = Config()
        config.enable_multi_agent = True
        contract = _contract(config)
        report = ReviewEngine(config).run(contract)
        self.assertGreater(report.score, 0)
        self.assertTrue(report.risks)
        self.assertEqual(set(report.severity_counts), {"high", "medium", "low", "info"})
        self.assertTrue(report.stats["multi_agent"]["enabled"])
        self.assertEqual(
            report.stats["multi_agent"]["order"][:4], ["intake", "orchestrator", "ner", "logic"]
        )
        self.assertTrue(report.stats["multi_agent"]["messages"])
        self.assertTrue(report.trace)


if __name__ == "__main__":
    unittest.main()
