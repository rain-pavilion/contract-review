"""主脑调度与分级复核的测试。

守住三件事：
1. 主脑开工前确实向五个业务节点派活，包括最终综合评估；
2. 没配强模型端点时，主脑不派复核（否则每条合同都在等一个不存在的服务）；
3. 插件位可用：`register_agent` 之后新节点进拓扑、能被调度、投递不被拒。
"""

from __future__ import annotations

import unittest

from zhilv_qiling_v2 import agents_multi as M
from zhilv_qiling_v2 import agents_multi_run as R
from zhilv_qiling_v2 import pipeline
from zhilv_qiling_v2.config import WORKSPACE, Config

SAMPLE = WORKSPACE / "samples" / "tech-development-contract.txt"


def _contract(config: Config | None = None):
    config = config or Config()
    doc = pipeline.prepare(SAMPLE.name, SAMPLE.read_bytes(), config=config)
    return pipeline.build_contract(doc, config)


class OrchestratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = Config()
        cls.out = R.run_multi_agent(_contract(config), config, strict_topology=True)

    def test_plan_dispatched_to_all_five_business_nodes(self):
        assigns = [m for m in self.out.bus.log if m.kind == "task.assign"]
        receivers = {m.receiver for m in assigns}
        self.assertEqual(receivers, {"ner", "logic", "subject", "statute", "judge"})
        for m in assigns:
            self.assertIn("tier", m.payload)
            self.assertEqual(m.sender, "orchestrator")

    def test_no_strong_endpoint_means_no_review_assignment(self):
        # 默认 Config 没有 strong 端点 → 主脑不该派复核
        assigns = [m for m in self.out.bus.log if m.kind == "review.assign"]
        self.assertEqual(assigns, [])
        # 但主脑确实被 logic 叫醒了（第二轮）
        planners = [s for s in self.out.steps if s["agent"] == "orchestrator"]
        self.assertGreaterEqual(len(planners), 2)

    def test_logic_reports_back_to_orchestrator(self):
        self.assertTrue(any(
            m.sender == "logic" and m.receiver == "orchestrator" and m.kind == "defects.detected"
            for m in self.out.bus.log
        ))

    def test_specs_cover_schedule(self):
        specs = M.agent_specs()
        for name in ("orchestrator", "reviewer", "judge", "statute"):
            self.assertIn(name, specs)
        self.assertEqual(set(M.TOPOLOGY), set(specs))


class PluginTest(unittest.TestCase):
    """插件位：加一个节点，拓扑与调度都要自动带上。"""

    def tearDown(self):
        M.PLUGIN_SPECS.pop("auditor", None)
        M.PLUGIN_HANDLERS.pop("auditor", None)
        M.TOPOLOGY.pop("auditor", None)
        M.TOPOLOGY["judge"] = ()

    def test_register_agent_extends_topology(self):
        def handle(ctx):
            ctx.emit("auditor", "judge", "audit.done", {"ok": True})
            return {"notes": ["插件节点跑过了"]}

        M.register_agent(
            "auditor",
            {"role": "合规复核", "executor": "custom", "model": "plugin"},
            handle,
            upstream=("statute",),
            downstream=("judge",),
        )
        self.assertTrue(M.MessageBus(strict=True).allowed("auditor", "judge"))
        self.assertTrue(M.MessageBus(strict=True).allowed("statute", "auditor"))
        self.assertIn("auditor", M.agent_specs())

        bus = M.MessageBus(strict=True)
        ctx = M.AgentContext(contract=_contract(), config=Config(), bus=bus)
        agent = R.build_agents(Config())["auditor"]
        agent.run(ctx)
        self.assertEqual(ctx.steps[-1]["status"], "ok")
        self.assertEqual(bus.stats()["rejected"], 0)
        self.assertEqual(len(bus.inbox("judge", kinds=("audit.done",))), 1)


if __name__ == "__main__":
    unittest.main()
