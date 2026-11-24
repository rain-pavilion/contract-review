"""Render the actual collaboration view for current and archived reports."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "zhilv_qiling_v2" / "webui.html"
NODE = shutil.which("node")


class SupervisorViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NODE:
            raise unittest.SkipTest("Node.js is required for renderer fixtures")
        html = PAGE.read_text(encoding="utf-8")
        cls.renderer = html[html.index("function renderGraph(){"):html.index("function renderFeedback(){")]
        cls.health_renderer = html[html.index("function modelReady("):html.index("function capability(")]
        cls.workflow_renderer = html[html.index("function hasSupervisor("):html.index("$('engine').addEventListener('change'")]

    def render(self, stats):
        report = {"stats": stats, "risks": [], "document": {"clauses": []}}
        script = """const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={report:JSON.parse(process.argv[1])},traceNames={},summary=x=>String(x||''),pending=()=>2,icon=()=>'',empty=x=>x,renderRuntimeSummary=()=>'',safeURL=()=>'',provenanceHTML=()=>'';
""" + self.renderer + "\nprocess.stdout.write(renderGraph());"
        result = subprocess.run([NODE, "-e", script, json.dumps(report, ensure_ascii=False)],
                                check=True, capture_output=True, encoding="utf-8")
        return result.stdout

    def test_multi_report_starts_with_supervisor_and_keeps_six_business_order(self):
        output = self.render({"multi_agent": {"enabled": True}, "supervisor": {
            "mode": "rules", "calls": 0, "decisions": [{"stage": "initial", "source": "rules",
                "action": "continue", "reason": "未配置云端主脑", "assignments": []}]}})
        self.assertIn("01 · 主脑规划与调度", output)
        self.assertIn("02 · 关键要素抽取", output)
        self.assertIn("06 · 综合评估与人工复核", output)
        self.assertIn("规则调度", output)
        self.assertIn("未配置云端主脑", output)

    def test_single_pipeline_does_not_invent_supervisor(self):
        output = self.render({"multi_agent": {"enabled": False}})
        self.assertNotIn("主脑规划与调度", output)
        self.assertIn("01 · 关键要素抽取", output)
        self.assertIn("05 · 综合评估与人工复核", output)

    def test_archived_multi_report_explains_missing_decisions(self):
        output = self.render({"multi_agent": {"enabled": True}, "agents": [
            {"agent": "orchestrator", "status": "ok", "model": "router"}]})
        self.assertIn("未保存主脑决策明细", output)
        self.assertNotIn("大模型决策", output)

    def test_supervisor_splits_to_peer_ner_and_logic_without_serial_edge(self):
        output = self.render({"multi_agent": {"enabled": True}})
        svg = ET.fromstring(output[output.index("<svg"):output.index("</svg>") + 6])
        nodes = {node.get("data-agent"): node for node in svg.findall(".//g[@data-agent]")}
        self.assertEqual(set(nodes), {"orchestrator", "ner", "logic", "subject", "statute", "judge"})
        self.assertEqual(nodes["ner"].find("rect").get("y"), nodes["logic"].find("rect").get("y"))
        edges = {edge.get("data-edge") for edge in svg.findall(".//path[@data-edge]")}
        self.assertIn("orchestrator->ner", edges)
        self.assertIn("orchestrator->logic", edges)
        self.assertNotIn("ner->logic", edges)
        self.assertIn("当前顺序调度，职责并列，不代表并发", output)

    def test_header_groups_peer_branches_and_retains_single_pipeline(self):
        def workflow(supervised):
            script = """const state={report:null},elements={engine:{value:'multi'},workflow:{innerHTML:'',setAttribute(){}},workflowLabel:{innerHTML:''}};
const $=id=>elements[id],icon=()=>'';
""" + self.workflow_renderer + "\nrenderWorkflow({stats:{multi_agent:{enabled:" + str(supervised).lower() + "}}});process.stdout.write(elements.workflow.innerHTML);"
            result = subprocess.run([NODE, "-e", script], check=True, capture_output=True, encoding="utf-8")
            return result.stdout
        multi = workflow(True)
        self.assertIn('data-branch-group="ner-logic"', multi)
        group = multi[multi.index('data-branch-group="ner-logic"'):multi.index('data-workflow-role="subject"')]
        self.assertIn('data-workflow-role="ner"', group)
        self.assertIn('data-workflow-role="logic"', group)
        self.assertIn("∥", group)
        single = workflow(False)
        self.assertNotIn("data-branch-group", single)
        self.assertNotIn('data-workflow-role="orchestrator"', single)
        self.assertEqual(single.count('data-workflow-role="'), 5)

    def test_decision_shows_model_and_manual_action_with_escaped_reason(self):
        output = self.render({"multi_agent": {"enabled": True}, "supervisor": {
            "mode": "llm", "calls": 1, "model_pool": [{"id": "expert", "model": "legal-expert"}],
            "decisions": [{"stage": "feedback", "source": "llm", "action": "manual",
                "reason": "证据不足<script>unsafe</script>", "clause_ids": ["c-1"],
                "assignments": [{"agent": "logic", "model_id": "expert", "reason": "核对授权"}]}]}})
        self.assertIn("大模型决策", output)
        self.assertIn("反馈调整", output)
        self.assertIn("转人工复核", output)
        self.assertIn("legal-expert", output)
        self.assertIn("核对授权", output)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", output)
        self.assertNotIn("<script>unsafe</script>", output)

    def health_controls(self, health):
        script = """const health=JSON.parse(process.argv[1]),state={health:null,busy:false,batchBusy:false},elements={};
const $=id=>elements[id]||(elements[id]={checked:false,disabled:true,textContent:'',className:''});
const api=async()=>health,updatePackageControls=()=>{};
""" + self.health_renderer + "\ncheckHealth().then(()=>process.stdout.write(JSON.stringify(elements)));"
        result = subprocess.run([NODE, "-e", script, json.dumps(health, ensure_ascii=False)],
                                check=True, capture_output=True, encoding="utf-8")
        return json.loads(result.stdout)

    def test_planner_only_configuration_enables_cloud_planner_without_expert_claim(self):
        controls = self.health_controls({"llm": {"endpoint": "", "enabled": False},
            "supervisor": {"configured": True, "enabled": True, "model": "cloud-planner"}})
        self.assertFalse(controls["useLlm"]["disabled"])
        self.assertTrue(controls["useLlm"]["checked"])
        self.assertIn("云端主脑", controls["modelSwitchText"]["textContent"])
        self.assertIn("cloud-planner", controls["modelHint"]["textContent"])
        self.assertIn("默认法律专家模型尚未配置", controls["modelHint"]["textContent"])

    def test_missing_or_disabled_planner_keeps_offline_toggle_disabled(self):
        for supervisor in ({"configured": False, "enabled": True},
                           {"configured": True, "enabled": False}):
            with self.subTest(supervisor=supervisor):
                controls = self.health_controls({"llm": {"endpoint": ""}, "supervisor": supervisor})
                self.assertTrue(controls["useLlm"]["disabled"])
                self.assertFalse(controls["useLlm"]["checked"])


if __name__ == "__main__":
    unittest.main()
