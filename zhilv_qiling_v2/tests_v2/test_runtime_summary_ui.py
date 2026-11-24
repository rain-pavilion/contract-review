"""Runtime summary fixtures execute the page's own renderer without a browser."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "zhilv_qiling_v2" / "webui.html"
NODE = shutil.which("node")


class RuntimeSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NODE:
            raise unittest.SkipTest("Node.js is required for the page renderer fixtures")
        html = PAGE.read_text(encoding="utf-8")
        cls.renderer = html[html.index("function renderRuntimeSummary(r){"):html.index("function renderGraph(){")]

    def render(self, logic=None, cross=None, **extra):
        report = {"stats": {"logic_llm": logic or {}, "cross_validation": cross or {}}, **extra}
        script = ("const esc=value=>String(value??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));\n"
                  + self.renderer + "\nprocess.stdout.write(renderRuntimeSummary(JSON.parse(process.argv[1])));")
        result = subprocess.run([NODE, "-e", script, json.dumps(report, ensure_ascii=False)],
                                check=True, capture_output=True, encoding="utf-8")
        return result.stdout

    def paragraph(self, output, label):
        start = output.index('<p class="rationale">' + label + '：')
        return output[start:output.index("</p>", start)]

    def test_full_text_with_pending_reference_is_incomplete(self):
        output = self.render({"text_coverage": 1, "explicit_reference_complete": False,
            "explicit_reference_pending_checks": 8,
            "explicit_reference_checks": [{"checked_checks": 1, "pending_checks": 8}]})
        self.assertIn("正文覆盖率 100.0%", output)
        self.assertIn("显式引用依赖：未完成；实际完成 1 次检查，待完成 8 次", output)
        self.assertIn("不代表显式引用依赖与逐项反馈核验全部完成", output)

    def test_unresolved_or_ambiguous_references_override_zero_pending(self):
        for field in ("explicit_reference_unresolved_targets", "explicit_reference_ambiguous_targets"):
            with self.subTest(field=field):
                output = self.render({"explicit_reference_complete": True,
                    "explicit_reference_pending_checks": 0, field: [{"reference_text": "第2条"}]})
                self.assertIn("显式引用依赖：未完成", output)

    def test_old_report_does_not_imply_completion(self):
        output = self.render({"text_coverage": 1, "complete": True})
        self.assertIn("显式引用依赖：未记录。", output)
        self.assertIn("逐项反馈核验：未记录。", output)

    def test_latest_success_does_not_repeat_historical_pending(self):
        failed = {"feedback_items": [{"feedback_item_id": "fb1"}], "feedback_complete": False,
                  "complete": False, "uncompleted_feedback_item_ids": ["fb1"]}
        success = {"feedback_items": [{"feedback_item_id": "fb1"}], "feedback_complete": True,
                   "complete": True, "verified_feedback_item_ids": ["fb1"]}
        output = self.render({"feedback_runs": [failed, success], "feedback_latest_runs": [success],
            "feedback_runs_complete": True, "feedback_all_attempts_complete": False},
            {"feedback_verification_complete": True, "pending_feedback_item_ids": []})
        self.assertIn("逐项反馈核验：已完成；已核验 1 项，待核验 0 项。", output)
        self.assertIn("历史尝试 2 次", output)

    def test_pending_without_item_ids_cannot_become_complete(self):
        for cross in ({"feedback_verification_complete": False},
                      {"feedback_verification_complete": True, "pending_feedback_fingerprints": ["fp"]}):
            with self.subTest(cross=cross):
                output = self.render({"feedback_latest_runs": [], "feedback_runs_complete": True}, cross)
                paragraph = self.paragraph(output, "逐项反馈核验")
                self.assertIn("逐项反馈核验：未完成", paragraph)
                self.assertIn("仍有待核验反馈，具体条目未记录", paragraph)

    def test_empty_feedback_is_not_claimed_as_performed(self):
        output = self.render({"feedback_items": [], "feedback_complete": True,
            "feedback_latest_runs": [], "feedback_runs_complete": True},
            {"feedback_verification_complete": True, "pending_feedback_item_ids": []})
        paragraph = self.paragraph(output, "逐项反馈核验")
        self.assertIn("本次未触发逐项反馈核验", paragraph)
        self.assertNotIn("已核验 0 项", paragraph)

    def test_incomplete_primary_text_run_does_not_invent_feedback(self):
        output = self.render({"complete": False, "feedback_complete": True,
            "feedback_items": [], "uncompleted_feedback_item_ids": []})
        self.assertIn("逐项反馈核验：本次未触发逐项反馈核验。", output)

    def test_reference_plan_is_not_actual_execution(self):
        output = self.render({"explicit_reference_plan_complete": True,
            "explicit_reference_dependencies": [{"scheduled_checks": 9, "pending_checks": 0}]})
        self.assertIn("显式引用依赖：未记录。", output)

    def test_missing_reference_details_do_not_turn_into_zero_counts(self):
        output = self.render({"explicit_reference_complete": True})
        paragraph = self.paragraph(output, "显式引用依赖")
        self.assertIn("实际完成 未记录 次检查", paragraph)
        self.assertNotIn("本次未检出", paragraph)

    def test_feedback_ids_are_deduplicated(self):
        run = {"feedback_items": [{"feedback_item_id": "f1"}, {"feedback_item_id": "f1"}],
               "verified_feedback_item_ids": ["f1", "f1"], "feedback_complete": True}
        output = self.render({"feedback_latest_runs": [run, run], "feedback_runs_complete": True})
        self.assertIn("逐项反馈核验：已完成；已核验 1 项，待核验 0 项。", output)

    def test_unverified_and_manual_items_stay_pending(self):
        run = {"feedback_items": [{"feedback_item_id": "f1", "article_quote_truncated": True},
                                  {"feedback_item_id": "f2"}, {"feedback_item_id": "f3"}],
               "verified_feedback_item_ids": ["f1", "f3"], "uncompleted_feedback_item_ids": ["f1"],
               "feedback_complete": False, "calls": [{"verifications": [
                   {"feedback_item_id": "f2", "verdict": "needs_manual"}]}]}
        output = self.render({"feedback_latest_runs": [run]})
        self.assertIn("逐项反馈核验：未完成；已核验 1 项，待核验 2 项。", output)
        self.assertIn("需人工核对或补充原文 2 项", output)

    def test_legacy_history_uses_latest_attempt_only(self):
        failed = {"uncompleted_feedback_item_ids": ["old"], "feedback_complete": False}
        success = {"feedback_items": [{"feedback_item_id": "new"}],
                   "verified_feedback_item_ids": ["new"], "feedback_complete": True}
        output = self.render({"feedback_runs": [failed, success]})
        self.assertIn("逐项反馈核验：已完成；已核验 1 项，待核验 0 项。", output)
        self.assertIn("旧报告仅展示最近一次反馈尝试", output)

    def test_subject_fact_manual_verification_is_counted(self):
        run = {"feedback_items": [{"feedback_item_id": "subject1", "feedback_kind": "subject_fact"}],
               "uncompleted_feedback_item_ids": ["subject1"], "feedback_complete": False,
               "calls": [{"subject_fact_verifications": [
                   {"feedback_item_id": "subject1", "verdict": "needs_manual"}]}]}
        output = self.render({"feedback_latest_runs": [run]})
        self.assertIn("逐项反馈核验：未完成；已核验 0 项，待核验 1 项。", output)
        self.assertIn("需人工核对或补充原文 1 项", output)

    def test_unverified_items_override_success_flags(self):
        output = self.render({"feedback_latest_runs": [{"feedback_items": [{"feedback_item_id": "f1"}],
            "feedback_complete": True, "complete": True}], "feedback_runs_complete": True},
            {"feedback_verification_complete": True})
        self.assertIn("逐项反馈核验：未完成；已核验 0 项，待核验 1 项。", output)

    def test_model_and_raw_details_are_escaped(self):
        output = self.render({"model": "<img src=x onerror=bad>", "calls": [{"status": "<script>bad</script>"}]})
        self.assertNotIn("<script>bad</script>", output)
        self.assertNotIn("<img src=x", output)
        self.assertIn("&lt;script&gt;bad&lt;/script&gt;", output)


if __name__ == "__main__":
    unittest.main()
