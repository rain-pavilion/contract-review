"""Context findings survive both engines, API storage and report export."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, ReviewReport


def context_contract():
    texts = [
        "甲方应在本设备最终验收合格后支付尾款。",
        "本设备收货后3日内未提出书面异议，视为最终验收合格。",
        "本设备最终验收必须包含性能测试，该测试最早在收货后第5日完成。",
    ]
    clauses = [Clause(f"c{i}", "第一条", title, 1, title, text, text,
        metadata={"document_id": f"doc{i}", "source_id": f"source{i}",
                  "source_title": title, "source_page_start": 1})
        for i, (title, text) in enumerate(zip(("主合同", "验收附件", "测试附件"), texts), 1)]
    return Contract("context-fixture", DocumentParse("合同及附件", "txt", "\n".join(texts), clauses=clauses))


def offline_config(root=None, **options):
    root = Path(root) if root else Path(tempfile.gettempdir())
    return Config(ner_enable_model=False, ner_use_bert=False, enable_llm=False,
        enable_subject_check=False, use_lsa=False, entailment_backend="rules",
        statutes_path=str(root / "context-review-no-statutes.jsonl"),
        subject_facts_path=str(root / "context-review-no-facts.json"), **options)


class ContextReviewIntegrationTests(unittest.TestCase):
    def test_engine_can_inject_the_future_completeness_checker(self):
        def completeness(contract, *, context):
            return {"status": "needs_manual", "message": "采购明细待补充",
                    "clause_count": context["summary"]["clause_count"]}
        report = ReviewEngine(offline_config(), retriever=False,
            completeness_check=completeness).run(context_contract())
        self.assertEqual(report.stats["contract_context_review"]["completeness"]["status"], "needs_manual")
        self.assertEqual(report.stats["contract_context_review"]["completeness"]["clause_count"], 3)

    def test_both_engines_expose_three_source_finding_and_point_receipt(self):
        original = context_contract()
        before = copy.deepcopy(original.to_dict())
        for multi in (False, True):
            with self.subTest(multi=multi):
                report = ReviewEngine(offline_config(enable_multi_agent=multi), retriever=False).run(original)
                context = report.stats["contract_context_review"]
                self.assertEqual(context["execution_status"], "completed")
                finding = next(row for row in context["findings"]
                    if row["kind"] == "acceptance_before_required_test")
                self.assertEqual({row["clause_id"] for row in finding["evidence"]}, {"c1", "c2", "c3"})
                self.assertTrue(finding["reasoning_steps"])
                self.assertTrue(finding["manual_required"])
                self.assertEqual(context["completeness"]["status"], "not_evaluated")
                point = next(row for row in report.stats["review_framework"]["assessments"]
                    if row["point_id"] == "performance.acceptance")
                self.assertEqual(point["status"], "issue_found")
                self.assertTrue(point["check_ids"])
                self.assertEqual(json.loads(report.to_json())["stats"]["contract_context_review"], context)
        self.assertEqual(original.review_context, before["review_context"])
        self.assertEqual(original.document.full_text, before["document"]["full_text"])
        self.assertNotIn("contract_context_review", original.document.stats)

    def test_api_archive_preserves_context_evidence_and_does_not_recompute_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = offline_config(temporary, store_path=str(Path(temporary) / "workspace.sqlite"))
            report = ReviewEngine(config, retriever=False).run(context_contract())
            payload = api._report_payload(report)
            payload["created_at"] = api._workspace_now()
            context = payload["contract_context_review"]
            self.assertEqual(context, payload["stats"]["contract_context_review"])
            api._save_workspace_report(payload, config)
            previous = api._SERVER_CONFIG
            api._SERVER_CONFIG = config
            try:
                ctx = api.RequestContext(handler=SimpleNamespace(), method="GET", path="/api/workspace/reports/report",
                    query={}, body=b"", json_body={}, trace_id="context-readback",
                    params={"report_id": payload["report_id"]})
                saved = api.handle_workspace_report(ctx)
            finally:
                api._SERVER_CONFIG = previous
            self.assertEqual(saved["contract_context_review"], context)
            self.assertTrue(saved["contract_context_review"]["findings"][0]["evidence"])

    def test_markdown_keeps_source_quotes_reasoning_and_completeness_state(self):
        report = ReviewEngine(offline_config(), retriever=False).run(context_contract())
        markdown = report.to_markdown()
        for expected in ("跨条款与跨附件上下文审查", "本设备收货后3日内未提出书面异议", "测试附件", "推导步骤", "信息齐全性", "尚未执行"):
            self.assertIn(expected, markdown)

    def test_markdown_escapes_untrusted_context_content_and_legacy_stays_readable(self):
        report = ReviewReport("test", "合同", stats={"contract_context_review": {
            "schema_version": "contract-context-review-v1", "summary": {},
            "completeness": {"status": "not_evaluated"}, "constraints": {},
            "findings": [{"title": "<script>alert(1)</script>", "status": "needs_manual",
                "actual_issue": "[危险](javascript:alert(1))", "evidence": [],
                "reasoning_steps": [{"claim": "<img src=x onerror=bad()>"}],
                "assumptions": [], "pending_reasons": []}]}})
        rendered = report.to_markdown()
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("[危险](javascript:", rendered)
        self.assertIn("未保存三层审查记录", ReviewReport("legacy", "旧合同").to_markdown())

    def test_markdown_labels_the_evidence_referenced_by_a_reasoning_step(self):
        from zhilv_qiling_v2.context_review_report import context_review_markdown
        context = {"findings": [{"title": "组合线索", "evidence": [
            {"evidence_id": "premiseA", "clause_id": "c1", "quote": "验收后付款"}],
            "reasoning_steps": [{"step_id": "s1", "claim": "付款以验收为条件", "premise_ids": ["premiseA"]}]}]}
        rendered = "\n".join(context_review_markdown(context))
        self.assertEqual(rendered.count("premiseA"), 2,
            "The step reference and its original evidence must both carry the evidence ID")


if __name__ == "__main__":
    unittest.main()
