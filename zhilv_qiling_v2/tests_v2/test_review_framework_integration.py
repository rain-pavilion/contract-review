"""Three-layer reports are produced by execution rather than client inference."""
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from zhilv_qiling_v2 import pipeline, nodes, cli
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, ReviewReport, FinalRisk
from zhilv_qiling_v2.llm_client import LLMPair
from zhilv_qiling_v2.supervisor import Supervisor


def config(**options):
    return Config(ner_enable_model=False, ner_use_bert=False, enable_llm=False,
        use_lsa=False, entailment_backend="rules", **options)


def contract(text="甲方应在验收合格后10个工作日内支付全部尾款。"):
    return Contract("source-id", DocumentParse("合同.txt", "txt", text,
        clauses=[Clause("c1", "第一条", "付款", 1, "付款", text, text)]))


class ReviewFrameworkIntegrationTests(unittest.TestCase):
    def test_pipeline_preserves_detected_type_and_content_identity(self):
        document = contract("房屋租赁合同，出租人每月收取租金。").document
        first = pipeline.build_contract(document, config())
        selected = pipeline.build_contract(document, config(review_contract_type="通用", review_position="party_b"))
        self.assertEqual(first.contract_id, selected.contract_id)
        self.assertEqual(selected.contract_type, "通用")
        self.assertEqual(selected.review_context["detected_contract_type"], "房屋租赁")
        self.assertEqual(selected.review_context["contract_type_source"], "user")

    def test_reused_document_does_not_mutate_an_existing_contract_context(self):
        document = contract().document
        first = pipeline.build_contract(document, config(review_position="party_a"))
        pipeline.build_contract(document, config(review_position="party_b"))
        self.assertEqual(first.document.stats["review_context"]["review_position"], "party_a")
        self.assertNotIn("review_context", document.stats)

    def test_engine_binds_request_context_before_running_without_mutating_input(self):
        source = contract()
        engine = ReviewEngine(config(review_contract_type="委托", review_position="party_b"), retriever=False)
        seen = []
        def run(item):
            seen.append(item)
            return ReviewReport(item.contract_id, item.document.source_name)
        with patch.object(engine, "_run_review", side_effect=run):
            report = engine.run(source)
        self.assertEqual(seen[0].review_context["review_position"], "party_b")
        self.assertEqual(source.contract_type, "通用")
        self.assertEqual(source.review_context, {})
        self.assertEqual(report.stats["review_context"]["contract_type"], "委托")
        self.assertEqual(report.stats["review_point_checks"][0]["status"], "no_issue_detected")
        self.assertIn("review_framework", report.stats)
        payment = next(row for row in report.stats["review_framework"]["assessments"]
                       if row["point_id"] == "commercial.payment_terms")
        self.assertEqual(payment["status"], "no_issue_detected")
        self.assertEqual(report.score, 100)
        self.assertEqual(report.risks, [])

    def test_single_and_multi_both_persist_all_three_layers(self):
        for multi in (False, True):
            with self.subTest(multi=multi):
                result = ReviewEngine(config(enable_multi_agent=multi), retriever=False).run(contract())
                framework = result.stats["review_framework"]
                self.assertEqual(framework["schema_version"], "contract-review-three-layers-v1")
                self.assertEqual(len(framework["review_points"]), len(framework["assessments"]))
                self.assertEqual([s["id"] for s in framework["workflow"]["stages"]],
                    ["context", "parse", "analysis", "verification", "assessment", "human_review"])
                self.assertIn("工作流", result.to_markdown())
                self.assertIn("审查点", result.to_markdown())
                self.assertIn("评估结果", result.to_markdown())

    def test_logic_receives_selected_context_as_data(self):
        item = contract()
        item.review_context = {"contract_type": "委托", "review_position": "party_b", "context_fingerprint": "chosen"}
        captured = []
        def request(settings, messages, **kwargs):
            captured.append(json.loads(messages[-1]["content"]))
            return SimpleNamespace(content={"defects": []}, model="offline", elapsed=0, raw={})
        cfg = replace(config(), enable_llm=True, llm_endpoint="https://offline.example/v1",
                      llm_native_json_schema=False, llm_native_context_preflight=False)
        with patch.object(nodes.L, "available", return_value=True), patch.object(nodes.L, "request_json", side_effect=request):
            nodes._llm_logic(item, cfg, item.document.clauses)
        self.assertTrue(captured)
        self.assertEqual(captured[0]["review_context"]["review_position"], "party_b")
        points = captured[0].get("review_points", [])
        payment = next((p for p in points if p["point_id"] == "commercial.payment_terms"), None)
        self.assertIsNotNone(payment, "Execution must receive the independent criteria, not only a result projection")
        self.assertTrue(payment["question"])
        self.assertTrue(payment["criterion"])
        self.assertEqual(payment["version"], "review-points-v1")

    def test_planner_and_judge_receive_review_context_as_request_data(self):
        item = contract()
        item.review_context = {"contract_type": "委托", "review_position": "party_b", "context_fingerprint": "chosen"}
        captured = []
        def request(settings, messages, **kwargs):
            captured.append(json.loads(messages[-1]["content"]))
            return LLMPair({"action": "continue", "reason": "offline", "assignments": [],
                            "clause_ids": [], "focus_risk_ids": ["r1"]}, "offline", 0)
        cfg = replace(config(), enable_llm=True, planner_endpoint="https://offline.example/v1",
            planner_model="offline", llm_endpoint="https://offline.example/v1")
        with patch.object(nodes.L, "available", return_value=True), patch.object(nodes.L, "request_json", side_effect=request):
            Supervisor(cfg).decide("initial", {"clause_ids": ["c1"], "review_context": item.review_context})
            nodes.node_judge([FinalRisk("r1", "payment_ambiguous", "履约风险", "medium", "付款期限不明确")], cfg, item)
        self.assertEqual(captured[0]["context"]["review_context"]["review_position"], "party_b")
        self.assertEqual(captured[1]["review_context"]["review_position"], "party_b")

    def test_cli_review_batch_explain_accept_the_same_context_options(self):
        for arguments in (("review", "contract.txt"), ("batch",), ("explain", "contract.txt")):
            options = cli.build_parser().parse_args([*arguments, "--contract-type", "委托",
                                                     "--review-position", "party_b"])
            self.assertEqual(options.contract_type, "委托")
            self.assertEqual(options.review_position, "party_b")

    def test_cli_does_not_infer_pass_from_empty_risks(self):
        report = ReviewEngine(config(), retriever=False).run(contract("双方应当保守商业秘密。"))
        report.risks = []
        out = io.StringIO()
        with redirect_stdout(out):
            cli._print_report(report)
        self.assertIn("工作流", out.getvalue())
        self.assertIn("审查点", out.getvalue())
        self.assertNotIn("未发现实质风险。", out.getvalue())


if __name__ == "__main__":
    unittest.main()
