"""Run the real context-review renderer and Markdown exporter with Node."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
WEB = ROOT / "zhilv_qiling_v2" / "webui.html"
ESC = r'''const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));'''


class Tags(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class InitialText(HTMLParser):
    """Collect text outside closed details, matching the initial visible content."""
    def __init__(self, html):
        super().__init__()
        self.hidden_details = 0
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "details" and (self.hidden_details or "open" not in dict(attrs)):
            self.hidden_details += 1

    def handle_endtag(self, tag):
        if tag == "details" and self.hidden_details:
            self.hidden_details -= 1

    def handle_data(self, data):
        if not self.hidden_details:
            self.text.append(data)


@unittest.skipUnless(NODE, "Node.js is required for the context-review renderer fixtures")
class ContextReviewUITests(unittest.TestCase):
    longMessage = False

    def block(self, start, end):
        source = WEB.read_text(encoding="utf-8")
        self.assertIn(start, source, "The real context renderer must exist before it can be exercised")
        self.assertIn(end, source)
        return source[source.index(start):source.index(end, source.index(start))]

    def js(self, source):
        result = subprocess.run([NODE, "-e", source], capture_output=True, text=True,
                                encoding="utf-8", check=True)
        return result.stdout

    def render(self, report, markdown=False):
        source = self.block("function contextReview(", "function renderRisks(")
        call = "contextReviewMarkdown(state.report).join('\\n')" if markdown else "renderContextReview()"
        output = self.js(ESC + "\nconst empty=x=>x,state={report:" + json.dumps(report, ensure_ascii=False)
                         + "};\n" + source + "\nconst before=JSON.stringify(state.report),html=" + call
                         + ";process.stdout.write(JSON.stringify({html,unchanged:before===JSON.stringify(state.report)}));")
        result = json.loads(output)
        self.assertTrue(result["unchanged"], "Rendering must not rewrite the saved report")
        return result["html"]

    def fixture(self):
        evidence = [
            {"evidence_id": "premise-accept", "clause_id": "accept-1", "quote": "主合同第六条：验收合格后支付尾款。",
             "start": 12, "end": 30, "source_sha256": "sha-main", "source_id": "main", "document_id": "doc-main",
             "source_title": "技术服务主合同", "source_page_start": 2, "source_page_end": 3},
            {"evidence_id": "premise-deadline", "clause_id": "deadline-1", "quote": "附件一：收到申请后三日未提出异议视为验收。",
             "start": 41, "end": 64, "source_sha256": "sha-annex-1", "source_id": "annex-1", "document_id": "doc-annex-1",
             "source_title": "验收约定", "source_page_start": 4},
            {"evidence_id": "premise-test", "clause_id": "test-1", "quote": "附件二：必要测试时间不少于十日。完整原文末尾。",
             "start": 75, "end": 100, "source_sha256": "sha-annex-2", "source_id": "annex-2", "document_id": "doc-annex-2",
             "source_title": "测试要求", "source_page_start": 5},
        ]
        context = {
            "schema_version": "contract-context-review-v1", "execution_status": "completed",
            "source_documents": [{"source_id": "main", "document_id": "doc-main", "source_title": "技术服务主合同"},
                                 {"source_id": "annex-1", "source_title": "验收约定"},
                                 {"source_id": "annex-2", "source_title": "测试要求"}],
            "proposition_graph": {"nodes": [{"clause_id": "accept-1"}, {"clause_id": "deadline-1"}, {"clause_id": "test-1"}]},
            "constraints": {"status": "needs_manual", "scope_complete": False,
                            "violations": [{"constraint_id": "binding-1", "kind": "source_binding", "clause_ids": ["test-1"],
                                            "message": "测试条款的来源页仍需核对"}],
                            "checks": [{"check_id": "constraint-check", "status": "needs_manual", "message": "来源约束检查记录"}]},
            "findings": [{"finding_id": "context-1", "kind": "acceptance_timing", "title": "默认验收期限短于必要测试时间",
                          "point_id": "performance", "status": "issue_found", "actual_issue": "三日视为验收与十日测试要求存在组合张力",
                          "impact": "尾款可能在测试完成前触发", "recommendation": "明确测试完成后再开始默认验收期间",
                          "evidence": evidence,
                          "reasoning_steps": [{"step_id": "step-1", "claim": "验收触发尾款支付", "premise_ids": ["premise-accept"]},
                                              {"step_id": "step-2", "claim": "三日期限早于十日测试完成", "premise_ids": ["premise-deadline", "premise-test", "step-1"]}],
                          "assumptions": ["三个约定适用于同一项交付"], "pending_reasons": ["尚需人工确认优先约定"], "manual_required": True}],
            "checks": [{"check_id": "context-check", "point_id": "performance", "status": "issue_found",
                        "scope": "所给三处验收与支付文本", "actual_issue": "组合时序检查记录"}],
            "completeness": {"status": "not_evaluated", "reason": "齐全性检查尚未接入"},
            "summary": {"clause_count": 3, "source_count": 3, "constraint_violation_count": 1, "finding_count": 1, "pending_count": 1},
        }
        return {"risks": [], "contract_context_review": context}

    def test_render_keeps_all_evidence_sources_offsets_and_reasoning(self):
        output = self.render(self.fixture())
        for value in ("上下文条款：3", "来源文档：3", "图约束问题：1", "组合发现：1", "待人工核对：1",
                      "默认验收期限短于必要测试时间", "三日视为验收与十日测试要求存在组合张力", "尾款可能在测试完成前触发",
                      "明确测试完成后再开始默认验收期间", "附件二：必要测试时间不少于十日。完整原文末尾。", "技术服务主合同",
                      "验收约定", "测试要求", "doc-main", "sha-annex-2", "75", "100", "三日期限早于十日测试完成",
                      "premise-deadline", "premise-test", "step-1", "三个约定适用于同一项交付", "尚需人工确认优先约定"):
            self.assertIn(value, output)
        locates = [attrs["data-locate"] for _, attrs in Tags(output).tags if "data-locate" in attrs]
        for clause in ("accept-1", "deadline-1", "test-1"):
            self.assertIn(clause, locates)

    def test_constraint_problems_and_receipts_are_visible_with_scope_limits(self):
        output = self.render(self.fixture())
        for value in ("binding-1", "source_binding", "测试条款的来源页仍需核对", "约束检查范围不完整",
                      "来源约束检查记录", "组合时序检查记录", "所给三处验收与支付文本", "信息齐全性：未执行", "齐全性检查尚未接入"):
            self.assertIn(value, output)
        self.assertNotIn("全部通过", output)

    def test_constraint_validity_is_scoped_and_unknown_status_is_unconfirmed(self):
        for status, label in (("valid", "本次图约束未检出问题"), ("invalid", "发现图约束问题"), ("pass", "状态待核对")):
            report = self.fixture()
            report["contract_context_review"]["constraints"]["status"] = status
            output = self.render(report)
            self.assertIn(label, output)
            self.assertNotIn("全部通过", output)

    def test_completeness_shows_human_notes_and_collapses_raw_technical_record(self):
        report = self.fixture()
        report["contract_context_review"]["completeness"] = {
            "status": "not_evaluated", "message": "齐全性算法尚未运行", "reason": "等待补充检查方案",
            "pending_reasons": ["需要验收清单", "需要授权材料"], "execution_status": "not_run",
            "legal_truth_established": False, "internal_trace": "raw-record-only",
        }
        output = self.render(report)
        visible = " ".join(InitialText(output).text)
        for value in ("信息齐全性：未执行", "齐全性算法尚未运行", "等待补充检查方案", "需要验收清单", "需要授权材料"):
            self.assertIn(value, visible)
        for value in ("execution_status", "legal_truth_established", "raw-record-only"):
            self.assertNotIn(value, visible)
            self.assertIn(value, output, "The full raw record must remain available after expanding details")
        self.assertIn('<details class="details"><summary>原始齐全性记录</summary>', output)

    def test_constraint_receipt_kind_has_readable_title_and_unknown_kind_is_safe(self):
        report = self.fixture()
        report["contract_context_review"]["constraints"]["checks"] = [
            {"kind": "unique_clause_ids", "status": "valid"},
            {"kind": "source_quote_bindings", "status": "valid"},
            {"kind": "edge_endpoints", "status": "valid"},
            {"kind": '<img src=x onerror="bad()">', "status": "needs_manual"},
        ]
        output = self.render(report)
        for title in ("条款标识唯一性", "原文引文与来源绑定", "图关系端点完整性", "&lt;img src=x onerror=&quot;bad()&quot;&gt;"):
            self.assertIn('</span> ' + title + '<p class="help">', output)
        self.assertNotIn("<img", output)

    def test_hash_and_offset_scopes_are_distinct_in_html_and_markdown(self):
        report = self.fixture()
        context = report["contract_context_review"]
        context["source_documents"][0].update(text_sha256="joinedhash", text_hash_scope="joined_supplied_clause_text", clause_count=3)
        context["findings"][0]["evidence"][0].update(offset_scope="clause_text", original_sha256="originalfilehash")
        for markdown in (False, True):
            output = self.render(report, markdown=markdown)
            for value in ("偏移范围：条款文本", "原件文件 SHA256：originalfilehash", "来源文本汇总 SHA256：joinedhash", "已提供条款文本按换行拼接"):
                self.assertIn(value, output)

    def test_missing_clause_binding_never_creates_a_locate_button(self):
        report = self.fixture()
        context = report["contract_context_review"]
        context["constraints"]["violations"] = []
        context["findings"][0]["evidence"] = [{"quote": "未绑定条款的原文", "source_title": "待核来源"}]
        output = self.render(report)
        self.assertIn("未绑定条款的原文", output)
        self.assertNotIn("data-locate", output)

    def test_empty_completed_and_legacy_reports_do_not_become_passed(self):
        for report in ({"risks": []}, {"risks": [], "contract_context_review": {"execution_status": "completed", "findings": [],
                                                                                  "completeness": {"status": "not_evaluated"}}}):
            output = self.render(report)
            self.assertNotIn("全部通过", output)
            self.assertNotIn(">审查通过<", output)
            self.assertIn("未记录" if "contract_context_review" not in report else "未记录组合发现", output)
        self.assertIn("信息齐全性：未执行", self.render({"contract_context_review": {"completeness": {"status": "not_evaluated"}}}))

    def test_stats_fallback_and_top_level_precedence(self):
        report = self.fixture()
        context = report.pop("contract_context_review")
        report["stats"] = {"contract_context_review": context}
        self.assertIn("默认验收期限短于必要测试时间", self.render(report))
        report["contract_context_review"] = {"findings": [{"title": "顶层结果优先"}]}
        output = self.render(report)
        self.assertIn("顶层结果优先", output)
        self.assertNotIn("默认验收期限短于必要测试时间", output)
        report["contract_context_review"] = []
        self.assertIn("默认验收期限短于必要测试时间", self.render(report))

    def test_every_untrusted_layer_is_html_escaped(self):
        report = self.fixture()
        context = report["contract_context_review"]
        hostile = '<img src=x onerror="bad()"><script>bad()</script>'
        context["execution_status"] = hostile
        context["source_documents"][0].update(source_title=hostile, document_id=hostile)
        context["constraints"]["violations"][0].update(constraint_id=hostile, kind=hostile, message=hostile, clause_ids=['" onclick="bad()'])
        finding = context["findings"][0]
        finding.update(finding_id=hostile, title=hostile, actual_issue=hostile, impact=hostile, recommendation=hostile,
                       assumptions=[hostile], pending_reasons=[hostile], reasoning_steps=[{"step_id": hostile, "claim": hostile, "premise_ids": [hostile]}])
        finding["evidence"] = [{"clause_id": '\" onclick="bad()', "quote": hostile, "source_title": hostile,
                                "source_id": hostile, "document_id": hostile, "source_sha256": hostile, "start": hostile, "end": hostile}]
        context["completeness"] = {"status": "needs_manual", "message": hostile}
        output = self.render(report)
        self.assertIn("&lt;img", output)
        self.assertNotIn("<img", output)
        self.assertNotIn("<script>", output)
        self.assertFalse(any("onclick" in attrs or "onerror" in attrs for _, attrs in Tags(output).tags))

    def test_markdown_preserves_trace_and_escapes_html_and_markdown_links(self):
        report = self.fixture()
        finding = report["contract_context_review"]["findings"][0]
        finding["recommendation"] += '<script>bad()</script>[打开](javascript:alert(1))![图](x)'
        output = self.render(report, markdown=True)
        for value in ("上下文审查", "图约束", "默认验收期限短于必要测试时间", "完整原文末尾", r"sha\-annex", "偏移", "75", "100",
                      "推导步骤", "premise", "假设", "三个约定适用于同一项交付", "待核原因", "尚需人工确认优先约定", "信息齐全性：未执行", "限制"):
            self.assertIn(value, output)
        self.assertNotIn("<script>", output)
        self.assertNotIn("[打开](javascript:", output)
        self.assertNotIn("![图](x)", output)
        self.assertIn(r"\[打开\]\(javascript:", output)

    def test_markdown_keeps_legacy_and_callback_completeness_states(self):
        self.assertIn("未记录", self.render({"risks": []}, markdown=True))
        report = self.fixture()
        report["contract_context_review"]["completeness"] = {"status": "needs_manual", "message": "回调要求补充验收清单", "missing": ["验收清单"]}
        output = self.render(report)
        self.assertIn("回调要求补充验收清单", output)
        self.assertIn("验收清单", self.render(report, markdown=True))

    def test_tab_selection_executes_real_context_renderer(self):
        context = self.block("function contextReview(", "function renderRisks(")
        tab = self.block("function renderTab(", "function zeroRiskPendingHTML(")
        names = ("renderReviewWorkflow", "renderReviewPoints", "renderRisks", "renderExplanations", "renderSource", "renderElements", "renderGraph", "renderPackageReport", "renderFeedback")
        stubs = ";".join("const " + name + "=()=>\"unexpected view\"" for name in names)
        output = self.js(ESC + "\nconst empty=x=>x,state={tab:'context',report:" + json.dumps(self.fixture(), ensure_ascii=False)
                         + "},element={innerHTML:''},$=()=>element;" + stubs + ";\n" + context + tab
                         + "\nrenderTab();process.stdout.write(element.innerHTML);")
        self.assertIn("默认验收期限短于必要测试时间", output)
        tabs = self.block("const tabs=", "function ocrWarning(")
        self.assertIn(["context", "上下文审查"], json.loads(self.js(tabs + "\nprocess.stdout.write(JSON.stringify(tabs));")))

    def test_export_markdown_includes_context_output(self):
        report = self.fixture()
        report.update(source_name="测试合同", contract_id="contract-1", score=100, stats={}, trace=[])
        context = self.block("function contextReview(", "function renderRisks(")
        export = WEB.read_text(encoding="utf-8")
        export = export[export.index("function exportMarkdown("):]
        export = export[:export.index("\nfunction ", 1)]
        output = self.js(ESC + "\nconst state={report:" + json.dumps(report, ensure_ascii=False)
                         + "},seconds=()=>0,sevCN={},labels={},frameworkMarkdown=()=>[],judgeEvidenceMarkdown=()=>[],packageMarkdown=()=>[],explanationMarkdown=()=>[],filename=()=>'',download=(data)=>process.stdout.write(data);\n"
                         + context + export + "\nexportMarkdown();")
        self.assertIn("默认验收期限短于必要测试时间", output)
        self.assertIn("信息齐全性：未执行", output)


if __name__ == "__main__":
    unittest.main()
