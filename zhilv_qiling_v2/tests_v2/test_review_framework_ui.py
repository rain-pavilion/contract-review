"""Execute the real three-layer review renderers with saved-report fixtures."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
WEB = ROOT / "zhilv_qiling_v2" / "webui.html"
LIBRARY = ROOT / "zhilv_qiling_v2" / "library.html"
ESC = r'''const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));'''


@unittest.skipUnless(NODE, "Node.js is required for the review-framework renderer fixtures")
class ReviewFrameworkUITests(unittest.TestCase):
    longMessage = False

    def block(self, path, start, end):
        source = path.read_text(encoding="utf-8")
        self.assertIn(start, source)
        self.assertIn(end, source)
        return source[source.index(start):source.index(end, source.index(start))]

    def js(self, source):
        result = subprocess.run([NODE, "-e", source], capture_output=True, text=True,
                                encoding="utf-8", check=True)
        return result.stdout

    def render(self, report, view="points"):
        block = self.block(WEB, "function reviewFramework(", "function renderRisks(")
        call = {"points": "renderReviewPoints()", "workflow": "renderReviewWorkflow()",
                "markdown": "frameworkMarkdown(state.report).join('\\n')"}[view]
        return self.js(ESC + "\nconst empty=x=>x,state={report:" + json.dumps(report, ensure_ascii=False)
                       + "};\n" + block + "\nprocess.stdout.write(" + call + ");")

    def fixture(self, status=None, **extra):
        point = {"point_id": "payment", "domain": "价款与支付", "question": "付款条件是否清楚？",
                 "criterion": "付款节点、金额与触发条件可对应", "responsible_agent": "logic",
                 "required_evidence": ["支付条款"], "scope": "全文"}
        assessment = {"point_id": "payment", "actual_issue": "付款时间待确认", "manual_required": False,
                      "impact": "可能影响回款", "recommendation": "补充付款节点", **extra}
        if status is not None:
            assessment["status"] = status
        return {"risks": [], "review_framework": {"schema_version": "contract-review-three-layers-v1",
                "context": {"contract_type": "技术开发", "review_position": "party_b", "position_label": "乙方"},
                "workflow": {"status": "pending", "manual_required": True, "pending_count": 1,
                             "stages": [{"id": "review", "label": "合同审查", "agents": ["ner", "logic"],
                                         "status": "pending", "gate": "主体授权材料待补"}]},
                "review_points": [point], "assessments": [assessment]}}

    def results(self, report, filter_="all"):
        framework = self.block(WEB, "function reviewFramework(", "function renderRisks(")
        risks = self.block(WEB, "function renderRisks(", "function safeURL(")
        result = json.loads(self.js(ESC + "\nconst state={report:" + json.dumps(report, ensure_ascii=False)
                            + ",filter:" + json.dumps(filter_) + "},empty=x=>x,icon=()=>'',zeroRiskPendingHTML=()=>'',riskHTML=r=>'<article data-legacy-risk=\"'+esc(r.risk_id)+'\">'+esc(r.message)+'</article>';\n"
                            + framework + risks + "\nconst before=JSON.stringify(state.report),html=renderRisks();process.stdout.write(JSON.stringify({html,unchanged:before===JSON.stringify(state.report)}));"))
        self.assertTrue(result["unchanged"])
        return result["html"]

    def test_results_show_independent_payment_finding_with_zero_legacy_risks(self):
        report = self.fixture("issue_found", check_ids=["check-payment"], risk_ids=[], scope="第一条付款期限",
                              actual_issue="未明确验收后应在多少日内付款", impact="可能延迟收款", recommendation="约定验收后30日内付款",
                              clause_ids=["clause-payment"], evidence=[{"clause_id": "clause-payment", "quote": "验收后支付", "located": True}])
        report.update(score=100, risk_count=0, saved=True)
        output = self.results(report)
        for text in ("独立审查结果", "未明确验收后应在多少日内付款", "可能延迟收款", "约定验收后30日内付款", "第一条付款期限", "验收后支付"):
            self.assertIn(text, output)
        self.assertIn('data-locate="clause-payment"', output)
        self.assertIn("不计入风险数量", output)

    def test_independent_results_exclude_unchecked_unreceipted_and_linked_risks(self):
        report = self.fixture("issue_found", check_ids=["check-independent"], question="独立付款发现")
        report["review_framework"]["assessments"] += [
            {"point_id": "unchecked", "question": "尚未运行不应展示", "check_ids": ["check-unrun"], "status": "not_checked"},
            {"point_id": "no-receipt", "question": "无检查记录不应展示", "status": "issue_found"},
            {"point_id": "linked", "question": "已关联旧风险不应重复", "check_ids": ["check-linked"], "risk_ids": ["legacy-risk"], "status": "issue_found"}]
        report.update(saved=True, risks=[{"risk_id": "legacy-risk", "severity": "medium", "message": "既有风险保留"}])
        output = self.results(report)
        self.assertIn("独立付款发现", output)
        self.assertIn("既有风险保留", output)
        for text in ("尚未运行不应展示", "无检查记录不应展示", "已关联旧风险不应重复"):
            self.assertNotIn(text, output)

    def test_independent_manual_and_negative_results_keep_scope_and_safe_text_under_risk_filter(self):
        for status, label in (("needs_manual", "待人工确认"), ("no_issue_detected", "本范围未检出问题")):
            report = self.fixture(status, check_ids=["check-1"], scope="仅第二条付款时间", actual_issue='<img src=x onerror="bad()">',
                                  recommendation="<script>不要执行</script>")
            report.update(saved=True, risks=[{"risk_id": "medium-risk", "severity": "medium", "message": "中风险筛掉"}])
            output = self.results(report, "high")
            self.assertIn(label, output)
            self.assertIn("仅第二条付款时间", output)
            self.assertIn("&lt;img", output)
            self.assertIn("风险级别筛选仅作用于风险条目", output)
            self.assertNotIn("中风险筛掉", output)
            self.assertNotIn("<script>", output)
            self.assertNotIn("<img", output)

    def test_results_without_framework_keep_original_empty_state(self):
        output = self.results({"risks": [], "saved": True, "score": 100})
        self.assertIn("当前筛选下没有检出风险", output)
        self.assertNotIn("独立审查结果", output)

    def test_evaluation_tab_labels_legacy_counter_as_risks(self):
        report = self.fixture("issue_found", check_ids=["check-payment"], risk_ids=[])
        report.update(saved=True, score=100, risk_count=0)
        tabs = self.block(WEB, "const tabs=", "function ocrWarning(")
        renderer = self.block(WEB, "function renderReport(", "function renderTab(")
        output = self.js(ESC + "\nconst state={report:" + json.dumps(report, ensure_ascii=False) + ",tab:'risks'},elements={},$=id=>elements[id]||(elements[id]={innerHTML:'',classList:{add(){}}});const icon=()=>'',renderWorkflow=()=>{},scoreColor=()=>'',seconds=()=>'-',pending=()=>0,ocrWarning=()=>'',packageReportNotice=()=>'',judgeEvidenceHTML=()=>'',renderTab=()=>{};\n"
                         + tabs + renderer + "\nrenderReport();process.stdout.write(elements.result.innerHTML);")
        self.assertRegex(output, r'data-tab="risks"[^>]*>评估结果 · 风险 0</button>')

    def test_legacy_report_does_not_invent_three_layer_records(self):
        for view in ("points", "workflow"):
            self.assertIn("未保存三层审查记录", self.render({"risks": [], "score": 100}, view))

    def test_missing_or_unknown_status_never_becomes_pass(self):
        for status in (None, "pass", "unknown"):
            output = self.render(self.fixture(status))
            self.assertIn("未记录", output)
            self.assertNotIn(">审查通过<", output)
            self.assertNotIn("本范围未检出问题", output)

    def test_catalog_point_without_assessment_stays_not_checked(self):
        report = self.fixture()
        report["review_framework"]["assessments"] = []
        output = self.render(report)
        self.assertIn("付款条件是否清楚", output)
        self.assertIn("尚未执行", output)

    def test_manual_requirement_overrides_positive_status(self):
        output = self.render(self.fixture("no_issue_detected", manual_required=True,
                                          pending_reasons=["主体授权仍待核验"]))
        self.assertIn("待人工确认", output)
        self.assertIn("主体授权仍待核验", output)
        self.assertNotIn("本范围未检出问题", output)

    def test_point_result_preserves_criterion_impact_recommendation_and_evidence_links(self):
        output = self.render(self.fixture("issue_found", risk_ids=["risk-1"], explanation_item_ids=["work-1"],
                            clause_ids=["clause-1"], evidence=[{"clause_id": "clause-1", "quote": "验收后付款", "located": True}]))
        for text in ("付款条件是否清楚", "付款节点、金额与触发条件可对应", "付款时间待确认", "可能影响回款", "补充付款节点", "验收后付款"):
            self.assertIn(text, output)
        for attr in ('data-locate="clause-1"', 'data-open-risk="risk-1"', 'data-open-explanation="work-1"', 'data-review="risk-1"'):
            self.assertIn(attr, output)

    def test_workflow_shows_actual_gates_and_keeps_peer_responsibilities(self):
        output = self.render(self.fixture("needs_manual"), "workflow")
        for text in ("技术开发", "乙方", "主体授权材料待补", "要素抽取", "逻辑审查", "并列职责", "顺序调度"):
            self.assertIn(text, output)
        self.assertNotIn("全部完成", output)
        self.assertIn("待处理记录：1 条", output)
        self.assertIn("包含关联证据、执行门槛和复核记录", output)
        self.assertIn("不等同于风险数量", output)

    def test_fixed_framework_enums_are_readable_and_unknown_values_stay_safe(self):
        report = self.fixture("issue_found", check_ids=["check-1"], impact_basis="explicit_check",
                              scope="provided_contract_text_payment_sentences_only 仅为该范围的文本检查。")
        report["review_framework"]["review_points"][0].update(domain="commercial", responsible_agent="logic")
        output = self.render(report)
        for text in ("商业条款", "逻辑审查", "独立检查记录", "仅限所给合同文本中的付款句"):
            self.assertIn(text, output)
        for text in ("commercial", "explicit_check", "provided_contract_text_payment_sentences_only"):
            self.assertNotIn(text, output)
        self.assertIn("仅限所给合同文本中的付款句", self.results(report))
        markdown = self.render(report, "markdown")
        self.assertIn("独立检查记录", markdown)
        self.assertIn("仅限所给合同文本中的付款句", markdown)
        report["review_framework"]["review_points"][0]["domain"] = "custom<script>unsafe</script>"
        self.assertIn("custom&lt;script&gt;unsafe&lt;/script&gt;", self.render(report))

    def test_engine_gate_objects_and_partial_stage_states_are_preserved(self):
        report = self.fixture("needs_manual")
        report["review_framework"]["workflow"] = {"status": "partially_checked", "stages": [
            {"id": "context", "status": "recorded", "gate": {"status": "open", "reasons": []}},
            {"id": "verification", "status": "partial", "gate": {"status": "blocked", "reasons": ["subject_evidence_pending<script>bad</script>"]}},
            {"id": "human_review", "status": "required", "gate": {"status": "pending", "reasons": ["human_review_not_recorded"]}}]}
        output = self.render(report, "workflow")
        for text in ("部分范围已有执行记录", "已有背景记录", "部分执行", "需要人工处理", "subject_evidence_pending", "human_review_not_recorded", "&quot;status&quot;:&quot;blocked&quot;"):
            self.assertIn(text, output)
        self.assertNotIn("<script>", output)

    def test_hostile_backend_fields_are_escaped_in_every_layer(self):
        report = self.fixture("issue_found", actual_issue='<img src=x onerror="alert(1)">',
                              evidence=[{"clause_id": '\" onclick="alert(1)', "quote": "<script>bad()</script>"}])
        report["review_framework"]["context"]["position_label"] = "<script>position</script>"
        report["review_framework"]["workflow"]["stages"][0]["gate"] = "<script>gate</script>"
        for view in ("points", "workflow"):
            output = self.render(report, view)
            self.assertNotIn("<script>", output)
            self.assertNotIn("<img src", output)
            self.assertNotIn('data-locate="" onclick=', output)
            self.assertIn("&lt;", output)

    def test_stats_fallback_and_explicit_no_issue_scope(self):
        report = self.fixture("no_issue_detected", scope="第3条付款时间")
        report["stats"] = {"review_framework": report.pop("review_framework")}
        output = self.render(report)
        self.assertIn("本范围未检出问题", output)
        self.assertIn("第3条付款时间", output)
        self.assertIn("不代表合同整体审查通过", output)

    def test_markdown_exports_layers_and_pending_without_raw_html(self):
        output = self.render(self.fixture("needs_manual", pending_reasons=["缺少授权"],
                                         actual_issue="<script>bad()</script>"), "markdown")
        for text in ("工作流", "审查点与评估结果", "技术开发", "乙方", "缺少授权", "付款条件是否清楚", "补充付款节点"):
            self.assertIn(text, output)
        self.assertNotIn("<script>", output)
        page = WEB.read_text(encoding="utf-8")
        self.assertIn("...frameworkMarkdown(r)", page)

    def test_context_choices_and_transport_use_canonical_fields(self):
        page = WEB.read_text(encoding="utf-8")
        for id_ in ("reviewContractType", "reviewPosition"):
            self.assertIn('id="' + id_ + '"', page)
        options = self.block(WEB, "function reviewContextOptions(", "function hasSupervisor(")
        result = json.loads(self.js("const elements={engine:{value:'multi'},useLlm:{checked:true},reviewContractType:{value:'政府采购'},reviewPosition:{value:'party_a'}},$=id=>elements[id];\n"
                                   + options + "\nprocess.stdout.write(JSON.stringify(runOptions()));"))
        self.assertEqual(result["contract_type"], "政府采购")
        self.assertEqual(result["review_position"], "party_a")
        self.assertIn("...runOptions(),engine:'multi'", page)
        self.assertRegex(page, r"function busy\(on\).*reviewContractType.*reviewPosition")
        for type_ in ("auto", "通用", "房屋租赁", "商品房预售", "房屋买卖", "委托", "中介", "保管", "数据提供", "技术开发", "政府采购", "承揽"):
            self.assertIn('value="' + type_ + '"', page)
        self.assertIn("['risks','评估结果']", page)
        self.assertIn("['workflow','工作流']", page)
        self.assertIn("['points','审查点']", page)

    def test_library_enqueue_and_report_include_context_and_layers(self):
        page = LIBRARY.read_text(encoding="utf-8")
        self.assertIn("reviewContractType", page)
        self.assertIn("reviewPosition", page)
        self.assertIn("contract_type:contractType.input.value", page)
        self.assertIn("review_position:position.input.value", page)
        self.assertIn("renderLibraryReviewFramework(payload)", page)
        block = self.block(LIBRARY, "  function renderLibraryReviewFramework(", "  async function openReport(")
        self.assertNotIn("innerHTML", block)
        self.assertIn("未保存三层审查记录", block)

    def test_json_multipart_and_model_queue_keep_selected_context(self):
        options = self.block(WEB, "function reviewContextOptions(", "function hasSupervisor(")
        request = self.block(WEB, "async function request(", "function busy(")
        source = """const elements={engine:{value:'multi'},useLlm:{checked:false},reviewContractType:{value:'房屋租赁'},reviewPosition:{value:'party_b'}},$=id=>elements[id];
const calls=[],post=async(path,body)=>{calls.push({path,body});return {batch_id:'batch-1'};},api=async(path,options)=>{calls.push({path,body:Object.fromEntries([...options.body].filter(([k])=>k!=='file'))});return {};};
const taskLink=()=>null,reportLink=()=>null,followSingleTask=async()=>({}),history={replaceState(){}},location={pathname:'/',search:''};
""" + options + request + """
(async()=>{await request(null,'合同原文','测试合同');await request(new Blob(['合同原文']),'','上传合同');elements.useLlm.checked=true;await request(null,'合同原文','模型队列');process.stdout.write(JSON.stringify(calls));})();"""
        calls = json.loads(self.js(source))
        payloads = [call["body"] for call in calls if call["path"] in ("/api/review", "/api/workspace/batches")]
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            self.assertEqual(payload["contract_type"], "房屋租赁")
            self.assertEqual(payload["review_position"], "party_b")

    def test_inline_scripts_are_valid_javascript(self):
        for path in (WEB, LIBRARY):
            for script in re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", path.read_text(encoding="utf-8")):
                result = subprocess.run([NODE, "--check"], input=script, text=True, encoding="utf-8", capture_output=True)
                self.assertEqual(result.returncode, 0, path.name + ": " + result.stderr)

    def test_context_controls_stay_locked_during_single_or_batch_work(self):
        busy = self.block(WEB, "function busy(", "async function run(")
        output = self.js("""const state={busy:false,batchBusy:false},elements={},$=id=>elements[id]||(elements[id]={});
const document={querySelectorAll:()=>[]},modelReady=()=>true,icon=()=>'',updatePackageControls=()=>{};
""" + busy + """
const states=[];busy(true);states.push([elements.reviewContractType.disabled,elements.reviewPosition.disabled]);busy(false);states.push([elements.reviewContractType.disabled,elements.reviewPosition.disabled]);state.batchBusy=true;busy(false);states.push([elements.reviewContractType.disabled,elements.reviewPosition.disabled]);process.stdout.write(JSON.stringify(states));""")
        self.assertEqual(json.loads(output), [[True, True], [False, False], [True, True]])

    def archived_inputs(self, report, contract_type="auto", position="neutral"):
        loader = self.block(WEB, "function showReport(", "$('historyWrap').addEventListener(")
        options = self.block(WEB, "function reviewContextOptions(", "function hasSupervisor(")
        source = "const report=" + json.dumps(report, ensure_ascii=False) + ",initial=" + json.dumps({"contract_type": contract_type, "review_position": position}, ensure_ascii=False) + ";\n" + r"""
const elements={engine:{value:'multi'},useLlm:{checked:false},reviewContractType:{value:initial.contract_type,options:['auto','通用','技术开发','政府采购','房屋租赁'].map(value=>({value}))},reviewPosition:{value:initial.review_position},sourceName:{value:''},text:{value:'',dispatchEvent(){}},file:{value:''},filetag:{classList:{add(){}}}},$=id=>elements[id];
const state={busy:false,batchId:'existing-batch',batchInfo:{options:{contract_type:'政府采购',review_position:'party_a'}}},history={replaceState(){}},inputMode=()=>{},showView=()=>{},renderReport=()=>{},toast=()=>{};
const reportBefore=JSON.stringify(report),batchBefore=JSON.stringify({id:state.batchId,info:state.batchInfo});
""" + options + loader + "\nshowReport(report);process.stdout.write(JSON.stringify({options:runOptions(),sourceName:elements.sourceName.value,text:elements.text.value,reportUnchanged:reportBefore===JSON.stringify(report),batchUnchanged:batchBefore===JSON.stringify({id:state.batchId,info:state.batchInfo})}));"
        result = json.loads(self.js(source))
        self.assertTrue(result["reportUnchanged"])
        self.assertTrue(result["batchUnchanged"])
        return result

    def test_archived_report_restores_explicit_type_and_position_for_review(self):
        report = self.fixture("needs_manual")
        report.update(source_name="归档开发合同", document={"full_text": "归档合同正文"})
        report["review_framework"]["context"].update(contract_type_source="user", contract_type="技术开发", review_position="party_b")
        result = self.archived_inputs(report)
        self.assertEqual(result["options"]["contract_type"], "技术开发")
        self.assertEqual(result["options"]["review_position"], "party_b")
        self.assertEqual(result["sourceName"], "归档开发合同")
        self.assertEqual(result["text"], "归档合同正文")

    def test_archived_auto_type_restores_auto_instead_of_detected_effective_type(self):
        report = self.fixture("needs_manual")
        report["review_framework"]["context"].update(contract_type_source="auto", contract_type="技术开发", review_position="party_b")
        result = self.archived_inputs(report, "政府采购", "party_a")
        self.assertEqual(result["options"]["contract_type"], "auto")
        self.assertEqual(result["options"]["review_position"], "party_b")

    def test_archived_stats_context_is_restored_without_inventing_old_context(self):
        for container in ({"review_context": {"contract_type_source": "user", "contract_type": "房屋租赁", "review_position": "party_a"}},
                          {"review_framework": {"context": {"contract_type_source": "user", "contract_type": "房屋租赁", "review_position": "party_a"}}}):
            result = self.archived_inputs({"risks": [], "stats": container})
            self.assertEqual(result["options"]["contract_type"], "房屋租赁")
            self.assertEqual(result["options"]["review_position"], "party_a")
        for legacy in ({"risks": [], "contract_type": "技术开发"}, {"risks": [], "review_framework": {"context": {"contract_type_source": "unknown", "contract_type": "unsupported", "review_position": "invalid"}}}):
            result = self.archived_inputs(legacy)
            self.assertEqual(result["options"]["contract_type"], "auto")
            self.assertEqual(result["options"]["review_position"], "neutral")

    def test_library_real_node_renderer_preserves_manual_state_and_inert_text(self):
        factory = self.block(LIBRARY, "  const list =", "  const replace =")
        structure = self.block(LIBRARY, "  function table(", "  function inputField(")
        renderer = self.block(LIBRARY, "  function renderLibraryReviewFramework(", "  async function openReport(")
        dom = r"""
class SafeNode {
 constructor(tag,text=''){this.tag=tag;this.text=text;this.children=[];this.attrs={};}
 set textContent(value){this.children=[new SafeNode('#text',String(value))];}
 setAttribute(key,value){this.attrs[key]=String(value);}
 addEventListener(){}
 append(child){this.children.push(child);}
 get outerHTML(){if(this.tag==='#text')return esc(this.text);return '<'+this.tag+Object.entries(this.attrs).map(([key,value])=>' '+key+'="'+esc(value)+'"').join('')+'>'+this.children.map(child=>child.outerHTML).join('')+'</'+this.tag+'>';}
}
const Node=SafeNode,document={createElement:tag=>new SafeNode(tag),createTextNode:text=>new SafeNode('#text',text)};
"""
        helpers = """const paragraph=(text,className='help')=>node('p',{className,text}),heading=text=>node('h3',{text}),actions=(...children)=>node('div',{},children),button=(text,fn)=>node('button',{text,onClick:fn}),empty=text=>paragraph(text);"""
        report = self.fixture("no_issue_detected", manual_required=True, impact_basis="explicit_check",
                              scope="provided_contract_text_payment_sentences_only", actual_issue='<img src=x onerror="bad()">', pending_reasons=["待补授权"])
        report["review_framework"]["review_points"][0].update(domain="commercial", responsible_agent="logic")
        rendered = self.js(ESC + dom + factory + structure + helpers + renderer + "\nprocess.stdout.write(renderLibraryReviewFramework("
                           + json.dumps(report, ensure_ascii=False) + ").outerHTML);")
        for text in ("待人工确认", "待补授权", "付款条件是否清楚", "判断标准", "可能影响", "处理建议", "&lt;img", "商业条款", "逻辑审查", "独立检查记录", "仅限所给合同文本中的付款句"):
            self.assertIn(text, rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("<summary>本范围未检出问题", rendered)


if __name__ == "__main__":
    unittest.main()
