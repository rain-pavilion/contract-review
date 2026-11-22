/* Native Node renderer checks. No browser, server, model, or network access.
 * Defaults to actual product files after applying the UI/API proposal.
 * --html and --api may select isolated candidate files before application.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const options = Object.fromEntries(process.argv.slice(2).reduce((rows, value, index, args) => {
  if (value === '--html' || value === '--api') rows.push([value.slice(2), args[index + 1]]);
  return rows;
}, []));
const pagePath = path.resolve(options.html || path.join(root, 'zhilv_qiling_v2', 'webui.html'));
const apiPath = path.resolve(options.api || path.join(root, 'zhilv_qiling_v2', 'api.py'));
const html = fs.readFileSync(pagePath, 'utf8');
const apiSource = fs.readFileSync(apiPath, 'utf8');
const renderer = html.slice(html.indexOf('function renderRuntimeSummary(r){'), html.indexOf('function renderGraph(){'));
const sourcesRenderer = html.slice(html.indexOf('async function loadFactSources(){'), html.indexOf('async function loadSystem(){'));
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const render = (logic = {}, cross = {}) => vm.runInNewContext(
  renderer + '\nrenderRuntimeSummary(input);',
  {esc, input: {stats: {logic_llm: logic, cross_validation: cross}}}, {timeout: 1000});
const paragraph = (output, label) => {
  const start = output.indexOf('<p class="rationale">' + label + '：');
  assert(start >= 0, 'missing paragraph: ' + label);
  return output.slice(start, output.indexOf('</p>', start));
};
const semantic = logic => paragraph(render(logic), '语义与证据状态');
let count = 0;
async function test(name, fn) {
  await fn(); count += 1; process.stdout.write('PASS ' + name + '\n');
}

(async () => {
  await test('protocol completion stays separate from semantic pending', () => {
    const success = {feedback_items: [{feedback_item_id: 'f1'}], feedback_complete: true,
      complete: true, verified_feedback_item_ids: ['f1']};
    const output = render({feedback_latest_runs: [success], feedback_runs_complete: true,
      feedback_consumption_current: [{protocol_complete: true, pending_manual: true, semantic_status: 'pending_manual'}]});
    assert(output.includes('逐项反馈核验：已完成；已核验 1 项，待核验 0 项。'));
    assert(paragraph(output, '语义与证据状态').includes('待人工核验 1 项'));
  });
  await test('missing or malformed ledger stays unrecorded', () => {
    for (const extra of [{}, {feedback_consumption_current: null}, {feedback_consumption_current: {complete: true}}]) {
      assert(semantic({complete: true, feedback_complete: true, ...extra}).includes('语义与证据状态：未记录。'));
    }
  });
  await test('upstream coverage and disagreement keep confirmed rows manual', () => {
    for (const limitation of [{input_truncated: true}, {coverage_complete: false},
      {disagreement: true}, {upstream_review_reasons: ['neural_rule_disagreement']}]) {
      const text = semantic({feedback_consumption_current: [{protocol_complete: true, pending_manual: false,
        semantic_status: 'provided_relation_confirmed', ...limitation}]});
      assert(text.includes('待人工核验 1 项'));
      assert(text.includes('给定关系模型已复核 0 项'));
      assert(text.includes('输入覆盖受限或上游分歧 1 项'));
    }
  });
  await test('incomplete and unknown semantics are not promoted', () => {
    const text = semantic({feedback_consumption_current: [{protocol_complete: false},
      {protocol_complete: true, verdicts: ['confirmed']}]});
    assert(text.includes('待人工核验 1 项')); assert(text.includes('状态未记录 1 项'));
    assert(text.includes('给定关系模型已复核 0 项'));
  });
  await test('a model-confirmed relation is not independent authentication', () => {
    const text = semantic({feedback_consumption_current: [{protocol_complete: true,
      pending_manual: false, semantic_status: 'provided_relation_confirmed'}]});
    assert(text.includes('给定关系模型已复核 1 项'));
    assert(text.includes('协议完成不代表法律结论或外部事实已独立核实'));
  });
  await test('source text and semantic details are escaped', () => {
    const output = render({feedback_consumption_current: [{pending_manual: true,
      source_quote: '<img src=x onerror=bad>', reason: '<script>bad</script>'}]});
    assert(!output.includes('<script>bad</script>')); assert(!output.includes('<img src=x'));
    assert(output.includes('&lt;script&gt;bad&lt;/script&gt;'));
    assert(output.includes('&lt;img src=x onerror=bad&gt;'));
  });
  await test('fact source descriptions distinguish provenance actors and escape data', async () => {
    const elements = {subjectEvidence: {innerHTML: ''}, factUpload: {addEventListener() {}}};
    const data = {total: 2, scope: '<script>scope</script>', records: [
      {source: {actor_kind: 'adapter', publisher: '<img src=x onerror=bad>'}}, {source: {actor_kind: 'human'}}]};
    const sandbox = {esc, $: id => elements[id], api: async () => data};
    await vm.runInNewContext(sourcesRenderer + '\nloadFactSources();', sandbox, {timeout: 1000});
    const output = elements.subjectEvidence.innerHTML;
    assert(output.includes('2 条带来源的主体证据')); assert(!output.includes('带来源的人工核验证据'));
    for (const text of ['source.actor_kind', 'human', 'adapter', 'legacy_unspecified', '来源标记不自动认证']) assert(output.includes(text));
    assert(!output.includes('<script>scope</script>')); assert(!output.includes('<img src=x'));
    assert(output.includes('&lt;script&gt;scope&lt;/script&gt;'));
    assert(output.includes('&lt;img src=x onerror=bad&gt;'));
  });
  await test('API descriptions remain neutral', () => {
    assert(apiSource.includes('带来源主体证据导入；source.actor_kind区分human、adapter和legacy_unspecified'));
    assert(apiSource.includes('records须为1至1000条带来源、提供者、方法和日期的主体证据'));
    assert(apiSource.includes('handle_subject_facts, "导入带来源主体证据"'));
    assert(!apiSource.includes('"scope": "人工核验证据导入'));
  });
  await test('existing old-report and explicit-reference wording is preserved', () => {
    const output = render({text_coverage: 1, complete: true});
    assert(output.includes('逐项反馈核验：未记录。')); assert(output.includes('显式引用依赖：未记录。'));
    const partial = render({text_coverage: 1, explicit_reference_complete: false,
      explicit_reference_pending_checks: 8, explicit_reference_checks: [{checked_checks: 1, pending_checks: 8}]});
    assert(partial.includes('正文覆盖率 100.0%'));
    assert(partial.includes('显式引用依赖：未完成；实际完成 1 次检查，待完成 8 次'));
  });
  await test('existing latest-attempt protocol counters keep historical failures separate', () => {
    const failed = {feedback_items: [{feedback_item_id: 'f1'}], complete: false,
      feedback_complete: false, uncompleted_feedback_item_ids: ['f1']};
    const success = {feedback_items: [{feedback_item_id: 'f1'}], complete: true,
      feedback_complete: true, verified_feedback_item_ids: ['f1']};
    const output = render({feedback_runs: [failed, success], feedback_latest_runs: [success], feedback_runs_complete: true});
    assert(output.includes('逐项反馈核验：已完成；已核验 1 项，待核验 0 项。'));
    assert(output.includes('历史尝试 2 次'));
  });
  await test('existing incomplete and empty feedback do not appear complete', () => {
    assert(render({feedback_latest_runs: [], feedback_runs_complete: true},
      {feedback_verification_complete: false}).includes('仍有待核验反馈，具体条目未记录'));
    assert(render({feedback_items: [], feedback_complete: true, feedback_latest_runs: [], feedback_runs_complete: true},
      {feedback_verification_complete: true}).includes('逐项反馈核验：本次未触发逐项反馈核验。'));
  });
  process.stdout.write(JSON.stringify({status: 'passed', tests: count, html: pagePath, api: apiPath,
    no_model_calls: true, no_network_calls: true, product_files_modified: false}) + '\n');
})().catch(error => {process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1;});
