/* Real-browser smoke tests for the local contract-review workspace.
 * Run with NODE_PATH pointing at a Playwright installation, then:
 *   node tools/test_web_workspace_browser.cjs http://127.0.0.1:8619
 * The test server should use an isolated SQLite database.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const base = process.argv[2] || 'http://127.0.0.1:8619';
const output = path.resolve(process.env.WEB_QA_OUTPUT || 'tools/browser_workspace_qa');
const sample = [
  '技术开发合同',
  '甲方：星海科技有限公司',
  '乙方：云图技术有限公司',
  '第一条 项目内容',
  '乙方为甲方开发合同管理系统。显示测试：<img src=x onerror="window.__workspaceXss=true">',
  '第二条 金额',
  '合同总价为人民币100000元。首付款60000元，尾款60000元。',
  '第三条 期限',
  '合同起始日期为2026年12月1日，终止日期为2026年11月1日。',
  '第四条 解除',
  '甲方有权随时单方解除合同且无需承担任何责任。',
].join('\n');

async function main() {
  await fs.mkdir(output, { recursive: true });
  const executablePath = process.env.CHROME_PATH || (process.platform === 'win32'
    ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : undefined);
  const browser = await chromium.launch({ headless: true, executablePath });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  const steps = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 500) errors.push(`HTTP ${response.status()}: ${response.url()}`);
  });
  async function step(name, run) {
    const started = Date.now();
    await run();
    steps.push({ name, milliseconds: Date.now() - started });
    console.log(`PASS ${name}`);
  }
  async function nav(view) {
    await page.locator(`[data-view="${view}"]`).click();
  }
  async function tab(name) {
    await page.locator(`[data-tab="${name}"]`).click();
  }
  async function download(action) {
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.locator(`[data-action="${action}"]`).click(),
    ]);
    const saved = path.join(output, download.suggestedFilename());
    await download.saveAs(saved);
    return { filename: download.suggestedFilename(), text: await fs.readFile(saved, 'utf8') };
  }
  let report;
  try {
    await step('desktop initial render', async () => {
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.locator('#run').waitFor({ state: 'visible' });
      await page.waitForFunction(() => /已连接|连接失败/.test(document.querySelector('#healthText').textContent));
      assert.match(await page.title(), /智律契灵/);
      await page.screenshot({ path: path.join(output, 'desktop-initial.png'), fullPage: true });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    });
    await step('empty input is rejected locally', async () => {
      let calls = 0;
      const listener = request => { if (request.url().includes('/api/review')) calls++; };
      page.on('request', listener);
      await page.locator('#run').click();
      await page.waitForTimeout(250);
      page.off('request', listener);
      assert.equal(calls, 0, 'empty review must not reach the backend');
    });
    await step('unsupported upload is rejected locally', async () => {
      await page.locator('#file').setInputFiles({ name: 'unsupported.exe', mimeType: 'application/octet-stream', buffer: Buffer.from('bad') });
      await page.waitForTimeout(100);
      assert.match(await page.locator('body').innerText(), /不支持|文件格式|格式.*支持|请选择 DOCX/);
    });
    await step('example loads a real contract', async () => {
      await page.locator('#demo').click();
      assert.ok((await page.locator('#text').inputValue()).length > 100);
    });
    const fixtureRoot = path.resolve('data/generated_contracts');
    const fixtures = await fs.readdir(fixtureRoot, { recursive: true }).catch(() => []);
    const docxFixture = fixtures.find(name => name.endsWith('.docx'));
    if (docxFixture) await step('DOCX upload is parsed as a real file', async () => {
      await page.locator('[data-input="file"]').click();
      await page.locator('#file').setInputFiles(path.join(fixtureRoot, docxFixture));
      const waiting = page.waitForResponse(response => response.url().endsWith('/api/analyze'));
      await page.locator('#analyze').click();
      const response = await waiting;
      assert.equal(response.status(), 200);
      const parsed = await response.json();
      assert.equal(parsed.document.file_type, 'docx');
      assert.ok(parsed.document.clauses.length > 0);
      assert.ok(parsed.document.full_text.length > 100);
      await page.waitForFunction(() => !document.querySelector('#analyze').disabled);
    });
    await step('enter a contract for source and review checks', async () => {
      await page.locator('[data-input="text"]').click();
      await page.locator('#text').fill(sample);
      await page.locator('#sourceName').fill('browser-workspace-test.txt');
    });
    await step('parse-only shows clauses', async () => {
      const waiting = page.waitForResponse(response => response.url().endsWith('/api/analyze') && response.request().method() === 'POST');
      await page.locator('#analyze').click();
      const response = await waiting;
      assert.equal(response.status(), 200);
      const parsed = await response.json();
      assert.ok(parsed.document.clauses.length >= 4);
      await page.waitForFunction(() => document.querySelector('#run') && !document.querySelector('#run').disabled);
      assert.match(await page.locator('#result').innerText(), /项目内容|第一条|条款/);
    });
    await step('multi-agent review returns source, risks and graph', async () => {
      await page.locator('#engine').selectOption('multi');
      if (await page.locator('#useLlm').isChecked()) await page.locator('#useLlm').uncheck();
      const waiting = page.waitForResponse(response => new URL(response.url()).pathname === '/api/review' && response.request().method() === 'POST');
      await page.locator('#run').click();
      const response = await waiting;
      assert.equal(response.status(), 200);
      report = await response.json();
      assert.ok(report.risks.length > 0);
      assert.ok(report.document.clauses.length >= 4);
      assert.ok(report.elements.parties.length > 0);
      assert.ok(report.graph.nodes.length > 0);
      assert.equal(report.saved, true);
      assert.ok(report.report_id);
      assert.equal(report.source_name, 'browser-workspace-test.txt');
      await page.locator('[data-tab="risks"]').waitFor({ state: 'visible' });
      await page.waitForFunction(() => !document.querySelector('#run').disabled);
      await page.locator('#toast.show').waitFor({ state: 'hidden', timeout: 6500 });
      await page.screenshot({ path: path.join(output, 'desktop-review.png'), fullPage: true });
    });
    await step('source, entities and graph tabs show real output', async () => {
      await tab('source');
      assert.match(await page.locator('#reportContent').innerText(), /100000|100,000/);
      assert.equal(await page.evaluate(() => Boolean(window.__workspaceXss)), false, 'contract text must not execute HTML');
      await tab('elements');
      assert.match(await page.locator('#reportContent').innerText(), /星海科技|云图技术/);
      await tab('graph');
      assert.ok(await page.locator('#reportContent svg').count() > 0, 'graph must be an actual SVG');
      assert.match(await page.locator('#reportContent').innerText(), /抽取|NER|主体|逻辑/);
    });
    await step('risk locates its source clause', async () => {
      await tab('risks');
      const linked = report.risks.find(risk => risk.clause_ids && risk.clause_ids.length);
      assert.ok(linked, 'at least one risk must link to its source clause');
      const expand = page.locator(`[data-expand="${linked.risk_id}"]`);
      await expand.click();
      await page.locator(`[data-risk-id="${linked.risk_id}"] [data-locate]`).click();
      assert.match(await page.locator('#reportContent').innerText(), /条|合同/);
      assert.ok(await page.locator(`[data-clause="${linked.clause_ids[0]}"].highlight`).count(), 'linked source clause must be highlighted');
    });
    await step('human review saves and survives archive reload', async () => {
      await tab('feedback');
      const first = page.locator('[data-feedback-risk]').first();
      const riskId = await first.getAttribute('data-feedback-risk');
      assert.ok(riskId);
      await first.locator('textarea').fill('浏览器端到端复核：确认此项风险。');
      const waiting = page.waitForResponse(response => response.url().endsWith('/feedback') && response.request().method() === 'POST');
      await first.locator('[data-feedback="confirmed"]').click();
      const response = await waiting;
      assert.equal(response.status(), 200);
      await page.reload({ waitUntil: 'networkidle' });
      await nav('records');
      await page.locator('#historySearch').fill('browser-workspace-test');
      await page.locator('#searchHistory').click();
      const open = page.locator(`[data-open-report="${report.report_id}"]`);
      await open.waitFor({ state: 'visible' });
      await open.click();
      await page.locator('[data-tab="feedback"]').waitFor({ state: 'visible' });
      await tab('feedback');
      const restored = page.locator(`[data-feedback-risk="${riskId}"]`);
      assert.equal(await restored.locator('textarea').inputValue(), '浏览器端到端复核：确认此项风险。');
      assert.match(await restored.innerText(), /已确认|确认风险|确认/);
    });
    await step('JSON and Markdown exports contain this report', async () => {
      await tab('risks');
      const json = await download('export-json');
      const data = JSON.parse(json.text);
      assert.equal(data.report_id, report.report_id);
      assert.equal(data.source_name, report.source_name);
      assert.ok(data.document.full_text.includes('100000'));
      assert.ok(Object.values(data.feedback).some(item => item.note.includes('浏览器端到端复核')));
      const markdown = await download('export-md');
      assert.match(markdown.text, /智律契灵|合同审查/);
      assert.ok(markdown.text.includes(report.source_name));
      assert.ok(markdown.text.length > 300);
      assert.ok(markdown.text.includes('浏览器端到端复核'));
    });
    await step('legal search displays corpus evidence', async () => {
      await nav('laws');
      await page.locator('#lawQuery').fill('租赁期限不得超过二十年');
      const waiting = page.waitForResponse(response => new URL(response.url()).pathname === '/api/retrieve');
      await page.locator('#lawSearch').click();
      const response = await waiting;
      assert.equal(response.status(), 200);
      const payload = await response.json();
      assert.ok(payload.hits.length > 0);
      await page.waitForFunction(() => !document.querySelector('#lawSearch').disabled);
      assert.match(await page.locator('#lawResults').innerText(), /民法典|第.*条/);
    });
    await step('batch reviews two real TXT files', async () => {
      await nav('batch');
      await page.locator('#batchFiles').setInputFiles([
        { name: 'batch-a.txt', mimeType: 'text/plain', buffer: Buffer.from(sample) },
        { name: 'batch-b.txt', mimeType: 'text/plain', buffer: Buffer.from(sample.replace('100000', '120000')) },
      ]);
      await page.locator('#runBatch').click();
      await page.waitForFunction(() => /已处理 2 \/ 2/.test(document.querySelector('#batchProgressText').textContent)
        && document.querySelector('#stopBatch').classList.contains('hidden'), undefined, { timeout: 90000 });
      const text = await page.locator('#batchResults').innerText();
      assert.ok(text.includes('batch-a.txt') && text.includes('batch-b.txt'));
      assert.match(text, /完成|分|风险/);
      assert.equal(await page.locator('[data-open-batch]').count(), 2, 'both batch files must have real reports');
    });
    await step('batch CSV export contains both contracts', async () => {
      const [file] = await Promise.all([
        page.waitForEvent('download'),
        page.locator('#exportBatch').click(),
      ]);
      const saved = path.join(output, file.suggestedFilename());
      await file.saveAs(saved);
      const csv = await fs.readFile(saved, 'utf8');
      assert.ok(csv.includes('batch-a.txt') && csv.includes('batch-b.txt'));
      assert.match(csv, /合同名称.*处理状态/);
    });
    await step('durable batch reload restores archived results', async () => {
      const batchId = await page.evaluate(() => localStorage.getItem('zlqlBatch'));
      assert.ok(batchId);
      await page.reload({ waitUntil: 'networkidle' });
      await nav('batch');
      await page.waitForFunction(id => document.querySelector('#batchHistory').value === id
        && document.querySelectorAll('[data-open-batch]').length === 2, batchId);
      assert.match(await page.locator('#batchResults').innerText(), /batch-a\.txt/);
      await page.locator('[data-open-batch]').first().click();
      await page.locator('[data-tab="risks"]').waitFor({ state: 'visible' });
    });
    await step('system exposes evidence import and safe model audit', async () => {
      await nav('system');
      await page.locator('#factUpload').waitFor({ state: 'attached' });
      await page.locator('#modelAuditEvents').waitFor({ state: 'visible' });
      assert.match(await page.locator('#subjectEvidence').innerText(), /人工核验证据|人工提供/);
      assert.match(await page.locator('#modelAuditEvents').innerText(), /不保存合同原文/);
      const receipts = await page.request.get(base + '/api/runtime/events?limit=3');
      assert.equal(receipts.status(), 200);
      assert.equal((await receipts.json()).contains_contract_text, false);
      await page.screenshot({ path: path.join(output, 'desktop-runtime.png'), fullPage: true });
    });
    await step('mobile navigation and report fit the viewport', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await nav('review');
      await tab('risks');
      await page.locator('#toast.show').waitFor({ state: 'hidden' });
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(output, 'mobile-review.png'), fullPage: true });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false, 'mobile must not overflow horizontally');
      await nav('system');
      await page.getByText('申报书指标与验证边界', { exact: true }).waitFor({ state: 'visible' });
      assert.match(await page.locator('body').innerText(), /能力|模型|状态/);
      await page.screenshot({ path: path.join(output, 'mobile-system.png'), fullPage: true });
    });
    assert.deepEqual(errors, [], 'browser must have no uncaught JS errors or HTTP 5xx');
    const result = { passed: true, base, steps, report_id: report.report_id, errors };
    await fs.writeFile(path.join(output, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
    await fs.writeFile(path.join(output, 'results.json'), JSON.stringify({ passed: false, steps, errors, failure: error.stack }, null, 2));
    throw error;
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
