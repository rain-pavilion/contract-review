# 复杂合同候选与运行收据核验审查

审查日期：2026年11月15日。范围是本地代码、离线反例和未应用补丁；本次审查没有调用 LegalOne、启动第二个模型、修改产品源码或改动正在进行的完整运行。离线模拟响应只证明工程护栏，不是法律质量或真实模型能力证据。

## 可以使用的核验产物

- `tools/verify_complex_postrun_v2.py`：独立运行后核验工具。必须提供现有报告、prepared 清单、run 结果及只读 SQLite 收据。
- `tools/complex_postrun_v2_tests_20261115.py`：27项离线核验反例测试，全部通过。
- `tools/receipt_binding_patch_20261115.patch`：尚未应用的4文件提案，显式连接窗口、任务、源文本摘要和真实调用收据，并导出当前反馈请求及轮次绑定。
- `tools/receipt_binding_tests_20261115_draft.py`：11项内存提案测试通过，其中嵌套执行18项既有长文本和反馈来源回归。
- `tools/implicit_plan_patch_20261115.patch`、`tools/implicit_plan_tests_20261115_draft.py`：原始候选规划提案保持不变，20项草案测试通过，其中含13项既有显式引用回归。

这些文件不会自动应用提案。v2 核验缺少显式链接时返回 `unproven`，不会按调用先后、数量或耗时重建历史链接。

## 严重核验问题与处理

1. 旧运行后工具会检查初始窗口清单及 `ok` 状态，因此正文调用遗漏、跨窗口 `error/partial` 确实被拒绝。但它只在全局配对模型 start/terminal 收据，一对成功收据也可能支撑多个自报成功窗口。v2 要求每个初始 window/job 有不同的真实 call_id，且 start 和 terminal 都包含完全一致的 window_id、job_id、source_sha256。
2. 旧工具的 effective 完成标志不结合 cross_validation 的待核验账本。v2 将 pending、当前请求清单、当前 attempt、每批任务和逐项回执一起核验；有 pending 时不报告有效审查完成。
3. 只检查 `stable` 会漏掉 `converged` 等替代字符串。v2 只接受生产者的规范状态 `stable` 与 `round_limit`，并核对 converged 布尔值和待核验账本；其余字符串全部拒绝。
4. 反馈摘要可能声称 planned/completed 相等，但实际 calls 为空；或反馈项被排到两个窗口，只在一个窗口回应；或引用的原文不属于该反馈项的条款。v2 按 feedback_windows.batches 重建 job 集合，逐批核对 exact IDs、调用状态、ack、pending、错误、源条款交集和引文。
5. 历史任意完整反馈 run 不能证明当前状态已完成。提案显式导出 attempt_id/index、fingerprint、run_sha256 与当前请求清单；v2 检查当前指纹的最后一次明确 attempt 及其快照，不用列表成员关系替代当前轮次证据。
6. 非空反馈 items 不能通过清空 focus/required 降为无需处理。v2 将任何非空 items 视为有反馈要求，并核对 focus 的来源对应关系。
7. 已截断的法条、主体事实或来源文本，空白 reason、未知额外反馈行，以及主体回执引用了其他 clause_id 的文字，都不能声称协议完成。
8. 上游深度法条关系或当前反馈中出现 `input_truncated`、`coverage_complete=false`、`manual_required`、`disagreement` 时，只阻止 `effective_model_review_complete`。模型已确认的逐项反馈仍可 `protocol_complete=true`，真实执行收据和规范协议也仍可通过 strict；语义限制单独列入 `upstream_semantic_review`，不能被“已消费”状态抹去。

训练子代理最终独立复跑27项核验测试与11项绑定提案测试，全部通过。其4个独立反例、empty-focus 反例和 truncated-source 反例均被 v2 拒绝；完整正例仍通过。最后增加的上游语义 guard 中12组位置/限制子测试确认 strict 与协议完成不受误伤，明确无语义限制的正例仍可完整完成。独立反例文件是 `tools/complex_postrun_v2_independent_audit_20261115.py`，由审查者保留，本次未覆盖。

## 数值候选提案仍有的边界

原始候选补丁未在这轮审查中修改。以下结果通过内存候选模块实际复现：

| 例子 | 结果 | 解释 |
|---|---|---|
| 本合同税前合同总价10000元 / 本合同税后合同总价11000元 | 1个差异候选，tax=unspecified | P2优先级误配。原提案只区分含税、不含税、未税，未覆盖税前、税后；不同口径不应提升同槽优先级。 |
| 本项目尾款70000元仅作示例 / 本项目尾款80000元仅作示例 | 1个差异候选 | P2精度缺口。示例性表述不是确定的付款义务。 |
| 本项目最终交付为验收后30日 / 45日 | 0个数值候选 | 相对期限不在固定日期窄语法内；一般主题窗口仍可能处理。 |
| 本项目尾款人民币柒万元 / 捌万元 | 0个数值候选 | 中文大写金额尚未规范化。 |
| 项目海河尾款70000元 / 80000元 | 0个数值候选 | 未加引号中文标的名保守放弃。 |
| 项目“海河”尾款70000元 / 80000元 | 1个差异候选 | 在明确命名范围内的支持正例。 |
| 旧合成诊断 prepared | 0个隐式数值候选 | 旧 fixture 缺局部明确 scope，部分日期还属于不同 stage；需要新的诊断 fixture。 |

上述两个误配只会提升模型审查窗口，不会直接输出法律矛盾；候选仍标 `legal_conflict_proven=False`。不过它们会占用有限 cross 预算，因此在应用提案前值得补上保守过滤。其余漏例不能用来宣称系统已穷尽复杂合同的金额与期限推理。

## 证明边界

- `strict_engineering_execution_verified` 证明所提供初始窗口和对应真实收据能够连接，且反馈状态没有自相矛盾；它不等于法律结论正确。
- `effective_model_review_complete` 还要求当前反馈协议完成、无待核验账本、显式引用范围完成、没有 LLM 降级且无上游已报告的截断、覆盖不足、人工核验或争议语义限制。它仍不证明穷尽全部合同条款关系。
- 深度判别成功调用的存在与降级数量分别报告。没有独立的应判别 pair 清单时，`complete_expected_pair_coverage_proven` 始终为 false。
- SQLite 交叉核验使用 `mode=ro`，按准确 trace_id 读取，将 event_id、时间、模型、状态、reason、details 等字段逐项与运行 stats 比较。SQLite 缺失、多出或改变事件都会失败。
- 模型审计保留技术 ID 和摘要，不新增合同原文、提示词或模型输出。当前反馈 full items 只进入原本已包含合同/证据的运行报告。
- 旧历史报告缺少 window/job-call_id 关联时，不能用本次工具替其追认真实逐窗口模型执行。

## 冻结指纹

| 文件 | SHA256 |
|---|---|
| verify_complex_postrun_v2.py | 4814D84FAF926ED762438CC697F98412AB46E1C77C1BC29B844C7008A5CEED24 |
| complex_postrun_v2_tests_20261115.py | 2F694C7B6292944A54C122CAAF4F734824FF902A9BE71F9E3DCF1950AF15A041 |
| receipt_binding_patch_20261115.patch | 1F98CF88293DE432741D0B6EE7FB641399C9BCE1027D442B0D56036C7BDA248B |
| receipt_binding_tests_20261115_draft.py | 1FD1B380898FFE61AE461673C0C7F61AB176A19868B47CB49006EAC444D86632 |
| implicit_plan_patch_20261115.patch | F7D40CD74FE6886B7ED3055008DBD2974E9ABB251EDE0CB64B7C61890398C656 |

两份提案同时 `git apply --check` 通过；`git diff --check` 返回0。冻结时产品 long_context.py 仍为 `16FEBF05914259F81408790EAC57A7AE57EE51FDC2E9F373C0C6423915CC2E71`，旧 full-run 工具仍为 `AA36AF83F6E34CEAC504FA6DF69AA0A9FEC2BBAEBAD5FD80CBD9AB885B3AD265`。后续应用和真实模型执行由父任务统一进行。
