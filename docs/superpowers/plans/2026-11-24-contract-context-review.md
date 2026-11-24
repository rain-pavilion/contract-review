# Contract Context Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将 source-bound 合同图转成可复核的跨条款/跨附件组合审查结果，并在现有工作台可见。

**Architecture:** 纯标准库组合检查模块复用 fact_slots/proposition_graph；统一引擎保存结果和审查点回执，API及工作台消费同一服务端结构。

**Tech Stack:** Python dataclasses/unittest，现有 HTTP API，原生 JavaScript 工作台。

**Spec:** `docs/superpowers/specs/2026-11-24-contract-context-review-design.md`

## Global Constraints

- 保留现有未提交改动；不读取 `.env`，不调用外部模型，不改变原风险评分口径。
- 原文证据必须核对偏移和哈希；不将相同名称或金额自动视为同一义务。
- 缺失、否定、例外、优先级及 OCR 待核不能改判为通过。
- 信息齐全性接口未接入时明确 `not_evaluated`。

## Task 1: 上下文组合审查模块

**Files:** Create `zhilv_qiling_v2/context_review.py`, `zhilv_qiling_v2/tests_v2/test_context_review.py`.
**Interface:** `evaluate_contract_context(contract, config=None, *, completeness_check=None) -> dict` produces constraints/findings/checks with exact evidence and reasoning steps.

- [x] 写失败测试并运行：三文件验收/默认期限/必要测试组合产生一个有三处原文的发现；删除任一前提不确认同一组合。
- [x] 实现来源绑定、图约束、已有命题图关系适配、保守时序规则和完整性回调接口。
- [x] 测试适用范围不一致、否定、例外、优先约定、OCR、引文篡改、重复ID和回调异常。

## Task 2: 来源内条款引用解析

**Files:** Modify `zhilv_qiling_v2/graph.py`; create `zhilv_qiling_v2/tests_v2/test_graph_source_scope.py`.
**Interface:** Existing `build_contract_graph` signature and dataclasses remain unchanged.

- [x] 写失败测试：主文和附件的“第一条”不会覆盖；无来源限定的歧义引用不任选目标。
- [x] 实现来源限定索引与显式目标解析，保留悬空或歧义信号。
- [x] 运行图及既有规则相关回归。

## Task 3: 引擎、API、Markdown与页面

**Files:** Modify `engine.py`, `api.py`, `models.py`, `webui.html`; create `tests_v2/test_context_review_integration.py`, `tests_v2/test_context_review_ui.py`.
**Interface:** `stats.contract_context_review` mirrored at API top level; `checks` appended to `review_point_checks`; new context view reads either location.

- [x] 写失败集成测试：单/多智能体报告均包含组合发现、原文链和可序列化的独立审查回执。
- [x] 接入引擎、API和报告导出，并在独立页面展示检查状态、证据、推导步骤、假设和人工待核事项。
- [x] 执行真实 JS 渲染验证转义、原文定位、旧报告兼容和 Markdown 证据保留。

## Task 4: 回归与交付记录

- [x] 执行新增测试及相关现有套件；记录基线已通过27项。
- [x] 独立审查改动，修正实际问题并复跑受影响测试。
- [x] 运行离线 HTTP/页面验证，更新功能说明与计划状态，报告工程证据及未验证边界。
