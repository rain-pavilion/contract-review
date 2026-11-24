# Supervisor First Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Make supervisor the first of six core agents, with validated model selection, bounded recovery and observable execution.

**Architecture:** Preserve the existing message bus and mandatory workflow. A focused supervisor module produces validated decisions and resolves registered model configurations; agents_multi_run applies decisions and records execution. UI consumes additive metadata with historical fallbacks.

**Tech Stack:** Python standard library, existing llm_client, unittest, existing HTML/JavaScript.

**Spec:** docs/superpowers/specs/2026-11-24-supervisor-first-design.md

## Global Constraints

- Preserve unrelated workspace edits and historical node IDs.
- No new dependency, external model call, key creation or deployment.
- Never expose model API keys in plans, messages, prompts or reports.
- Preserve deterministic scoring, evidence checks and pending status.
- Six core roles: orchestrator, ner, logic, subject, statute, judge.

## Task 1: Validated supervisor and registered model pool

Files: create zhilv_qiling_v2/supervisor.py; modify config.py; create tests_v2/test_supervisor.py.

Interface: Supervisor(config), .decide(stage, context)->dict, .config_for(agent, model_id)->Config, .snapshot()->dict. Decisions contain stage/source/action/reason/assignments/clause_ids; assignments contain agent/model_id/reason. Supported stages initial/review/feedback. Supported actions continue/review/manual. Registry contains default/strong plus configured named routes. Public snapshots never include secrets.

- [x] Write failing tests for a named model resolved into a distinct endpoint and key, unknown model rejection, stage/action validation, fallback after failed planning and bounded planner calls.
- [x] Confirm missing functionality with failing tests; use `.venv/Scripts/python.exe -m unittest zhilv_qiling_v2.tests_v2.test_supervisor -q` because pytest is not installed.
- [x] Implement schema-validated planning with bounded summaries and explicit rules fallback; test using mocked network boundary only.
- [x] Run the same tests and confirm success.

## Task 2: Real routing and first-node metadata

Files: agents_multi.py, agents_multi_run.py, engine.py, tests_v2/test_supervisor_integration.py.

Consumes Task 1 interfaces. Reports add stats.supervisor and stats.multi_agent.core_nodes; stats.nodes includes additive core_order. Legacy business IDs remain unchanged.

- [x] Write failing integration tests demonstrating route selection affects N.node_logic config and manual requests preserve pending.
- [x] Implement initial/review/feedback decisions, registered reviewer route selection, bounded replanning and execution receipts.
- [x] Preserve rules-only path and existing workflow dependencies.
- [x] Run integration and existing multi-agent/feedback tests.

## Task 3: Six-node UI and deployment documentation

Files: webui.html, docs/主脑节点配置与验证.md, README.md.

Consumes stats.supervisor and additive core metadata. UI shows node 01 supervisor then existing business stages 02-06, real decision source and route reasons; old reports remain readable.

- [x] Update collaboration display and operational text, with bounded actual decision details.
- [x] Document planner endpoint/model/key configuration, model pool fields and offline/cloud validation limits.
- [x] Run authenticated API and renderer checks, then inspect an actual sample review in the browser.

## Task 4: Verification and independent review

- [x] Run focused supervisor, multi-agent, feedback, report and web tests.
- [x] Run broader relevant suite once; investigate actual failures.
- [x] Review changed files for workflow bypass, secret exposure, unbounded retries and compatibility.
- [x] Record actual checks and the absence of live cloud validation.

## Observed validation

Final combined unittest run: 273 tests passed, including 25 supervisor unit tests, 11 supervisor integration regressions, renderer fixtures, business/feedback/evidence tests and 13 authenticated library API tests. Independent read-only review reran all 11 supervisor integration tests and found no remaining significant issue in scope.

Actual browser review in an isolated local test service showed supervisor first, all five business stages, three rules-sourced planning stages, explicit unresolved work and no actual model-call claim. Browser warning/error log was empty. Screenshot: outputs/supervisor-qa-20261124/supervisor-first.jpg.

The older test_web_workspace suite lacks login setup and fails with 401 against the existing authenticated API. Authentication was preserved; the newer authenticated API suite passed. No live cloud inference or performance comparison was performed.
