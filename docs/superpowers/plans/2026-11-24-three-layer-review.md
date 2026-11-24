# Three Layer Review Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox syntax. User explicitly requested implementation in this session.

**Goal:** Connect workflow, independent review criteria and evidence-bound assessments throughout the existing project.

**Architecture:** Extend existing context and reports additively, with pure catalog/projection modules and one bounded payment check. Keep the six core agents and NER/logic peer relationship, all existing risk scores and pending ledgers.

**Tech Stack:** Existing Python standard library and unittest; existing HTML/JavaScript and Node renderer tests.

**Spec:** docs/superpowers/specs/2026-11-24-three-layer-review-design.md

## Global Constraints

- Preserve unrelated dirty workspace changes; no commits, staging or resets.
- No new dependency, cloud key, external model call or relaxed authentication.
- No absent-risk-to-pass inference; no success without evidence and scope.
- Preserve existing risk identities, scoring, pending work and historical archives.

## Task 1: Context and persisted execution options

Files: new review_context.py; config.py, pipeline.py, models.py, api.py, library_api.py, workspace_queue.py; new context tests.

- [x] Write tests for validated type/position, cloned config and queue persistence; observe missing functionality.
- [x] Implement normalize_review_options, apply_review_options, build_review_context with fingerprint; propagate selected options through every task entry.
- [x] Bind context to Contract before engine execution; selected type takes precedence while inferred type remains visible.
- [x] Add context to planner, logic and judge data, without replacing system instructions or evidence gates.

## Task 2: Review points and result projection

Files: new review_points.py, review_assessment.py, tests_v2/test_review_framework.py; engine.py, models.py.

- [x] Write failing tests for missing execution, unknown facts, exact quotation binding and immutable original risks.
- [x] Implement review_point_catalog and build_review_framework with separate workflow stages, criteria and per-point assessments.
- [x] Add bounded deterministic payment checks with exact source quotes and conservative uncertain states.
- [x] Assemble framework in ReviewEngine.run for both engine modes and expose it in API, JSON and Markdown.

## Task 3: Targeted workspace and library UI

Files: webui.html, library.html, tests_v2/test_review_framework_ui.py.

- [x] Add type/position controls and consistent synchronous/batch/library options.
- [x] Add workflow and review-point views; display issues, possible impact, recommendations, evidence and manual gates.
- [x] Preserve existing evidence/manual/graph views and historical reports; never infer pass in the browser.
- [x] Run renderer tests and whole-script syntax checks; inspect desktop and narrow-screen layouts.

## Task 4: Integration and verification

- [x] Run new context, payment, projection, renderer and authenticated API tests.
- [x] Run relevant supervisor, business, feedback, queue, package and report regressions.
- [x] Review field propagation, secret exposure, missing-evidence states and scoring preservation.
- [x] Record observed results and unresolved limitations in user documentation.

## Observed completion evidence — 2026-11-24

- Final combined relevant unittest run: **470 tests passed in 51.635 seconds**. Includes actual authenticated HTTP review/archive/access-isolation tests, existing supervisor/feedback/scoring regressions, bounded payment checks, context binding and both full-page script syntax checks. This is a selected regression suite, not a claim that every old test in the dirty workspace passes.
- Earlier unmodified `test_api_cli.ApiTest` still has eight failures from its obsolete unauthenticated API/root-page expectations; existing authentication remains enabled. The authenticated integration suite tests 400/202/401/404 and saved snapshots without bypassing authentication.
- CUA tested the actual temporary local backend: login → select technology-development/party_b → review → workflow/points/independent payment result → exact source jump → archive reload. Saved perspective is restored for re-review. Desktop 1440×1000 and narrow 390×844 had no page-level horizontal overflow; captured browser warning/error log was empty. Temporary tab was closed, viewport reset and owned server stopped.
- Added late safeguards from review: invoice dates do not clear payment dates; malformed Chinese periods, omitted payment sentences, stale clause text and unverified OCR cannot clear the bounded check. Reusing a parsed document does not contaminate the earlier review context.
- User instructions and coverage limitations are documented in `docs/三层审查结构与使用.md`. No real cloud request, cloud deployment, benchmark accuracy claim, credential creation, staging or commit was performed.
