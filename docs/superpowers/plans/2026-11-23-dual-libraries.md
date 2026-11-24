# Dual Libraries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Deliver owner-isolated contract storage and an official-source multidomain law library integrated with the current app.
**Architecture:** Two SQLite libraries, private immutable versions, authenticated HTTP boundary, durable review queue, staged law publication and immutable retrieval snapshots.
**Tech Stack:** Existing Python standard library HTTP/SQLite, existing parsers and review engine, vanilla HTML/CSS/JS.
**Spec:** docs/superpowers/specs/2026-11-23-dual-libraries-design.md

## Global Constraints
- Preserve all pre-existing edits and legacy databases; work on codex/dual-libraries-20261123 branch in current tree because current product source includes untracked work.
- No default credentials, user-supplied ownership, anonymous legacy access, or file paths from ordinary users.
- Tests and smoke servers use temporary isolated storage, no live model startup.
- Source documents and published snapshots are immutable; unknown legal validity stays unknown.

### Task 1: Contract repository, accounts, sessions and immutable versions
Files: create zhilv_qiling_v2/contract_library.py and tests_v2/test_contract_library.py.
Produces: ContractLibrary(path), LibraryError(message,status); bootstrap_admin/create_user/login/authenticate/logout; upload/list/get/download/update/delete contract; save/get/list reports; durable task enqueue/claim/finish; events and idempotent legacy import. Exact interfaces live in .superpowers/sdd/2026-11-23-dual-libraries/contract-interface.md.
- [x] Write behavioral tests for owner isolation, password/session, immutable revision and reports, admin deletion and migration idempotency; run to observe failure.
- [x] Implement SQLite transactions and access checks. Example critical assertion: `with self.assertRaises(LibraryError): store.get_contract(user_b, contract_a['contract_id'])`.
- [x] Run `.venv/Scripts/python.exe -m unittest zhilv_qiling_v2.tests_v2.test_contract_library -v` and review changes.

### Task 2: Law repository, source adapters, crawl and publication
Files: create law_library.py, law_crawler.py, law_sources.json and tests_v2/test_law_library.py/test_law_crawler.py.
Produces: versioned ingest, administrator publication, immutable compatible snapshots, filtered search, official-domain crawler with source fixtures. Interfaces in .superpowers/sdd/2026-11-23-dual-libraries/law-interface.md.
- [x] Write failing tests for duplicate/changed source, draft visibility, source hash checks, known vs unknown validity, temporal/regional filters, SSRF/redirect/oversize protections.
- [x] Implement repository and staged publication, parse full official text into article or item units; preserve locations.
- [x] Run `.venv/Scripts/python.exe -m unittest zhilv_qiling_v2.tests_v2.test_law_library zhilv_qiling_v2.tests_v2.test_law_crawler -v`.

### Task 3: Authenticated HTTP, durable review and migration CLI
Files: create library_api.py/library_cli.py/tests_v2/test_library_api.py; modify api.py/config.py/retrieve.py narrowly.
Consumes Task 1/2 interfaces. Produces /api/auth, /api/library/contracts,/reports,/tasks,/users,/laws endpoints and admin-only legacy access.
- [x] Write real HTTP integration tests: unauthorized401; two users and admin; cross-user404; CSRF403; upload readonly; private download; reports/tasks ownership; old routes403; hidden source paths; admin publication.
- [x] Install authentication before old route dispatch, attach actor, secure headers/cookie and CSRF; require admin for every non-allowlisted legacy endpoint.
- [x] New review task reuses parser and `_review_document(...,save=False)` against bound contract version and published-law snapshot; use isolated subject facts for ordinary users and deny arbitrary engine/path parameters.
- [x] Add CLI init-admin/create-user/migrate/crawl/publish/status and safe resumable imports. `migrate` must be idempotent and leave legacy files unchanged.
- [x] Run integration tests and relevant engine/retrieval regressions; inspect failures before editing.

### Task 4: Contract and law library UI
Files: create library.html (separate from existing webui.html), add UI behavioral smoke test outside repo. Consumes Task 3 API contract in .superpowers/sdd/2026-11-23-dual-libraries/http-interface.md.
- [x] Implement login/logout; user upload/list/detail/download/review/report; administrator revision/delete/users; law browse/search/detail; administrator sources/crawl/drafts/publish/status.
- [x] Escape text and use DOM textContent; no long-term browser credential or private-data caches; reset all state on logout/account switch.
- [x] Browser flow login as two users and administrator, upload, own-only list, immutable revisions, law draft/publication; desktop and mobile screenshot QA.

### Task 5: Production bootstrap, actual migration/crawl and whole change verification
Files: add docs/双库使用说明.md and operational scripts if necessary.
- [x] Bootstrap admin via CLI with generated strong password stored only in a local owner-readable file; never print password in tool logs. No credentials in tracked files.
- [x] Execute resumable import of current workspace/defense/old reports and legal corpus. Run actual bounded official multidomain crawl with success/failure receipts; publish only verified records and keep missing statuses explicit.
- [x] Verify counts and immutable hashes; run all new suites plus selected existing engine/retrieval tests and browser tests.
- [x] Obtain independent security/spec review, fix findings, rerun affected checks; write final progress and user operating instructions.
