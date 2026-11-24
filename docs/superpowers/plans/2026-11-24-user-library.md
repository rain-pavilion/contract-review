# User Library and Registration Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Existing user instruction authorizes this feature; proceed with bounded implementation and verification.

**Goal:** Separate account storage into users.sqlite and deliver public ordinary-user registration.
**Architecture:** UserLibrary owns identity/session/account audit; ContractLibrary keeps stable principal references and reads attached account state for authorization. API and UI add anonymous registration before the authenticated boundary.
**Tech Stack:** Python SQLite/HTTP, existing vanilla HTML/JS.
**Spec:** docs/superpowers/specs/2026-11-24-user-library-design.md

## Global Constraints
- Preserve existing edits, IDs, private contract ownership, immutable versions and reports.
- No administrator self-registration, plaintext passwords, reset of existing accounts or re-import of revoked sessions.
- Independent file ownership for parallel work; no production database writes from subagents.
- Reuse current checkout: all previous implementation and existing unrelated work are uncommitted.

## Task 1 — identity repository and contract integration
Files: user_library.py, contract_library.py, test_user_library.py, targeted test_contract_library.py adaptations.
- [x] Write and observe failing separation/registration/migration/revocation/race tests.
- [x] Implement UserLibrary, idempotent migration and delegated ContractLibrary account methods.
- [x] Run repository/migration regressions and review authorization boundaries.

## Task 2 — HTTP, paths and local operations
Files: library_api.py, config.py, library_cli.py, start-libraries-web.cmd, test_library_api.py/test_library_cli.py.
- [x] Write and observe failing real HTTP registration policy/collision/CSRF-origin/rate tests.
- [x] Integrate users path and registration status/endpoint/rate limiting.
- [x] Keep CLI and worker connected to configured user library; expose users_db status.

## Task 3 — registration UI
File: library.html only.
- [x] Add registration form and login switches, field validation, pending/error/success states and password cleanup.
- [x] Exercise form helpers/syntax; run browser signup/login/owner isolation/admin management on temporary data.

## Task 4 — migration and delivery
Files: docs/双库使用说明.md, task ledger.
- [x] Obtain independent security/migration review; fix concrete findings and verify related tests.
- [x] Stop actual service, back up current libraries, run actual account migration and confirm contract/law integrity.
- [x] Restart service, verify old admin and registration UI; run related full regressions and update operating instructions.
