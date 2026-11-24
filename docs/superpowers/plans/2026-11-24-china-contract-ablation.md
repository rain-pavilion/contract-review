# 中国法合同消融实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement tasks with independent review. User already authorized execution; continue without another approval round.

**Goal:** Deliver frozen Chinese contract inputs/targets, official-source annotation candidates, honest ablation controls and locally verified dev baselines.

**Architecture:** New isolated experiment package imports production interfaces only where required. Supervised public benchmark, unlabelled official material, CPU baselines and pending production-system experiments remain distinct in manifests and metrics.

**Tech Stack:** Python 3.12, unittest, stdlib, installed numpy/scipy/scikit-learn; existing local production modules.

**Spec:** `docs/superpowers/specs/2026-11-24-china-contract-ablation-design.md`.

## Global Constraints

- No production-file changes, no rollback/commit/cleanup of existing user modifications.
- No credentials, paid API or automatic model startup. All experiment writes stay in the new package directory, except these design/plan records.
- Use released labels only for supervision. Official guidance and unannotated material are not automatic risk gold.
- Test set frozen; first executed comparisons use dev. Test requires an explicit CLI flag.
- Reusing a frozen output with changed bytes/configuration fails closed.

## Task 1: Freeze Chinese supervised benchmark

Files: `experiments/china_contract_ablation_20261124/dataset.py`, `tests/test_dataset.py` under the same experiment directory.

- [ ] First write and observe failing behavioral tests for input/target separation, nested law titles, duplicates, conflicting labels and frozen drift.
- [ ] Implement `prepare_ccl(source_dir, output_dir, seed=20261124)` and `load_jsonl(path)` with exact-priority and article-holdout protocols.
- [ ] Run unittest, prepare actual 9,716 released rows and verify resulting disjoint inputs/targets/corpus hashes.

## Task 2: Prepare official-source and blind-annotation pack

Files: experiment `sources.py`, `tests/test_sources.py`, `sources/` outputs.

- [ ] First test public source types, guidance not becoming gold, nonempty quotes and missing/unadjudicated annotation rejection.
- [ ] Implement `prepare_sources(project_root, output_dir, seed, annotation_limit)` and annotation validation.
- [ ] Generate actual source catalog and 30 blind candidates with empty reviewer files, retain pending-human status.

## Task 3: Prepare real production ablation matrix

Files: experiment `system_ablation.py`, `tests/test_system_ablation.py`, `system_matrix/` outputs.

- [ ] First test config isolation, no-endpoint blocking, profile-neutralized flags and secret-bearing endpoint rejection.
- [ ] Implement `prepare_system_matrix(project_root, output_dir, endpoint='')`, explicit Config and static consumption evidence.
- [ ] Prepare matrix and report blockers; do not manufacture formal results.

## Task 4: Implement executable CPU controls and scoring

Files: experiment `metrics.py`, `baselines.py`, `cli.py`, `__main__.py`, package init, `tests/test_metrics.py`, `tests/test_baselines.py`.

- [ ] Hand-compute metric fixtures; first run tests missing implementations and retain red output.
- [ ] Implement BM25/TF-IDF lexical index, risk metrics and train-only-fit 2×2 role/evidence baseline using real retrieval.
- [ ] Add prepare/run-baselines/verify CLI with output immutability and explicit test-unlock behavior.
- [ ] Run CPU baseline on dev, save predictions, timing, actual features/hashes and paired joint-accuracy intervals; do not call it an LLM formal ablation.

## Task 5: Integrate, independently review and document

- [ ] Run all focused unittests and actual-data preparation/freeze verification.
- [ ] Request an independent agent review of leakage paths, misleading experimental claims and no-effect controls; fix issues and re-run tests.
- [ ] Add experiment README with exact commands, matrix, counts, pending human work and all measured limitations.
- [ ] Record final outputs and blockers in `experiments/china_contract_ablation_20261124/PROGRESS.md`; do not push or publish.

## Execution rulings

- Keep existing dirty `codex/dual-libraries-20261123` checkout: a fresh worktree would omit user-owned untracked models/data and production edits being evaluated. Isolate all new experiment writes; do not alter production or `.gitignore`.
- Use current explicit authorization instead of repeating the skill's generic design approval gate.
- No unavailable live model is substituted with mocks or rule fallbacks in formal results. CPU-only runs are explicitly separate development experiments.
