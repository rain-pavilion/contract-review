"""Apply the frozen planner and narrow precision follow-up in memory only.

This runner reuses the original 20 draft tests and the unmodified five product
regressions. It never changes product source, test inputs or model run evidence.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import re
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / 'zhilv_qiling_v2/long_context.py'
V1_PATCH = ROOT / 'tools/implicit_plan_patch_20261115.patch'
FOLLOWUP_PATCH = ROOT / 'tools/implicit_plan_precision_followup_20261115.patch'


def apply_unified(source: str, proposal: str) -> str:
    original = source.splitlines(keepends=True)
    result, position, active = [], 0, False
    for line in proposal.splitlines(keepends=True):
        if line.startswith(('--- ', '+++ ')):
            continue
        if line.startswith('@@'):
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', line)
            if not match:
                raise AssertionError('malformed patch header')
            start = int(match.group(1)) - 1
            if start < position:
                raise AssertionError('overlapping or reordered patch hunks')
            result.extend(original[position:start])
            position, active = start, True
        elif active and line.startswith((' ', '-')):
            if original[position] != line[1:]:
                raise AssertionError('proposal context/removal drift')
            if line[0] == ' ':
                result.append(line[1:])
            position += 1
        elif active and line.startswith('+'):
            result.append(line[1:])
        elif line.startswith('\\ No newline'):
            continue
        else:
            raise AssertionError('unexpected patch line')
    if not active:
        raise AssertionError('empty proposal')
    return ''.join(result + original[position:])


def run() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'tools'))
    watched = [PRODUCT, V1_PATCH, ROOT / 'tools/implicit_plan_extra_tests_20261115.py']
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
    base = apply_unified(PRODUCT.read_text(encoding='utf-8'), V1_PATCH.read_text(encoding='utf-8'))
    candidate_source = apply_unified(base, FOLLOWUP_PATCH.read_text(encoding='utf-8'))
    name = 'zhilv_qiling_v2._unapplied_implicit_plan_precision_followup'
    candidate = types.ModuleType(name)
    candidate.__package__ = 'zhilv_qiling_v2'
    sys.modules[name] = candidate
    exec(compile(candidate_source, '<unapplied-implicit-plan-precision-followup>', 'exec'), candidate.__dict__)
    draft = importlib.import_module('implicit_plan_tests_20261115_draft')
    extra = importlib.import_module('implicit_plan_extra_tests_20261115')
    original_candidate, original_builder = draft.CANDIDATE, extra.build_review_plan
    try:
        draft.CANDIDATE = candidate
        extra.build_review_plan = candidate.build_review_plan
        suite = unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromTestCase(draft.UnappliedPlanDraftTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(extra.ConservativeValueCandidates),
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        draft.CANDIDATE, extra.build_review_plan = original_candidate, original_builder
        for path, expected in before.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise AssertionError('protected source changed: ' + str(path))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(run())
