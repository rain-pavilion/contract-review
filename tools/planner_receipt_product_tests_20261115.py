"""Run the 20 planner and 11 receipt tests against normal product imports.

No proposal is read or applied. Only the original tests' proposal-loading
scaffolding is omitted; their assertions and nested regression suites are kept.
All audit storage is temporary, and unmocked transport/inference is forbidden.
Run only after the planner and receipt-binding patches have been applied.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
sys.dont_write_bytecode = True

PLANNER_TESTS = ROOT / 'tools/implicit_plan_tests_20261115_draft.py'
BINDING_TESTS = ROOT / 'tools/receipt_binding_tests_20261115_draft.py'
PRODUCT_NAMES = ('long_context', 'llm_client', 'nodes', 'model_audit', 'agents_multi_run')
# Captured before planner application from product SHA256 16febf...2e71.
# Product mode makes CURRENT and CANDIDATE identical in one inherited test, so
# independently retain that test fixture's full primary-window baseline here.
PRIMARY_BASELINE_SHA256 = 'aa0cbb1d020919530efe35f87848ee293361597ab10724edac2ae4db38fc23db'


class OfflineViolation(BaseException):
    """Do not let product fallback handlers hide an unintended real operation."""


def forbidden(*_args, **_kwargs):
    raise OfflineViolation('real network or model inference is forbidden in product regression tests')


def _load_test_definitions(path: Path, name: str, injected: dict,
                           omitted_functions: set[str], omitted_assignments: set[str]):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in omitted_functions:
            continue
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            attributes = {ast.unparse(target) for target in node.targets if isinstance(target, ast.Attribute)}
            if (names | attributes) & omitted_assignments:
                continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            continue
        kept.append(node)
    tree.body = kept
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(injected)
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    for forbidden_name in omitted_functions - injected.keys():
        if forbidden_name in module.__dict__:
            raise AssertionError('proposal helper must not exist in product test mode')
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true', help='load product tests and report counts without running them')
    args = parser.parse_args()
    watched = [ROOT / 'zhilv_qiling_v2' / (name + '.py') for name in PRODUCT_NAMES]
    watched += [PLANNER_TESTS, BINDING_TESTS]
    before = {path: _sha(path) for path in watched}
    result = None
    with tempfile.TemporaryDirectory(prefix='zlql_product_regression_') as temporary, ExitStack() as guards:
        audit_root = Path(temporary).resolve()
        guards.enter_context(patch.dict(os.environ, {
            'ZLQL_MODEL_AUDIT_PATH': str(audit_root / 'outer_audit.sqlite'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        }))
        guards.enter_context(patch('socket.create_connection', side_effect=forbidden))
        guards.enter_context(patch.object(socket.socket, 'connect', forbidden))
        guards.enter_context(patch.object(socket.socket, 'connect_ex', forbidden))
        products = {name: importlib.import_module('zhilv_qiling_v2.' + name) for name in PRODUCT_NAMES}
        audit = products['model_audit']
        original_connect = audit._connect

        def isolated_audit_connect():
            path = Path(os.environ['ZLQL_MODEL_AUDIT_PATH']).resolve()
            if not path.is_relative_to(audit_root):
                raise OfflineViolation('audit path escaped the temporary regression directory')
            return original_connect()

        guards.enter_context(patch.object(audit, '_connect', side_effect=isolated_audit_connect))
        guards.enter_context(patch.object(products['llm_client'], '_complete', side_effect=forbidden))
        for module_name, attribute in (
            ('bert_entail', 'load_predictor'), ('bert_entail', 'load_model'),
            ('bert_ner', 'load_model'), ('bert_embed', 'shared_embedder'),
        ):
            module = importlib.import_module('zhilv_qiling_v2.' + module_name)
            guards.enter_context(patch.object(module, attribute, side_effect=forbidden))

        @contextmanager
        def product_runtime():
            with tempfile.TemporaryDirectory(prefix='case_', dir=audit_root) as case, \
                 patch.dict(os.environ, {'ZLQL_MODEL_AUDIT_PATH': str(Path(case) / 'audit.sqlite')}):
                yield Path(case) / 'audit.sqlite'

        planner = _load_test_definitions(
            PLANNER_TESTS, '_planner_product_regression_20261115',
            {'CANDIDATE': products['long_context']}, {'candidate_module'}, {'CANDIDATE'})
        binding = _load_test_definitions(
            BINDING_TESTS, '_receipt_product_regression_20261115', {
                'AUDIT': audit, 'CLIENT': products['llm_client'], 'NODES': products['nodes'],
                'GRAPH': products['agents_multi_run'], 'runtime': product_runtime,
            }, {'proposed_sources', 'load_candidate', 'runtime'},
            {'SOURCES', 'AUDIT', 'CLIENT', 'NODES', 'NODES.L', 'GRAPH', 'GRAPH.N'})
        # runtime is deliberately injected rather than compiled from the draft.
        binding.runtime = product_runtime
        if planner.CANDIDATE is not products['long_context'] or any(
                getattr(binding, alias) is not products[name] for alias, name in (
                    ('AUDIT', 'model_audit'), ('CLIENT', 'llm_client'),
                    ('NODES', 'nodes'), ('GRAPH', 'agents_multi_run'))):
            raise AssertionError('test targets must be the normally imported product modules')
        planner_suite = unittest.defaultTestLoader.loadTestsFromTestCase(planner.UnappliedPlanDraftTests)
        binding_suite = unittest.defaultTestLoader.loadTestsFromTestCase(binding.ReceiptBindingDraftTests)
        counts = {'planner_outer': planner_suite.countTestCases(), 'binding_outer': binding_suite.countTestCases()}
        if counts != {'planner_outer': 20, 'binding_outer': 11}:
            raise AssertionError('original test count changed; review wrapper coverage explicitly')
        baseline_rows = [planner.clause('a', '付款材料。' * 100 + '本项目尾款为70000元。'),
                         planner.clause('b', '交付记录。' * 100 + '本项目尾款为80000元。')]
        primary, _ = products['long_context'].build_review_plan(baseline_rows, chunk_chars=300)
        primary_json = json.dumps([window.as_dict() for window in primary if window.kind == 'text'],
            ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if hashlib.sha256(primary_json.encode()).hexdigest() != PRIMARY_BASELINE_SHA256:
            raise AssertionError('primary text windows changed from the independently frozen original fixture')
        for module_name, class_name, minimum in (
            ('test_explicit_reference_context', 'ExplicitReferenceContextTest', 13),
            ('test_core_completion', 'LongContractTest', 9),
            ('test_feedback_grounding', 'FeedbackGroundingTests', 9),
        ):
            module = importlib.import_module('zhilv_qiling_v2.tests_v2.' + module_name)
            count = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, class_name)).countTestCases()
            if count < minimum:
                raise AssertionError('nested regression coverage unexpectedly decreased')
            counts[class_name] = count
        print(json.dumps({'mode': 'normal_product_imports_no_proposal_loading', 'counts': counts,
            'audit_storage': 'temporary_only', 'unmocked_transport_and_model_loading': 'blocked',
            'source_sha256': {path.name: digest for path, digest in before.items()}}, sort_keys=True))
        if not args.collect_only:
            result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([planner_suite, binding_suite]))
        for path, digest in before.items():
            if _sha(path) != digest:
                raise AssertionError('protected source changed during product tests: ' + str(path))
    return 0 if args.collect_only or result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
