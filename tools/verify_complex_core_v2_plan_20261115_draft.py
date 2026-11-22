"""Offline review of the v2 fixture against the UNAPPLIED planner proposal.

Uses the existing draft's in-memory candidate only. It never calls a model,
starts a server, accesses the network, applies a patch, or rewrites product or
frozen integration-tool source. Outputs use a new isolated diagnostic directory.
The fixture and all assertions are engineering checks, not legal-quality gold.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import implicit_plan_tests_20261115_draft as draft
import verify_complex_core_v2 as wrapper
from zhilv_qiling_v2 import long_context as current
from zhilv_qiling_v2.pipeline import prepare

CANDIDATE = draft.CANDIDATE
BASE = wrapper.base
FROZEN = [ROOT / name for name in (
    'zhilv_qiling_v2/long_context.py',
    'tools/verify_complex_core_workflow.py',
    'tools/verify_complex_core_v2.py',
    'tools/implicit_plan_patch_20261115.patch',
)]
BEFORE = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in FROZEN}
OUTPUT = Path(tempfile.mkdtemp(prefix='complex_v2_plan_draft_',
                               dir=ROOT / 'tools/acceptance_results'))
CHECKPOINT = ROOT / 'zhilv_qiling_v2/models/bert_mamba_entail'
OBSERVED = {}


def planned(chunk_chars=1400, cross_windows=4):
    cfg = BASE.config_for(OUTPUT, checkpoint=CHECKPOINT, chunk_chars=chunk_chars,
                          cross_windows=cross_windows)
    raw = wrapper.scoped_contract()
    document = prepare('SYNTHETIC-complex-core-v2.txt', raw.encode(), config=cfg)
    windows, plan = CANDIDATE.build_review_plan(document.clauses,
        chunk_chars=chunk_chars, overlap=cfg.llm_chunk_overlap,
        cross_pairs=cross_windows)
    return document, windows, plan


def compact_plan(windows, plan):
    keys = ('total_clauses', 'total_chars', 'text_windows', 'cross_windows',
        'cross_window_budget', 'explicit_reference_required_checks',
        'explicit_reference_pending_checks', 'explicit_reference_plan_complete',
        'obligation_value_windows_candidates', 'obligation_value_windows_scheduled',
        'obligation_value_windows_pending', 'cross_plan_complete',
        'cross_max_window_chars')
    return {**{key: plan[key] for key in keys},
        'value_candidates': plan['obligation_value_candidates'],
        'scheduled_cross_windows': [{'window_id': item.window_id,
            'reason': item.reason, 'dependency_ids': list(item.dependency_ids),
            'segments': [{'clause_id': segment.clause_id, 'start': segment.start,
                          'end': segment.end} for segment in item.segments]}
            for item in windows if item.kind == 'cross_clause']}


def run_wrapper_offline(directory, *, chunk_chars=1400):
    originals = {name: getattr(BASE, name)
                 for name in ('synthetic_contract', 'source_provenance', 'prepare_case')}
    try:
        with mock.patch.object(current, 'build_review_plan', CANDIDATE.build_review_plan), \
             mock.patch.object(current, 'coverage', CANDIDATE.coverage), \
             mock.patch.object(BASE, 'request', side_effect=AssertionError('network/model request forbidden')), \
             mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('socket connection forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            return wrapper.main(['--run-dir', str(directory), '--chunk-chars', str(chunk_chars),
                                 '--cross-windows', '4'])
    finally:
        for name, value in originals.items():
            setattr(BASE, name, value)


class V2FixturePlanDraftTests(unittest.TestCase):
    def test_default_budget_schedules_exact_two_scoped_value_pairs(self):
        document, windows, plan = planned()
        candidates = plan['obligation_value_candidates']
        self.assertEqual(len(candidates), 2)
        by_slot = {item['slot']: item for item in candidates}
        self.assertEqual(set(by_slot), {'payment:tail:tax=unspecified', 'date:delivery:final'})
        self.assertEqual(by_slot['payment:tail:tax=unspecified']['scope'], 'contract')
        self.assertEqual({row['source_value'] for row in by_slot['payment:tail:tax=unspecified']['values']},
                         {'70000元', '80000元'})
        self.assertEqual(by_slot['date:delivery:final']['scope'], 'project:this')
        self.assertEqual({row['source_value'] for row in by_slot['date:delivery:final']['values']},
                         {'2026年12月1日', '2027年1月15日'})
        scheduled = {window.window_id for window in windows if window.kind == 'cross_clause'}
        self.assertEqual(plan['cross_windows'], 4)
        self.assertTrue(all(item['window_id'] in scheduled for item in candidates))
        self.assertEqual(plan['obligation_value_windows_scheduled'], 2)
        self.assertEqual(plan['obligation_value_windows_pending'], 0)
        self.assertTrue(CANDIDATE.coverage(document.clauses, windows,
                         {item.window_id for item in windows})['text_complete'])
        OBSERVED['default_1400_4'] = compact_plan(windows, plan)

    def test_fixture_negative_stage_event_tax_and_object_do_not_enter_value_pairs(self):
        document, _, plan = planned()
        labels = {clause.clause_id: clause.label for clause in document.clauses}
        candidate_ids = {row['clause_id'] for item in plan['obligation_value_candidates']
                         for row in item['values']}
        candidate_labels = {labels[cid] for cid in candidate_ids}
        self.assertEqual(candidate_labels, {'第二条', '第三条', '第七条', '第八条'})
        # Fourth article: advance and a different tax/stage; sixth: preliminary
        # acceptance and equipment B. Neither is a same-obligation difference.
        self.assertFalse(candidate_labels & {'第四条', '第六条'})
        self.assertTrue(all(item['candidate_only'] and item['legal_conflict_proven'] is False
                            for item in plan['obligation_value_candidates']))
        OBSERVED['negative_control_candidate_labels'] = sorted(candidate_labels)

    def test_actual_literal_offsets_and_tail_are_preserved(self):
        document, windows, plan = planned()
        originals = {clause.clause_id: clause.text for clause in document.clauses}
        labels = {clause.clause_id: clause.label for clause in document.clauses}
        self.assertIn(BASE.TAIL, document.full_text)
        self.assertTrue(any(BASE.TAIL in segment.text for item in windows if item.kind == 'text'
                            for segment in item.segments))
        for item in plan['obligation_value_candidates']:
            for row in item['values']:
                self.assertEqual(originals[row['clause_id']][row['start']:row['end']], row['source_value'])
                self.assertGreater(row['start'], 300)
                if labels[row['clause_id']] in {'第三条', '第七条', '第八条'}:
                    self.assertGreater(row['start'], 1400)
                    self.assertGreater(row['anchor']['start'], 0)
                    self.assertEqual(row['anchor']['end'], len(originals[row['clause_id']]))
                scheduled = next(window for window in windows if window.window_id == item['window_id'])
                self.assertTrue(any(segment.clause_id == row['clause_id'] and
                    segment.start <= row['start'] and segment.end >= row['end'] and
                    row['source_value'] in segment.text for segment in scheduled.segments))
        OBSERVED['literal_offsets_beyond_300_chars'] = True
        OBSERVED['three_long_clause_values_beyond_1400_with_tail_anchors'] = True

    def test_cross_char_budget_is_bounded_without_claiming_token_measurement(self):
        _, windows, plan = planned()
        cross = [window for window in windows if window.kind == 'cross_clause']
        self.assertEqual(plan['cross_max_window_chars'], 4200)
        totals = [sum(len(segment.text) for segment in window.segments) for window in cross]
        self.assertTrue(all(total <= plan['cross_max_window_chars'] for total in totals))
        self.assertTrue(all(len(segment.text) <= 1400 for window in cross for segment in window.segments))
        OBSERVED['cross_window_text_chars'] = totals
        OBSERVED['cross_window_char_cap'] = plan['cross_max_window_chars']
        OBSERVED['token_measurement'] = 'not_performed_offline; actual prompt including schema and completion budget must be checked by serving runtime'

    def test_reciprocal_explicit_dependencies_share_one_default_window(self):
        _, windows, plan = planned()
        self.assertEqual(len(plan['explicit_reference_dependencies']), 2)
        self.assertTrue(plan['explicit_reference_plan_complete'])
        explicit = [item for item in windows if item.reason == 'explicit_reference']
        self.assertEqual(len(explicit), 1)
        self.assertEqual(len(explicit[0].dependency_ids), 2)
        OBSERVED['reciprocal_dependency_window_count'] = 1

    def test_zero_budget_keeps_pending_without_losing_primary_text(self):
        document, windows, plan = planned(cross_windows=0)
        self.assertEqual(plan['obligation_value_windows_pending'], 2)
        self.assertGreater(plan['explicit_reference_pending_checks'], 0)
        self.assertFalse(plan['explicit_reference_plan_complete'])
        self.assertFalse(plan['cross_plan_complete'])
        self.assertTrue(CANDIDATE.coverage(document.clauses, windows,
                         {item.window_id for item in windows})['text_complete'])
        OBSERVED['zero_budget'] = compact_plan(windows, plan)

    def test_small_primary_windows_truthfully_exhaust_four_cross_windows(self):
        _, windows, plan = planned(chunk_chars=300)
        self.assertEqual(plan['cross_windows'], 4)
        self.assertEqual(plan['obligation_value_windows_scheduled'], 0)
        self.assertEqual(plan['obligation_value_windows_pending'], 2)
        self.assertEqual(plan['explicit_reference_pending_checks'], 2)
        self.assertFalse(plan['cross_plan_complete'])
        with self.assertRaisesRegex(AssertionError, 'outside explicit run budget'):
            run_wrapper_offline(OUTPUT / 'wrapper_small_budget_rejected', chunk_chars=300)
        OBSERVED['small_300_4_is_not_accepted'] = compact_plan(windows, plan)

    def test_wrapper_default_offline_prepares_new_directory_without_source_changes(self):
        directory = OUTPUT / 'wrapper_default'
        self.assertEqual(run_wrapper_offline(directory), 0)
        prepared = json.loads((directory / 'prepared.json').read_text(encoding='utf-8'))
        self.assertEqual(prepared['status'], 'prepared_offline')
        self.assertTrue(prepared['no_model_calls'])
        self.assertFalse(prepared['legal_quality_evaluated'])
        self.assertEqual(prepared['raw_text'], wrapper.scoped_contract())
        self.assertEqual(prepared['source_manifest']['fixture_version'], 'scoped_value_tail_v2')
        self.assertEqual(prepared['plan']['obligation_value_windows_scheduled'], 2)
        self.assertFalse((directory / 'result.json').exists())
        self.assertFalse((directory / 'workspace.sqlite').exists())
        self.assertFalse((directory / 'model_calls.sqlite').exists())
        after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in FROZEN}
        self.assertEqual(after, BEFORE)
        OBSERVED['frozen_source_sha256_unchanged'] = after
        OBSERVED['wrapper_prepared_json'] = str(directory / 'prepared.json')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(V2FixturePlanDraftTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'uses_unapplied_in_memory_candidate': True, 'model_calls': 0, 'network_calls': 0,
        'legal_quality_evaluated': False, 'observed': OBSERVED,
        'limitation': 'Default chunk 1400 and four cross windows admit both value pairs. At chunk 300, explicit dependencies exhaust four windows and the wrapper rejects the run; this is not exhaustive cross-contract or legal-quality verification.'}
    (OUTPUT / 'draft-plan-review.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'tests_run': result.testsRun,
                      'evidence': str(OUTPUT / 'draft-plan-review.json')}, ensure_ascii=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
