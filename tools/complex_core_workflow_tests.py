"""Offline guards for the complex/public contract validation tool; no models."""
from __future__ import annotations
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import verify_complex_core_workflow as V
from zhilv_qiling_v2.long_context import build_review_plan, coverage
from zhilv_qiling_v2.models import Clause
from zhilv_qiling_v2.subject_facts import validate_fact, resolve_subject


class ComplexWorkflowGuards(unittest.TestCase):
    def test_public_source_inventory_and_all_text(self):
        raw, name, manifest = V.public_contract_source(V.ROOT/'data/gov_contracts/3529a49034df')
        files = sorted((V.ROOT/'data/gov_contracts/3529a49034df/text').glob('page_*.txt'))
        self.assertEqual(raw, '\n'.join(path.read_text(encoding='utf-8') for path in files))
        self.assertEqual(manifest['page_text_count'], 18)
        self.assertFalse(manifest['source_refetched_this_run'])
        self.assertTrue(manifest['no_page_text_omitted'])
        self.assertEqual(len(manifest['files']), 37)
        self.assertTrue(name.startswith('PUBLIC-'))

    def test_public_source_outside_registry_refused(self):
        with self.assertRaises(ValueError):
            V.public_contract_source(V.ROOT/'tools')

    def test_fixture_source_and_tail_are_deterministic(self):
        self.assertEqual(V.synthetic_contract(), V.synthetic_contract())
        self.assertGreater(len(V.synthetic_contract()), 3000)
        self.assertIn(V.TAIL, V.synthetic_contract())
        self.assertIn('非人工法律金标', V.synthetic_contract())

    def test_expired_wrong_identity_and_absent_credit_never_promoted(self):
        facts = [validate_fact(row) for row in V.synthetic_facts('ct_test', verified_at='2026-11-14')]
        result = resolve_subject(V.SUBJECT, facts, subject_ids=[V.BUYER_ID],
                                 contract_id='ct_test', today=date(2026,11,15))
        self.assertEqual(result['checks']['license'], 'unknown')
        self.assertEqual(result['checks']['credit'], 'unknown')
        self.assertFalse(result['independently_verified'])
        self.assertTrue(any('expired' in row['review_reasons'] for row in result['records']))
        self.assertTrue(all(row['effective_status']=='unknown' for row in result['records']
                            if row['subject_id']==V.SELLER_ID))

    def test_zero_budget_reports_pending_even_when_all_text_reviewed(self):
        clauses=[Clause('a','第一条','',1,'','','按第二条支付。'+'说明'*350),
                 Clause('b','第二条','',1,'','','收到款项后交付。'+'材料'*350)]
        windows, plan=build_review_plan(clauses, chunk_chars=300, cross_pairs=0)
        status=coverage(clauses,windows,{window.window_id for window in windows})
        self.assertTrue(status['text_complete'])
        self.assertFalse(status['complete'])
        self.assertGreater(plan['explicit_reference_pending_checks'],0)
        self.assertTrue(any(segment.end==len(clauses[1].text) for window in windows
                            for segment in window.segments if segment.clause_id=='b'))

    def test_pending_cannot_claim_stability_or_unverified_model_identity(self):
        report={'document':{'full_text':'example-tail'}, 'stats':{
            'logic_llm':{}, 'cross_validation':{'termination':'stable',
                'pending_feedback_item_ids':['pending1']},
            'model_calls':[{'component':'entailment','operation':'predict_pair','status':'succeeded'},
                           {'component':'llm','operation':'request_json','status':'started'}]}}
        result=V.assess_complex(report,{'parsed_text':'example-tail','tail_text':'tail'})
        self.assertFalse(result['engineering_invariants_passed'])
        self.assertFalse(result['engineering_invariants']['pending_does_not_claim_stable'])
        self.assertFalse(result['engineering_invariants']['verified_actual_llm_receipts'])
        self.assertFalse(result['legal_quality_evaluated'])


if __name__=='__main__':
    unittest.main()
