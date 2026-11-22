"""Offline replay evidence guards; no endpoint or pretrained model is called."""
from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import replay_feedback_batch as R
from zhilv_qiling_v2 import nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse


def archived_report(*, count=1, selected=0, quote='合成条文正文', current_ids=False):
    clause = Clause('clause-1', '', '', 1, '', '合成合同条款', '合成合同条款', ['block-1'])
    contract = Contract('synthetic-contract', DocumentParse('fixture.txt', 'txt', clause.text, clauses=[clause]))
    checks = [{'candidate_id':'candidate-' + str(index), 'article_uid':'law-' + str(index),
        'clause_ids':['clause-1'], 'article_quote':quote, 'score':0.5, 'source':'offline-fixture'} for index in range(count)]
    # Produce archived IDs through the product's real preparation path while
    # forcing the transport unavailable. No model function is substituted.
    supplied = checks
    if current_ids:
        from zhilv_qiling_v2.agents_multi_run import _statute_feedback_item
        supplied = [_statute_feedback_item(check) for check in checks]
    with patch.object(N.L, 'available', return_value=False):
        _, meta, _ = N._llm_logic(contract, Config(), [clause], verification_feedback=supplied)
    chosen = meta['feedback_items'][selected]['feedback_item_id']
    meta['calls'] = [{'job_id':'original-failed-job', 'status':'feedback_incomplete',
        'subject_fact_feedback_items':0, 'feedback_item_ids':[chosen], 'clause_ids':['clause-1']}]
    return {'contract_id':contract.contract_id, 'document':{'clauses':[dataclasses.asdict(clause)],
        'full_text':clause.text}, 'stats':{'logic_llm':{'feedback_runs':[meta]}, 'statute_checks':checks}}


def assessment_fixture(kind='subject', *, pending=False, verdict='confirmed'):
    item = {'feedback_item_id':'feedback-1', 'fact_id':'fact-1', 'source_id':'source-1',
        'subject_name':'合成主体', 'fact_type':'license', 'clause_id':'clause-1', 'clause_ids':['clause-1'],
        'candidate_id':'candidate-1', 'article_uid':'law-1', 'source_complete':True,
        'effective_status':'valid', 'upstream_pending':False, 'source_evidence_text':'合成来源'}
    schema = N._feedback_json_schema([item], subject_fact=kind == 'subject')
    digest = R.digest(json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
    common = {'call_id':'actual-call-1', 'native_json_schema_requested':True,
        'native_json_schema_sent':True, 'json_schema_sha256':digest}
    calls = [{'component':'llm', 'status':'started', 'details':json.dumps(common)},
        {'component':'llm', 'status':'succeeded', 'details':json.dumps({**common, 'finish_reason':'stop',
            'model_identity_verified':True, 'actual_model':'legalone-8b'})}]
    key = 'subject_fact_verifications' if kind == 'subject' else 'verifications'
    row = {'feedback_item_id':'feedback-1', 'verdict':verdict, 'reason':'合成协议确认',
        'clause_quote':'合成合同条款', 'fact_quote':'合成事实', 'source_quote':'合成来源', 'article_quote':'合成条文'}
    meta = {'calls':[{key:[row]}], 'uncompleted_feedback_item_ids':['feedback-1'] if pending else [],
        'verified_feedback_item_ids':[] if pending else ['feedback-1'], 'feedback_complete':not pending}
    return meta, calls, [item], kind, schema


def archived_subject_report(*, legacy_ids=True):
    from zhilv_qiling_v2.agents_multi_run import _subject_fact_feedback
    from zhilv_qiling_v2.models import stable_id
    name = '合成主体'
    clause = Clause('clause-1', '', '', 1, '', name + '应维持资质。', name + '应维持资质。', ['block-1'])
    contract = Contract('synthetic-subject', DocumentParse('fixture.txt', 'txt', clause.text, clauses=[clause]))
    source = {'url':'https://source.invalid/fixture', 'publisher':'合成来源', 'reviewer':'fixture',
        'verified_at':'2026-11-14', 'verification_method':'仅用于离线协议测试'}
    record = {'fact_id':'fact-1', 'subject_name':name, 'subject_id':'TEST-ID', 'fact_type':'license',
        'effective_status':'invalid', 'identity_match':'name_and_identifier', 'source':source}
    payload = {'fact_checks':[{'subject_name':name, 'subject_node_id':'S:buyer:fixture', 'subject_role':'buyer',
        'identity_binding':{'status':'bound', 'identifiers':['TEST-ID']}, 'checks':{'license':'invalid'},
        'clause_ids':['clause-1'], 'records':[record]}], 'verification_results':[],
        'candidates':[{'candidate_id':'candidate-old', 'clause_ids':['clause-1'], 'extra':{'subject_name':name}}]}
    bus = SimpleNamespace(inbox=lambda agent, kinds: [])
    items = _subject_fact_feedback(SimpleNamespace(contract=contract, bus=bus), payload)
    if legacy_ids:
        for item in items:
            for field in ('feedback_item_id','candidate_binding_status','candidate_binding_errors'):
                item.pop(field)
            # This is the known archived V2 input, before strict identity
            # association. Hash its complete content independently of replay.
            item['candidate_ids'] = ['candidate-old']
            item['feedback_item_id'] = stable_id('subject_feedback',
                json.dumps(item, sort_keys=True, ensure_ascii=False), length=24)
    with patch.object(N.L, 'available', return_value=False):
        _, meta, _ = N._llm_logic(contract, Config(), [clause], subject_fact_feedback=items)
    meta['calls'] = [{'job_id':'original-subject-failure', 'status':'feedback_incomplete',
        'subject_fact_feedback_items':1, 'feedback_item_ids':[items[0]['feedback_item_id']], 'clause_ids':['clause-1']}]
    return {'contract_id':contract.contract_id, 'document':{'clauses':[dataclasses.asdict(clause)], 'full_text':clause.text},
        'stats':{'logic_llm':{'feedback_runs':[meta]}, 'statute_checks':[]},
        'subject_checks':{'stats':{key:value for key,value in payload.items() if key != 'candidates'},
            'candidates':payload['candidates']}}


class RestoreReplayGuards(unittest.TestCase):
    def test_subject_v2_association_is_recomputed_without_upgrading_identity(self):
        _, _, failed, items = R.restore_batch(archived_subject_report(), 'subject')
        self.assertEqual(items[0]['feedback_item_id'], failed['feedback_item_ids'][0])
        self.assertEqual(items[0]['candidate_ids'], ['candidate-old'])
        self.assertNotIn('candidate_binding_status', items[0])

    def test_current_subject_input_keeps_current_binding_metadata(self):
        _, _, _, items = R.restore_batch(archived_subject_report(legacy_ids=False), 'subject')
        self.assertEqual(items[0]['candidate_binding_status'], 'pending_manual')
        self.assertEqual(items[0]['candidate_ids'], [])

    def test_subject_v2_changed_source_or_candidate_cannot_reuse_old_id(self):
        for changed in ('source','candidate'):
            with self.subTest(changed=changed):
                report = archived_subject_report()
                if changed == 'source':
                    report['subject_checks']['stats']['fact_checks'][0]['records'][0]['source']['publisher'] = '修改后的来源'
                else:
                    report['subject_checks']['candidates'][0]['candidate_id'] = 'different-candidate'
                with self.assertRaisesRegex(ValueError, 'original feedback IDs'):
                    R.restore_batch(report, 'subject')

    def test_legacy_request_recovery_rejects_duplicate_request_ids(self):
        item = {'request_id':'request-1', 'subject_name':'合成主体', 'clause_id':'a', 'candidate_ids':[]}
        requests = [{'request_id':'request-1','candidate_id':'a'}, {'request_id':'request-1','candidate_id':'b'}]
        self.assertIsNone(R._legacy_subject_item(item, [], requests))

    def test_current_full_content_id_restores_without_copying_the_archived_id(self):
        report = archived_report(count=2, selected=1, current_ids=True)
        _, _, failed, items = R.restore_batch(report, 'statute')
        self.assertEqual(items[0]['article_uid'], 'law-1')
        self.assertEqual(items[0]['feedback_item_id'], failed['feedback_item_ids'][0])
        self.assertTrue(items[0]['feedback_item_id'].startswith('statute_feedback_'))

    def test_current_id_binds_the_full_quote_tail_before_transport_truncation(self):
        report = archived_report(quote='测' * 1100, current_ids=True)
        _, _, _, items = R.restore_batch(report, 'statute')
        self.assertEqual(len(items[0]['article_quote']), 1100)
        self.assertTrue(items[0]['article_quote_truncated'])
        report['stats']['statute_checks'][0]['article_quote'] = '测' * 1000 + '改' * 100
        with self.assertRaisesRegex(ValueError, 'content-bound'):
            R.restore_batch(report, 'statute')

    def test_current_content_id_rejects_changed_source_and_an_upstream_stale_id(self):
        report = archived_report(current_ids=True)
        record = report['stats']['statute_checks'][0]
        record['feedback_item_id'] = report['stats']['logic_llm']['feedback_runs'][0]['feedback_items'][0]['feedback_item_id']
        record['source'] = 'changed-source'
        with self.assertRaisesRegex(ValueError, 'content-bound'):
            R.restore_batch(report, 'statute')

    def test_original_content_and_global_index_restore_exactly(self):
        report = archived_report(count=2, selected=1)
        _, clauses, failed, items = R.restore_batch(report, 'statute')
        self.assertEqual(items[0]['article_uid'], 'law-1')
        self.assertEqual(items[0]['feedback_item_id'], failed['feedback_item_ids'][0])
        self.assertEqual([clause.clause_id for clause in clauses], ['clause-1'])

    def test_changed_article_quote_cannot_reuse_an_archived_id(self):
        report = archived_report()
        report['stats']['statute_checks'][0]['article_quote'] = '修改后的其他证据'
        with self.assertRaisesRegex(ValueError, 'content-bound'):
            R.restore_batch(report, 'statute')

    def test_other_changed_record_fields_are_also_content_bound(self):
        for field, value in (('source','changed-source'), ('score',0.9), ('clause_ids',['other-clause'])):
            with self.subTest(field=field):
                report = archived_report(); report['stats']['statute_checks'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'content-bound'):
                    R.restore_batch(report, 'statute')

    def test_duplicate_pair_does_not_overwrite_the_matching_original_record(self):
        report = archived_report()
        other = dict(report['stats']['statute_checks'][0], article_quote='其他版本条文')
        report['stats']['statute_checks'].append(other)
        _, _, _, items = R.restore_batch(report, 'statute')
        self.assertEqual(items[0]['article_quote'], '合成条文正文')

    def test_original_quote_truncation_is_reproduced_before_hashing(self):
        report = archived_report(quote='测'*1100)
        _, _, _, items = R.restore_batch(report, 'statute')
        self.assertEqual(len(items[0]['article_quote']), 1000)
        self.assertTrue(items[0]['article_quote_truncated'])

    def test_duplicate_archived_descriptions_or_batch_ids_are_rejected(self):
        for target in ('feedback_items','feedback_item_ids'):
            with self.subTest(target=target):
                report = archived_report(); run = report['stats']['logic_llm']['feedback_runs'][0]
                if target == 'feedback_items': run[target].append(copy.deepcopy(run[target][0]))
                else: run['calls'][0][target] *= 2
                with self.assertRaisesRegex(ValueError, 'duplicate archived'):
                    R.restore_batch(report, 'statute')


class ReplaySchemaSourceGuards(unittest.TestCase):
    def test_rebuilt_schema_matches_transport_and_excludes_unsent_tail(self):
        text = '甲' * 200 + 'TAIL-MUST-NOT-ENTER-FIRST-WINDOW'
        clause = Clause('clause-1', '', '', 1, '', text, text, ['block-1'])
        contract = Contract('schema-fixture', DocumentParse('fixture.txt', 'txt', text, clauses=[clause]))
        item = {'feedback_item_id':'feedback-1', 'candidate_id':'candidate-1', 'article_uid':'law-1',
            'clause_ids':['clause-1'], 'article_quote':'法条不是合同引文'}
        config = Config(enable_llm=True, llm_native_json_schema=True, llm_chunk_chars=200,
            llm_chunk_overlap=0, llm_cross_clause_pairs=0, llm_max_calls=1)
        expected = R.expected_replay_schema(config, [clause], [item], 'statute')
        with patch.object(N.L, 'available', return_value=True), \
             patch.object(N.L, 'request_json', return_value=SimpleNamespace(content={'verifications':[]}, model='fixture')) as request_json:
            N._llm_logic(contract, config, [clause], verification_feedback=[item])
        self.assertEqual(request_json.call_args.kwargs['json_schema'], expected)
        quote_schema = expected['properties']['verifications']['items']['properties']['clause_quote']
        self.assertTrue(quote_schema['enum'])
        self.assertTrue(all(value in '甲' * 200 for value in quote_schema['enum']))
        self.assertNotIn('TAIL-MUST-NOT-ENTER-FIRST-WINDOW', json.dumps(expected))
        self.assertNotIn('法条不是合同引文', quote_schema['enum'])

    def test_rebuilt_schema_uses_bound_clause_intersection_and_preserves_whitespace(self):
        clauses = [Clause('a', '', '', 1, '', '甲方原文\n 含空白', '甲方原文\n 含空白', ['a']),
            Clause('b', '', '', 1, '', '未绑定的其他合同条款', '未绑定的其他合同条款', ['b'])]
        item = {'feedback_item_id':'feedback-1', 'fact_id':'fact-1', 'source_id':'source-1',
            'subject_name':'合成主体', 'fact_type':'license', 'clause_id':'a', 'clause_ids':['a','forged'],
            'source_evidence_text':'合成来源不是合同原文'}
        schema = R.expected_replay_schema(Config(llm_cross_clause_pairs=0), clauses, [item], 'subject')
        values = schema['properties']['subject_fact_verifications']['items']['properties']['clause_quote']['enum']
        self.assertEqual(values, ['甲方原文\n 含空白'])

    def test_rebuilt_schema_refuses_missing_source_window(self):
        with self.assertRaisesRegex(ValueError, 'source window'):
            R.expected_replay_schema(Config(), [], [], 'statute')


class ReplayAssessmentGuards(unittest.TestCase):
    def test_exact_native_schema_audit_passes_only_protocol_scope(self):
        result = R.evaluate_replay(*assessment_fixture())
        self.assertTrue(result['passed'])
        self.assertTrue(result['native_schema_sha256_verified'])
        for field in ('accuracy_evaluated','full_http_revalidated','legal_truth_independently_evaluated'):
            self.assertIs(result[field], False)

    def test_sent_flag_requires_an_actual_boolean_true(self):
        for value in (False, 'true', 1, None):
            with self.subTest(value=value):
                args = assessment_fixture(); details = json.loads(args[1][1]['details'])
                details['native_json_schema_sent'] = value; args[1][1]['details'] = json.dumps(details)
                result = R.evaluate_replay(*args)
                self.assertFalse(result['native_schema_sent'])
                self.assertFalse(result['passed'])

    def test_missing_wrong_or_started_hash_mismatch_never_passes(self):
        for index, value in ((0,'0'*64), (1,''), (1,'1'*64), (1,None)):
            with self.subTest(index=index,value=value):
                args = assessment_fixture(); details = json.loads(args[1][index]['details'])
                if value is None: details.pop('json_schema_sha256')
                else: details['json_schema_sha256'] = value
                args[1][index]['details'] = json.dumps(details)
                result = R.evaluate_replay(*args)
                self.assertFalse(result['native_schema_sha256_verified'])
                self.assertFalse(result['passed'])

    def test_hash_must_match_the_actual_current_batch_schema(self):
        args = assessment_fixture(); args[4]['title'] = 'different-schema'
        self.assertFalse(R.evaluate_replay(*args)['passed'])

    def test_missing_or_unpaired_started_receipts_do_not_prove_one_call(self):
        for mode in ('missing','wrong-id','extra-start'):
            with self.subTest(mode=mode):
                args = assessment_fixture()
                if mode == 'missing': args[1].pop(0)
                elif mode == 'extra-start': args[1].append(copy.deepcopy(args[1][0]))
                else:
                    details = json.loads(args[1][0]['details']); details['call_id']='other-call'
                    args[1][0]['details']=json.dumps(details)
                result = R.evaluate_replay(*args)
                self.assertFalse(result['single_call_audit_verified'])
                self.assertFalse(result['passed'])

    def test_completion_requires_boolean_identity_verification_and_exact_actual_model(self):
        for changes in ({'model_identity_verified':False}, {'model_identity_verified':'true'},
                        {'actual_model':'legalone-1.7b'}, {'actual_model':''}, {'actual_model':None}):
            with self.subTest(changes=changes):
                args = assessment_fixture(); details = json.loads(args[1][1]['details'])
                details.update(changes); args[1][1]['details'] = json.dumps(details)
                result = R.evaluate_replay(*args)
                self.assertFalse(result['completion_model_identity_verified'])
                self.assertFalse(result['model_response_not_truncated'])
                self.assertFalse(result['passed'])

    def test_unknown_source_and_status_must_remain_pending(self):
        for changes in ({'source_complete':False}, {'source_id':''}, {'effective_status':'unknown'},
                        {'effective_status':'conflicting_evidence'}, {'effective_status':None}):
            with self.subTest(changes=changes):
                args = assessment_fixture(); args[2][0].update(changes)
                self.assertFalse(R.evaluate_replay(*args)['passed'])
                args = assessment_fixture(pending=True); args[2][0].update(changes)
                result = R.evaluate_replay(*args)
                self.assertTrue(result['passed'])
                self.assertTrue(result['pending'])
                self.assertEqual(result['required_pending_feedback_item_ids'], ['feedback-1'])
                self.assertFalse(result['legal_truth_independently_evaluated'])

    def test_upstream_pending_and_each_subject_truncation_must_be_preserved(self):
        for field in ('upstream_pending','fact_evidence_text_truncated','source_evidence_text_truncated'):
            with self.subTest(field=field):
                args = assessment_fixture(); args[2][0][field] = True
                self.assertFalse(R.evaluate_replay(*args)['passed'])
                args = assessment_fixture(pending=True); args[2][0][field] = True
                self.assertTrue(R.evaluate_replay(*args)['passed'])

    def test_truncated_statute_source_must_remain_pending(self):
        args = assessment_fixture('statute'); args[2][0]['article_quote_truncated'] = True
        self.assertFalse(R.evaluate_replay(*args)['passed'])
        args = assessment_fixture('statute', pending=True); args[2][0]['article_quote_truncated'] = True
        self.assertTrue(R.evaluate_replay(*args)['passed'])

    def test_model_needs_manual_is_pending_even_with_complete_source(self):
        self.assertFalse(R.evaluate_replay(*assessment_fixture(verdict='needs_manual'))['passed'])
        self.assertTrue(R.evaluate_replay(*assessment_fixture(pending=True,verdict='needs_manual'))['passed'])

    def test_complete_and_verified_accounting_cannot_contradict_pending_ids(self):
        for field, value in (('feedback_complete',True), ('verified_feedback_item_ids',['feedback-1']),
                              ('uncompleted_feedback_item_ids',['feedback-1','feedback-1'])):
            with self.subTest(field=field):
                args = assessment_fixture(pending=True); args[0][field] = value
                self.assertFalse(R.evaluate_replay(*args)['pending_accounting_consistent'])
                self.assertFalse(R.evaluate_replay(*args)['passed'])

    def test_missing_or_foreign_returned_ids_and_nonstop_outputs_do_not_pass(self):
        for mode in ('missing','wrong-id','length','degraded'):
            with self.subTest(mode=mode):
                args = assessment_fixture()
                if mode == 'missing': args[0]['calls'][0]['subject_fact_verifications'] = []
                elif mode == 'wrong-id': args[0]['calls'][0]['subject_fact_verifications'][0]['feedback_item_id']='wrong'
                elif mode == 'degraded': args[1][1]['status']='degraded'
                else:
                    details=json.loads(args[1][1]['details']); details['finish_reason']='length'
                    args[1][1]['details']=json.dumps(details)
                self.assertFalse(R.evaluate_replay(*args)['passed'])


if __name__ == '__main__':
    unittest.main()
