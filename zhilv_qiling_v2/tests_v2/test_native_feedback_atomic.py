"""Offline atomic statute-feedback protocol tests; synthetic stubs are not legal gold."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import llm_client as L, long_context as C, model_audit as AUDIT, nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse


class NativeFeedbackAtomicTests(unittest.TestCase):
    def setUp(self):
        self.a = Clause('a', '', '', 1, '', '甲方应在最终验收后支付尾款。',
                        '甲方应在最终验收后支付尾款。', ['block-a'])
        self.b = Clause('b', '', '', 1, '', '乙方应在收款后启动最终验收。',
                        '乙方应在收款后启动最终验收。', ['block-b'])
        self.rows = [self.a, self.b]
        self.contract = Contract('offline-atomic-contract', DocumentParse(
            'SYNTHETIC-atomic-protocol.txt', 'txt', '\n'.join(row.text for row in self.rows), clauses=self.rows))
        self.item = {'feedback_item_id': 'f-1', 'candidate_id': 'candidate-1', 'article_uid': 'law-1',
                     'clause_ids': ['a'], 'article_quote': '合成法条一：当事人应按约履行付款义务。'}

    def items(self, count=3):
        return [{**self.item, 'feedback_item_id': f'f-{index}', 'candidate_id': f'candidate-{index}',
                 'article_uid': f'law-{index}', 'article_quote': f'合成法条{index}：当事人应按约履行义务。'}
                for index in range(1, count + 1)]

    def review(self, items=None, *, native=True, facts=(), mutate=None, max_calls=0,
               available=True, fail_transport=False, direct=False, finish='stop'):
        items = [self.item] if items is None else items
        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1', llm_model='legalone-8b',
                        llm_native_json_schema=native, llm_chunk_chars=6000, llm_chunk_overlap=0,
                        llm_cross_clause_pairs=0, llm_max_calls=max_calls)
        captured = []

        def response(messages, options):
            payload = json.loads(messages[1]['content'])
            schema = options.get('json_schema')
            entry = {'payload': copy.deepcopy(payload), 'schema': copy.deepcopy(schema),
                     'system': messages[0]['content']}
            captured.append(entry)
            if fail_transport:
                raise L.LLMClientError('OFFLINE-STUB deliberate transport failure')
            statute = payload.get('verification_feedback', [])
            subject = payload.get('subject_fact_feedback', [])
            root = 'subject_fact_verifications' if subject else 'verifications'
            rows = []
            for item in statute or subject:
                keys = ('feedback_item_id', 'fact_id', 'source_id', 'subject_name', 'fact_type', 'clause_id') if subject else (
                    'feedback_item_id', 'candidate_id', 'article_uid')
                row = {key: item[key] for key in keys}
                if schema:
                    props = schema['properties'][root]['items']['properties']
                    clause_quote = props['clause_quote']['enum'][0]
                else:
                    clause_quote = payload['clause_sources'][item['bound_clause_ids'][0]][0]['text']
                row.update(verdict='confirmed', reason='合成协议检查，不作法律认证。', clause_quote=clause_quote)
                if subject:
                    row.update(fact_quote=item['fact_evidence_text'], source_quote=item['source_evidence_text'])
                else:
                    row['article_quote'] = (props['article_quote']['enum'][0]
                                            if schema and 'enum' in props['article_quote'] else item['article_quote'][:80])
                rows.append(row)
            content = {root: rows} if statute or subject else {'defects': [], 'more_findings_possible': False}
            if mutate:
                content = mutate(copy.deepcopy(content), payload, schema, len(captured))
            entry['response'] = copy.deepcopy(content)
            return content

        def complete(_config, messages, _max_tokens, **options):
            content = response(messages, options)
            return json.dumps(content, ensure_ascii=False), {'model': 'legalone-8b',
                'choices': [{'finish_reason': finish}], 'usage': {'completion_tokens': 100}}, 0.001

        def request(_config, messages, **options):
            return SimpleNamespace(content=response(messages, options), model='offline-direct-stub',
                                   call_id='offline-direct-call', degraded='')

        with patch.object(L, 'available', return_value=available):
            with patch.object(L, 'request_json', side_effect=request) if direct else patch.object(L, '_complete', side_effect=complete):
                with AUDIT.review_audit('OFFLINE-ATOMIC-STUB', self.contract.contract_id) as audit:
                    candidates, meta, degraded = N._llm_logic(self.contract, config, self.rows,
                        verification_feedback=items, subject_fact_feedback=facts)
        return candidates, meta, degraded, captured, audit.events

    def test_native_batches_are_single_items_with_truthful_plan_and_receipts(self):
        _, meta, _, captured, events = self.review(self.items(4))
        self.assertEqual([len(row['payload']['verification_feedback']) for row in captured], [1] * 4)
        self.assertEqual(meta['planned_calls'], 4)
        self.assertEqual(meta['completed_calls'], 4)
        self.assertEqual(meta['feedback_max_items_per_batch'], 1)
        self.assertEqual(meta['native_statute_feedback_planned_jobs'], 4)
        self.assertEqual(meta['native_statute_feedback_not_started_job_ids'], [])
        batches = meta['feedback_windows'][0]['batches']
        self.assertEqual([row['batch_index'] for row in batches], [1, 2, 3, 4])
        self.assertTrue(all(row['batch_count'] == 4 for row in batches))
        self.assertEqual([call['feedback_items_in_other_batches'] for call in meta['calls']], [3] * 4)
        self.assertTrue(meta['feedback_complete'])
        self.assertTrue(all(call['call_id'] and call['source_sha256'] for call in meta['calls']))
        self.assertEqual(len([event for event in events if event['status'] == 'succeeded']), 4)

    def test_exact_one_schema_binds_only_this_item_ids_and_both_visible_sources(self):
        _, _, _, captured, _ = self.review([{**self.item, 'clause_ids': ['b']}])
        schema = captured[0]['schema']
        array = schema['properties']['verifications']
        self.assertEqual((array['minItems'], array['maxItems']), (1, 1))
        props = array['items']['properties']
        for field in ('feedback_item_id', 'candidate_id', 'article_uid'):
            self.assertEqual(props[field]['enum'], [self.item[field]])
        self.assertEqual(props['clause_quote']['enum'], [self.b.text])
        self.assertEqual(props['article_quote']['enum'], [self.item['article_quote']])
        self.assertEqual(set(props['verdict']['enum']), {'confirmed', 'rejected', 'needs_manual'})
        self.assertFalse(schema['additionalProperties'])
        self.assertFalse(array['items']['additionalProperties'])

    def test_article_candidates_are_exact_continuous_overlapping_visible_slices(self):
        article = '\n ' + ''.join(chr(0x4e00 + index) for index in range(171)) + '\n'
        schema, issues = N._native_statute_feedback_schema([{**self.item, 'article_quote': article}], {'a': [self.a.text]})
        self.assertEqual(issues, [])
        choices = schema['properties']['verifications']['items']['properties']['article_quote']['enum']
        self.assertEqual(choices, [article[start:start + 80] for start in (0, 40, 80, 120)])
        self.assertTrue(all(quote in article and len(quote) <= 80 for quote in choices))
        self.assertTrue(choices[-1].endswith(article[-1]))

    def test_bound_contract_sources_do_not_join_segments_or_admit_other_clause(self):
        item = {**self.item, 'clause_ids': ['a', 'b'], 'bound_clause_ids': ['a']}
        schema, issues = N._native_statute_feedback_schema([item], {'a': ['片段甲。', '片段乙。'], 'b': [self.b.text]})
        self.assertEqual(issues, [])
        self.assertEqual(schema['properties']['verifications']['items']['properties']['clause_quote']['enum'], ['片段甲。', '片段乙。'])

    def assert_schema_failure_pending(self, mutate):
        candidates, meta, _, captured, events = self.review(mutate=mutate)
        self.assertEqual(candidates, [])
        self.assertFalse(meta['feedback_complete'])
        self.assertEqual(meta['uncompleted_feedback_item_ids'], ['f-1'])
        self.assertEqual(meta['calls'][0]['status'], 'error')
        self.assertEqual(meta['completed_calls'], 0)
        self.assertTrue(captured)
        self.assertIn('native_json_schema_validation_failed', {row.get('reason') for row in events})
        return captured

    def test_empty_model_array_fails_schema_without_synthesizing_return(self):
        captured = self.assert_schema_failure_pending(lambda content, *_: {'verifications': []})
        self.assertTrue(all(row['response'] == {'verifications': []} for row in captured))

    def test_duplicate_model_rows_fail_schema(self):
        self.assert_schema_failure_pending(lambda content, *_: {'verifications': content['verifications'] * 2})

    def test_wrong_item_candidate_and_uid_cannot_be_cross_paired(self):
        for field in ('feedback_item_id', 'candidate_id', 'article_uid'):
            with self.subTest(field=field):
                def mutate(content, *_):
                    content['verifications'][0][field] = 'other-item-value'
                    return content
                self.assert_schema_failure_pending(mutate)

    def test_wrong_article_or_contract_quote_fails_schema(self):
        for field, quote in (('article_quote', '另一项法条，不是本项提供的正文。'), ('clause_quote', self.b.text)):
            with self.subTest(field=field):
                def mutate(content, *_):
                    content['verifications'][0][field] = quote
                    return content
                self.assert_schema_failure_pending(mutate)

    def test_direct_stub_bypassing_schema_still_cannot_complete_bad_binding(self):
        def mutate(content, *_):
            content['verifications'][0]['article_uid'] = 'other-item-value'
            return content
        _, meta, _, captured, _ = self.review(mutate=mutate, direct=True)
        self.assertFalse(meta['feedback_complete'])
        self.assertEqual(meta['calls'][0]['verification_errors'][0]['reason'], 'article_uid_mismatch')
        self.assertEqual(captured[0]['response']['verifications'][0]['article_uid'], 'other-item-value')

    def test_all_three_verdicts_remain_model_choices_and_manual_stays_pending(self):
        verdicts = ['confirmed', 'rejected', 'needs_manual']
        def mutate(content, payload, *_):
            index = int(payload['verification_feedback'][0]['feedback_item_id'].split('-')[-1]) - 1
            content['verifications'][0]['verdict'] = verdicts[index]
            return content
        candidates, meta, _, _, _ = self.review(self.items(3), mutate=mutate)
        self.assertEqual(candidates, [])
        self.assertEqual([call['verifications'][0]['verdict'] for call in meta['calls']], verdicts)
        self.assertEqual(meta['verified_feedback_item_ids'], ['f-1', 'f-2'])
        self.assertEqual(meta['uncompleted_feedback_item_ids'], ['f-3'])
        self.assertFalse(meta['feedback_complete'])

    def test_missing_or_invalid_article_source_skips_model_and_remains_pending(self):
        for source in ('', ' \n', None, {'not': 'legal text'}):
            with self.subTest(source=source):
                _, meta, _, captured, events = self.review([{**self.item, 'article_quote': source}])
                self.assertEqual(captured, [])
                self.assertEqual(events, [])
                self.assertFalse(meta['feedback_complete'])
                self.assertEqual(meta['calls'][0]['status'], 'feedback_source_unavailable')
                self.assertIn('no_bound_article_quote_selections', meta['calls'][0]['source_errors'])
                self.assertEqual(len(meta['native_statute_feedback_source_unavailable_job_ids']), 1)

    def test_missing_contract_source_has_no_empty_enum_request(self):
        window = C.ReviewWindow('empty-source', 'text', (C.Segment('a', 0, 1, ' '),))
        with patch.object(C, 'build_review_plan', return_value=([window], {'cross_windows': 0})):
            _, meta, _, captured, events = self.review()
        self.assertEqual(captured, [])
        self.assertEqual(events, [])
        self.assertFalse(meta['feedback_complete'])
        self.assertIn('no_bound_contract_quote_selections', meta['calls'][0]['source_errors'])

    def test_missing_uid_skips_unsatisfiable_binding_without_filling_it(self):
        _, meta, _, captured, _ = self.review([{**self.item, 'article_uid': ''}])
        self.assertEqual(captured, [])
        self.assertEqual(meta['feedback_items'][0]['article_uid'], '')
        self.assertIn('native_statute_feedback_binding_missing:article_uid', meta['calls'][0]['source_errors'])
        self.assertFalse(meta['feedback_complete'])

    def test_truncated_visible_article_never_completes_even_if_model_confirms(self):
        source = ''.join(chr(0x4e00 + index) for index in range(1100)) + 'NOT-SENT-TAIL'
        _, meta, _, captured, _ = self.review([{**self.item, 'article_quote': source}])
        sent = captured[0]['payload']['verification_feedback'][0]
        choices = captured[0]['schema']['properties']['verifications']['items']['properties']['article_quote']['enum']
        self.assertEqual(sent['article_quote'], source[:1000])
        self.assertTrue(sent['article_quote_truncated'])
        self.assertTrue(all(quote in sent['article_quote'] for quote in choices))
        self.assertTrue(all('NOT-SENT-TAIL' not in quote for quote in choices))
        self.assertEqual(meta['calls'][0]['verifications'][0]['verdict'], 'confirmed')
        self.assertIn('article_source_truncated', {row['reason'] for row in meta['calls'][0]['verification_errors']})
        self.assertFalse(meta['feedback_complete'])

    def test_model_output_truncation_remains_pending(self):
        _, meta, _, _, _ = self.review(finish='length')
        self.assertEqual(meta['calls'][0]['status'], 'partial')
        self.assertFalse(meta['feedback_complete'])

    def test_budget_preserves_all_unstarted_single_item_jobs(self):
        _, meta, _, captured, _ = self.review(self.items(4), max_calls=1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(meta['planned_calls'], 4)
        self.assertEqual(meta['completed_calls'], 1)
        self.assertEqual(len(meta['native_statute_feedback_not_started_job_ids']), 3)
        self.assertEqual(meta['uncompleted_feedback_item_ids'], ['f-2', 'f-3', 'f-4'])
        self.assertEqual([row['status'] for row in meta['feedback_windows'][0]['batches']], ['ok', 'not_started', 'not_started', 'not_started'])
        self.assertFalse(meta['feedback_complete'])

    def test_circuit_breaker_keeps_unstarted_jobs_and_items(self):
        _, meta, _, captured, _ = self.review(self.items(5), fail_transport=True)
        self.assertEqual(len(captured), 3)
        self.assertEqual(meta['planned_calls'], 5)
        self.assertEqual(len(meta['native_statute_feedback_not_started_job_ids']), 2)
        self.assertEqual(len(meta['uncompleted_feedback_item_ids']), 5)
        self.assertFalse(meta['feedback_complete'])

    def test_unavailable_endpoint_does_not_count_jobs_as_started(self):
        _, meta, _, captured, _ = self.review(self.items(4), available=False)
        self.assertEqual(captured, [])
        self.assertEqual(meta['calls'], [])
        self.assertEqual(len(meta['native_statute_feedback_not_started_job_ids']), 4)
        self.assertEqual(len(meta['uncompleted_feedback_item_ids']), 4)

    def test_duplicate_input_id_is_not_repaired_by_single_item_batches(self):
        _, meta, _, _, _ = self.review([self.item, copy.deepcopy(self.item)])
        self.assertEqual([row['feedback_item_id'] for row in meta['feedback_items']], ['f-1', 'f-1'])
        self.assertEqual(meta['verified_feedback_item_ids'], [])
        self.assertFalse(meta['feedback_complete'])
        self.assertTrue(all('duplicate_feedback_input_id' in {row['reason'] for row in call['verification_errors']}
                            for call in meta['calls']))

    def test_oversize_and_unmatched_items_remain_visible_pending(self):
        item = {**self.item, 'note': '未截断字段' * 1500}
        _, meta, _, captured, _ = self.review([item])
        self.assertEqual(captured, [])
        self.assertEqual(meta['feedback_windows'][0]['oversize_feedback_item_ids'], ['f-1'])
        self.assertFalse(meta['feedback_complete'])
        _, meta, _, _, _ = self.review([{**self.item, 'clause_ids': ['not-in-contract']}])
        self.assertEqual(meta['unmatched_feedback_item_ids'], ['f-1'])
        self.assertEqual(meta['uncompleted_feedback_item_ids'], ['f-1'])

    def test_non_native_keeps_three_item_batches_without_new_schema_or_policy(self):
        _, meta, _, captured, _ = self.review(self.items(7), native=False)
        self.assertEqual([len(row['payload']['verification_feedback']) for row in captured], [3, 3, 1])
        self.assertEqual(meta['feedback_max_items_per_batch'], 3)
        self.assertTrue(all(row['schema'] is None for row in captured))
        self.assertNotIn('native_statute_feedback_policy', meta)
        self.assertTrue(meta['feedback_complete'])

    def test_subject_native_schema_and_unknown_semantics_are_unchanged(self):
        fact = {'feedback_item_id': 'sf-1', 'fact_id': 'fact-1', 'source_id': '', 'subject_name': '甲方',
                'fact_type': 'license', 'clause_id': 'a', 'clause_ids': ['a'], 'source_complete': False,
                'effective_status': 'unknown', 'fact_evidence_text': '合成未知记录。', 'source_evidence_text': ''}
        _, meta, _, captured, _ = self.review([], facts=[fact, {**fact, 'feedback_item_id': 'sf-2'}])
        self.assertEqual(len(captured), 2)
        array = captured[0]['schema']['properties']['subject_fact_verifications']
        self.assertEqual((array['minItems'], array['maxItems']), (0, 1))
        self.assertNotIn('enum', array['items']['properties']['fact_quote'])
        self.assertNotIn('enum', array['items']['properties']['source_quote'])
        self.assertNotIn('article_quote_mode', captured[0]['payload'])
        self.assertNotIn('native_statute_feedback_policy', meta)
        self.assertFalse(meta['feedback_complete'])

    def test_mixed_native_statute_and_subject_jobs_stay_separate(self):
        fact = {'feedback_item_id': 'sf-1', 'fact_id': 'fact-1', 'source_id': 'source-1', 'subject_name': '甲方',
                'fact_type': 'license', 'clause_id': 'a', 'clause_ids': ['a'], 'source_complete': True,
                'effective_status': 'valid', 'fact_evidence_text': '合成记录。', 'source_evidence_text': '合成来源。'}
        _, meta, _, captured, _ = self.review(self.items(3), facts=[fact])
        self.assertEqual(len(captured), 4)
        self.assertEqual(meta['native_statute_feedback_planned_jobs'], 3)
        self.assertEqual(meta['planned_calls'], 4)
        self.assertTrue(all(bool(row['payload'].get('verification_feedback')) != bool(row['payload'].get('subject_fact_feedback')) for row in captured))

    def test_helper_rejects_multiple_items_and_does_not_conflate_source_with_truth(self):
        schema, issues = N._native_statute_feedback_schema(self.items(2), {'a': [self.a.text]})
        self.assertIsNone(schema)
        self.assertEqual(issues, ['native_statute_feedback_requires_single_item'])
        _, meta, _, captured, _ = self.review()
        self.assertTrue(meta['native_statute_feedback_policy']['schema_failure_is_not_legal_accuracy'])
        self.assertIn('不得为完成协议强行确认', captured[0]['system'])
        self.assertIn('不表示相关性、完整性、适用性或法律结论正确', captured[0]['system'])
        self.assertIn('not_legal_truth', meta['calls'][0]['article_quote_scope'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
