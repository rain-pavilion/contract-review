"""Offline candidate protocol tests. Stubs never establish legal quality."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents as A, llm_client as L, model_audit as AUDIT, nodes as N
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.long_context import build_review_plan
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, RISK_CODES


def clause(cid, text):
    return Clause(cid, '', '', 1, '', text, text, ['block-' + cid])


def finding(cid='alpha', quote='约定付款条件应与验收记录一致。', **changes):
    value = {'clause_id': cid, 'code': 'fee_mismatch', 'severity': 'medium',
             'message': '需要结合合同约定复核。', 'quote': quote, 'related_quotes': []}
    value.update(changes)
    return value


class InitialNativeOutputTests(unittest.TestCase):
    def review(self, content, *, rows=None, native=True, finish='stop', direct=False):
        rows = rows or [clause('alpha', '约定付款条件应与验收记录一致。')]
        contract = Contract('offline-initial-candidate', DocumentParse(
            'synthetic-protocol.txt', 'txt', '\n'.join(c.text for c in rows), clauses=rows))
        config = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:1/v1',
                        llm_model='legalone-8b', llm_native_json_schema=native,
                        llm_chunk_chars=6000, llm_chunk_overlap=0,
                        llm_cross_clause_pairs=0, llm_max_tokens=4096,
                        # These tests isolate one native response. Multi-attempt
                        # execution is covered by test_initial_review_nodes.
                        llm_initial_max_attempts=1)
        captured = []

        def complete(_config, messages, _max_tokens, **kwargs):
            captured.append({'payload': json.loads(messages[1]['content']),
                             'messages': copy.deepcopy(messages), **copy.deepcopy(kwargs)})
            return json.dumps(content, ensure_ascii=False), {
                'model': 'legalone-8b', 'choices': [{'finish_reason': finish}],
                'usage': {'completion_tokens': 4096 if finish == 'length' else 100}}, 0.001

        def direct_reply(_config, messages, **kwargs):
            captured.append({'payload': json.loads(messages[1]['content']), **copy.deepcopy(kwargs)})
            return SimpleNamespace(content=copy.deepcopy(content), model='offline-stub',
                                   call_id='offline-direct-stub', degraded='')

        with patch.object(L, 'request_json', side_effect=direct_reply) if direct else \
                patch.object(L, '_complete', side_effect=complete):
            with AUDIT.review_audit('offline-candidate-test', contract.contract_id) as audit:
                candidates, meta, degraded = N._llm_logic(contract, config, rows)
        return candidates, meta, degraded, captured, audit.events, contract, config

    def test_initial_flag_sends_schema_and_preserves_source_and_bindings(self):
        source = '自由选取的合同原文片段必须保留。' * 9
        row = clause('alpha', source)
        candidates, meta, _, sent, events, _, _ = self.review(
            {'defects': [], 'more_findings_possible': False}, rows=[row])
        self.assertEqual(candidates, [])
        self.assertTrue(meta['complete'])
        self.assertEqual(''.join(s['text'] for s in sent[0]['payload']['segments']), source)
        self.assertEqual(set(sent[0]['json_schema']['properties']), {'defects', 'more_findings_possible'})
        self.assertNotIn('clause_quote_mode', sent[0]['payload'])
        self.assertEqual([e['status'] for e in events], ['started', 'succeeded'])
        for key in ('call_id', 'window_id', 'job_id', 'source_sha256'):
            self.assertEqual(events[0]['details'][key], events[1]['details'][key])
            self.assertTrue(events[0]['details'][key])
        self.assertTrue(all(e['details']['native_json_schema_sent'] for e in events))
        self.assertTrue(all(e['details']['json_schema_sha256'] for e in events))
        self.assertFalse(meta['initial_output_policy']['token_completion_guaranteed'])

    def test_flag_false_keeps_ordinary_behavior_without_new_fields(self):
        _, meta, _, sent, events, _, _ = self.review({'defects': []}, native=False)
        self.assertTrue(meta['complete'])
        self.assertNotIn('json_schema', sent[0])
        self.assertNotIn('output_policy', sent[0]['payload'])
        self.assertNotIn('initial_output_policy', meta)
        self.assertFalse(events[0]['details']['native_json_schema_requested'])

    def test_schema_allows_empty_and_all_substantive_codes_but_not_status_sentinel(self):
        rows = [clause('alpha', '付款条件。'), clause('beta', '履行期限。')]
        windows, _ = build_review_plan(rows, chunk_chars=6000, overlap=0, cross_pairs=0)
        schema = N._logic_output_json_schema(windows[0])
        props = schema['properties']['defects']['items']['properties']
        self.assertEqual(set(props['code']['enum']), set(RISK_CODES) - {'no_substantive_risk'})
        self.assertEqual(set(props['clause_id']['enum']), {'alpha', 'beta'})
        self.assertNotIn('enum', props['quote'])
        self.assertNotIn('enum', props['related_quotes']['items']['properties']['quote'])
        L._validate_json_schema({'defects': [], 'more_findings_possible': False}, schema)
        for code in sorted(set(RISK_CODES) - {'no_substantive_risk'}):
            L._validate_json_schema({'defects': [finding(quote='付款条件。', code=code)],
                                    'more_findings_possible': False}, schema)
        with self.assertRaises(L.LLMClientError):
            L._validate_json_schema({'defects': [finding(code='no_substantive_risk')],
                                    'more_findings_possible': False}, schema)
        self.assertEqual(schema['properties']['defects']['minItems'], 0)

    def test_free_continuous_quote_need_not_be_a_sampled_source_extract(self):
        source = 'abcdefghijklmnopqrstuvwxyz' * 5
        quote = source[17:58]
        values, meta, _, sent, _, _, _ = self.review(
            {'defects': [finding(quote=quote)], 'more_findings_possible': False},
            rows=[clause('alpha', source)])
        self.assertEqual(len(values), 1)
        self.assertTrue(meta['complete'])
        self.assertEqual(values[0].evidence[0].extra['quote'], quote)
        self.assertTrue(values[0].evidence[0].extra['quote_exact_source_match'])
        self.assertNotIn('native_source_extract_selection', sent[0]['messages'][0]['content'])

    def assert_pending(self, meta, reason):
        self.assertFalse(meta['complete'])
        self.assertFalse(meta['text_complete'])
        self.assertEqual(meta['completed_calls'], 0)
        self.assertEqual(meta['calls'][0]['status'], 'partial')
        self.assertIn(reason, meta['calls'][0]['output_pending_reasons'])
        self.assertFalse(meta['calls'][0]['output_scope_complete'])

    def test_model_more_findings_stays_pending_but_keeps_bound_candidate(self):
        values, meta, _, _, events, _, _ = self.review(
            {'defects': [finding()], 'more_findings_possible': True})
        self.assert_pending(meta, 'initial_output_more_findings_possible')
        self.assertEqual(len(values), 1)
        self.assertTrue(values[0].evidence[0].extra['partial_model_output'])
        self.assertEqual(events[-1]['status'], 'succeeded')
        self.assertEqual(meta['calls'][0]['reason'], 'initial_output_budget_or_source_pending')

    def test_eight_findings_capacity_is_pending_even_if_model_says_no_more(self):
        quotes = [f'合同第{i}项需要对照履行。' for i in range(8)]
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote=q) for q in quotes], 'more_findings_possible': False},
            rows=[clause('alpha', ''.join(quotes))])
        self.assert_pending(meta, 'initial_output_findings_capacity_reached')
        self.assertEqual(len(values), 8)
        self.assertTrue(all(e.extra['partial_model_output'] for c in values for e in c.evidence))

    def test_direct_over_capacity_reply_is_never_sliced(self):
        quotes = [f'合同第{i}项需要对照履行。' for i in range(11)]
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote=q) for q in quotes], 'more_findings_possible': False},
            rows=[clause('alpha', ''.join(quotes))], direct=True)
        self.assert_pending(meta, 'initial_output_findings_capacity_reached')
        self.assertIn('initial_output_schema_invalid', meta['calls'][0]['output_pending_reasons'])
        self.assertEqual(len(values), 11)
        self.assertEqual(meta['accepted'], 11)

    def test_actual_transport_rejects_over_capacity_without_clipping(self):
        content = {'defects': [finding() for _ in range(9)], 'more_findings_possible': False}
        _, meta, _, sent, events, _, _ = self.review(content)
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['calls'][0]['status'], 'error')
        self.assertTrue(all(e['reason'] == 'native_json_schema_validation_failed'
                            for e in events if e['status'] == 'failed'))
        self.assertEqual(len(content['defects']), 9)
        # Native initial retries are explicit window attempts, with no hidden
        # request_json parse retry outside llm_initial_max_attempts.
        self.assertEqual(len(sent), 1)

    def test_message_character_capacity_stays_pending_and_is_not_clipped(self):
        message = '说明' * 80
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(message=message)], 'more_findings_possible': False})
        self.assert_pending(meta, 'initial_output_message_capacity_reached')
        self.assertEqual(values[0].message, message)

    def test_quote_character_capacity_stays_pending_and_is_preserved(self):
        quote = '甲' * 160
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote=quote)], 'more_findings_possible': False},
            rows=[clause('alpha', quote)])
        self.assert_pending(meta, 'initial_output_quote_capacity_reached')
        self.assertEqual(values[0].evidence[0].extra['quote'], quote)

    def test_related_capacity_stays_pending_and_keeps_all_bound_sources(self):
        rows = [clause('alpha', '主条款。'), clause('beta', '第二来源。'), clause('gamma', '第三来源。')]
        item = finding(quote=rows[0].text, related_quotes=[{'clause_id': r.clause_id, 'quote': r.text} for r in rows[1:]])
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [item], 'more_findings_possible': False}, rows=rows)
        self.assert_pending(meta, 'initial_output_related_capacity_reached')
        self.assertEqual(set(values[0].clause_ids), {'alpha', 'beta', 'gamma'})
        self.assertEqual(len(values[0].evidence[0].extra['bound_model_quotes']), 3)

    def test_more_related_sources_can_be_reported_pending_below_capacity(self):
        rows = [clause('alpha', '主条款。'), clause('beta', '第二来源。')]
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote=rows[0].text, related_quotes=[{'clause_id': 'beta', 'quote': rows[1].text}])],
             'more_findings_possible': True}, rows=rows)
        self.assert_pending(meta, 'initial_output_more_findings_possible')
        self.assertEqual(len(values), 1)

    def test_nonexistent_primary_source_never_establishes_coverage(self):
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote='不存在于发送窗口的文字。')], 'more_findings_possible': False})
        self.assert_pending(meta, 'initial_output_source_not_exactly_bound')
        self.assertEqual(values, [])

    def test_nonexistent_related_source_never_establishes_coverage(self):
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(related_quotes=[{'clause_id': 'alpha', 'quote': '伪造的相关引文。'}])],
             'more_findings_possible': False})
        self.assert_pending(meta, 'initial_output_source_not_exactly_bound')
        self.assertEqual(values, [])

    def test_whitespace_normalized_only_quote_is_pending_not_exact_proof(self):
        values, meta, _, _, _, _, _ = self.review(
            {'defects': [finding(quote='甲乙付款。')], 'more_findings_possible': False},
            rows=[clause('alpha', '甲 乙付款。')])
        self.assert_pending(meta, 'initial_output_source_not_exactly_bound')
        self.assertEqual(len(values), 1)
        self.assertFalse(values[0].evidence[0].extra['quote_exact_source_match'])
        self.assertTrue(values[0].evidence[0].extra['partial_model_output'])

    def test_real_length_finish_is_partial_even_for_complete_valid_json(self):
        values, meta, _, _, events, _, _ = self.review(
            {'defects': [finding()], 'more_findings_possible': False}, finish='length')
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['calls'][0]['reason'], 'output_truncated')
        self.assertEqual(events[-1]['status'], 'degraded')
        self.assertEqual(events[-1]['reason'], 'output_truncated')
        self.assertEqual(events[-1]['details']['finish_reason'], 'length')
        self.assertTrue(values[0].evidence[0].extra['partial_model_output'])

    def test_missing_budget_status_is_pending_for_direct_callers(self):
        values, meta, _, _, _, _, _ = self.review({'defects': [finding()]}, direct=True)
        self.assert_pending(meta, 'initial_output_schema_invalid')
        self.assertEqual(len(values), 1)

    def test_source_id_outside_window_is_rejected_by_transport_schema(self):
        _, meta, _, _, events, _, _ = self.review(
            {'defects': [finding(cid='not-sent')], 'more_findings_possible': False})
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['calls'][0]['status'], 'error')
        self.assertTrue(any(e['reason'] == 'native_json_schema_validation_failed' for e in events))

    def test_pending_metadata_does_not_change_legal_score_or_evidence_weight(self):
        complete, _, _, _, _, contract, config = self.review(
            {'defects': [finding()], 'more_findings_possible': False})
        pending, _, _, _, _, _, _ = self.review(
            {'defects': [finding()], 'more_findings_possible': True})
        before = A.fuse(copy.deepcopy(complete), [], contract, config)
        after = A.fuse(copy.deepcopy(pending), [], contract, config)
        self.assertEqual(A.score_risks(before, config), A.score_risks(after, config))
        self.assertEqual([(e.kind, e.text, e.strength, e.source) for e in before[0].evidence],
                         [(e.kind, e.text, e.strength, e.source) for e in after[0].evidence])


if __name__ == '__main__':
    unittest.main()
