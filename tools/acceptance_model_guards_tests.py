"""Isolated acceptance-tool guardrails; no server, model, or benchmark run."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import acceptance_lawbench as lawbench
import acceptance_model_http as model_http


class Response:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps(self.value).encode()


def native_payload(parameters=8195978240, alias='legalone-8b'):
    return {'data':[{'id':alias, 'owned_by':'llamacpp',
                     'meta':{'n_params':parameters, 'n_ctx':8192, 'ftype':'Q4_K - Medium'}}]}


class HttpCompletionGuards(unittest.TestCase):
    def setUp(self):
        isolation=patch.dict('os.environ',{'ZHILV_ACCEPTANCE_SKIP_GIT_PROVENANCE_REASON':'isolated acceptance guard tests do not invoke external git'})
        isolation.start();self.addCleanup(isolation.stop)

    def report(self):
        details = {'call_id':'actual-call-1', 'actual_model':'legalone-8b',
                   'model_identity_verified':True}
        return {'stats':{'logic_llm':{'complete':True, 'text_windows':1,
                    'text_coverage':1.0, 'unreviewed_clause_ids':[],
                    'cross_windows':1, 'cross_windows_completed':1,
                    'planned_calls':2, 'completed_calls':2,
                    'feedback_complete':True, 'uncompleted_feedback_item_ids':[],
                    'unmatched_feedback_item_ids':[], 'feedback_runs':[], 'feedback_runs_complete':True},
                'cross_validation':{'termination':'stable', 'converged':True, 'rounds':[{},{}]},
                'degraded':[], 'model_calls':[
                    {'component':'llm','operation':'request_json','status':'started', 'details':details},
                    {'component':'llm','operation':'request_json','status':'succeeded', 'details':details}]}}

    def test_complete_success_has_separate_duration_and_completion(self):
        result = model_http.assess_report(self.report(), 161.427465)
        self.assertTrue(result['duration_within_three_minutes'])
        self.assertTrue(result['meets_three_minute_for_this_sample'])
        self.assertEqual(result['legal_quality_acceptance'], 'not_evaluated_no_independent_human_gold')

    def test_judge_success_cannot_cover_missing_logic(self):
        report = self.report(); report['stats']['logic_llm'] = {}
        result = model_http.assess_report(report, 2)
        self.assertTrue(result['duration_within_three_minutes'])
        self.assertFalse(result['meets_three_minute_for_this_sample'])

    def test_feedback_context_truncation_blocks_effective_complete(self):
        report = self.report()
        report['stats']['degraded'] = ['node2_logic:llm_feedback_context_partial:window-id']
        result = model_http.assess_report(report, 161.427465)
        self.assertFalse(result['model_review_completion']['effective_complete'])
        self.assertFalse(result['meets_three_minute_for_this_sample'])

    def test_unmatched_call_receipt_cannot_pass(self):
        report = self.report()
        report['stats']['model_calls'][1]['details'] = {
            **report['stats']['model_calls'][1]['details'], 'call_id':'different-call'}
        self.assertFalse(model_http.assess_report(report, 2)['meets_three_minute_for_this_sample'])

    def test_cross_window_failure_cannot_pass(self):
        report = self.report(); report['stats']['logic_llm']['cross_windows_completed'] = 0
        self.assertFalse(model_http.assess_report(report, 2)['meets_three_minute_for_this_sample'])

    def with_feedback_run(self):
        report=self.report();logic=report['stats']['logic_llm']
        run={key:copy.deepcopy(value) for key,value in logic.items()
             if key not in ('feedback_runs','feedback_runs_complete')}
        logic['feedback_runs']=[run]
        return report,run

    def test_initial_feedback_flag_is_required_without_degradation_text(self):
        report=self.report();report['stats']['logic_llm']['feedback_complete']=False
        result=model_http.assess_report(report,2)
        self.assertFalse(result['model_review_completion']['feedback_verification_complete'])
        self.assertFalse(result['meets_three_minute_for_this_sample'])

    def test_every_feedback_run_requires_complete_and_feedback_complete(self):
        for key in ('complete','feedback_complete'):
            with self.subTest(key=key):
                report,run=self.with_feedback_run();run[key]=False
                result=model_http.assess_report(report,2)
                self.assertFalse(result['model_review_completion']['feedback_verification_complete'])
                self.assertFalse(result['meets_three_minute_for_this_sample'])

    def test_each_feedback_run_requires_all_calls_and_cross_windows(self):
        for key in ('completed_calls','cross_windows_completed'):
            with self.subTest(key=key):
                report,run=self.with_feedback_run();run[key]=0
                self.assertFalse(model_http.assess_report(report,2)['meets_three_minute_for_this_sample'])

    def test_each_feedback_run_requires_empty_and_present_pending_ids(self):
        for key in ('uncompleted_feedback_item_ids','unmatched_feedback_item_ids','unreviewed_clause_ids'):
            with self.subTest(key=key):
                report,run=self.with_feedback_run();run[key]=['pending-evidence-1']
                self.assertFalse(model_http.assess_report(report,2)['meets_three_minute_for_this_sample'])
                del run[key]
                self.assertFalse(model_http.assess_report(report,2)['meets_three_minute_for_this_sample'])

    def test_summary_true_cannot_override_failed_run_and_missing_new_fields(self):
        report,run=self.with_feedback_run();del run['planned_calls']
        self.assertTrue(report['stats']['logic_llm']['feedback_runs_complete'])
        self.assertFalse(model_http.assess_report(report,2)['meets_three_minute_for_this_sample'])

    def test_legacy_feedback_is_readable_but_not_proven_complete(self):
        report=self.report();logic=report['stats']['logic_llm']
        for key in ('feedback_complete','planned_calls','completed_calls','feedback_runs',
                    'feedback_runs_complete','uncompleted_feedback_item_ids','unmatched_feedback_item_ids'):
            del logic[key]
        result=model_http.assess_report(report,161.427465)
        self.assertEqual(result['model_review_completion']['feedback_evidence_status'],'legacy_fields_unavailable')
        self.assertIsNone(result['model_review_completion']['feedback_verification_complete'])
        self.assertFalse(result['meets_three_minute_for_this_sample'])

    def test_round_limit_is_separate_from_completed_feedback_duration(self):
        report,run=self.with_feedback_run()
        report['stats']['cross_validation']={'termination':'round_limit','converged':False,'rounds':[{}, {}, {}]}
        report['stats']['degraded']=['cross_validation_round_limit']
        result=model_http.assess_report(report,161.427465)
        self.assertTrue(result['model_review_completion']['feedback_verification_complete'])
        self.assertTrue(result['meets_three_minute_for_this_sample'])
        self.assertEqual(result['cross_validation']['termination'],'round_limit')
        self.assertFalse(result['cross_validation']['converged'])
        self.assertEqual(result['cross_validation']['rounds_completed'],3)

    def test_archive_reanalysis_keeps_original_time(self):
        with tempfile.TemporaryDirectory() as folder:
            prior = Path(folder)/'prior.json'; output = Path(folder)/'validated.json'
            original = {'report_id':'report-1', 'client_upload_to_archive_response_seconds':161.427465,
                        'text_sha256':model_http.digest(model_http.SAMPLE), 'errors':[]}
            prior.write_text(json.dumps(original), encoding='utf-8')
            report = self.report(); report['document'] = {'full_text':model_http.SAMPLE.strip()}
            argv = ['acceptance_model_http.py', '--existing-report','report-1',
                    '--prior-result',str(prior),'--out',str(output)]
            with patch('sys.argv', argv), patch.object(model_http, 'request', return_value=(200,report,0.002)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(model_http.main(), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['client_upload_to_archive_response_seconds'],161.427465)
            self.assertEqual(result['archive_reread_seconds'],0.002)
            self.assertEqual(result['new_review_or_model_calls'],0)
            self.assertEqual(json.loads(prior.read_text(encoding='utf-8')),original)


class LawBenchRuntimeGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data_file = self.folder/'official-fixture.json'
        self.rows = [{'instruction':'请选一项。','question':'合成协议测试问题','answer':'A'} for _ in range(500)]
        self.data_file.write_text(json.dumps(self.rows),encoding='utf-8')
        self.a = SimpleNamespace(snapshot=self.folder, run_dir=self.folder/'run',
            model='legalone-8b',model_revision='operator-declared-test-only-not-real-weights',
            endpoint='http://not-a-server.invalid/v1',shot='zero_shot',tasks='1-2',
            max_items=1,item_limit=1,max_tokens=8,temperature=0,timeout=1)
        self.identity = lawbench.native_identity(native_payload(),self.a.model)
        self.patches = [patch.object(lawbench,'source',return_value={}),
            patch.object(lawbench,'load_task',return_value=(self.data_file,self.rows)),
            patch.object(lawbench,'provenance',return_value={'isolated_guard_test':True}),
            patch.object(lawbench,'verify_native_identity',return_value=self.identity),
            patch.object(lawbench,'preflight_context',return_value={'method':'isolated_native_budget_fixture',
                'prompt_tokens':10,'server_context_tokens':8192,'reserved_output_tokens':8,
                'reserved_stop_token':1,'required_context_tokens':19,'fits':True,'input_truncation_permitted':False}),
            patch.object(lawbench,'evaluation_preflight',return_value={'ready':True,'errors':[],'runtime':{'isolated_fixture':True}})]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def reply(self, model='legalone-8b'):
        return {'model':model,'choices':[{'message':{'content':'A'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}}

    def state(self):
        return json.loads((self.a.run_dir/'state.json').read_text(encoding='utf-8'))

    def test_alias_and_parameter_identity_are_both_required(self):
        with self.assertRaises(ValueError):
            lawbench.native_identity(native_payload(parameters=1700000000),self.a.model)
        with self.assertRaises(ValueError):
            lawbench.native_identity(native_payload(alias='other-model'),self.a.model)
        with self.assertRaises(ValueError):
            lawbench.native_identity(native_payload(), 'legalone-8b-pretend')

    def test_started_is_durable_before_request_and_wrong_response_is_failed(self):
        def wrong_response(request, timeout):
            paths=list((self.a.run_dir/'receipts').glob('*/*.json'))
            self.assertEqual(len(paths),1)
            self.assertEqual(json.loads(paths[0].read_text())['status'],'started')
            return Response(self.reply('different-model'))
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=wrong_response):
            with self.assertRaisesRegex(ValueError,'response model identity'):
                lawbench.infer(self.a)
        receipt=json.loads(next((self.a.run_dir/'receipts').glob('*/*.json')).read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(receipt['response_model'],'different-model')
        self.assertEqual(self.state()['model_calls'],1)
        self.assertFalse((self.a.run_dir/'predictions').exists())

    def test_each_success_has_usage_finish_and_identity_receipt(self):
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())):
            lawbench.infer(self.a)
        receipt=json.loads(next((self.a.run_dir/'receipts').glob('*/*.json')).read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'],'succeeded')
        self.assertTrue(receipt['model_identity_verified'])
        self.assertEqual(receipt['finish_reason'],'stop')
        self.assertEqual(receipt['usage']['completion_tokens'],1)
        self.assertGreaterEqual(receipt['elapsed_seconds'],0)
        self.assertEqual(self.state()['completed']['1-2']['items'],1)
        self.assertFalse(self.state()['full_official_benchmark'])
        self.assertNotIn('reasoning_effort',self.state()['configuration'])
        self.assertNotIn('reasoning_effort',receipt)
        snapshot=self.a.run_dir/self.state()['adapter_source_snapshot']
        self.assertEqual(snapshot.read_bytes(),Path(lawbench.__file__).read_bytes())

    def test_reasoning_mode_only_changes_explicit_runtime_configuration(self):
        default_config=lawbench.config_for(self.a)
        self.a.reasoning_effort='none'
        mode_config=lawbench.config_for(self.a)
        without_mode=dict(mode_config);del without_mode['reasoning_effort']
        self.assertEqual(default_config,without_mode)
        self.assertNotEqual(lawbench.json_digest(default_config),lawbench.json_digest(mode_config))
        def capture(request,timeout):
            body=json.loads(request.data)
            self.assertEqual(body['reasoning_effort'],'none')
            self.assertEqual(body['messages'],[{'role':'user','content':lawbench.prompt(self.rows[0])}])
            self.assertEqual(body['max_tokens'],8)
            return Response(self.reply())
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=capture):
            lawbench.infer(self.a)
        receipt=json.loads(next((self.a.run_dir/'receipts').glob('*/*.json')).read_text(encoding='utf-8'))
        self.assertEqual(receipt['reasoning_effort'],'none')
        self.assertEqual(self.state()['configuration_sha256'],lawbench.json_digest(mode_config))

    def test_mode_and_source_snapshot_drift_refuse_resume(self):
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())):
            lawbench.infer(self.a)
        self.a.reasoning_effort='none'
        with self.assertRaisesRegex(ValueError,'resume configuration changed'):
            lawbench.infer(self.a)
        self.a.reasoning_effort=None
        snapshot=self.a.run_dir/'adapter_snapshot.py'
        snapshot.write_bytes(snapshot.read_bytes()+b'\n# simulated drift\n')
        with self.assertRaisesRegex(ValueError,'source snapshot drift'):
            lawbench.infer(self.a)

    def test_success_receipt_recovers_crash_before_prediction_without_new_call(self):
        original=lawbench.atomic_json
        def interrupted_write(path,value):
            if 'predictions' in path.parts:
                raise OSError('isolated simulated crash after response receipt')
            return original(path,value)
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())), patch.object(lawbench,'atomic_json',side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError,'simulated crash'):
                lawbench.infer(self.a)
        self.assertFalse((self.a.run_dir/'predictions').exists())
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=AssertionError('must not make another inference call')):
            result=lawbench.infer(self.a)
        self.assertEqual(result['new_actual_model_responses'],0)
        self.assertEqual(self.state()['model_calls'],1)
        self.assertEqual(self.state()['completed']['1-2']['items'],1)

    def test_changed_prediction_refuses_resume(self):
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())):
            lawbench.infer(self.a)
        output=self.a.run_dir/'predictions'/self.a.model/'1-2.json'
        value=json.loads(output.read_text(encoding='utf-8'));value['0']['prediction']='tampered'
        lawbench.atomic_json(output,value)
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=AssertionError('no new request')):
            with self.assertRaisesRegex(ValueError,'matching successful receipt'):
                lawbench.infer(self.a)

    def test_unfinished_call_refuses_automatic_duplicate(self):
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                lawbench.infer(self.a)
        with patch.object(lawbench.urllib.request,'urlopen',side_effect=AssertionError('no duplicate')):
            with self.assertRaisesRegex(ValueError,'outcome unknown'):
                lawbench.infer(self.a)

    def test_two_model_folders_cannot_be_aggregated(self):
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())):
            lawbench.infer(self.a)
        (self.a.run_dir/'predictions'/'another-model').mkdir()
        with patch.object(lawbench.subprocess,'run',side_effect=AssertionError('official scorer must not run')):
            with self.assertRaisesRegex(ValueError,'mixed model'):
                lawbench.evaluate(self.a)

    def evaluate_without_running_a_scorer(self):
        evaluator=self.folder/'evaluation'/'main.py';evaluator.parent.mkdir(exist_ok=True)
        evaluator.write_text('# isolated evaluator fixture; never executed',encoding='utf-8')
        self.a.python=Path(sys.executable);self.a.out=self.folder/'evaluation-fixture-result.json'
        def fake_evaluator(*args,**kwargs):
            (self.a.run_dir/'official_results.csv').write_text('model,task,score\nfixture,1-2,0\n',encoding='utf-8')
            return SimpleNamespace(returncode=0,stdout='isolated fixture',stderr='')
        with patch.object(lawbench.subprocess,'run',side_effect=fake_evaluator):
            return lawbench.evaluate(self.a)

    def test_two_actual_receipts_are_checked_but_not_a_full_benchmark(self):
        self.a.item_limit=2;self.a.max_items=2
        with patch.object(lawbench.urllib.request,'urlopen',return_value=Response(self.reply())):
            lawbench.infer(self.a)
        result=self.evaluate_without_running_a_scorer()
        self.assertFalse(result['full_official_benchmark'])
        self.assertEqual(result['evaluated_items_per_task'],{'1-2':2})
        self.assertEqual(result['actual_model_calls'],2)
        self.assertEqual(len(result['actual_call_receipts']),2)

    def test_declared_empty_output_fixture_is_not_a_model_score(self):
        lawbench.fixture(self.a)
        result=self.evaluate_without_running_a_scorer()
        self.assertTrue(result['fixture_only'])
        self.assertFalse(result['model_score'])
        self.assertEqual(result['actual_model_calls'],0)
        self.assertFalse(result['full_official_benchmark'])


if __name__=='__main__':
    unittest.main()
