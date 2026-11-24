"""Supervisor decisions must change execution without changing evidence gates."""
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import agents_multi as M, agents_multi_run as R, nodes as N, pipeline
from zhilv_qiling_v2.config import Config, WORKSPACE
from zhilv_qiling_v2.engine import ReviewEngine
from zhilv_qiling_v2.models import FinalRisk


def contract_fixture():
    source = WORKSPACE / 'samples' / 'tech-development-contract.txt'
    config = Config()
    return pipeline.build_contract(pipeline.prepare(source.name, source.read_bytes(), config=config), config)


class PlannedSupervisor:
    manual = False

    def __init__(self, config):
        self.config = config
        self.decisions = []

    def decide(self, stage, context):
        decision = {'stage': stage, 'source': 'llm', 'action': 'manual' if self.manual and stage == 'feedback' else 'continue',
                    'reason': 'offline validated plan', 'assignments': [], 'clause_ids': []}
        if stage == 'initial':
            decision['assignments'] = [{'agent': 'logic', 'model_id': 'cloud_legal', 'reason': 'legal reasoning'}]
        if stage == 'review' and not self.manual:
            decision.update(action='review', clause_ids=context['clause_ids'][:1],
                            assignments=[{'agent': 'reviewer', 'model_id': 'cloud_legal', 'reason': 'independent review'}])
        self.decisions.append(decision)
        return decision

    def config_for(self, agent, model_id):
        return replace(self.config, enable_llm=self.config.enable_llm, llm_endpoint='https://offline.example/v1',
                       llm_model='selected-legal', llm_api_key='offline-route-key', llm_native_json_schema=False,
                       llm_native_context_preflight=False)

    def snapshot(self):
        return {'enabled': True, 'mode': 'llm', 'calls': len(self.decisions), 'decisions': self.decisions,
                'model_pool': [{'id': 'cloud_legal', 'model': 'selected-legal', 'agents': ['logic', 'reviewer']}], 'degraded': []}


class SupervisorIntegrationTest(unittest.TestCase):
    def test_peer_analysis_branches_receive_independent_tasks_and_no_ner_to_logic_message(self):
        config = Config(enable_llm=False)
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        R._handle_intake(ctx)
        R._handle_orchestrator(ctx)
        for agent in ('ner', 'logic'):
            self.assertEqual(len(ctx.bus.inbox(agent, kinds=['task.assign'])), 1)
        R._handle_ner(ctx)
        self.assertEqual(ctx.bus.inbox('logic', kinds=['elements.extracted']), [])
        self.assertTrue(ctx.bus.inbox('subject', kinds=['elements.extracted']))
        self.assertTrue(ctx.bus.inbox('judge', kinds=['elements.extracted']))

    def test_logic_results_do_not_require_ner_branch_to_execute_first(self):
        results = []
        for ner_first in (False, True):
            ctx = M.AgentContext(contract=contract_fixture(), config=Config(enable_llm=False), bus=M.MessageBus())
            R._handle_intake(ctx)
            R._handle_orchestrator(ctx)
            if ner_first:
                R._handle_ner(ctx)
            R._handle_logic(ctx)
            results.append(sorted((item.code, tuple(item.clause_ids), item.message)
                                  for item in ctx.shared['logic'].candidates))
        self.assertTrue(results[0])
        self.assertEqual(results[0], results[1])

    def test_review_executor_exception_keeps_assignment_pending(self):
        config = Config(enable_llm=True)
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        ctx.shared['clauses'] = list(ctx.contract.document.clauses)
        ctx.shared['supervisor'] = PlannedSupervisor(config)
        ctx.emit('orchestrator', 'reviewer', 'review.assign', {'model_id': 'cloud_legal',
            'assignment_id': 'raised-review', 'clause_ids': [ctx.contract.document.clauses[0].clause_id]})
        with patch.object(N, 'node_logic', side_effect=RuntimeError('offline executor failure')):
            R.build_agents(config)['reviewer'].run(ctx)
        assert 'raised-review' in ctx.shared.get('review_pending', {})
        assert 'raised-review' not in ctx.shared.get('review_completed', set())

    def test_provider_error_does_not_expose_selected_route_key_in_report(self):
        secret = 'SYNTHETIC_ROUTE_SECRET'
        config = Config(enable_llm=True, llm_endpoint='https://offline.example/v1',
                        llm_model='offline-expert', llm_api_key=secret, llm_route_explicit=True)
        contract = contract_fixture()
        with patch.object(N.L, 'request_json', side_effect=N.L.LLMClientError('HTTP 401 echo ' + secret)):
            outcome = N.node_logic(contract, config, clauses=contract.document.clauses[:1])
        assert secret not in json.dumps(outcome.llm)
        assert secret not in json.dumps(outcome.degraded)

    def test_zero_risk_report_preserves_supervisor_and_expert_manual_obligations(self):
        from zhilv_qiling_v2.review_explanations import build_review_explanations
        payload = {'contract_id': 'manual-contract', 'risks': [], 'stats': {'supervisor': {
            'manual_review_requested': True, 'decisions': [{'action': 'manual', 'reason': '授权来源不足'}],
            'pending_reviews': [{'model_id': 'expert', 'clause_ids': ['c-1'], 'attempts': 2}]}}}
        result = build_review_explanations(payload)
        assert result['summary']['manual_required'] == 2
        assert {item['kind'] for item in result['items']} == {'supervisor_manual_review', 'expert_review'}
        assert all(item['legal_truth_established'] is False for item in result['items'])

    def test_switching_model_does_not_reuse_old_model_feedback_as_execution(self):
        config = Config(enable_llm=True)
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        ctx.shared.update(supervisor=PlannedSupervisor(config), model_assignments={'logic': 'cloud_legal'},
                          logic_executed_route='default', processed_feedback={'same-feedback'},
                          logic=N.NodeOutcome(node='node2_logic'),
                          incremental_dispatch=SimpleNamespace(plan=lambda *args: {
                              'reused_work_item_ids': ['old-model-work'], 'dispatch_work_item_ids': []}))
        cid = ctx.contract.document.clauses[0].clause_id
        observed = []
        def execute(contract, selected, **kwargs):
            observed.append(selected.llm_model)
            return N.NodeOutcome(node='node2_logic', llm={'complete': True, 'feedback_complete': True})
        with patch.object(R, '_feedback_request', return_value=('same-feedback', [], {}, {cid}, [], [])), \
             patch.object(R, '_review_work_state', return_value=([], {})), patch.object(N, 'node_logic', execute):
            R._handle_logic(ctx)
        assert observed == ['selected-legal']
        assert ctx.shared['logic'].payload['feedback_run_bindings'][0]['model_id'] == 'cloud_legal'

    def test_manual_request_puts_existing_risk_in_review_queue(self):
        ctx = M.AgentContext(contract=contract_fixture(), config=Config(), bus=M.MessageBus())
        ctx.shared['supervisor_manual'] = True
        risk = FinalRisk(risk_id='supported-risk', code='fee_mismatch', category='履约风险',
                         severity='medium', message='金额需核对', review_required=False)
        with patch.object(R.A, 'fuse', return_value=[risk]), \
             patch.object(N, 'node_judge', return_value=N.NodeOutcome(node='node5_judge', payload={'score': 90})):
            R._handle_judge(ctx)
        assert ctx.shared['fused'][0].review_required is True
        assert '主脑' in ctx.shared['fused'][0].rationale

    def test_explicit_route_never_inherits_main_environment_credentials(self):
        config = Config(llm_endpoint='https://expert.example/v1', llm_model='expert', llm_api_key='')
        config.llm_route_explicit = True
        with patch.dict(os.environ, {'ZLQL_LLM_API_KEY': 'unrelated-main-secret'}):
            settings = N.llm_settings(config)
        assert settings.api_key == ''
        assert settings.endpoint == 'https://expert.example/v1'

    def test_incomplete_review_remains_pending_and_can_be_retried(self):
        config = Config(enable_llm=True)
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        ctx.shared['clauses'] = list(ctx.contract.document.clauses)
        ctx.shared['supervisor'] = PlannedSupervisor(config)
        ctx.emit('orchestrator', 'reviewer', 'review.assign', {
            'model_id': 'cloud_legal', 'clause_ids': [ctx.contract.document.clauses[0].clause_id],
            'assignment_id': 'retry-review'})
        partial = N.NodeOutcome(node='node2_logic', llm={'complete': False}, degraded=['llm_timeout'])
        completed = N.NodeOutcome(node='node2_logic', llm={'complete': True})
        with patch.object(N, 'node_logic', side_effect=[partial, completed]):
            R._handle_reviewer(ctx)
            assert 'retry-review' not in ctx.shared.get('review_completed', set())
            assert ctx.bus.inbox('orchestrator', kinds=['review.pending'])
            R._handle_reviewer(ctx)
        assert 'retry-review' in ctx.shared['review_completed']

    def test_initial_plan_routes_the_actual_logic_executor(self):
        config = Config(enable_llm=True)
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        ctx.shared['clauses'] = list(ctx.contract.document.clauses)
        observed = []

        def execute(contract, chosen, **kwargs):
            observed.append((chosen.llm_endpoint, chosen.llm_model, chosen.llm_api_key))
            return N.NodeOutcome(node='node2_logic')

        with patch.object(R, 'Supervisor', PlannedSupervisor, create=True), patch.object(N, 'node_logic', execute):
            R._handle_orchestrator(ctx)
            R._handle_logic(ctx)
        assert observed == [('https://offline.example/v1', 'selected-legal', 'offline-route-key')]
        assert ctx.bus.inbox('statute', kinds=['defects.detected'])
        assert 'offline-route-key' not in json.dumps(ctx.bus.dump())


    def test_selected_reviewer_uses_its_route_not_the_main_key(self):
        config = Config(enable_llm=True, llm_api_key='main-offline-key')
        ctx = M.AgentContext(contract=contract_fixture(), config=config, bus=M.MessageBus())
        ctx.shared['clauses'] = list(ctx.contract.document.clauses)
        observed = []

        def execute(contract, chosen, **kwargs):
            observed.append((chosen.llm_model, chosen.llm_api_key, len(kwargs['clauses'])))
            return N.NodeOutcome(node='node2_logic')

        with patch.object(R, 'Supervisor', PlannedSupervisor, create=True), patch.object(N, 'node_logic', execute):
            R._handle_orchestrator(ctx)
            ctx.emit('logic', 'orchestrator', 'defects.detected', {'defects': [
                {'severity': 'high', 'confidence': .9, 'clause_ids': [ctx.contract.document.clauses[0].clause_id]}]})
            R._handle_orchestrator(ctx)
            R._handle_reviewer(ctx)
        assert observed == [('selected-legal', 'offline-route-key', 1)]
        assert 'offline-route-key' not in json.dumps(ctx.bus.dump())


    def test_manual_decision_retains_pending_and_final_business_evaluation(self):
        class ManualSupervisor(PlannedSupervisor):
            manual = True

        config = Config(enable_llm=False, review_max_rounds=3)
        subject = N.NodeOutcome(node='node3_subject', payload={'hint': {}, 'stats': {
            'verification_complete': False, 'pending_verification_request_ids': ['authorization-request-1']}})
        with patch.object(R, 'Supervisor', ManualSupervisor, create=True), \
             patch.object(N, 'node_logic', return_value=N.NodeOutcome(node='node2_logic')), \
             patch.object(N, 'node_subject', return_value=subject), \
             patch.object(N, 'node_statute', return_value=N.NodeOutcome(node='node4_statute', payload={'risk_checks': []})), \
             patch.object(N, 'node_judge', return_value=N.NodeOutcome(node='node5_judge', payload={'score': 100})):
            run = R.run_multi_agent(contract_fixture(), config, retriever=object())
        cross = run.diagnostics['cross_validation']
        assert cross['termination'] == 'supervisor_manual_review'
        assert cross['converged'] is False
        assert cross['pending_subject_verification_request_ids'] == ['authorization-request-1']
        assert run.order[-1] == 'judge'
        assert all(role in run.order for role in ('ner', 'logic', 'subject', 'statute'))


    def test_report_exposes_six_core_nodes_and_rule_decision_provenance(self):
        from zhilv_qiling_v2 import entailment
        with patch.object(entailment, 'backend', return_value='rules'):
            report = ReviewEngine(Config(enable_multi_agent=True)).run(contract_fixture())
        core = report.stats['multi_agent']['core_nodes']
        assert [(node['number'], node['id']) for node in core] == [
            (1, 'orchestrator'), (2, 'ner'), (3, 'logic'), (4, 'subject'), (5, 'statute'), (6, 'judge')]
        assert report.stats['supervisor']['decisions'][0]['source'] == 'rules'
        assert report.stats['nodes']['core_order'][0] == 'orchestrator'
        assert report.stats['nodes']['order'][0] == 'node1_ner'
        self.assertTrue('core_stages' in report.stats['multi_agent'], 'Report must describe peer analysis branches')
        self.assertEqual(report.stats['multi_agent']['core_stages'], [
            ['orchestrator'], ['ner', 'logic'], ['subject'], ['statute'], ['judge']])
        self.assertEqual(report.stats['multi_agent']['execution_mode'], 'sequential')
