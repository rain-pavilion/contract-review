"""One real model-enabled HTTP review, including upload, archive and reread.

Uses a clearly synthetic contract. It never substitutes mocked model output or
turns a successful request into an accuracy or three-minute acceptance claim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from acceptance_common import ROOT, digest, multipart, provenance, request, require, write_json
from acceptance_http import compact_stats

SAMPLE = """合成验收用技术服务合同
甲方：合成验收甲科技有限公司
乙方：合成验收乙技术有限公司
第一条 服务内容
乙方向甲方提供软件维护服务。
第二条 价款
服务费人民币10000元，服务完成并验收后十日内支付。
第三条 期限
服务期限为2026年11月15日至2026年12月15日。
第四条 责任
双方按约履行，违约方依法承担违约责任。争议由有管辖权人民法院处理。
"""


def feedback_run_completion(run):
    """Check explicit feedback coverage, not only human-readable degradation text."""
    pending_keys = ('uncompleted_feedback_item_ids', 'unmatched_feedback_item_ids',
                    'unreviewed_clause_ids')
    pending_known = all(key in run and isinstance(run[key], list) for key in pending_keys)
    pending = {key: run.get(key) for key in pending_keys}
    planned, completed = run.get('planned_calls'), run.get('completed_calls')
    calls_complete = (type(planned) is int and planned >= 0 and
                      type(completed) is int and completed == planned)
    cross_planned, cross_done = run.get('cross_windows'), run.get('cross_windows_completed')
    cross_complete = (type(cross_planned) is int and cross_planned >= 0 and
                      type(cross_done) is int and cross_done == cross_planned)
    complete = bool(run.get('complete') is True and run.get('feedback_complete') is True
                    and calls_complete and cross_complete and pending_known
                    and not any(pending.values()))
    return {'verification_complete': complete,
            'text_complete': run.get('complete') is True,
            'feedback_complete': run.get('feedback_complete') is True,
            'planned_calls': planned, 'completed_calls': completed,
            'scheduled_calls_complete': calls_complete,
            'targeted_cross_clause_plan_complete': cross_complete,
            'pending_id_fields_present': pending_known, 'pending_ids': pending}


def assess_report(report, seconds):
    """Separate elapsed time from effective model completion and legal quality."""
    stats = report.get('stats', {})
    logic = stats.get('logic_llm') or {}
    calls = stats.get('model_calls', [])
    terminal = [c for c in calls if c.get('component') == 'llm'
                and c.get('operation') == 'request_json' and c.get('status') != 'started']
    starts = [c for c in calls if c.get('component') == 'llm'
              and c.get('operation') == 'request_json' and c.get('status') == 'started']
    start_ids = [c.get('details', {}).get('call_id') for c in starts]
    terminal_ids = [c.get('details', {}).get('call_id') for c in terminal]
    successful = [c for c in terminal if c.get('status') == 'succeeded']
    complete_json = [c for c in terminal if c.get('status') in ('succeeded', 'degraded')
                     and c.get('reason') != 'output_truncated']
    degradations = list(stats.get('degraded') or [])
    llm_degradations = [d for d in degradations if 'llm_' in str(d) or 'output_truncated' in str(d)]
    text_complete = bool(logic.get('complete') and logic.get('text_windows', 0) > 0
                         and logic.get('text_coverage') == 1
                         and not logic.get('unreviewed_clause_ids'))
    cross_complete = bool('cross_windows' in logic and 'cross_windows_completed' in logic
                          and logic['cross_windows_completed'] == logic['cross_windows'])
    all_succeeded = bool(starts and None not in start_ids and '' not in start_ids
                         and len(set(start_ids)) == len(start_ids)
                         and len(set(terminal_ids)) == len(terminal_ids)
                         and set(start_ids) == set(terminal_ids)
                         and len(starts) == len(terminal) == len(successful)
                         and all(c.get('details', {}).get('model_identity_verified') for c in terminal)
                         and all(c.get('details', {}).get('actual_model') == 'legalone-8b' for c in terminal))
    feedback_keys = ('feedback_complete', 'planned_calls', 'completed_calls',
                     'feedback_runs', 'feedback_runs_complete', 'uncompleted_feedback_item_ids')
    legacy_feedback_schema = not any(key in logic for key in feedback_keys)
    initial_feedback = None if legacy_feedback_schema else feedback_run_completion(logic)
    runs = logic.get('feedback_runs')
    run_assessments = ([{'run_index': index, **feedback_run_completion(run)}
                        if isinstance(run, dict) else {'run_index':index, 'verification_complete':False,
                                                     'error':'feedback run is not an object'}
                        for index, run in enumerate(runs, 1)] if isinstance(runs, list) else [])
    feedback_complete = (None if legacy_feedback_schema else
                         bool(initial_feedback['verification_complete'] and isinstance(runs, list)
                              and logic.get('feedback_runs_complete') is True
                              and all(run['verification_complete'] for run in run_assessments)))
    cross_validation = stats.get('cross_validation') or {}
    termination = cross_validation.get('termination')
    effective = bool(text_complete and cross_complete and all_succeeded
                     and feedback_complete is True and not llm_degradations)
    duration_ok = isinstance(seconds, (int, float)) and 0 <= seconds <= 180
    return {
        'duration_within_three_minutes': bool(duration_ok),
        'meets_three_minute_for_this_sample': bool(duration_ok and effective),
        'model_review_completion': {
            'initial_logic_text_complete': text_complete,
            'initial_logic_targeted_cross_clause_plan_complete': cross_complete,
            'feedback_verification_complete': feedback_complete,
            'feedback_evidence_status': ('legacy_fields_unavailable' if legacy_feedback_schema
                                         else 'explicit_coverage_metadata'),
            'initial_logic_feedback_verification': initial_feedback,
            'feedback_runs_reported_complete': logic.get('feedback_runs_complete'),
            'feedback_runs': run_assessments,
            'all_llm_requests_succeeded_with_verified_response_identity': all_succeeded,
            'effective_complete': effective,
            'llm_review_degradations': llm_degradations,
            'meaning': 'requires initial text/cross coverage, explicit initial and every feedback-run coverage, matched successful call receipts and no reported LLM context/output degradation; legacy missing feedback proof cannot establish completion; not legal correctness or exhaustive pairwise inference'},
        'cross_validation': {
            'termination': termination, 'converged': cross_validation.get('converged'),
            'round_limit_reached': termination == 'round_limit',
            'rounds_completed': len(cross_validation['rounds']) if isinstance(cross_validation.get('rounds'),list) else None,
            'meaning': 'structural convergence is not legal correctness; round_limit alone does not invalidate measured duration when all scheduled feedback was completed'},
        'all_requested_capabilities_without_degradation': not degradations,
        'reported_degradations': degradations,
        'complete_llm_json_events': len(complete_json),
        'legal_quality_acceptance': 'not_evaluated_no_independent_human_gold',
        'status': ('completed_effective_model_review' if effective else
                   'completed_model_review_with_degradation' if complete_json else
                   'completed_without_successful_llm_event'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8610')
    parser.add_argument('--model-base', default='http://127.0.0.1:8620')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--out', type=Path, default=ROOT/'tools/acceptance_results/http_legalone8b.json')
    parser.add_argument('--existing-report', help='reread an archived report without starting another review')
    parser.add_argument('--prior-result', type=Path, help='original measurement JSON, required with --existing-report')
    args = parser.parse_args()
    result = {'status': 'running', 'accuracy_evaluated': False, 'synthetic_input': True,
              'provenance': provenance(), 'errors': [],
              'timing_scope': 'client immediately before multipart upload through full archived HTTP response; archive reread checked separately'}
    try:
        if args.existing_report:
            if not args.prior_result:
                raise ValueError('--existing-report requires --prior-result; reread latency is not review latency')
            prior = json.loads(args.prior_result.read_text(encoding='utf-8'))
            if prior.get('report_id') != args.existing_report:
                raise ValueError('prior measurement/report ID mismatch')
            seconds = prior.get('client_upload_to_archive_response_seconds')
            if not isinstance(seconds, (int, float)) or seconds < 0:
                raise ValueError('original measured upload-to-response duration is missing')
            status, report, reread_seconds = request(args.base, '/api/workspace/reports/'+args.existing_report)
            require(status, report)
            if (digest(SAMPLE) != prior.get('text_sha256') or
                    report['document']['full_text'].strip().replace('\r\n', '\n') != SAMPLE.strip().replace('\r\n', '\n')):
                raise ValueError('archived text does not match the measured input')
            result = {**prior, **assess_report(report, seconds),
                      'reanalysis_provenance': provenance(),
                      'original_measurement_file': str(args.prior_result.resolve()),
                      'original_measurement_sha256': digest(args.prior_result.read_bytes()),
                      'archive_reread_seconds': reread_seconds,
                      'new_review_or_model_calls': 0,
                      'timing_scope': 'original measured upload-to-archive-response duration retained unchanged; archive reread timed separately'}
            write_json(args.out, result)
            print(json.dumps({'status': result['status'], 'out': str(args.out),
                              'duration_seconds': seconds,
                              'meets_three_minute_for_this_sample': result['meets_three_minute_for_this_sample']}, ensure_ascii=False))
            return 0 if result['model_review_completion']['effective_complete'] else 2
        health = require(*request(args.base, '/api/health', timeout=30)[:2])
        model = require(*request(args.model_base, '/health', timeout=30)[:2])
        model_ready = bool(model.get('ready') or model.get('status') == 'ok')
        if not model.get('data') and model.get('status') == 'ok':
            model = {**model, 'data': require(*request(args.model_base, '/v1/models', timeout=30)[:2]).get('data', [])}
        result['configuration_observed'] = {'llm': health.get('llm'), 'runtime': health.get('runtime'),
                                            'actual_model_service': model}
        assert health.get('llm', {}).get('enabled'), 'LLM is disabled'
        served = (model.get('data') or [model])[0]
        assert (served.get('ready') or model_ready) and served.get('model_id', served.get('id')) == 'legalone-8b', model
        if served.get('owned_by') == 'llamacpp':
            assert served.get('meta', {}).get('n_params') == 8195978240, 'native model parameter identity mismatch'
        raw, content_type = multipart('acceptance-synthetic-LegalOne8B.txt', SAMPLE.encode('utf-8'),
                                        {'store': 'true', 'engine': 'multi', 'llm': 'on'})
        print('Starting one actual LegalOne-8B HTTP contract review; no automatic retry.', flush=True)
        status, report, seconds = request(args.base, '/api/review', raw=raw,
                                         content_type=content_type, timeout=args.timeout)
        require(status, report)
        assert report.get('saved') and report.get('report_id'), report
        archived = require(*request(args.base, '/api/workspace/reports/'+report['report_id'])[:2])
        assert archived['document']['full_text'] == report['document']['full_text']
        assert archived['stats'] == report['stats'], 'archive statistics differ'
        calls = report.get('stats', {}).get('model_calls', [])
        counters = Counter((x.get('component'), x.get('status')) for x in calls)
        result.update(client_upload_to_archive_response_seconds=round(seconds, 6),
                      text_chars=len(SAMPLE), text_sha256=digest(SAMPLE), upload_bytes=len(raw),
                      report_id=report['report_id'], trace_id=report.get('trace_id'),
                      server_timings=report.get('stats', {}).get('timings'),
                      stats=compact_stats(report.get('stats', {})),
                      event_status_counts=[{'component': c, 'status': s, 'count': n}
                                           for (c, s), n in sorted(counters.items())],
                      full_model_calls=calls,
                      scope='one synthetic contract, one request, warm existing model process; not an accuracy benchmark or general throughput claim')
        result.update(assess_report(report, seconds))
    except Exception as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    write_json(args.out, result)
    print(json.dumps({'status': result['status'], 'out': str(args.out), 'errors': result['errors']}, ensure_ascii=False))
    return 0 if result.get('model_review_completion', {}).get('effective_complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
