"""Read-only independent coverage checks for an archived complex core run.

Does not call models, alter reports, or convert pending legal/evidence states to
success. The stronger engineering flag requires every prepared initial window
to have completed and every real request to have a matched verified receipt.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acceptance_common import digest, write_json
from acceptance_model_http import assess_report
from zhilv_qiling_v2.long_context import Segment, ReviewWindow, coverage
from zhilv_qiling_v2.models import Clause


def check(prepared, report):
    logic=report.get('stats',{}).get('logic_llm') or {}
    cross=report.get('stats',{}).get('cross_validation') or {}
    calls=logic.get('calls') or []
    windows=[ReviewWindow(row['window_id'],row['kind'],
        tuple(Segment(**segment) for segment in row['segments']),row.get('reason',''),
        tuple(row.get('dependency_ids',[]))) for row in prepared['windows']]
    window_ids=[window.window_id for window in windows]
    actual_ids=[call.get('window_id') for call in calls]
    valid_fields={field.name for field in fields(Clause)}
    clauses=[Clause(**{key:value for key,value in row.items() if key in valid_fields})
             for row in prepared['clauses']]
    done={call.get('window_id') for call in calls if call.get('status')=='ok'}
    independent=coverage(clauses,windows,done)
    receipt_assessment=assess_report(report,0)['model_review_completion']
    deep_calls=[call for call in report.get('stats',{}).get('model_calls',[])
                if call.get('component')=='entailment' and call.get('operation')=='predict_pair'
                and call.get('status')=='succeeded']
    pending=cross.get('pending_feedback_item_ids')
    pending_requests=cross.get('pending_subject_verification_request_ids')
    ledger_present=isinstance(pending,list) and isinstance(pending_requests,list)
    missing=set(window_ids)-set(actual_ids)
    unknown=set(actual_ids)-set(window_ids)
    scheduled_finished=all(window_id in done for window_id in window_ids)
    pending_safe=bool(ledger_present and not ((pending or pending_requests)
        and (cross.get('converged') is True or cross.get('termination')=='stable')))
    checks={'document_text_equals_prepared':report.get('document',{}).get('full_text')==prepared['parsed_text'],
        'initial_plan_ids_unique':len(set(window_ids))==len(window_ids),
        'initial_call_ids_match_plan':not missing and not unknown and len(actual_ids)==len(window_ids),
        'every_prepared_initial_window_completed':scheduled_finished,
        'independent_text_coverage_complete':independent['text_complete'],
        'reported_text_complete_agrees':logic.get('text_complete')==independent['text_complete'],
        'reported_text_coverage_agrees':logic.get('text_coverage')==independent['text_coverage'],
        'reported_reference_status_agrees':logic.get('explicit_reference_complete')==independent['explicit_reference_complete'],
        'all_real_requests_have_verified_success_receipts':receipt_assessment['all_llm_requests_succeeded_with_verified_response_identity'],
        'actual_deep_pair_calls_present':bool(deep_calls),
        'pending_ledger_present_and_no_false_stability':pending_safe}
    return {'status':'passed_with_explicit_pending' if all(checks.values()) else 'failed_strict_checks',
        'strict_engineering_execution_verified':all(checks.values()), 'checks':checks,
        'missing_initial_window_ids':sorted(missing),'unknown_initial_window_ids':sorted(unknown),
        'independent_source_coverage':independent,
        'explicit_reference_scope_complete':independent['explicit_reference_complete'],
        'explicit_reference_pending_is_engineering_failure':False,
        'feedback_completion':{'all_protocol_feedback_complete':receipt_assessment['feedback_verification_complete'],
            'pending_feedback_item_ids':pending, 'pending_subject_verification_request_ids':pending_requests,
            'termination':cross.get('termination'),'converged':cross.get('converged')},
        'effective_model_review_complete':receipt_assessment['effective_complete'],
        'legal_correctness_evaluated':False,
        'meaning':'strict execution proves supplied full text and every initial scheduled window were processed with verified model receipts; reference ambiguity, unscheduled relations and pending feedback remain separate and cannot prove legal correctness'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    directory=args.run_dir.resolve()
    if not directory.is_relative_to(ROOT/'tools/acceptance_results'):
        parser.error('run-dir must remain under tools/acceptance_results')
    output=args.out or directory/'postrun_strict.json'
    if output.exists():
        parser.error('output already exists; preserve prior postrun evidence')
    paths={name:directory/name for name in ('prepared.json','report.json','result.json')}
    contents={name:json.loads(path.read_text(encoding='utf-8')) for name,path in paths.items()}
    result=check(contents['prepared.json'],contents['report.json'])
    result['input_sha256']={name:digest(path.read_bytes()) for name,path in paths.items()}
    result['tool_sha256']=digest(Path(__file__).read_bytes())
    result['archived_http_roundtrip_reported']=contents['result.json'].get('http_archive_roundtrip_verified')
    result['checks']['archive_roundtrip_verified_by_run']=result['archived_http_roundtrip_reported'] is True
    result['checks']['saved_report_identity_matches_run']=(contents['report.json'].get('saved') is True
        and bool(contents['report.json'].get('report_id'))
        and contents['report.json'].get('report_id')==contents['result.json'].get('report_id'))
    result['strict_engineering_execution_verified']=all(result['checks'].values())
    result['status']='passed_with_explicit_pending' if result['strict_engineering_execution_verified'] else 'failed_strict_checks'
    write_json(output,result)
    print(json.dumps({'status':result['status'],'evidence':str(output),
        'strict_engineering_execution_verified':result['strict_engineering_execution_verified'],
        'effective_model_review_complete':result['effective_model_review_complete']},ensure_ascii=True))
    return 0 if result['strict_engineering_execution_verified'] else 2


if __name__=='__main__':
    raise SystemExit(main())
