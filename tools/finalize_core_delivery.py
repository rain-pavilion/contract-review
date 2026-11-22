"""Summarize explicitly supplied core evidence without erasing earlier failures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from acceptance_common import ROOT, digest, provenance, write_json
from acceptance_deep_http import SAMPLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statute', type=Path, required=True)
    parser.add_argument('--subject', type=Path, required=True)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--current-deep-http', type=Path, required=True)
    parser.add_argument('--current-deep-readback', type=Path, required=True)
    args = parser.parse_args()
    evidence_root = ROOT / 'tools/acceptance_results'

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(evidence_root):
            raise ValueError('evidence must remain under tools/acceptance_results')
        value = json.loads(path.read_text(encoding='utf-8'))
        return path.relative_to(evidence_root).as_posix(), value

    statute_path, statute = read(args.statute)
    subject_path, subject = read(args.subject)
    regression_path, regression = read(args.regression)
    runtime_path, runtime = read(args.runtime)
    deep_path, deep = read(args.current_deep_http)
    readback_path, readback = read(args.current_deep_readback)
    archive_path, archive = read(evidence_root / readback['archive_evidence'])
    pipeline_path, pipeline = read(evidence_root / 'core_technical_workflow_subject_bound_v2/result.json')
    training_path, training = read(evidence_root / 'mamba_entail_training/run.json')
    annotation_path, annotation = read(evidence_root / 'annotation_pack.json')
    current = provenance()
    deep_sources = deep.get('provenance', {}).get('product_source_sha256', {})
    deep_sources_match = bool(deep_sources) and all(value == current['product_source_sha256'].get(name)
        for name, value in deep_sources.items())
    deep_calls = archive.get('stats', {}).get('model_calls', [])
    deep_count = sum(call.get('component') == 'entailment' and call.get('operation') == 'predict_pair'
        and call.get('status') == 'succeeded' for call in deep_calls)
    loaded = deep.get('runtime', {}).get('bert_metadata', {}).get('loaded_architecture', {})
    deep_ok = (deep.get('status') == 'passed' and deep.get('errors') == []
        and deep.get('llm_requested') is False and readback.get('status') == 'passed'
        and readback.get('source_evidence') == deep_path
        and readback.get('source_evidence_sha256') == digest(args.current_deep_http.read_bytes())
        and readback.get('archive_evidence_sha256') == digest((evidence_root / archive_path).read_bytes())
        and deep.get('report_id') == readback.get('report_id') == archive.get('report_id')
        and deep.get('text_sha256') == readback.get('raw_input_sha256') == digest(SAMPLE)
        and SAMPLE.endswith('\n') and archive.get('document', {}).get('full_text') == SAMPLE[:-1]
        and readback.get('stored_text_sha256') == digest(SAMPLE[:-1])
        and deep_calls == deep.get('actual_model_calls') and deep_count > 0
        and deep_count == readback.get('actual_deep_pair_calls')
        and not any(call.get('component') == 'llm' and call.get('status') == 'started' for call in deep_calls)
        and loaded.get('mamba') is True and loaded.get('strict_checkpoint_load') is True)
    relevant = ('nodes.py', 'llm_client.py', 'config.py', 'model_audit.py')
    feedback_sources_match = all(all(item.get('provenance', {}).get('product_source_sha256', {}).get(name)
        == current['product_source_sha256'].get(name) for name in relevant) for item in (statute, subject))
    accepted = ('complete_bound_batch_returned', 'bound_records_returned_pending')
    protocol_ok = all(item.get('status') in accepted
        and item.get('protocol_bindings_and_quotes_valid') is True
        and item.get('native_schema_sha256_verified') is True
        and item.get('single_call_audit_verified') is True
        and item.get('pending_accounting_consistent') is True for item in (statute, subject))
    regression_ok = (regression.get('unit', {}).get('success') is True
        and regression.get('guards', {}).get('success') is True
        and regression.get('source_unchanged_during_regression') is True
        and regression.get('success') is True)
    regression_sources = regression.get('source_sha256_end', {})
    regression_sources_match = bool(regression_sources) and all((ROOT / name).is_file()
        and digest((ROOT / name).read_bytes()) == value for name, value in regression_sources.items())
    target = evidence_root / 'final_acceptance_summary.json'
    history = None
    if target.exists():
        original = target.read_bytes()
        historical = evidence_root / ('historical_final_acceptance_summary_' + digest(original)[:16] + '.json')
        if historical.exists():
            if historical.read_bytes() != original:
                raise ValueError('historical evidence collision; preserve existing files')
        else:
            historical.write_bytes(original)
        history = {'evidence':historical.name, 'sha256':digest(original), 'bytes_preserved_exactly':True}
    summary = {
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'status':('core_implemented_with_actual_models_and_manual_evidence_dependencies'
                  if protocol_ok and feedback_sources_match and regression_ok and regression_sources_match
                  and runtime.get('core_runtime_verified') is True and deep_ok and deep_sources_match
                  else 'core_implemented_with_explicit_pending_validation'),
        'priority':'core_technology_first_per_user; efficiency_and_full_benchmarks_deferred',
        'web_url':'http://127.0.0.1:8610', 'launcher':'start-core-web.cmd',
        'provenance':current, 'previous_summary_preserved':history,
        'current_product_hashes_match_final_http_measurement':deep_sources_match,
        'current_llm_enabled_full_http_revalidated':False,
        'http_proof_scope':'Current LLM-disabled real Mamba HTTP archive + current targeted real 8B feedback replays; latest full LLM-enabled HTTP review not rerun',
        'current_deep_http':{'evidence':deep_path, 'readback_evidence':readback_path,
            'archive_evidence':archive_path, 'status':deep.get('status'),
            'verified':deep_ok, 'sources_match_current':deep_sources_match,
            'actual_deep_pair_calls':deep_count, 'llm_requested':False,
            'archive_text_verification':'exact SAMPLE[:-1]; only one terminal LF removed',
            'scope':'synthetic multipart upload, multi-agent review, archive and readback with actual Mamba; LLM explicitly off; not legal accuracy or performance acceptance'},
        'archive_pipeline':{'evidence':pipeline_path, 'status':pipeline['status'],
            'http_archive_roundtrip_verified':pipeline.get('http_archive_roundtrip_verified'),
            'actual_deep_pair_calls':pipeline.get('actual_deep_pair_calls'),
            'effective_complete':pipeline.get('model_review_completion', {}).get('effective_complete'),
            'historical_feedback_failures_preserved':True},
        'actual_mamba_training':{'evidence':training_path, 'status':training['status'],
            'completed_optimizer_steps':training['completed_optimizer_steps'],
            'weak_supervision':True, 'human_gold':False,
            'scope':'actual selective-state-space parameter training, strict reload and real predictions; not sufficient legal training'},
        'feedback_replays':[],
        'regression':{'evidence':regression_path, 'unit':regression.get('unit'),
            'guards':regression.get('guards'), 'source_unchanged':regression.get('source_unchanged_during_regression'),
            'sources_match_current':regression_sources_match,
            'scope':'offline unit and tool guards; optional pretrained caches isolated; no legal-quality score'},
        'runtime':{'evidence':runtime_path, 'core_runtime_verified':runtime.get('core_runtime_verified')},
        'annotation_package':{'evidence':annotation_path, 'records':annotation['records'],
            'status':annotation['package_status'], 'human_gold_ready':annotation['gold_ready']},
        'human_gold_ready':False, 'legal_accuracy_accepted':False,
        'current_annotation_page_browser_visual_verified':False,
        'remaining_data_dependencies':['independent human annotation/adjudication',
            'real subject evidence or a business fact provider; import entry is implemented',
            'independent legal-quality validation of the weakly supervised model'],
        'deferred_by_user':['full LawBench and ProvBench model scores', 'throughput and storage/scheduler scale claims',
            'performance targets and full timing acceptance'],
        'failed_history_not_overwritten':['core_technical_workflow', 'core_technical_workflow_bound_v1',
            'core_technical_workflow_subject_bound_v2', 'feedback_schema_real_statute_v1', 'feedback_schema_real_statute_v2',
            'feedback_schema_real_statute_v3_diagnostic'],
    }
    for path, item in ((statute_path, statute), (subject_path, subject)):
        hashes = item.get('provenance', {}).get('product_source_sha256', {})
        summary['feedback_replays'].append({'evidence':path, 'kind':item['feedback_kind'], 'status':item['status'],
            'protocol_bindings_and_quotes_valid':item.get('protocol_bindings_and_quotes_valid'),
            'pending_ids':item.get('feedback', {}).get('uncompleted_feedback_item_ids', []),
            'clause_quote_modes':sorted({str(call.get('clause_quote_mode', 'unspecified'))
                for call in item.get('feedback', {}).get('calls', [])}),
            'feedback_component_sources_match_current':all(hashes.get(name) == current['product_source_sha256'].get(name)
                                                         for name in relevant),
            'full_http_revalidated':False, 'legal_truth_independently_evaluated':False})
    write_json(target, summary)
    print(json.dumps({'status':summary['status'], 'evidence':str(target),
                      'protocol_replays_ok':protocol_ok, 'regression_ok':regression_ok,
                      'current_deep_http_verified':deep_ok, 'current_deep_sources_match':deep_sources_match}, ensure_ascii=True))


if __name__ == '__main__':
    main()
