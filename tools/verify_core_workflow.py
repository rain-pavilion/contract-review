"""Real LegalOne + opt-in trained BERT/Mamba HTTP integration, no quality score.

Runs one explicitly synthetic contract against the existing native 8B process,
with an isolated web server/database and no second large-model instance.
"""
from __future__ import annotations
import json
import argparse
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.web-deps'))
from acceptance_common import digest, multipart, provenance, request, require, write_json
from acceptance_model_http import SAMPLE, assess_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'tools/acceptance_results/core_technical_workflow')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--native-thinking', choices=['on','off'], default='off')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Explicit retrieval setting for this integration run; not an accuracy score')
    parser.add_argument('--review-rounds', type=int, default=3,
                        help='Explicit application round bound; unresolved items must remain pending')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if (not directory.is_relative_to(ROOT/'tools/acceptance_results') or args.max_tokens < 1
            or args.top_k < 1 or args.review_rounds < 1):
        parser.error('fresh directory must remain in tools/acceptance_results; numeric limits must be positive')
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists() or (directory / 'workspace.sqlite').exists():
        raise ValueError('requires fresh evidence directory; preserve prior experiments')
    os.environ['ZLQL_ENTAILMENT_MODEL_DIR'] = str(ROOT/'zhilv_qiling_v2/models/bert_mamba_entail')
    os.environ['ZLQL_ENTAILMENT_DEVICE'] = 'cpu'
    os.environ['ZLQL_MODEL_AUDIT_PATH'] = str(directory / 'model_calls.sqlite')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from zhilv_qiling_v2 import api
    from zhilv_qiling_v2.config import Config
    api._SERVER_CONFIG = Config(entailment_backend='bert', enable_llm=True,
        enable_multi_agent=True, llm_endpoint='http://127.0.0.1:8620/v1', llm_model='legalone-8b',
        top_k=args.top_k, review_max_rounds=args.review_rounds,
        llm_timeout=900, llm_max_tokens=args.max_tokens, llm_native_thinking=args.native_thinking, llm_native_reasoning_format='deepseek',
        subject_facts_path=str(directory/'subject_facts.json'),
        store_path=str(directory/'workspace.sqlite'))
    server = api.serve('127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    result = {'status': 'running', 'scope': 'actual_core_integration_not_accuracy_or_throughput',
        'synthetic_input': True, 'accuracy_evaluated': False, 'provenance': provenance(), 'errors': [],
        'validation_configuration': {'top_k':args.top_k, 'review_max_rounds':args.review_rounds,
                                     'max_tokens':args.max_tokens, 'native_thinking':args.native_thinking},
        'http_archive_roundtrip_verified': False}
    try:
        health = require(*request(base, '/api/health', timeout=30)[:2])
        result['runtime'] = health['runtime']
        result['llm_configuration'] = health['llm']
        architecture = health['runtime']['bert_metadata']['loaded_architecture']
        assert architecture['mamba'] and architecture['strict_checkpoint_load'], architecture
        raw, content_type = multipart('SYNTHETIC-core-LegalOne-BERT-Mamba.txt', SAMPLE.encode(),
            {'store':'true', 'llm':'on', 'engine':'multi'})
        print('Starting one actual core HTTP review (LegalOne + trained BERT/Mamba); application feedback retries remain enabled.', flush=True)
        status, report, seconds = request(base, '/api/review', raw=raw, content_type=content_type, timeout=3600)
        require(status, report)
        assert report.get('saved') and report.get('report_id'), report
        archived = require(*request(base, '/api/workspace/reports/'+report['report_id'])[:2])
        assert archived['document'] == report['document'] and archived['stats'] == report['stats']
        result['http_archive_roundtrip_verified'] = True
        calls = report.get('stats', {}).get('model_calls', [])
        deep = [c for c in calls if c.get('operation')=='predict_pair' and c.get('status')=='succeeded']
        result.update(assess_report(report, seconds))
        result.update(report_id=report['report_id'], archived_report='report.json',
            source_sha256=digest(SAMPLE), client_upload_to_archive_response_seconds=seconds,
            actual_deep_pair_calls=len(deep), full_model_calls=calls,
            five_node_trace=report.get('trace'), subject_scope=report.get('subject_checks'),
            statute_checks=report.get('stats', {}).get('statute_checks'))
        write_json(directory/'report.json', report)
        assert deep, 'no actual BERT/Mamba classification call'
        if result['model_review_completion']['effective_complete']:
            result['status'] = 'core_http_pipeline_completed_with_actual_models'
    except Exception as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        write_json(result_path, result)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(json.dumps({'status': result['status'], 'evidence':str(result_path),
        'errors':result['errors']}, ensure_ascii=True))
    return 0 if result.get('model_review_completion',{}).get('effective_complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
