"""Check the running core web configuration without generating a model answer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

from acceptance_common import ROOT, digest, provenance, request, require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'tools/acceptance_results') or output.exists():
        parser.error('requires a fresh evidence path under tools/acceptance_results')
    before = provenance()
    health = require(*request('http://127.0.0.1:8610', '/api/health', timeout=90)[:2])
    models = require(*request('http://127.0.0.1:8620', '/v1/models', timeout=15)[:2])
    llm, runtime = health['llm'], health['runtime']
    metadata = runtime['bert_metadata']
    architecture, checkpoint = metadata['loaded_architecture'], metadata['loaded_checkpoint']
    checks = {
        'llm_enabled':llm.get('enabled') is True,
        'multi_agent_default':health.get('multi_agent', {}).get('default') is True,
        'actual_native_8b_identity':len([item for item in models.get('data', [])
            if item.get('id') == 'legalone-8b' and item.get('meta', {}).get('n_params') == 8195978240]) == 1,
        'output_budget_4096':llm.get('max_output_tokens') == 4096,
        'native_schema_enabled':llm.get('native_json_schema') is True,
        'native_thinking_off':llm.get('native_thinking') == 'off',
        'native_reasoning_boundary':llm.get('native_reasoning_format') == 'deepseek',
        'bert_loaded':runtime.get('active_backend') == 'bert' and runtime.get('bert_loaded') is True,
        'mamba_loaded':architecture.get('mamba') is True,
        'strict_checkpoint_load':architecture.get('strict_checkpoint_load') is True,
        'explicit_new_checkpoint':Path(checkpoint.get('directory', '')).resolve()
            == (ROOT / 'zhilv_qiling_v2/models/bert_mamba_entail').resolve(),
        'weak_supervision_disclosed':checkpoint.get('weak_supervision') is True and checkpoint.get('human_gold') is False,
    }
    pages = []
    for route, filename in (('/', 'webui.html'), ('/annotations', 'annotation.html')):
        with urllib.request.urlopen('http://127.0.0.1:8610' + route, timeout=15) as response:
            content = response.read()
            source = ROOT / 'zhilv_qiling_v2' / filename
            same_text = content.decode('utf-8-sig').replace('\r\n', '\n') == source.read_text(encoding='utf-8-sig')
            page = {'route':route, 'status':response.status, 'content_type':response.headers.get('Content-Type', ''),
                    'sha256':digest(content), 'source_file_sha256':digest(source.read_bytes()),
                    'matches_current_source':same_text, 'comparison':'UTF-8 text, CRLF normalized'}
        checks['page_' + filename] = page['status'] == 200 and page['matches_current_source']
        pages.append(page)
    after = provenance()
    checks['product_sources_unchanged'] = before['product_source_sha256'] == after['product_source_sha256']
    result = {'core_runtime_verified':all(checks.values()), 'scope':'running web health, actual model identity and HTTP page mounting',
        'checks':checks, 'health':health, 'models':models, 'pages':pages,
        'model_generation_requests_by_this_check':0, 'full_contract_review_by_this_check':False,
        'browser_visual_verified':False, 'legal_accuracy_evaluated':False,
        'provenance':before, 'source_sha256_after':after['product_source_sha256']}
    write_json(output, result)
    print(json.dumps({'core_runtime_verified':result['core_runtime_verified'], 'checks':checks,
                      'evidence':str(output)}, ensure_ascii=True))
    return 0 if result['core_runtime_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
