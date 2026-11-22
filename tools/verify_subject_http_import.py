"""Import a real captured evidence bundle through an isolated HTTP API.

No public network or model request is made. Evidence stays unknown. The existing
business registry and website process are never changed by this diagnostic.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.web-deps'))

from acceptance_common import digest, request, require, write_json
import subject_evidence_verify as evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--today', type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    output = args.out_dir.resolve()
    if not output.is_relative_to(ROOT / 'tools/acceptance_results') or output.exists():
        parser.error('a fresh directory under tools/acceptance_results is required')
    output.mkdir(parents=True)
    direct = evidence.exercise_import(args.bundle, output / 'direct', today=args.today)
    records = json.loads((output / 'direct/subject_facts.isolated.json').read_text(encoding='utf-8'))['records']
    if not all(row.get('source_actor_kind') == 'adapter' and row['status'] == 'unknown'
               and row.get('independently_verified') is False for row in records):
        raise ValueError('explicit adapter provenance and unknown status are required')

    from zhilv_qiling_v2 import api
    from zhilv_qiling_v2.config import Config

    api._SERVER_CONFIG = Config(enable_llm=False, entailment_backend='rules',
        ner_enable_model=False, subject_facts_path=str(output / 'http_subject_facts.json'),
        store_path=str(output / 'http_workspace.sqlite'))
    server = api.serve('127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        posted = require(*request(base, '/api/workspace/subject-facts',
            data={'records': records}, timeout=60)[:2])
        loaded = require(*request(base, '/api/workspace/subject-facts', timeout=60)[:2])
        assert posted['imported'] == loaded['total'] == len(records)
        assert loaded['degraded'] == []
        assert {row['fact_id'] for row in records} == {row['fact_id'] for row in loaded['records']}
        assert all(row['source_actor_kind'] == 'adapter' and row['status'] == 'unknown'
                   and row['independently_verified'] is False for row in loaded['records'])
        assert posted['independently_verified'] is False
        result = {'status': 'verified_actual_http_import_readback_unknown',
            'bundle_sha256': digest(args.bundle.read_bytes()),
            'tool_sha256': digest(Path(__file__).read_bytes()),
            'product_import_identity_binding_status': direct['status'],
            'http_post_result': posted, 'http_get_result': loaded,
            'same_fact_ids': True, 'current_fact_status': 'unknown',
            'network_source_reauthenticated': False, 'independently_verified': False,
            'isolated_registry': str(output / 'http_subject_facts.json'),
            'business_registry_modified': False, 'model_calls': 0,
            'implementation_sha256': {name: digest((ROOT / name).read_bytes()) for name in
                ('zhilv_qiling_v2/subject_facts.py', 'zhilv_qiling_v2/api.py',
                 'tools/subject_evidence_verify.py')}}
        write_json(output / 'http_verification.json', result)
        # A usable JSON sample for the website, carrying the actual captured
        # evidence reference and its explicit pending status, not a fake approval.
        write_json(output / 'website-import-unknown.json', {'records': loaded['records']})
        print(json.dumps({'status': result['status'], 'evidence': str(output / 'http_verification.json')},
                         ensure_ascii=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
