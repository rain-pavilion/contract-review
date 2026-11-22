"""Prepare or execute an isolated complex-contract core integration diagnostic.

Preparation is offline and never calls a model. Execution explicitly uses the
already running loopback LegalOne service; this tool never starts an 8B model.
The default contract and subject records are synthetic test fixtures. An
explicit local public-contract directory can instead supply all page text.
Neither input is independent legal gold. Pending evidence is an expected state, not a failed fact
that may be silently promoted into a legal conclusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.web-deps'))
from acceptance_common import digest, multipart, request, require, write_json, provenance

AS_OF = date(2026, 11, 15)
SUBJECT = '合成同名技术有限公司'
BUYER_ID = 'SYNTHETICBUYER00001'
SELLER_ID = 'SYNTHETICSELLER0001'
TAIL = 'TAIL-COMPLEX-20261115'


def source_provenance():
    """Hash actual sources without launching Git or the optional WMI query."""
    return provenance()


def synthetic_contract() -> str:
    """A reproducible diagnostic with distant anchors, not an accuracy dataset."""
    descriptions = [
        ('第一条 服务范围', '乙方提供软件维护、数据迁移和交付文档服务，合同价款为人民币100000元。'),
        ('第二条 付款安排', '甲方在合同生效后支付人民币30000元；依据第九条完成最终验收后支付尾款人民币70000元。'),
        ('第三条 交付安排', '全部系统、源代码和交付文档应在2026年12月1日前交付并经双方签字确认。'),
        ('第四条 文档保管', '维护记录应按模块编号归档，双方各保留一份；记录包含时间、操作人员及结果。'),
        ('第五条 信息保密', '双方应对履约获取的技术资料和业务数据承担保密义务；经书面同意方可向第三方披露。'),
        ('第六条 变更记录', '需求变更应由双方签署书面文件，记载版本、影响范围和处理安排，原记录不得直接覆盖。'),
        ('第七条 补充付款约定', '甲方还应在系统上线前支付尾款人民币80000元，该笔与预付款共同构成本合同全部服务费。'),
        ('第八条 服务期限', '乙方完成源代码和交付文档的最终交付日期为2027年1月15日，服务应在此后结束。'),
        ('第九条 验收安排', '验收以第二条所列付款安排为前提，甲方支付全部合同价款后，乙方才启动最终验收。'),
        ('第十条 责任及附件', '甲方无论因何原因违约，均不承担任何赔偿责任。附录末尾标识：' + TAIL + '。'),
    ]
    lines = ['合成诊断合同：复杂跨章节技术服务合同（非真实交易、非人工法律金标）',
             '甲方：' + SUBJECT, '甲方统一社会信用代码：' + BUYER_ID,
             '乙方：' + SUBJECT, '乙方统一社会信用代码：' + SELLER_ID]
    for index, (heading, central) in enumerate(descriptions, 1):
        # Factual administrative detail creates distance between the anchors;
        # it does not contain instructions to the model or fake legal citations.
        notes = ''.join(f'事项{index}-{j}的材料包括版本号、经办人、收到日期和完整性记录；'
                        '材料应放入对应目录，双方通过书面记录确认收到情况。' for j in range(1, 6))
        lines.extend((heading, central + notes))
    return '\n'.join(lines) + '\n'


def synthetic_facts(contract_id: str, *, verified_at: str | None = None):
    verified = verified_at or min(date.today(), AS_OF).isoformat()
    base = {'subject_name': SUBJECT, 'contract_id': contract_id,
            'source': {'url': 'https://example.invalid/synthetic-complex-core',
                       'publisher': 'SYNTHETIC diagnostic fixture; not a real registry',
                       'reviewer': 'SYNTHETIC fixture builder',
                       'verification_method': 'Synthetic test attestation only; never fetched or claimed official',
                       'verified_at': verified}}
    return [{**base, 'subject_id': BUYER_ID, 'fact_type': 'registration', 'status': 'active',
             'evidence_ref': 'SYNTHETIC buyer registration fixture'},
            {**base, 'subject_id': SELLER_ID, 'fact_type': 'registration', 'status': 'active',
             'evidence_ref': 'SYNTHETIC seller registration fixture'},
            {**base, 'subject_id': BUYER_ID, 'fact_type': 'license', 'status': 'valid',
             'valid_until': '2025-12-31', 'evidence_ref': 'SYNTHETIC deliberately expired license'},
            {**base, 'subject_id': SELLER_ID, 'fact_type': 'credit', 'status': 'unknown',
             'evidence_ref': 'SYNTHETIC unknown credit; no adverse record supplied'}]


def public_contract_source(directory: Path):
    """Read a pre-existing public copy without claiming a new source fetch."""
    source = directory.resolve()
    allowed = (ROOT/'data/gov_contracts').resolve()
    if not source.is_relative_to(allowed) or not source.is_dir():
        raise ValueError('public contract source must be an existing data/gov_contracts directory')
    metadata = json.loads((source/'meta.json').read_text(encoding='utf-8'))
    text_files = sorted((source/'text').glob('page_*.txt'))
    if not text_files or len(text_files) != metadata.get('page_count'):
        raise ValueError('public source page text is incomplete')
    files = [source/'meta.json', *text_files, *sorted((source/'pages').glob('*'))]
    texts = [path.read_text(encoding='utf-8') for path in text_files]
    raw = '\n'.join(texts)
    if not raw.strip():
        raise ValueError('public source text is empty')
    record = {'mode':'existing_public_procurement_local_copy',
        'source_directory':str(source.relative_to(ROOT)), 'title':metadata.get('title'),
        'detail_url':metadata.get('detail_url'), 'pdf_url':metadata.get('pdf_url'),
        'original_crawled_at':metadata.get('crawled_at'),
        'source_refetched_this_run':False,
        'source_availability_note':'existing local public procurement copy; source site was not fetched in this run',
        'page_text_count':len(text_files), 'no_page_text_omitted':True,
        'pdf_file_in_local_copy':any(path.suffix.lower()=='.pdf' for path in files),
        'files':[{'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
                  'sha256':digest(path.read_bytes())} for path in files if path.is_file()],
        'page_text_sha256':[digest(text) for text in texts]}
    return raw, f'PUBLIC-{source.name}-all-pages.txt', record


def config_for(directory: Path, *, checkpoint: Path, chunk_chars: int = 1400,
               cross_windows: int = 4, rounds: int = 2, max_tokens: int = 4096,
               top_k: int = 2, execute: bool = False, model_base: str = 'http://127.0.0.1:8620'):
    from zhilv_qiling_v2.config import Config
    return Config(entailment_backend='bert' if execute else 'rules', enable_llm=execute,
        enable_multi_agent=True, llm_endpoint=model_base.rstrip('/') + '/v1',
        llm_model='legalone-8b', top_k=top_k, review_max_rounds=rounds,
        llm_timeout=900, llm_max_tokens=max_tokens, llm_native_thinking='off',
        llm_native_reasoning_format='deepseek', llm_native_json_schema=True,
        llm_chunk_chars=chunk_chars, llm_cross_clause_pairs=cross_windows,
        ner_enable_model=execute, ner_use_bert=execute, llm_max_calls=0,
        subject_facts_path=str(directory/'subject_facts.json'),
        store_path=str(directory/'workspace.sqlite'))


def prepare_case(directory: Path, *, checkpoint: Path, chunk_chars=1400, cross_windows=4,
                 rounds=2, max_tokens=4096, top_k=2, public_contract_dir: Path | None = None):
    from zhilv_qiling_v2.pipeline import prepare, build_contract
    from zhilv_qiling_v2.long_context import build_review_plan, coverage
    from zhilv_qiling_v2.subject_facts import validate_fact, resolve_subject
    cfg = config_for(directory, checkpoint=checkpoint, chunk_chars=chunk_chars,
        cross_windows=cross_windows, rounds=rounds, max_tokens=max_tokens, top_k=top_k)
    raw, source_name, source_manifest = (public_contract_source(public_contract_dir) if public_contract_dir
        else (synthetic_contract(), 'SYNTHETIC-complex-core.txt', {'mode':'synthetic_diagnostic_not_real_transaction'}))
    document = prepare(source_name, raw.encode(), config=cfg)
    contract = build_contract(document, cfg)
    windows, plan = build_review_plan(document.clauses, chunk_chars=chunk_chars,
        overlap=cfg.llm_chunk_overlap, cross_pairs=cross_windows)
    planned_coverage = coverage(document.clauses, windows, {window.window_id for window in windows})
    assert planned_coverage['text_complete']
    assert all(segment.text == next(c for c in document.clauses if c.clause_id == segment.clause_id).text[
        segment.start:segment.end] for window in windows for segment in window.segments)
    tail = document.full_text[-40:]
    assert tail and any(tail in segment.text for window in windows if window.kind == 'text' for segment in window.segments)
    if not public_contract_dir:
        assert TAIL in document.full_text
        assert plan['explicit_reference_dependencies'], 'diagnostic reference anchors were not parsed'
    # Prove that the exact same input does not acquire false reference coverage
    # when the cross-window budget is intentionally exhausted.
    limited, limited_meta = build_review_plan(document.clauses, chunk_chars=chunk_chars,
        overlap=cfg.llm_chunk_overlap, cross_pairs=0)
    limited_cov = coverage(document.clauses, limited, {window.window_id for window in limited})
    assert limited_cov['text_complete']
    if plan['explicit_reference_dependencies']:
        assert not limited_cov['explicit_reference_complete']
        assert limited_meta['explicit_reference_pending_checks'] > 0
    facts, identities = [], {}
    if not public_contract_dir:
        facts = [validate_fact(record) for record in synthetic_facts(contract.contract_id)]
        buyer = resolve_subject(SUBJECT, facts, contract_id=contract.contract_id,
                                subject_ids=[BUYER_ID], today=AS_OF)
        seller = resolve_subject(SUBJECT, facts, contract_id=contract.contract_id,
                                 subject_ids=[SELLER_ID], today=AS_OF)
        assert buyer['checks']['license'] == 'unknown'
        assert any('expired' in row.get('review_reasons', []) for row in buyer['records']
                   if row['subject_id'] == BUYER_ID and row['fact_type'] == 'license')
        assert all(row['effective_status'] == 'unknown' for row in buyer['records'] if row['subject_id'] == SELLER_ID)
        assert all(row['effective_status'] == 'unknown' for row in seller['records'] if row['subject_id'] == BUYER_ID)
        assert seller['checks']['credit'] == 'unknown'
        identities = {'buyer':buyer, 'seller':seller}
    # This bound is deliberately a conservative resource estimate, not a
    # promise that candidate-dependent feedback calls equal this number.
    initial = len(windows)
    return {'status': 'prepared_offline', 'synthetic_input': not bool(public_contract_dir),
        'legal_quality_evaluated': False, 'no_model_calls': True,
        'source_manifest':source_manifest,
        'source_name': document.source_name, 'contract_id': contract.contract_id,
        'raw_text': raw, 'raw_text_sha256': digest(raw),
        'parsed_text':document.full_text, 'tail_text':tail,
        'parsed_text_sha256': digest(document.full_text), 'text_chars': len(document.full_text),
        'configuration': {'chunk_chars': chunk_chars, 'cross_windows': cross_windows,
            'rounds': rounds, 'max_tokens': max_tokens, 'top_k': top_k, 'checkpoint': str(checkpoint)},
        'clauses': [clause.to_dict() for clause in document.clauses],
        'windows': [window.as_dict() for window in windows], 'plan': plan,
        'planned_coverage': planned_coverage, 'zero_cross_budget_coverage': limited_cov,
        'subject_records': facts, 'subject_identity_diagnostics': identities,
        'case_expectations': ([
            {'id': 'explicit_reference_cycle', 'kind': 'diagnostic_text_relationship',
             'description': '第二条等待第九条验收，第九条等待全部付款；模型能否检出单独记录，不当人工金标'},
            {'id': 'implicit_payment_conflict', 'kind': 'diagnostic_text_relationship',
             'description': '第二条尾款70000元，第七条尾款80000元；无显式互引'},
            {'id': 'implicit_delivery_conflict', 'kind': 'diagnostic_text_relationship',
             'description': '第三条2026-12-01交付，第八条2027-01-15最终交付；是否可协调需人工'},
            {'id': 'tail_preserved', 'kind': 'exact_text_invariant', 'marker': TAIL},
            {'id': 'same_name_identity_isolation', 'kind': 'protocol_invariant'},
            {'id': 'expired_and_unknown_stay_pending', 'kind': 'protocol_invariant'}] if not public_contract_dir else [
            {'id':'all_public_page_text_supplied', 'kind':'exact_text_invariant'},
            {'id':'tail_preserved', 'kind':'exact_text_invariant'},
            {'id':'missing_external_facts_stay_pending', 'kind':'protocol_invariant'}]),
        'call_budget_estimate': {'exact_initial_logic_windows': initial,
            'feedback_calls_candidate_dependent': True,
            'no_global_call_cap': True,
            'rough_total_range': [initial+6, initial+30],
            'note': 'feedback items and repeated windows depend on actual model and candidate outputs; this is not an execution count or performance guarantee'},
        'provenance': source_provenance(), 'tool_sha256': digest(Path(__file__).read_bytes())}


def assess_complex(report, prepared):
    from acceptance_model_http import assess_report
    # Performance is explicitly not an acceptance criterion for this task.
    basic = assess_report(report, 0)
    for key in ('duration_within_three_minutes', 'meets_three_minute_for_this_sample'):
        basic.pop(key, None)
    logic = report.get('stats', {}).get('logic_llm') or {}
    cross = report.get('stats', {}).get('cross_validation') or {}
    calls = report.get('stats', {}).get('model_calls') or []
    deep = [call for call in calls if call.get('component') == 'entailment'
            and call.get('operation') == 'predict_pair' and call.get('status') == 'succeeded']
    pending = list(cross.get('pending_feedback_item_ids') or [])
    pending_requests = list(cross.get('pending_subject_verification_request_ids') or [])
    by_code = Counter(str(risk.get('code', '')) for risk in report.get('risks', []))
    invariants = {'source_text_matches_prepared': report.get('document', {}).get('full_text') == prepared['parsed_text'],
        'tail_preserved': prepared['tail_text'] in report.get('document', {}).get('full_text', ''),
        'actual_mamba_pair_events': bool(deep),
        'actual_llm_started': any(call.get('component') == 'llm' and call.get('status') == 'started' for call in calls),
        'verified_actual_llm_receipts': basic['model_review_completion']['all_llm_requests_succeeded_with_verified_response_identity'],
        'pending_does_not_claim_stable': not ((pending or pending_requests) and cross.get('termination') in ('stable', 'converged'))}
    return {'engineering_invariants': invariants, 'engineering_invariants_passed': all(invariants.values()),
        'actual_deep_pair_calls': len(deep), 'pending_feedback_ids': pending,
        'pending_subject_request_ids': pending_requests, 'model_diagnostic': basic,
        'risk_codes': dict(by_code), 'case_expectation_scope': 'diagnostic relationships; no independent human legal gold',
        'legal_quality_evaluated': False}


def monitor_audit(directory, stopped):
    """Write safe live counters while a long synchronous HTTP request runs."""
    previous = None
    while not stopped.wait(10):
        path = directory/'model_calls.sqlite'
        if not path.exists():
            continue
        try:
            with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True, timeout=3) as db:
                rows = db.execute('SELECT component,status,count(*) FROM model_calls GROUP BY component,status').fetchall()
            counts = [{'component':c, 'status':s, 'count':n} for c,s,n in rows]
            progress = {'at_utc':datetime.now(timezone.utc).isoformat(), 'status':'review_running',
                        'safe_audit_counts':counts, 'no_contract_text_or_model_output':True}
            write_json(directory/'progress.json', progress)
            if counts != previous:
                print(json.dumps(progress, ensure_ascii=True), flush=True)
                previous = counts
        except (OSError, sqlite3.Error) as exc:
            print(json.dumps({'audit_monitor_error':type(exc).__name__}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='explicitly allow actual requests to existing loopback 8B')
    parser.add_argument('--model-base', default='http://127.0.0.1:8620')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'zhilv_qiling_v2/models/bert_mamba_entail')
    parser.add_argument('--public-contract-dir', type=Path,
                        help='existing local public procurement copy; source is not refetched by this tool')
    parser.add_argument('--chunk-chars', type=int, default=1400)
    parser.add_argument('--cross-windows', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT/'tools/acceptance_results'):
        parser.error('run-dir must remain under tools/acceptance_results')
    parsed = urlsplit(args.model_base)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
        parser.error('only the existing loopback model service may be used')
    if min(args.rounds, args.top_k, args.max_tokens) < 1 or args.chunk_chars < 200 or args.cross_windows < 0:
        parser.error('invalid numeric budget')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists() or (directory/'workspace.sqlite').exists():
        parser.error('fresh execution directory required; prior evidence is never overwritten')
    prepared = prepare_case(directory, checkpoint=args.checkpoint.resolve(), chunk_chars=args.chunk_chars,
        cross_windows=args.cross_windows, rounds=args.rounds, top_k=args.top_k, max_tokens=args.max_tokens,
        public_contract_dir=args.public_contract_dir)
    prepare_path = directory/'prepared.json'
    if prepare_path.exists():
        prior = json.loads(prepare_path.read_text(encoding='utf-8'))
        if any(prior.get(key) != prepared.get(key) for key in ('raw_text_sha256', 'configuration', 'tool_sha256')):
            parser.error('prepared evidence differs; use a fresh directory')
    else:
        write_json(prepare_path, prepared)
        (directory/'input-contract.txt').write_text(prepared['raw_text'], encoding='utf-8')
    if not args.execute:
        print(json.dumps({'status':'prepared_offline', 'evidence':str(prepare_path),
            'text_chars':prepared['text_chars'], 'plan':prepared['plan'],
            'call_budget_estimate':prepared['call_budget_estimate']}, ensure_ascii=True))
        return 0
    if not (args.checkpoint/'model.pt').is_file():
        parser.error('explicit trained checkpoint is missing')
    model = require(*request(args.model_base, '/v1/models', timeout=30)[:2])
    served = model.get('data') or []
    if len(served) != 1 or served[0].get('id') != 'legalone-8b' or served[0].get('meta', {}).get('n_params') != 8195978240:
        parser.error('existing service is not the expected actual LegalOne-8B')
    os.environ['ZLQL_ENTAILMENT_MODEL_DIR'] = str(args.checkpoint.resolve())
    os.environ['ZLQL_ENTAILMENT_DEVICE'] = 'cpu'
    os.environ['ZLQL_MODEL_AUDIT_PATH'] = str(directory/'model_calls.sqlite')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from zhilv_qiling_v2 import api
    from zhilv_qiling_v2.subject_facts import import_facts
    api._SERVER_CONFIG = config_for(directory, checkpoint=args.checkpoint.resolve(), chunk_chars=args.chunk_chars,
        cross_windows=args.cross_windows, rounds=args.rounds, max_tokens=args.max_tokens,
        top_k=args.top_k, execute=True, model_base=args.model_base)
    if prepared['subject_records']:
        import_facts(prepared['subject_records'], directory/'subject_facts.json')
    server = api.serve('127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    result = {'status':'running', 'synthetic_input':prepared['synthetic_input'], 'legal_quality_evaluated':False,
        'provenance':source_provenance(), 'tool_sha256':digest(Path(__file__).read_bytes()),
        'prepared_sha256':digest(prepare_path.read_bytes()), 'model_service_identity':model,
        'errors':[], 'http_archive_roundtrip_verified':False,
        'scope':'actual model-enabled complete contract HTTP workflow, isolated archive; not accuracy or performance acceptance',
        'source_manifest':prepared['source_manifest']}
    write_json(directory/'result.json', result)
    stopped = threading.Event()
    monitor = threading.Thread(target=monitor_audit, args=(directory, stopped), daemon=True)
    monitor.start()
    try:
        health = require(*request(base, '/api/health', timeout=60)[:2])
        result['runtime'] = health['runtime']
        result['llm_configuration'] = health['llm']
        architecture = health['runtime']['bert_metadata']['loaded_architecture']
        assert architecture['mamba'] and architecture['strict_checkpoint_load']
        raw, content_type = multipart(prepared['source_name'], prepared['raw_text'].encode(),
            {'store':'true', 'llm':'on', 'engine':'multi'})
        print('Starting one actual complex HTTP review; this may retain expected pending evidence.', flush=True)
        status, report, elapsed = request(base, '/api/review', raw=raw, content_type=content_type, timeout=14400)
        require(status, report)
        write_json(directory/'report.json', report)
        assert report.get('saved') and report.get('report_id')
        archived = require(*request(base, '/api/workspace/reports/'+report['report_id'], timeout=60)[:2])
        assert archived == report, 'archive differs from actual returned payload'
        result.update(assess_complex(report, prepared))
        result.update(http_archive_roundtrip_verified=True, report_id=report['report_id'],
            client_upload_to_archive_response_seconds=elapsed,
            measured_duration_is_performance_acceptance=False,
            full_model_calls=report.get('stats', {}).get('model_calls', []))
        result['status'] = ('complex_core_http_verified_with_pending' if result['engineering_invariants_passed']
            else 'complex_core_http_completed_with_invariant_failure')
    except Exception as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        stopped.set()
        monitor.join(timeout=5)
        write_json(directory/'result.json', result)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(json.dumps({'status':result['status'], 'evidence':str(directory/'result.json'),
        'errors':result['errors']}, ensure_ascii=True))
    return 0 if result.get('engineering_invariants_passed') and result['http_archive_roundtrip_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
