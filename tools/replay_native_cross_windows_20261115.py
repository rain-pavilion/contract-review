"""Replay three archived cross windows through current _llm_logic, not HTTP review.

Only the planner return is restricted. sys.setprofile observes real transport
frames without replacing transport, prompts, validators, answers, or retries.
"""
from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time
import urllib.request
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/acceptance_results/complex_scoped_20261115_run_v2'
OUT = ROOT / 'tools/acceptance_results/native_cross_window_replay_20261115_v1'
EXPECTED_NODES = '4c396cf63c77633f27121e7f8c398a4553e3f3b0d47878cb271ba411415ad647'
TARGETS = (
    ('amount_2_7', '8c32eab022926421', ('第二条', '第七条')),
    ('delivery_3_8', 'fb7779ee0069bde4', ('第三条', '第八条')),
    ('payment_acceptance_2_9', 'f23aae98529bf78a', ('第二条', '第九条')),
)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
NETWORK = False


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'tensorflow', 'keras',
                'sentence_transformers', 'vllm', 'bitsandbytes', 'llama_cpp'}:
            raise ImportError('Real model imports are forbidden in the replay client')
        return None


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name == '.env' or path.name.startswith('.env.'):
            raise PermissionError('.env access forbidden')
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and not path.is_relative_to(OUT):
            raise PermissionError('Replay writes must remain in its new output directory')
    elif event == 'sqlite3.connect':
        if not Path(args[0]).resolve().is_relative_to(OUT):
            raise PermissionError('Default or external audit database forbidden')
    elif event == 'socket.connect':
        address = args[1]
        if not NETWORK or not isinstance(address, tuple) or address[:2] != ('127.0.0.1', 8620):
            raise PermissionError('Only the explicitly enabled existing local 8B endpoint is allowed')
    elif event == 'socket.getaddrinfo':
        if not NETWORK or args[0] not in ('127.0.0.1', b'127.0.0.1') or args[1] != 8620:
            raise PermissionError('External resolution forbidden')


def sha(value):
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def frozen_hashes():
    paths = list((ROOT / 'zhilv_qiling_v2').rglob('*.py')) + [ROOT / 'zhilv_qiling_v2/webui.html']
    for directory in (SOURCE, ROOT / 'tools/proposals/initial_native_output_schema_20261115',
                      ROOT / 'tools/proposals/native_feedback_atomic_20261115'):
        paths.extend(path for path in directory.rglob('*') if path.is_file())
    return {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in sorted(set(paths))}


def restore():
    from zhilv_qiling_v2 import nodes as N, long_context as LC
    from zhilv_qiling_v2.config import Config
    from zhilv_qiling_v2.models import Clause, Contract, DocumentParse, RISK_CODES
    prepared = json.loads((SOURCE / 'prepared.json').read_text(encoding='utf-8'))
    report = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    assert sha((ROOT / 'zhilv_qiling_v2/nodes.py').read_bytes()) == EXPECTED_NODES
    assert prepared['synthetic_input'] is True
    assert sha(prepared['parsed_text']) == prepared['parsed_text_sha256']
    assert prepared['contract_id'] == report['contract_id']
    assert prepared['parsed_text'] == report['document']['full_text']
    assert prepared['clauses'] == report['document']['clauses']
    clauses = [Clause(**row) for row in prepared['clauses']]
    assert [clause.to_dict() for clause in clauses] == prepared['clauses']
    contract = Contract(prepared['contract_id'], DocumentParse(prepared['source_name'], 'txt',
        prepared['parsed_text'], clauses=clauses), contract_type=report['contract_type'])
    cfg = Config(enable_llm=True, llm_endpoint='http://127.0.0.1:8620/v1', llm_model='legalone-8b',
        llm_max_tokens=4096, llm_timeout=900, llm_native_thinking='off',
        llm_native_reasoning_format='deepseek', llm_native_json_schema=True,
        llm_chunk_chars=1400, llm_chunk_overlap=160, llm_cross_clause_pairs=4, llm_max_calls=1)
    windows, plan = LC.build_review_plan(clauses, chunk_chars=1400, overlap=160, cross_pairs=4)
    assert [window.as_dict() for window in windows] == prepared['windows']
    assert plan == prepared['plan']
    selected = []
    clause_map = {clause.clause_id: clause for clause in clauses}
    for name, window_id, labels in TARGETS:
        matches = [window for window in windows if window.window_id == window_id]
        assert len(matches) == 1
        window = matches[0]
        assert window.kind == 'cross_clause'
        assert tuple(clause_map[s.clause_id].label for s in window.segments) == labels
        assert all(segment.text == clause_map[segment.clause_id].text[segment.start:segment.end]
                   for segment in window.segments)
        schema = N._logic_output_json_schema(window)
        properties = schema['properties']['defects']['items']['properties']
        assert set(properties['code']['enum']) == set(RISK_CODES)
        assert 'enum' not in properties['quote']
        assert set(properties['clause_id']['enum']) == {s.clause_id for s in window.segments}
        selected.append((name, window, schema))
    return N, LC, contract, clauses, cfg, selected


def planner_for(window, clauses):
    def selected_plan(scope, **options):
        assert [c.to_dict() for c in scope] == [c.to_dict() for c in clauses]
        assert options == {'chunk_chars': 1400, 'overlap': 160, 'cross_pairs': 4}
        return [window], {'replay_scope': 'one_archived_cross_window_not_full_http_review',
            'text_windows': 0, 'cross_windows': 1, 'cross_plan_complete': False,
            'cross_scope': 'targeted_replay_only_not_exhaustive', 'chunk_chars': 1400, 'overlap': 160}
    return selected_plan


def selfcheck():
    N, LC, contract, clauses, cfg, selected = restore()
    original = {'logic': N._llm_logic, 'request_json': N.L.request_json, 'complete': N.L._complete}
    checks = []
    for name, window, schema in selected:
        with patch.object(LC, 'build_review_plan', planner_for(window, clauses)), patch.object(N.L, 'available', return_value=False):
            candidates, meta, degraded = N._llm_logic(contract, cfg, clauses)
        assert not candidates and not meta['calls'] and meta['planned_calls'] == 1
        assert meta['planned_windows'] == 1 and meta['completed_calls'] == 0
        assert meta['text_complete'] is False and meta['complete'] is False
        assert meta['feedback_windows'][0]['window_id'] == window.window_id
        checks.append({'name': name, 'window_id': window.window_id,
            'source_sha256': sha(canonical([part.as_dict() for part in window.segments])),
            'schema_sha256': sha(canonical(schema)), 'offline_no_calls': True})
    assert original == {'logic': N._llm_logic, 'request_json': N.L.request_json, 'complete': N.L._complete}
    assert not {'torch', 'transformers'}.intersection(sys.modules)
    return checks


def health():
    result = {}
    for name, path in (('models', '/v1/models'), ('slots', '/slots')):
        with urllib.request.urlopen('http://127.0.0.1:8620' + path, timeout=20) as response:
            result[name] = json.loads(response.read())
    models = [item for item in result['models'].get('data', [])
              if item.get('id') == 'legalone-8b' and item.get('meta', {}).get('n_params') == 8195978240]
    assert len(models) == 1, 'Existing native LegalOne-8B parameter identity not verified'
    assert isinstance(result['slots'], list) and result['slots']
    assert all(slot.get('is_processing') is False for slot in result['slots']), '8B server is busy; no model call sent'
    return result


def observations(raw, candidates, window):
    sources = {segment.clause_id: [] for segment in window.segments}
    for segment in window.segments:
        sources[segment.clause_id].append(segment.text)
    rows = []
    for attempt in raw:
        try:
            content = json.loads(attempt.get('content_text', ''))
        except (TypeError, ValueError):
            continue
        for item in content.get('defects', []) if isinstance(content, dict) else []:
            if not isinstance(item, dict):
                continue
            quotes = [{'clause_id': item.get('clause_id'), 'quote': item.get('quote')}]
            quotes += item.get('related_quotes', []) if isinstance(item.get('related_quotes'), list) else []
            bound = [q for q in quotes if isinstance(q, dict) and isinstance(q.get('quote'), str)
                     and q['quote'] and any(q['quote'] in text for text in sources.get(q.get('clause_id'), []))]
            rows.append({'call_id': attempt['call_id'], 'code': item.get('code'), 'message': item.get('message'),
                'quotes': quotes, 'exact_bound_quotes': bound,
                'both_window_sides_quoted': set(sources).issubset({q['clause_id'] for q in bound}),
                'nonempty_model_explanation': isinstance(item.get('message'), str) and bool(item['message'].strip())})
    return {'raw_model_findings': rows, 'accepted_candidate_count': len(candidates),
        'raw_model_has_both_sides_and_explanation': any(row['both_window_sides_quoted'] and row['nonempty_model_explanation'] for row in rows),
        'legal_quality_evaluated': False, 'rule_evidence_merged': False}


def replay(name, window, schema, N, LC, contract, clauses, cfg):
    from zhilv_qiling_v2 import model_audit
    folder = OUT / name
    folder.mkdir(exist_ok=False)
    os.environ['ZLQL_MODEL_AUDIT_PATH'] = str(folder / 'model_calls.sqlite')
    raw, pairs = [], []
    def observe(frame, event, arg):
        if frame.f_code is N.L._complete.__code__:
            if event == 'call':
                config = frame.f_locals['config']
                raw.append({'call_id': config.extra_headers.get('X-ZLQL-Call-ID', ''),
                    'request_messages': copy.deepcopy(frame.f_locals['messages']),
                    'json_schema': copy.deepcopy(frame.f_locals.get('json_schema')),
                    'configuration': {'endpoint': config.endpoint, 'model': config.model,
                        'max_tokens': frame.f_locals.get('max_tokens') or config.max_tokens,
                        'native_thinking': config.native_thinking, 'native_reasoning_format': config.native_reasoning_format,
                        'native_json_schema': config.native_json_schema}})
            elif event == 'return':
                row = raw[-1]
                row.update(content_text=frame.f_locals.get('content_text'),
                    raw_response_base64=base64.b64encode(frame.f_locals.get('raw_bytes', b'')).decode('ascii'),
                    envelope=copy.deepcopy(frame.f_locals.get('envelope')))
        elif frame.f_code is N.L.request_json.__code__ and event == 'return' and isinstance(arg, N.L.LLMPair):
            pairs.append(dataclasses.asdict(arg))
    trace_id = 'native-cross-replay-' + uuid.uuid4().hex
    started = time.perf_counter()
    with model_audit.review_audit(trace_id, contract.contract_id) as collector:
        try:
            sys.setprofile(observe)
            with patch.object(LC, 'build_review_plan', planner_for(window, clauses)):
                candidates, meta, degraded = N._llm_logic(contract, cfg, clauses)
        finally:
            sys.setprofile(None)
            write(folder / 'raw_transport_observations.json', raw)
            write(folder / 'raw_client_pairs.json', pairs)
            write(folder / 'collector_events.json', collector.events)
    write(folder / 'candidates.json', [candidate.to_dict() for candidate in candidates])
    write(folder / 'logic_result.json', {'meta': meta, 'degraded': degraded})
    with sqlite3.connect(folder / 'model_calls.sqlite') as database:
        database.row_factory = sqlite3.Row
        events = [dict(row) for row in database.execute('select * from model_calls order by rowid')]
    for event in events:
        event['details'] = json.loads(event['details'])
    write(folder / 'audit_events.json', events)
    source_hash = sha(canonical([part.as_dict() for part in window.segments]))
    schema_hash = sha(canonical(schema))
    job_id = window.window_id + ':batch:1'
    started_events = [event for event in events if event['status'] == 'started']
    terminal_events = [event for event in events if event['status'] != 'started']
    audit_ok = bool(started_events) and len(started_events) == len(terminal_events) == len(raw)
    for begin in started_events:
        call_id = begin['details'].get('call_id')
        terminal = [event for event in terminal_events if event['details'].get('call_id') == call_id]
        observed = [row for row in raw if row['call_id'] == call_id]
        audit_ok = audit_ok and len(terminal) == len(observed) == 1
        for event in [begin, *terminal]:
            details = event['details']
            audit_ok = audit_ok and event['trace_id'] == trace_id and event['contract_id'] == contract.contract_id
            audit_ok = audit_ok and details.get('native_json_schema_sent') is True
            audit_ok = audit_ok and details.get('window_id') == window.window_id and details.get('job_id') == job_id
            audit_ok = audit_ok and details.get('source_sha256') == source_hash and details.get('json_schema_sha256') == schema_hash
            audit_ok = audit_ok and details.get('max_tokens') == 4096 and details.get('native_thinking') == 'off'
    call_ids = {event['details'].get('call_id') for event in terminal_events}
    for candidate in candidates:
        assert candidate.source == 'node2_llm'
        for evidence in candidate.evidence:
            if evidence.source == 'node2_llm':
                assert evidence.extra['window_id'] == window.window_id
                assert evidence.extra['job_id'] == job_id and evidence.extra['call_id'] in call_ids
                assert evidence.extra['source_sha256'] == source_hash
    assessment = observations(raw, candidates, window)
    assessment.update(name=name, window_id=window.window_id, trace_id=trace_id,
        scope='one_archived_synthetic_diagnostic_cross_window_not_full_http_or_accuracy',
        source_sha256=source_hash, schema_sha256=schema_hash, audit_binding_assertions_passed=bool(audit_ok),
        actual_transport_attempts=len(raw), elapsed_seconds=round(time.perf_counter() - started, 6),
        call_statuses=[call['status'] for call in meta['calls']],
        audit_terminal_statuses=[event['status'] for event in terminal_events],
        model_identity_verified=bool(terminal_events) and all(event['details'].get('model_identity_verified') is True for event in terminal_events),
        finish_reasons=[event['details'].get('finish_reason') for event in terminal_events],
        model_audit_status=model_audit.status(),
        text_complete=meta.get('text_complete'), full_http_revalidated=False)
    write(folder / 'receipt.json', assessment)
    assert audit_ok, 'Native schema or window/job/call/source audit binding failed'
    assert not assessment['model_audit_status']['errors'], 'Persistent model audit failed'
    print(json.dumps({'name': name, 'window_id': window.window_id, 'attempts': len(raw),
        'two_sided_model_observation': assessment['raw_model_has_both_sides_and_explanation'],
        'statuses': assessment['call_statuses']}, ensure_ascii=True), flush=True)
    return assessment


def main():
    global NETWORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare_only == args.execute:
        parser.error('Choose exactly one phase')
    os.environ['ZLQL_LLM_API_KEY'] = ''
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    sys.meta_path.insert(0, NoModelImports())
    sys.addaudithook(guard)
    before = frozen_hashes()
    if args.prepare_only:
        OUT.mkdir(exist_ok=False)
        checks = selfcheck()
        assert frozen_hashes() == before
        write(OUT / 'frozen_hashes.json', before)
        write(OUT / 'source_fixture.json', json.loads((SOURCE / 'prepared.json').read_text(encoding='utf-8')))
        write(OUT / 'offline_selfcheck.json', {'status': 'passed', 'checks': checks,
            'source_prepared_sha256': sha((SOURCE / 'prepared.json').read_bytes()),
            'source_report_sha256': sha((SOURCE / 'report.json').read_bytes()),
            'tool_sha256': sha(Path(__file__).read_bytes()), 'expected_nodes_sha256': EXPECTED_NODES,
            'network_blocked': True, 'real_model_imports_blocked': True,
            'no_model_calls': True, 'no_invented_answers': True, 'frozen_bytes_unchanged': True})
        print(json.dumps({'status': 'offline_selfcheck_passed', 'windows': checks}, ensure_ascii=True), flush=True)
        return 0
    prior = json.loads((OUT / 'offline_selfcheck.json').read_text(encoding='utf-8'))
    assert prior['status'] == 'passed' and prior['tool_sha256'] == sha(Path(__file__).read_bytes())
    assert before == json.loads((OUT / 'frozen_hashes.json').read_text(encoding='utf-8'))
    assert not (OUT / 'execution_started.json').exists(), 'Execution already attempted; do not duplicate real calls'
    N, LC, contract, clauses, cfg, selected = restore()
    NETWORK = True
    initial_health = health()
    write(OUT / 'preflight_health.json', initial_health)
    write(OUT / 'execution_started.json', {'status': 'started', 'scope': 'three_target_windows_only',
        'tool_sha256': sha(Path(__file__).read_bytes()), 'nodes_sha256': EXPECTED_NODES,
        'synthetic_input': True, 'full_http_revalidated': False})
    results = []
    error = None
    try:
        for name, window, schema in selected:
            write(OUT / (name + '_source_window.json'), window.as_dict())
            results.append(replay(name, window, schema, N, LC, contract, clauses, cfg))
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        NETWORK = False
        after = frozen_hashes()
        unchanged = before == after
        write(OUT / 'execution_receipt.json', {'status': 'completed_targeted_replay' if error is None else 'failed_or_incomplete',
            'scope': 'three_archived_cross_windows_only_not_full_http_review', 'results': results, 'error': error,
            'source_prepared_sha256': prior['source_prepared_sha256'], 'source_report_sha256': prior['source_report_sha256'],
            'nodes_sha256': EXPECTED_NODES, 'tool_sha256': sha(Path(__file__).read_bytes()),
            'frozen_bytes_unchanged': unchanged, 'frozen_file_count': len(before),
            'torch_imported': 'torch' in sys.modules, 'transformers_imported': 'transformers' in sys.modules,
            'synthetic_input': True, 'legal_quality_evaluated': False, 'full_http_revalidated': False,
            'rule_evidence_merged': False, 'prompt_or_transport_replaced': False,
            'meaning': 'Observed model quotes and explanations only. Missing/partial remains missing/partial; no accuracy or completion guarantee.'})
    assert unchanged, 'Frozen product, proposals or source v2 artifacts changed'
    if error:
        print(json.dumps({'status': 'failed_or_incomplete', 'error': error}, ensure_ascii=True), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
