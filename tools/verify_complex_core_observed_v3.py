"""Observe the unchanged v2 synthetic HTTP diagnostic without replacing calls.

Only --execute installs Python profile hooks.  No transport, prompt, model
return, product source, or v2 fixture is patched.  Captures are diagnostic data,
not legal gold or performance acceptance.  Configuration objects, headers and
environment variables are never inspected by this observer.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time

import verify_complex_core_v2 as v2

base = v2.base
ROOT = base.ROOT
METHOD = 'sys.setprofile+threading.setprofile; exact Python code identity; passive v3'
META_KEYS = frozenset(('call_id', 'requested_model', 'attempt', 'max_tokens', 'prompt_chars',
    'native_thinking', 'native_reasoning_format', 'native_json_schema_requested',
    'native_json_schema_sent', 'json_schema_sha256', 'window_id', 'job_id', 'source_sha256'))
PAIR_KEYS = ('content', 'model', 'elapsed', 'raw', 'prompt_chars', 'degraded', 'reason_code', 'call_id')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plain(value):
    """Copy JSON primitives only; never invoke arbitrary serializers/properties."""
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind in (list, tuple):
        return [plain(item) for item in value]
    if kind is dict and all(type(key) is str for key in value):
        return {key: plain(item) for key, item in value.items()}
    raise TypeError('capture value is not a plain JSON value')


def safe_meta(value):
    if type(value) is not dict:
        return {}
    return {key: plain(value[key]) for key in sorted(META_KEYS) if key in value}


def bytes_record(value):
    if type(value) is not bytes:
        raise TypeError('original byte buffer unavailable')
    return {'encoding': 'base64', 'data': base64.b64encode(value).decode('ascii'),
            'bytes': len(value), 'sha256': sha(value)}


class PassiveCapture:
    """A failure-contained profiler for two exact functions and future threads."""

    def __init__(self, directory: Path, complete_code, request_code):
        self.directory = Path(directory)
        self.complete_code = complete_code
        self.request_code = request_code
        self.lock = threading.RLock()
        self.frames = {}
        self.files = []
        self.errors = []
        self.incomplete = []
        self.counts = {'complete_calls': 0, 'complete_returns': 0,
                       'request_calls': 0, 'request_returns': 0}
        self.thread_ids = set()
        self.binding_counts = {'bound': 0, 'not_supplied_by_caller': 0, 'partial': 0}
        self.complete_call_ids = []
        self.active = False
        self.installed = False
        self.elapsed_in_hook = 0.0
        self.hook_events = 0
        self.previous_sys = None
        self.previous_threading = None
        self._hook = self._profile

    def _error(self, stage, exc):
        # Exception text may contain private objects; retain only type/stage.
        try:
            self.errors.append({'stage': stage, 'error_type': type(exc).__name__})
        except BaseException:
            pass

    def _write(self, name, value):
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        path = self.directory / name
        with path.open('xb') as handle:
            handle.write(data)
        self.files.append({'path': 'raw_captures/' + name, 'bytes': len(data), 'sha256': sha(data)})

    def start(self):
        if self.installed:
            raise RuntimeError('capture is already installed')
        self.previous_sys = sys.getprofile()
        self.previous_threading = threading.getprofile()
        if self.previous_sys is not None or self.previous_threading is not None:
            raise RuntimeError('existing profile hook must not be silently replaced')
        self.directory.mkdir(parents=False, exist_ok=False)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.preexisting_other_threads = sum(t.ident != threading.get_ident() for t in threading.enumerate())
        self.active = True
        self.installed = True
        try:
            threading.setprofile(self._hook)
            sys.setprofile(self._hook)
        except BaseException:
            self.active = False
            sys.setprofile(self.previous_sys)
            threading.setprofile(self.previous_threading)
            self.installed = False
            raise

    def stop(self):
        if not self.installed:
            return
        self.active = False
        sys.setprofile(self.previous_sys)
        threading.setprofile(self.previous_threading)
        self.installed = False

    def _profile(self, frame, event, arg):
        # Python exceptions escaping a profile hook would disable it and alter
        # the observed program.  Contain all observer failures, including IO.
        if not self.active or event not in ('call', 'return'):
            return
        code = frame.f_code
        if code is not self.complete_code and code is not self.request_code:
            return
        started = None
        kind = 'profile'
        try:
            started = time.perf_counter()
            with self.lock:
                self.hook_events += 1
                self.thread_ids.add(threading.get_ident())
                kind = 'complete' if code is self.complete_code else 'request'
                if event == 'call':
                    self._on_call(kind, frame)
                else:
                    self._on_return(kind, frame, arg)
        except BaseException as exc:
            self._error(kind + '_' + event, exc)
        finally:
            try:
                if started is not None:
                    self.elapsed_in_hook += time.perf_counter() - started
            except BaseException as exc:
                self._error('hook_timing', exc)

    def _on_call(self, kind, frame):
        self.counts[kind + '_calls'] += 1
        number = self.counts[kind + '_calls']
        observation_id = f'{kind}_{number:06d}'
        row = {'observation_id': observation_id, 'thread_id': threading.get_ident()}
        self.frames[id(frame)] = row
        local = frame.f_locals
        row['metadata'] = safe_meta(local.get('meta')) if kind == 'request' else {}
        if kind == 'complete':
            caller = frame.f_back
            if caller is not None and caller.f_code is self.request_code:
                row['metadata'] = safe_meta(caller.f_locals.get('meta'))
                parent = self.frames.get(id(caller))
                row['request_observation_id'] = parent['observation_id'] if parent else None
            if not row['metadata'].get('call_id'):
                self.incomplete.append({'observation_id': observation_id, 'reason': 'source_call_id_unavailable'})
            self.complete_call_ids.append(row['metadata'].get('call_id'))
            bindings = [bool(row['metadata'].get(key)) for key in ('window_id', 'job_id', 'source_sha256')]
            binding_kind = 'bound' if all(bindings) else 'partial' if any(bindings) else 'not_supplied_by_caller'
            row['source_window_binding'] = binding_kind
            self.binding_counts[binding_kind] += 1
            if binding_kind == 'partial':
                self.incomplete.append({'observation_id': observation_id, 'reason': 'source_window_binding_unavailable'})
        payload = {**row, 'event': 'call', 'messages': plain(local.get('messages')),
                   'max_tokens': plain(local.get('max_tokens')), 'json_schema': plain(local.get('json_schema'))}
        self._write(observation_id + '_input.json', payload)

    def _on_return(self, kind, frame, arg):
        self.counts[kind + '_returns'] += 1
        row = self.frames.pop(id(frame), None)
        if row is None:
            self.incomplete.append({'kind': kind, 'reason': 'return_without_observed_entry'})
            return
        local = frame.f_locals
        payload = {**row, 'event': 'return',
                   'return_kind': 'returned_value' if arg is not None else 'no_return_value_or_unwind'}
        if kind == 'complete':
            for key in ('body', 'raw_bytes'):
                if key in local:
                    payload[key] = bytes_record(local[key])
            for key in ('raw_text', 'envelope', 'content_text', 'elapsed'):
                if key in local:
                    payload[key] = plain(local[key])
            if type(arg) is tuple and len(arg) == 3:
                payload['original_return'] = plain(arg)
            elif arg is not None:
                self.incomplete.append({'observation_id': row['observation_id'], 'reason': 'unexpected_complete_return'})
            if 'body' not in payload:
                self.incomplete.append({'observation_id': row['observation_id'], 'reason': 'request_bytes_unavailable'})
            if 'raw_bytes' not in payload:
                # HTTPError only exposes a truncated diagnostic in the original
                # transport.  Do not read its response or retry it here.
                self.incomplete.append({'observation_id': row['observation_id'], 'reason': 'full_response_bytes_unavailable'})
        else:
            payload['metadata'] = safe_meta(local.get('meta'))
            if arg is not None:
                fields = object.__getattribute__(arg, '__dict__')
                if type(fields) is not dict or not all(key in fields for key in PAIR_KEYS):
                    raise TypeError('request result is not a complete plain LLMPair')
                payload['parsed_pair'] = {key: plain(fields[key]) for key in PAIR_KEYS}
        self._write(row['observation_id'] + '_return.json', payload)

    def summary(self):
        with self.lock:
            complete_files = len(self.files) == 2 * sum(self.counts[key] for key in ('complete_calls', 'request_calls'))
            paired = not self.frames and all(self.counts[k + '_calls'] == self.counts[k + '_returns']
                                            for k in ('complete', 'request'))
            integrity = bool(self.counts['complete_calls']) and complete_files and paired and not self.errors
            return {'method': METHOD, 'counts': dict(self.counts), 'files': list(self.files),
                'capture_errors': list(self.errors), 'evidence_gaps': list(self.incomplete),
                'unpaired_frames': [dict(row) for row in self.frames.values()],
                'capture_integrity_complete': integrity,
                'all_raw_model_responses_available': integrity and not any(
                    gap['reason'] == 'full_response_bytes_unavailable' for gap in self.incomplete),
                'source_call_bindings_complete': integrity and not any(
                    gap['reason'].startswith('source_') for gap in self.incomplete),
                'source_window_binding_counts': dict(self.binding_counts),
                'source_complete_call_ids': list(self.complete_call_ids),
                'binding_scope_note': 'All source call_ids required; window/job/source copied only when caller supplies them. Unbound judge calls are not assigned guessed windows.',
                'evidence_sufficient_for_full_response_analysis': integrity and not self.incomplete,
                'profile_thread_coverage': {'main_thread_and_future_threads': True,
                    'preexisting_other_thread_count': self.preexisting_other_threads,
                    'preexisting_thread_hook_not_retrofitted': True,
                    'observed_thread_ids': sorted(self.thread_ids)},
                'started_at': self.started_at,
                'ended_at': datetime.now(timezone.utc).isoformat(),
                'capture_hook_events': self.hook_events,
                'cumulative_hook_seconds': self.elapsed_in_hook,
                'hook_time_is_instrumentation_not_performance_acceptance': True,
                'configuration_headers_environment_read_by_observer': False}


def wrapper_provenance():
    return {'observation_wrapper': Path(__file__).name,
            'observed_runner_sha256': sha(Path(__file__).read_bytes()),
            'observation_method': METHOD,
            'observation_scope': 'unchanged scoped_value_tail_v2 synthetic diagnostic; not legal gold'}


def check_prepared(prepared, expected_provenance):
    """Fail closed on legacy/stale preparation, including product provenance."""
    if type(prepared) is not dict or prepared.get('synthetic_input') is not True:
        raise ValueError('only the fixed synthetic preparation is supported')
    raw = v2.scoped_contract()
    if prepared.get('raw_text') != raw or prepared.get('raw_text_sha256') != sha(raw.encode('utf-8')):
        raise ValueError('prepared fixture differs from the unchanged v2 fixture')
    if prepared.get('tool_sha256') != sha(Path(base.__file__).read_bytes()):
        raise ValueError('prepared base runner hash differs')
    provenance = prepared.get('provenance') or {}
    expected = {**wrapper_provenance(),
        'diagnostic_runner': Path(v2.__file__).name,
        'diagnostic_runner_sha256': sha(Path(v2.__file__).read_bytes()),
        'fixture_version': 'scoped_value_tail_v2', 'fixture_is_independent_legal_gold': False,
        'product_source_sha256': expected_provenance.get('product_source_sha256')}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('prepared observation/diagnostic/product provenance differs')
    source = prepared.get('source_manifest') or {}
    if (source.get('mode') != 'synthetic_diagnostic_not_real_transaction'
            or source.get('fixture_version') != 'scoped_value_tail_v2'
            or source.get('runner_sha256') != expected['diagnostic_runner_sha256']):
        raise ValueError('prepared source manifest differs')


def result_binding(directory, expected_provenance):
    prepared_path = directory / 'prepared.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    check_prepared(prepared, expected_provenance)
    result_path = directory / 'result.json'
    binding = {'prepared_sha256': sha(prepared_path.read_bytes()),
               'prepared_provenance': prepared['provenance'], 'result_available': result_path.is_file(),
               'prepared_result_provenance_match': False}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        binding['result_sha256'] = sha(result_path.read_bytes())
        if (result.get('prepared_sha256') != binding['prepared_sha256']
                or result.get('tool_sha256') != prepared.get('tool_sha256')
                or result.get('source_manifest') != prepared.get('source_manifest')
                or any((result.get('provenance') or {}).get(key) != value
                       for key, value in prepared['provenance'].items() if key != 'captured_at')):
            raise ValueError('actual result is not bound to the prepared sources')
        binding['prepared_result_provenance_match'] = True
    current = base.source_provenance()['product_source_sha256']
    if current != expected_provenance['product_source_sha256']:
        raise ValueError('product source changed during observed execution')
    return binding


def reconcile_calls(directory, summary):
    """Independently account for started calls from the isolated read-only audit."""
    path = directory / 'model_calls.sqlite'
    db = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True, timeout=3)
    try:
        rows = db.execute("SELECT details FROM model_calls WHERE component='llm' "
                          "AND operation='request_json' AND status='started'").fetchall()
    finally:
        db.close()
    actual = [json.loads(row[0]).get('call_id') for row in rows]
    captured = summary['source_complete_call_ids']
    valid = all(type(value) is str and value for value in actual + captured)
    unique = len(actual) == len(set(actual)) and len(captured) == len(set(captured))
    matches = bool(actual) and valid and unique and Counter(actual) == Counter(captured)
    return {'source': 'isolated model_calls.sqlite opened mode=ro',
        'audit_started_count': len(actual), 'captured_complete_count': len(captured),
        'all_call_ids_present': valid, 'call_ids_unique_per_attempt': unique,
        'audit_started_calls_exactly_match_captured_attempts': matches,
        'missing_capture_call_ids': sorted(value for value in (set(actual) - set(captured)) if type(value) is str),
        'unexpected_capture_call_ids': sorted(value for value in (set(captured) - set(actual)) if type(value) is str),
        'audit_sha256': sha(path.read_bytes())}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--public-contract-dir', type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.public_contract_dir is not None:
        raise ValueError('observed v3 only captures the fixed synthetic v2 diagnostic')
    directory = args.run_dir.resolve()
    if not directory.is_relative_to((ROOT / 'tools/acceptance_results').resolve()):
        raise ValueError('run-dir must remain under tools/acceptance_results')
    if (directory / 'raw_captures').exists() or (directory / 'capture_manifest.json').exists():
        raise ValueError('fresh capture evidence directory required')
    original = {name: getattr(base, name) for name in ('synthetic_contract', 'prepare_case', 'source_provenance')}
    expected_provenance = original['source_provenance']()
    prepared_path = directory / 'prepared.json'
    if prepared_path.exists():
        check_prepared(json.loads(prepared_path.read_text(encoding='utf-8')), expected_provenance)
    observer = None

    def provenance():
        return {**original['source_provenance'](), **wrapper_provenance()}

    base.source_provenance = provenance
    try:
        if args.execute:
            # This module is stdlib-only; no torch/model import is required by
            # capture installation.  The original runner owns real execution.
            from zhilv_qiling_v2 import llm_client
            directory.mkdir(parents=True, exist_ok=True)
            observer = PassiveCapture(directory / 'raw_captures', llm_client._complete.__code__,
                                      llm_client.request_json.__code__)
            observer.start()
        return v2.main(argv)
    finally:
        try:
            if observer is not None:
                try:
                    observer.stop()
                except BaseException as exc:
                    observer._error('stop', exc)
                    observer.active = False
                    for setter, previous in ((sys.setprofile, observer.previous_sys),
                                             (threading.setprofile, observer.previous_threading)):
                        try:
                            setter(previous)
                        except BaseException as restore_exc:
                            observer._error('stop_restore_hook', restore_exc)
                try:
                    manifest = observer.summary()
                    manifest['wrapper_provenance'] = wrapper_provenance()
                    try:
                        manifest['source_binding'] = result_binding(directory, expected_provenance)
                        if not manifest['source_binding']['prepared_result_provenance_match']:
                            manifest['evidence_sufficient_for_full_response_analysis'] = False
                    except BaseException as exc:
                        manifest['source_binding_error_type'] = type(exc).__name__
                        manifest['evidence_sufficient_for_full_response_analysis'] = False
                    try:
                        manifest['actual_call_reconciliation'] = reconcile_calls(directory, manifest)
                        if not manifest['actual_call_reconciliation']['audit_started_calls_exactly_match_captured_attempts']:
                            manifest['evidence_sufficient_for_full_response_analysis'] = False
                    except BaseException as exc:
                        manifest['call_reconciliation_error_type'] = type(exc).__name__
                        manifest['evidence_sufficient_for_full_response_analysis'] = False
                    data = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
                    with (directory / 'capture_manifest.json').open('xb') as handle:
                        handle.write(data)
                except BaseException as exc:
                    # Do not replace the original runner's return or exception.
                    try:
                        print('Observation evidence insufficient; finalization error: ' + type(exc).__name__, file=sys.stderr)
                    except BaseException:
                        pass
        finally:
            for name, value in original.items():
                setattr(base, name, value)


if __name__ == '__main__':
    raise SystemExit(main())
