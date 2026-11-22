"""Passively observe the base runner using an existing complete public contract.

Without --execute this only prepares local text. With --execute the unchanged
base HTTP runner owns every real call. The v3 PassiveCapture preserves original
request/response bytes, attempt identities, and isolated SQLite reconciliation.
No source is refetched, no synthetic fixture is substituted, and no legal gold
or public-subject authenticity is claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

import verify_complex_core_workflow as base
from verify_complex_core_observed_v3 import PassiveCapture, METHOD, plain, reconcile_calls, sha

ROOT = base.ROOT


class PublicPassiveCapture(PassiveCapture):
    """Reuse v3 hooks, additionally retaining the new public receipt field."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_request_receipts = None

    def _on_return(self, kind, frame, arg):
        # The inherited profile hook already holds the RLock. Observe only the
        # plain public audit field; never read config/headers or alter the pair.
        try:
            if kind == 'request' and arg is not None:
                fields = object.__getattribute__(arg, '__dict__')
                if type(fields) is dict and 'request_receipts' in fields:
                    self._observed_request_receipts = plain(fields['request_receipts'])
            return super()._on_return(kind, frame, arg)
        finally:
            self._observed_request_receipts = None

    def _write(self, name, value):
        if self._observed_request_receipts is not None and 'parsed_pair' in value:
            value = {**value, 'parsed_pair': {**value['parsed_pair'],
                      'request_receipts': self._observed_request_receipts}}
        return super()._write(name, value)


def wrapper_provenance():
    return {
        'observation_wrapper': Path(__file__).name,
        'observed_runner_sha256': sha(Path(__file__).read_bytes()),
        'observation_method': METHOD,
        'observation_scope': 'unchanged base public-contract HTTP runner; existing local copy; not legal gold',
        'diagnostic_runner': Path(base.__file__).name,
        'diagnostic_runner_sha256': sha(Path(base.__file__).read_bytes()),
        'passive_capture_provider': 'verify_complex_core_observed_v3.py',
        'passive_capture_provider_sha256': sha((Path(__file__).parent / 'verify_complex_core_observed_v3.py').read_bytes()),
        'acceptance_common_sha256': sha((Path(__file__).parent / 'acceptance_common.py').read_bytes()),
        'public_source_refetched_this_run': False,
        'input_is_independent_legal_gold': False,
    }


def public_source_binding(directory):
    """Independently bind all numbered local pages, without fetching a site."""
    source = Path(directory).resolve()
    raw, name, manifest = base.public_contract_source(source)
    metadata = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
    pages = sorted((source / 'text').glob('page_*.txt'))
    count = metadata.get('page_count')
    expected_names = [f'page_{number:04d}.txt' for number in range(1, count + 1)] if type(count) is int and count > 0 else []
    if not expected_names or [path.name for path in pages] != expected_names:
        raise ValueError('all consecutively numbered public pages are required')
    if any(not path.read_text(encoding='utf-8').strip() for path in pages):
        raise ValueError('every public page must contain local extracted text')
    return {'source_directory': str(source), 'source_name': name, 'raw_text': raw,
            'raw_text_sha256': sha(raw.encode('utf-8')), 'raw_text_chars': len(raw),
            'source_manifest': manifest, 'all_numbered_page_texts_supplied': True,
            'page_text_count': count, 'source_refetched_this_run': False,
            'input_is_independent_legal_gold': False,
            'text_scope': 'all existing extracted page text; OCR correctness not independently evaluated',
            'subject_fact_scope': 'no synthetic subject records or external authenticity assumptions'}


def check_prepared(prepared, expected_provenance, expected_source):
    if type(prepared) is not dict or prepared.get('synthetic_input') is not False:
        raise ValueError('an explicit real public local copy is required')
    if prepared.get('legal_quality_evaluated') is not False:
        raise ValueError('preparation cannot claim legal quality evaluation')
    if (prepared.get('raw_text') != expected_source['raw_text']
            or prepared.get('raw_text_sha256') != expected_source['raw_text_sha256']
            or prepared.get('source_name') != expected_source['source_name']
            or prepared.get('source_manifest') != expected_source['source_manifest']):
        raise ValueError('prepared public source bytes, page inventory, or manifest differ')
    if prepared.get('tool_sha256') != sha(Path(base.__file__).read_bytes()):
        raise ValueError('prepared base runner hash differs')
    provenance = prepared.get('provenance') or {}
    expected = {**wrapper_provenance(),
                'product_source_sha256': expected_provenance.get('product_source_sha256')}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('prepared observation/base/product provenance differs')
    if 'fixture_version' in provenance or 'fixture_version' in prepared['source_manifest']:
        raise ValueError('a public source must not carry a synthetic fixture version')
    if prepared.get('subject_records') != [] or prepared.get('subject_identity_diagnostics') != {}:
        raise ValueError('public preparation must not inherit synthetic subject records')
    parsed = prepared.get('parsed_text')
    if not isinstance(parsed, str) or prepared.get('parsed_text_sha256') != sha(parsed.encode('utf-8')):
        raise ValueError('prepared parsed text hash differs')


def result_binding(directory, expected_provenance, expected_source):
    prepared_path = directory / 'prepared.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    check_prepared(prepared, expected_provenance, expected_source)
    current_source = public_source_binding(Path(expected_source['source_directory']))
    if current_source != expected_source:
        raise ValueError('local public source changed during observed execution')
    if base.source_provenance()['product_source_sha256'] != expected_provenance['product_source_sha256']:
        raise ValueError('product source changed during observed execution')
    binding = {key: value for key, value in expected_source.items() if key != 'raw_text'}
    binding.update(prepared_sha256=sha(prepared_path.read_bytes()),
                   parsed_text_sha256=prepared['parsed_text_sha256'],
                   prepared_provenance=prepared['provenance'], result_available=False,
                   prepared_result_provenance_match=False)
    result_path = directory / 'result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        binding.update(result_available=True, result_sha256=sha(result_path.read_bytes()))
        if (result.get('synthetic_input') is not False or result.get('legal_quality_evaluated') is not False
                or result.get('prepared_sha256') != binding['prepared_sha256']
                or result.get('tool_sha256') != prepared['tool_sha256']
                or result.get('source_manifest') != prepared['source_manifest']
                or any((result.get('provenance') or {}).get(key) != value
                       for key, value in prepared['provenance'].items() if key != 'captured_at')):
            raise ValueError('actual result is not bound to the prepared public sources')
        binding['prepared_result_provenance_match'] = True
    input_path = directory / 'input-contract.txt'
    if not input_path.is_file() or input_path.read_text(encoding='utf-8') != expected_source['raw_text']:
        raise ValueError('saved input differs from the complete local public text')
    binding['saved_input_sha256'] = sha(input_path.read_bytes())
    report_path = directory / 'report.json'
    binding['report_available'] = report_path.is_file()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        binding['report_sha256'] = sha(report_path.read_bytes())
        binding['report_text_matches_prepared'] = report.get('document', {}).get('full_text') == prepared['parsed_text']
        if not binding['report_text_matches_prepared']:
            raise ValueError('returned report does not preserve the prepared complete public text')
    return binding


def finalize_observation(observer, directory, expected_provenance, expected_source):
    """Contain observer failures; never replace the base return/exception."""
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
        manifest['independent_legal_gold'] = False
        manifest['public_subject_authenticity_verified'] = False
        try:
            manifest['source_binding'] = result_binding(directory, expected_provenance, expected_source)
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
        try:
            print('Observation evidence insufficient; finalization error: ' + type(exc).__name__, file=sys.stderr)
        except BaseException:
            pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--public-contract-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args, _ = parser.parse_known_args(argv)
    directory = args.run_dir.resolve()
    if not directory.is_relative_to((ROOT / 'tools/acceptance_results').resolve()):
        raise ValueError('run-dir must remain under tools/acceptance_results')
    if any((directory / name).exists() for name in ('raw_captures', 'capture_manifest.json',
            'result.json', 'workspace.sqlite', 'model_calls.sqlite')):
        raise ValueError('fresh execution/capture directory required; preserve prior evidence')
    expected_source = public_source_binding(args.public_contract_dir)
    original_provenance = base.source_provenance
    expected_provenance = original_provenance()
    prepared_path = directory / 'prepared.json'
    if prepared_path.exists():
        check_prepared(json.loads(prepared_path.read_text(encoding='utf-8')), expected_provenance, expected_source)
    observer = None

    def provenance():
        return {**original_provenance(), **wrapper_provenance()}

    base.source_provenance = provenance
    try:
        if args.execute:
            from zhilv_qiling_v2 import llm_client
            directory.mkdir(parents=True, exist_ok=True)
            observer = PublicPassiveCapture(directory / 'raw_captures', llm_client._complete.__code__,
                                            llm_client.request_json.__code__)
            observer.start()
        return base.main(argv)
    finally:
        try:
            if observer is not None:
                finalize_observation(observer, directory, expected_provenance, expected_source)
        finally:
            base.source_provenance = original_provenance


if __name__ == '__main__':
    raise SystemExit(main())
