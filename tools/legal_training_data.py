"""Hash-pinned original legal-pair labels with law-family isolation.

This module never creates labels, mines random negatives or invents a licence.
An explicit manifest must describe the actual released files and label mapping.
Derived splits retain original row IDs and split names; this is not an official
benchmark result or an independent project human-gold acceptance set.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def _records(path: Path, spec: dict[str, Any]):
    kind = spec.get('format', path.suffix.lstrip('.')).lower()
    if kind == 'jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if kind == 'json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        for key in spec.get('record_path', []):
            data = data[key]
        if not isinstance(data, list):
            raise ValueError('JSON record_path must identify an array')
        return data
    if kind in ('csv', 'tsv'):
        with path.open(encoding=spec.get('encoding', 'utf-8-sig'), newline='') as stream:
            return list(csv.DictReader(stream, delimiter='\t' if kind == 'tsv' else ','))
    if kind == 'xlsx':
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook[spec['sheet']] if spec.get('sheet') else workbook.active
            rows = iter(sheet.values)
            headers = [str(x) for x in next(rows)]
            return [dict(zip(headers, row)) for row in rows]
        finally:
            workbook.close()
    raise ValueError('unsupported original source format: ' + kind)


def _field(record, key):
    value = record
    for part in key.split('.'):
        value = value[part]
    if value is None:
        return ''
    return str(value).strip()


def prepare(manifest_path: str | Path, *, seed=20261115, caps=None):
    path = Path(manifest_path).resolve()
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    for key in ('dataset_name', 'task', 'official_source', 'license', 'label_provenance', 'files', 'fields', 'label_mapping'):
        if not manifest.get(key):
            raise ValueError('missing documented original-source field: ' + key)
    if not manifest['license'].get('training_permitted') or not manifest['license'].get('url'):
        raise ValueError('explicit verified training licence is required')
    if manifest['label_provenance'].get('kind') != 'original_released_labels':
        raise ValueError('only original released labels are accepted; no project-generated labels')
    if not manifest['label_provenance'].get('description'):
        raise ValueError('original annotation method must be described')
    fields = manifest['fields']
    if any(not fields.get(k) for k in ('left_text', 'right_text', 'left_family', 'right_family', 'label')):
        raise ValueError('explicit original text, family and label columns required')
    mapping = manifest['label_mapping']
    if any(not isinstance(label, str) or not label.strip() for label in mapping.values()):
        raise ValueError('every original class needs an explicit nonempty target name')
    family_scope = manifest.get('family_scope', 'both_query_and_reference')
    if family_scope not in ('both_query_and_reference', 'query_source_family'):
        raise ValueError('unknown law-family isolation scope')
    rows, files, pair_seen, rejected = [], [], {}, []
    specs = list(manifest['files'])
    if manifest.get('split_policy') == 'provided_disjoint_families':
        # Keep the original held-out copy when the release contains identical
        # pairs in more than one split. Training copies are removed, never the
        # other way round merely because train.jsonl appeared first.
        priority = {'test':0, 'valid':1, 'validation':1, 'dev':1, 'train':2}
        specs.sort(key=lambda item:priority.get(item.get('split'), 3))
    for file_index, spec in enumerate(specs):
        source = (path.parent / spec['path']).resolve()
        payload = source.read_bytes()
        actual_hash = sha256(payload)
        if actual_hash != spec.get('sha256'):
            raise ValueError('original source SHA-256 mismatch: ' + str(source))
        files.append({'path':str(source), 'sha256':actual_hash, 'original_split':spec.get('split')})
        for index, record in enumerate(_records(source, spec)):
            raw_label = _field(record, fields['label'])
            if raw_label not in mapping:
                raise ValueError('unmapped original label: ' + raw_label)
            left, right = _field(record, fields['left_text']), _field(record, fields['right_text'])
            query_family = _field(record, fields['left_family'])
            reference_family = _field(record, fields['right_family'])
            families = sorted(set((query_family, reference_family))) if family_scope == 'both_query_and_reference' else [query_family]
            if not left or not right or not query_family or not reference_family:
                rejected.append({'file':file_index, 'row':index, 'reason':'empty_text_or_law_family'})
                continue
            pair = sha256('\x1f'.join(sorted((left, right))))
            label = mapping[raw_label]
            if pair in pair_seen:
                if pair_seen[pair] != label:
                    raise ValueError('conflicting original labels for identical/reversed pair')
                rejected.append({'file':file_index, 'row':index, 'reason':'duplicate_or_reversed_pair'})
                continue
            pair_seen[pair] = label
            original_id = _field(record, fields['id']) if fields.get('id') else f'{file_index}:{index}'
            rows.append({'pair_id':pair, 'clause':left, 'article':right, 'label':label,
                'law_families':families, 'source_file_index':file_index, 'source_row':index,
                'reference_family':reference_family,
                'original_id':original_id, 'original_label':raw_label, 'original_split':spec.get('split')})
    policy = manifest.get('split_policy', 'family_hash')
    buckets = {'train':[], 'valid':[], 'test':[]}
    if policy == 'family_hash':
        def assignment(family):
            n = int(sha256(str(seed) + '\x1f' + family)[:8], 16) % 1000
            return 'test' if n < 150 else 'valid' if n < 300 else 'train'
        for row in rows:
            target = {assignment(family) for family in row['law_families']}
            if len(target) != 1:
                rejected.append({'pair_id':row['pair_id'], 'reason':'cross_split_law_families'})
            else:
                buckets[target.pop()].append(row)
    elif policy == 'provided_disjoint_families':
        aliases = {'train':'train', 'valid':'valid', 'validation':'valid', 'dev':'valid', 'test':'test'}
        heldout = set()
        for split in ('test', 'valid', 'train'):
            selected = [row for row in rows if aliases.get(row['original_split']) == split]
            for row in selected:
                if heldout.intersection(row['law_families']):
                    rejected.append({'pair_id':row['pair_id'], 'reason':'original_split_law_family_overlap'})
                else:
                    buckets[split].append(row)
            heldout.update(family for row in buckets[split] for family in row['law_families'])
        if any(row['original_split'] not in aliases for row in rows):
            raise ValueError('all original rows must have an explicit supported split')
    else:
        raise ValueError('unknown split policy')
    for split, limit in (caps or {}).items():
        ordered = sorted(buckets[split], key=lambda r:sha256(str(seed) + r['pair_id']))
        labels = sorted(set(row['label'] for row in ordered))
        if limit < len(labels):
            raise ValueError('cap must preserve all available original labels')
        chosen = []
        for label in labels:
            chosen.extend([row for row in ordered if row['label'] == label][:limit // len(labels)])
        chosen_ids = {row['pair_id'] for row in chosen}
        chosen.extend(row for row in ordered if row['pair_id'] not in chosen_ids and len(chosen) < limit)
        buckets[split] = chosen
    families = {split:{f for row in records for f in row['law_families']} for split, records in buckets.items()}
    if any(families[a] & families[b] for a,b in (('train','valid'),('train','test'),('valid','test'))):
        raise AssertionError('law family leakage')
    if any(not bucket for bucket in buckets.values()):
        raise ValueError('source cannot provide nonempty law-family-disjoint train/valid/test splits')
    labels = set(mapping.values())
    if len(labels) < 2 or any(set(row['label'] for row in bucket) != labels for bucket in buckets.values()):
        coverage = {split:{label:sum(row['label'] == label for row in records) for label in sorted(labels)}
            for split,records in buckets.items()}
        raise ValueError('each law-family-disjoint split must contain every released target class: '
            +json.dumps({'coverage':coverage,'family_counts':{k:len(v) for k,v in families.items()},
                         'rejected':len(rejected)},ensure_ascii=True))
    return {'manifest':manifest, 'manifest_sha256':sha256(raw_manifest), 'files':files,
        'splits':buckets, 'law_families':{k:sorted(v) for k,v in families.items()},
        'rejected':rejected, 'counts':{k:len(v) for k,v in buckets.items()},
        'split_policy':policy, 'family_scope':family_scope, 'law_family_overlap':False, 'project_human_gold':False,
        'reference_family_overlap_allowed':family_scope == 'query_source_family',
        'limitation':'original released labels for the documented auxiliary legal task; not official contract-law benchmark scores or independent project human-gold acceptance'}
