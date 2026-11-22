"""Prepare original LCR-CN conflict-type labels; never derive new labels.

The original query provisions and expert-released conflict types are used for
the auxiliary legislative-conflict task. golden_content and distractors are
not training examples and never become invented non-conflict labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from legal_training_data import sha256


def family(title: str) -> str:
    title = re.sub(r'\s+', '', title)
    # Versions of the same named source regulation belong to one query family.
    title = re.sub(r'[（(][^）)]*(?:19|20)\d{2}[^）)]*[）)]', '', title)
    title = re.sub(r'[（(](?:修订|修正|试行|暂行|草案)[）)]', '', title)
    return title


def references(value):
    if not isinstance(value, list) or not value:
        raise ValueError('original high_level_laws must be a nonempty list')
    texts, titles = [], []
    for item in value:
        if isinstance(item, str):
            # No law name is inferred from legal prose. The complete released
            # reference remains a shared-reference audit identity.
            text = item.strip()
            title = 'released-reference:' + sha256(text)[:20]
        elif isinstance(item, dict):
            if 'clause' not in item or 'title' not in item:
                raise ValueError('unknown original high_level_laws dictionary schema')
            text = str(item['clause']).strip()
            title = str(item['title']).strip()
        else:
            raise ValueError('unknown original high_level_laws item type')
        if not text or not title:
            raise ValueError('empty original superior-law reference')
        texts.append(text)
        titles.append(title)
    return '\n'.join(texts), '\x1f'.join(sorted(set(titles)))


def provision_families(records, threshold=.85):
    """Group reused underlying provisions without feeding revisions to a model.

    Exact normalized query/revision matches merge globally. Near-duplicates
    merge only within the same anonymized source-document title. Expert
    revisions are used for leakage exclusion only, never as model inputs or
    newly labelled examples.
    """
    parent=list(range(len(records)))
    def root(index):
        while parent[index] != index:
            parent[index]=parent[parent[index]]
            index=parent[index]
        return index
    def join(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    def canonical(text):return re.sub(r'[^0-9a-zA-Z\u3400-\u9fff]','',text).lower()
    seen,by_title,grams={},{},[]
    for index,row in enumerate(records):
        texts=[canonical(row['query']),canonical(row['expert_revised_content_for_family_only'])]
        texts=[text for text in set(texts) if text]
        for text in texts:
            if text in seen:join(index,seen[text])
            else:seen[text]=index
        grams.append([set(text[i:i+5] for i in range(max(1,len(text)-4))) for text in texts])
        by_title.setdefault(family(row['original_title']),[]).append(index)
    comparisons=0
    for indexes in by_title.values():
        for slot,a in enumerate(indexes):
            for b in indexes[slot+1:]:
                if root(a)==root(b):continue
                for ga in grams[a]:
                    for gb in grams[b]:
                        if not ga or not gb or min(len(ga),len(gb))/max(len(ga),len(gb))<threshold:continue
                        comparisons+=1
                        overlap=len(ga&gb)
                        if overlap/(len(ga)+len(gb)-overlap)>=threshold:
                            join(a,b)
                            break
                    if root(a)==root(b):break
    keys={}
    for index,row in enumerate(records):
        key=sha256(family(row['original_title'])+'\x1f'+canonical(row['query']))
        group=root(index)
        keys[group]=min(keys.get(group,key),key)
    return ['provision-family:'+keys[root(i)] for i in range(len(records))],{
        'groups':len(keys),'near_duplicate_comparisons':comparisons,'char_ngram_size':5,
        'jaccard_threshold':threshold,'revision_text_model_input':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family-level', choices=('document','provision'), default='document')
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('refuse existing prepared output')
    meta_path=args.source/'figshare_metadata.json'
    metadata=json.loads(meta_path.read_text(encoding='utf-8'))
    licence=metadata.get('license',{})
    if licence.get('name') != 'CC BY 4.0' or licence.get('url') != 'https://creativecommons.org/licenses/by/4.0/':
        raise ValueError('source licence differs from verified CC BY 4.0 metadata')
    if metadata.get('id') != 30315955 or metadata.get('version') != 4:
        raise ValueError('source DOI/version differs from verified LCR-CN v4')
    args.output.mkdir(parents=True,exist_ok=True)
    specifications={item['name']:item for item in metadata['files']}
    normalized,counts,raw_hashes,labels=[],{},[],set()
    prepared_counts={}
    quarantined=[]
    raw_dir=args.source/'raw' if (args.source/'raw').is_dir() else args.source
    for original,split in (('train.jsonl','train'),('val.jsonl','valid'),('test.jsonl','test')):
        source=raw_dir/original
        spec=specifications[original]
        md5=hashlib.md5()
        sha=hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):
                md5.update(chunk);sha.update(chunk)
        if source.stat().st_size != spec['size'] or md5.hexdigest() != spec['computed_md5']:
            raise ValueError('original Figshare size/MD5 verification failed: '+original)
        raw_hashes.append({'file':original,'sha256':sha.hexdigest(),'md5':md5.hexdigest(),'original_split':original[:-6]})
        target=args.output/original
        counts[split]={}
        prepared_counts[split]=0
        with source.open('rb') as stream,target.open('w',encoding='utf-8') as out:
            offset=0
            for line_index,line in enumerate(stream):
                line_offset=offset
                offset+=len(line)
                if not line.strip():continue
                try:
                    row=json.loads(line.decode('utf-8-sig'))
                except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                    # The hash-verified published train file currently ends in
                    # an incomplete UTF-8/JSON record. Quarantine only that
                    # incomplete last line; never repair or invent its label.
                    if offset != spec['size'] or line.endswith(b'\n'):
                        raise
                    quarantined.append({'file':original,'original_row':line_index,
                        'byte_offset':line_offset,'bytes':len(line),'sha256':sha256(line),
                        'reason':'published_hash_verified_file_incomplete_terminal_record',
                        'error_type':type(exc).__name__,'error':str(exc)})
                    continue
                label=str(row['conflict_type']).strip()
                if not label:raise ValueError('empty original conflict_type')
                labels.add(label)
                counts[split][label]=counts[split].get(label,0)+1
                if not row.get('high_level_laws') or not str(row.get('title') or '').strip() or not str(row.get('content') or '').strip():
                    quarantined.append({'file':original,'original_row':line_index,'original_id':row.get('id'),
                        'byte_offset':line_offset,'bytes':len(line),'sha256':sha256(line),
                        'reason':'missing_original_query_text_title_or_reference','original_label':label})
                    continue
                upper,upper_family=references(row['high_level_laws'])
                title=str(row['title']).strip()
                text=str(row['content']).strip()
                if not title or not text:raise ValueError('empty original query title/content')
                prepared_counts[split]+=1
                record={'id':str(row['id']),'query':text,'reference':upper,'query_family':family(title),
                    'reference_family':upper_family,'conflict_type':label,'original_title':title,
                    'original_conflict_type_en':row.get('conflict_type_en'),
                    'original_row':line_index,'original_url':row.get('url')}
                if args.family_level=='provision':
                    record['expert_revised_content_for_family_only']=str(row.get('golden_content') or '')
                out.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
        normalized.append({'path':original,'format':'jsonl','sha256':sha256(target.read_bytes()),'split':split})
    family_audit={'level':'document'}
    if args.family_level=='provision':
        all_rows=[]
        for spec in normalized:
            all_rows.extend(json.loads(line) for line in (args.output/spec['path']).read_text(encoding='utf-8').splitlines())
        families,family_audit=provision_families(all_rows)
        position=0
        for spec in normalized:
            target=args.output/spec['path']
            rows=[json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
            for row in rows:
                row['query_family']=families[position]
                position+=1
            target.write_text('\n'.join(json.dumps(row,ensure_ascii=False,sort_keys=True) for row in rows)+'\n',encoding='utf-8')
            spec['sha256']=sha256(target.read_bytes())
    manifest={'dataset_name':'LCR-CN v4','task':'original_legislative_conflict_type_classification',
        'official_source':{'repository':'https://github.com/Zhaozhao77/LCR-CN',
            'revision':'c8672137493899accaaac63f6858d0364762ba18',
            'doi':'10.6084/m9.figshare.30315955.v4','metadata_sha256':sha256(meta_path.read_bytes())},
        'license':{'training_permitted':True,'name':licence['name'],'url':licence['url'],
            'attribution':metadata.get('citation')},
        'label_provenance':{'kind':'original_released_labels',
            'description':'Unmodified conflict_type labels released by the LCR-CN authors; no golden_content/distractor relabeling and no project-generated negatives.'},
        'fields':{'id':'id','left_text':'query','right_text':'reference','left_family':'query_family',
            'right_family':'reference_family','label':'conflict_type'},
        'label_mapping':{label:label for label in sorted(labels)},
        'files':normalized,'original_files':raw_hashes,'original_label_counts':counts,'prepared_pair_counts':prepared_counts,
        'quarantined_source_records':quarantined,
        'split_policy':'provided_disjoint_families','family_scope':'query_source_family',
        'family_definition':('Source regulation title, whitespace and parenthesized revision/year removed.' if args.family_level=='document'
            else 'Underlying provision families: exact normalized query/expert-revision matches globally; 5-character shingle Jaccard >=0.85 within anonymized source-title families. Expert revisions are used only for leakage exclusion.'),
        'family_grouping':family_audit,
        'original_split_policy':'Original splits retained; train/valid rows sharing held-out query families removed, never reassign original test rows into training.',
        'reference_corpus_scope':'Superior laws are shared legal reference material and can recur across splits; exact/reversed text pairs are deduplicated. This is not a claim of reference-law-disjoint evaluation.',
        'project_human_gold':False,'contract_law_entailment_labels':False,
        'limitations':'Auxiliary legislative conflict-type task, not contract-versus-statute entailment, not PROVDATA, not current-law validity verification or official full benchmark reproduction.'}
    path=args.output/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'prepared_original_labels','manifest':str(path),'counts':counts,
        'class_names':sorted(labels),'contract_law_entailment_labels':False},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
