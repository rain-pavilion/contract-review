"""Independent human-label workflow. Never generate gold using project rules.

init creates UNLABELED templates; validate/freeze require real independent reviews.
freeze records deterministic group splits and hashes, refusing train/test leakage.
"""
from __future__ import annotations
import argparse, datetime, json, random, re, sys
from pathlib import Path
from acceptance_common import digest, write_json

SCHEMA='zlql-human-gold-v1'
def normalized(text): return re.sub(r'\s+', '', text).lower()
def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def validate(rows, training=None, require_gold=False):
    errors=[]; ids=set(); normalized_seen={}; groups={}; gold=0
    train_hashes=set(); train_groups=set()
    if training:
        inventory=read_rows(training)
        if not inventory: errors.append('training inventory is empty; no leakage clearance can be issued')
        for x in inventory:
            train_hashes.add(digest(normalized(x['text']))); train_groups.add(x['source']['group_id'])
    for i,row in enumerate(rows,1):
        prefix=f'line {i}: '
        missing=[k for k in ('schema_version','id','text','text_sha256','source','status','split','annotations','reviews') if k not in row]
        if missing: errors.append(prefix+'missing '+','.join(missing)); continue
        if row['schema_version']!=SCHEMA: errors.append(prefix+'wrong schema_version')
        if row['id'] in ids: errors.append(prefix+'duplicate id')
        ids.add(row['id']); text=row['text']; h=digest(normalized(text))
        if row['text_sha256']!=digest(text): errors.append(prefix+'source text hash changed')
        source=row['source']; group=source.get('group_id')
        if not all(source.get(k) for k in ('origin','group_id','license_or_permission','collected_at')):
            errors.append(prefix+'source provenance/permission/group/date required')
        split=row['split']
        if split not in ('unassigned','train','dev','test'): errors.append(prefix+'invalid split')
        if h in normalized_seen and normalized_seen[h]!=split: errors.append(prefix+'normalized duplicate across splits')
        normalized_seen[h]=split
        if group in groups and groups[group]!=split: errors.append(prefix+'same source group across splits')
        groups[group]=split
        if split=='test' and (h in train_hashes or group in train_groups): errors.append(prefix+'test leakage against training inventory')
        status=row['status']
        if status not in ('unlabeled','in_review','adjudicated'): errors.append(prefix+'invalid status')
        if status=='unlabeled' and (row['annotations'] or row['reviews']): errors.append(prefix+'unlabeled rows must not contain labels/reviews')
        if status=='adjudicated':
            gold+=1
            reviews=row['reviews']; people={r.get('reviewer_id') for r in reviews if r.get('reviewer_id')}
            if len(people)<2: errors.append(prefix+'two distinct real human reviewers required')
            for review in reviews:
                if review.get('method')!='manual' or not all(review.get(k) for k in ('reviewer_id','reviewed_at','decision','annotations_sha256')):
                    errors.append(prefix+'manual review audit fields required')
                if review.get('used_system_predictions') is not False:
                    errors.append(prefix+'blind review must record used_system_predictions=false')
                if not isinstance(review.get('annotations'),list):
                    errors.append(prefix+'each reviewer must retain their independent annotation list')
                else:
                    rh=digest(json.dumps(review['annotations'],ensure_ascii=False,sort_keys=True,separators=(',',':')))
                    if review.get('annotations_sha256')!=rh: errors.append(prefix+'independent reviewer annotation hash mismatch')
            adjudication=row.get('adjudication',{})
            if not all(adjudication.get(k) for k in ('adjudicator_id','adjudicated_at','rationale')): errors.append(prefix+'adjudication audit required')
            ah=digest(json.dumps(row['annotations'],ensure_ascii=False,sort_keys=True,separators=(',',':')))
            if adjudication.get('annotations_sha256')!=ah: errors.append(prefix+'adjudicated annotation hash mismatch')
        elif require_gold: errors.append(prefix+'not adjudicated; cannot freeze as gold')
        for ann in row['annotations']:
            if ann.get('type') not in ('entity','risk','law_reference'): errors.append(prefix+'invalid annotation type')
            if not all(k in ann for k in ('start','end','label','evidence')): errors.append(prefix+'annotation span/label/evidence required'); continue
            a,b=ann['start'],ann['end']
            if not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=len(text): errors.append(prefix+'invalid character span')
            elif text[a:b]!=ann['evidence']: errors.append(prefix+'evidence does not match source span')
        if require_gold and split=='unassigned': errors.append(prefix+'split is not frozen')
    return {'status':'passed' if not errors else 'failed','records':len(rows),'adjudicated_records':gold,
            'unlabeled_records':sum(x.get('status')=='unlabeled' for x in rows),'errors':errors,
            'gold_ready':bool(rows) and gold==len(rows) and not errors,
            'training_inventory_checked':bool(training),
            'scope':'schema/audit/exact-normalized-duplicate/group leakage checks; cannot establish human identity or detect all paraphrase leakage'}

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('init'); q.add_argument('--input-dir',type=Path,required=True); q.add_argument('--out',type=Path,required=True); q.add_argument('--permission',required=True); q.add_argument('--origin',required=True)
    for action in ('validate','freeze'):
        q=sub.add_parser(action); q.add_argument('--data',type=Path,required=True); q.add_argument('--training-inventory',type=Path); q.add_argument('--out',type=Path,required=True)
        if action=='freeze': q.add_argument('--seed',type=int,default=20261114)
    a=p.parse_args()
    if a.action=='init':
        rows=[]
        for file in sorted(a.input_dir.rglob('*.txt')):
            text=file.read_text(encoding='utf-8-sig'); relative=file.relative_to(a.input_dir).as_posix()
            rows.append({'schema_version':SCHEMA,'id':digest(relative+text)[:24],'text':text,'text_sha256':digest(text),
                'source':{'origin':a.origin,'path':relative,'group_id':relative.split('/')[0] if '/' in relative else file.stem,
                'license_or_permission':a.permission,'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},
                'status':'unlabeled','split':'unassigned','annotations':[],'reviews':[]})
        a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
        print(json.dumps({'status':'unlabeled','records':len(rows),'gold_created':False},ensure_ascii=False)); return 0
    rows=read_rows(a.data)
    if a.action=='freeze':
        if not a.training_inventory: p.error('freeze requires --training-inventory (complete training/development source inventory)')
        groups=sorted({x['source']['group_id'] for x in rows}); random.Random(a.seed).shuffle(groups)
        assignment={group:('test' if i%10<2 else 'dev' if i%10==2 else 'train') for i,group in enumerate(groups)}
        for row in rows: row['split']=assignment[row['source']['group_id']]
    result=validate(rows,a.training_inventory,require_gold=a.action=='freeze')
    if a.action=='freeze' and result['gold_ready']:
        frozen=a.out.with_suffix('.jsonl'); frozen.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
        result.update({'frozen_file':str(frozen),'sha256':digest(frozen.read_bytes()),'seed':a.seed,'schema_version':SCHEMA,
                       'training_inventory_sha256':digest(a.training_inventory.read_bytes()),'immutable_policy':'do not overwrite; version a new dataset after any edit'})
    elif a.action=='freeze': result['status']='not_run'; result['reason']='No frozen gold created: incomplete human annotation or leakage validation failed'
    write_json(a.out,result); print(json.dumps(result,ensure_ascii=False)); return 0 if result['status']=='passed' else 2
if __name__=='__main__': raise SystemExit(main())
