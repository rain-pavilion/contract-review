"""Inspect original-label splits and exact local-tokenizer budgets, no model load."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from legal_training_data import prepare, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=Path('zhilv_qiling_v2/models/bert_mamba_entail'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('preserve existing inspection evidence')
    d=prepare(a.manifest,caps={'train':1200,'valid':128,'test':128})
    from tokenizers import Tokenizer
    tokenizer_file=a.tokenizer/'tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    result={'status':'inspected_original_labels_not_trained','manifest_sha256':d['manifest_sha256'],
        'counts':d['counts'],'law_family_overlap':d['law_family_overlap'],
        'family_grouping':d['manifest'].get('family_grouping'),
        'reference_family_overlap_allowed':d['reference_family_overlap_allowed'],
        'quarantined_source_records':d['manifest'].get('quarantined_source_records'),
        'rejected_reasons':dict(Counter(row['reason'] for row in d['rejected'])),
        'splits':{},'used_for_model_selection':False,
        'tokenizer_json_sha256':sha256(tokenizer_file.read_bytes()),
        'tokenizer_backend':'saved fast-tokenizer JSON; add_special_tokens=True; no truncation or padding'}
    for split,rows in d['splits'].items():
        lengths=[(row,len(tokenizer.encode(row['clause'],add_special_tokens=True).ids),
            len(tokenizer.encode(row['article'],add_special_tokens=True).ids)) for row in rows]
        summary={}
        for index,name in ((1,'query'),(2,'reference')):
            values=sorted(r[index] for r in lengths)
            summary[name]={'min':min(values),'p50':values[int((len(values)-1)*.5)],
                'p90':values[int((len(values)-1)*.9)],'p95':values[int((len(values)-1)*.95)],'max':max(values)}
        result['splits'][split]={'token_lengths':summary,'per_class':dict(Counter(r['label'] for r in rows)),
            'complete_rows_within_each_tower_budget':{str(budget):dict(Counter(row['label'] for row,l,r in lengths if max(l,r)<=budget))
                for budget in (128,256,384,512)}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
