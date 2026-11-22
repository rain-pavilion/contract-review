"""Read back original-label training evidence without loading any model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args()
    target=a.evidence/'verification.json'
    if target.exists():p.error('preserve existing verification')
    r=json.loads((a.evidence/'run.json').read_text(encoding='utf-8'))
    if r.get('status')!='trained_strict_reloaded_predictions_completed':
        p.error('training is not complete; read the existing run status instead')
    rows=[json.loads(line) for line in (a.evidence/'used_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    splits={split:[row for row in rows if row['derived_split']==split] for split in ('train','valid','test')}
    assert all(splits.values())
    families={s:{f for row in bucket for f in row['law_families']} for s,bucket in splits.items()}
    assert not any(families[x]&families[y] for x,y in (('train','valid'),('train','test'),('valid','test')))
    label_map=r['source']['manifest']['label_mapping']
    for split,bucket in splits.items():
        for row in bucket:
            assert row['original_split']==split
            assert row['label']==label_map[row['original_label']]
    for split in ('valid','test'):
        expected=[(row['pair_id'],row['original_id'],row['label']) for row in splits[split]]
        for moment in ('before','after'):
            actual=r[f'{moment}_{split}']['predictions']
            assert [(row['pair_id'],row['original_id'],row['expected']) for row in actual]==expected
    assert r['completed_optimizer_steps']==math.ceil(len(splits['train'])/r['batch_size'])*r['epochs']
    assert r['mamba_core_sha256_before']!=r['mamba_core_sha256_after']
    assert any(value>0 and name.startswith('mamba.') and any(x in name for x in ('A_log','.D','x_proj','dt_proj'))
        for name,value in r['parameter_changes'].items())
    assert r['strict_reload']['strict_checkpoint_load'] is True
    assert r['strict_reload_predictions_match_after_test'] is True
    assert r['compatible_with_contract_entailment_runtime'] is False
    assert r['product_default_replaced'] is False and r['training_labels_generated'] is False
    assert digest(Path(r['initial_checkpoint']['path'])/'model.pt')==r['initial_checkpoint']['sha256']
    assert digest(r['checkpoint']['path'])==r['checkpoint']['sha256']
    assert all(digest(ROOT/name)==value for name,value in r['implementation_sha256'].items())
    for file in r['source']['files']:
        assert digest(file['path'])==file['sha256']
    result={'status':'verified_actual_original_label_auxiliary_training',
        'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'run_sha256':digest(a.evidence/'run.json'),'checkpoint':r['checkpoint'],
        'used_pair_counts':{k:len(v) for k,v in splits.items()},
        'per_class':{k:dict(Counter(row['label'] for row in v)) for k,v in splits.items()},
        'completed_optimizer_steps':r['completed_optimizer_steps'],
        'supported_class_macro_f1_before':r['before_test']['macro_f1_supported_labels'],
        'supported_class_macro_f1_after':r['after_test']['macro_f1_supported_labels'],
        'test_accuracy_before':r['before_test']['accuracy'],'test_accuracy_after':r['after_test']['accuracy'],
        'law_provision_family_overlap':False,'shared_reference_corpus_disclosed':True,
        'original_split_and_labels_preserved':True,'same_heldout_before_after':True,
        'strict_reload_verified':True,'initial_checkpoint_unchanged':True,'implementation_hashes_match':True,
        'compatible_with_contract_entailment_runtime':False,'product_default_replaced':False,
        'official_full_benchmark_score':False,'independent_project_legal_accuracy_accepted':False,
        'scope':r['training_scope']}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
