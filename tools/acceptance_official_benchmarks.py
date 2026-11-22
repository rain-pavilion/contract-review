"""Manifest-driven adapter to LOCAL OFFICIAL benchmark evaluators.

Never maps project-generated examples to an official benchmark name. Evaluation
is not_run until official dataset, revision, protocol, tasks, predictions and
unmodified official evaluation script are supplied and hash-verified locally.
"""
from __future__ import annotations
import argparse, datetime, json, subprocess, sys
from pathlib import Path
from acceptance_common import ROOT, digest, write_json

def template(name):
    return {'schema_version':'official-benchmark-v1','benchmark':name,'status':'not_run',
        'protocol_status':'protocol_unverified','official_source':{'repository':None,'protocol_url':None,'verified_by':None,'verified_at':None},
        'candidate_source_for_manual_verification':'https://github.com/open-compass/LawBench' if name=='LawBench' else None,
        'checkout':None,'revision':None,'dataset_files':[],'prediction_files':[],'task_mapping':[],
        'model':{'name':None,'revision_or_weights_sha256':None,'configuration_file':None,'configuration_sha256':None},
        'evaluation':{'script':None,'script_sha256':None,'arguments':[],'metrics_file':None},
        'requirements':['Identify the exact official benchmark. ProvBench name alone is insufficient to identify a publication/repository.',
            'Verify the primary repository/paper and official evaluation protocol; record URL, verifier and time.',
            'Provide pinned local official checkout and git revision; preserve official dataset split/task IDs.',
            'Provide every official dataset file and prediction file with SHA-256, task ID, split, expected item count.',
            'Provide explicit supported and unsupported official tasks. Do not substitute project-owned task names or data.',
            'Generate predictions with a declared model revision/weight hash and configuration hash.',
            'Invoke the unchanged, hash-pinned official Python evaluator. The adapter does not invent metrics.'],
        'limitations':'This manifest is an integration entry point, not a completed formal benchmark or protocol verification.'}

def check(m):
    errors=[]
    if m.get('protocol_status')!='verified_primary_source':errors.append('protocol_unverified: official primary protocol has not been verified')
    source=m.get('official_source',{})
    if not all(source.get(x) for x in ('repository','protocol_url','verified_by','verified_at')):errors.append('official source provenance incomplete')
    root=Path(m['checkout']).resolve() if m.get('checkout') else None
    if not root or not root.is_dir():errors.append('local official checkout missing');return errors
    if m.get('checkout_kind')=='official_commit_sparse_source_snapshot':
        try:
            from acceptance_lawbench import source
            meta=source(root)
            if meta['revision']!=m.get('revision'):errors.append('official snapshot revision mismatch')
            snapshot_manifest=root/'official_snapshot.json'
            if digest(snapshot_manifest.read_bytes())!=m.get('source_snapshot_sha256'):errors.append('official snapshot manifest hash mismatch')
        except (OSError,ValueError,KeyError) as exc:errors.append('official source snapshot integrity failed: '+str(exc))
    else:
        try:
            revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
            if revision!=m.get('revision'):errors.append('official checkout revision mismatch')
            dirty=subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True).stdout
            if dirty.strip():errors.append('official checkout has modified tracked files')
        except (OSError,subprocess.CalledProcessError):errors.append('cannot establish official git revision')
    mapping=m.get('task_mapping',[])
    if not mapping:errors.append('official task mapping missing')
    task_ids={x.get('official_task_id') for x in mapping if x.get('supported') is True}
    if not task_ids:errors.append('no official supported task')
    for kind in ('dataset_files','prediction_files'):
        files=m.get(kind,[])
        if not files:errors.append(kind+' missing')
        for f in files:
            path=Path(f.get('path',''));path=path if path.is_absolute() else root/path
            if not path.is_file():errors.append(kind+' file missing: '+str(path));continue
            if digest(path.read_bytes())!=f.get('sha256'):errors.append(kind+' hash mismatch: '+str(path))
            if f.get('official_task_id') not in task_ids or not f.get('official_split') or not isinstance(f.get('item_count'),int) or f['item_count']<1:
                errors.append(kind+' official task/split/item_count incomplete')
    model=m.get('model',{})
    if not model.get('name') or not model.get('revision_or_weights_sha256'):errors.append('model identity/revision missing')
    cp=Path(model.get('configuration_file') or '')
    if not cp.is_file() or digest(cp.read_bytes())!=model.get('configuration_sha256'):errors.append('model configuration file/hash missing or mismatched')
    ev=m.get('evaluation',{});script=(root/(ev.get('script') or '')).resolve()
    if not script.is_relative_to(root) or not script.is_file() or script.suffix!='.py':errors.append('official Python evaluator script missing/outside checkout')
    elif digest(script.read_bytes())!=ev.get('script_sha256'):errors.append('official evaluator hash mismatch')
    if not isinstance(ev.get('arguments'),list) or not all(isinstance(x,str) for x in ev.get('arguments',[])):errors.append('evaluator arguments must be an explicit string array')
    if not ev.get('metrics_file'):errors.append('official metrics output path missing')
    return errors

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','check','run']);p.add_argument('--benchmark',choices=['LawBench','ProvBench']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
    if a.action=='init':
        if not a.benchmark:p.error('init requires --benchmark')
        if a.manifest.exists():p.error('refuse overwrite existing manifest')
        write_json(a.manifest,template(a.benchmark));print('not_run: local official inputs/protocol required');return 0
    m=json.loads(a.manifest.read_text(encoding='utf-8-sig'));errors=check(m)
    result={'benchmark':m.get('benchmark'),'status':'not_run' if errors else 'ready','protocol_status':m.get('protocol_status'),
        'manifest_sha256':digest(a.manifest.read_bytes()),'errors':errors,'formal_metrics':None,
        'project_custom_evaluation_is_official':False}
    if a.action=='run' and not errors:
        root=Path(m['checkout']).resolve();ev=m['evaluation'];script=(root/ev['script']).resolve();metric=(root/ev['metrics_file']).resolve()
        if metric.exists():result.update({'status':'not_run','errors':['metrics_file already exists; select a fresh output to avoid stale-score attribution']})
        else:
            try:
                cwd=(root/ev.get('working_directory','.')).resolve()
                if not cwd.is_relative_to(root):raise ValueError('official evaluator cwd must remain in source checkout')
                run=subprocess.run([sys.executable,str(script),*ev['arguments']],cwd=cwd,capture_output=True,text=True,timeout=a.timeout)
                result.update({'evaluator_exit_code':run.returncode,'stdout_tail':run.stdout[-4000:],'stderr_tail':run.stderr[-4000:]})
                if run.returncode or not metric.is_file():result['status']='failed';result['errors'].append('official evaluator failed or no metrics output created')
                else:result.update({'status':'completed_official_evaluator','official_metrics_file':str(metric),'official_metrics_sha256':digest(metric.read_bytes()),'formal_metrics':'see unchanged official output file; adapter does not reinterpret scores'})
            except (OSError,subprocess.TimeoutExpired) as exc:result.update({'status':'failed','errors':[str(exc)]})
    write_json(a.out or a.manifest.with_suffix('.result.json'),result);print(json.dumps(result,ensure_ascii=False));return 0 if result['status'] in ('ready','completed_official_evaluator') else 2
if __name__=='__main__':raise SystemExit(main())
