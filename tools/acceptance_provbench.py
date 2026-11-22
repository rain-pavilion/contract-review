"""ProvBench paper/source verification and strict local evaluation adapter.

The paper is ACL 2025 2025.acl-long.312. Its actual code footnote is labpaper/
ProvBench. Repository availability is separate from benchmark identification.
Paper-metric reimplementation never claims to be the unavailable official code.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, subprocess, sys, urllib.error, urllib.request
from html.parser import HTMLParser
from pathlib import Path
from acceptance_common import ROOT, digest, provenance, write_json

PAPER='https://aclanthology.org/2025.acl-long.312/'
PDF='https://aclanthology.org/2025.acl-long.312.pdf'
REPO='https://github.com/labpaper/ProvBench'
SCHEMA='provbench-paper-adapter-v1'
LABELS=('entailment','contradiction')

class Metadata(HTMLParser):
    def __init__(self):super().__init__();self.values={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='meta' and a.get('name','').startswith('citation_'):
            self.values.setdefault(a['name'],[]).append(a.get('content',''))

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ZLQL-ProvBench-primary-protocol-verification'})
    try:
        with urllib.request.urlopen(req,timeout=25) as response:return {'url':url,'http_status':response.status,'final_url':response.url},response.read()
    except urllib.error.HTTPError as exc:return {'url':url,'http_status':exc.code,'error':str(exc)},b''
    except OSError as exc:return {'url':url,'http_status':None,'error':f'{type(exc).__name__}: {exc}'},b''

def probe(a):
    folder=a.sources;folder.mkdir(parents=True,exist_ok=True)
    page_status,page=fetch(PAPER);pdf_status,pdf=fetch(PDF)
    parser=Metadata();parser.feed(page.decode('utf-8','replace'));links=set();pages=None;identity=False;evidence=[]
    if page:(folder/'paper.html').write_bytes(page)
    if pdf:
        (folder/'paper.pdf').write_bytes(pdf)
        import fitz
        doc=fitz.open(stream=pdf,filetype='pdf');pages=len(doc)
        full_text='\n'.join(p.get_text() for p in doc)
        for number,p in enumerate(doc,1):
            text=p.get_text()
            topics=[]
            if number==1:topics.append('title and actual code footnote')
            if 'Evaluation Metrics' in text:topics.append('section 5.2: hit R@k, exact gold set, gold-top3 conflicts, overall F1 without averaging')
            if '2,698' in text and '448' in text:topics.append('Table 2: clause and relation split counts')
            if '0 means there is no semantic' in text:topics.append('Appendix C: numeric binary output mapping')
            if topics:evidence.append({'pdf_page_1_based':number,'printed_footer':text.strip().splitlines()[-1],
                'topics':topics,'extracted_page_text_sha256':digest(text)})
            for link in p.get_links():
                url=link.get('uri','')
                if 'github.com' in url:links.add(url.rstrip('/'))
        identity='ProvBench' in full_text and REPO in links
        # Store only a short protocol index, not a substituted dataset or labels.
        write_json(folder/'paper_index.json',{'pages':pages,'title':parser.values.get('citation_title'),
            'github_footnote_urls':sorted(links),'contains_recall_and_top3_em':'Top-3 EM' in full_text,
            'contains_mrr':'MRR' in full_text,'contains_ndcg':'NDCG' in full_text,'protocol_evidence_pages':evidence})
    availability=[]
    for url in (REPO,REPO.replace('github.com/','raw.githubusercontent.com/')+'/main/README.md',REPO.replace('github.com/','raw.githubusercontent.com/')+'/master/README.md'):
        status,content=fetch(url);availability.append(status)
        if status['http_status']==200 and url.endswith('README.md'):(folder/('README_'+('main' if '/main/' in url else 'master')+'.md')).write_bytes(content)
    title=(parser.values.get('citation_title') or [None])[0]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source={'checked_at':now,'paper_page':page_status,'paper_pdf':pdf_status,'title':title,
        'authors':parser.values.get('citation_author',[]),'doi':(parser.values.get('citation_doi') or [None])[0],
        'paper_page_sha256':digest(page) if page else None,'paper_pdf_sha256':digest(pdf) if pdf else None,
        'pdf_pages':pages,'actual_pdf_code_links':sorted(links),'identity_verified':identity,'protocol_evidence_pages':evidence,
        'repository':REPO,'repository_checks':availability,
        'availability_interpretation':'HTTP status describes public access at this check only; 404 does not distinguish private, removed or unpublished repository',
        'incorrect_candidate_repository':'https://github.com/LittleFishG/ProvBench (not the official paper footnote)',
        'source_extraction':'ACL citation metadata + all PDF pages + PDF URI annotations; no user-supplied title accepted without verification'}
    write_json(folder/'source_check.json',source)
    manifest={'schema_version':SCHEMA,'benchmark':'ProvBench','status':'not_run',
        'identity_status':'verified_acl_primary_paper' if identity else 'not_verified',
        'protocol_status':'paper_protocol_verified_official_release_unavailable' if identity else 'protocol_unverified',
        'official_source':{'paper':PAPER,'pdf':PDF,'title':title,'doi':source['doi'],'repository_from_pdf_footnote':REPO,
            'source_check':str((folder/'source_check.json').resolve()),'source_check_sha256':digest((folder/'source_check.json').read_bytes())},
        'availability':{'checked_at':now,'repository_checks':availability,'official_data_downloaded':False,'official_evaluator_downloaded':False},
        'paper_protocol':{'paper_sections':['2','3.3','5.2'],'dataset_name':'PROVDATA','clauses':3550,'candidate_provisions':663,
            'split_clause_counts':{'train':2698,'validation':404,'test':448},'gold_provisions_per_clause':3,
            'table2_validation_consistency_warning':'Table 2 reports Val contradiction 589 + entailment 704 = 1293, whereas 404 clauses x 3 = 1212; preserve published values and resolve against authentic released files, never silently repair',
            'binary_label_mapping':{'0':'entailment','1':'contradiction'},
            'recommendation_metrics':{'R@1':'fraction of clauses with at least one relevant provision in top 1',
                'R@3':'fraction of clauses with at least one relevant provision in top 3','R@5':'fraction of clauses with at least one relevant provision in top 5',
                'Top-3 EM':'set equality between the top-three recommendation and the gold set'},
            'conflict_protocol':'evaluate the three gold-standard provision pairs per clause; not the model-retrieved top-three',
            'conflict_metrics':['entailment class accuracy','contradiction class accuracy','overall F1'],
            'f1_averaging_from_official_code':None,'MRR_NDCG_are_official_metrics':False,
            'metric_source_location':'section 5.2, printed page 6245 (PDF page 6); Table 2 at printed 6243 (PDF page 4); binary mapping Appendix C printed 6254 (PDF page 15)'},
        'local_inputs':{'normalized_gold_jsonl':None,'normalized_candidate_corpus_jsonl':None,'predictions_jsonl':None,
            'raw_official_files':[],'import_mapping_script':None,'import_mapping_script_sha256':None,
            'split_mapping_verified':False,'model':None,'model_revision_or_weights_sha256':None,'model_configuration_sha256':None},
        'official_execution':{'checkout':None,'revision':None,'script':None,'script_sha256':None,'working_directory':None,'arguments':[],'metrics_file':None},
        'formal_metrics':None,'missing_requirements':['publicly accessible official repository/data release or a supplied authentic local official checkout',
            'pinned raw PROVDATA files + the 663-item candidate library + unchanged official split IDs',
            'released schema and exact import mapping, both hash-pinned','actual model predictions, model revision/weights hash and configuration',
            'official evaluator entry point/version; resolve F1 averaging from official code'],
        'adapter_scope':'paper-defined diagnostic metrics available locally; never reported as official-code scores',
        'provenance':provenance()}
    write_json(a.manifest,manifest)
    return {'status':'paper_verified_release_not_run' if identity else 'verification_incomplete','manifest':str(a.manifest),
        'title':title,'actual_pdf_code_links':sorted(links),'repository_http_statuses':[x['http_status'] for x in availability],'formal_metrics':None}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def validate_inputs(gold,predictions,corpus,*,fixture=False):
    if not gold or not predictions or not corpus:raise ValueError('nonempty gold, predictions and candidate corpus are required')
    cids=[x['provision_id'] for x in corpus]
    if len(cids)!=len(set(cids)):raise ValueError('duplicate provision IDs')
    allowed=set(cids);gold_ids=[x['id'] for x in gold];pred_ids=[x['id'] for x in predictions]
    if len(gold_ids)!=len(set(gold_ids)) or len(pred_ids)!=len(set(pred_ids)):raise ValueError('duplicate clause IDs')
    if set(gold_ids)!=set(pred_ids):raise ValueError('prediction IDs must exactly cover selected official/fixture rows')
    if not fixture and (len(corpus)!=663 or len(gold)!=448):raise ValueError('full paper test protocol requires 663 candidates and 448 test clauses; subsets must be explicitly declared fixture/diagnostic')
    pred_by_id={x['id']:x for x in predictions}
    for row in gold:
        if row.get('official_split')!='test':raise ValueError('only the unchanged test split can be evaluated here')
        pair=row.get('gold_provisions',[]);gids=[x['provision_id'] for x in pair]
        if len(gids)!=3 or len(set(gids))!=3 or not set(gids)<=allowed:raise ValueError('three unique gold provisions from candidate corpus required')
        if any(x.get('label') not in LABELS for x in pair):raise ValueError('labels must be entailment/contradiction, not project risk categories')
        pred=pred_by_id[row['id']];rank=pred.get('ranked_provision_ids',[])
        if len(rank)<5 or len(rank)!=len(set(rank)) or not set(rank)<=allowed:raise ValueError('ranked list needs >=5 unique candidate IDs')
        pairs=pred.get('conflicts_on_gold',{})
        if set(pairs)!=set(gids) or any(label not in LABELS for label in pairs.values()):raise ValueError('conflict predictions must cover GOLD top-three pairs, independent of retrieved ranking')

def metrics(gold,predictions,corpus,*,fixture=False):
    validate_inputs(gold,predictions,corpus,fixture=fixture);by_id={x['id']:x for x in predictions}
    hits={1:0,3:0,5:0};em=0;cm={a:{b:0 for b in LABELS} for a in LABELS}
    for row in gold:
        p=by_id[row['id']];truth={x['provision_id'] for x in row['gold_provisions']};rank=p['ranked_provision_ids']
        for k in hits:hits[k]+=bool(truth.intersection(rank[:k]))
        em+=set(rank[:3])==truth
        for pair in row['gold_provisions']:cm[pair['label']][p['conflicts_on_gold'][pair['provision_id']]]+=1
    per_class={};f1=[];correct=0
    for label in LABELS:
        other=next(x for x in LABELS if x!=label);tp=cm[label][label];fn=cm[label][other];fp=cm[other][label]
        support=tp+fn;score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        per_class[label]={'accuracy_on_gold_class':tp/support if support else None,'support':support,'one_vs_rest_f1':score};f1.append(score);correct+=tp
    return {'recommendation':{**{'R@'+str(k):v/len(gold) for k,v in hits.items()},'Top-3 EM':em/len(gold)},
        'conflict_on_gold_top3':{'confusion_matrix':cm,'per_class':per_class,'macro_f1_diagnostic':sum(f1)/2,
            'micro_f1_diagnostic':correct/(3*len(gold)),'contradiction_positive_f1_diagnostic':per_class['contradiction']['one_vs_rest_f1'],
            'official_overall_F1':None,'f1_note':'paper does not specify averaging in the inspected text; official code unavailable, so no variant is silently designated official'},
        'clauses':len(gold),'gold_pairs':3*len(gold)}

def fixture(a):
    if a.fixture_dir.exists():raise ValueError('fixture directory must be fresh')
    a.fixture_dir.mkdir(parents=True)
    corpus=[{'provision_id':x,'text':'SYNTHETIC METRIC FIXTURE '+x} for x in 'ABCDE']
    gold=[{'id':'fixture-1','official_split':'test','clause':'SYNTHETIC fixture one','gold_provisions':[{'provision_id':'A','label':'entailment'},{'provision_id':'B','label':'contradiction'},{'provision_id':'C','label':'entailment'}]},
          {'id':'fixture-2','official_split':'test','clause':'SYNTHETIC fixture two','gold_provisions':[{'provision_id':'B','label':'contradiction'},{'provision_id':'D','label':'contradiction'},{'provision_id':'E','label':'entailment'}]}]
    pred=[{'id':'fixture-1','ranked_provision_ids':list('DABCE'),'conflicts_on_gold':{'A':'entailment','B':'entailment','C':'entailment'}},
          {'id':'fixture-2','ranked_provision_ids':list('EDBAC'),'conflicts_on_gold':{'B':'contradiction','D':'contradiction','E':'entailment'}}]
    for name,rows in [('corpus',corpus),('gold',gold),('predictions',pred)]:
        (a.fixture_dir/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    values=metrics(gold,pred,corpus,fixture=True)
    assert values['recommendation']=={'R@1':.5,'R@3':1.,'R@5':1.,'Top-3 EM':.5}
    assert values['conflict_on_gold_top3']['contradiction_positive_f1_diagnostic']==.8
    result={'status':'passed','benchmark':'ProvBench','scope':'paper_metric_reimplementation_synthetic_fixture_only','fixture_only':True,
        'official_data':False,'official_evaluator_executed':False,'formal_score':False,'model_calls':0,'metrics':values,'provenance':provenance()}
    write_json(a.out,result);return result

def check_manifest(m):
    errors=[];inputs=m.get('local_inputs',{})
    if m.get('identity_status')!='verified_acl_primary_paper':errors.append('ACL primary paper identity not verified')
    if not inputs.get('split_mapping_verified'):errors.append('official original split mapping not verified')
    for key in ('normalized_gold_jsonl','normalized_candidate_corpus_jsonl','predictions_jsonl'):
        if not inputs.get(key) or not Path(inputs[key]).is_file():errors.append(key+' missing')
    for key in ('model','model_revision_or_weights_sha256','model_configuration_sha256'):
        if not inputs.get(key):errors.append(key+' missing')
    raw=inputs.get('raw_official_files',[])
    if not raw:errors.append('hash-pinned raw official PROVDATA/candidate corpus files missing')
    for item in raw:
        p=Path(item.get('path',''))
        if p.name=='.env' or not p.is_file():errors.append('raw official file missing or disallowed');continue
        if digest(p.read_bytes())!=item.get('sha256'):errors.append('raw official file hash mismatch')
    mapping=inputs.get('import_mapping_script')
    if not mapping or not Path(mapping).is_file():errors.append('released-schema import mapping script missing')
    elif digest(Path(mapping).read_bytes())!=inputs.get('import_mapping_script_sha256'):errors.append('import mapping hash mismatch')
    return errors

def evaluate(a):
    m=json.loads(a.manifest.read_text(encoding='utf-8-sig'));errors=check_manifest(m)
    if errors:
        result={'benchmark':'ProvBench','status':'not_run','identity_status':m.get('identity_status'),'protocol_status':m.get('protocol_status'),
            'errors':errors,'formal_metrics':None,'reason':'official paper identified; authentic released files/model outputs/import mapping not yet supplied'}
    else:
        x=m['local_inputs'];values=metrics(read_jsonl(x['normalized_gold_jsonl']),read_jsonl(x['predictions_jsonl']),read_jsonl(x['normalized_candidate_corpus_jsonl']))
        result={'benchmark':'ProvBench','status':'paper_metric_diagnostics_computed','official_evaluator_executed':False,'formal_score':False,
            'metrics':values,'manifest_sha256':digest(a.manifest.read_bytes()),'scope':'paper-metric reimplementation; not official-source-code evaluation'}
    write_json(a.out,result);return result

def run_official(a):
    m=json.loads(a.manifest.read_text(encoding='utf-8-sig'));errors=check_manifest(m);e=m.get('official_execution',{});root=Path(e['checkout']).resolve() if e.get('checkout') else None
    if not root or not root.is_dir():errors.append('official checkout unavailable')
    elif root:
        try:
            origin=subprocess.run(['git','-C',str(root),'remote','get-url','origin'],capture_output=True,text=True,check=True).stdout.strip().rstrip('/').removesuffix('.git')
            revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
            dirty=subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True).stdout.strip()
            if origin.lower()!=REPO.lower() or revision!=e.get('revision') or dirty:errors.append('official origin/revision/unmodified checkout validation failed')
        except (OSError,subprocess.CalledProcessError):errors.append('could not verify official Git checkout')
        script=(root/(e.get('script') or '')).resolve();cwd=(root/(e.get('working_directory') or '.')).resolve()
        if not script.is_relative_to(root) or not script.is_file() or script.suffix!='.py':errors.append('official Python evaluator script missing/outside checkout')
        elif digest(script.read_bytes())!=e.get('script_sha256'):errors.append('official evaluator SHA-256 mismatch')
        if not cwd.is_relative_to(root) or not cwd.is_dir():errors.append('invalid official working directory')
    if not e.get('metrics_file'):errors.append('fresh official metrics output path missing')
    if errors:result={'status':'not_run','benchmark':'ProvBench','errors':errors,'formal_metrics':None,'official_evaluator_executed':False}
    else:
        output=Path(e['metrics_file']);output=output if output.is_absolute() else root/output
        if output.exists():raise ValueError('refuse existing/stale official metrics output')
        executed=subprocess.run([str(a.python.resolve()),str(script),*e.get('arguments',[])],cwd=cwd,capture_output=True,text=True,timeout=600)
        result={'benchmark':'ProvBench','status':'official_evaluator_executed' if executed.returncode==0 and output.is_file() else 'failed',
            'official_evaluator_executed':True,'exit_code':executed.returncode,'stdout_tail':executed.stdout[-3000:],'stderr_tail':executed.stderr[-3000:],
            'official_metrics_file':str(output),'official_metrics_sha256':digest(output.read_bytes()) if output.is_file() else None}
    write_json(a.out,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['probe','fixture','check','evaluate','run-official'])
    p.add_argument('--manifest',type=Path,default=ROOT/'docs/验收/ProvBench.manifest.json');p.add_argument('--sources',type=Path,default=ROOT/'docs/验收/ProvBench来源核验')
    p.add_argument('--fixture-dir',type=Path,default=ROOT/'tools/acceptance_results/provbench_fixture');p.add_argument('--out',type=Path,default=ROOT/'tools/acceptance_results/ProvBench.not_run.json')
    p.add_argument('--python',type=Path,default=ROOT/'.venv/Scripts/python.exe');a=p.parse_args()
    if a.action=='fixture' and a.out==ROOT/'tools/acceptance_results/ProvBench.not_run.json':a.out=ROOT/'tools/acceptance_results/provbench_fixture.result.json'
    if a.action=='probe':result=probe(a)
    elif a.action=='fixture':result=fixture(a)
    elif a.action=='run-official':result=run_official(a)
    else:result=evaluate(a)
    print(json.dumps(result,ensure_ascii=False));return 0 if result['status'] in ('passed','paper_verified_release_not_run','paper_metric_diagnostics_computed','official_evaluator_executed') else 2
if __name__=='__main__':raise SystemExit(main())
