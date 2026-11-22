"""Real HTTP multipart upload-to-archive timing; no mocked review.

Default starts a fresh isolated offline server, not the user's production service.
An existing model-enabled server may be selected explicitly with --base --llm on.
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
from collections import Counter
from pathlib import Path
from acceptance_common import ROOT, digest, multipart, provenance, request, require, summary, write_json

def compact_stats(stats):
    """Keep auditable model counts without copying enormous nested evidence."""
    calls=stats.get('model_calls',[])
    counts=Counter((str(c.get('component','')),str(c.get('model','')),str(c.get('operation','')),str(c.get('status','')),str(c.get('reason',''))) for c in calls if isinstance(c,dict))
    return {'full_stats_sha256':digest(json.dumps(stats,ensure_ascii=False,sort_keys=True,separators=(',',':'))),
        'full_stats_retained_in_archived_report':True,
        **{k:stats.get(k) for k in ('elapsed_seconds','clauses','blocks','candidates','contract_type','degraded','logic_llm','judge_llm')},
        'model_call_events':len(calls),'model_call_breakdown':[{'component':key[0],'model':key[1],'operation':key[2],'status':key[3],'reason':key[4],'count':n} for key,n in sorted(counts.items())]}

def text_for(chars):
    base='技术开发合同\n甲方：星海科技有限公司\n乙方：云图技术有限公司\n第一条 标的\n乙方为甲方开发信息系统。\n第二条 价款\n合同价款100000元，首期50000元，验收后50000元。\n第三条 期限\n期限自2026年1月1日至2026年12月31日。\n第四条 责任\n双方按约定承担违约责任，争议协商解决，协商不成向有管辖权人民法院起诉。\n'
    extra='第五条 履约说明\n双方应妥善保存项目验收材料并及时书面通知对方。\n'
    return (base+extra*max(0,(chars-len(base))//len(extra)+1))[:max(chars,len(base))]

def review(base,text,name,llm,engine):
    raw,ctype=multipart(name,text.encode(),{'store':'true','llm':llm,'engine':engine})
    status,payload,seconds=request(base,'/api/review',raw=raw,content_type=ctype)
    require(status,payload)
    assert payload.get('saved') and payload.get('report_id'),payload
    saved=require(*request(base,'/api/workspace/reports/'+payload['report_id'])[:2])
    assert saved['document']['full_text']==payload['document']['full_text']
    assert saved['trace']==payload['trace'] and saved['graph']==payload['graph']
    assert saved['document']['source_name']==name
    return payload,{'client_upload_to_response_seconds':seconds,'text_chars':len(text),'upload_bytes':len(raw),
        'text_sha256':digest(text),'report_id':payload['report_id'],
        'server_timings':payload.get('stats',{}).get('timings',{}),
        'stats':compact_stats(payload.get('stats',{})),
        'model_invocation_evidence':'see compact stats.model_call_breakdown and full archived stats.model_calls; not inferred from agent display labels'}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--base'); p.add_argument('--out',type=Path,default=ROOT/'tools/acceptance_results/http_offline.json')
    p.add_argument('--lengths',default='500,3000,10000');p.add_argument('--samples',type=int,default=5)
    p.add_argument('--llm',choices=['off','on'],default='off');p.add_argument('--engine',choices=['multi','single'],default='multi')
    p.add_argument('--queue-tasks',type=int,default=2);p.add_argument('--_serve',type=int);p.add_argument('--db',type=Path)
    a=p.parse_args()
    if a._serve is not None:
        from zhilv_qiling_v2 import api
        from zhilv_qiling_v2.config import Config
        api._SERVER_CONFIG=Config(store_path=str(a.db),enable_llm=False,ner_enable_model=False,ner_use_bert=False,use_lsa=False)
        server=api.serve('127.0.0.1',a._serve)
        try:server.serve_forever()
        finally:server.server_close()
        return 0
    if a.samples<1 or a.queue_tasks<1:p.error('samples and queue-tasks must be positive')
    a.out.parent.mkdir(parents=True,exist_ok=True);child=None;log=None
    result={'status':'running','synthetic_inputs':True,'reviews_executed':True,'accuracy_evaluated':False,'provenance':provenance(),
        'requested_configuration':{'engine':a.engine,'llm':a.llm},'length_groups':[],
        'timing_scope':'client timer begins immediately before multipart HTTP upload and ends after complete archived response; report reread measured separately by checks',
        'cold_definition':'fresh Python server process; OS file cache/GPU/model-service caches not cleared',
        'p95_caution':'small sample empirical percentile; not a production SLA or population estimate','errors':[]}
    try:
        if a.base:
            base=a.base;result['cold_start']={'status':'not_run','reason':'existing server process; cannot claim process cold start'}
        else:
            if a.llm=='on':p.error('use an explicitly configured --base for model acceptance')
            db=a.db or a.out.with_suffix('.sqlite')
            if db.exists():raise ValueError('isolated DB already exists; choose a new --out/--db')
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            log=a.out.with_suffix('.server.log').open('w',encoding='utf-8')
            started=time.perf_counter()
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--_serve',str(port),'--db',str(db)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            base=f'http://127.0.0.1:{port}'
            deadline=time.monotonic()+120
            while True:
                if child.poll() is not None:raise RuntimeError('isolated server exited; inspect server log')
                try:
                    status,health,_=request(base,'/api/health',timeout=2)
                    if status==200:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('server startup timeout')
                time.sleep(.1)
            result['cold_start']={'status':'completed','process_to_health_seconds':time.perf_counter()-started,'isolated_database':str(db)}
        health=require(*request(base,'/api/health')[:2]);result['base']=base
        result['configuration_observed']={'version':health.get('version'),'tag':health.get('tag'),'llm_enabled':health.get('llm',{}).get('enabled'),'llm_model':health.get('llm',{}).get('model'),'multi_agent':health.get('multi_agent')}
        if a.llm=='on' and not health.get('llm',{}).get('enabled'):raise AssertionError('model-enabled acceptance refused: health says LLM disabled')
        lengths=[int(x) for x in a.lengths.split(',')]
        _,cold=review(base,text_for(lengths[0]),'acceptance-cold.txt',a.llm,a.engine)
        result['first_request']=cold;result['first_request']['is_process_cold']=bool(child)
        for chars in lengths:
            records=[]
            for i in range(a.samples):
                _,record=review(base,text_for(chars),f'acceptance-{chars}-{i}.txt',a.llm,a.engine);records.append(record)
            result['length_groups'].append({'requested_chars':chars,'samples':records,'client_upload_to_archive':summary([x['client_upload_to_response_seconds'] for x in records]),
                'server_total':summary([x['server_timings']['server_total_seconds'] for x in records if 'server_total_seconds' in x['server_timings']])})
        listing=require(*request(base,'/api/workspace/reports?q=acceptance-&size=100')[:2]);assert listing['total']>=1
        report,record=review(base,text_for(500),'acceptance-feedback.txt',a.llm,a.engine)
        if report['risks']:
            risk=report['risks'][0]['risk_id'];path='/api/workspace/reports/'+report['report_id']
            require(*request(base,path+'/feedback',{'risk_id':risk,'status':'pending','note':'验收工具自动持久性检查；不是人工金标或实际人工复核结论'})[:2])
            restored=require(*request(base,path)[:2]);assert restored['feedback'][risk]['status']=='pending'
            result['feedback_persistence']='passed (pending only; no invented human confirmation)'
        batch=require(*request(base,'/api/workspace/batches',{'engine':a.engine,'llm':a.llm})[:2]);bid=batch['batch_id'];started=time.perf_counter()
        task_ids=[]
        for i in range(a.queue_tasks):
            raw,ctype=multipart(f'acceptance-queue-{i}.txt',text_for(500).encode())
            task=require(*request(base,f'/api/workspace/batches/{bid}/tasks',raw=raw,content_type=ctype)[:2]);task_ids.append(task['task_id'])
        require(*request(base,f'/api/workspace/batches/{bid}/start',{})[:2]);deadline=time.monotonic()+600
        while True:
            state=require(*request(base,f'/api/workspace/batches/{bid}?size=1000')[:2])
            if state['status'] in ('completed','failed','cancelled'):break
            if time.monotonic()>deadline:raise TimeoutError('persistent queue end-to-end timeout')
            time.sleep(.1)
        assert state['counts'].get('completed')==a.queue_tasks,state
        for item in state['items']:
            assert item['report_id']==item['task_id']
            require(*request(base,'/api/workspace/reports/'+item['report_id'])[:2])
        result['persistent_queue']={'status':'passed','batch_id':bid,'task_count':a.queue_tasks,'upload_through_completed_archive_seconds':time.perf_counter()-started,'counts':state['counts']}
        result['status']='passed'
        result['model_claim']='offline rules/retrieval/graph execution only' if a.llm=='off' else 'requested LLM; inspect per-report invocation evidence, enabled health alone does not prove successful generation'
    except Exception as exc:
        result['status']='failed';result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if child is not None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if log:log.close()
        write_json(a.out,result)
    print(json.dumps({'status':result['status'],'out':str(a.out),'errors':result['errors']},ensure_ascii=False));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
