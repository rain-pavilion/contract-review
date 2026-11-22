"""Synthetic full-workspace archive and persistent-queue capacity acceptance.

No ReviewEngine, model, extraction or real contract review is executed. Archive
and queue use different fresh SQLite files; no live worker sees synthetic tasks.
"""
from __future__ import annotations
import argparse, datetime, json, random, sqlite3, threading, time
from contextlib import closing
from pathlib import Path
from acceptance_common import ROOT, digest, provenance, request, require, summary, write_json

def payload(i):
    chars=(512 if i%100<80 else 2048 if i%100<98 else 8192)
    text=(f'合成存储样本 {i}，不是实际合同，不含模型审查结论。\n'+'合成正文用于验证完整原文、分段、图结构和执行记录的往返一致性。'*300)[:chars]
    key=f'{i:032x}'; clause='synthetic-c1'
    return {'schema_version':'synthetic-storage-v1','synthetic':True,'review_executed':False,
        'contract_id':'synthetic-'+str(i),'source_name':f'synthetic-{i:06d}.txt','score':0,'risk_count':0,'risks':[],
        'category_counts':{},'severity_counts':{},'review_queue':[],'conflicts':[],
        'trace':[{'node':'synthetic_storage_fixture','status':'not_run','message':'No risk review/model executed','elapsed_seconds':0}],
        'stats':{'synthetic':True,'review_executed':False,'text_chars':len(text)},
        'created_at':'2026-11-14T00:00:00+00:00','contract_type':'合成存储验收',
        'document':{'source_name':f'synthetic-{i:06d}.txt','file_type':'txt','full_text':text,
            'blocks':[{'block_id':'synthetic-b1','text':text,'page':1,'kind':'paragraph'}],
            'clauses':[{'clause_id':clause,'text':text,'title':'合成样本','block_ids':['synthetic-b1']}],
            'stats':{'synthetic':True,'chars':len(text)}},
        'elements':{'parties':[],'amounts':[],'dates':[]},
        'graph':{'nodes':[{'id':clause,'kind':'clause','text':text[:100]}],'edges':[]},
        'subject_checks':{'status':'not_run','verification_scope':'synthetic; no subject checked'},
        'report_id':key,'saved':True,'feedback':{}},key

def run(a):
    from zhilv_qiling_v2 import api
    from zhilv_qiling_v2.config import Config
    from zhilv_qiling_v2.workspace_queue import QueueStore
    a.out.parent.mkdir(parents=True,exist_ok=True)
    archive=a.out.with_suffix('.archive.sqlite');queue=a.out.with_suffix('.queue.sqlite')
    if archive.exists() or queue.exists():raise ValueError('refuse overwrite: select a new --out')
    result={'status':'running','synthetic':True,'real_contracts':0,'model_calls':0,'contracts_reviewed':0,'count':a.count,'provenance':provenance(),
        'scope':'full JSON snapshot capacity, HTTP retrieval, persistent queue capacity and sampled state transitions; not review throughput or legal accuracy',
        'archive_database':str(archive),'queue_database':str(queue),'errors':[]}
    config=Config(store_path=str(archive),enable_llm=False,ner_enable_model=False,ner_use_bert=False,use_lsa=False)
    expected={};sizes=[];started=time.perf_counter()
    with closing(api._workspace_db(config)) as db:
        for start in range(0,a.count,a.chunk):
            rows=[]
            for i in range(start,min(a.count,start+a.chunk)):
                value,key=payload(i);serialized=json.dumps(value,ensure_ascii=False,separators=(',',':'));sizes.append(len(serialized.encode()))
                rows.append((key,value['contract_id'],value['source_name'],value['contract_type'],0,0,0,value['created_at'],serialized))
                if i in (0,a.count//2,a.count-1):expected[key]=value
            with db:db.executemany('INSERT INTO workspace_reports VALUES (?,?,?,?,?,?,?,?,?)',rows)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        assert db.execute('SELECT count(*) FROM workspace_reports').fetchone()[0]==a.count
    result['archive']={'records':a.count,'bulk_insert_seconds':time.perf_counter()-started,
        'insertion_method':'real workspace schema + SQLite executemany; not 100k HTTP review calls',
        'payload_bytes':{'min':min(sizes),'max':max(sizes),'p50':sorted(sizes)[len(sizes)//2],'total':sum(sizes)},
        'source_char_distribution':{'512':sum(i%100<80 for i in range(a.count)),'2048':sum(80<=i%100<98 for i in range(a.count)),'8192':sum(i%100>=98 for i in range(a.count))}}
    previous=api._SERVER_CONFIG;api._SERVER_CONFIG=config;server=api.serve('127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_address[1]}'
    timings=[]
    try:
        for key,value in expected.items():
            status,restored,secs=request(base,'/api/workspace/reports/'+key);require(status,restored);assert restored==value;timings.append(secs)
        for page in (1,max(1,(a.count+99)//100)):
            status,listing,secs=request(base,f'/api/workspace/reports?page={page}&size=100');require(status,listing);assert listing['total']==a.count;assert listing['items'];timings.append(secs)
        for i in (0,a.count//2,a.count-1):
            name=f'synthetic-{i:06d}.txt';status,listing,secs=request(base,'/api/workspace/reports?q='+name);require(status,listing);assert listing['total']==1;timings.append(secs)
        result['archive']['http_readback']='passed: complete payload equality, first/last pagination, exact-name search'
        result['archive']['sampled_http_latency']=summary(timings)
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5);api._SERVER_CONFIG=previous
    # Queue is intentionally separate: api.serve's worker only sees the archive DB.
    store=QueueStore(queue);batch=store.create_batch({'engine':'multi','llm':'off'});bid=batch['batch_id'];started=time.perf_counter()
    for start in range(0,a.count,a.chunk):
        items=[(f'synthetic-{i:06d}.txt','txt',f'SYNTHETIC QUEUE CAPACITY ONLY {i}'.encode()) for i in range(start,min(a.count,start+a.chunk))]
        store.add_many(bid,items)
    result['queue']={'tasks':a.count,'add_many_seconds':time.perf_counter()-started,'model_runner':'not_run; no worker launched',
        'batch_id':bid,'payload_kind':'short synthetic UTF-8 blobs; not representative PDF/DOCX upload bytes'}
    reopened=QueueStore(queue);state=reopened.get_batch(bid,page=max(1,(a.count+999)//1000),size=1000)
    assert state['total']==a.count and state['counts'].get('pending')==a.count
    assert state['items'];reopened.start_batch(bid);sample=min(a.dispatch_sample,a.count);started=time.perf_counter()
    ids=set()
    for _ in range(sample):
        task=reopened.claim_next('synthetic-acceptance-worker');assert task and task['task_id'] not in ids;ids.add(task['task_id'])
        assert reopened.complete(task['task_id'],task['task_id'],{'synthetic':True,'review_executed':False},worker_id='synthetic-acceptance-worker')
    result['queue']['sampled_dispatch']={'tasks':sample,'seconds':time.perf_counter()-started,'synthetic_completion_only':True,'archives_created_by_dispatch':0}
    if a.count>sample:
        task=reopened.claim_next('interrupted-synthetic-worker',lease_seconds=.01);assert task
        time.sleep(.04);assert QueueStore(queue).recover()>=1
        recovered=QueueStore(queue).claim_next('recovery-synthetic-worker');assert recovered and recovered['task_id']==task['task_id']
        assert reopened.complete(recovered['task_id'],recovered['task_id'],{'synthetic':True},worker_id='recovery-synthetic-worker')
        result['queue']['lease_recovery']='passed after independent QueueStore reopen'
    reopened.cancel_batch(bid);cancelled=QueueStore(queue).get_batch(bid,size=1)
    assert cancelled['total']==a.count and sum(cancelled['counts'].values())==a.count
    reopened.retry_batch(bid);retried=QueueStore(queue).get_batch(bid,size=1)
    assert retried['counts'].get('completed',0)==sample+(1 if a.count>sample else 0)
    assert sum(retried['counts'].values())==a.count
    result['queue']['reopen_pagination_cancel_retry']='passed';result['queue']['final_counts']=retried['counts']
    with closing(reopened.connect()) as db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    result['archive']['database_bytes']=archive.stat().st_size;result['queue']['database_bytes']=queue.stat().st_size
    result['status']='passed';return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=100000);p.add_argument('--chunk',type=int,default=1000)
    p.add_argument('--dispatch-sample',type=int,default=100);p.add_argument('--out',type=Path,default=ROOT/'tools/acceptance_results/scale_100k.json');a=p.parse_args()
    if not 1<=a.count<=100000 or a.chunk<1 or a.dispatch_sample<0:p.error('count 1..100000, chunk>0, dispatch-sample>=0')
    try:result=run(a)
    except Exception as exc:result={'status':'failed','synthetic':True,'contracts_reviewed':0,'error':f'{type(exc).__name__}: {exc}'}
    write_json(a.out,result);print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
