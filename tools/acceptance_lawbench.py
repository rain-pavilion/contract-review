"""Real, hash-pinned official LawBench 20-task adapter and resumable inference.

plan: validate both official shot splits, export data/protocol manifest.
infer: real OpenAI-compatible HTTP calls, one atomic checkpoint per response.
fixture: explicit empty-prediction integration fixture, never a model score.
evaluate: invoke unchanged official evaluation/main.py from its required cwd.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path
from acceptance_common import ROOT, digest, provenance, write_json
from acceptance_fetch_lawbench import REPOSITORY, REVISION

TASKS={
 '1-1':('法条背诵','ROUGE-L'),'1-2':('知识问答','Accuracy'),
 '2-1':('文件校对','F0.5'),'2-2':('纠纷焦点识别','F1'),'2-3':('婚姻纠纷鉴定','F1'),
 '2-4':('问题主题识别','Accuracy'),'2-5':('阅读理解','rc-F1'),'2-6':('命名实体识别','soft-F1'),
 '2-7':('舆情摘要','ROUGE-L'),'2-8':('论点挖掘','Accuracy'),'2-9':('事件检测','F1'),'2-10':('触发词提取','soft-F1'),
 '3-1':('法条预测(基于事实)','F1'),'3-2':('法条预测(基于场景)','ROUGE-L'),'3-3':('罪名预测','F1'),
 '3-4':('刑期预测(无法条内容)','Normalized log-distance'),'3-5':('刑期预测(给定法条内容)','Normalized log-distance'),
 '3-6':('案例分析','Accuracy'),'3-7':('犯罪金额计算','Accuracy'),'3-8':('咨询','ROUGE-L')}

def source(snapshot):
    snapshot=snapshot.resolve();meta=json.loads((snapshot/'official_snapshot.json').read_text(encoding='utf-8'))
    if meta.get('repository')!=REPOSITORY or meta.get('revision')!=REVISION:raise ValueError('wrong official repository/revision')
    for entry in meta['files']:
        path=(snapshot/entry['path']).resolve()
        if not path.is_relative_to(snapshot):raise ValueError('unsafe snapshot path')
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if digest(data)!=entry['sha256'] or blob!=entry['git_blob_sha']:raise ValueError('official snapshot modified: '+entry['path'])
    return meta

def load_task(snapshot,shot,task):
    if task not in TASKS:raise ValueError('unknown official task ID '+task)
    path=snapshot/'data'/shot/(task+'.json');data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,list) or len(data)!=500:raise ValueError('official task must contain 500 records: '+str(path))
    if any(not all(isinstance(x.get(k),str) for k in ('instruction','question','answer')) for x in data):raise ValueError('wrong official data schema')
    return path,data

def prompt(row):return row['instruction']+'\n'+row['question']

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def plan(a):
    meta=source(a.snapshot);files=[]
    for shot in ('zero_shot','one_shot'):
        for task in TASKS:
            path,rows=load_task(a.snapshot,shot,task)
            files.append({'path':str(path.resolve()),'sha256':digest(path.read_bytes()),'official_task_id':task,'official_split':shot,'item_count':len(rows)})
    result={'schema_version':'official-benchmark-v1','benchmark':'LawBench','status':'not_run',
      'protocol_status':'verified_primary_source','data_status':'ready_verified_official_snapshot',
      'official_source':{'repository':REPOSITORY,'protocol_url':REPOSITORY+'/blob/'+REVISION+'/README.md#-如何评估模型',
        'verified_by':'coding-agent primary-source protocol inspection (not a human annotation identity)',
        'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},
      'checkout':str(a.snapshot.resolve()),'checkout_kind':'official_commit_sparse_source_snapshot','revision':REVISION,
      'source_snapshot_manifest':str((a.snapshot/'official_snapshot.json').resolve()),
      'source_snapshot_sha256':digest((a.snapshot/'official_snapshot.json').read_bytes()),'dataset_files':files,'prediction_files':[],
      'task_mapping':[{'official_task_id':task,'official_name':v[0],'metric':v[1],'supported':True,
        'adapter':'official instruction + newline + official question; task identity unchanged'} for task,v in TASKS.items()],
      'model':{'name':None,'revision_or_weights_sha256':None,'configuration_file':None,'configuration_sha256':None},
      'evaluation':{'script':'evaluation/main.py','working_directory':'evaluation','script_sha256':digest((a.snapshot/'evaluation/main.py').read_bytes()),
        'arguments':['-i','ABSOLUTE_PREDICTIONS_ROOT','-o','ABSOLUTE_FRESH_RESULTS_CSV'],'metrics_file':None},
      'formal_metrics':None,'official_task_count':20,'items_per_task_per_shot':500,'prepared_items_both_shots':sum(f['item_count'] for f in files),
      'remaining_requirements':['actual model identity/weights hash/configuration','actual complete predictions for selected official shot','unchanged official evaluator and required dependencies',
        'task 2-1 official ChERRANT invokes python3 and depends on LTP resources; configure a compatible environment, do not silently replace official scoring'],
      'license_notice':'official repository Apache-2.0; dataset-specific source licenses still apply per official README',
      'model_run_status':'not_run; data readiness is not a model benchmark score','provenance':provenance()}
    write_json(a.out,result);return {'status':'data_and_protocol_ready','manifest':str(a.out),'official_tasks':20,'items_both_shots':20000,'full_model_benchmark':'not_run'}

MODEL_ID = 'legalone-8b'
MODEL_PARAMETERS = 8195978240


def json_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def selected_tasks(value):
    tasks = list(TASKS) if value == 'all' else value.split(',')
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError('select unique official task IDs or all')
    return tasks


def native_identity(payload, model):
    """Validate observable architecture/alias, not an unprovable remote weights hash."""
    if model != MODEL_ID:
        raise ValueError('this acceptance run requires the exact legalone-8b alias')
    matches = [row for row in payload.get('data', []) if row.get('id') == model]
    if len(matches) != 1:
        raise ValueError('native server must advertise one exact requested model alias')
    row = matches[0]
    if row.get('owned_by') != 'llamacpp' or row.get('meta', {}).get('n_params') != MODEL_PARAMETERS:
        raise ValueError('expected actual llama.cpp LegalOne-8B architecture (8195978240 parameters)')
    observed = {key: row[key] for key in ('id', 'owned_by', 'meta')}
    return {'observed_native_model': observed, 'identity_sha256': json_digest(observed),
            'assurance': 'exact native alias and observable parameter architecture verified; declared weights revision is not remotely cryptographically proven'}


def verify_native_identity(endpoint, model, timeout):
    with urllib.request.urlopen(endpoint.rstrip('/')+'/models', timeout=timeout) as response:
        return native_identity(json.loads(response.read()), model)


def ensure_adapter_snapshot(run_dir):
    """Preserve exact source bytes and refuse to overwrite a different snapshot."""
    data = Path(__file__).read_bytes()
    target = run_dir/'adapter_snapshot.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError('adapter source snapshot drift; use a new run directory')
    else:
        with target.open('xb') as handle:
            handle.write(data)
    return target


def config_for(a):
    config = {'benchmark':'LawBench','source_revision':REVISION,'shot':a.shot,'model':a.model,'model_revision_or_weights_sha256':a.model_revision,
        'endpoint':a.endpoint.rstrip('/'),'temperature':a.temperature,'max_tokens':a.max_tokens,'timeout_seconds':a.timeout,
        'prompt_composition':'official instruction + newline + official question, no reference answer appended','prompt_truncation':False,
        'item_limit_per_task':a.item_limit,'selected_official_tasks':selected_tasks(a.tasks),
        'weights_revision_assurance':'operator-declared identifier; native alias/architecture separately observed, not a remote hash attestation',
        'context_preflight_policy':'native_template_and_tokenize_before_each_generation_v1'}
    if getattr(a,'reasoning_effort',None) is not None:
        config['reasoning_effort'] = a.reasoning_effort
    if getattr(a, 'native_thinking', 'default') != 'default':
        config['chat_template_kwargs'] = {'enable_thinking': a.native_thinking == 'on'}
    if getattr(a, 'native_reasoning_format', 'default') != 'default':
        config['reasoning_format'] = a.native_reasoning_format
    return config


def request_body(config, row):
    body={'model':config['model'],'messages':[{'role':'user','content':prompt(row)}],
          'temperature':config['temperature'],'max_tokens':config['max_tokens']}
    for key in ('reasoning_effort','chat_template_kwargs','reasoning_format'):
        if key in config:body[key]=config[key]
    return body


def native_json(endpoint, route, body, timeout):
    parsed=urllib.parse.urlsplit(endpoint.rstrip('/'))
    if parsed.path.rstrip('/') != '/v1' or parsed.query or parsed.fragment:
        raise ValueError('native LawBench endpoint must end with /v1 without query or fragment')
    url=urllib.parse.urlunsplit((parsed.scheme,parsed.netloc,route,'',''))
    request=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:return json.loads(response.read())


def preflight_context(a, identity, row):
    """Apply the actual server template and tokenizer without model generation."""
    config=config_for(a)
    context=identity['observed_native_model'].get('meta',{}).get('n_ctx')
    if type(context) is not int or context<1:raise ValueError('native per-slot context size missing')
    formatted=native_json(a.endpoint,'/apply-template',request_body(config,row),a.timeout).get('prompt')
    if not isinstance(formatted,str):raise ValueError('native template did not return text')
    tokens=native_json(a.endpoint,'/tokenize',{'content':formatted,'add_special':True,'parse_special':True},a.timeout).get('tokens')
    if not isinstance(tokens,list) or any(type(t) is not int for t in tokens):
        raise ValueError('native tokenizer did not return token IDs')
    required=len(tokens)+a.max_tokens+1
    result={'method':'actual_native_chat_template_and_tokenizer_no_generation','prompt_tokens':len(tokens),
            'formatted_prompt_sha256':digest(formatted),'server_context_tokens':context,
            'reserved_output_tokens':a.max_tokens,'reserved_stop_token':1,'required_context_tokens':required,
            'fits':required<=context,'input_truncation_permitted':False}
    if not result['fits']:
        raise ValueError(f'context budget exceeded before generation: prompt {len(tokens)} + output {a.max_tokens} + stop 1 > native slot {context}; input was not truncated')
    return result


def evaluation_environment():
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'tools/acceptance_vendor/eval_deps'),str(ROOT/'.web-deps'),env.get('PYTHONPATH','')])
    env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
    return env


def evaluation_preflight(snapshot, python, tasks):
    """Check the exact interpreter used by unchanged official scoring code."""
    code="""import importlib, json, os, shutil, signal
result={'python':__import__('sys').executable,'os_name':os.name,'imports':{},'python3_command':shutil.which('python3'),'sigalrm_available':hasattr(signal,'SIGALRM')}
for module in ['pandas','numpy','rouge_chinese','jieba','nltk','cn2an','evaluation_functions.jec_kd']:
    try:importlib.import_module(module);result['imports'][module]={'ok':True}
    except Exception as exc:result['imports'][module]={'ok':False,'error':type(exc).__name__+': '+str(exc)}
print(json.dumps(result))"""
    process=subprocess.run([str(python.resolve()),'-X','utf8','-X','faulthandler','-u','-c',code],cwd=snapshot/'evaluation',
                           env=evaluation_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    errors=[]
    if process.returncode:
        metadata={'probe_exit_code':process.returncode,'probe_exit_code_hex':hex(process.returncode & 0xffffffff),
                  'stderr':process.stderr[-6000:],'stdout':process.stdout[-1000:]}
        errors.append('official evaluator import probe failed')
    else:
        metadata=json.loads(process.stdout.strip().splitlines()[-1])
        errors.extend('missing evaluation dependency: '+name for name,row in metadata['imports'].items() if not row['ok'])
    if '2-1' in tasks:
        # The unchanged ChERRANT first pass uses timeout_decorator's SIGALRM.
        # On Windows it can turn every exception into a timeout and substitute
        # targets with "无" in the second pass. Do not silently call that a score.
        if not metadata.get('sigalrm_available'):
            errors.append('official task 2-1 ChERRANT requires POSIX SIGALRM; this interpreter would change timeout behavior')
        if not metadata.get('python3_command'):errors.append('official task 2-1 requires an accessible compatible python3 command')
        extra="import importlib,json; names=['ltp','pypinyin','opencc','timeout_decorator','Levenshtein']; result={}\nfor name in names:\n try:importlib.import_module(name);result[name]=True\n except Exception:result[name]=False\nprint(json.dumps(result))"
        probe=subprocess.run([str(python.resolve()),'-X','utf8','-X','faulthandler','-u','-c',extra],cwd=snapshot/'evaluation',env=evaluation_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        modules=json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode==0 else {}
        metadata['cherrant_imports']=modules
        if probe.returncode:metadata['cherrant_probe_failure']={'exit_code':probe.returncode,'exit_code_hex':hex(probe.returncode & 0xffffffff),'stderr':probe.stderr[-6000:]}
        errors.extend('missing ChERRANT dependency: '+name for name,ok in modules.items() if not ok)
        if probe.returncode:errors.append('ChERRANT dependency probe failed')
    return {'ready':not errors,'errors':errors,'runtime':metadata,'official_source_modified':False}


def completion_summary(config, completed, receipts):
    """Task completeness and output truncation are distinct from a benchmark score."""
    counts={task:row['items'] for task,row in completed.items()}
    selected=config['selected_official_tasks']
    complete=[task for task in selected if counts.get(task)==500]
    successful=[row for row in receipts if row.get('status')=='succeeded']
    truncated=[row for row in successful if row.get('finish_reason')=='length']
    input_problems=[row for row in successful if row.get('prompt_token_count_matches_preflight') is False
                    or row.get('native_context_diagnostics',{}).get('truncated') is True]
    return {'selected_official_tasks':selected,'evaluated_items_per_task':counts,
            'completed_official_tasks':complete,'full_selected_official_tasks':bool(selected) and set(complete)==set(selected),
            'full_official_benchmark':set(complete)==set(TASKS),
            'successful_response_count':len(successful),'truncated_response_count':len(truncated),
            'truncated_items':[{'task':row['task'],'item_index':row['item_index']} for row in truncated],
            'all_responses_untruncated':bool(successful) and not truncated,
            'native_input_context_problem_count':len(input_problems),
            'native_input_context_problem_items':[{'task':row['task'],'item_index':row['item_index']} for row in input_problems],
            'interpretation':'500 outputs complete one official task only; all 20 tasks require 10000 outputs in one shot; truncation is reported without rewriting outputs'}


def local_tokenizer_preflight(a,tasks):
    # Isolate native library imports: a host DLL failure must produce a failed
    # preflight report instead of terminating the acceptance process silently.
    config=config_for(a);kwargs=dict(config.get('chat_template_kwargs',{}))
    if 'enable_thinking' not in kwargs:kwargs['enable_thinking']=True
    code="""import hashlib,json,sys
from pathlib import Path
from transformers import AutoTokenizer
c=json.loads(sys.argv[1]); tokenizer=AutoTokenizer.from_pretrained(c['tokenizer'],local_files_only=True)
budgets={};maximum=[0,None,None]
for task in c['tasks']:
    path=Path(c['snapshot'])/'data'/c['shot']/(task+'.json')
    data=path.read_bytes();rows=json.loads(data)[:c['item_limit']]
    messages=[[{'role':'user','content':row['instruction']+'\\n'+row['question']}] for row in rows]
    prompts=[tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=True,**c['template_kwargs']) for m in messages]
    lengths=[len(x) for x in tokenizer(prompts,add_special_tokens=False)['input_ids']]
    index=max(range(len(lengths)),key=lengths.__getitem__)
    budgets[task]={'items':len(lengths),'max_prompt_tokens':lengths[index],'max_prompt_item_index':index,'official_data_sha256':hashlib.sha256(data).hexdigest()}
    if lengths[index]>maximum[0]:maximum=[lengths[index],task,index]
print(json.dumps({'tasks':budgets,'maximum':maximum}))"""
    settings={'tokenizer':str(ROOT/'zhilv_qiling_v2/models/pretrained/legalone-8b'),
              'snapshot':str(a.snapshot.resolve()),'shot':a.shot,'tasks':tasks,'item_limit':a.item_limit,'template_kwargs':kwargs}
    process=subprocess.run([str(a.python.resolve()),'-X','utf8','-X','faulthandler','-u','-c',code,json.dumps(settings)],
        cwd=ROOT,env=evaluation_environment(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    if process.returncode:
        return {'ready':False,'exit_code':process.returncode,'exit_code_hex':hex(process.returncode & 0xffffffff),
                'stderr':process.stderr[-6000:],'stdout':process.stdout[-1000:]}
    return {'ready':True,**json.loads(process.stdout.strip().splitlines()[-1])}


def preflight(a):
    source(a.snapshot);tasks=selected_tasks(a.tasks)
    evaluation=evaluation_preflight(a.snapshot,a.python,tasks)
    local=local_tokenizer_preflight(a,tasks)
    budgets=local.get('tasks',{});maximum=local.get('maximum',[0,None,None])
    native={'ready':False,'error':None}
    try:
        identity=verify_native_identity(a.endpoint,a.model,a.timeout)
        if not local['ready']:raise ValueError('local all-input tokenizer scan failed; cannot select maximum native check')
        _,rows=load_task(a.snapshot,a.shot,maximum[1])
        native={'ready':True,'identity':identity,'largest_local_prompt_native_check':preflight_context(a,identity,rows[maximum[2]])}
    except Exception as exc:native['error']=f'{type(exc).__name__}: {exc}'
    result={'benchmark':'LawBench','status':'ready' if evaluation['ready'] and local['ready'] and native['ready'] else 'not_ready',
            'actual_generation_calls':0,'official_source_integrity':'passed','tasks':budgets,'shot':a.shot,
            'largest_local_prompt_tokens':maximum[0],'reserved_output_tokens':a.max_tokens,
            'local_required_context_tokens':maximum[0]+a.max_tokens+1 if local['ready'] else None,
            'native':native,'evaluation':evaluation,'local_tokenizer_probe':local,
            'scope':'preflight only; not predictions or a model score','tokenizer_assurance':'local official model tokenizer scans selected prompts; infer additionally verifies every prompt with the actual native template and tokenizer'}
    write_json(a.out,result);return result


def reconcile_run(a, state, *, repair):
    """Recover a committed response before prediction/state writes; refuse unknown outcomes."""
    config = state['configuration']; model = config['model']
    if json_digest(config) != state['configuration_sha256']:
        raise ValueError('run configuration hash changed')
    snapshot_name=state.get('adapter_source_snapshot','adapter_snapshot.py')
    if snapshot_name != 'adapter_snapshot.py':
        raise ValueError('inference adapter snapshot must remain inside the run directory')
    snapshot = a.run_dir/snapshot_name
    if not snapshot.is_file() or digest(snapshot.read_bytes()) != state.get('adapter_sha256'):
        raise ValueError('recorded inference adapter source snapshot is missing or changed')
    identity = state['model_identity']
    observed = native_identity({'data': [identity['observed_native_model']]}, model)
    if observed['identity_sha256'] != identity['identity_sha256']:
        raise ValueError('stored native identity hash changed')
    root = a.run_dir/'predictions'
    if root.exists() and any(p.name != model for p in root.iterdir() if p.is_dir()):
        raise ValueError('mixed model prediction directories are forbidden')
    grouped = {}; receipts = []; task_cache = {}
    def official(task):
        if task not in task_cache:
            task_cache[task] = load_task(a.snapshot, config['shot'], task)
        return task_cache[task]
    for path in sorted((a.run_dir/'receipts').glob('*/*.json')):
        receipt = json.loads(path.read_text(encoding='utf-8'))
        task = path.parent.name; index = receipt.get('item_index')
        if (task not in config['selected_official_tasks'] or receipt.get('task') != task or
                type(index) is not int or not 0 <= index < config['item_limit_per_task']):
            raise ValueError('receipt task/index is outside the fixed official selection')
        if (receipt.get('configuration_sha256') != state['configuration_sha256'] or
                receipt.get('native_identity_sha256') != identity['identity_sha256'] or
                receipt.get('requested_model') != model):
            raise ValueError('receipt identity/configuration changed')
        if (('reasoning_effort' in config) != ('reasoning_effort' in receipt) or
                receipt.get('reasoning_effort') != config.get('reasoning_effort')):
            raise ValueError('receipt reasoning mode differs from fixed configuration')
        for key in ('chat_template_kwargs', 'reasoning_format'):
            if (key in config) != (key in receipt) or receipt.get(key) != config.get(key):
                raise ValueError('receipt native reasoning mode differs from fixed configuration')
        _, official_rows = official(task)
        if receipt.get('prompt_sha256') != digest(prompt(official_rows[index])):
            raise ValueError('receipt prompt hash differs from official item')
        status = receipt.get('status')
        if status == 'started':
            raise ValueError('unfinished call receipt: outcome unknown; refusing automatic duplicate inference')
        if status not in ('succeeded', 'failed'):
            raise ValueError('unknown call receipt status')
        receipts.append({'path':str(path.relative_to(a.run_dir)), 'sha256':digest(path.read_bytes()),
                         'status':status, 'task':task, 'item_index':index})
        if status == 'succeeded':
            if receipt.get('response_model') != model or receipt.get('model_identity_verified') is not True:
                raise ValueError('successful receipt lacks verified response model identity')
            if config.get('context_preflight_policy'):
                budget=receipt.get('context_preflight',{})
                if (budget.get('fits') is not True or budget.get('input_truncation_permitted') is not False
                        or budget.get('reserved_output_tokens')!=config['max_tokens']
                        or budget.get('server_context_tokens')!=identity['observed_native_model']['meta'].get('n_ctx')
                        or type(budget.get('prompt_tokens')) is not int
                        or budget.get('required_context_tokens')!=budget['prompt_tokens']+config['max_tokens']+1
                        or budget['required_context_tokens']>budget['server_context_tokens']):
                    raise ValueError('successful receipt context preflight is missing or inconsistent')
                if receipt.get('output_truncated') != (receipt.get('finish_reason')=='length'):
                    raise ValueError('successful receipt output truncation flag differs from native finish reason')
            entry = receipt.get('prediction_entry')
            if not isinstance(entry, dict) or json_digest(entry) != receipt.get('prediction_entry_sha256'):
                raise ValueError('receipt prediction payload/hash mismatch')
            if 'response_message' in receipt:
                message = receipt['response_message']
                if (not isinstance(message, dict) or json_digest(message) != receipt.get('response_message_sha256')
                        or message.get('content') != entry.get('prediction')):
                    raise ValueError('receipt original response message/hash differs from prediction')
            target = grouped.setdefault(task, {})
            if str(index) in target:
                raise ValueError('multiple successful calls for one official item require explicit reconciliation')
            target[str(index)] = entry
    paths = {p.stem:p for p in (root/model).glob('*.json')}
    completed = {}; predictions_by_task = {}
    for task in sorted(set(paths) | set(grouped)):
        if task not in config['selected_official_tasks']:
            raise ValueError('prediction task is outside fixed official selection')
        data_path, rows = official(task)
        path = paths.get(task, root/model/(task+'.json'))
        existing = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        committed = grouped.get(task, {})
        if set(committed) != {str(i) for i in range(len(committed))}:
            raise ValueError('successful receipt indices must be contiguous official first-N items')
        for index, entry in committed.items():
            row = rows[int(index)]
            if (entry.get('refr') != row['answer'] or
                    entry.get('origin_prompt') != [{'role':'HUMAN','prompt':prompt(row)}] or
                    not isinstance(entry.get('prediction'), str)):
                raise ValueError('receipt prediction does not align with official data')
        if any(index not in committed or entry != committed[index] for index, entry in existing.items()):
            raise ValueError('prediction lacks a matching successful receipt or was changed')
        if existing != committed:
            if not repair:
                raise ValueError('committed response requires infer resume to recover its prediction file')
            atomic_json(path, committed)
        predictions_by_task[task] = committed
        if committed:
            completed[task] = {'items':len(committed),'sha256':digest(path.read_bytes()),
                               'official_data_sha256':digest(data_path.read_bytes())}
    return predictions_by_task, completed, receipts

def infer(a):
    source(a.snapshot);config=config_for(a);config_hash=json_digest(config)
    if not a.model_revision:raise ValueError('declared model revision/weights identifier is required before inference')
    adapter_snapshot = ensure_adapter_snapshot(a.run_dir)
    identity = verify_native_identity(a.endpoint, a.model, a.timeout)
    state_path=a.run_dir/'state.json';state={'scope':'actual_model_http_inference','synthetic':False,'model_calls':0,'completed':{},'errors':[],
        'configuration':config,'configuration_sha256':config_hash,'adapter_sha256':digest(Path(__file__).read_bytes()),
        'adapter_source_snapshot':str(adapter_snapshot.relative_to(a.run_dir)),
        'status':'running','full_official_benchmark':False,'provenance':provenance()}
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['configuration_sha256']!=config_hash:raise ValueError('resume configuration changed; use new run directory')
        if state.get('adapter_sha256') != digest(Path(__file__).read_bytes()):
            raise ValueError('resume adapter source changed; use a new run directory')
        if state.get('model_identity', {}).get('identity_sha256') != identity['identity_sha256']:
            raise ValueError('resume native model identity changed')
    else:
        state['model_identity'] = identity
        atomic_json(state_path,state)
    selected=selected_tasks(a.tasks);generated=0
    recovered, completed, receipts = reconcile_run(a, state, repair=True)
    state.update(completed=completed, receipt_files=receipts, model_calls=len(receipts))
    atomic_json(state_path, state)
    for task in selected:
        data_path,rows=load_task(a.snapshot,a.shot,task);out=a.run_dir/'predictions'/a.model/(task+'.json')
        predictions=recovered.get(task, {})
        if len(predictions)>a.item_limit:raise ValueError('existing predictions exceed selected official subset')
        for i,row in enumerate(rows[:a.item_limit]):
            if str(i) in predictions:continue
            if a.max_items and generated>=a.max_items:break
            try:
                budget=preflight_context(a,identity,row)
            except Exception as exc:
                state['status']='blocked_before_generation'
                state['context_preflight_error']={'task':task,'item_index':i,'error':f'{type(exc).__name__}: {exc}'}
                atomic_json(state_path,state)
                raise
            body=request_body(config,row)
            request=urllib.request.Request(config['endpoint']+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            call_id=uuid.uuid4().hex
            receipt_path=a.run_dir/'receipts'/task/(f'{i:04d}-{call_id}.json')
            receipt={'schema_version':'lawbench-model-call-v1','call_id':call_id,'task':task,'item_index':i,
                'status':'started','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'configuration_sha256':config_hash,'native_identity_sha256':identity['identity_sha256'],
                'requested_model':a.model,'declared_weights_revision':a.model_revision,
                'prompt_sha256':digest(prompt(row)),'max_tokens':a.max_tokens,'context_preflight':budget}
            if 'reasoning_effort' in config:
                receipt['reasoning_effort'] = config['reasoning_effort']
            for key in ('chat_template_kwargs', 'reasoning_format'):
                if key in config:
                    receipt[key] = config[key]
            atomic_json(receipt_path,receipt)
            state['model_calls']+=1;state['active_call_receipt']=str(receipt_path.relative_to(a.run_dir));atomic_json(state_path,state)
            started=time.perf_counter()
            try:
                with urllib.request.urlopen(request,timeout=a.timeout) as response:data=json.loads(response.read())
                receipt.update(response_model=data.get('model'),usage=data.get('usage'),
                               finish_reason=data.get('choices',[{}])[0].get('finish_reason'),
                               output_truncated=data.get('choices',[{}])[0].get('finish_reason')=='length')
                receipt['native_context_diagnostics']={key:data[key] for key in ('truncated','tokens_evaluated','timings') if key in data}
                actual_prompt=(data.get('usage') or {}).get('prompt_tokens')
                receipt['prompt_token_count_matches_preflight']=(abs(actual_prompt-budget['prompt_tokens'])<=1) if type(actual_prompt) is int else None
                if data.get('model') != a.model:
                    raise ValueError('response model identity differs from requested LegalOne-8B alias')
                answer=data['choices'][0]['message']['content']
                if not isinstance(answer,str):raise ValueError('model response content is not text')
                # Persist every returned message field, including separated reasoning.
                # Scoring still receives the unchanged content, never a gold-guided extraction.
                receipt['response_message'] = data['choices'][0]['message']
                receipt['response_message_sha256'] = json_digest(receipt['response_message'])
                entry={'origin_prompt':[{'role':'HUMAN','prompt':prompt(row)}],'prediction':answer,'refr':row['answer']}
                receipt.update(status='succeeded',model_identity_verified=True,
                    ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                    prediction_entry=entry,prediction_entry_sha256=json_digest(entry))
                # Commit the complete response first. Recovery can replay it without a duplicate model call.
                atomic_json(receipt_path,receipt)
            except Exception as exc:
                receipt.update(status='failed',ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-started,error=f'{type(exc).__name__}: {exc}')
                atomic_json(receipt_path,receipt)
                state['status']='interrupted_resumable';state['errors'].append({'task':task,'item_index':i,'error':f'{type(exc).__name__}: {exc}'})
                state.pop('active_call_receipt', None)
                state['receipt_files'].append({'path':str(receipt_path.relative_to(a.run_dir)),
                    'sha256':digest(receipt_path.read_bytes()),'status':'failed','task':task,'item_index':i})
                state['model_calls']=len(state['receipt_files'])
                atomic_json(state_path,state);raise
            predictions[str(i)]=entry;atomic_json(out,predictions);generated+=1
            state['completed'][task]={'items':len(predictions),'sha256':digest(out.read_bytes()),
                                      'official_data_sha256':digest(data_path.read_bytes())}
            state['receipt_files'].append({'path':str(receipt_path.relative_to(a.run_dir)),
                'sha256':digest(receipt_path.read_bytes()),'status':'succeeded','task':task,'item_index':i})
            state['model_calls']=len(state['receipt_files'])
            state.pop('active_call_receipt', None);atomic_json(state_path,state)
        if a.max_items and generated>=a.max_items:break
    quality=completion_summary(config,state['completed'],[json.loads((a.run_dir/row['path']).read_text(encoding='utf-8')) for row in state['receipt_files']])
    state['completion_summary']=quality
    state['full_official_benchmark']=quality['full_official_benchmark']
    state['status']='complete_official_shot_predictions' if state['full_official_benchmark'] else 'partial_predictions_resumable'
    atomic_json(state_path,state);return {'status':state['status'],'new_actual_model_responses':generated,
        'full_official_benchmark':state['full_official_benchmark'],'completion_summary':quality}

def fixture(a):
    source(a.snapshot)
    if a.run_dir.exists():raise ValueError('fixture requires fresh run directory')
    _,rows=load_task(a.snapshot,'zero_shot','1-2')
    predictions={str(i):{'origin_prompt':[{'role':'HUMAN','prompt':prompt(row)}],'prediction':'','refr':row['answer']} for i,row in enumerate(rows[:3])}
    atomic_json(a.run_dir/'predictions'/'adapter_empty_output_fixture'/'1-2.json',predictions)
    atomic_json(a.run_dir/'state.json',{'scope':'official_evaluator_adapter_fixture_only','fixture_only':True,'model_calls':0,
        'model_score':False,'configuration':{'shot':'zero_shot','source_revision':REVISION},'full_official_benchmark':False,
        'description':'3 official task 1-2 records + deliberately empty outputs to exercise official abstention parsing; NOT model predictions'})
    return {'status':'fixture_ready','scope':'not_model_evaluation','official_items':3,'model_calls':0}

def evaluate(a):
    source(a.snapshot);state=json.loads((a.run_dir/'state.json').read_text(encoding='utf-8'));shot=state['configuration']['shot'];count=0;task_set=set()
    fixture_only=bool(state.get('fixture_only'))
    receipt_files=[]
    if fixture_only:
        expected_model='adapter_empty_output_fixture'
        if state.get('scope') != 'official_evaluator_adapter_fixture_only' or state.get('model_calls') != 0:
            raise ValueError('invalid non-model fixture declaration')
    else:
        expected_model=state['configuration']['model']
        _, completed, receipt_files = reconcile_run(a, state, repair=False)
        expected_receipts=sorted(state.get('receipt_files', []),key=lambda row:row['path'])
        if completed != state.get('completed') or receipt_files != expected_receipts or len(receipt_files) != state.get('model_calls'):
            raise ValueError('state/receipt/prediction ledger differs; resume infer to reconcile committed responses')
    prediction_root=a.run_dir/'predictions'
    if any(p.name != expected_model for p in prediction_root.iterdir() if p.is_dir()):
        raise ValueError('only the single declared model directory may be evaluated')
    item_counts={}
    for path in (prediction_root/expected_model).glob('*.json'):
        task=path.stem;_,rows=load_task(a.snapshot,shot,task);pred=json.loads(path.read_text(encoding='utf-8'))
        if set(pred)!={str(i) for i in range(len(pred))}:raise ValueError('non-contiguous official output keys')
        if not pred or len(pred)>500:raise ValueError('invalid official output count')
        for index,item in pred.items():
            row=rows[int(index)]
            if item.get('refr')!=row['answer'] or item.get('origin_prompt')!=[{'role':'HUMAN','prompt':prompt(row)}]:raise ValueError('prediction does not align with official data')
        count+=len(pred);task_set.add(task);item_counts[task]=len(pred)
    if not count:raise ValueError('no actual or declared fixture predictions')
    dependencies=evaluation_preflight(a.snapshot,a.python,sorted(task_set))
    if not dependencies['ready']:
        result={'benchmark':'LawBench','status':'failed','scope':'official_evaluator_preflight',
                'official_evaluator_executed':False,'full_official_benchmark':False,
                'evaluated_items':0,'prediction_items_available':item_counts,'dependency_preflight':dependencies,
                'errors':dependencies['errors']}
        write_json(a.out,result);return result
    metrics=a.run_dir/'official_results.csv'
    if metrics.exists():raise ValueError('refuse stale metrics output; select new run or remove only a reviewed stale result')
    env=evaluation_environment()
    command=[str(a.python.resolve()),'main.py','-i',str((a.run_dir/'predictions').resolve()),'-o',str(metrics.resolve())]
    started=time.perf_counter();executed=subprocess.run(command,cwd=a.snapshot/'evaluation',env=env,capture_output=True,text=True,timeout=600,encoding='utf-8',errors='replace')
    result={'benchmark':'LawBench','scope':state.get('scope'),'status':'official_evaluator_executed' if executed.returncode==0 and metrics.exists() else 'failed',
        'fixture_only':fixture_only,'model_score':not fixture_only,
        'full_official_benchmark':not fixture_only and set(item_counts)==set(TASKS) and all(n==500 for n in item_counts.values()),
        'single_model_directory':expected_model,'evaluated_items_per_task':item_counts,
        'model_configuration':state.get('configuration'),'observed_model_identity':state.get('model_identity'),
        'actual_call_receipts':receipt_files,'actual_model_calls':len(receipt_files),
        'official_source_revision':REVISION,'official_evaluator_sha256':digest((a.snapshot/'evaluation/main.py').read_bytes()),
        'dependency_preflight':dependencies,
        'inference_adapter_sha256':state.get('adapter_sha256'),
        'inference_adapter_source_snapshot':str((a.run_dir/state.get('adapter_source_snapshot','adapter_snapshot.py')).resolve()) if not fixture_only else None,
        'evaluated_items':count,'evaluated_official_tasks':sorted(task_set),'seconds':time.perf_counter()-started,
        'command':command,'working_directory':str((a.snapshot/'evaluation').resolve()),'exit_code':executed.returncode,
        'stdout_tail':executed.stdout[-4000:],'stderr_tail':executed.stderr[-4000:],'provenance':provenance()}
    if metrics.exists():
        result['official_metrics_sha256']=digest(metrics.read_bytes())
        with metrics.open(encoding='utf-8-sig',newline='') as f:result['official_output_rows']=list(csv.DictReader(f))
    if result['fixture_only']:result['interpretation']='integration fixture only; empty output should score 0 with abstention 1; not a model result or leaderboard submission'
    else:
        result['completion_summary']=completion_summary(state['configuration'],state['completed'],
            [json.loads((a.run_dir/row['path']).read_text(encoding='utf-8')) for row in receipt_files])
    write_json(a.out,result);source(a.snapshot);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','preflight','infer','fixture','evaluate']);p.add_argument('--snapshot',type=Path,default=ROOT/'tools/acceptance_vendor/LawBench')
    p.add_argument('--out',type=Path,default=ROOT/'docs/验收/LawBench.manifest.json');p.add_argument('--run-dir',type=Path,default=ROOT/'tools/acceptance_results/lawbench_run')
    p.add_argument('--python',type=Path,default=ROOT/'.venv/Scripts/python.exe');p.add_argument('--endpoint',default='http://127.0.0.1:8620/v1');p.add_argument('--model',default='legalone-8b')
    p.add_argument('--model-revision',default='');p.add_argument('--shot',choices=['zero_shot','one_shot'],default='zero_shot');p.add_argument('--tasks',default='all')
    p.add_argument('--max-items',type=int,default=1,help='new responses this invocation; 0 explicitly requests all remaining items')
    p.add_argument('--item-limit',type=int,default=500,help='official first N rows per task, 1..500; use 10 to resume a fixed first-10 subset without extending it')
    p.add_argument('--reasoning-effort',default=None,help='optional native inference mode; none disables thinking; omitted preserves the server default')
    p.add_argument('--native-thinking', choices=['default','on','off'], default='default', help='explicit llama.cpp chat-template thinking control; original prompt is unchanged')
    p.add_argument('--native-reasoning-format', choices=['default','none','deepseek'], default='default', help='explicit native response reasoning separation; raw message is retained')
    p.add_argument('--max-tokens',type=int,default=512);p.add_argument('--temperature',type=float,default=0);p.add_argument('--timeout',type=float,default=600);a=p.parse_args()
    if a.max_items<0 or a.max_tokens<1 or not 1<=a.item_limit<=500:p.error('max-items>=0, max-tokens>0, item-limit 1..500 required')
    if not a.model or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in a.model):p.error('model folder identifier must be safe alphanumeric/dot/underscore/hyphen')
    if a.action=='evaluate' and a.out==ROOT/'docs/验收/LawBench.manifest.json':a.out=a.run_dir/'evaluation.result.json'
    if a.action=='preflight' and a.out==ROOT/'docs/验收/LawBench.manifest.json':a.out=a.run_dir/'preflight.result.json'
    try:result=globals()[a.action](a)
    except Exception as exc:
        result={'status':'failed','error':f'{type(exc).__name__}: {exc}','full_official_benchmark':False};print(json.dumps(result,ensure_ascii=False));return 1
    print(json.dumps(result,ensure_ascii=False));return 1 if result['status'] in ('failed','not_ready') else 0
if __name__=='__main__':raise SystemExit(main())
