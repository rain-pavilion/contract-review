"""Prepare verified b12375 CUDA 12.4/13.5 files separately; never execute them."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import stat
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TAG='b12375'


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def official_assets(manifest,cuda_version='12.4'):
    if cuda_version not in ('12.4','13.5'):raise ValueError('unsupported pinned CUDA version')
    names=(f'llama-{TAG}-bin-win-cuda-{cuda_version}-x64.zip',f'cudart-llama-bin-win-cuda-{cuda_version}-x64.zip')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if data['tag_name']!=TAG:raise ValueError('fixed release b12375 required')
    selected=[]
    for name in names:
        entries=[entry for entry in data['assets'] if entry['name']==name]
        if len(entries)!=1:raise ValueError('missing or ambiguous fixed asset: '+name)
        entry=entries[0]
        url=f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{name}'
        digest=entry.get('digest','')
        if entry['browser_download_url']!=url or not digest.startswith('sha256:') or len(digest)!=71:
            raise ValueError('official asset identity/digest unavailable')
        int(digest[7:],16)
        selected.append({'name':name,'url':url,'official_sha256':digest[7:],'declared_bytes':entry.get('size',0)})
    return selected


def download(asset,directory,progress=None):
    target=directory/asset['name']
    if target.exists():
        if sha256(target)!=asset['official_sha256']:raise ValueError('existing archive digest mismatch; retained: '+str(target))
        return {'path':str(target),'bytes':target.stat().st_size,'sha256':asset['official_sha256'],'official_digest_verified':True,'reused':True}
    partial=target.with_suffix('.zip.part')
    request=urllib.request.Request(asset['url'],headers={'User-Agent':'ZLQL-pinned-CUDA-runtime-preparation'})
    started=time.perf_counter();last_progress=0;last_report=started;count=0;value=hashlib.sha256()
    print(json.dumps({'asset':asset['name'],'status':'connecting','url':asset['url'],'socket_timeout_seconds':20}),flush=True)
    with urllib.request.urlopen(request,timeout=20) as response,partial.open('wb') as handle:
        expected=response.headers.get('Content-Length')
        for chunk in iter(lambda:response.read(256*1024),b''):
            handle.write(chunk);value.update(chunk);count+=len(chunk)
            now=time.perf_counter()
            if count-last_progress>=32*1024*1024 or now-last_report>=30:
                update={'asset':asset['name'],'status':'downloading','received_bytes':count,'content_length':expected,'elapsed_seconds':round(now-started,2)}
                print(json.dumps(update),flush=True)
                if progress:progress(update)
                last_progress=count;last_report=now
            if now-started>600:raise TimeoutError('asset download exceeded bounded 600-second total; partial retained')
    if expected and count!=int(expected):raise ValueError('HTTP Content-Length mismatch')
    if asset['declared_bytes'] and count!=asset['declared_bytes']:raise ValueError('official size mismatch')
    if value.hexdigest()!=asset['official_sha256']:raise ValueError('download digest mismatch; partial retained')
    partial.replace(target)
    return {'path':str(target),'bytes':count,'sha256':value.hexdigest(),'official_digest_verified':True,'seconds':time.perf_counter()-started}


def extract_verified(archive,destination):
    base=destination.resolve();destination.mkdir(parents=True,exist_ok=True);files=[]
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            target=(base/entry.filename).resolve()
            if not target.is_relative_to(base) or stat.S_ISLNK(entry.external_attr>>16):raise ValueError('unsafe zip member')
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            content=package.read(entry);digest=hashlib.sha256(content).hexdigest()
            if target.exists():
                if sha256(target)!=digest:raise ValueError('different existing extraction member retained: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as handle:handle.write(content)
            files.append({'path':str(target.relative_to(base)),'bytes':len(content),'sha256':digest})
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-manifest',type=Path,default=ROOT/'.model-runtime/llama.cpp/latest/release.json')
    parser.add_argument('--cuda-version',choices=['12.4','13.5'],default='12.4')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--download',action='store_true',help='download/verify/extract official assets; no binary execution')
    args=parser.parse_args();destination=(args.out or ROOT/f'.model-runtime/llama.cpp/.cuda/{TAG}-win-cuda{args.cuda_version}-x64').resolve()
    if not destination.is_relative_to((ROOT/'.model-runtime/llama.cpp/.cuda').resolve()):parser.error('output must remain inside separate .cuda directory')
    result={'schema':'cuda-runtime-preparation-v1','captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'planned_not_downloaded','repository':'ggml-org/llama.cpp','release_tag':TAG,'cuda_version':args.cuda_version,
            'release_manifest':str(args.release_manifest.resolve()),'release_manifest_sha256':sha256(args.release_manifest),
            'output_directory':str(destination),'downloaded_code_executed':False,'model_loaded':False,
            'old_cpu_runtime_modified':False,'weights_modified':False,'assets':official_assets(args.release_manifest,args.cuda_version),
            'gpu_compatibility':'not_tested; CUDA backend and actual GPU offload require later runtime verification'}
    try:
        if args.download:
            result['status']='downloading'
            write(destination/'preparation.json',result)
            for asset in result['assets']:
                destination.mkdir(parents=True,exist_ok=True)
                def progress(update):
                    result['active_download']=update;write(destination/'preparation.json',result)
                archive=download(asset,destination,progress)
                asset['download']=archive
                result.pop('active_download',None)
                write(destination/'preparation.json',result)
            expanded=[]
            for asset in result['assets']:
                expanded.extend(extract_verified(Path(asset['download']['path']),destination/'runtime'))
            server=list((destination/'runtime').rglob('llama-server.exe'))
            cuda=list((destination/'runtime').rglob('ggml-cuda.dll'))
            if len(server)!=1 or len(cuda)!=1:raise ValueError('server or CUDA backend missing/ambiguous')
            result.update(status='verified_and_extracted_not_executed',server=str(server[0]),cuda_backend=str(cuda[0]),
                          files=expanded,archive_bytes=sum(a['download']['bytes'] for a in result['assets']),
                          expanded_bytes=sum(e['bytes'] for e in expanded))
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    write(destination/'preparation.json',result)
    stamp=result['captured_at'].replace(':','').replace('-','').replace('.','').replace('+','')
    write(destination/'attempts'/(stamp+'.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
