"""Fetch a sparse official LawBench commit snapshot, verifying Git blob hashes.

Alternative to Git smart HTTP when that transport is unavailable. This is an
official source snapshot, not a fabricated Git repository or benchmark result.
"""
from __future__ import annotations
import argparse, concurrent.futures, datetime, hashlib, json, time, urllib.request
from pathlib import Path
from acceptance_common import ROOT, digest, write_json

REPOSITORY='https://github.com/open-compass/LawBench'
REVISION='e30981bb3ff54c41571f222e0b23e92d27375388'

def fetch(url):
    last=None
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'ZLQL-official-benchmark-acceptance','Accept':'application/vnd.github+json'})
            with urllib.request.urlopen(request,timeout=30) as response:return response.read()
        except OSError as exc:last=exc;time.sleep(attempt+.1)
    raise last

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dest',type=Path,default=ROOT/'tools/acceptance_vendor/LawBench');a=p.parse_args()
    if a.dest.exists() and any(a.dest.iterdir()):p.error('destination must be empty; source snapshots are immutable')
    a.dest.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/open-compass/LawBench/git/trees/{REVISION}?recursive=1'))
    if tree.get('truncated'):raise RuntimeError('refuse truncated official Git tree')
    entries=[x for x in tree['tree'] if x['type']=='blob' and (x['path'].startswith(('data/','evaluation/')) or x['path'] in ('README.md','README_EN.md','LICENSE','LICENSE.md','requirements.txt'))]
    def download(entry):
        relative=entry['path'];path=a.dest/relative
        if not path.resolve().is_relative_to(a.dest.resolve()):raise ValueError('unsafe upstream path')
        url=f'https://raw.githubusercontent.com/open-compass/LawBench/{REVISION}/{relative}'
        data=fetch(url);blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=entry['sha']:raise ValueError('official Git blob mismatch: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return {'path':relative,'git_blob_sha':blob,'sha256':digest(data),'bytes':len(data),'source_url':url}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(download,entries))
    result={'schema_version':'official-source-snapshot-v1','repository':REPOSITORY,'revision':REVISION,
        'git_tree_sha':tree['sha'],'fetched_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'transport':'GitHub primary Git Tree API + immutable raw commit URLs','snapshot_is_git_checkout':False,
        'integrity':'each file verified against pinned official Git blob SHA-1 and separately SHA-256',
        'files':sorted(files,key=lambda x:x['path'])}
    write_json(a.dest/'official_snapshot.json',result)
    print(json.dumps({'status':'downloaded_verified_official_snapshot','revision':REVISION,'files':len(files),'bytes':sum(x['bytes'] for x in files)},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
