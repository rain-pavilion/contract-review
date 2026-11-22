"""Read-only public primary-source availability audit; never fabricates PROVDATA."""
from __future__ import annotations
import argparse
import datetime
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from acceptance_common import ROOT, digest, provenance, write_json

PAPER='https://aclanthology.org/2025.acl-long.312/'
REPO='https://github.com/labpaper/ProvBench'


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        row=dict(attrs)
        if tag=='a' and row.get('href'):
            self.links.append(row['href'])


def fetch(url,folder,key,timeout=30):
    result={'url':url,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    req=urllib.request.Request(url,headers={'User-Agent':'ZLQL primary-source availability audit','Accept':'*/*'})
    try:
        response=urllib.request.urlopen(req,timeout=timeout)
    except urllib.error.HTTPError as exc:
        response=exc
    except OSError as exc:
        result.update(http_status=None,error=f'{type(exc).__name__}: {exc}')
        return result,b''
    with response:
        raw=response.read(32*1024*1024+1)
        result.update(http_status=response.status,final_url=response.url,
                      content_type=response.headers.get('Content-Type'))
    if len(raw)>32*1024*1024:
        raise ValueError('unexpected response larger than 32 MiB')
    target=folder/'raw'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    result.update(bytes=len(raw),sha256=digest(raw),saved_response=str(target.resolve()))
    return result,raw


def run(a):
    if a.out_dir.exists():
        raise ValueError('use a fresh --out-dir; preserve previous availability checks')
    a.out_dir.mkdir(parents=True)
    (a.out_dir/'.gitignore').write_text('raw/\n',encoding='utf-8')
    checks=[]
    def get(url,name):
        result,raw=fetch(url,a.out_dir,name,a.timeout);checks.append(result);return result,raw
    page_check,page=get(PAPER,'paper.html')
    pdf_check,pdf=get(PAPER.rstrip('/')+'.pdf','paper.pdf')
    parser=Links();parser.feed(page.decode('utf-8','replace'))
    absolute=[urllib.parse.urljoin(PAPER,url) for url in parser.links]
    attachments=sorted(set(url for url in absolute if
        (re.search(r'\.(zip|tar|gz|tgz)$',urllib.parse.urlparse(url).path,re.I) or
         re.search(r'2025\.acl-long\.312\.(software|attachments|supp)',url,re.I))))
    attachment_checks=[]
    for index,url in enumerate(attachments):
        status,_=get(url,f'attachment_{index}.bin');attachment_checks.append(status)
    pdf_urls=[]
    if pdf_check.get('http_status')==200:
        import fitz
        document=fitz.open(stream=pdf,filetype='pdf')
        pdf_urls=sorted(set(link['uri'] for page in document for link in page.get_links() if link.get('uri')))
    for url,name in [
        (REPO,'repository.html'),
        ('https://api.github.com/repos/labpaper/ProvBench','repository_api.json'),
        ('https://api.github.com/repos/labpaper/ProvBench/releases','releases.json'),
        ('https://raw.githubusercontent.com/labpaper/ProvBench/main/README.md','README_main.md'),
        ('https://raw.githubusercontent.com/labpaper/ProvBench/master/README.md','README_master.md')]:
        get(url,name)
    metadata_records=[]
    # ACL exposes its deployed source commit in the page footer. Pin that commit,
    # rather than assuming the moving master branch describes this paper page.
    commits=[re.search(r'/acl-anthology/tree/([0-9a-f]{40})',url) for url in absolute]
    commit=next((match.group(1) for match in commits if match),None)
    if commit:
        for filename in ('2025.acl.xml','2025.acl-long.xml'):
            url=f'https://raw.githubusercontent.com/acl-org/acl-anthology/{commit}/data/xml/{filename}'
            status,raw=get(url,filename)
            if status.get('http_status')!=200:continue
            collection=ET.fromstring(raw)
            for volume in collection.findall('volume'):
                for paper in volume.findall('paper'):
                    title=''.join(paper.find('title').itertext()) if paper.find('title') is not None else ''
                    if paper.get('id')=='312' and 'ProvBench' in title:
                        entry=ET.tostring(paper,encoding='unicode')
                        (a.out_dir/'official_paper_metadata.xml').write_text(entry,encoding='utf-8')
                        metadata_records.append({'source_url':url,'deployed_commit':commit,'volume_id':volume.get('id'),
                            'record_sha256':digest(entry),'attachment_elements':[ET.tostring(x,encoding='unicode') for x in paper if 'attach' in x.tag or 'supp' in x.tag],
                            'all_record_urls':sorted(set(re.findall(r'https?://[^\s<>\"]+',entry)))})
    discovery=[]
    for query in ('ProvBench','ProvData'):
        status,raw=get('https://huggingface.co/api/datasets?search='+query+'&limit=100',f'huggingface_{query}.json')
        if status.get('http_status')==200:
            data=json.loads(raw)
            discovery.append({'query':query,'matching_dataset_ids':[row.get('id') for row in data],
                'assurance':'discovery metadata only; an unlinked third-party match is not an authenticated official release'})
    repo_status=next((row for row in checks if row['url']==REPO),{})
    paper_ok=page_check.get('http_status')==200 and REPO in [url.rstrip('/') for url in pdf_urls]
    result={'status':'official_release_still_unavailable' if paper_ok and repo_status.get('http_status')==404 else 'inspect_availability_evidence',
        'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'benchmark':'ProvBench','official_identity_verified':paper_ok,'paper_url':PAPER,
        'paper_pdf_uri_annotations':pdf_urls,'paper_page_data_or_software_attachment_links':attachments,
        'attachment_checks':attachment_checks,'acl_official_metadata_records':metadata_records,
        'dataset_discovery':discovery,'checks':checks,
        'official_original_data_downloaded':False,'official_evaluator_downloaded':False,'formal_benchmark_run':False,
        'interpretation':'HTTP 404 establishes unavailable public access at this time, not whether private, deleted, renamed or unpublished. No synthetic or third-party data is substituted for official PROVDATA.',
        'provenance':provenance()}
    write_json(a.out_dir/'availability.json',result)
    blocked=any('10013' in str(row.get('error','')) for row in checks)
    return result,4 if blocked else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'tools/acceptance_results/provbench_recheck_20261114')
    parser.add_argument('--timeout',type=float,default=30)
    args=parser.parse_args()
    result,code=run(args)
    print(json.dumps({'status':result['status'],'official_identity_verified':result['official_identity_verified'],
        'out':str(args.out_dir/'availability.json'),'http_statuses':[(row['url'],row.get('http_status')) for row in result['checks']]},ensure_ascii=False))
    return code


if __name__=='__main__':raise SystemExit(main())
