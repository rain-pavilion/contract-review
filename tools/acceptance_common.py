"""Shared standard-library helpers for reproducible acceptance tools."""
from __future__ import annotations
import datetime, hashlib, json, math, os, platform, re, subprocess, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.web-deps').exists():
    sys.path.insert(0, str(ROOT / '.web-deps'))

def digest(data):
    if isinstance(data, str): data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def runtime_platform():
    # Python 3.12 platform.platform() queries WMI on Windows. A missing native
    # WMI DLL can terminate the interpreter rather than raise a Python error.
    # GetVersionEx supplies the OS version needed for reproducibility directly.
    if sys.platform == 'win32':
        version = sys.getwindowsversion()
        return f'Windows-{version.major}.{version.minor}.{version.build}-{version.service_pack or "SP0"}'
    return platform.platform()

def read_git_revision(root):
    """Read a full SHA-1/SHA-256 revision without executing Git or following escapes."""
    try:
        git_dir=Path(root)/'.git'
        if not git_dir.is_dir():return None
        git_dir=git_dir.resolve()
        def read_within(relative):
            target=(git_dir/relative).resolve()
            if not target.is_relative_to(git_dir):return None
            return target.read_text(encoding='utf-8').strip()
        def revision(value):
            return value.lower() if value and re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})',value) else None
        head=read_within('HEAD')
        direct=revision(head)
        if direct:return direct
        if not head or not head.startswith('ref: '):return None
        ref=head[5:]
        if not re.fullmatch(r'refs/(?:[A-Za-z0-9_-][A-Za-z0-9._-]*/)*[A-Za-z0-9_-][A-Za-z0-9._-]*',ref):return None
        if any(part in ('.','..') for part in ref.split('/')) or '..' in ref:return None
        try:loose=read_within(ref)
        except FileNotFoundError:loose=None
        if loose is not None:return revision(loose)
        try:packed=read_within('packed-refs')
        except FileNotFoundError:return None
        for line in (packed or '').splitlines():
            if line.startswith(('#','^')):continue
            parts=line.split(' ',1)
            if len(parts)==2 and parts[1]==ref:return revision(parts[0])
    except (OSError,UnicodeError):pass
    return None

def provenance():
    skip_reason = os.environ.get('ZHILV_ACCEPTANCE_SKIP_GIT_PROVENANCE_REASON', '').strip()
    revision = None
    if not skip_reason:
        revision=read_git_revision(ROOT)
    names=sorted(p.name for p in (ROOT/'zhilv_qiling_v2').glob('*.py'))+['webui.html']
    return {'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':sys.version,'platform':runtime_platform(),'git_revision':revision,
        'platform_method':'sys.getwindowsversion' if sys.platform == 'win32' else 'platform.platform',
        'git_revision_status':'captured' if revision else 'not_captured',
        'git_revision_method':'explicit_skip' if skip_reason else 'git_metadata_read',
        'git_revision_not_captured_reason':skip_reason or ('safe local .git metadata unavailable or invalid' if revision is None else None),
        'product_source_sha256':{name:digest((ROOT/'zhilv_qiling_v2'/name).read_bytes()) for name in names},
        'warning':'git revision alone excludes uncommitted changes; source SHA-256 values identify selected actual files'}

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def percentile(values, q):
    values = sorted(values)
    if not values: return None
    x = (len(values)-1)*q; low = int(x); high = min(low+1, len(values)-1)
    return round(values[low] + (values[high]-values[low])*(x-low), 6)

def summary(values):
    return {'n': len(values), 'p50_seconds': percentile(values,.5),
            'p95_seconds': percentile(values,.95), 'min_seconds': min(values) if values else None,
            'max_seconds': max(values) if values else None}

def request(base, path, data=None, *, raw=None, content_type='application/json', timeout=600):
    if data is not None: raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(base.rstrip('/')+path, data=raw, headers={'Content-Type':content_type})
    started=time.perf_counter()
    try: response=urllib.request.urlopen(req,timeout=timeout)
    except urllib.error.HTTPError as exc: response=exc
    with response:
        content=response.read(); status=response.status
    elapsed=time.perf_counter()-started
    try: value=json.loads(content)
    except ValueError: value={'raw':content.decode('utf-8','replace')[:1000]}
    return status,value,elapsed

def multipart(name,content,fields=None):
    boundary='Acceptance'+uuid.uuid4().hex
    parts=[]
    for key,value in (fields or {}).items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode())
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+content+b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts),'multipart/form-data; boundary='+boundary

def require(status,value,expected=200):
    if status != expected: raise AssertionError(f'HTTP {status} expected {expected}: {str(value)[:800]}')
    return value
