"""Download the pinned, author-linked LCR-CN v4 release, verifying every byte.

Downloads data only. Never executes source-repository code or creates labels.
Large raw files stay outside version control; incomplete files are preserved.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'tools/acceptance_vendor/LCR_CN_20261115'
FILES = (
    ('train.jsonl', 61538128, 555483136, '274201bbc5a054cfb9d9d831fb4c2376'),
    ('val.jsonl', 61538131, 80302328, '1fffc40d630701c8b6ff10020dcd812b'),
    ('test.jsonl', 61538143, 160535784, '96d898bb9c8705683d210c4a91b02858'),
)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return size, md5.hexdigest(), sha.hexdigest()


def main():
    metadata = json.loads((DEST / 'figshare_metadata.json').read_text(encoding='utf-8'))
    if metadata['doi'] != '10.6084/m9.figshare.30315955.v4' or metadata['license']['name'] != 'CC BY 4.0':
        raise ValueError('Pinned release or licence changed; inspect primary metadata')
    raw_dir = DEST / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    (DEST / '.gitignore').write_text('/raw/\n', encoding='ascii')
    records = []
    for name, identifier, expected_size, expected_md5 in FILES:
        url = f'https://ndownloader.figshare.com/files/{identifier}'
        official = next(row for row in metadata['files'] if row['id'] == identifier)
        if (official['name'], official['size'], official['computed_md5'], official['download_url']) != (
                name, expected_size, expected_md5, url):
            raise ValueError('File metadata differs from pinned release')
        target = raw_dir / name
        partial = raw_dir / (name + '.partial')
        if not target.exists():
            if partial.exists():
                raise ValueError(f'Incomplete download preserved: {partial}; inspect before resuming')
            request = urllib.request.Request(url, headers={'User-Agent': 'ZLQL-source-verification'})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open('xb') as output:
                final_url = response.url
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    size += len(block)
                    if size > expected_size:
                        raise ValueError('Response exceeded pinned file size')
                    output.write(block)
            actual_size, md5, sha = hashes(partial)
            if actual_size != expected_size or md5 != expected_md5:
                raise ValueError('Partial evidence preserved: official byte count or MD5 mismatch')
            partial.rename(target)
        else:
            final_url = 'existing locally; original release URL re-used only for identity'
            actual_size, md5, sha = hashes(target)
            if actual_size != expected_size or md5 != expected_md5:
                raise ValueError('Existing source file differs from official release; not overwritten')
        row = {'file': str(target.relative_to(ROOT)), 'source_url': url, 'final_url': final_url,
               'bytes': actual_size, 'official_md5': md5, 'sha256': sha}
        records.append(row)
        print(json.dumps({'verified': name, 'bytes': actual_size, 'sha256': sha}), flush=True)
    receipt = {'status': 'official_release_downloaded_hash_verified',
               'captured_at_utc': datetime.now(timezone.utc).isoformat(),
               'author_repository': 'https://github.com/Zhaozhao77/LCR-CN',
               'repository_commit': 'c8672137493899accaaac63f6858d0364762ba18',
               'doi': metadata['doi'], 'license': metadata['license'], 'files': records,
               'task': 'superior/subordinate legislative conflict; not contract-to-law PROVDATA',
               'labels_created_or_changed': False, 'source_code_executed': False}
    out = DEST / 'download_receipt.json'
    if out.exists():
        raise ValueError('Existing receipt is preserved; inspect rather than overwrite')
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
