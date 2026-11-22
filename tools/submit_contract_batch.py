"""Upload a local directory into the durable web queue, without a browser limit."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def call(base, path, data=None, content_type="application/json"):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
    request = urllib.request.Request(base.rstrip("/")+path, data=data,
        headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", "replace")[:1000]) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="合同目录，只读上传DOCX/PDF/TXT/MD")
    parser.add_argument("--base", default="http://127.0.0.1:8610")
    parser.add_argument("--llm", action="store_true", help="明确启用模型；耗时取决于合同和算力")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--draft", action="store_true", help="仅上传，暂不启动审查")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("root必须是现有合同目录")
    if not 1 <= args.limit <= 100000:
        parser.error("limit必须介于1与100000")
    paths = sorted(p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() in (".docx", ".pdf", ".txt", ".md"))[:args.limit]
    if not paths:
        parser.error("目录下没有可上传合同")
    batch = call(args.base, "/api/workspace/batches", {"engine":"multi", "llm":bool(args.llm)})
    key = batch["batch_id"]
    print(f"batch_id={key}", flush=True)
    uploaded, failed = 0, []
    for path in paths:
        try:
            if not 0 < path.stat().st_size <= 25*1024*1024:
                raise ValueError("文件须非空且不超过25MB")
            boundary = "ContractBatch"+uuid.uuid4().hex
            safe_name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
            raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe_name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
            raw += path.read_bytes()+f"\r\n--{boundary}--\r\n".encode()
            call(args.base, f"/api/workspace/batches/{key}/tasks", raw, f"multipart/form-data; boundary={boundary}")
            uploaded += 1
            if uploaded % 100 == 0: print(f"uploaded={uploaded}/{len(paths)}", flush=True)
        except (ValueError, OSError, RuntimeError) as exc:
            failed.append({"file":str(path), "error":str(exc)})
            print(f"UPLOAD FAILED {path.name}: {exc}", file=sys.stderr, flush=True)
    if uploaded and not args.draft:
        call(args.base, f"/api/workspace/batches/{key}/start", {})
    print(json.dumps({"batch_id":key, "uploaded":uploaded, "upload_failed":failed,
        "submitted":bool(uploaded and not args.draft), "status_path":f"/api/workspace/batches/{key}",
        "note":"上传失败项未自动重传，避免响应丢失时重复入队；请核对服务端批次"}, ensure_ascii=False, indent=2))
    return int(not uploaded or bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
