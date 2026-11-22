"""从 hf-mirror 分块并行下载 HuggingFace 仓库（Windows 单连接会被掐断，必须分块）。

用法：
    python tools\\_dl_hf.py --repo CSHaitao/LegalOne-1.7B --dest zhilv_qiling_v2\\models\\pretrained\\legalone-1.7b
    python tools\\_dl_hf.py --repo CSHaitao/LegalOne-8B --dest ... --only "config.json,tokenizer*"
    python tools\\_dl_hf.py --repo X/Y --dest Z --list          # 只列清单

设计要点（都是踩过的坑）：
* 单连接下载到 800 MB 左右会被限速到 0，必须 `Range` 分块 + 多线程；
* 镜像的 `HEAD` 不一定回 `Content-Length`，`total_size()` 回落到 `Range: bytes=0-0` 读
  `Content-Range: bytes 0-0/<total>`；
* 已下完的块记在 `<file>.chunks.json`，中断后重跑接着下；
* `.part` 预分配后随机写，最后 `os.replace` 成正式名。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CHUNK = 16 * 1024 * 1024


def list_repo(repo: str, *, mirror: str = MIRROR) -> list[dict]:
    url = f"{mirror}/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("siblings") or [])


def _request(url: str, *, method: str = "GET", headers: dict | None = None, timeout: int = 60):
    merged = dict(UA)
    if headers:
        merged.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=merged, method=method), timeout=timeout)


def total_size(url: str, *, mirror: str = MIRROR, timeout: int = 60) -> int:
    """先试 HEAD，拿不到 Content-Length 就用 1 字节 Range 探 Content-Range。

    镜像会偶发 503 与 SSL 握手超时（实测同一条 URL 前一秒 503、后一秒 206），
    所以这里必须自带重试，否则整份下载在第一步就崩。
    """
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            with _request(url, method="HEAD", timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                if length and int(length) > 0:
                    return int(length)
        except Exception as exc:
            last = exc
        try:
            with _request(url, headers={"Range": "bytes=0-0"}, timeout=timeout) as resp:
                content_range = resp.headers.get("Content-Range") or ""
                if "/" in content_range:
                    return int(content_range.rsplit("/", 1)[1])
                length = resp.headers.get("Content-Length")
                if length:
                    return int(length)
        except Exception as exc:
            last = exc
        time.sleep(2.0 * attempt)
    raise RuntimeError(f"拿不到文件大小：{url}（{last}）")


def download(url: str, dest: pathlib.Path, *, mirror: str = MIRROR, threads: int = 4,
             chunk: int = CHUNK, known_size: int = 0, delay: float = 0.0) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.name}（已存在 {dest.stat().st_size/1e9:.2f} GB）")
        return dest

    size = known_size or total_size(url, mirror=mirror)

    # 小文件（配置/词表）直接整份 GET：镜像对小文件偶发 503，走 Range 分块反而更容易
    # 命中限流；整份 GET 一次成功就完事，失败重试 8 次。
    if size <= 1_000_000:
        last: Exception | None = None
        for attempt in range(1, 13):
            try:
                with _request(url, timeout=60) as resp:
                    payload = resp.read()
                if len(payload) != size:
                    raise IOError(f"长度不符 {len(payload)} != {size}")
                dest.write_bytes(payload)
                print(f"  small {dest.name}  {size/1e3:.0f} KB")
                return dest
            except Exception as exc:
                last = exc
                time.sleep(min(6.0, 0.8 * attempt))
        print(f"  WARN  {dest.name} 跳过（{last}）")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    state_path = dest.with_suffix(dest.suffix + ".chunks.json")
    done: set[int] = set()
    if state_path.exists():
        try:
            done = set(json.loads(state_path.read_text(encoding="utf-8")).get("done") or [])
        except Exception:
            done = set()
    if not part.exists() or part.stat().st_size != size:
        with open(part, "wb") as handle:
            handle.truncate(size)

    offsets = list(range(0, size, chunk))
    todo = [start for start in offsets if start not in done]
    total_chunks = len(offsets)
    print(f"  file  {dest.name}  {size/1e9:.2f} GB  {total_chunks} 块  待下 {len(todo)} 块")
    if not todo:
        os.replace(part, dest)
        state_path.unlink(missing_ok=True)
        return dest

    lock = threading.Lock()
    fetched = {"bytes": 0, "chunks": 0}
    began = time.time()

    def fetch(start: int) -> None:
        end = min(start + chunk, size) - 1
        for attempt in range(1, 9):
            try:
                with _request(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90) as resp:
                    payload = resp.read()
                if len(payload) != end - start + 1:
                    raise IOError(f"块长度不符 {len(payload)} != {end - start + 1}")
                with lock:
                    with open(part, "r+b") as handle:
                        handle.seek(start)
                        handle.write(payload)
                    done.add(start)
                    fetched["bytes"] += len(payload)
                    fetched["chunks"] += 1
                    if fetched["chunks"] % 8 == 0 or fetched["chunks"] == len(todo):
                        rate = fetched["bytes"] / max(1e-6, time.time() - began) / 1e6
                        print(f"    {fetched['chunks']}/{len(todo)} 块  {rate:5.1f} MB/s", flush=True)
                    state_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
                if delay:
                    time.sleep(delay)
                return
            except Exception as exc:
                if attempt == 5:
                    raise RuntimeError(f"块 {start} 下载失败：{exc}") from exc
                time.sleep(min(12.0, 1.5 * attempt * attempt))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(fetch, todo))

    if len(done) >= total_chunks:
        os.replace(part, dest)
        state_path.unlink(missing_ok=True)
    return dest


def resolve_url(repo: str, filename: str, *, mirror: str = MIRROR) -> str:
    return f"{mirror}/{repo}/resolve/main/{filename}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hf-mirror 分块并行下载")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--only", default="", help="逗号分隔的通配符，例如 'config.json,tokenizer*'")
    parser.add_argument("--skip", default="fig/,*.md,*.png", help="逗号分隔的排除通配符")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mirror", default=MIRROR)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.0, help="每块完成后停顿秒数（限流时用）")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    files = list_repo(args.repo, mirror=args.mirror)
    total = sum(int(item.get("size") or 0) for item in files)
    if args.list:
        for item in files:
            print(f"  {item['rfilename']:<46} {(item.get('size') or 0)/1e9:8.3f} GB")
        print(f"  TOTAL {total/1e9:.2f} GB")
        return 0

    patterns = [p.strip() for p in args.only.split(",") if p.strip()]
    skips = [p.strip() for p in args.skip.split(",") if p.strip()]

    def wanted(name: str) -> bool:
        if patterns:
            return any(fnmatch.fnmatch(name, pat) for pat in patterns)
        return not any(fnmatch.fnmatch(name, pat) or name.startswith(pat) for pat in skips)

    dest_root = pathlib.Path(args.dest)
    picked = [item for item in files if wanted(item["rfilename"])]
    heavy = sum(int(item.get("size") or 0) for item in picked)
    print(f"仓库 {args.repo}：{len(files)} 个文件 / 全量 {total/1e9:.2f} GB；本次下 {len(picked)} 个 / {heavy/1e9:.2f} GB")
    began = time.time()
    small = [i for i in picked if int(i.get("size") or 0) <= 1_000_000]
    heavy_files = [i for i in picked if int(i.get("size") or 0) > 1_000_000]

    def one(item: dict) -> None:
        name = item["rfilename"]
        try:
            download(resolve_url(args.repo, name, mirror=args.mirror), dest_root / name,
                     mirror=args.mirror, threads=args.threads, known_size=int(item.get("size") or 0),
                     chunk=args.chunk_mb * 1024 * 1024, delay=args.delay)
        except Exception as exc:
            print(f"  FAIL  {name}: {type(exc).__name__} {str(exc)[:120]}")

    # 小文件并发下：镜像抖动是概率性的，串行时一个 503 会把整轮拖成十几分钟
    if small:
        print(f"小文件 {len(small)} 个，4 路并发")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, small))
    for item in heavy_files:
        one(item)
    print(json.dumps({"repo": args.repo, "dest": str(dest_root), "files": len(picked),
                      "gb": round(sum((dest_root / i["rfilename"]).stat().st_size for i in picked
                                      if (dest_root / i["rfilename"]).exists()) / 1e9, 2),
                      "seconds": round(time.time() - began, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
