"""分块并行下载器：给单连接会被限速/掐断的镜像站用。

背景：`mirror.sjtu.edu.cn` 前 20 MB 能跑到 9 MB/s，但单连接下到 800 MB 左右就停住，
`Invoke-WebRequest` 与 `curl -C -` 都救不回来。这个脚本把文件按固定大小切块，
多线程各自用 `Range` 请求，每块失败重试，已完成块记进 `.chunks.json`，
中断后重跑只补缺块。

用法：
    python tools/_dl_chunked.py <url> <out_path> [--threads 6] [--chunk 16] [--retries 6]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def total_size(url: str, session: requests.Session) -> int:
    """拿文件总大小。

    有些镜像站（如交大 pytorch-wheels）HEAD 不返回 Content-Length，
    改用 `Range: bytes=0-0` 读 `Content-Range: bytes 0-0/<total>`。
    """
    try:
        resp = session.head(url, headers=UA, allow_redirects=True, timeout=30)
        size = int(resp.headers.get("Content-Length") or 0)
        if size > 0:
            return size
    except Exception:
        pass

    resp = session.get(url, headers={**UA, "Range": "bytes=0-0"}, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    content_range = resp.headers.get("Content-Range") or ""
    if "/" in content_range:
        tail = content_range.split("/")[-1].strip()
        if tail.isdigit():
            return int(tail)
    size = int(resp.headers.get("Content-Length") or 0)
    if size > 0:
        return size
    raise RuntimeError(f"服务器没有返回文件大小（Content-Range={content_range!r}）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分块并行下载")
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=16, help="每块 MB")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + ".chunks.json")
    part = out.with_suffix(out.suffix + ".part")

    session = requests.Session()
    size = total_size(args.url, session)
    chunk_bytes = args.chunk * 1024 * 1024
    ranges = [(start, min(start + chunk_bytes, size) - 1) for start in range(0, size, chunk_bytes)]
    done: set[int] = set()
    if state_path.exists():
        try:
            done = set(json.loads(state_path.read_text(encoding="utf-8"))["done"])
        except Exception:
            done = set()
    print(f"[info] 总大小 {size / 1048576:.1f} MB，共 {len(ranges)} 块，已完成 {len(done)} 块", flush=True)

    if not part.exists():
        with part.open("wb") as handle:
            handle.truncate(size)

    lock = threading.Lock()
    finished = 0
    failures: list[int] = []

    def worker(index: int) -> None:
        nonlocal finished
        if index in done:
            with lock:
                finished += 1
            return
        start, end = ranges[index]
        for attempt in range(1, args.retries + 1):
            try:
                resp = session.get(args.url, headers={**UA, "Range": f"bytes={start}-{end}"}, timeout=120)
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                data = resp.content
                if len(data) < (end - start + 1):
                    raise RuntimeError(f"短读 {len(data)}/{end - start + 1}")
                with lock:
                    with part.open("r+b") as handle:
                        handle.seek(start)
                        handle.write(data)
                    done.add(index)
                    finished += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == args.retries:
                    with lock:
                        failures.append(index)
                    print(f"[x] 块 {index} 失败：{exc}", flush=True)
                else:
                    time.sleep(1.5 * attempt)

    pending = [i for i in range(len(ranges)) if i not in done]
    threads: list[threading.Thread] = []
    cursor = 0
    cursor_lock = threading.Lock()

    def feed() -> None:
        nonlocal cursor
        while True:
            with cursor_lock:
                if cursor >= len(pending):
                    return
                index = pending[cursor]
                cursor += 1
            worker(index)

    began = time.time()
    for _ in range(max(1, args.threads)):
        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        threads.append(thread)

    last_report = 0.0
    while any(thread.is_alive() for thread in threads):
        time.sleep(2)
        now = time.time()
        if now - last_report > 10:
            last_report = now
            state_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
            written = part.stat().st_size
            elapsed = max(now - began, 0.001)
            print(f"[进度] 完成块 {len(done)}/{len(ranges)}，文件 {written / 1048576:.0f} MB，用时 {elapsed:.0f}s", flush=True)

    state_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
    if failures:
        print(f"[!] 仍有 {len(failures)} 块失败：{failures[:10]}，重跑本脚本可续传", flush=True)
        return 2

    out.write_bytes(part.read_bytes())
    part.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    print(f"[ok] {out} {out.stat().st_size / 1048576:.1f} MB，用时 {time.time() - began:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
