"""从 ModelScope 拉取模型仓库文件，支持断点续传。

用法::

    python tools/_dl_modelscope.py Qwen/Qwen2.5-1.5B-Instruct zhilv_qiling_v2/models/pretrained/qwen2.5-1.5b-instruct

只下权重与分词器（config/tokenizer/generation_config/*.safetensors），跳过 `.gitattributes`
与示例图。HF 直连在本机不通，ModelScope 是唯一可用源，单连接会被掐，这里按 8 MB 分块请求
并在本地 `.part` 累积，中断后重跑会跳过已完成文件。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import requests

SESS = requests.Session()
SESS.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
RAW = "https://www.modelscope.cn/models/{repo}/resolve/master/{path}"
KEEP = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "model.safetensors.index.json",
)
CHUNK = 8 * 1024 * 1024
RETRY = 5


def list_files(repo: str) -> list[dict]:
    resp = SESS.get(API.format(repo=repo), timeout=30)
    resp.raise_for_status()
    files = (resp.json().get("Data") or {}).get("Files") or []
    out = []
    for item in files:
        path = str(item.get("Path") or "")
        size = int(item.get("Size") or 0)
        if not path or size <= 0:
            continue
        name = path.rsplit("/", 1)[-1]
        if name not in KEEP and not name.endswith(".safetensors"):
            continue
        out.append({"path": path, "size": size})
    return out


def download(repo: str, path: str, size: int, dest: pathlib.Path) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return f"skip {path} ({size/1e9:.2f} GB 已完整)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    started = time.time()
    url = RAW.format(repo=repo, path=path)
    while have < size:
        end = min(have + CHUNK - 1, size - 1)
        headers = {"Range": f"bytes={have}-{end}"}
        for attempt in range(RETRY):
            try:
                with SESS.get(url, headers=headers, timeout=90, stream=True) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    data = resp.content
                if len(data) != end - have + 1:
                    raise RuntimeError(f"块长度不符 {len(data)} != {end-have+1}")
                with part.open("ab") as handle:
                    handle.write(data)
                have += len(data)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == RETRY - 1:
                    raise
                print(f"  重试 {attempt+1}/{RETRY}: {exc}")
                time.sleep(2 + attempt * 2)
    part.replace(dest)
    cost = time.time() - started
    return f"ok   {path} ({size/1e9:.2f} GB / {cost:.0f}s / {size/1e6/max(cost,1e-6):.1f} MB/s)"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("用法: python tools/_dl_modelscope.py <repo> <dest_dir>")
        return 2
    repo, dest = argv[1], pathlib.Path(argv[2])
    files = list_files(repo)
    total = sum(item["size"] for item in files)
    print(f"仓库 {repo}：{len(files)} 个文件 / {total/1e9:.2f} GB → {dest}")
    done = 0
    for item in files:
        print(" ", download(repo, item["path"], item["size"], dest / item["path"].rsplit("/", 1)[-1]))
        done += item["size"]
    print(json.dumps({"repo": repo, "dest": str(dest), "files": len(files), "gb": round(done / 1e9, 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
