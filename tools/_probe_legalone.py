"""取证：LegalOne-8B 的权重在各镜像站到底能不能取到。

只做 HTTP 探测，不下载权重。输出每个候选地址的**状态码与耗时**，用于回答
"为什么权重缺失"——区分三件事：站点不通 / 仓库不存在 / 仓库存在但要鉴权。
"""
from __future__ import annotations

import json
import sys
import time

import requests

PROBES = [
    ("HuggingFace 主页", "https://huggingface.co/"),
    ("HuggingFace 仓库页", "https://huggingface.co/CSHaitao/LegalOne-8B"),
    ("HuggingFace API", "https://huggingface.co/api/models/CSHaitao/LegalOne-8B"),
    ("HF 镜像 主页", "https://hf-mirror.com/"),
    ("HF 镜像 仓库", "https://hf-mirror.com/CSHaitao/LegalOne-8B/resolve/main/config.json"),
    ("ModelScope 主页", "https://www.modelscope.cn/"),
    ("ModelScope 该仓库", "https://www.modelscope.cn/models/CSHaitao/LegalOne-8B"),
    ("ModelScope 搜索 legalone", "https://www.modelscope.cn/api/v1/dolphin/models?PageSize=10&PageNumber=1&Name=legalone"),
    ("ModelScope 可用对照 Qwen", "https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct"),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for name, url in PROBES:
        started = time.perf_counter()
        try:
            resp = requests.get(url, timeout=20, allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0 zlql-probe"})
            code = resp.status_code
            size = len(resp.content)
            note = ""
            if code == 404:
                note = "仓库/文件不存在"
            elif code == 401 or code == 403:
                note = "需要鉴权"
            elif code == 200:
                note = "可达"
        except Exception as exc:  # noqa: BLE001
            code, size, note = -1, 0, f"{type(exc).__name__}"
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({"name": name, "url": url, "status": code, "bytes": size,
                     "ms": round(elapsed), "note": note})
        print(f"{name:<26} status={code:<5} {size:>8} B  {elapsed:8.0f} ms  {note}")

    # ModelScope 搜索里到底有没有可用的法律模型
    print()
    try:
        resp = requests.get(
            "https://www.modelscope.cn/api/v1/dolphin/models",
            params={"PageSize": 8, "PageNumber": 1, "Name": "LegalOne"},
            timeout=20, headers={"User-Agent": "Mozilla/5.0 zlql-probe"},
        )
        payload = resp.json()
        models = (payload.get("Data") or {}).get("Model", {}).get("Models") or []
        print(f"ModelScope 搜 'LegalOne' → {len(models)} 条")
        for item in models[:8]:
            print("   ", item.get("Path") or item.get("Name"), "|", item.get("ChineseName") or "")
    except Exception as exc:  # noqa: BLE001
        print("ModelScope 搜索失败：", type(exc).__name__, exc)

    out = {"probes": rows}
    json.dump(out, open("zhilv_qiling_v2/reports/legalone_probe.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写 zhilv_qiling_v2/reports/legalone_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
