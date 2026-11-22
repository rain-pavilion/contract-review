"""用节点真实提示词打一次模型，原样打印输出，定位"修补后仍失败"。"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import nodes as N  # noqa: E402
from zhilv_qiling_v2 import pipeline  # noqa: E402

URL = "http://127.0.0.1:8622/v1/chat/completions"


def call(messages: list[dict[str, str]], max_tokens: int = 400) -> str:
    body = json.dumps(
        {"model": "legalone-1.7b", "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode())
    return data["choices"][0]["message"]["content"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "samples" / "tech-development-contract.txt"
    doc = pipeline.prepare(path.name, path.read_bytes())
    clause = [c for c in doc.clauses if len(c.text) >= 20][0]

    messages = [
        {"role": "system", "content": N.LOGIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"合同类型：技术开发\n条款编号：第1条\n条款文本：\n{clause.text[:1200]}",
        },
        {"role": "system", "content": "只输出 JSON，字段要求：defects:array"},
    ]
    started = time.perf_counter()
    text = call(messages)
    print(f"耗时 {time.perf_counter() - started:.1f}s / {len(text)} 字")
    print("---- 完整输出 ----")
    print(text)
    print("---- 结束 ----")

    from zhilv_qiling_v2.llm_client import _extract_json

    try:
        print("解析:", _extract_json(text))
    except Exception as exc:  # noqa: BLE001
        print("解析失败:", str(exc)[:100])


if __name__ == "__main__":
    main()
