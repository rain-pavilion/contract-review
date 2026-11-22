"""LocalLLM 通道冒烟：验证 --no-think 后的输出形态与耗时。"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")

from zhilv_qiling_v2.llm_client import _extract_json  # noqa: E402

URL = "http://127.0.0.1:8622/v1/chat/completions"
PROMPT = (
    "你是政府采购合同审查专家。只依据给定条款判断缺陷。"
    "可用风险码：vague_delivery, payment_unclear, unilateral_modify。"
    '只输出 JSON：{"defects":[{"code":"...","severity":"medium","message":"...","quote":"原文片段"}]}，'
    '没有缺陷返回 {"defects":[]}。\n条款：乙方应当在甲方要求时及时交付成果。'
)


def ask(prompt: str, max_tokens: int = 600) -> str:
    body = json.dumps(
        {
            "model": "legalone-1.7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode())
    return data["choices"][0]["message"]["content"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.perf_counter()
    text = ask(PROMPT)
    elapsed = time.perf_counter() - started
    has_think = "<think" in text
    print(f"{elapsed:.1f}s / {len(text)} 字 / 含 think={has_think}")
    print("尾部:", text[-200:].replace("\n", " "))
    try:
        print("解析:", _extract_json(text))
    except Exception as exc:  # noqa: BLE001
        print("解析失败:", str(exc)[:120])


if __name__ == "__main__":
    main()
