"""对比：同一提示词走 `llm_client.request_json` 与裸 POST，定位差异。"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import llm_client as L  # noqa: E402
from zhilv_qiling_v2 import nodes as N  # noqa: E402
from zhilv_qiling_v2 import pipeline  # noqa: E402

URL = "http://127.0.0.1:8622/v1/chat/completions"


def raw_post(messages: list[dict[str, str]], max_tokens: int) -> str:
    body = json.dumps(
        {"model": "legalone-1.7b", "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())["choices"][0]["message"]["content"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "samples" / "tech-development-contract.txt"
    doc = pipeline.prepare(path.name, path.read_bytes())
    clause = [c for c in doc.clauses if len(c.text) >= 20][0]
    messages = [
        {"role": "system", "content": N.LOGIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"合同类型：技术开发\n条款编号：第1条\n条款文本：\n{clause.text[:1200]}"},
    ]

    print("=== 裸 POST（无 schema_hint 合并） ===")
    text = raw_post(messages, 512)
    print(f"{len(text)} 字 / 尾部 {text[-60:]!r}")

    print("=== request_json + schema_hint ===")
    cfg = L.LLMConfig(endpoint="http://127.0.0.1:8622/v1", model="legalone-1.7b",
                      timeout=300.0, max_tokens=512)
    try:
        pair = L.request_json(cfg, messages, schema_hint="defects:array")
        print("OK", pair.elapsed, pair.content)
    except L.LLMClientError as exc:
        print("ERR", str(exc)[:90])
        print("RAW:", (exc.raw or "")[:400])


if __name__ == "__main__":
    main()
