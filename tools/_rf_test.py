"""验证 response_format 是否是"模型只吐骨架"的开关。"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import llm_client as L, nodes as N, pipeline as P  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "samples" / "tech-development-contract.txt"
    doc = P.prepare(path.name, path.read_bytes())
    clause = [c for c in doc.clauses if len(c.text) >= 20][0]
    messages = [
        {"role": "system", "content": N.LOGIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"合同类型：技术开发\n条款编号：第1条\n条款文本：\n{clause.text[:1200]}",
        },
    ]
    for fmt in ("json_object", ""):
        cfg = L.LLMConfig(
            endpoint="http://127.0.0.1:8622/v1",
            model="legalone-1.7b",
            timeout=300.0,
            max_tokens=512,
            response_format=fmt,
        )
        try:
            pair = L.request_json(cfg, messages, schema_hint="")
            defects = pair.content.get("defects") or []
            print(f"rf={fmt!r:14} OK {pair.elapsed:6.1f}s defects={len(defects)}")
            print("   ", (pair.raw or "")[:150].replace("\n", " "))
        except L.LLMClientError as exc:
            print(f"rf={fmt!r:14} ERR {pair if False else str(exc)[:60]}")
            print("    RAW:", (exc.raw or "")[:150].replace("\n", " "))


if __name__ == "__main__":
    main()
