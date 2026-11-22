"""直接探 LegalOne 通道：单条款逻辑判断 + 复核意见，打印原始返回与异常。

用途：区分"模型能力不足（返回空 defects）"与"调用路径 bug（抛异常/解析失败）"。
跑法：
    python tools\\_legalone_probe.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import llm_client as L  # noqa: E402
from zhilv_qiling_v2 import nodes as N  # noqa: E402
from zhilv_qiling_v2 import pipeline  # noqa: E402

ENDPOINT = "http://127.0.0.1:8622/v1"
MODEL = "legalone-1.7b"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "samples" / "tech-development-contract.txt"
    doc = pipeline.prepare(path.name, path.read_bytes())
    contract_type = "技术开发"
    clauses = [c for c in doc.clauses if len(c.text) >= 20][:2]
    print(f"条款 {len(clauses)} 条，类型 {contract_type}")

    cfg = L.LLMConfig(endpoint=ENDPOINT, model=MODEL, timeout=400.0)
    print("available:", L.available(cfg))

    for clause in clauses:
        messages = [
            {"role": "system", "content": N.LOGIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"合同类型：{contract_type}\n"
                    f"条款编号：{clause.label}\n条款文本：\n{clause.text[:1200]}"
                ),
            },
        ]
        try:
            pair = L.request_json(cfg, messages, schema_hint="defects:array")
        except L.LLMClientError as exc:
            print(f"\n[{clause.label}] 异常 {str(exc)[:160]}")
            print("   raw:", (exc.raw or "")[:200].replace("\n", " "))
            continue
        defects = pair.content.get("defects") or []
        print(f"\n[{clause.label}] {pair.elapsed:.1f}s defects={len(defects)}")
        for item in list(defects)[:3]:
            code = N._coerce_code(str(item.get("code", "")))
            quote = str(item.get("quote") or "")
            ok = N._locate_quote(clause, quote)
            print(f"   code={code or '(码表外)'} quote_ok={ok} msg={str(item.get('message'))[:50]}")
        if not defects:
            print("   原文:", clause.text[:80].replace("\n", " "))

    judge_messages = [
        {"role": "system", "content": N.JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "审查结果：评分 83，风险 3 条（vague_delivery 0.73 / need_authorization 0.63 / "
                "missing_party_info 0.55）。合同类型：技术开发。请给复核意见。"
            ),
        },
    ]
    try:
        pair = L.request_json(cfg, judge_messages, schema_hint="summary:string, focus:array")
        print(f"\n[judge] {pair.elapsed:.1f}s ->", json.dumps(pair.content, ensure_ascii=False)[:200])
    except L.LLMClientError as exc:
        print(f"\n[judge] 异常 {str(exc)[:200]}")
        print("   raw:", (exc.raw or "")[:300].replace("\n", " "))


if __name__ == "__main__":
    main()

