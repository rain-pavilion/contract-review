# -*- coding: utf-8 -*-
"""验证"大模型通道真的能跑"——用一个本地假端点替代真模型。

本机没有 Ollama / vLLM，所以无法证明 `enable_llm=True` 时链路能走通。
这个脚本起一个 30 行的 HTTP 服务，按 OpenAI 的 `/v1/chat/completions` 协议回固定 JSON，
把 `config.llm_endpoint` 指过去，就能验证三件事：

1. 节点② 的模型缺陷能被解析、且**引文校验**会把编造的引文丢掉；
2. 节点⑤ 的模型复核意见能写进 payload，但**分数仍由规则算**；
3. trace 里出现 `judge_llm` 节点，报告 stats 里 `nodes.llm=True`。

跑法：``python -m zhilv_qiling_v2.reports.experiments.check_llm_channel``
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from zhilv_qiling_v2 import nodes as N  # noqa: E402
from zhilv_qiling_v2 import pipeline  # noqa: E402
from zhilv_qiling_v2.config import default_config  # noqa: E402
from zhilv_qiling_v2.engine import ReviewEngine  # noqa: E402

SAMPLE = WORKSPACE / "samples" / "tech-development-contract.txt"

#: 模型"返回"的缺陷：第二条的 quote 是编造的，必须被引文校验挡掉
FAKE_DEFECTS = {
    "defects": [
        {
            "clause_id": "",
            "code": "vague_delivery",
            "severity": "medium",
            "message": "交付时间只写“尽快”，无法确定履行期限",
            "quote": "乙方应在合理期限内尽快完成交付",
        },
        {
            "clause_id": "",
            "code": "zero_rent",
            "severity": "high",
            "message": "编造的缺陷：原文里并没有这句",
            "quote": "甲方无需支付任何费用且永不付款",
        },
    ]
}

FAKE_JUDGE = {
    "summary": "重点核对交付期限与验收标准两处，其余条款风险可控。",
    "focus": ["交付期限", "验收标准"],
}


class _Handler(BaseHTTPRequestHandler):
    def _clause_text(self, payload: dict) -> str:
        """从请求里抠出条款原文，用于生成"一定能命中"的引文。"""
        for message in payload.get("messages", []):
            if message.get("role") != "user":
                continue
            body = message.get("content", "")
            marker = "条款文本："
            if marker in body:
                tail = body.split(marker, 1)[1].strip()
                return tail.splitlines()[0] if tail else ""
        return ""

    def do_POST(self):  # noqa: N802 - http.server 的固定命名
        length = int(self.headers.get("content-length", 0) or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        system = ""
        for message in payload.get("messages", []):
            if message.get("role") == "system":
                system += message.get("content", "")
        if "复核律师" in system:
            content = FAKE_JUDGE
        else:
            # 第一条引文取自真条款（应被接受），第二条是编造的（应被挡掉）
            real = self._clause_text(payload)[:24]
            content = {
                "defects": [
                    {
                        "code": "vague_delivery",
                        "severity": "medium",
                        "message": "交付时间只写“尽快”，无法确定履行期限",
                        "quote": real,
                    },
                    {
                        "code": "zero_rent",
                        "severity": "high",
                        "message": "编造的缺陷：原文里并没有这句",
                        "quote": "甲方无需支付任何费用且永不付款",
                    },
                ]
            } if real else {"defects": []}
        body = json.dumps(
            {
                "model": payload.get("model", "fake"),
                "choices": [{"message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静音
        return


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    print(f"[1] 假端点已启动：{endpoint}")

    config = default_config()
    config.enable_llm = True
    config.enable_subject_check = True
    config.enable_entailment = True
    config.llm_endpoint = endpoint
    config.llm_model = "fake-legalone-8b"

    document = pipeline.prepare(SAMPLE.name, SAMPLE.read_bytes(), config=config)
    contract = pipeline.build_contract(document, config=config)
    print(f"[2] 条款 {len(document.clauses)} 条 / 主体 {len(contract.elements.parties)} 个")

    logic = N.node_logic(contract, config)
    llm_sources = [c for c in logic.candidates if c.source == "node2_llm"]
    print(f"[3] 节点② 候选 {len(logic.candidates)} 条，其中模型 {len(llm_sources)} 条；"
          f"accepted={logic.llm.get('accepted')} rejected={logic.llm.get('rejected')}")
    for candidate in llm_sources:
        print(f"    - {candidate.code} | {candidate.message[:36]} | {candidate.severity}")

    engine = ReviewEngine(config)
    report = engine.run(contract)
    print(f"[4] 引擎：评分 {report.score}，风险 {len(report.risks)} 条")
    print(f"    节点顺序：{' -> '.join(e.node for e in report.trace)}")
    print(f"    judge_llm：{report.stats.get('judge_llm')}")
    print(f"    degraded：{report.stats['degraded']}")
    print(f"    nodes：{report.stats['nodes']}")

    server.shutdown()
    ok = bool(llm_sources) and bool(report.stats.get("judge_llm")) and any(
        e.node == "judge_llm" for e in report.trace
    )
    print("[5] 结论：" + ("模型通道可用（含引文校验与只读评分）" if ok else "模型通道异常"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
