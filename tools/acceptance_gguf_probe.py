"""One-shot real native LegalOne-8B transport/JSON smoke tests, never a benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import llm_client as C, model_audit as A


def get(base: str, path: str):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=5) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8620")
    parser.add_argument("--out", type=Path, default=ROOT / "tools/acceptance_results/legalone8b_gguf_probe.json")
    args = parser.parse_args()
    result = {"status": "running", "synthetic_input": True, "accuracy_benchmark": False,
              "automatic_retries": False, "scope": "warm native model process, one token transport, minimal JSON and short synthetic cross-clause JSON only"}
    trace = "gguf-probe-" + uuid.uuid4().hex[:12]
    config = C.LLMConfig(endpoint=args.base + "/v1", model="legalone-8b", timeout=120, max_tokens=160)
    try:
        result["health"] = get(args.base, "/health")
        result["models"] = get(args.base, "/v1/models")
        if not any(item.get("id") == "legalone-8b" for item in result["models"].get("data", [])):
            raise RuntimeError("Actual model endpoint does not expose legalone-8b alias")
        with A.review_audit(trace, "synthetic-gguf-probe") as collector:
            call_id = uuid.uuid4().hex
            meta = {"call_id": call_id, "max_tokens": 1}
            A.record("llm", config.model, "one_token_transport", "started", **meta)
            started = time.perf_counter()
            try:
                raw_config = C.LLMConfig(endpoint=config.endpoint, model=config.model, timeout=config.timeout, response_format="")
                content, envelope, elapsed = C._complete(raw_config, [{"role": "user", "content": "只回答数字1。"}], 1)
                if envelope.get("model") != config.model:
                    raise RuntimeError("Actual model identity mismatch")
                A.record("llm", config.model, "one_token_transport", "succeeded", time.perf_counter()-started, **meta)
                result["one_token"] = {"content": content, "response": envelope, "elapsed_seconds": round(elapsed, 6),
                                       "interpretation": "transport/generation smoke only; expected one-token limit is not complete JSON or review success"}
            except Exception as exc:
                A.record("llm", config.model, "one_token_transport", "failed", time.perf_counter()-started,
                         "transport_or_identity_failure", error_type=type(exc).__name__, **meta)
                raise
            print("One real token returned; testing complete JSON once.", flush=True)
            pair = C.request_json(config, [{"role": "system", "content": "只输出JSON。"},
                                           {"role": "user", "content": '请原样返回这个JSON对象：{"ok":true}'}], max_tokens=32, retries=0)
            result["minimal_json"] = {"content": pair.content, "model": pair.model, "elapsed_seconds": pair.elapsed,
                                       "degraded": pair.degraded, "reason_code": pair.reason_code}
            if pair.content.get("ok") is not True or pair.degraded:
                raise RuntimeError("Minimal JSON was incorrect or degraded")
            print("Complete JSON returned; testing short synthetic contract once.", flush=True)
            contract = "C2：甲方应在验收后十日内付款。C5：甲方应在验收后五日内付款。"
            pair = C.request_json(config, [{"role": "system", "content": "你是合同审查助手。只输出JSON，不写思考过程。"},
                                           {"role": "user", "content": contract + '\n判断付款期限是否冲突。返回{"conflict":布尔值,"clause_ids":[条款编号],"reason":"简短中文依据"}。'}],
                                  max_tokens=160, retries=0)
            result["short_contract"] = {"input": contract, "content": pair.content, "model": pair.model,
                                         "elapsed_seconds": pair.elapsed, "degraded": pair.degraded, "reason_code": pair.reason_code}
            if pair.degraded or pair.content.get("conflict") is not True:
                raise RuntimeError("Short synthetic contract JSON did not complete correctly")
            result["status"] = "passed_smoke_only"
            result["model_calls"] = collector.events
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        result["model_calls"] = A.recent_events(limit=50, trace_id=trace)
    result["trace_id"] = trace
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out), "error": result.get("error")}, ensure_ascii=False), flush=True)
    return int(result["status"] != "passed_smoke_only")


if __name__ == "__main__":
    raise SystemExit(main())
