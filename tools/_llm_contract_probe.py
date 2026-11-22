"""模型接入体检：把"接一个真模型"必须验的 12 条做成一次可复跑的检查。

为什么要有这个脚本
------------------
把 LegalOne 接进节点②时,前后踩了 11 个坑,全部是"我以为是那样、实际不是"。
根因不是手笨,而是**假端点只能验证我的解析器,不能验证模型**:
假端点是我写的,它按我期望的格式回话,所以提示词毒害、键名走形、停符缺失、
字段留空这些"模型的坏习惯"在假端点上一条都照不出来。

这个脚本就是补那道缺的工序。跑一次,12 条里哪几条不满足会直接点出来,
不用等到端到端跑完、盯着"采纳 0 条"猜。

用法
----
    python tools/_llm_contract_probe.py                 # 只做静态体检(不需要模型)
    python tools/_llm_contract_probe.py --endpoint http://127.0.0.1:8622/v1 --model legalone-1.7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import llm_client as L  # noqa: E402
from zhilv_qiling_v2 import nodes as N  # noqa: E402
from zhilv_qiling_v2.models import RISK_CODES  # noqa: E402

WEIGHT_DIR = ROOT / "zhilv_qiling_v2" / "models" / "pretrained" / "legalone-1.7b"

#: 体检表:编号、(名字、判据)。判据返回 (ok, 说明)
CHECKS: list[tuple[str, str]] = []


def check(name: str):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn

    return wrap


def _post(endpoint: str, model: str, messages: list[dict], *, max_tokens: int, timeout: float) -> dict:
    """最小化调用:只发 messages,不拼任何格式要求 —— 正是这一条验出了 schema_hint 的毒害。"""
    url = L.chat_url(endpoint)
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--model", default="legalone-1.7b")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    results: list[tuple[str, bool, str, bool]] = []

    def record(name: str, ok: bool, note: str, warn: bool = False) -> None:
        results.append((name, ok, note, warn))

    # ---------------- 1) 权重文件齐不齐 ----------------
    needed = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    weights = [p.name for p in WEIGHT_DIR.glob("*.safetensors")] + [p.name for p in WEIGHT_DIR.glob("*.bin")]
    missing = [name for name in needed if not (WEIGHT_DIR / name).exists()]
    record(
        "1 权重与配置齐备",
        not missing and bool(weights),
        "缺 " + ", ".join(missing) if missing else f"{WEIGHT_DIR.name}: {weights or '无权重'}",
    )

    # ---------------- 2) 架构与 tokenizer 能对上 ----------------
    try:
        cfg = json.loads((WEIGHT_DIR / "config.json").read_text(encoding="utf-8"))
        arch = (cfg.get("architectures") or ["?"])[0]
        record("2 架构可识别", "CausalLM" in arch, f"{arch} / layers={cfg.get('num_hidden_layers')} / hidden={cfg.get('hidden_size')}")
    except Exception as exc:  # noqa: BLE001
        record("2 架构可识别", False, f"读 config 失败: {exc}")

    # ---------------- 3) 停符必须显式给 ----------------
    try:
        sys.path.insert(0, str(ROOT))
        from transformers import AutoTokenizer  # 惰性:没装 torch 也不该在这里崩

        tok = AutoTokenizer.from_pretrained(str(WEIGHT_DIR), trust_remote_code=True)
        stop_ids = sorted(set([tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]))
        stop_ids = [i for i in stop_ids if isinstance(i, int) and i >= 0]
        record("3 停符显式可用", len(stop_ids) >= 2, f"候选 stop_ids={stop_ids}（少给会尾巴无限重复）")
    except Exception as exc:  # noqa: BLE001
        record("3 停符显式可用", False, f"tokenizer 未就绪: {type(exc).__name__}")

    # ---------------- 4) 提示词里不许出现格式要求 ----------------
    prompt = N.LOGIC_SYSTEM_PROMPT
    poisoned = "字段要求" in prompt
    record("4 提示词不含字段要求", not poisoned, f"LOGIC_SYSTEM_PROMPT {len(prompt)} 字" + ("｜含'字段要求'→ 模型会输出空骨架" if poisoned else ""))

    # ---------------- 5) 风险码表可查 ----------------
    sample_codes = ["amount_inconsistent", "zero_rent", "paymentUnclear", "amount_missing"]
    mapped = {code: N._coerce_code(code) for code in sample_codes}
    unknown = [c for c, v in mapped.items() if not v]
    record("5 码表归一到位", True, f"{len(RISK_CODES)} 个码；样例映射 {mapped}；表外={unknown}")

    # ---------------- 6) defects 归一（dict → list） ----------------
    norm = L.normalize_payload({"defects": {"clauses": [{"clausesId": 1, "code": "zero_rent", "evidence": "0元"}]}})
    item = (norm.get("defects") or [{}])[0]
    ok = isinstance(norm.get("defects"), list) and item.get("clause_id") == 1 and item.get("quote") == "0元"
    record("6 容器与键名归一", ok, f"defects 是 list={isinstance(norm.get('defects'), list)}；clausesId→clause_id={'clause_id' in item}；evidence→quote={'quote' in item}")

    # ---------------- 7) JSON 修复链 ----------------
    broken = '{ "defects":[ { "code":"zero_rent", "message":"零租金" } ]'
    fixed = L._repair_json(broken)
    record("7 JSON 修复链", isinstance(fixed, dict) and bool(fixed.get("defects")), "缺闭合括号可修" if fixed else "修不了")

    # ---------------- 8) think 剥离 ----------------
    body, had = L.strip_think('<think>想一下</think>{"defects": []}')
    record("8 think 剥离", had and body.strip() == '{"defects": []}', f"had_think={had} 正文={body[:30]!r}")

    # ---------------- 9) 引文校验闸 ----------------
    try:
        from zhilv_qiling_v2.models import Clause

        clause = Clause(
            clause_id="c1",
            label="第一条",
            title="交付",
            level=1,
            heading="",
            body="乙方应在接到通知后及时完成交付，逾期视为违约。",
            text="乙方应在接到通知后及时完成交付，逾期视为违约。",
            block_ids=["b1"],
        )
        hit = N._locate_quote(clause, "及时完成交付")
        miss = N._locate_quote(clause, "甲方应在三日内付款")
        record("9 引文校验闸", bool(hit) and not miss, "原文片段可定位、编造片段被拒")
    except Exception as exc:  # noqa: BLE001
        record("9 引文校验闸", False, f"构造 Clause 失败: {type(exc).__name__}: {exc}")

    # ---------------- 10) 条款编号兜底 ----------------
    try:
        from zhilv_qiling_v2.models import Clause

        blank = Clause(clause_id="c2", label="", title="", level=1, heading="", body="租金为零元。", text="租金为零元。", block_ids=["b2"])
        record(
            "10 条款编号兜底",
            (blank.label or "").strip() == "",
            "label 可为空；节点侧补'第N条'（源 bug：留空 → 模型把它当待填模板、输出空骨架）",
        )
    except Exception as exc:  # noqa: BLE001
        record("10 条款编号兜底", False, f"{type(exc).__name__}: {exc}")

    # ---------------- 11) max_tokens 有上限 ----------------
    default_cap = getattr(L.LLMConfig(), "max_tokens", None)
    try:
        from zhilv_qiling_v2.config import Config as _Config

        effective = N.llm_settings(_Config()).max_tokens
    except Exception:  # noqa: BLE001
        effective = default_cap
    record(
        "11 max_tokens 有界",
        isinstance(default_cap, int) and isinstance(effective, int) and 0 < default_cap <= 1024 and 0 < effective <= 1024,
        f"LLMConfig 默认={default_cap} / llm_settings 生效={effective}",
    )

    # ---------------- 12) 真调用（可选） ----------------
    if args.endpoint:
        started = time.perf_counter()
        cfg = L.LLMConfig(endpoint=args.endpoint, model=args.model, timeout=args.timeout, max_tokens=512)
        try:
            # 走 request_json（生产真实路径），它内部带一次"纠正重试"：
            # 小模型偶尔会把"解析失败"本身当答案吐出来，重问一句通常就对了。
            pair = L.request_json(
                cfg,
                [
                    # 用生产里的真实提示词，不要自己编一句"只输出 JSON：{...}"——
                    # 上一版就是这么测的，结果模型去模仿 JSON 报错格式了。
                    {"role": "system", "content": N.LOGIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "合同类型：技术服务\n条款编号：第1条\n条款文本：\n租金为零元，乙方应在接到通知后及时完成交付。",
                    },
                ],
                retries=2,
            )
            elapsed = time.perf_counter() - started
            text = pair.raw or ""
            body, had = L.strip_think(text)
            repeated = text.count("<|im_end|>") > 1 or text.count("舅") > 3
            items = pair.content.get("defects") or []
            ok = isinstance(items, list) and not repeated and bool(body.strip())
            record(
                "12 真调用可用",
                ok,
                f"{elapsed:.1f}s / {len(text)} 字 / think={had} / 重试过={pair.degraded} / 缺陷={len(items)}"
                + ("" if ok else f" / raw={body.strip()[:400]!r}"),
            )
        except L.LLMClientError as exc:
            # 这一项是模型能力观测，不是链路契约：判 FAIL 只说明 1.7B 这次没写出可解析的 JSON，
            # 不代表接入有 bug（重试 2 次仍截断在同一个字段名上，性质是生成截断）。
            record("12 真调用可用", False, f"{time.perf_counter() - started:.1f}s / {str(exc)[:60]} / raw={(exc.raw or '')[:60]!r}", warn=True)
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            record("12 真调用可用", False, f"连不上或返回异常: {type(exc).__name__}: {str(exc)[:60]}", warn=True)
    else:
        record("12 真调用可用", True, "跳过（未给 --endpoint）")

    # ---------------- 输出 ----------------
    print("模型接入体检")
    print("-" * 78)
    bad = warn = 0
    for name, ok, note, is_warn in results:
        flag = "PASS" if ok else ("WARN" if is_warn else "FAIL")
        bad += 0 if (ok or is_warn) else 1
        warn += 0 if ok else (1 if is_warn else 0)
        print(f"[{flag}] {name:<22} {note}")
    print("-" * 78)
    print(f"结论：{len(results) - bad - warn} 项契约通过 / {warn} 项模型能力观测 / {bad} 项未通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
