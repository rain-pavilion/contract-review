# -*- coding: utf-8 -*-
"""大模型客户端：申报书里五个节点要"把子任务派发给专门的大模型"。

这个模块只做三件事：拼 URL、发请求、把回复里的 JSON 抠出来。
刻意不引入第三方 SDK —— 只要目标端点兼容 OpenAI 的 /v1/chat/completions，
无论是本地 Ollama、vLLM、LegalOne-8B 还是云端 API，都能直接用。

设计约束（对应申报书"可溯源"的要求）：
1. **可缺席**：没有端点时 `available()` 为假，调用方必须走降级分支，
   而不是抛异常把整条链路打断；
2. **要 JSON**：节点的输入输出都是结构化信息，所以默认要求返回 JSON；
   通用端点不支持 `response_format` 时回落到提示词约束；显式原生 schema 被拒绝时不降级抹除；
3. **留证据**：原始回复与耗时都返回给调用方，写进 TraceEvent，便于复核。
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import time
import uuid
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

DEFAULT_TIMEOUT = 90.0
#: 逐条款只需要几条缺陷。默认给 1500 会让小模型继续絮叨、写越界成重复或乱码，
#: 反而把 JSON 撑坏（LegalOne-1.7B 实测：1500 → 44~72 s 且可能不闭合；512 → 5.7 s 稳定）。
DEFAULT_MAX_TOKENS = 512


class LLMClientError(RuntimeError):
    """请求失败、超时或回复不含可解析 JSON 时抛出，由调用方决定是否降级。"""

    def __init__(self, message: str, *, status: int | None = None, raw: str = "",
                 call_id: str = "", request_receipts: Sequence[dict[str, Any]] = ()) -> None:
        super().__init__(message)
        self.status = status
        self.raw = raw
        self.call_id = call_id
        self.request_receipts = [dict(row) for row in request_receipts]


class LLMContextError(LLMClientError):
    """Native capacity check failed before any generation was submitted."""

    def __init__(self, message: str, reason: str, details: dict[str, Any]):
        super().__init__(message)
        self.reason = reason
        self.details = details


@dataclass
class LLMPair:
    """一次完整调用的回执：内容 + 元信息，供 trace 与降级记录使用。"""

    content: dict[str, Any]
    model: str
    elapsed: float
    raw: str = ""
    prompt_chars: int = 0
    degraded: str = ""
    reason_code: str = ""
    call_id: str = ""
    request_receipts: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        if self.degraded:
            return f"degraded:{self.degraded}"
        return f"{self.model} {self.elapsed:.2f}s {len(self.raw)}B"


@dataclass
class LLMConfig:
    """端点配置。字段与 v1 的 AgentReviewConfig 对齐，便于两边切换。"""

    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    disable_thinking: bool = True
    response_format: str = "json_object"
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Explicit llama.cpp-only controls. Generic OpenAI endpoints receive neither.
    native_thinking: str = ""
    native_reasoning_format: str = ""
    native_json_schema: bool = False
    native_context_preflight: bool = False

    @property
    def ready(self) -> bool:
        return bool(self.endpoint and self.model)

    def redacted(self) -> dict[str, Any]:
        """写日志用：不出现 api_key。"""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout": self.timeout,
            "has_key": bool(self.api_key),
            "native_json_schema": self.native_json_schema is True,
            "native_context_preflight": self.native_context_preflight is True,
        }


def llm_config_from_env(prefix: str = "ZLQL") -> LLMConfig:
    """从环境变量读配置。

    ZLQL_LLM_ENDPOINT  例如 http://127.0.0.1:11434/v1/chat/completions 或 https://api.xxx.com/v1
    ZLQL_LLM_MODEL     例如 qwen2.5:7b / LegalOne-8B
    ZLQL_LLM_API_KEY   可为空（本地端点通常不需要）
    """
    endpoint = os.environ.get(f"{prefix}_LLM_ENDPOINT", "").strip()
    model = os.environ.get(f"{prefix}_LLM_MODEL", "").strip()
    api_key = os.environ.get(f"{prefix}_LLM_API_KEY", "").strip()
    timeout_raw = os.environ.get(f"{prefix}_LLM_TIMEOUT", "").strip()
    timeout = float(timeout_raw) if timeout_raw.replace(".", "", 1).isdigit() else DEFAULT_TIMEOUT
    native_schema = os.environ.get(f"{prefix}_LLM_NATIVE_JSON_SCHEMA", "").strip().lower() in ("1", "true", "on", "yes")
    return LLMConfig(endpoint=endpoint, model=model, api_key=api_key, timeout=timeout,
                     native_json_schema=native_schema,
                     native_context_preflight=os.environ.get(f"{prefix}_LLM_NATIVE_CONTEXT_PREFLIGHT", "").strip().lower()
                     in ("1", "true", "on", "yes"))


def chat_url(endpoint: str) -> str:
    """把各种写法的端点补成 chat/completions 完整地址。

    支持三种输入：
    * https://host/v1                     → https://host/v1/chat/completions
    * https://host/v1/chat/completions    → 原样返回
    * https://host                         → https://host/v1/chat/completions
    """
    url = endpoint.strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/v1/"):
        return url.rstrip("/") + "/chat/completions"
    return url + "/v1/chat/completions"


def available(config: LLMConfig | None) -> bool:
    return bool(config and config.ready)


_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.S | re.I)


def strip_think(text: str) -> tuple[str, bool]:
    """剥掉 LegalOne / Qwen3 的 `<think>…</think>` 推理块。

    返回 (正文, 是否有推理块)。LegalOne 的 think 里会复述提示词、举例 JSON，
    直接交给 `_extract_json` 的"首个 { 到末个 }"切片会切出错误对象；
    而且截断时 think 未闭合、正文为空 —— 这两种情况都必须先剥再解析。
    """
    if not text:
        return "", False
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate, re.I)
    if fenced:
        candidate = fenced.group(1)
    try:
        if isinstance(json.loads(candidate), dict):
            return text.strip(), False
    except (ValueError, TypeError):
        pass
    has = bool(_THINK_RE.search(text)) or "<think" in text.lower()
    body = _THINK_RE.sub("", text)
    # A template may prefill the opening tag, so the generated text contains
    # only its closing tag. Never parse JSON examples inside that reasoning.
    if re.search(r"</think\s*>", body, re.I):
        body = re.split(r"</think\s*>", body, flags=re.I)[-1]
        has = True
    # 未闭合的 think（生成被 max_new_tokens 截断）：think 之后的内容全部丢弃
    if "<think" in body.lower():
        body = re.split(r"<think\b[^>]*>", body, maxsplit=1, flags=re.I)[0]
    # LegalOne's model card also documents this explicit final-answer format.
    # Restrict it to a line boundary, rather than a phrase inside JSON strings.
    final_marker = re.compile(r"^\s*\[最终回答\]\s*[:：]?\s*", re.M)
    if final_marker.search(body):
        body = final_marker.split(body)[-1]
        has = True
    elif re.match(r"^\s*(?:\[)?思考过程(?:\])?\s*[:：]?", body):
        return "", True
    return body.strip(), has


def _balance(text: str) -> str:
    """补齐未闭合的 `{`/`[`，并裁掉多余的闭合符。

    小模型（LegalOne-1.7B 实测）经常写成 `{"defects":[{...}, {...}] }` 里少一个 `]`，
    或者在数组中间被 max_new_tokens 截断。括号平衡后往往就能解析出来，
    比直接丢弃整条回复划算得多。
    """
    stack: list[str] = []
    pairs = {"}": "{", "]": "["}
    in_string = False
    escaped = False
    trimmed: list[str] = []
    for char in text:
        if in_string:
            trimmed.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            trimmed.append(char)
        elif char in "{[":
            stack.append(char)
            trimmed.append(char)
        elif char in "}]":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
                trimmed.append(char)
            # 多余的闭合符直接丢掉
        else:
            trimmed.append(char)
    if in_string:
        trimmed.append('"')
    out = "".join(trimmed)
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def _repair_json(body: str) -> dict[str, Any] | None:
    """修小模型写坏的结构：裸键名、单引号、括号不配平、尾逗号。"""
    text = body
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    # { code: → { "code":
    text = re.sub(r'([{,\[]\s*)([A-Za-z_][\w-]*)(\s*:)', r'\1"\2"\3', text)
    # 引号跑到键名后面：{ message": → { "message":（LegalOne 实测就写这个）
    text = re.sub(r'([{,\[]\s*)([A-Za-z_][\w-]*)"(\s*:)', r'\1"\2"\3', text)
    # 'key': 'value' → "key": "value"
    text = re.sub(r"'([^'\n]{0,80})'", r'"\1"', text)
    # 重复冒号：LegalOne 实测写过 `"clausesId"::"1"` —— 两个冒号直接让 json.loads 失败
    text = re.sub(r':\s*:+', ": ", text)
    # 尾逗号
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # 漏掉的元素分隔逗号："…风险。" "quote":"…" —— 小模型常把对象成员之间的逗号吃掉
    text = re.sub(r'(?<=[}"\]\d])\s+(?="[A-Za-z_][\w-]*"\s*:)', ", ", text)
    text = _balance(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json(text: str) -> dict[str, Any]:
    """从模型回复里抠出第一个 JSON 对象。

    顺序：剥 think → 整体解析 → ```json 代码块 → 首个 '{' 到末个 '}' 的切片。
    """
    body, had_think = strip_think(text)
    if not body and had_think:
        raise LLMClientError(
            "推理块未闭合：生成被 max_new_tokens 截断，think 之后没有正文",
            raw=(text or "")[:400],
        )
    body = body.strip()
    if not body:
        raise LLMClientError("回复为空")
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    if "```" in body:
        for chunk in body.split("```"):
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    start, end = body.find("{"), body.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(body[start: end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # 最后一道：修补裸键名/单引号/括号不配平后再解析
    for candidate in (body[start:] if start >= 0 else body, body):
        repaired = _repair_json(candidate)
        if repaired is not None:
            return repaired
    raise LLMClientError("回复不是合法 JSON（修补后仍失败）", raw=body[:400])


def normalize_payload(content: dict[str, Any]) -> dict[str, Any]:
    """把"小模型的常见走形"归一成调用方约定的形状。

    实测 LegalOne-1.7B 会把这三种走形都写出来：
    1. 单条缺陷输出成对象而不是数组 → `{"defects": {"code": ...}}`；
    2. 干脆省掉外层键 → `{"code": "...", "quote": "..."}`；
    3. 数组里塞字符串 → `{"focus": "交付期限"}`。

    这三类如果不归一，`content["defects"][:6]` 会直接抛 `KeyError: slice` 或
    `TypeError: unhashable type`，节点表现得像"模型没找出东西"，实际是它找出来了
    但格式不对 —— 这类静默降级比崩溃更难查。
    """
    out = dict(content or {})
    for key in ("defects", "findings", "issues", "focus", "items"):
        if key not in out:
            continue
        value = out[key]
        if value is None:
            out[key] = []
        elif isinstance(value, dict):
            # 多套一层容器：实测 LegalOne 会写 `{"defects": {"clauses": [...]}}`
            nested = None
            for inner in ("clauses", "items", "list", "defects", "results", "data"):
                candidate = value.get(inner)
                if isinstance(candidate, list):
                    nested = candidate
                    break
            out[key] = nested if nested is not None else [value]
        elif isinstance(value, str):
            out[key] = [value] if value.strip() else []
        elif isinstance(value, (list, tuple)):
            out[key] = [item for item in value if item not in (None, "", {})]
        else:
            out[key] = [value]

    # 省掉外层键：把形似缺陷的顶层对象包一层
    if "defects" not in out and any(k in out for k in ("code", "quote", "risk_code")):
        out["defects"] = [dict(out)]
    return _canonical_keys(out)


#: 实测 LegalOne 会把 `clause_id` 写成 `clausesId`、`clauses_id`，把 `quote` 写成 `evidence`。
#: 键名不归一，字段读出来全是空值，"模型给出了结论"会退化成"什么都没找到"。
_KEY_ALIASES = {
    "clausesid": "clause_id",
    "clauses_id": "clause_id",
    "clauseid": "clause_id",
    "clause": "clause_id",
    "riskcode": "code",
    "risk_code": "code",
    "msg": "message",
    "desc": "message",
    "description": "message",
    "evidence": "quote",
    "snippet": "quote",
    "level": "severity",
}


def _canonical_keys(value: Any) -> Any:
    """递归把已知的键名别名改写成调用方约定的键名。"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = _KEY_ALIASES.get(str(key).strip().lower(), str(key))
            if name in out and out[name] not in (None, "", [], {}):
                # 已有非空同义键时就保留原值，不互相覆盖
                continue
            out[name] = _canonical_keys(item)
        return out
    if isinstance(value, list):
        return [_canonical_keys(item) for item in value]
    return value


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Check the bounded protocol subset locally if a server ignores its grammar.

    This is a deliberately small, dependency-free validator for our feedback
    schemas, not a general JSON Schema implementation. Unknown assertions fail
    closed; harmless schema annotations do not affect acceptance.
    """
    allowed = {"type", "required", "properties", "additionalProperties", "items", "minItems", "maxItems",
               "minLength", "maxLength", "enum", "const", "$schema", "title", "description"}
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise LLMClientError("原生JSON Schema包含本地校验器不支持的约束")
    kind = schema.get("type")
    types = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
             "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
             "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
             "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
             "null": lambda v: v is None}
    if kind is not None and (not isinstance(kind, str) or kind not in types or not types[kind](value)):
        raise LLMClientError("原生JSON Schema类型不匹配：" + path)
    if "enum" in schema and (not isinstance(schema["enum"], list) or value not in schema["enum"]):
        raise LLMClientError("原生JSON Schema枚举不匹配：" + path)
    if "const" in schema and value != schema["const"]:
        raise LLMClientError("原生JSON Schema固定值不匹配：" + path)
    for lower, upper, target in (("minItems", "maxItems", list), ("minLength", "maxLength", str)):
        for key in (lower, upper):
            bound = schema.get(key)
            if key in schema and (not isinstance(bound, int) or isinstance(bound, bool) or bound < 0):
                raise LLMClientError("原生JSON Schema长度约束不合法")
        if isinstance(value, target) and ((lower in schema and len(value) < schema[lower])
                                         or (upper in schema and len(value) > schema[upper])):
            raise LLMClientError("原生JSON Schema长度不匹配：" + path)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise LLMClientError("原生JSON Schema对象约束不合法")
        if any(key not in value for key in required):
            raise LLMClientError("原生JSON Schema缺少必需字段：" + path)
        extra = schema.get("additionalProperties", True)
        if not isinstance(extra, bool):
            raise LLMClientError("原生JSON Schema附加字段约束不合法")
        if extra is False and any(key not in properties for key in value):
            raise LLMClientError("原生JSON Schema包含额外字段：" + path)
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], path + "." + key)
    if isinstance(value, list) and "items" in schema:
        if not isinstance(schema["items"], dict):
            raise LLMClientError("原生JSON Schema数组约束不合法")
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], path + f"[{index}]")


def _request_payload(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": max_tokens or config.max_tokens,
        "stream": False,
    }
    if config.native_json_schema is True and json_schema is not None:
        payload["response_format"] = {"type": "json_object", "schema": json_schema}
    elif config.response_format:
        payload["response_format"] = {"type": config.response_format}
    if config.native_thinking:
        if config.native_thinking not in ("on", "off"):
            raise LLMClientError("native_thinking须为on/off或空")
        payload["chat_template_kwargs"] = {"enable_thinking": config.native_thinking == "on"}
    if config.native_reasoning_format:
        if config.native_reasoning_format not in ("none", "deepseek"):
            raise LLMClientError("native_reasoning_format须为none/deepseek或空")
        payload["reasoning_format"] = config.native_reasoning_format
    # schema_hint 只作调用方的自文档，**不拼进提示词**。
    # 实测教训：往 system 里追加"只输出 JSON，字段要求：defects:array"，
    # LegalOne-1.7B 会理解成"请列出字段名"，吐出 `{"defects":[{"clause_id":", "code":", ...}]}`
    # —— 键名齐全、值全空，看着像 JSON，实际什么都没判断。
    return payload


def preflight_native_context(config: LLMConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Count the actual native template, preserving all source and pending data.

    Native-only and explicitly enabled by the verified llama.cpp launcher.
    Missing metadata/tokenizer fails closed; no character-count approximation,
    source truncation, output-budget reduction or model generation is allowed.
    """
    details: dict[str, Any] = {"method": "actual_native_chat_template_and_tokenizer_no_generation",
        "input_truncation_permitted": False, "generation_sent": False}
    try:
        parsed = urllib.parse.urlsplit(chat_url(config.endpoint))
        if parsed.path != "/v1/chat/completions" or parsed.query or parsed.fragment:
            raise ValueError("native endpoint must use /v1 without a query or fragment")
        output = payload.get("max_tokens")
        if type(output) is not int or output < 1:
            raise ValueError("reserved output must be a positive integer")
        headers = {"Content-Type": "application/json", **config.extra_headers}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        def read(route: str, body: dict[str, Any] | None = None) -> Any:
            url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, route, "", ""))
            data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=min(config.timeout, 30)) as response:
                return json.loads(response.read())

        models = read("/v1/models")
        candidates = models.get("data", []) if isinstance(models, dict) else []
        selected = [row for row in candidates if isinstance(row, dict) and row.get("id") == config.model]
        context = selected[0].get("meta", {}).get("n_ctx") if len(selected) == 1 else None
        if type(context) is not int or context < 1:
            raise ValueError("actual model per-slot context missing")
        formatted = read("/apply-template", payload).get("prompt")
        if not isinstance(formatted, str) or not formatted:
            raise ValueError("native template did not return prompt text")
        tokens = read("/tokenize", {"content": formatted, "add_special": True, "parse_special": True}).get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("native tokenizer did not return token IDs")
        required = len(tokens) + output + 1
        details.update(prompt_tokens=len(tokens), reserved_output_tokens=output, reserved_stop_token=1,
            required_context_tokens=required, server_context_tokens=context, fits=required <= context,
            formatted_prompt_sha256=hashlib.sha256(formatted.encode("utf-8")).hexdigest())
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        raise LLMContextError("生成前容量检查不可用，未提交模型生成：" + type(exc).__name__,
                              "context_preflight_unavailable", details) from exc
    if not details["fits"]:
        raise LLMContextError(
            f"上下文容量不足：输入{len(tokens)} + 预留输出{output} + 结束1 > 服务容量{context}；"
            "未提交生成，原文与未决事项均未截断。请扩大模型上下文或重新规划完整证据窗口。",
            "context_budget_exceeded", details)
    return details


def _complete(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    json_schema: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], float]:
    """发一次请求，只负责拿到回复正文，不负责解析。"""
    url = chat_url(config.endpoint)
    payload = _request_payload(config, messages, max_tokens, json_schema)

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    headers.update(config.extra_headers)

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.perf_counter()
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise LLMClientError(f"HTTP {exc.code}: {detail}", status=exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMClientError(f"连接失败：{exc}") from exc
    elapsed = time.perf_counter() - started

    raw_text = raw_bytes.decode("utf-8", "replace")
    try:
        envelope = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"外层不是 JSON：{exc}", raw=raw_text[:400]) from exc

    if not isinstance(envelope, dict):
        raise LLMClientError("回复外层必须是 JSON 对象")
    choices = envelope.get("choices") or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMClientError("回复缺少 choices", raw=raw_text[:400])
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise LLMClientError("回复 message 必须是对象")
    content_text = message.get("content")
    if content_text is None:
        # reasoning_content is analysis, not a completed business answer.
        content_text = ""
    return str(content_text), envelope, elapsed


def request_json(
    config: LLMConfig,
    messages: Sequence[dict[str, str]],
    *,
    schema_hint: str = "",
    json_schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    retries: int = 2,
    audit_context: dict[str, str] | None = None,
    max_transport_attempts: int | None = None,
) -> LLMPair:
    """发对话请求并解析 JSON；解析失败按 retries 做"纠正重试"。

    为什么必须重试：小模型偶尔把"解析失败"本身当成答案。实测 LegalOne-1.7B 输出
    `"code":"NOCOMPLIANCE":("line":1,"column":1` —— 它在模仿 json.JSONDecodeError 的格式。
    这种情况换一句更硬的指令重问通常就对了，比直接降级划算。把跑偏的回复贴回去再纠正，
    比空口重问有效（不贴回去模型会重复同样的错）。
    """
    from . import model_audit
    if max_transport_attempts is not None and (type(max_transport_attempts) is not int or max_transport_attempts < 1):
        raise LLMClientError("真实请求剩余预算必须为正整数")
    binding: dict[str, str] = {}
    if audit_context is not None:
        required = {"window_id", "job_id", "source_sha256"}
        if (not isinstance(audit_context, dict) or set(audit_context) != required
                or not all(isinstance(audit_context.get(key), str) for key in required)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", audit_context.get("window_id", ""))
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", audit_context.get("job_id", ""))
                or not re.fullmatch(r"[0-9a-f]{64}", audit_context.get("source_sha256", ""))):
            raise LLMClientError("调用审计绑定字段不完整或无效")
        binding = dict(audit_context)
    schema_sent = config.native_json_schema is True and json_schema is not None
    schema_hash = ""
    bound_schema = None
    if schema_sent:
        try:
            if not isinstance(json_schema, dict) or json_schema.get("type") != "object":
                raise ValueError("schema root must be an object")
            encoded = json.dumps(json_schema, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
            if len(encoded) > 128000:
                raise ValueError("schema is too large")
            bound_schema = json.loads(encoded)
            schema_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        except (TypeError, ValueError, OverflowError) as exc:
            model_audit.record("llm", config.model, "request_json", "failed", reason="native_json_schema_invalid",
                               native_json_schema_requested=True, native_json_schema_sent=False)
            raise LLMClientError("原生JSON Schema不合法，未发送请求") from exc
    if not config.ready:
        model_audit.record("llm", config.model, "request_json", "skipped", reason="endpoint_or_model_missing")
        raise LLMClientError("未配置端点或模型")

    convo = [dict(item) for item in messages]
    last: LLMClientError | None = None
    current = config
    started = time.perf_counter()
    attempt = 0
    parse_failures = 0
    format_retry = False
    receipts: list[dict[str, Any]] = []

    def bind_error(exc: LLMClientError) -> LLMClientError:
        exc.call_id = receipts[-1]["call_id"] if receipts else ""
        exc.request_receipts = [dict(row) for row in receipts]
        return exc

    while parse_failures <= max(0, retries):
        if max_transport_attempts is not None and attempt >= max_transport_attempts:
            raise bind_error(last or LLMClientError("真实请求预算已耗尽，当前任务未完成"))
        attempt += 1
        call_started = time.perf_counter()
        call_id = uuid.uuid4().hex
        meta = {"call_id": call_id, "requested_model": config.model, "attempt": attempt, "max_tokens": max_tokens or config.max_tokens,
                "prompt_chars": sum(len(item.get("content", "")) for item in convo),
                "native_thinking": config.native_thinking, "native_reasoning_format": config.native_reasoning_format,
                "native_json_schema_requested": json_schema is not None,
                "native_json_schema_sent": schema_sent, "json_schema_sha256": schema_hash, **binding}
        receipt = {"call_id": call_id, "transport_attempt": attempt,
                   "status": "started", **binding}
        receipts.append(receipt)
        transport = replace(current, extra_headers={**current.extra_headers, "X-ZLQL-Call-ID": call_id,
                                                    "X-ZLQL-Trace-ID": model_audit.scope_metadata()["trace_id"]})
        if transport.native_context_preflight is True:
            try:
                preflight = preflight_native_context(transport, _request_payload(transport, convo, max_tokens, bound_schema))
            except LLMContextError as exc:
                receipt.update(status="skipped", reason=exc.reason, generation_sent=False, context_preflight=exc.details)
                model_audit.record("llm", config.model, "request_json", "skipped", time.perf_counter() - call_started,
                    exc.reason, generation_sent=False,
                    **{"context_" + key: value for key, value in exc.details.items()}, **meta)
                raise bind_error(exc)
            meta.update({"context_" + key: value for key, value in preflight.items()})
            receipt["context_preflight"] = preflight
        receipt["generation_sent"] = True
        meta["generation_sent"] = True
        model_audit.record("llm", config.model, "request_json", "started", **meta)
        try:
            if schema_sent:
                content_text, envelope, elapsed = _complete(transport, convo, max_tokens, json_schema=bound_schema)
            else:
                content_text, envelope, elapsed = _complete(transport, convo, max_tokens)
        except LLMClientError as exc:
            can_retry_format = (not schema_sent and exc.status in (400, 422) and bool(current.response_format) and not format_retry
                                and any(token in str(exc).lower() for token in
                                        ("response_format", "response format", "json_object", "unsupported format")))
            model_audit.record("llm", config.model, "request_json", "failed", time.perf_counter() - call_started,
                               "response_format_retry" if can_retry_format else
                               "native_json_schema_rejected" if schema_sent and exc.status in (400, 422) else "transport_failed",
                               http_status=exc.status, error_type=type(exc).__name__, **meta)
            receipt.update(status="failed", reason="response_format_retry" if can_retry_format else "transport_failed")
            bind_error(exc)
            last = exc
            if can_retry_format:
                format_retry = True
                current = replace(current, response_format="")
                continue
            raise
        actual_reported = str(envelope.get("model") or "").strip()
        actual = actual_reported or config.model
        requested_name = config.model.strip().rsplit("/", 1)[-1].casefold()
        actual_name = actual_reported.rsplit("/", 1)[-1].casefold()
        response_meta = {"actual_model": actual, "model_identity_verified": bool(actual_reported) and actual_name == requested_name}
        usage = envelope.get("usage")
        if isinstance(usage, dict):
            response_meta.update({key: value for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                                  if isinstance((value := usage.get(key)), int) and not isinstance(value, bool) and value >= 0})
        finish_reason = str(((envelope.get("choices") or [{}])[0]).get("finish_reason") or "")
        response_meta["finish_reason"] = finish_reason
        # Only the explicitly required LegalOne-8B identity is strict. Generic
        # gateways may legitimately map an alias to a dated snapshot ID.
        if requested_name == "legalone-8b" and actual_reported and actual_name != requested_name:
            model_audit.record("llm", actual, "request_json", "failed", time.perf_counter() - call_started,
                               "model_identity_mismatch", **response_meta, **meta)
            receipt.update(status="failed", reason="model_identity_mismatch")
            raise bind_error(LLMClientError("服务实际返回的模型与请求模型不一致，拒绝静默替换"))
        try:
            if schema_sent:
                try:
                    content = json.loads(content_text)
                except (TypeError, ValueError) as exc:
                    raise LLMClientError("原生JSON Schema模式必须返回完整JSON，不能修补截断输出") from exc
                _validate_json_schema(content, bound_schema)
                # Feedback IDs and root names must survive exactly as produced.
                # The ordinary fuzzy alias/shape repair path is never strict.
            else:
                content = normalize_payload(_extract_json(content_text))
        except LLMClientError as exc:
            last = exc
            if finish_reason in ("length", "time_limit"):
                model_audit.record("llm", actual, "request_json", "failed",
                                   time.perf_counter() - call_started, "output_truncated",
                                   reason_code="generation_time_limit" if finish_reason == "time_limit" else "token_limit",
                                   error_type=type(exc).__name__, **response_meta, **meta)
                # Repeating an unfinished JSON object with the identical output
                # budget does not restore coverage and can multiply CPU latency.
                receipt.update(status="failed", reason="output_truncated")
                raise bind_error(LLMClientError("模型输出已截断，本窗口未完成；请缩小分块或增加输出预算")) from exc
            model_audit.record("llm", str(envelope.get("model") or config.model), "request_json", "failed",
                               time.perf_counter() - call_started,
                               "native_json_schema_validation_failed" if schema_sent else "invalid_json",
                               error_type=type(exc).__name__, **response_meta, **meta)
            receipt.update(status="failed", reason="native_json_schema_validation_failed" if schema_sent else "invalid_json")
            bind_error(exc)
            parse_failures += 1
            if parse_failures > max(0, retries):
                raise
            convo = convo + [
                {"role": "assistant", "content": content_text[:600]},
                {
                    "role": "user",
                    "content": "上一条不是合法 JSON（" + str(exc)[:40] + "）。只输出一个 JSON 对象，不要解释，不要输出错误信息。",
                },
            ]
            continue
        reason = ("output_truncated" if finish_reason in ("length", "time_limit")
                  else "json_retry" if parse_failures else "response_format_retry" if format_retry else "")
        reason_code = "generation_time_limit" if finish_reason == "time_limit" else "token_limit" if finish_reason == "length" else reason
        model_audit.record("llm", actual, "request_json", "degraded" if reason else "succeeded",
                           time.perf_counter() - call_started, reason, reason_code=reason_code, **response_meta, **meta)
        receipt.update(status="degraded" if reason else "succeeded", reason=reason)
        return LLMPair(
            content=content,
            model=actual,
            elapsed=time.perf_counter() - started,
            raw=content_text,
            prompt_chars=sum(len(item.get("content", "")) for item in convo),
            degraded=reason,
            reason_code=reason_code,
            call_id=call_id,
            request_receipts=[dict(row) for row in receipts],
        )
    raise last or LLMClientError("未知失败")


def probe(config: LLMConfig, *, max_tokens: int = 8) -> dict[str, Any]:
    """轻量体检：只问一个必须回 JSON 的问题，用于 doctor / 部署自检。"""
    if not config.ready:
        return {"ok": False, "reason": "未配置端点或模型"}
    try:
        pair = request_json(
            config,
            [
                {"role": "system", "content": "你是自检探针，只回 JSON。"},
                {"role": "user", "content": '回一个 JSON：{"ok": true}'},
            ],
            max_tokens=max_tokens,
        )
    except LLMClientError as exc:
        return {"ok": False, "reason": str(exc), "status": exc.status}
    return {"ok": bool(pair.content.get("ok")), "model": pair.model, "elapsed": round(pair.elapsed, 3)}
