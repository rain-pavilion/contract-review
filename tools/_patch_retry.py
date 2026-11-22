"""一次性补丁：给 llm_client.request_json 加"纠正重试"。

为什么必须重试：小模型偶尔把"解析失败"本身当答案输出。
实测 LegalOne-1.7B 吐过 `"code":"NOCOMPLIANCE":("line":1,"column":1` ——
它在模仿 json.JSONDecodeError 的格式。换一句更硬的指令重问通常就对了。
"""

import pathlib
import re

p = pathlib.Path("zhilv_qiling_v2/llm_client.py")
t = p.read_text(encoding="utf-8")
start = t.index("def request_json(")
m = re.search(r"\ndef [a-zA-Z_]", t[start:])
end = start + m.start() + 1 if m else len(t)

new = '''def _complete(
    config: LLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int | None,
) -> tuple[str, dict[str, Any], float]:
    """发一次请求，只负责拿到回复正文，不负责解析。"""
    url = chat_url(config.endpoint)
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": max_tokens or config.max_tokens,
        "stream": False,
    }
    if config.response_format:
        payload["response_format"] = {"type": config.response_format}
    # schema_hint 只作调用方的自文档，**不拼进提示词**。
    # 实测教训：往 system 里追加"只输出 JSON，字段要求：defects:array"，
    # LegalOne-1.7B 会理解成"请列出字段名"，吐出 `{"defects":[{"clause_id":", "code":", ...}]}`
    # —— 键名齐全、值全空，看着像 JSON，实际什么都没判断。

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

    choices = envelope.get("choices") or []
    if not choices:
        raise LLMClientError("回复缺少 choices", raw=raw_text[:400])
    message = choices[0].get("message") or {}
    content_text = message.get("content")
    if content_text is None:
        # 部分推理模型把结果放在 reasoning_content
        content_text = message.get("reasoning_content") or ""
    return str(content_text), envelope, elapsed


def request_json(
    config: LLMConfig,
    messages: Sequence[dict[str, str]],
    *,
    schema_hint: str = "",
    max_tokens: int | None = None,
    retries: int = 1,
) -> LLMPair:
    """发对话请求并解析 JSON；解析失败按 retries 做"纠正重试"。

    为什么必须重试：小模型偶尔把"解析失败"本身当成答案。实测 LegalOne-1.7B 输出
    `"code":"NOCOMPLIANCE":("line":1,"column":1` —— 它在模仿 json.JSONDecodeError 的格式。
    这种情况换一句更硬的指令重问通常就对了，比直接降级划算。把跑偏的回复贴回去再纠正，
    比空口重问有效（不贴回去模型会重复同样的错）。
    """
    if not config.ready:
        raise LLMClientError("未配置端点或模型")

    convo = [dict(item) for item in messages]
    last: LLMClientError | None = None
    for attempt in range(max(0, retries) + 1):
        content_text, envelope, elapsed = _complete(config, convo, max_tokens)
        try:
            content = normalize_payload(_extract_json(content_text))
        except LLMClientError as exc:
            last = exc
            if attempt >= retries:
                raise
            convo = convo + [
                {"role": "assistant", "content": content_text[:600]},
                {
                    "role": "user",
                    "content": "上一条不是合法 JSON（" + str(exc)[:40] + "）。只输出一个 JSON 对象，不要解释，不要输出错误信息。",
                },
            ]
            continue
        return LLMPair(
            content=content,
            model=str(envelope.get("model") or config.model),
            elapsed=elapsed,
            raw=content_text,
            prompt_chars=sum(len(item.get("content", "")) for item in convo),
            degraded=attempt > 0,
        )
    raise last or LLMClientError("未知失败")


'''

t = t[:start] + new + t[end:]
p.write_text(t, encoding="utf-8")
print("rewrote request_json; total lines =", len(t.splitlines()))
print("has _complete:", "_complete(" in t)
print("retries param:", "retries: int = 1" in t)
