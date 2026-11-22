"""本地 LegalOne 推理服务，提供兼容的 POST /v1/chat/completions。

默认加载明确的 `models/pretrained/legalone-8b`。指定模型缺失或加载失败时直接报错，
不会悄悄替换成小模型。显存不足可显式选择4bit量化和CPU offload；实际模型身份、
量化、设备、输入上限与调用结果均可查询或审计。没有静默截断合同文本。

为什么不是直接 `import transformers` 在进程内生成：

1. **一次加载、多次复用**：模型加载要几秒到几十秒，把它做成独立服务，审查进程只在第一次
   调用时付一次冷启动成本；
2. **故障隔离**：模型 OOM 或崩掉，审查进程不会跟着死（`llm_client` 本来就有超时与降级）；
3. **协议一致**：换成远程 LegalOne-8B 或 OpenAI 兼容网关时，只改 `ZLQL_LLM_ENDPOINT`，
   `nodes._llm_logic` 一行不用动。

CLI:
    python -m zhilv_qiling_v2.local_llm --serve --port 8620
    python -m zhilv_qiling_v2.local_llm --prompt "把下面这句话压缩成 JSON：..."   # 冒烟

生成参数默认贪心（temperature=0），因为审查任务要可复现；`<think>` 段若有会原样返回，
由 `nodes` 侧剥掉（对齐 LegalOne 的 `<think>思考</think>[最终回答]` 输出契约）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import MODEL_DIR

DEFAULT_MODEL_DIR = MODEL_DIR / "pretrained" / "legalone-8b"
FALLBACK_DIRS = (DEFAULT_MODEL_DIR,)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.S | re.I)


def find_model_dir(preferred: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """按"真有 config.json + 权重"的顺序挑一个可加载的模型目录。"""
    # A caller requesting 8B must never silently receive a smaller fallback.
    candidates = [pathlib.Path(preferred)] if preferred else list(FALLBACK_DIRS)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_config = (candidate / "config.json").is_file()
        weights = list(candidate.glob("*.safetensors")) + list(candidate.glob("pytorch_model*.bin"))
        index = candidate / "model.safetensors.index.json"
        if index.exists():
            try:
                required = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
                if not required or any(not (candidate / name).is_file() for name in required):
                    continue
            except (OSError, ValueError):
                continue
        if has_config and weights and all(w.stat().st_size > 0 for w in weights):
            return candidate
    return None


def strip_think(text: str) -> str:
    """去掉 `<think>…</think>` 推理段（LegalOne 的输出契约），保留最终回答。"""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里尽量抠出一个 JSON 对象（容忍前后废话与 ```json 围栏）。"""
    body = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.+?)```", body, re.S)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(body[start : end + 1])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class LocalLLM:
    """把 transformers 的因果语言模型包成"聊天"接口。"""

    model_dir: pathlib.Path | None = None
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    disable_thinking: bool = False
    quant: str = ""
    dtype: str = "auto"
    cpu_offload: bool = False
    max_gpu_memory: str = ""
    max_cpu_memory: str = "20GiB"
    max_input_tokens: int = 8192
    cpu_threads: int = 4
    max_generate_seconds: float = 120.0
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _lock: Any = field(default_factory=threading.RLock, repr=False)
    load_seconds: float = 0.0
    calls: int = 0
    last_error: str = ""
    last_generation: dict[str, Any] = field(default_factory=dict)
    torch_threads: int = 0
    compute_dtype: str = ""
    cpu_capability: str = ""

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _torch_dtype(self, torch):
        """把 dtype 字符串翻译成 torch 类型；auto 交给 transformers 自己判断。"""
        table = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = (self.dtype or "auto").strip().lower()
        if key in ("", "auto"):
            return None
        if key not in table:
            raise ValueError(f"不认识的 dtype：{self.dtype}（可用 bf16 / fp16 / fp32 / auto）")
        return table[key]

    def load(self) -> "LocalLLM":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = find_model_dir(self.model_dir)
        if path is None:
            raise FileNotFoundError("没有可加载的本地模型目录（请先跑 tools/_dl_modelscope.py）")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(max(1, int(self.cpu_threads)))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # This process may already have initialized the interop pool.
        self.torch_threads = torch.get_num_threads()
        self.cpu_capability = torch.backends.cpu.get_cpu_capability()
        started = time.perf_counter()
        self.model_dir = path
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)

        kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": True}
        mode = (self.quant or "").strip().lower()
        if mode in ("4bit", "8bit"):
            # 8B fp16 要 ~16 GB 显存，本机只有 8 GB；nf4 后权重约 4.5 GB 才塞得下
            from transformers import BitsAndBytesConfig
            compute_dtype = self._torch_dtype(torch)
            if compute_dtype is None:
                if self.device == "cpu":
                    from bitsandbytes.functional import has_avx512bf16
                    compute_dtype = torch.bfloat16 if has_avx512bf16() else torch.float32
                else:
                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.compute_dtype = str(compute_dtype).replace("torch.", "")
            # AVX2 CPUs emulate BF16 and can be orders of magnitude slower.
            # Quantized weights remain NF4; only compute and unquantized layers use FP32.
            kwargs["dtype"] = compute_dtype

            if mode == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    llm_int8_enable_fp32_cpu_offload=self.cpu_offload,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=self.cpu_offload)
            kwargs["device_map"] = "auto" if self.cpu_offload else {"": 0 if self.device == "cuda" else "cpu"}
            if self.cpu_offload:
                kwargs["max_memory"] = {"cpu": self.max_cpu_memory}
                if self.device == "cuda":
                    kwargs["max_memory"][0] = self.max_gpu_memory or "3GiB"
        else:
            dtype = self._torch_dtype(torch)
            if dtype is None:
                kwargs["dtype"] = "auto"  # transformers >= 4.56
            else:
                kwargs["dtype"] = dtype

        try:
            model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        except TypeError:
            # transformers < 4.56 用 torch_dtype
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)

        if mode in ("4bit", "8bit"):
            self._model = model  # device_map 已负责搬移，再 .to() 会报错
        else:
            self._model = model.to(self.device)
        self._model.eval()
        self.load_seconds = time.perf_counter() - started
        return self

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        disable_thinking: bool | None = None,
    ) -> str:
        """按 chat template 生成，返回原始文本（含 `<think>`，由调用方决定剥不剥）。

        `disable_thinking=True` 走 Qwen3 官方模板开关 `enable_thinking=False`：不生成
        `… ` 段，直接出正文。实测 LegalOne-1.7B 的 think 会吃掉 800+ token、
        一半概率被 max_new_tokens 截断在 think 里，关掉后单条款从 72 s 降到十几秒，
        且不再出现 JSON 之后的乱码重复 token。
        """
        if not self.ready:
            self.load()
        import torch

        want_no_think = self.disable_thinking if disable_thinking is None else bool(disable_thinking)
        with self._lock:
            template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if want_no_think:
                template_kwargs["enable_thinking"] = False
            try:
                prompt = self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # 模板不认 enable_thinking（非 Qwen3 系）→ 退回普通渲染
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            requested_tokens = max_new_tokens or self.max_new_tokens
            if prompt_tokens > self.max_input_tokens:
                raise ValueError(f"context_limit_exceeded: {prompt_tokens}>{self.max_input_tokens}; input was not truncated")
            inputs = inputs.to(self._model.get_input_embeddings().weight.device)
            stop_ids = self._stop_ids()
            kwargs: dict[str, Any] = {
                "max_new_tokens": requested_tokens,
                "pad_token_id": stop_ids[0],
                "eos_token_id": stop_ids,
                # 只加轻度重复惩罚：no_repeat_ngram_size 会把模型逼到罕见 token 上，
                # 输出退化成「）冻得到取出）」这种循环乱码，JSON 直接断在半路。
                "repetition_penalty": 1.05,
            }
            if self.temperature and self.temperature > 0:
                kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
            else:
                kwargs.update(do_sample=False)
            if self.max_generate_seconds > 0:
                kwargs["max_time"] = float(self.max_generate_seconds)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(**inputs, **kwargs)
            generated = output[0][inputs["input_ids"].shape[-1] :]
            generation_elapsed = time.perf_counter() - generation_started
            ended = len(generated) > 0 and int(generated[-1]) in stop_ids
            token_limit = int(generated.shape[-1]) >= requested_tokens
            finish_reason = ("time_limit" if self.max_generate_seconds > 0 and generation_elapsed >= self.max_generate_seconds and not ended and not token_limit
                             else "length" if token_limit else "stop")
            self.calls += 1
            self.last_generation = {"prompt_tokens": prompt_tokens, "completion_tokens": int(generated.shape[-1]),
                                    "generation_seconds": round(generation_elapsed, 3), "finish_reason": finish_reason}
            return self._decode(generated, set(stop_ids))

    def _stop_ids(self) -> list[int]:
        """所有"该停"的 token id。

        Qwen3 系的聊天模板以 `<|im_end|>` 收尾，但 `eos_token_id` 只有
        `<|endoftext|>`。不把 `<|im_end|>` 一起交给 `generate`，模型会在输出完 JSON
        之后**无限重复 `<|im_end|>`**，`skip_special_tokens` 又把它解码成"舅"，
        于是下游看到的是"JSON 后面挂一串乱码" —— 实测 LegalOne-1.7B 就是这样
        （生成 token 全是同一个 id 101849，decode 出 `舅舅舅舅…`）。
        """
        ids: list[int] = []
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            ids.append(eos)
        for token in ("<|im_end|>", "<|endoftext|>"):
            value = self._tokenizer.convert_tokens_to_ids(token)
            if isinstance(value, int) and value >= 0 and value not in ids:
                ids.append(value)
        return ids or [0]

    def _decode(self, generated: Any, stop_ids: set[int]) -> str:
        """在第一个停止符处截断再解码，避免把尾部的重复特殊符带进文本。"""
        tokens = generated.tolist()
        cut = len(tokens)
        for index, token in enumerate(tokens):
            if token in stop_ids:
                cut = index
                break
        return self._tokenizer.decode(tokens[:cut], skip_special_tokens=True)

    def chat_json(self, messages: list[dict[str, str]], *, max_new_tokens: int | None = None) -> dict[str, Any]:
        return extract_json(self.chat(messages, max_new_tokens=max_new_tokens))

    def describe(self) -> dict[str, Any]:
        path = pathlib.Path(self.model_dir) if self.model_dir else find_model_dir()
        return {
            "model_dir": str(path) if path else "",
            "device": self.device,
            "ready": self.ready,
            "load_seconds": round(self.load_seconds, 2),
            "calls": self.calls,
            "max_new_tokens": self.max_new_tokens,
            "model_id": path.name if path else "",
            "quant": self.quant,
            "dtype": self.dtype,
            "cpu_offload": self.cpu_offload,
            "max_input_tokens": self.max_input_tokens,
            "last_error": self.last_error,
            "cpu_threads": self.torch_threads or self.cpu_threads,
            "compute_dtype": self.compute_dtype,
            "cpu_capability": self.cpu_capability,
            "max_generate_seconds": self.max_generate_seconds,
        }


_ENGINE: LocalLLM | None = None
_ENGINE_LOCK = threading.Lock()


def engine(**kwargs: Any) -> LocalLLM:
    """进程级单例，避免每次调用都重新加载权重。"""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = LocalLLM(**kwargs)
        return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None


# --------------------------------------------------------------------------- #
# OpenAI 兼容服务：让 llm_client.request_json 直接可用
# --------------------------------------------------------------------------- #
def make_handler(llm: LocalLLM) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "zlql-local-llm/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # 静音，避免污染 stdout
            return

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/health", "/v1/models", "/models"):
                info = llm.describe()
                self._send(200, {"object": "list", "ready": llm.ready, "model": info["model_id"],
                                 "data": [{"id": info["model_id"], **info}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": {"message": "invalid content length"}})
                return
            if length < 1 or length > 2 * 1024 * 1024:
                self._send(413, {"error": {"message": "request body exceeds limit"}})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "invalid json"}})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": {"message": "JSON object required"}})
                return
            info = llm.describe()
            if body.get("model") not in (None, "", info["model_id"], str(llm.model_dir)):
                self._send(404, {"error": {"message": "requested model is not served", "code": "model_not_found"}})
                return
            messages = body.get("messages") or []
            if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
                self._send(400, {"error": {"message": "messages must contain text content"}})
                return
            try:
                max_new = int(body.get("max_tokens") or llm.max_new_tokens)
            except (TypeError, ValueError):
                self._send(400, {"error": {"message": "max_tokens must be an integer"}})
                return
            if max_new < 1 or max_new > 4096:
                self._send(400, {"error": {"message": "max_tokens must be 1..4096"}})
                return
            from . import model_audit
            call_id = self.headers.get("X-ZLQL-Call-ID", "")
            if not re.fullmatch(r"[a-fA-F0-9]{32}", call_id):
                call_id = uuid.uuid4().hex
            trace_id = self.headers.get("X-ZLQL-Trace-ID", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{0,128}", trace_id):
                trace_id = ""
            with model_audit.review_audit(trace_id), llm._lock:
                started = time.perf_counter()
                model_audit.record("local_llm", info["model_id"], "generate", "started", call_id=call_id,
                                   device=llm.device, quant=llm.quant, max_tokens=max_new)
                try:
                    text = llm.chat(messages, max_new_tokens=max_new)
                    usage = dict(llm.last_generation)
                except Exception as exc:  # 生成失败也要给 llm_client 一个可判读的错误
                    llm.last_error = type(exc).__name__
                    model_audit.record("local_llm", info["model_id"], "generate", "failed", time.perf_counter() - started,
                                       "context_limit_exceeded" if isinstance(exc, ValueError) else "generation_failed",
                                       call_id=call_id, error_type=type(exc).__name__, device=llm.device, quant=llm.quant)
                    self._send(422 if isinstance(exc, ValueError) else 500,
                               {"error": {"message": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}})
                    return
                elapsed = round(time.perf_counter() - started, 3)
                llm.last_error = ""
                model_audit.record("local_llm", info["model_id"], "generate", "succeeded", elapsed,
                                   call_id=call_id, device=llm.device, quant=llm.quant, finish_reason=usage.get("finish_reason", "stop"))
            self._send(
                200,
                {
                    "id": f"chatcmpl-local-{llm.calls}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": info["model_id"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": usage.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                        "elapsed_seconds": elapsed,
                    },
                },
            )

    return Handler


class LocalModelServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        # Windows SO_REUSEADDR permits duplicate listeners and hence duplicate
        # model loads; use an exclusive listener for the local model process.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(host: str = "127.0.0.1", port: int = 8620, *, model_dir: str | None = None,
          warmup: bool = True, max_new_tokens: int = 512,
          disable_thinking: bool = False, quant: str = "", dtype: str = "auto", device: str = "auto",
          cpu_offload: bool = False, max_gpu_memory: str = "", max_cpu_memory: str = "20GiB",
          max_input_tokens: int = 8192, cpu_threads: int = 4, max_generate_seconds: float = 120.0) -> None:
    llm = engine(
        model_dir=pathlib.Path(model_dir) if model_dir else None,
        max_new_tokens=max_new_tokens,
        disable_thinking=disable_thinking,
        quant=quant, dtype=dtype, device=device, cpu_offload=cpu_offload,
        max_gpu_memory=max_gpu_memory, max_cpu_memory=max_cpu_memory, max_input_tokens=max_input_tokens, cpu_threads=cpu_threads,
        max_generate_seconds=max_generate_seconds,
    )
    # Reserve the port before expensive loading. A second launcher must fail
    # before allocating another copy of the 8B checkpoint.
    server = LocalModelServer((host, port), make_handler(llm))
    if warmup:
        from . import model_audit
        started = time.perf_counter()
        try:
            llm.load()
        except Exception as exc:
            model_audit.record("local_llm", pathlib.Path(model_dir).name if model_dir else DEFAULT_MODEL_DIR.name,
                               "load", "failed", time.perf_counter() - started, "model_load_failed",
                               error_type=type(exc).__name__, quant=quant, device=device)
            server.server_close()
            raise
        model_audit.record("local_llm", llm.describe()["model_id"], "load", "succeeded", time.perf_counter() - started,
                           quant=quant, device=llm.device)
        print(f"[local_llm] 已加载 {llm.describe()}", flush=True)
    print(f"[local_llm] serving on http://{host}:{port}/v1/chat/completions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[local_llm] bye", flush=True)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地 LegalOne-8B 服务（兼容聊天接口；不隐式替换模型）")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8620)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="关掉 Qwen3 的 think 段（enable_thinking=False）。1.7B 已改为默认关闭，见 --think",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="显式打开 think 段。实测 1.7B 会写 900+ 字推理且常不闭合 </think>，正文被挤空",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--quant",
        default="",
        choices=["", "4bit", "8bit"],
        help="量化加载。8B(fp16≈16GB) 在 8GB 显存上要用 4bit（nf4，权重约 4.5GB）",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="非量化时的权重精度；CPU 上跑 8B 用 bf16（约 15GB 内存）",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cpu-offload", action="store_true", help="Allow explicit CPU offload, retaining the requested 8B model")
    parser.add_argument("--max-gpu-memory", default="", help="GPU weight budget, e.g. 3GiB, when CPU offload is enabled")
    parser.add_argument("--max-cpu-memory", default="20GiB")
    parser.add_argument("--max-input-tokens", type=int, default=8192, help="Reject oversize inputs; never silently truncate contracts")
    parser.add_argument("--cpu-threads", type=int, default=4, help="CPU inference thread budget; avoids oversubscription")
    parser.add_argument("--max-generation-seconds", "--max-generate-seconds", dest="max_generate_seconds", type=float, default=120,
                        help="Soft generation deadline checked between tokens; 0 disables it")
    args = parser.parse_args(argv)
    # 1.7B 实测：带 think 时会写 900+ 字推理且常不闭合 </think>，512 token 全被吃掉、正文为空，
    # 于是 OpenOne 风格的解析器只能拿到空串。审查场景要的是结论，所以默认关掉，需要用 --think 显式打开。
    no_think = True if not args.think else False

    if args.serve:
        serve(
            args.host,
            args.port,
            model_dir=args.model_dir or None,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=no_think,
            quant=args.quant, dtype=args.dtype, device=args.device, cpu_offload=args.cpu_offload,
            max_gpu_memory=args.max_gpu_memory, max_cpu_memory=args.max_cpu_memory,
            max_input_tokens=args.max_input_tokens,
            cpu_threads=args.cpu_threads,
            max_generate_seconds=args.max_generate_seconds,
        )
        return 0

    llm = LocalLLM(
        model_dir=pathlib.Path(args.model_dir) if args.model_dir else None,
        max_new_tokens=args.max_new_tokens,
        disable_thinking=no_think,
        quant=args.quant,
        dtype=args.dtype,
        device=args.device,
        cpu_offload=args.cpu_offload, max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory, max_input_tokens=args.max_input_tokens,
        cpu_threads=args.cpu_threads,
        max_generate_seconds=args.max_generate_seconds,
    )
    info = find_model_dir(args.model_dir or None)
    if info is None:
        print(json.dumps({"ready": False, "reason": "没有可加载的模型目录"}, ensure_ascii=False))
        return 1
    llm.load()
    if not args.prompt:
        print(json.dumps(llm.describe(), ensure_ascii=False, indent=2))
        return 0
    started = time.perf_counter()
    text = llm.chat([{"role": "user", "content": args.prompt}])
    elapsed = time.perf_counter() - started
    payload = {"describe": llm.describe(), "elapsed": round(elapsed, 2), "raw": text,
               "stripped": strip_think(text), "json": extract_json(text)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
