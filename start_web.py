"""Start the local proposal-aligned web workspace without a frontend build step."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGALONE_ALIAS = "legalone-8b"
LEGALONE_PARAMETERS = 8195978240
os.environ.setdefault("ZLQL_BERT_DEVICE", "cpu")
sys.path.insert(0, str(ROOT))
if (ROOT / ".web-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".web-deps"))


def local_model_status(port=8620, *, strict_identity=False):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            data = json.loads(response.read())
        if not isinstance(data, dict):
            return None
        if data.get("status") == "ok":
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                models = json.loads(response.read())
            candidates = models.get("data", []) if isinstance(models, dict) else []
            for item in candidates if isinstance(candidates, list) else []:
                if not isinstance(item, dict):
                    continue
                meta = item.get("meta")
                if (str(item.get("id", "")).strip() == LEGALONE_ALIAS
                        and isinstance(meta, dict) and meta.get("n_params") == LEGALONE_PARAMETERS):
                    return {**item, "ready": True, "backend": "llama.cpp", "_native_identity_verified": True}
            if strict_identity:
                raise RuntimeError("The native model is ready but its exact LegalOne-8B alias/parameter identity did not match.")
            return None
        candidates = data.get("data", [data])
        if not isinstance(candidates, list):
            return None
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model_id") or item.get("id") or item.get("model") or item.get("model_dir") or "")
            name = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()
            if (item.get("ready") is True or item.get("loaded") is True) and name == LEGALONE_ALIAS:
                return {key: value for key, value in item.items() if key != "_native_identity_verified"}
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        pass
    return None


def llama_runtime_paths(args):
    """Only use explicit files or the dedicated same-model runtime directory."""
    runtime = ROOT / ".model-runtime"
    model_root = runtime / "models" / "legalone-8b"

    def manifest_path(filename, key, allowed_root):
        try:
            receipt = json.loads((ROOT / "tools" / "acceptance_results" / filename).read_text(encoding="utf-8"))
            if receipt.get("status") not in ("prepared_not_executed", "quantized_not_inference_validated"):
                return None
            value = receipt
            for part in key:
                value = value[part]
            path = (ROOT / value).resolve()
            return path if path.is_relative_to(allowed_root.resolve()) and path.is_file() else None
        except (OSError, ValueError, TypeError, KeyError):
            return None

    binary = Path(args.llama_server).expanduser().resolve() if args.llama_server else manifest_path(
        "llama_runtime_preparation.json", ("server",), runtime)
    model = Path(args.gguf_model).expanduser().resolve() if args.gguf_model else manifest_path(
        "legalone8b_gguf_conversion.json", ("gguf", "path"), model_root)
    if binary is None and not args.llama_server:
        binaries = sorted(runtime.rglob("llama-server.exe")) if runtime.is_dir() else []
        binary = binaries[0] if binaries else None
    if model is None and not args.gguf_model:
        models = sorted(model_root.glob("*Q4_K_M.gguf")) if model_root.is_dir() else []
        model = models[0] if models else None
    return binary, model


def stop_owned_model(child):
    """Reap only a Popen handle created by this launcher."""
    if child is None or child.poll() is not None:
        return
    try:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Could not finish stopping the model process started here: {error}", file=sys.stderr)


def model_port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def connect_local_model(args):
    """Return only the child we started; never stop an existing model service."""
    if not args.with_model:
        os.environ["ZLQL_ENABLE_LLM"] = "0"
        print("Explicit offline mode: the main LLM channel is disabled.", flush=True)
        return None
    child = None
    try:
        status = local_model_status(args.model_port)
        if args.with_model and not status:
            if model_port_in_use(args.model_port):
                raise RuntimeError("The model port is occupied but its LegalOne-8B identity/readiness could not be verified.")
            llama_binary, gguf_model = llama_runtime_paths(args)
            use_llama = args.model_backend == "llama.cpp" or (
                args.model_backend == "auto" and llama_binary and gguf_model)
            log = ROOT / "zhilv_qiling_v2" / "reports" / "model-server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if use_llama:
                if not llama_binary or not llama_binary.is_file() or not gguf_model or not gguf_model.is_file():
                    raise RuntimeError("The LegalOne-8B GGUF file or llama-server executable is missing.")
                command = [str(llama_binary), "--model", str(gguf_model), "--alias", LEGALONE_ALIAS,
                           "--host", "127.0.0.1", "--port", str(args.model_port),
                           "--ctx-size", str(getattr(args, "model_context_tokens", 16384)), "--threads", str(args.cpu_threads or 8),
                           "--threads-batch", str(args.cpu_threads or 8), "--n-gpu-layers", str(getattr(args, "model_gpu_layers", 0)),
                           "--parallel", "1", "--jinja", "--reasoning", "off",
                           "--no-context-shift", "--predict", "512"]
                if getattr(args, "model_gpu_layers", 0) > 0:
                    command += ["--flash-attn", "on", "--cache-type-k", getattr(args, "model_cache_type", "f16"),
                                "--cache-type-v", getattr(args, "model_cache_type", "f16"),
                                "--batch-size", "256", "--ubatch-size", "128"]
                device_note = (f'GPU offload up to {args.model_gpu_layers} layers' if getattr(args, 'model_gpu_layers', 0) > 0 else 'CPU')
                print(f"Starting the converted LegalOne-8B Q4_K_M model with llama.cpp on {device_note}.", flush=True)
                with log.open("a", encoding="utf-8") as handle:
                    child = subprocess.Popen(command, cwd=str(llama_binary.parent), stdout=handle,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                child = start_transformers_model(args, log)
            print("Loading LegalOne-8B. Log: " + str(log), flush=True)
            deadline, last = time.monotonic() + 900, 0
            while not status:
                if child.poll() is not None:
                    raise RuntimeError("LegalOne-8B failed to start; see " + str(log))
                if time.monotonic() > deadline:
                    raise RuntimeError("Model startup timed out; see " + str(log))
                if time.monotonic() - last >= 15:
                    print("Waiting for the actual 8B model to load...", flush=True)
                    last = time.monotonic()
                time.sleep(1)
                status = local_model_status(args.model_port, strict_identity=True)
        if status:
            os.environ["ZLQL_LLM_ENDPOINT"] = f"http://127.0.0.1:{args.model_port}/v1"
            os.environ["ZLQL_LLM_MODEL"] = LEGALONE_ALIAS
            os.environ["ZLQL_LLM_TIMEOUT"] = "600"
            os.environ.setdefault("ZLQL_LLM_MAX_TOKENS", "4096")
            os.environ.setdefault("ZLQL_LLM_NATIVE_THINKING", "off")
            os.environ.setdefault("ZLQL_LLM_NATIVE_REASONING_FORMAT", "deepseek")
            # Only local_model_status's verified native branch may enable this
            # automatically. Model aliases or a ready Transformers service are
            # not proof that the endpoint supports llama.cpp's schema extension.
            if status.get("backend") == "llama.cpp" and status.get("_native_identity_verified") is True:
                os.environ.setdefault("ZLQL_LLM_NATIVE_JSON_SCHEMA", "1")
                os.environ.setdefault("ZLQL_LLM_NATIVE_CONTEXT_PREFLIGHT", "1")
            os.environ["ZLQL_ENABLE_LLM"] = "1"
            print("Connected to the loaded LegalOne-8B service.", flush=True)
        return child
    except BaseException:
        # Ownership has not transferred to main until this function returns.
        stop_owned_model(child)
        raise


def start_transformers_model(args, log):
    """Retain the explicit Transformers backend as an alternative."""
    device = args.model_device
    if device == "auto":
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
            required = 6.5 if args.quant == "4bit" else 11 if args.quant == "8bit" else 18
            device = "cuda" if free >= required else "cpu"
            print(f"Model device: {device}; free GPU memory {free:.2f} GiB. Same LegalOne-8B weights; no model substitution.", flush=True)
            if device == "cpu":
                print("CPU mode can be slow; the three-minute target is not guaranteed.", flush=True)
        except ImportError as exc:
            raise RuntimeError("Model dependencies missing. Install deploy/requirements-web-model-locked.txt in .venv.") from exc
    folder = ROOT / "zhilv_qiling_v2" / "models" / "pretrained" / "legalone-8b"
    command = [sys.executable, "-m", "zhilv_qiling_v2.local_llm", "--serve",
               "--host", "127.0.0.1", "--port", str(args.model_port),
               "--model-dir", str(folder), "--quant", args.quant,
               "--device", device, "--dtype", args.model_dtype,
               "--no-think", "--max-new-tokens", "512", "--cpu-threads", str(args.cpu_threads or 4)]
    with log.open("a", encoding="utf-8") as handle:
        child = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return child


def workspace_status(url, timeout=1):
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as response:
            health = json.loads(response.read())
            own_server = response.headers.get("Server", "").startswith("ZhilvQiling/")
        if own_server and isinstance(health, dict) and health.get("tag") and "multi_agent" in health:
            return health
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def existing_web_model_ready(health, model_port):
    llm = health.get("llm", {})
    if not isinstance(llm, dict) or not llm.get("enabled"):
        return False
    if str(llm.get("model", "")).strip().rsplit("/", 1)[-1].casefold() != LEGALONE_ALIAS:
        return False
    try:
        endpoint = urllib.parse.urlsplit(str(llm.get("endpoint", "")))
        local_endpoint = (endpoint.scheme == "http" and endpoint.hostname in ("127.0.0.1", "localhost")
                          and endpoint.port == model_port
                          and endpoint.path.rstrip("/") in ("", "/v1", "/v1/chat/completions"))
    except ValueError:
        return False
    return bool(local_endpoint and local_model_status(model_port))


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Zhilv Qiling local web workspace")
    parser.add_argument("--port", type=int, default=8610)
    parser.add_argument("--no-browser", action="store_true")
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument("--with-model", dest="with_model", action="store_true",
                            help="启动或复用实际LegalOne-8B模型后再启动网页（默认）")
    model_mode.add_argument("--without-model", dest="with_model", action="store_false",
                            help="显式离线模式；关闭主大模型通道")
    parser.set_defaults(with_model=True)
    parser.add_argument("--keep-model-running", action="store_true",
                        help="网页成功启动后保留本次启动的模型服务，供下次启动复用")
    parser.add_argument("--model-port", type=int, default=8620)
    parser.add_argument("--model-context-tokens", type=int, default=16384,
                        help="Native per-slot context when starting a new model; existing services are left running")
    parser.add_argument("--quant", choices=["4bit", "8bit", ""], default="4bit")
    parser.add_argument("--model-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--model-backend", choices=["auto", "transformers", "llama.cpp"], default="auto")
    parser.add_argument("--llama-server", default="", help="Explicit llama-server executable path")
    parser.add_argument("--gguf-model", default="", help="Explicit converted LegalOne-8B GGUF path")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 selects the backend's conservative default")
    parser.add_argument("--model-gpu-layers", type=int, default=0,
                        help="Explicit native GPU layer offload; 0 preserves CPU startup; requires a compatible runtime")
    parser.add_argument("--model-cache-type", choices=["f16", "q8_0"], default="f16",
                        help="Native GPU KV cache type for a newly launched model")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.model_port <= 65535 or args.port == args.model_port:
        parser.error("web/model ports must be different and between 1 and 65535")
    if args.cpu_threads < 0:
        parser.error("cpu-threads must be zero or a positive integer")
    if args.model_gpu_layers < 0:
        parser.error("model-gpu-layers must be zero or a positive integer")
    if args.model_context_tokens < 1024:
        parser.error("model-context-tokens must be at least 1024")
    url = f"http://127.0.0.1:{args.port}"
    # Windows SO_REUSEADDR can let two HTTPServer instances bind one port.
    # Probe before binding as well as handling the ordinary port-in-use error.
    health = workspace_status(url)
    if health:
        if args.with_model and not existing_web_model_ready(health, args.model_port):
            print("The existing web server is not connected to a verified, ready local LegalOne-8B service. Stop it before starting --with-model.", file=sys.stderr)
            return 1
        print("Already running: " + url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    model_child, server, retain_model = None, None, False
    try:
        model_child = connect_local_model(args)
        from zhilv_qiling_v2 import api
        try:
            server = api.serve("127.0.0.1", args.port)
        except OSError as error:
            health = workspace_status(url, timeout=2)
            if health and (not args.with_model or existing_web_model_ready(health, args.model_port)):
                retain_model = bool(args.keep_model_running and args.with_model)
                print("Already running: " + url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
            print(f"Cannot start on port {args.port}: {error}", file=sys.stderr)
            print("Use --port 8611 to choose another port.", file=sys.stderr)
            return 1
        print("Web workspace: " + url, flush=True)
        retain_model = bool(args.keep_model_running and args.with_model)
        if retain_model and model_child is not None:
            print(f"Resident LegalOne-8B service: PID {model_child.pid}, port {args.model_port}; closing this web workspace will retain it for reuse.", flush=True)
        archive_path = os.environ.get("ZLQL_STORE_PATH") or "zhilv_qiling_v2/reports/workspace.sqlite"
        print(f"Press Ctrl+C to stop. Reports stay in {archive_path}.", flush=True)
        if not args.no_browser:
            timer = threading.Timer(0.5, lambda: webbrowser.open(url))
            timer.daemon = True
            timer.start()
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            if not retain_model:
                stop_owned_model(model_child)


if __name__ == "__main__":
    raise SystemExit(main())
