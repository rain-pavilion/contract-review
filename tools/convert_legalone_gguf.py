"""Convert the existing LegalOne-8B weights using the verified official runtime.

No network, model substitution, inference, or package installation is performed.
The original safetensors remain untouched. Conversion and quantization run
sequentially on CPU, with a provenance receipt written after each stage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "tools/acceptance_results/llama_runtime_preparation.json"
EVIDENCE = ROOT / "tools/acceptance_results/legalone8b_gguf_conversion.json"
MODEL = ROOT / "zhilv_qiling_v2/models/pretrained/legalone-8b"
OUTPUT = ROOT / ".model-runtime/models/legalone-8b"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    result = {"status": "preparing", "source_model": str(MODEL.relative_to(ROOT)),
              "model_substituted": False, "original_weights_modified": False, "quantization": "Q4_K_M"}
    started = time.perf_counter()
    def save():
        EVIDENCE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        prep = json.loads(PREP.read_text(encoding="utf-8"))
        if prep.get("status") != "prepared_not_executed":
            raise RuntimeError("Official runtime preparation has not completed")
        converter, quantizer = (ROOT / prep[key] for key in ("converter", "quantizer"))
        config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
        if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("hidden_size") != 4096 or config.get("num_hidden_layers") != 36:
            raise RuntimeError("Existing requested LegalOne-8B architecture mismatch")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        f16, quant = OUTPUT / "legalone-8b-f16.gguf", OUTPUT / "legalone-8b-Q4_K_M.gguf"
        quant_partial = quant.with_suffix(".gguf.part")
        if f16.exists() or quant.exists() or quant_partial.exists():
            raise RuntimeError("Conversion output already exists; inspect it before rerunning")
        free = shutil.disk_usage(OUTPUT).free
        if free < 26 * 1024**3:
            raise RuntimeError("At least 26 GiB free disk is required for bounded conversion")
        result.update(runtime_tag=prep["tag"], runtime_commit=prep["commit"], free_disk_before_bytes=free,
                      source_config_sha256=digest(MODEL / "config.json"), source_index_sha256=digest(MODEL / "model.safetensors.index.json"))
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
        stages = [
            ("conversion", [sys.executable, str(converter), str(MODEL), "--outfile", str(f16), "--outtype", "f16", "--model-name", "legalone-8b"]),
            ("quantization", [str(quantizer), str(f16), str(quant_partial), "Q4_K_M", "4"]),
        ]
        result["stages"] = []
        for name, command in stages:
            log = OUTPUT / (name + ".log")
            entry = {"name": name, "command": command, "log": str(log.relative_to(ROOT)), "status": "running"}
            result["stages"].append(entry)
            result["status"] = name + "_running"
            save()
            print(f"Starting {name}; log={log}", flush=True)
            stage_started = time.perf_counter()
            with log.open("w", encoding="utf-8") as output:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
            entry.update(exit_code=process.returncode, elapsed_seconds=round(time.perf_counter() - stage_started, 3),
                         status="succeeded" if process.returncode == 0 else "failed")
            save()
            if process.returncode:
                raise RuntimeError(f"{name} failed; inspect its local log")
            if name == "quantization":
                # Native launchers must not discover a half-written .gguf.
                quant_partial.replace(quant)
            print(f"Finished {name}: {entry['elapsed_seconds']} seconds", flush=True)
        result.update(status="quantized_not_inference_validated", elapsed_seconds=round(time.perf_counter() - started, 3),
                      intermediate={"path": str(f16.relative_to(ROOT)), "bytes": f16.stat().st_size, "sha256": digest(f16)},
                      gguf={"path": str(quant.relative_to(ROOT)), "bytes": quant.stat().st_size, "sha256": digest(quant)})
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=round(time.perf_counter()-started, 3))
    save()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "quantized_not_inference_validated" else 1


if __name__ == "__main__":
    sys.exit(main())
