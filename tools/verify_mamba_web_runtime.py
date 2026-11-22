"""Isolated CPU startup/health and real prediction check for the opt-in branch.

No LegalOne call, benchmark claim, GPU use, or production database is involved.
The synthetic predictions are engineering diagnostics, never human-gold scores.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ZLQL_ENTAILMENT_DEVICE"] = "cpu"
    os.environ["ZLQL_ENTAILMENT_MODEL_DIR"] = str(ROOT / "zhilv_qiling_v2/models/bert_mamba_entail")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ZLQL_LLM_ENDPOINT"] = ""
    os.environ["ZLQL_LLM_MODEL"] = ""
    started = time.perf_counter()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from zhilv_qiling_v2 import api, entailment, model_audit
    from zhilv_qiling_v2.config import Config
    output = ROOT / "tools/acceptance_results/mamba_entail_training/web_runtime.json"
    with tempfile.TemporaryDirectory(prefix="mamba-web-runtime-") as temp:
        os.environ["ZLQL_MODEL_AUDIT_PATH"] = str(Path(temp) / "audit.sqlite")
        api._SERVER_CONFIG = Config(entailment_backend="bert", enable_llm=False,
            ner_enable_model=False, ner_use_bert=False, use_bge=False,
            store_path=str(Path(temp) / "workspace.sqlite"))
        server = api.serve("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/health", timeout=10) as response:
                health = json.load(response)
            runtime = health["runtime"]
            meta = runtime["bert_metadata"]
            assert runtime["active_backend"] == "bert", runtime
            assert meta["loaded_architecture"]["mamba"] is True
            assert meta["loaded_architecture"]["strict_checkpoint_load"] is True
            assert meta["loaded_checkpoint"]["selection"] == "environment"
            assert meta["loaded_checkpoint"]["weak_supervision"] is True
            assert meta["loaded_checkpoint"]["human_gold"] is False
            assert Path(meta["loaded_checkpoint"]["directory"]) == Path(os.environ["ZLQL_ENTAILMENT_MODEL_DIR"])
            pairs = [
                "合同终止后三年内接收方仍负保密义务。",
                "合同终止后三十日保密义务即全部终止。",
                "本项目使用蓝色文件夹存放纸质资料。",
            ]
            with model_audit.review_audit("mamba-web-runtime-synthetic") as audit:
                predictions = [dataclasses.asdict(entailment.judge_pair(text,
                    {"content": "合成规范：接收方应在合同终止后三年内继续保密。", "article_no": "synthetic-family-02"})) for text in pairs]
            assert all(item["source"] == "bert_entail" for item in predictions)
            assert len([item for item in audit.events if item["operation"] == "predict_pair"]) == 3
            result = {"status": "actual_cpu_web_startup_health_and_predictions_passed",
                "scope": "Actual API startup and HTTP health; three direct neural predictions in the same isolated process. No HTTP contract review or LegalOne inference.",
                "llm_calls": 0, "temporary_database": True, "loopback_ephemeral_port": True,
                "gpu_device_visibility": "disabled", "cpu_threads": 2,
                "runtime": runtime, "predictions": predictions, "audit_events": audit.events,
                "elapsed_seconds": time.perf_counter() - started,
                "limitation": "Synthetic weak-supervision diagnostics, not independent human-gold validation or proof of legal accuracy."}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "predicted_labels": [item["label"] for item in result["predictions"]],
        "evidence": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
