"""Offline regression receipts; no real model, benchmark or performance claims."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "regression_core_native_schema_v4_20261114"
OUT = ROOT / "tools" / "acceptance_results"
SUMMARY = OUT / (PREFIX + ".summary.json")
NODE = Path(r"C:\Users\ROG\dev-tools\node-v22.20.0-win-x64\node.exe")
SOURCE_RUN = OUT / "core_technical_workflow_subject_bound_v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes(paths):
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def save(value):
    SUMMARY.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if SUMMARY.exists() or list(OUT.glob(PREFIX + "_*.txt")):
        raise RuntimeError("This independently versioned regression receipt already exists; do not overwrite it")
    OUT.mkdir(parents=True, exist_ok=True)
    for key, value in {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONIOENCODING": "utf-8"}.items():
        os.environ[key] = value
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    sources = [path for path in (ROOT / "zhilv_qiling_v2").rglob("*") if path.is_file()
               and path.suffix in (".py", ".html", ".js", ".css")
               and "tests_v2" not in path.parts and "__pycache__" not in path.parts]
    sources.append(ROOT / "start_web.py")
    test_sources = list((ROOT / "zhilv_qiling_v2" / "tests_v2").glob("test_*.py"))
    test_sources += list((ROOT / "tools").glob("acceptance*_tests.py"))
    test_sources += [ROOT / "tools" / "replay_feedback_batch.py", Path(__file__).resolve()]
    summary = {"scope": "offline regression after independent typed feedback prompts and explicit native JSON schema; no model-quality or benchmark claim",
        "python": sys.executable, "started_at_utc": now(), "source_sha256_start": hashes(sources),
        "test_and_runner_sha256_start": hashes(test_sources), "gpu_visibility": "disabled only in this process and its child checks",
        "hf_offline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "neural_cache_isolation": {"entailment": "loaded=True, predictor=None, OfflineUnitIsolation, offline_unit_no_neural_weights, load_seconds=0.0",
                                   "ner": "loaded=True, model=None, tokenizer=None"},
        "previous_logs_overwritten": False, "product_functions_replaced": False,
        "isolation_scope": "runner changes only optional neural cache state; normal test-local mocks remain part of their own fixtures",
        "scope_exclusions": ["real pretrained model inference", "official benchmark", "independent human annotation",
                             "end-to-end performance timing", "archive/scheduler scale test"]}
    save(summary)
    from zhilv_qiling_v2 import entailment, ner
    entailment._BERT.update(loaded=True, predictor=None, error="OfflineUnitIsolation",
                           reason="offline_unit_no_neural_weights", load_seconds=0.0)
    ner._BERT_CACHE.update(loaded=True, model=None, tokenizer=None)

    def tests(name, start, pattern, top):
        log = OUT / (PREFIX + "_" + name + ".txt")
        tick = time.perf_counter()
        with log.open("x", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            suite = unittest.TestLoader().discover(str(start), pattern=pattern, top_level_dir=str(top))
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        details = {"tests": result.testsRun, "executed_without_skip": result.testsRun - len(result.skipped),
            "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
            "skip_details": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
            "success": result.wasSuccessful(), "elapsed_seconds": time.perf_counter() - tick,
            "log": str(log), "finished_at_utc": now()}
        summary[name] = details
        save(summary)
        print(json.dumps({"phase": name, **details}, ensure_ascii=True), flush=True)

    tests("unit", ROOT / "zhilv_qiling_v2" / "tests_v2", "test_*.py", ROOT)
    tests("guards", ROOT / "tools", "acceptance*_tests.py", ROOT / "tools")
    pages = [path for path in sources if path.suffix == ".html"]
    script = r'''const fs=require('fs');let bad=false;const results=[];
for(const path of process.argv.slice(1)){const html=fs.readFileSync(path,'utf8');let scripts=0;for(const match of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)){if(/\btype\s*=\s*["'](?:application\/json|application\/ld\+json)["']/i.test(match[1]))continue;if(!match[2].trim())continue;scripts++;try{new Function(match[2]);}catch(e){bad=true;results.push({path,script:scripts,error:e.message});}}results.push({path,inline_scripts:scripts,status:'syntax_checked_without_execution'});}process.stdout.write(JSON.stringify({success:!bad,pages:results},null,2));process.exitCode=bad?1:0;'''
    js_log = OUT / (PREFIX + "_page_js.txt")
    tick = time.perf_counter()
    result = subprocess.run([str(NODE), "-e", script, *map(str, pages)], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8")
    js_log.write_text(result.stdout + result.stderr, encoding="utf-8")
    summary["page_javascript"] = {"success": result.returncode == 0, "returncode": result.returncode,
        "node": str(NODE), "pages": [str(path) for path in pages], "log": str(js_log),
        "elapsed_seconds": time.perf_counter() - tick, "scope": "parse inline JavaScript without browser, DOM, execution or visual QA"}
    save(summary)
    print(json.dumps({"phase": "page_javascript", **summary["page_javascript"]}, ensure_ascii=True), flush=True)

    restored = []
    for kind in ("statute", "subject"):
        target = OUT / (PREFIX + "_restore_only_" + kind)
        log = OUT / (PREFIX + "_restore_only_" + kind + ".txt")
        command = [sys.executable, str(ROOT / "tools" / "replay_feedback_batch.py"),
            "--source-run", str(SOURCE_RUN), "--out-dir", str(target), "--kind", kind, "--prepare-only"]
        tick = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        receipt_path = target / "result.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        entry = {"kind": kind, "success": result.returncode == 0 and receipt.get("status") == "evidence_restored_without_model_call",
            "returncode": result.returncode, "status": receipt.get("status"), "log": str(log),
            "receipt": str(receipt_path), "source_run": str(SOURCE_RUN), "elapsed_seconds": time.perf_counter() - tick,
            "scope": "restore archived supplied evidence only; no 8620 request, model output or model-quality result",
            "audit_database_created": (target / "model_calls.sqlite").exists()}
        restored.append(entry)
        summary["replay_restore_only"] = restored
        save(summary)
        print(json.dumps({"phase": "replay_restore_only", **entry}, ensure_ascii=True), flush=True)
    summary["source_sha256_end"] = hashes(sources)
    summary["test_and_runner_sha256_end"] = hashes(test_sources)
    summary["source_unchanged_during_regression"] = summary["source_sha256_start"] == summary["source_sha256_end"]
    summary["tests_and_runner_unchanged_during_regression"] = summary["test_and_runner_sha256_start"] == summary["test_and_runner_sha256_end"]
    summary["success"] = all(summary[key]["success"] for key in ("unit", "guards", "page_javascript")) and all(item["success"] for item in restored) and summary["source_unchanged_during_regression"] and summary["tests_and_runner_unchanged_during_regression"]
    summary["finished_at_utc"] = now()
    save(summary)
    print(json.dumps({"summary": str(SUMMARY), "success": summary["success"],
        "source_unchanged": summary["source_unchanged_during_regression"]}, ensure_ascii=True), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
