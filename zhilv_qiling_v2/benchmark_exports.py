"""Keep actual LawBench generation separate from unchanged Linux scoring.

The existing acceptance adapter owns all per-response checkpoints and receipts.
This module only pins a selection and exports verified evidence; it never
manufactures predictions, converts scores, or runs an evaluator on Windows.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "tools/acceptance_vendor/LawBench"
# Exact previously verified sparse snapshot, in addition to every upstream blob hash.
OFFICIAL_SNAPSHOT_MANIFEST_SHA256 = "4a029fea21a066215e2005a8079b6784db25e294f26abee1eb862e8b7247b5c8"


def _adapter():
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("acceptance_lawbench")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selection(snapshot, tasks, shot, limit):
    adapter = _adapter()
    snapshot = Path(snapshot).resolve()
    if shot not in ("zero_shot", "one_shot"):
        raise ValueError("shot must be zero_shot or one_shot")
    if type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("explicit limit 1..500 per official task is required")
    selected = adapter.selected_tasks(tasks)
    adapter.source(snapshot)
    manifest_sha = adapter.digest((snapshot / "official_snapshot.json").read_bytes())
    if manifest_sha != OFFICIAL_SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("official snapshot manifest differs from the exact pinned snapshot")
    files = []
    for task in selected:
        path, rows = adapter.load_task(snapshot, shot, task)
        files.append({"task": task, "path": path.relative_to(snapshot).as_posix(),
                      "sha256": adapter.digest(path.read_bytes()), "official_items": len(rows),
                      "selected_first_indices": [0, limit - 1], "selected_items": limit})
    return {"version": "lawbench-split-selection-v1", "benchmark": "LawBench",
            "repository": adapter.REPOSITORY, "revision": adapter.REVISION,
            "official_snapshot_manifest_sha256": manifest_sha,
            "shot": shot, "tasks": selected, "limit_per_task": limit,
            "selected_items": len(selected) * limit, "official_shot_items": 10000,
            "is_subset": set(selected) != set(adapter.TASKS) or limit != 500,
            "dataset_files": files,
            "selection_policy": "unchanged official first-N records per selected task; no sampling or answer filtering",
            "scoring_status": "not_scored", "formal_metrics": None}


def prepare(snapshot, run_dir, *, tasks, shot, limit):
    """Pin only source/selection; do not create or rewrite inference state."""
    adapter = _adapter()
    run_dir = Path(run_dir).resolve()
    selection = _selection(snapshot, tasks, shot, limit)
    target = run_dir / "split_selection.json"
    if target.exists():
        if _read(target) != selection:
            raise ValueError("fixed split selection changed; use a new run directory")
    else:
        if (run_dir / "state.json").exists():
            raise ValueError("existing inference run has no split selection; use a new run directory")
        adapter.atomic_json(target, selection)
    return {"status": "prepared", "selection": selection,
            "selection_path": str(target), "actual_generation_calls": 0,
            "official_evaluator_executed": False, "formal_metrics": None}


def infer(snapshot, run_dir, *, tasks, shot, limit, endpoint, model_revision,
          max_new_items=1, max_tokens=512, temperature=0.0, timeout=600.0,
          native_thinking="off", native_reasoning_format="deepseek", reasoning_effort=None):
    """Delegate every actual call and recovery to acceptance_lawbench.infer."""
    adapter = _adapter()
    if type(max_new_items) is not int or max_new_items < 0:
        raise ValueError("max_new_items must be >=0; 0 explicitly means all remaining selected items")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not math.isfinite(temperature) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("temperature must be finite and timeout must be positive")
    if not model_revision:
        raise ValueError("an operator-declared model revision/weights identifier is required")
    if native_thinking not in ("default", "on", "off") or native_reasoning_format not in ("default", "deepseek"):
        raise ValueError("unsupported native reasoning configuration")
    prepare(snapshot, run_dir, tasks=tasks, shot=shot, limit=limit)
    args = SimpleNamespace(snapshot=Path(snapshot).resolve(), run_dir=Path(run_dir).resolve(),
        tasks=tasks, shot=shot, item_limit=limit, max_items=max_new_items,
        endpoint=endpoint, model=adapter.MODEL_ID, model_revision=model_revision,
        max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        native_thinking=native_thinking, native_reasoning_format=native_reasoning_format,
        reasoning_effort=reasoning_effort)
    result = adapter.infer(args)
    return {**result, "selected_limit_per_task": limit,
            "official_evaluator_executed": False, "scoring_status": "not_scored", "formal_metrics": None}


VERIFY_SCRIPT = '''"""Verify immutable exported bytes before running any official evaluator."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify():
    entries=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in entries.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('missing or unsafe bundle file: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('bundle SHA-256 mismatch: '+name)
    return {'verified_files':len(entries),'status':'verified'}
if __name__=='__main__':print(json.dumps(verify()))
'''


LINUX_SCRIPT = '''"""Run the unchanged pinned LawBench evaluator only on Linux."""
import csv,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
from verify_bundle import verify
ROOT=Path(__file__).resolve().parent
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    if not sys.platform.startswith('linux'):
        raise SystemExit('Linux required; no Windows-equivalent score is provided')
    verify()
    sys.path.insert(0,str(ROOT/'adapter'))
    import acceptance_lawbench as adapter
    manifest=json.loads((ROOT/'export_manifest.json').read_text(encoding='utf-8'))
    results=ROOT/'results'
    results.mkdir(exist_ok=True)
    status_path=results/'scoring_status.json'
    metrics=results/'official_results.csv'
    if status_path.exists() or metrics.exists():
        raise SystemExit('refusing to overwrite scoring evidence; extract a fresh bundle for another attempt')
    status={'benchmark':'LawBench','status':'not_scored','official_evaluator_executed':False,
            'formal_metrics':None,'full_official_benchmark':False,
            'prediction_scope':manifest['prediction_scope'],'python':sys.executable,
            'official_revision':manifest['official_revision']}
    save(status_path,status)
    preflight=adapter.evaluation_preflight(ROOT/'LawBench',Path(sys.executable),manifest['generated_official_tasks'])
    status['dependency_preflight']=preflight
    if not preflight['ready']:
        status['status']='blocked_before_official_scoring';save(status_path,status)
        print(json.dumps(status,ensure_ascii=False));return 1
    command=[sys.executable,'main.py','-i',str(ROOT/'run/predictions'),'-o',str(metrics)]
    started=time.perf_counter()
    process=subprocess.run(command,cwd=ROOT/'LawBench/evaluation',capture_output=True,text=True,encoding='utf-8',errors='replace')
    (results/'official_stdout.txt').write_text(process.stdout,encoding='utf-8')
    (results/'official_stderr.txt').write_text(process.stderr,encoding='utf-8')
    status.update(official_evaluator_executed=True,command=command,exit_code=process.returncode,
                  elapsed_seconds=time.perf_counter()-started)
    try:verify()
    except Exception as exc:
        status.update(status='bundle_changed_during_scoring',integrity_error=type(exc).__name__+': '+str(exc))
        save(status_path,status);print(json.dumps(status,ensure_ascii=False));return 1
    if process.returncode==0 and metrics.is_file():
        with metrics.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        expected=set(manifest['generated_official_tasks'])
        try:finite_metrics=all(math.isfinite(float(row['score'])) and math.isfinite(float(row['abstention_rate'])) for row in rows)
        except (KeyError,TypeError,ValueError):finite_metrics=False
        if len(rows)!=len(expected) or {row.get('task') for row in rows}!=expected or any(row.get('model_name')!='legalone-8b' for row in rows) or not finite_metrics:
            status['status']='official_output_schema_or_task_mismatch'
        else:
            status.update(status='official_evaluator_executed',official_output_rows=rows,
                          official_metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
                          full_official_benchmark=manifest['full_official_shot_predictions'])
            status['interpretation']='Raw official output for the exact archived items; subset results are not full LawBench acceptance.'
    else:status['status']='official_evaluator_failed'
    save(status_path,status);print(json.dumps(status,ensure_ascii=False))
    return 0 if status['status']=='official_evaluator_executed' else 1
if __name__=='__main__':raise SystemExit(main())
'''


def export_linux(snapshot, run_dir, output):
    """Package unchanged official sources plus receipt-reconciled predictions."""
    adapter = _adapter()
    snapshot, run_dir, output = (Path(path).resolve() for path in (snapshot, run_dir, output))
    if output.exists() or output.is_relative_to(snapshot):
        raise ValueError("export requires a fresh output path outside the official snapshot")
    official = adapter.source(snapshot)
    selection = _read(run_dir / "split_selection.json")
    expected = _selection(snapshot, ",".join(selection["tasks"]), selection["shot"], selection["limit_per_task"])
    if selection != expected:
        raise ValueError("fixed source/selection manifest changed")
    state = _read(run_dir / "state.json")
    if state.get("fixture_only") or state.get("synthetic") is not False or state.get("scope") != "actual_model_http_inference":
        raise ValueError("only actual inference runs may be exported; fixtures are not model evidence")
    config = state["configuration"]
    if (config.get("source_revision") != adapter.REVISION or config.get("shot") != selection["shot"]
            or config.get("selected_official_tasks") != selection["tasks"]
            or config.get("item_limit_per_task") != selection["limit_per_task"]):
        raise ValueError("inference configuration differs from fixed official selection")
    args = SimpleNamespace(snapshot=snapshot, run_dir=run_dir)
    _, completed, receipts = adapter.reconcile_run(args, state, repair=False)
    if (completed != state.get("completed")
            or receipts != sorted(state.get("receipt_files", []), key=lambda row: row["path"])
            or len(receipts) != state.get("model_calls")):
        raise ValueError("state/receipt/prediction ledger differs; resume original infer to reconcile")
    if not completed:
        raise ValueError("no verified model predictions to export")
    payload = {}
    def add(name, path):
        payload[name] = Path(path).read_bytes()
    for entry in official["files"]:
        add("LawBench/" + entry["path"], snapshot / entry["path"])
        if adapter.digest(payload["LawBench/" + entry["path"]]) != entry["sha256"]:
            raise ValueError("official source changed while copying export")
    add("LawBench/official_snapshot.json", snapshot / "official_snapshot.json")
    add("run/state.json", run_dir / "state.json")
    add("run/adapter_snapshot.py", run_dir / "adapter_snapshot.py")
    add("split_selection.json", run_dir / "split_selection.json")
    for task in completed:
        add("run/predictions/" + adapter.MODEL_ID + "/" + task + ".json",
            run_dir / "predictions" / adapter.MODEL_ID / (task + ".json"))
    for receipt in receipts:
        add("run/" + Path(receipt["path"]).as_posix(), run_dir / receipt["path"])
        if adapter.digest(payload["run/" + Path(receipt["path"]).as_posix()]) != receipt["sha256"]:
            raise ValueError("inference receipt changed while copying export")
    if (_read(run_dir / "state.json") != state or json.loads(payload["run/state.json"]) != state
            or json.loads(payload["split_selection.json"]) != selection
            or adapter.digest(payload["run/adapter_snapshot.py"]) != state["adapter_sha256"]):
        raise ValueError("inference state or adapter changed while copying export")
    for task, row in completed.items():
        if adapter.digest(payload["run/predictions/" + adapter.MODEL_ID + "/" + task + ".json"]) != row["sha256"]:
            raise ValueError("prediction changed while copying export")
    for name in ("acceptance_lawbench.py", "acceptance_common.py", "acceptance_fetch_lawbench.py"):
        add("adapter/" + name, ROOT / "tools" / name)
    add("export_sources/benchmark_exports.py", Path(__file__))
    add("export_sources/lawbench_split.py", ROOT / "tools/selected_solutions_20261118/lawbench_split.py")
    raw_receipts = [_read(run_dir / row["path"]) for row in receipts]
    quality = adapter.completion_summary(config, completed, raw_receipts)
    scope = {"shot": config["shot"], "selected_tasks": selection["tasks"],
             "selected_limit_per_task": selection["limit_per_task"],
             "generated_items_per_task": {task: row["items"] for task, row in completed.items()},
             "selected_items_complete": all(completed.get(task, {}).get("items", 0) == selection["limit_per_task"] for task in selection["tasks"]),
             "is_official_subset": not quality["full_official_benchmark"]}
    manifest = {"version": "lawbench-linux-export-v1", "benchmark": "LawBench",
        "official_repository": adapter.REPOSITORY, "official_revision": adapter.REVISION,
        "official_snapshot_manifest_sha256": adapter.digest(payload["LawBench/official_snapshot.json"]),
        "official_evaluator_sha256": adapter.digest(payload["LawBench/evaluation/main.py"]),
        "configuration_sha256": state["configuration_sha256"], "configuration": config,
        "model_identity": state["model_identity"], "actual_call_receipts": receipts,
        "prediction_scope": scope, "prediction_completion_summary": quality,
        "generated_official_tasks": sorted(completed), "full_official_shot_predictions": quality["full_official_benchmark"],
        "export_scoring_status": "not_scored", "official_evaluator_executed": False,
        "formal_metrics": None, "full_official_benchmark": False,
        "run_command": ["python3", "run_official_linux.py"],
        "unchanged_official_command": ["python3", "main.py", "-i", "ABSOLUTE_BUNDLE/run/predictions", "-o", "ABSOLUTE_BUNDLE/results/official_results.csv"],
        "official_working_directory": "LawBench/evaluation",
        "dependency_status": "Linux dependencies and ChERRANT/LTP resources must be verified on target; not bundled as Windows binaries",
        "source_integrity": "official snapshot and original inference evidence copied byte-for-byte; SHA256SUMS.json covers all exported payload files"}
    def json_bytes(value):
        return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    payload["configuration.json"] = json_bytes(config)
    payload["export_manifest.json"] = json_bytes(manifest)
    payload["NOT_SCORED.json"] = json_bytes({"status": "not_scored_at_export", "formal_metrics": None,
        "official_evaluator_executed": False, "meaning": "Export receipt only. Target Linux scoring writes a separate results/scoring_status.json."})
    payload["verify_bundle.py"] = VERIFY_SCRIPT.encode("utf-8")
    payload["run_official_linux.py"] = LINUX_SCRIPT.encode("utf-8")
    payload["README_LINUX.txt"] = ("LawBench generation and Linux official scoring are separate.\n"
        "This archive has NOT been scored. No Windows-equivalent or converted score exists.\n"
        "1. Extract into a fresh directory on Linux.\n2. python3 verify_bundle.py\n"
        "3. In an isolated Linux Python environment, install the dependencies listed in LawBench/README.md, plus evaluator imports (pandas, numpy, jieba, nltk).\n"
        "   Task 2-1 additionally needs working LTP resources, POSIX SIGALRM and a compatible python3 command. No resources were downloaded by this export.\n"
        "4. python3 run_official_linux.py\n"
        "The wrapper checks dependencies then invokes unchanged LawBench/evaluation/main.py from evaluation/.\n"
        "Results and failure evidence go to results/. Run only once per fresh extraction.\n"
        "A first-N subset remains a subset even after official scoring; inspect export_manifest.json.\n"
        "Preserve the ZIP SHA-256 reported by the export CLI as the external archive fingerprint.\n").encode("utf-8")
    payload["SHA256SUMS.json"] = json_bytes({name: adapter.digest(data) for name, data in sorted(payload.items())})
    # Validate again immediately before committing the archive. No evidence is repaired here.
    adapter.source(snapshot)
    for row in receipts:
        if adapter.digest((run_dir / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("inference receipt changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr(name, data)
    return {"status": "exported_not_scored", "archive": str(output),
            "archive_sha256": adapter.digest(output.read_bytes()), "archive_files": len(payload),
            "prediction_scope": scope, "official_evaluator_executed": False,
            "formal_metrics": None, "full_official_benchmark": False}
