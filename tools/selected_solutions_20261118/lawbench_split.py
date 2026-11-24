"""Pin a LawBench subset, resume actual generation, export unchanged Linux scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import benchmark_exports as B


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "infer", "export_linux"):
        command = commands.add_parser(action)
        command.add_argument("--snapshot", type=Path, default=B.DEFAULT_SNAPSHOT)
        command.add_argument("--run-dir", type=Path, required=True)
        if action == "export_linux":
            command.add_argument("--output", type=Path, required=True)
            continue
        command.add_argument("--tasks", required=True, help="comma-separated official IDs, or explicitly all")
        command.add_argument("--shot", choices=["zero_shot", "one_shot"], default="zero_shot")
        command.add_argument("--limit", type=int, required=True, help="fixed official first-N items PER TASK (1..500); a subset is not a full benchmark")
        if action == "infer":
            command.add_argument("--endpoint", default="http://127.0.0.1:8621/v1")
            command.add_argument("--model-revision", required=True)
            command.add_argument("--max-new-items", type=int, default=1, help="new responses this invocation; 0 explicitly means all remaining selected items")
            command.add_argument("--max-tokens", type=int, default=512)
            command.add_argument("--temperature", type=float, default=0.0)
            command.add_argument("--timeout", type=float, default=600.0)
            command.add_argument("--native-thinking", choices=["default", "on", "off"], default="off")
            command.add_argument("--native-reasoning-format", choices=["default", "deepseek"], default="deepseek")
            command.add_argument("--reasoning-effort", default=None)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    try:
        result = getattr(B, action)(**args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "official_evaluator_executed": False, "formal_metrics": None}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
