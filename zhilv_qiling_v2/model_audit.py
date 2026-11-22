"""Persistent model-call receipts without prompts, responses, keys or contract text."""
from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import REPORT_DIR


@dataclass
class AuditCollector:
    trace_id: str = ""
    contract_id: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


_CONTEXT: contextvars.ContextVar[AuditCollector | None] = contextvars.ContextVar("model_audit", default=None)
_LOCK = threading.RLock()
_ERRORS: list[str] = []
_ALLOWED = {"call_id", "window_id", "job_id", "source_sha256", "requested_model", "actual_model", "attempt", "max_tokens", "prompt_chars", "input_sha256",
            "finish_reason", "pairs", "backend", "device", "quant", "load_seconds", "error_type",
            "http_status", "response_format", "parse_retries", "missing_keys", "unexpected_keys", "reason_code",
            "prompt_tokens", "completion_tokens", "total_tokens", "model_identity_verified",
            "native_thinking", "native_reasoning_format", "native_json_schema_requested",
            "native_json_schema_sent", "json_schema_sha256",
            "generation_sent", "context_method", "context_input_truncation_permitted", "context_generation_sent",
            "context_prompt_tokens", "context_reserved_output_tokens", "context_reserved_stop_token",
            "context_required_context_tokens", "context_server_context_tokens", "context_fits",
            "context_formatted_prompt_sha256",
            "clause_token_start", "clause_token_end", "article_token_start", "article_token_end",
            "planned_pairs", "executed_pairs", "token_coverage_complete", "pair_coverage_complete",
            "semantic_pending", "window_disagreement", "decision_scope"}


@contextlib.contextmanager
def review_audit(trace_id: str = "", contract_id: str = "") -> Iterator[AuditCollector]:
    collector = AuditCollector(str(trace_id), str(contract_id))
    token = _CONTEXT.set(collector)
    try:
        yield collector
    finally:
        _CONTEXT.reset(token)


def _path() -> Path:
    return Path(os.environ.get("ZLQL_MODEL_AUDIT_PATH") or Path(REPORT_DIR) / "model_calls.sqlite")


def scope_metadata() -> dict[str, str]:
    collector = _CONTEXT.get()
    return {"trace_id": collector.trace_id if collector else ""}


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute("""CREATE TABLE IF NOT EXISTS model_calls(
        event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, trace_id TEXT,
        contract_id TEXT, component TEXT NOT NULL, model TEXT NOT NULL,
        operation TEXT NOT NULL, status TEXT NOT NULL, elapsed_seconds REAL,
        reason TEXT, details TEXT NOT NULL)""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_model_calls_trace ON model_calls(trace_id,created_at)")
    return db


def record(component: str, model: str, operation: str, status: str, elapsed_seconds: float = 0,
           reason: str = "", **details: Any) -> dict[str, Any]:
    """Only constant reason codes and allowlisted numeric/metadata details are persisted."""
    collector = _CONTEXT.get()
    safe = {k: v for k, v in details.items() if k in _ALLOWED and isinstance(v, (str, int, float, bool, type(None)))}
    row = {"event_id": uuid.uuid4().hex, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "trace_id": collector.trace_id if collector else "", "contract_id": collector.contract_id if collector else "",
           "component": str(component), "model": str(model), "operation": str(operation), "status": str(status),
           "elapsed_seconds": round(max(0, float(elapsed_seconds)), 6), "reason": str(reason), "details": safe}
    row.update(attempt=safe.get("attempt", 1), elapsed=row["elapsed_seconds"],
               fallback=row["reason"] if row["status"] in ("degraded", "skipped") else "")
    if collector is not None:
        collector.events.append(row)
    try:
        with _LOCK, contextlib.closing(_connect()) as db, db:
            db.execute("INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                row["event_id"], row["created_at"], row["trace_id"], row["contract_id"], row["component"],
                row["model"], row["operation"], row["status"], row["elapsed_seconds"], row["reason"],
                json.dumps(safe, ensure_ascii=False)))
    except (OSError, sqlite3.Error) as exc:
        # Inference remains usable; this visible diagnostic prevents silent loss of auditability.
        _ERRORS.append(type(exc).__name__)
        del _ERRORS[:-10]
    return row


def recent_events(limit: int = 50, trace_id: str = "") -> list[dict[str, Any]]:
    with _LOCK, contextlib.closing(_connect()) as db:
        db.row_factory = sqlite3.Row
        where, params = (" WHERE trace_id=?", [trace_id]) if trace_id else ("", [])
        params.append(max(1, min(500, int(limit))))
        rows = db.execute("SELECT * FROM model_calls" + where + " ORDER BY created_at DESC LIMIT ?", params).fetchall()
    result = []
    for item in rows:
        row = dict(item)
        row["details"] = json.loads(row["details"])
        row.update(attempt=row["details"].get("attempt", 1), elapsed=row["elapsed_seconds"],
                   fallback=row["reason"] if row["status"] in ("degraded", "skipped") else "")
        result.append(row)
    return result


def trace_progress(trace_id: str) -> dict[str, Any]:
    """Read-only SQL counts for one exact trace, including every stored event.

    Started/terminal events are separate counters, not additive inference
    completions. A skipped preflight never enters ``inference_attempts``.
    This observes request execution only, not legal accuracy or completion.
    """
    if not isinstance(trace_id, str) or not trace_id or len(trace_id) > 256:
        raise ValueError("a nonempty exact trace_id is required")
    result: dict[str, Any] = {
        "available": False, "trace_id": trace_id, "total_events": 0,
        "llm": {status: 0 for status in ("started", "succeeded", "failed", "degraded", "skipped")},
        "inference_attempts": 0,
        "entailment": {"predict_pair_succeeded": 0, "neural_rule_disagreement": 0},
        "latest_event": None, "contains_contract_text": False,
        "scope": "exact_trace_all_stored_events; execution_not_legal_completion",
    }
    path = _path()
    if not path.is_file():
        result["reason"] = "audit_database_not_present"
        return result
    try:
        # Do not call _connect(): polling must not create a database, table,
        # index or audit event. A transaction gives both reads one snapshot.
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            grouped = db.execute("""SELECT component,operation,status,reason,COUNT(*) AS n
                FROM model_calls WHERE trace_id=? GROUP BY component,operation,status,reason""", (trace_id,)).fetchall()
            latest = db.execute("""SELECT created_at,component,operation,status,details
                FROM model_calls WHERE trace_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1""", (trace_id,)).fetchone()
        for row in grouped:
            result["total_events"] += row["n"]
            if row["component"] == "llm" and row["operation"] == "request_json" and row["status"] in result["llm"]:
                result["llm"][row["status"]] += row["n"]
            if row["component"] == "entailment":
                if row["operation"] == "predict_pair" and row["status"] == "succeeded":
                    result["entailment"]["predict_pair_succeeded"] += row["n"]
                if row["operation"] == "cross_check" and row["status"] == "degraded" and row["reason"] == "neural_rule_disagreement":
                    result["entailment"]["neural_rule_disagreement"] += row["n"]
        result["inference_attempts"] = result["llm"]["started"]
        if latest is not None:
            details = json.loads(latest["details"])
            result["latest_event"] = {key: latest[key] for key in ("created_at", "component", "operation", "status")}
            result["latest_event"]["job_id"] = details.get("job_id", "") if isinstance(details, dict) else ""
        result["available"] = True
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        # Audit availability must not hide the durable task itself.
        result["reason"] = "audit_read_failed:" + type(exc).__name__
    return result


def status() -> dict[str, Any]:
    return {"storage": str(_path()), "errors": list(_ERRORS), "stores_contract_text": False,
            "stores_api_keys": False}
