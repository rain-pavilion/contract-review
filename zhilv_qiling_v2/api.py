"""对外 HTTP 接口：纯标准库 `http.server`，零第三方 Web 依赖。

接口清单（全部 JSON over HTTP/1.1）：

===========================  ======  ==========================================
路径                         方法    说明
===========================  ======  ==========================================
`/api/health`                GET     能力矩阵、模型清单、版本号
`/api/review`                POST    上传文件（multipart）或传 JSON 文本 → 审查报告
`/api/batch`                 POST    传目录清单 → 批量审查汇总
`/api/retrieve`              POST    传条款文本 → 法条检索结果（调试/演示用）
`/api/models`                GET     已训练的权重清单与训练元信息
`/api/train`                 POST    触发训练（仅 boundary/ner/rerank/risk/calibration）
`/api/crawl`                 POST    触发爬虫（源、数量、是否 OCR）
`/api/ocr`                   POST    单图/单 PDF OCR
`/`                          GET     内嵌 Web UI
===========================  ======  ==========================================

相对 v1 `web_server.py` 的升级：
1. 路由用"表 + 正则"声明式注册，不再是 if/elif 长链；
2. 统一的 `_json()` 出参与异常兜底（任何异常都变成 4xx/5xx JSON，不吐 traceback 给前端）；
3. multipart 自己解析（v1 只吃 JSON 文本），支持真实文件上传；
4. 每个响应都带 `trace_id`，与 `ReviewReport.trace` 对得上；
5. 所有能力缺失（无 OCR、无 LLM）都在 `/api/health` 里显式声明，前端据此隐藏按钮。
"""

from __future__ import annotations

from collections import Counter
import json
import mimetypes
import sqlite3
import threading
import time
import traceback
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from . import __version__
from .config import Config, capabilities, config_from_env, default_config
from .version import VERSION_TAG

#: 单次请求体上限（上传的合同很少超过 40MB）
MAX_BODY_BYTES = 48 * 1024 * 1024

#: 训练/爬虫这类长任务在后台线程里跑，接口立刻返回任务 id
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_QUEUE_LOCK = threading.Lock()
_LAST_MODEL_PROBE: dict[str, Any] = {}


@dataclass
class Route:
    """一条路由：方法 + 正则 + 处理函数。"""

    method: str
    pattern: str
    handler: Callable[["RequestContext"], Any]
    doc: str = ""


@dataclass
class BinaryResponse:
    body: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)


_SERVER_CONFIG: Config | None = None


@dataclass
class RequestContext:
    """一次请求的全部上下文，处理函数只依赖它。"""

    handler: BaseHTTPRequestHandler
    method: str
    path: str
    query: dict[str, list[str]]
    body: bytes
    json_body: Any
    trace_id: str
    params: dict[str, str] = field(default_factory=dict)

    @property
    def config(self) -> Config:
        # 进程级缓存 + 读环境变量。
        # 原来这里直接 `default_config()`，等于**完全不看 ZLQL_LLM_ENDPOINT**——
        # 服务端配了模型地址却永远走规则，是最难查的一类静默失效。
        global _SERVER_CONFIG
        if _SERVER_CONFIG is None:
            _SERVER_CONFIG = config_from_env()
        return _SERVER_CONFIG

    def text(self) -> str:
        """取请求里的合同文本：JSON 的 `text` 字段优先，其次 URL 编码的表单体。"""
        if isinstance(self.json_body, dict) and isinstance(self.json_body.get("text"), str):
            return self.json_body["text"]
        if self.body and self.body[:1] not in (b"{", b"-"):
            decoded = self.body.decode("utf-8", "ignore")
            if "=" in decoded and "&" in decoded:
                form = parse_qs(decoded)
                if form.get("text"):
                    return form["text"][0]
            return decoded
        return ""


# ---------------------------------------------------------------------------
# multipart 解析（只处理单文件 + 普通字段，够用且无依赖）
# ---------------------------------------------------------------------------

def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """解析 `multipart/form-data`，返回 (字段, 文件)。

    文件用 `{字段名: (文件名, 内容)}` 表示。v1 只支持 JSON 文本，前端必须把文件
    读成 base64 再塞进 JSON，体积膨胀 33%；这里直接吃原始字节。
    """
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    marker = "boundary="
    if marker not in content_type:
        return fields, files
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    if not boundary:
        return fields, files
    delimiter = b"--" + boundary.encode("latin-1")
    for chunk in body.split(delimiter):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        header_text = head.decode("utf-8", "ignore")
        name = ""
        filename = ""
        for line in header_text.splitlines():
            lower = line.lower()
            if lower.startswith("content-disposition"):
                for part in line.split(";"):
                    part = part.strip()
                    if part.startswith("name="):
                        name = part[5:].strip('"')
                    elif part.startswith("filename="):
                        filename = part[9:].strip('"')
        if not name:
            continue
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", "ignore")
    return fields, files


# ---------------------------------------------------------------------------
# 处理函数
# ---------------------------------------------------------------------------

def _report_payload(report) -> dict[str, Any]:
    """把 ReviewReport 压成前端好用的结构（保留 trace 与统计）。"""
    # The legacy no-candidate sentinel is an execution notice, never a risk
    # finding or evidence that all pending obligations have passed.
    risks = [risk for risk in report.risks if risk.code != "no_substantive_risk"]
    notices = [risk for risk in report.risks if risk.code == "no_substantive_risk"]
    notice_ids = {risk.risk_id for risk in notices}
    return {
        "contract_id": report.contract_id,
        "source_name": report.source_name,
        "score": report.score,
        "score_breakdown": report.score_breakdown,
        "category_counts": dict(Counter(risk.category for risk in risks)),
        "severity_counts": dict(Counter(risk.severity for risk in risks)),
        "risk_count": len(risks),
        "review_queue": [item for item in report.review_queue
                         if (item if isinstance(item, str) else item.get("id", item.get("risk_id"))) not in notice_ids],
        "conflicts": [c.to_dict() if hasattr(c, "to_dict") else str(c) for c in report.conflicts],
        "risks": [risk.to_dict() if hasattr(risk, "to_dict") else str(risk) for risk in risks],
        "notices": [risk.to_dict() for risk in notices],
        "trace": [event.to_dict() if hasattr(event, "to_dict") else str(event) for event in report.trace],
        "stats": report.stats,
        "schema_version": report.schema_version,
    }


def handle_health(ctx: RequestContext) -> dict[str, Any]:
    caps = capabilities()
    from .learned import registry

    config = ctx.config
    endpoint = (getattr(config, "llm_endpoint", "") or "").strip()
    weights = {}
    try:
        import json as _json

        from .config import MODEL_DIR

        for name in ("legalone-8b", "legalone-1.7b", "qwen2.5-1.5b-instruct"):
            folder = MODEL_DIR / "pretrained" / name
            if not folder.exists():
                continue
            index = folder / "model.safetensors.index.json"
            if index.exists():
                meta = _json.loads(index.read_text(encoding="utf-8"))
                expected = sorted(set((meta.get("weight_map") or {}).values()))
                present = [
                    n for n in expected
                    if (folder / n).exists() and not (folder / (n + ".part")).exists()
                ]
                gb = round(sum((folder / n).stat().st_size for n in present) / 2**30, 2)
                weights[name] = {
                    "shards": len(expected),
                    "present": len(present),
                    "gb": gb,
                    "ready": len(present) == len(expected) and bool(expected),
                    "downloading": len(present) < len(expected),
                }
                continue
            single = folder / "model.safetensors"
            if single.exists():
                weights[name] = {
                    "shards": 1,
                    "present": 1,
                    "gb": round(single.stat().st_size / 2**30, 2),
                    "ready": single.stat().st_size > 50 * 2**20,
                }
    except Exception:
        pass

    from .model_runtime import runtime_status
    return {
        "version": __version__,
        "tag": VERSION_TAG,
        "capabilities": caps.to_dict() if hasattr(caps, "to_dict") else vars(caps),
        "models": registry().summary(),
        "llm": {
            "enabled": bool(getattr(config, "enable_llm", False)),
            "endpoint": endpoint,
            "model": getattr(config, "llm_model", ""),
            "coverage_policy": "all_nonempty_clauses_with_explicit_window_results",
            "chunk_chars": getattr(config, "llm_chunk_chars", 1800),
            "chunk_overlap": getattr(config, "llm_chunk_overlap", 160),
            "max_calls": getattr(config, "llm_max_calls", 0),
            "feedback_window_policy": config.llm_feedback_window_policy,
            "freeze_feedback_cycle": config.freeze_feedback_cycle,
            "initial_source_protocol": config.llm_initial_source_protocol,
            "max_output_tokens": getattr(config, "llm_max_tokens", 512),
            "native_thinking": getattr(config, "llm_native_thinking", ""),
            "native_reasoning_format": getattr(config, "llm_native_reasoning_format", ""),
            "native_json_schema": getattr(config, "llm_native_json_schema", False) is True,
            "native_context_preflight": getattr(config, "llm_native_context_preflight", False) is True,
            "cross_clause_pairs": getattr(config, "llm_cross_clause_pairs", 12),
            "strong": {
                "endpoint": (getattr(config, "llm_strong_endpoint", "") or "").strip(),
                "model": getattr(config, "llm_strong_model", "") or "legalone-8b",
                "top_k": getattr(config, "llm_review_top_k", 0),
                "timeout": getattr(config, "llm_strong_timeout", 0),
            },
            "weights": weights,
        },
        "multi_agent": {"available": True, "default": bool(getattr(config, "enable_multi_agent", False))},
        "runtime": runtime_status(),
        "last_model_probe": _LAST_MODEL_PROBE,
        "trace_id": ctx.trace_id,
    }


def handle_models(ctx: RequestContext) -> dict[str, Any]:
    from .learned import registry

    summary = registry().summary()
    detail: dict[str, Any] = {}
    from .config import MODEL_DIR

    for name, filename in registry().FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        detail[name] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "trained_at": meta.get("trained_at"),
            "keys": sorted(payload.keys())[:8],
        }
    return {"summary": summary, "files": detail, "trace_id": ctx.trace_id}


_ENGINE_ALIASES = {"single": False, "pipeline": False, "multi": True, "goa": True, "agents": True}


def _engine_config(ctx: "RequestContext", config):
    """按请求里的 engine 参数决定走单进程链路还是多智能体编排。

    多智能体要一个新 config（`enable_multi_agent` 是构造期字段语义），
    直接改传入对象会污染服务端常驻配置——并发请求之间会互相串。
    """
    from dataclasses import replace

    raw = ""
    try:
        # ctx.query 来自 parse_qs：值是 list，直接 str() 会得到 "['multi']"
        values = (ctx.query or {}).get("engine") or []
        raw = str(values[0]) if values else ""
    except Exception:
        raw = ""
    if not raw and isinstance(ctx.json_body, dict):
        raw = str(ctx.json_body.get("engine") or "")
    raw = raw.strip().lower()
    if raw not in _ENGINE_ALIASES:
        return config
    try:
        return replace(config, enable_multi_agent=_ENGINE_ALIASES[raw])
    except TypeError:
        config.enable_multi_agent = _ENGINE_ALIASES[raw]
        return config


_LLM_ALIASES = {
    "off": False, "rules": False, "rule": False, "0": False, "false": False,
    "on": True, "llm": True, "model": True, "1": True, "true": True,
}


class LLMConfigurationError(RuntimeError):
    """An explicit main-model request must not become a rules-only review."""


def _ensure_llm_request_configured(raw, config):
    if (_LLM_ALIASES.get(str(raw).strip().lower()) is True
            and not (getattr(config, "llm_endpoint", "") or "").strip()):
        raise LLMConfigurationError("本次明确请求大模型审查，但未配置主模型端点；未执行审查。请启动 start-defense-web.cmd 后重试。")


def _llm_config(ctx: "RequestContext", config):
    """按请求里的 llm 参数决定这一份合同要不要走大模型通道。

    与 `_engine_config` 同一套路：query / JSON body / multipart 字段三种写法都认，
    并且用 `replace` 克隆，避免并发请求互相污染常驻配置。
    """
    from dataclasses import replace

    raw = ""
    try:
        values = (ctx.query or {}).get("llm") or []
        raw = str(values[0]) if values else ""
    except Exception:
        raw = ""
    if not raw and isinstance(ctx.json_body, dict):
        raw = str(ctx.json_body.get("llm") or "")
    raw = raw.strip().lower()
    if raw not in _LLM_ALIASES:
        return config
    want = _LLM_ALIASES[raw]
    _ensure_llm_request_configured(raw, config)
    try:
        return replace(config, enable_llm=want)
    except TypeError:
        config.enable_llm = want
        return config


def handle_review(ctx: RequestContext) -> dict[str, Any]:
    started = time.perf_counter()
    config = _engine_config(ctx, ctx.config)
    config = _llm_config(ctx, config)
    fields, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
    if fields.get("engine"):
        from dataclasses import replace as _replace

        try:
            config = _replace(config, enable_multi_agent=str(fields["engine"]).lower() in ("multi", "goa", "agents"))
        except TypeError:
            pass
    if fields.get("llm") and str(fields["llm"]).lower() in _LLM_ALIASES:
        from dataclasses import replace as _replace2

        want = _LLM_ALIASES[str(fields["llm"]).lower()]
        _ensure_llm_request_configured(fields["llm"], config)
        if not want or (getattr(config, "llm_endpoint", "") or "").strip():
            try:
                config = _replace2(config, enable_llm=want)
            except TypeError:
                pass
    document, error = _request_document(ctx, config, fields, files)
    if error:
        return error
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    save = fields.get("store", body.get("store", False))
    return _review_document(document, config, ctx.trace_id, started,
        time.perf_counter() - started,
        save=save is True or str(save).strip().lower() in ("true", "1", "yes"))


def _review_document(document, config, trace_id, started, parse_seconds, *, save=True, report_id=None, queue_worker_id=None):
    """Shared HTTP/queue path; server timing includes parsing through archiving."""
    from .engine import ReviewEngine
    from .model_audit import review_audit
    from .pipeline import build_contract

    config = _package_review_config(document, config)

    with review_audit(trace_id) as audit:
        stage = time.perf_counter()
        contract = build_contract(document, config)
        audit.contract_id = contract.contract_id
        preparation = time.perf_counter() - stage
        stage = time.perf_counter()
        report = ReviewEngine(config).run(contract)
        review_seconds = time.perf_counter() - stage
        payload = _report_payload(report)
        stage = time.perf_counter()
        payload.update(_workspace_context(contract, config))
        context_seconds = time.perf_counter() - stage
    payload.update({"feedback": {}, "saved": False, "report_id": None,
                    "created_at": _workspace_now(), "trace_id": trace_id})
    timings = {"parse_seconds": round(parse_seconds, 4),
               "prepare_seconds": round(preparation, 4), "review_seconds": round(review_seconds, 4),
               "context_seconds": round(context_seconds, 4), "archive_seconds": 0,
               "server_total_seconds": round(time.perf_counter() - started, 4),
               "scope": "服务端处理至归档，不含客户端网络传输和排队等待"}
    payload.setdefault("stats", {})["timings"] = timings
    payload["stats"]["model_calls"] = audit.events
    from .review_explanations import build_review_explanations
    payload["review_explanations"] = build_review_explanations(payload)
    if save:
        try:
            stage = time.perf_counter()
            _save_workspace_report(payload, config, report_id=report_id, queue_worker_id=queue_worker_id)
            timings["archive_seconds"] = round(time.perf_counter() - stage, 4)
            timings["server_total_seconds"] = round(time.perf_counter() - started, 4)
            with closing(_workspace_db(config)) as connection, connection:
                raw = json.dumps(payload, ensure_ascii=False)
                if queue_worker_id:
                    updated = connection.execute("""UPDATE workspace_reports SET payload=?
                        WHERE report_id=? AND EXISTS (SELECT 1 FROM workspace_tasks
                            WHERE task_id=? AND status='running' AND worker_id=?)""",
                        (raw, payload["report_id"], payload["report_id"], queue_worker_id))
                    if updated.rowcount != 1:
                        raise OSError("任务租约已移交，旧执行者未更新归档")
                else:
                    connection.execute("UPDATE workspace_reports SET payload=? WHERE report_id=?",
                        (raw, payload["report_id"]))
        except (sqlite3.Error, OSError) as exc:
            payload.update({"saved": False, "report_id": None,
                            "storage_warning": f"审查已完成，但档案保存失败（{type(exc).__name__}）；"
                                               "请先导出报告，检查存储权限或稍后重试。"})
    return payload


def _request_document(ctx: RequestContext, config: Config, fields=None, files=None):
    """Common upload/text parsing; pasted names never decide the byte parser."""
    from .parsers import ParseError
    from .pipeline import prepare

    if fields is None or files is None:
        fields, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}

    def finish(document):
        if not document.full_text.strip() or not any(c.text.strip() for c in document.clauses):
            return None, {"error": "未识别到可审查正文，请上传文本型 PDF 或启用 OCR。",
                          "status": 422, "trace_id": ctx.trace_id,
                          "needs_ocr_pages": document.stats.get("needs_ocr_pages", [])}
        return document, None

    try:
        package_id = str(body.get("package_id") or fields.get("package_id") or "")
        if package_id:
            from .contract_packages import prepare_package_document
            include = body.get("include_tender", fields.get("include_tender", False))
            document = prepare_package_document(package_id, config=config,
                include_tender=include is True or str(include).lower() in ("1", "true"),
                store_root=config.contract_packages_path or None)
            return finish(document)
        if files:
            name, content = next(iter(files.values()))
            name = str(name).replace("\\", "/").rsplit("/", 1)[-1][:240]
            supported = {".docx", ".pdf", ".txt", ".md", ".html", ".htm", ".doc",
                         ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
            if Path(name).suffix.lower() not in supported:
                return None, {"error": "不支持此文件格式，请上传 DOCX、PDF、TXT 或 Markdown。",
                              "status": 415, "trace_id": ctx.trace_id}
            document = prepare(name, content, config=config)
            document.file_type = Path(name).suffix.lower().lstrip(".")
            return finish(document)
        text = ctx.text() or fields.get("text", "")
        if text.strip():
            document = prepare("upload.txt", text.encode("utf-8"), config=config)
            supplied = str(body.get("source_name") or fields.get("source_name") or "upload.txt")
            document.source_name = supplied.replace("\\", "/").rsplit("/", 1)[-1][:240]
            document.file_type = "txt"
            return finish(document)
        path_value = str(body.get("path") or "")
        if path_value:
            path = Path(path_value)
            if not path.is_file():
                return None, {"error": "路径不存在或不是文件", "status": 404}
            document = prepare(path.name, path=path, config=config)
            document.file_type = path.suffix.lower().lstrip(".")
            return finish(document)
    except (ParseError, ValueError, OSError) as exc:
        return None, {"error": str(exc), "status": 400, "trace_id": ctx.trace_id}
    return None, {"error": "缺少文件或文本", "hint": "multipart 上传 file 字段，或 JSON 传 text/path",
                  "status": 400, "trace_id": ctx.trace_id}


def handle_analyze(ctx: RequestContext) -> dict[str, Any]:
    """Parse and split only. No LLM, risk review, or archive side effects."""
    document, error = _request_document(ctx, ctx.config)
    if error:
        return error
    return {"document": document.to_dict(), "stats": document.stats, "trace_id": ctx.trace_id}


def _workspace_context(contract, config: Config) -> dict[str, Any]:
    from .graph import build_contract_graph
    from .subject_graph import check_subjects

    # Text-derived hints are explicitly distinct from an external registry check.
    subject = check_subjects(contract, config)
    checks = asdict(subject)
    checks.update({"paths": subject.paths(), "hint": subject.query_hint(),
                   "verification_scope": "合同文本线索＋带来源导入证据；未连接外部在线登记或信用接口，未核验或过期事实保持unknown，来源标记不代表独立认证"})
    graph = build_contract_graph(contract.document.clauses, contract.elements)
    result = {"document": contract.document.to_dict(), "elements": contract.elements.to_dict(),
            "subject_checks": checks, "graph": graph.to_dict(),
            "contract_type": contract.contract_type}
    package = contract.document.stats.get("contract_package")
    if package:
        result["contract_package"] = package
    return result


def _package_review_config(document, config):
    """Bind this run to a verified package index without mutating the server."""
    from dataclasses import replace
    from .contract_packages import package_evidence_summary
    package = document.stats.get("contract_package") or {}
    if not package.get("package_id"):
        return config
    current = package_evidence_summary(package["package_id"], store_root=config.contract_packages_path or None)
    if current["manifest_sha256"] != package.get("manifest_sha256"):
        raise ValueError("合同包证据版本已变化，请重新加载后审查")
    return replace(config, statutes_path=current["statutes_path"])


def handle_contract_packages(ctx):
    from .contract_packages import (list_contract_packages, load_contract_package,
        package_evidence_summary, package_source_path, search_package_background)
    from urllib.parse import quote
    store_root = ctx.config.contract_packages_path or None
    key, action = ctx.params.get("package_id"), ctx.params.get("action")
    try:
        if not key:
            return {"items": list_contract_packages(store_root=store_root), "trace_id": ctx.trace_id}
        source_id = ctx.params.get("source_id")
        if source_id:
            source = package_source_path(key, source_id, store_root=store_root)
            content_type = mimetypes.guess_type(str(source))[0] or "application/octet-stream"
            inline = source.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".txt")
            if not inline:
                content_type = "application/octet-stream"
            return BinaryResponse(source.read_bytes(), content_type,
                {"Content-Disposition": ("inline" if inline else "attachment") + "; filename*=UTF-8''" + quote(source.name),
                 "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})
        if action == "background":
            query = str((ctx.query.get("q") or [""])[0])[:1000]
            return {"package_id": key, "query": query,
                    "items": search_package_background(key, query, store_root=store_root),
                    "scope": "招标文件原文检索；空白模板不证明实际签约授权，检索命中不代表模型已完成审查。"}
        if action == "review" and ctx.method == "POST":
            requested = ctx.json_body if isinstance(ctx.json_body, dict) else {}
            _ensure_llm_request_configured(requested.get("llm", "1" if ctx.config.enable_llm else "0"), ctx.config)
        evidence = package_evidence_summary(key, store_root=store_root)
        if action == "review" and ctx.method == "POST":
            body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
            worker = _ensure_queue_worker(ctx.handler.server, ctx.config)
            options = {"engine": "single" if body.get("engine") == "single" else "multi",
                       "llm": body.get("llm", "1" if ctx.config.enable_llm else "0")}
            batch = worker.store.create_batch(options)
            descriptor = {"package_id": key, "manifest_sha256": evidence["manifest_sha256"],
                          "include_tender": body.get("include_tender") is True}
            task = worker.store.add_task(batch["batch_id"], key + ".contract_package", "contract_package",
                json.dumps(descriptor, ensure_ascii=False).encode("utf-8"))
            worker.store.start_batch(batch["batch_id"])
            return {"batch_id": batch["batch_id"], "task_id": task["task_id"], "package_id": key,
                    "status": "queued", "manifest_sha256": evidence["manifest_sha256"],
                    "include_tender": descriptor["include_tender"], "trace_id": ctx.trace_id}
        return {"package_id": key, "manifest": load_contract_package(key, store_root=store_root, verify_files=False),
                "evidence_summary": evidence, "trace_id": ctx.trace_id}
    except (FileNotFoundError, KeyError) as exc:
        return {"error": str(exc), "status": 404, "trace_id": ctx.trace_id}
    except (ValueError, OSError) as exc:
        return {"error": str(exc), "status": 409, "trace_id": ctx.trace_id}


def _workspace_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace_db(config: Config) -> sqlite3.Connection:
    """Full source/report snapshots, separate from the compact legacy archive."""
    from .config import REPORT_DIR

    path = Path(config.store_path) if config.store_path else REPORT_DIR / "workspace.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS workspace_reports (
            report_id TEXT PRIMARY KEY,
            contract_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            contract_type TEXT NOT NULL,
            score REAL NOT NULL,
            risk_count INTEGER NOT NULL,
            high_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS workspace_reports_created
            ON workspace_reports(created_at DESC);
        CREATE TABLE IF NOT EXISTS workspace_feedback (
            report_id TEXT NOT NULL REFERENCES workspace_reports(report_id) ON DELETE CASCADE,
            risk_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('confirmed','dismissed','pending')),
            note TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY(report_id,risk_id)
        );
        CREATE TABLE IF NOT EXISTS workspace_item_feedback (
            report_id TEXT NOT NULL REFERENCES workspace_reports(report_id) ON DELETE CASCADE,
            item_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('supported','unsupported','pending')),
            reviewer TEXT NOT NULL,
            note TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(report_id,item_id)
        );
    """)
    return connection


def _save_workspace_report(payload: dict[str, Any], config: Config, *, report_id=None, queue_worker_id=None) -> None:
    # Each run is a distinct snapshot: old contract IDs use name+length and can collide.
    report_id = report_id or uuid.uuid4().hex
    payload.update({"report_id": report_id, "saved": True})
    with closing(_workspace_db(config)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        if queue_worker_id:
            task = connection.execute("SELECT status,worker_id FROM workspace_tasks WHERE task_id=?", (report_id,)).fetchone()
            if not task or task["status"] != "running" or task["worker_id"] != queue_worker_id:
                raise OSError("任务租约已移交，旧执行者未写入归档")
        connection.execute("""INSERT INTO workspace_reports
            (report_id,contract_id,source_name,contract_type,score,risk_count,high_count,created_at,payload)
            VALUES (?,?,?,?,?,?,?,?,?)""", (
                report_id, payload["contract_id"], payload["source_name"],
                payload.get("contract_type", "通用"), payload["score"], payload["risk_count"],
                sum(r.get("severity") == "high" for r in payload["risks"]),
                payload["created_at"], json.dumps(payload, ensure_ascii=False),
            ))


def _workspace_feedback(connection: sqlite3.Connection, report_id: str) -> dict[str, Any]:
    return {row["risk_id"]: dict(row) for row in connection.execute(
        "SELECT risk_id,status,note,updated_at FROM workspace_feedback WHERE report_id=?", (report_id,))}


def handle_workspace_reports(ctx: RequestContext) -> dict[str, Any]:
    try:
        page = max(1, int((ctx.query.get("page") or ["1"])[0]))
        size = max(1, min(100, int((ctx.query.get("size") or ["20"])[0])))
    except (TypeError, ValueError):
        return {"error": "page 和 size 必须为整数", "status": 400}
    query = str((ctx.query.get("q") or [""])[0]).strip()[:200]
    where = ("WHERE source_name LIKE ? ESCAPE '\\' OR contract_type LIKE ? ESCAPE '\\' "
             "OR contract_id LIKE ? ESCAPE '\\'") if query else ""
    literal = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    values = (f"%{literal}%",) * 3 if query else ()
    with closing(_workspace_db(ctx.config)) as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM workspace_reports {where}", values).fetchone()[0]
        rows = connection.execute(f"""SELECT report_id,contract_id,source_name,contract_type,score,
            risk_count,high_count,created_at,
            (SELECT COUNT(*) FROM workspace_feedback f
             WHERE f.report_id=workspace_reports.report_id AND f.status!='pending') AS reviewed_count
            FROM workspace_reports {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?""",
            (*values, size, (page - 1) * size)).fetchall()
    items = [dict(row) for row in rows]
    for item in items:
        item["pending_count"] = max(0, item["risk_count"] - item["reviewed_count"])
    return {"items": items, "total": total, "page": page, "size": size, "trace_id": ctx.trace_id}


def handle_workspace_report(ctx: RequestContext) -> dict[str, Any]:
    report_id = ctx.params.get("report_id", "")
    with closing(_workspace_db(ctx.config)) as connection:
        row = connection.execute("SELECT payload FROM workspace_reports WHERE report_id=?", (report_id,)).fetchone()
        if row is None:
            return {"error": "审查记录不存在", "status": 404}
        payload = json.loads(row["payload"])
        payload["feedback"] = _workspace_feedback(connection, report_id)
        payload["item_feedback"] = _workspace_item_feedback(connection, report_id)
    from .review_explanations import build_review_explanations
    payload["review_explanations"] = build_review_explanations(payload)
    return payload


def _workspace_item_feedback(connection, report_id):
    return {row["item_id"]: dict(row) for row in connection.execute(
        "SELECT item_id,status,reviewer,note,updated_at FROM workspace_item_feedback WHERE report_id=?",
        (report_id,))}


def handle_workspace_item_feedback(ctx):
    """Human-declared opinion only; machine ledgers and facts stay immutable."""
    from .review_explanations import build_review_explanations
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    item_id, status, reviewer, note = (body.get(k, "") for k in ("item_id", "status", "reviewer", "note"))
    if (not all(isinstance(v, str) for v in (item_id, status, reviewer, note))
            or status not in ("supported", "unsupported", "pending")
            or not reviewer.strip() or len(reviewer) > 120 or not note.strip() or len(note) > 4000):
        return {"error": "请填写复核人、核对依据和有效意见；姓名限120字，依据限4000字", "status": 400}
    report_id = ctx.params["report_id"]
    with closing(_workspace_db(ctx.config)) as connection, connection:
        row = connection.execute("SELECT payload FROM workspace_reports WHERE report_id=?", (report_id,)).fetchone()
        if row is None:
            return {"error": "审查记录不存在", "status": 404}
        payload = json.loads(row["payload"])
        if item_id not in {item["item_id"] for item in build_review_explanations(payload)["items"]}:
            return {"error": "该事项不属于此报告", "status": 404}
        connection.execute("""INSERT INTO workspace_item_feedback
            (report_id,item_id,status,reviewer,note,updated_at) VALUES (?,?,?,?,?,?)
            ON CONFLICT(report_id,item_id) DO UPDATE SET status=excluded.status,
            reviewer=excluded.reviewer,note=excluded.note,updated_at=excluded.updated_at""",
            (report_id, item_id, status, reviewer.strip(), note.strip(), _workspace_now()))
        feedback = _workspace_item_feedback(connection, report_id)
    return {"report_id": report_id, "item_feedback": feedback, "saved": True,
            "scope": "人工填写意见单独保存；不改变机器结论、主体事实或回环待办"}


def handle_workspace_feedback(ctx: RequestContext) -> dict[str, Any]:
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    risk_id = str(body.get("risk_id") or "")
    status = str(body.get("status") or "")
    note = str(body.get("note") or "")
    if not risk_id or status not in ("confirmed", "dismissed", "pending") or len(note) > 4000:
        return {"error": "需要有效 risk_id 和 status（confirmed / dismissed / pending），备注限4000字", "status": 400}
    report_id = ctx.params.get("report_id", "")
    with closing(_workspace_db(ctx.config)) as connection, connection:
        row = connection.execute("SELECT payload FROM workspace_reports WHERE report_id=?", (report_id,)).fetchone()
        if row is None:
            return {"error": "审查记录不存在", "status": 404}
        payload = json.loads(row["payload"])
        if risk_id not in {str(r.get("risk_id")) for r in payload.get("risks", [])}:
            return {"error": "该风险不属于此审查记录", "status": 404}
        connection.execute("""INSERT INTO workspace_feedback(report_id,risk_id,status,note,updated_at)
            VALUES (?,?,?,?,?) ON CONFLICT(report_id,risk_id) DO UPDATE SET
            status=excluded.status,note=excluded.note,updated_at=excluded.updated_at""",
            (report_id, risk_id, status, note, _workspace_now()))
        feedback = _workspace_feedback(connection, report_id)
    return {"report_id": report_id, "feedback": feedback, "saved": True, "trace_id": ctx.trace_id}


def _queue_path(config):
    from .config import REPORT_DIR
    return Path(config.store_path) if config.store_path else REPORT_DIR / "workspace.sqlite"


def _run_workspace_task(task, base_config):
    from dataclasses import replace
    from .pipeline import prepare
    from .workspace_queue import PermanentTaskError

    config = replace(base_config, enable_multi_agent=task["options"]["engine"] == "multi",
        enable_llm=bool(task["options"]["llm"] and base_config.llm_endpoint))
    # The task ID is also the archive ID: recovering a lease never duplicates a saved review.
    with closing(_workspace_db(config)) as connection:
        existing = connection.execute("SELECT payload FROM workspace_reports WHERE report_id=?", (task["task_id"],)).fetchone()
    if existing:
        payload = json.loads(existing["payload"])
    else:
        if task["options"]["llm"] and not (base_config.llm_endpoint or "").strip():
            raise PermanentTaskError("任务要求主大模型，但当前服务未配置模型端点；未改为规则审查。")
        started = time.perf_counter()
        try:
            if task["file_type"] == "contract_package":
                from .contract_packages import prepare_package_document, package_evidence_summary
                descriptor = json.loads(bytes(task["content"]))
                evidence = package_evidence_summary(descriptor["package_id"], store_root=config.contract_packages_path or None)
                if evidence["manifest_sha256"] != descriptor["manifest_sha256"]:
                    raise ValueError("排队后合同包发生变化，已停止审查；请重新提交当前证据版本")
                document = prepare_package_document(descriptor["package_id"], config=config,
                    include_tender=descriptor.get("include_tender") is True,
                    store_root=config.contract_packages_path or None)
            else:
                document = prepare(task["source_name"], bytes(task["content"]), config=config)
        except (ValueError, OSError) as exc:
            raise PermanentTaskError(str(exc)) from exc
        if not document.full_text.strip() or not any(c.text.strip() for c in document.clauses):
            raise PermanentTaskError("没有可审查正文；请先完成OCR或补充文本版本")
        document.file_type = task["file_type"]
        payload = _review_document(document, config, task["task_id"][:12], started,
            time.perf_counter() - started, save=True, report_id=task["task_id"],
            queue_worker_id=task.get("worker_id"))
        if not payload.get("saved"):
            raise OSError(payload.get("storage_warning", "报告未归档"))
    summary = {"score": payload["score"], "risk_count": payload["risk_count"],
        "high_count": payload.get("severity_counts", {}).get("high", 0)}
    return payload["report_id"], summary


def _ensure_queue_worker(server, config):
    from .workspace_queue import QueueStore, QueueWorker

    path = str(_queue_path(config).resolve())
    with _QUEUE_LOCK:
        workers = getattr(server, "workspace_workers", None)
        if workers is None:
            workers = server.workspace_workers = {}
        if path not in workers:
            worker = QueueWorker(QueueStore(path), lambda task: _run_workspace_task(task, config))
            workers[path] = worker
            worker.start()
        return workers[path]


def handle_workspace_batches(ctx):
    worker = _ensure_queue_worker(ctx.handler.server, ctx.config)
    if ctx.method == "POST":
        body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
        _ensure_llm_request_configured(body.get("llm", "0"), ctx.config)
        return worker.store.create_batch(body)
    return {"items": worker.store.list_batches(), "persistent": True,
            "recovery": "过期租约最多30秒后恢复；已归档任务不会重复生成报告"}


def handle_workspace_batch(ctx):
    from .workspace_queue import QueueError

    store = _ensure_queue_worker(ctx.handler.server, ctx.config).store
    key, action = ctx.params["batch_id"], ctx.params.get("action")
    try:
        if ctx.method == "GET":
            page = int((ctx.query.get("page") or [1])[0])
            size = int((ctx.query.get("size") or [100])[0])
            result = store.get_batch(key, page=page, size=size)
            from .model_audit import trace_progress
            for item in result.get("items", []):
                if item.get("status") == "running":
                    item["runtime_progress"] = trace_progress(item["task_id"][:12])
            return result
        if action == "tasks":
            _, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
            if len(files) != 1:
                return {"error": "每次上传一个file字段，文件会持久保存后再提交批次", "status": 400}
            name, content = next(iter(files.values()))
            return store.add_task(key, name, Path(name).suffix, content)
        if action == "start":
            store.start_batch(key)
        elif action == "cancel":
            store.cancel_batch(key)
        elif action == "retry":
            store.retry_batch(key)
        else:
            return {"error": "未知队列操作", "status": 400}
        return store.get_batch(key)
    except (QueueError, TypeError, ValueError) as exc:
        return {"error": str(exc), "status": 400}


def handle_subject_facts(ctx):
    from .subject_facts import import_facts, load_facts
    from .config import REPORT_DIR

    path = ctx.config.subject_facts_path or str(REPORT_DIR / "subject_facts.json")
    if ctx.method == "GET":
        records, degraded = load_facts(path)
        return {"records": records, "total": len(records), "degraded": degraded,
                "scope": "带来源主体证据导入；source.actor_kind区分human、adapter和legacy_unspecified；不会访问来源URL，导入或来源标记不代表独立认证"}
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    records = body.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        return {"error": "records须为1至1000条带来源、提供者、方法和日期的主体证据", "status": 400}
    try:
        return import_facts(records, path)
    except (ValueError, OSError) as exc:
        return {"error": str(exc), "status": 400}


def handle_subject_materials(ctx):
    from .subject_materials import SubjectMaterialStore, decode_content
    from .config import REPORT_DIR
    path = Path(ctx.config.subject_facts_path or str(REPORT_DIR / "subject_facts.json"))
    store = SubjectMaterialStore(path.parent / (path.stem + "_materials"), path)
    if ctx.method == "GET":
        return store.list_materials()
    body = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    try:
        return store.import_material(body.get("filename"), decode_content(body.get("content_base64")), body.get("metadata"))
    except (ValueError, OSError, TypeError) as exc:
        return {"error": str(exc), "status": 400}


def handle_model_events(ctx):
    from .model_audit import recent_events
    try:
        count = int((ctx.query.get("limit") or [50])[0])
    except (ValueError, TypeError):
        return {"error": "limit必须是整数", "status": 400}
    return {"items": recent_events(count), "contains_contract_text": False}


def handle_annotations(ctx):
    from .annotation_workspace import AnnotationWorkspace, handle_request
    import os
    root = Path(os.environ.get("ZLQL_ANNOTATION_WORKSPACE") or
                Path(__file__).resolve().parents[1] / "data" / "annotation_workspace")
    workspace = AnnotationWorkspace(root)
    return handle_request(workspace, ctx.method, ctx.path, ctx.json_body,
        ctx.handler.headers.get("Authorization", ""),
        {key: values[-1] for key, values in ctx.query.items() if values})


def handle_model_probe(ctx):
    from .llm_client import LLMConfig, probe
    from .model_audit import review_audit
    cfg = ctx.config
    if not cfg.llm_endpoint:
        return {"error": "尚未配置模型端点", "status": 409}
    with review_audit(ctx.trace_id, "runtime-probe") as audit:
        result = probe(LLMConfig(endpoint=cfg.llm_endpoint, model=cfg.llm_model,
            api_key=cfg.llm_api_key, timeout=cfg.llm_timeout), max_tokens=16)
    global _LAST_MODEL_PROBE
    _LAST_MODEL_PROBE = {"tested_at": _workspace_now(), "result": result, "calls": audit.events}
    return _LAST_MODEL_PROBE


def handle_batch(ctx: RequestContext) -> dict[str, Any]:
    from .pipeline import batch_review, dataset_files

    config = _engine_config(ctx, ctx.config)
    config = _llm_config(ctx, config)
    payload = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    paths: list[Path] = []
    for item in payload.get("paths", []) or []:
        candidate = Path(str(item))
        if candidate.exists():
            paths.append(candidate)
    if not paths:
        root = payload.get("root")
        if root:
            paths = dataset_files(Path(str(root)))
        else:
            paths = dataset_files()
    if not paths:
        return {"error": "没有可用样本", "status": 400}
    limit = int(payload.get("limit") or 0)
    if limit > 0:
        paths = paths[:limit]
    summary = batch_review(paths, config=config)
    reports = list(summary.reports)
    scores = [report.score for report in reports]
    codes: dict[str, int] = {}
    for report in reports:
        for risk in report.risks:
            codes[risk.code] = codes.get(risk.code, 0) + 1
    compact = {
        "count": len(reports),
        "failed": summary.failed,
        "elapsed_seconds": round(summary.elapsed, 2),
        "score_mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "risks_mean": round(sum(len(r.risks) for r in reports) / len(reports), 2) if reports else 0.0,
        "code_histogram": dict(sorted(codes.items(), key=lambda pair: -pair[1])),
        "items": [
            {"contract_id": r.contract_id, "score": r.score, "risks": len(r.risks)}
            for r in reports[:200]
        ],
    }
    return {"summary": compact, "trace_id": ctx.trace_id}


def handle_retrieve(ctx: RequestContext) -> dict[str, Any]:
    from .retrieve import get_retriever, rewrite_query
    from .config import V1_STATUTES

    payload = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    query = str(payload.get("text") or ctx.text() or "").strip()
    if not query:
        return {"error": "缺少 text", "status": 400}
    contract_type = str(payload.get("contract_type") or "通用")
    top_k = int(payload.get("top_k") or ctx.config.top_k)
    plan = rewrite_query(query, contract_type=contract_type,
                         codes=payload.get("codes") or ())
    config = ctx.config
    if payload.get("package_id"):
        from dataclasses import replace
        from .contract_packages import package_evidence_summary
        try:
            package = package_evidence_summary(str(payload["package_id"]), store_root=config.contract_packages_path or None)
            config = replace(config, statutes_path=package["statutes_path"])
        except (ValueError, OSError) as exc:
            return {"error": str(exc), "status": 409}
    retriever = get_retriever(config.statute_file(), config)
    result = retriever.search(plan, top_k=top_k)
    return {
        "query": plan.raw,
        "rewritten": result.stats.get("query", plan.rewritten),
        "rewrite_candidate": plan.rewritten,
        "effective_query": result.stats.get("query", plan.raw),
        "slots": plan.slots,
        "tags": result.stats.get("tags", plan.tags),
        "hits": [
            {
                "rank": hit.rank,
                "score": round(hit.score, 6),
                "law": hit.law_name,
                "article": hit.article_no,
                "content": hit.content[:220],
                "risk_tags": hit.risk_tags,
                "article_uid": hit.article_uid,
                "source_url": hit.source_url,
                "provenance": hit.provenance,
            }
            for hit in result.hits
        ],
        "stats": result.stats,
        "package_id": payload.get("package_id"),
        "trace_id": ctx.trace_id,
    }


def _start_job(kind: str, func: Callable[[], Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"id": job_id, "kind": kind, "status": "running",
                         "started_at": time.time(), "result": None, "error": None}

    def runner() -> None:
        try:
            result = func()
            _JOBS[job_id].update(status="done", result=result,
                                 finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 - 任务线程必须吞掉异常并记录
            _JOBS[job_id].update(status="failed", error=f"{exc}",
                                 traceback=traceback.format_exc()[-1200:],
                                 finished_at=time.time())

    threading.Thread(target=runner, name=f"zlql-job-{job_id}", daemon=True).start()
    return job_id


def handle_job(ctx: RequestContext) -> dict[str, Any]:
    job_id = ctx.params.get("job_id", "")
    job = _JOBS.get(job_id)
    if not job:
        return {"error": "任务不存在", "id": job_id, "status": 404}
    return job


def handle_train(ctx: RequestContext) -> dict[str, Any]:
    from . import train as train_module

    payload = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    only = payload.get("only") or []
    quick = bool(payload.get("quick"))
    job_id = _start_job("train", lambda: train_module.main(
        (["--only", ",".join(only)] if only else []) + (["--quick"] if quick else [])
    ))
    return {"job_id": job_id, "kind": "train", "only": only, "quick": quick}


def handle_crawl(ctx: RequestContext) -> dict[str, Any]:
    from .crawl import SOURCES, crawl

    payload = ctx.json_body if isinstance(ctx.json_body, dict) else {}
    wanted = payload.get("sources") or ["ccgp_beijing"]
    limit = int(payload.get("limit") or 5)
    pages = int(payload.get("pages") or 1)
    use_ocr = bool(payload.get("use_ocr"))
    token = payload.get("token")
    rate = float(payload.get("rate") or 1.0)
    names = [name for name in wanted if name in SOURCES]
    if not names:
        return {"error": "未知数据源", "known": sorted(SOURCES), "status": 400}

    def job() -> Any:
        return crawl(limit=limit, pages=pages, sources=names, use_ocr=use_ocr,
                     token=token, rate=rate)

    return {"job_id": _start_job("crawl", job), "sources": names, "limit": limit,
            "pages": pages, "use_ocr": use_ocr}


def handle_ocr(ctx: RequestContext) -> dict[str, Any]:
    from .config import CACHE_DIR
    from .ocr import OcrBudget, get_engine

    engine = get_engine()
    if engine is None:
        return {"error": "OCR 引擎不可用", "detail": capabilities().to_dict(), "status": 503}
    fields, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
    if not files:
        return {"error": "缺少文件", "status": 400}
    name, content = next(iter(files.values()))
    suffix = Path(name).suffix.lower() or ".png"
    temp_dir = CACHE_DIR / "api_ocr"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{ctx.trace_id}{suffix}"
    temp_path.write_bytes(content)
    budget = OcrBudget(
        max_pages=int(fields.get("max_pages") or 30),
        max_seconds=float(fields.get("max_seconds") or 300),
        dpi=int(fields.get("dpi") or 200),
    )
    if suffix == ".pdf":
        result = engine.recognize_pdf(temp_path, budget=budget)
        page_payload = [
            {"page": getattr(page, "page", index + 1),
             "chars": len(getattr(page, "text", "") or ""),
             "confidence": round(float(getattr(page, "confidence", 0.0) or 0.0), 4),
             "source": getattr(page, "source", "")}
            for index, page in enumerate(getattr(result, "pages", []) or [])
        ]
        payload = {
            "engine": getattr(result, "engine", getattr(engine, "engine_name", "")),
            "pages": len(page_payload),
            "avg_confidence": round(
                sum(item["confidence"] for item in page_payload) / len(page_payload), 4
            ) if page_payload else 0.0,
            "chars": sum(item["chars"] for item in page_payload),
            "page_details": page_payload[:60],
            "stats": getattr(result, "stats", {}),
            "degraded": getattr(result, "degraded", []),
            "text_preview": "".join(
                getattr(page, "text", "") or "" for page in (getattr(result, "pages", []) or [])
            )[:1500],
        }
    else:
        page = engine.recognize_image(temp_path)
        payload = {
            "engine": getattr(engine, "engine_name", ""),
            "pages": 1,
            "avg_confidence": round(float(getattr(page, "confidence", 0.0) or 0.0), 4),
            "chars": len(getattr(page, "text", "") or ""),
            "text_preview": (getattr(page, "text", "") or "")[:1500],
        }
    return {"file": name, "result": payload, "trace_id": ctx.trace_id}


# ---------------------------------------------------------------------------
# 路由表
# ---------------------------------------------------------------------------

def handle_history(ctx: RequestContext) -> dict[str, Any]:
    """审查历史（分页）。库不存在时返回空表而不是 500 —— 首次启动就是这个状态。"""
    from .store import default_store

    try:
        store = default_store(ctx.config)
    except Exception as exc:  # noqa: BLE001 - 库打不开也要给前端一个可用响应
        return {"items": [], "total": 0, "page": 1, "size": 0, "note": f"存储不可用：{exc}"}
    query = ctx.query
    page = int((query.get("page") or ["1"])[0] or 1)
    size = int((query.get("size") or ["20"])[0] or 20)
    code = (query.get("code") or [""])[0]
    severity = (query.get("severity") or [""])[0]
    try:
        # list_contracts() 返回的是 (rows, total) 元组，不是裸列表
        rows, total = store.list_contracts(page=page, size=size, code=code, severity=severity)
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "total": 0, "page": page, "size": size, "note": f"查询失败：{exc}"}
    items = [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in rows]
    return {"items": items, "total": int(total), "page": page, "size": size, "trace_id": ctx.trace_id}


def handle_history_detail(ctx: RequestContext) -> dict[str, Any]:
    """单份记录的风险明细（含引用条文）。"""
    from .store import default_store

    contract_id = ctx.params.get("contract_id", "")
    try:
        store = default_store(ctx.config)
        risks = store.get_risks(contract_id)
    except Exception as exc:  # noqa: BLE001
        return {"contract_id": contract_id, "risks": [], "note": f"读取失败：{exc}"}
    return {
        "contract_id": contract_id,
        "risks": [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in risks],
        "trace_id": ctx.trace_id,
    }


def handle_store_stats(ctx: RequestContext) -> dict[str, Any]:
    """全库统计：合同数、风险数、严重度分布、高频风险码。"""
    from .store import default_store

    try:
        stats = default_store(ctx.config).stats()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"存储不可用：{exc}"}
    payload = stats.as_dict() if hasattr(stats, "as_dict") else dict(stats)
    payload["available"] = True
    payload["trace_id"] = ctx.trace_id
    return payload


def handle_case_references(ctx: RequestContext) -> dict[str, Any]:
    """Historical reference metadata only; never invoke review, training or facts."""
    from .case_references import case_references
    return {**case_references((ctx.query.get("q") or [""])[0]), "trace_id": ctx.trace_id}


ROUTES: list[Route] = [
    Route("GET", r"^/api/workspace/contract-packages$", handle_contract_packages, "真实合同包清单"),
    Route("GET", r"^/api/workspace/contract-packages/(?P<package_id>[A-Za-z0-9_-]+)$", handle_contract_packages, "合同包来源与缺件"),
    Route("GET", r"^/api/workspace/contract-packages/(?P<package_id>[A-Za-z0-9_-]+)/sources/(?P<source_id>[A-Za-z0-9_-]+)$", handle_contract_packages, "完整性校验后读取原件"),
    Route("GET", r"^/api/workspace/contract-packages/(?P<package_id>[A-Za-z0-9_-]+)/(?P<action>background)$", handle_contract_packages, "招标背景逐页检索"),
    Route("POST", r"^/api/workspace/contract-packages/(?P<package_id>[A-Za-z0-9_-]+)/(?P<action>review)$", handle_contract_packages, "提交真实合同包持久审查"),
    Route("GET", r"^/api/annotations(?:/.*)?$", handle_annotations, "独立人工盲标工作台"),
    Route("POST", r"^/api/annotations(?:/.*)?$", handle_annotations, "保存盲标、仲裁与冻结"),
    Route("GET", r"^/api/health$", handle_health, "能力与版本"),
    Route("POST", r"^/api/runtime/probe$", handle_model_probe, "真实模型调用探测"),
    Route("GET", r"^/api/runtime/events$", handle_model_events, "逐次调用与降级审计"),
    Route("GET", r"^/api/models$", handle_models, "模型清单"),
    Route("GET", r"^/api/references/cases$", handle_case_references, "公开历史案例参考摘要"),
    Route("POST", r"^/api/review$", handle_review, "单份审查"),
    Route("POST", r"^/api/analyze$", handle_analyze, "只解析与切分"),
    Route("GET", r"^/api/workspace/reports$", handle_workspace_reports, "完整审查档案"),
    Route("GET", r"^/api/workspace/reports/(?P<report_id>[0-9a-f]{32})$", handle_workspace_report, "完整报告与原文"),
    Route("POST", r"^/api/workspace/reports/(?P<report_id>[0-9a-f]{32})/feedback$", handle_workspace_feedback, "保存人工复核"),
    Route("POST", r"^/api/workspace/reports/(?P<report_id>[0-9a-f]{32})/item-feedback$", handle_workspace_item_feedback, "保存逐项证据核对意见"),
    Route("GET", r"^/api/workspace/batches$", handle_workspace_batches, "持久批次列表"),
    Route("POST", r"^/api/workspace/batches$", handle_workspace_batches, "创建持久批次"),
    Route("GET", r"^/api/workspace/batches/(?P<batch_id>[0-9a-f]{32})$", handle_workspace_batch, "批次进度与分页任务"),
    Route("POST", r"^/api/workspace/batches/(?P<batch_id>[0-9a-f]{32})/(?P<action>tasks|start|cancel|retry)$", handle_workspace_batch, "上传/提交/停止/重试"),
    Route("GET", r"^/api/workspace/subject-facts$", handle_subject_facts, "查看主体事实来源"),
    Route("POST", r"^/api/workspace/subject-facts$", handle_subject_facts, "导入带来源主体证据"),
    Route("GET", r"^/api/workspace/subject-materials$", handle_subject_materials, "查看主体原件与完整性"),
    Route("POST", r"^/api/workspace/subject-materials$", handle_subject_materials, "导入主体原件并绑定待核验事实"),
    Route("POST", r"^/api/batch$", handle_batch, "批量审查"),
    Route("POST", r"^/api/retrieve$", handle_retrieve, "法条检索"),
    Route("POST", r"^/api/train$", handle_train, "触发训练"),
    Route("POST", r"^/api/crawl$", handle_crawl, "触发爬虫"),
    Route("POST", r"^/api/ocr$", handle_ocr, "OCR"),
    Route("GET", r"^/api/jobs/(?P<job_id>[0-9a-f]+)$", handle_job, "任务状态"),
    Route("GET", r"^/api/history$", handle_history, "审查历史（分页）"),
    Route("GET", r"^/api/history/(?P<contract_id>[^/]+)$", handle_history_detail, "单份历史明细"),
    Route("GET", r"^/api/stats$", handle_store_stats, "全库统计"),
]

#: Web UI 文件（与包同目录）
UI_FILE = Path(__file__).resolve().parent / "webui.html"


class Handler(BaseHTTPRequestHandler):
    """把请求交给路由表；任何异常都转成 JSON 错误。"""

    server_version = f"ZhilvQiling/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- 输出 ---------------------------------------------------------
    def _send(self, status: int, payload: bytes, content_type: str, headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_BODY_BYTES:
            return b""
        return self.rfile.read(length)

    # -- 分发 ---------------------------------------------------------
    def _dispatch(self, method: str) -> None:
        import re

        trace_id = uuid.uuid4().hex[:12]
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        if method == "OPTIONS":
            self._send(204, b"", "text/plain; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            if not UI_FILE.exists():
                self._json({"error": "webui.html 缺失", "status": 404}, 404)
                return
            body = UI_FILE.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if method in ("GET", "HEAD") and path in ("/annotations", "/annotation.html"):
            from .annotation_workspace import page_bytes
            self._send(200, page_bytes(), "text/html; charset=utf-8")
            return
        body = self._read_body()
        json_body: Any = None
        if body[:1] in (b"{", b"["):
            try:
                json_body = json.loads(body.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                json_body = None
        for route in ROUTES:
            if route.method != method:
                continue
            match = re.match(route.pattern, path)
            if not match:
                continue
            ctx = RequestContext(
                handler=self, method=method, path=path,
                query=parse_qs(parsed.query), body=body, json_body=json_body,
                trace_id=trace_id, params=match.groupdict(),
            )
            try:
                result = route.handler(ctx)
            except LLMConfigurationError as exc:
                self._json({"error": str(exc), "trace_id": trace_id,
                            "llm_requested": True, "review_executed": False}, 503)
                return
            except Exception as exc:  # noqa: BLE001 - 统一兜底
                self._json({"error": f"{type(exc).__name__}: {exc}",
                            "trace_id": trace_id,
                            "traceback": traceback.format_exc()[-1500:]}, 500)
                return
            if isinstance(result, BinaryResponse):
                self._send(200, result.body, result.content_type, result.headers)
                return
            status = 200
            if isinstance(result, dict) and isinstance(result.get("status"), int):
                status = result["status"]
                result = {key: value for key, value in result.items() if key != "status"}
            self._json(result, status)
            return
        self._json({"error": "未知路径", "path": path, "method": method,
                    "routes": [f"{r.method} {r.pattern}" for r in ROUTES]}, 404)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # 默认实现会往 stderr 刷日志；这里压成一行，方便挂后台跑
        if "--verbose" in getattr(self.server, "flags", ()):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)


def _env_host() -> str:
    """监听地址：容器/云平台必须 0.0.0.0，本地默认只监听回环。

    优先级 ZLQL_HOST > HOST > 127.0.0.1；只要检测到 PaaS 常见的 PORT 变量
    就默认放通到 0.0.0.0，避免"部署成功但探针连不上"这类最常见的翻车。
    """
    import os

    explicit = os.environ.get("ZLQL_HOST") or os.environ.get("HOST")
    if explicit:
        return explicit
    if os.environ.get("PORT") or os.environ.get("RENDER") or os.environ.get("SPACE_ID"):
        return "0.0.0.0"
    return "127.0.0.1"


def _env_port(default: int = 8610) -> int:
    """监听端口：ZLQL_PORT > PORT > 8610。"""
    import os

    for key in ("ZLQL_PORT", "PORT"):
        raw = os.environ.get(key)
        if raw and raw.strip().isdigit():
            return int(raw.strip())
    return default


class WorkspaceServer(ThreadingHTTPServer):
    def server_close(self):
        for worker in getattr(self, "workspace_workers", {}).values():
            worker.stop()
        super().server_close()


def serve(host: str | None = None, port: int | None = None, *, verbose: bool = False) -> ThreadingHTTPServer:
    """启动 HTTP 服务（阻塞前先打印一行启动信息）。"""
    from .model_runtime import initialize_backend
    initialize_backend(_SERVER_CONFIG or config_from_env())
    host = host or _env_host()
    port = _env_port() if port is None else port
    httpd = WorkspaceServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.flags = ("--verbose",) if verbose else ()  # type: ignore[attr-defined]
    _ensure_queue_worker(httpd, _SERVER_CONFIG or config_from_env())
    print(f"[api] http://{host}:{httpd.server_address[1]}  版本 {VERSION_TAG}", flush=True)
    return httpd


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="智律契灵 v2 HTTP 接口")
    parser.add_argument("--host", default=None, help="默认取 ZLQL_HOST/HOST，PaaS 环境自动 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="默认取 ZLQL_PORT/PORT，否则 8610")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    httpd = serve(args.host, args.port, verbose=args.verbose)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[api] 收到中断，退出", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
