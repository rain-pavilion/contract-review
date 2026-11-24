"""Authenticated boundary for the two business libraries.

Legacy global workspaces are administrator-only. Ordinary users only access
owner-aware repositories here; no client supplied path reaches the parser.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import date
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import quote

COOKIE = "zlql_session"
LOG = logging.getLogger(__name__)
_WORKER_LOCK = threading.Lock()


def library_paths(config=None):
    from .config import WORKSPACE
    root = Path(os.environ.get("ZLQL_LIBRARY_ROOT") or WORKSPACE / "data" / "libraries").resolve()
    contracts = Path(getattr(config, "contract_library_path", "") or
        os.environ.get("ZLQL_CONTRACT_LIBRARY_PATH") or root / "contracts.sqlite").resolve()
    laws = Path(getattr(config, "law_library_path", "") or
        os.environ.get("ZLQL_LAW_LIBRARY_PATH") or root / "laws.sqlite").resolve()
    if contracts == laws:
        raise ValueError("合同库与法条库必须使用不同文件")
    return contracts, laws


def libraries(config=None):
    from .contract_library import ContractLibrary
    from .law_library import LawLibrary
    contracts, laws = library_paths(config)
    return ContractLibrary(contracts, users_path=user_library_path(config)), LawLibrary(laws)


def user_library_path(config=None):
    contracts, laws = library_paths(config)
    users = Path(getattr(config, "user_library_path", "") or
        os.environ.get("ZLQL_USER_LIBRARY_PATH") or contracts.parent / "users.sqlite").resolve()
    if users in (contracts, laws):
        raise ValueError("用户库、合同库与法条库必须使用不同文件")
    return users


def _contracts(config=None):
    from .contract_library import ContractLibrary
    return ContractLibrary(library_paths(config)[0], users_path=user_library_path(config))


def _config():
    from . import api
    from .config import config_from_env
    return api._SERVER_CONFIG or config_from_env()


def _token(handler):
    try:
        cookie = SimpleCookie(handler.headers.get("Cookie", ""))
        return cookie[COOKIE].value if COOKIE in cookie else ""
    except Exception:
        return ""


def _same_origin(handler):
    origin = handler.headers.get("Origin", "")
    if handler.headers.get("Sec-Fetch-Site", "") == "cross-site":
        return False
    if not origin:
        return True
    host = handler.headers.get("Host", "")
    # A reverse proxy must preserve Host. Never trust client X-Forwarded-*.
    return origin in ("http://" + host, "https://" + host)


def _user(actor):
    return {k: actor[k] for k in ("user_id", "username", "role", "active") if k in actor}


def _public_data(value):
    """Never expose repository paths or trusted worker context over HTTP."""
    if isinstance(value, dict):
        return {k: _public_data(v) for k, v in value.items()
                if k not in ("path", "raw_path", "source_root", "_session_hash", "actor")}
    if isinstance(value, list):
        return [_public_data(v) for v in value]
    return value


def _body(handler):
    length = int(handler.headers.get("Content-Length") or 0)
    if length < 0 or length > 42 * 1024 * 1024:
        raise HttpError("上传内容超过40MB限制", 413)
    raw = handler._read_body()
    ct = handler.headers.get("Content-Type", "").lower()
    if "multipart/form-data" in ct:
        from .api import parse_multipart
        fields, files = parse_multipart(raw, handler.headers.get("Content-Type", ""))
        return fields, files
    if not raw:
        return {}, {}
    if "application/json" not in ct:
        raise HttpError("请求须使用JSON或文件表单", 415)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HttpError("JSON格式不正确", 400)
    if not isinstance(data, dict):
        raise HttpError("请求内容须为对象", 400)
    return data, {}


class HttpError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _admin(actor):
    if actor.get("role") != "admin":
        raise HttpError("只有管理员可以执行此操作", 403)


def _integer(query, key, default, maximum=100):
    try:
        value = int((query.get(key) or [default])[0])
    except (ValueError, TypeError):
        raise HttpError("分页参数不正确")
    if value < 0 or value > maximum:
        raise HttpError("分页参数超出范围")
    return value


def _query(parsed):
    from urllib.parse import parse_qs
    return parse_qs(parsed.query)


def _binary(handler, content, filename, content_type=None):
    handler._send(200, content, content_type or "application/octet-stream", {
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe=""),
        "Content-Security-Policy": "sandbox; default-src 'none'",
    })


def dispatch_request(handler, method, parsed):
    """Return True when handled; False allows an authorized legacy route."""
    path = parsed.path or "/"
    new_route = path.startswith("/api/library/") or path.startswith("/api/auth/")
    try:
        if method == "OPTIONS":
            handler._send(204, b"", "text/plain")
            return True
        if path in ("/", "/index.html", "/library", "/library/"):
            handler._send(200, Path(__file__).with_name("library.html").read_bytes(),
                "text/html; charset=utf-8", {"Content-Security-Policy":
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"})
            return True
        if path == "/favicon.ico":
            handler._send(204, b"", "image/x-icon")
            return True
        if path == "/api/health" and method == "GET":
            # Retain launcher's health handshake, stripping sensitive paths/keys.
            from .version import VERSION_TAG
            cfg = _config()
            from urllib.parse import urlsplit
            endpoint = urlsplit(str(cfg.llm_endpoint or ""))
            local_endpoint = (f"http://127.0.0.1:{endpoint.port}{endpoint.path}" if endpoint.hostname in
                ("127.0.0.1", "localhost") and endpoint.port else "")
            handler._json({"status": "ok", "tag": VERSION_TAG, "multi_agent": bool(cfg.enable_multi_agent),
                "authentication_required": True, "llm": {"enabled": bool(cfg.enable_llm),
                "model": cfg.llm_model, "endpoint": local_endpoint}})
            return True
        if method != "GET" and not _same_origin(handler):
            raise HttpError("拒绝跨站请求", 403)
        store = _contracts(_config())
        if path == "/api/auth/status" and method == "GET":
            authenticated = False
            if _token(handler):
                try:
                    store.authenticate(_token(handler))
                    authenticated = True
                except Exception as exc:
                    if getattr(exc, "status", None) != 401:
                        raise
            initialized = store.initialized()
            handler._json({"initialized": initialized, "authenticated": authenticated,
                "registration_enabled": initialized and _registration_open()})
            return True
        if path == "/api/auth/register" and method == "POST":
            if not _registration_open():
                raise HttpError("当前未开放注册，请联系管理员创建账号", 403)
            if not store.initialized():
                raise HttpError("请先在服务器初始化管理员", 409)
            _check_register_rate(handler)
            if int(handler.headers.get("Content-Length") or 0) > 8192:
                raise HttpError("注册表单过大", 413)
            if "application/json" not in handler.headers.get("Content-Type", "").lower():
                raise HttpError("注册请求须使用JSON", 415)
            body, _ = _body(handler)
            if set(body) - {"username", "password", "password_confirm"}:
                raise HttpError("注册不能指定角色、权限或其他账户属性", 403)
            username, password, confirm = (body.get(k) for k in ("username", "password", "password_confirm"))
            if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff.\-]{2,64}", username.strip()):
                raise HttpError("用户名须为2至64位中文、字母、数字、下划线、点或短横线")
            if not isinstance(password, str) or not 12 <= len(password) <= 512:
                raise HttpError("密码须为12至512位")
            if not isinstance(confirm, str) or password != confirm:
                raise HttpError("两次输入的密码不一致")
            user = store.register(username.strip(), password)
            handler._json({"registered": True, "username": user["username"], "message": "注册成功，请登录"}, 201)
            return True
        if path == "/api/auth/login" and method == "POST":
            _check_login_rate(handler)
            body, _ = _body(handler)
            try:
                session = store.login(str(body.get("username") or ""), str(body.get("password") or ""))
            except Exception:
                _login_failure(handler)
                raise
            secure = "; Secure" if os.environ.get("ZLQL_COOKIE_SECURE", "").lower() in ("1", "true") else ""
            handler._send(200, json.dumps({"user": _user(session["user"]),
                "csrf_token": session["csrf_token"]}).encode(), "application/json; charset=utf-8", {
                "Set-Cookie": f"{COOKIE}={session['token']}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400" + secure})
            return True
        token = _token(handler)
        if not token:
            raise HttpError("请先登录", 401)
        actor = store.authenticate(token)
        if method not in ("GET", "HEAD"):
            csrf = handler.headers.get("X-CSRF-Token", "")
            if not csrf or not store.validate_csrf(token, csrf):
                raise HttpError("会话校验失败，请刷新页面后重试", 403)
        if path == "/api/auth/me" and method == "GET":
            handler._json({"user": _user(actor), "csrf_token": actor["csrf_token"]})
            return True
        if path == "/api/auth/logout" and method == "POST":
            store.logout(token)
            handler._send(200, b'{"ok":true}', "application/json", {
                "Set-Cookie": f"{COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"})
            return True
        if new_route:
            body, files = _body(handler) if method == "POST" else ({}, {})
            result = _handle_library(handler, method, parsed, body, files, store, actor)
            if result is not None:
                payload, status = result if isinstance(result, tuple) else (result, 200)
                handler._json(_public_data(payload), status)
            return True
        # Deny by default: every old endpoint can expose global data or mutate
        # server resources. Administrator-only keeps historical archives usable.
        _admin(actor)
        handler.library_actor = actor
        if path in ("/workspace", "/workspace/", "/workspace.html", "/annotations", "/annotation.html"):
            from . import api
            page = (api.UI_FILE.read_text(encoding="utf-8") if path.startswith("/workspace") else
                Path(__file__).with_name("annotation.html").read_text(encoding="utf-8"))
            bridge = """<script>(()=>{const original=window.fetch.bind(window);const session=original('/api/auth/me',
              {credentials:'same-origin'}).then(r=>r.json());window.fetch=async(input,init={})=>{
              const request=new Request(input,init);if(new URL(request.url).origin===location.origin&&
              !['GET','HEAD','OPTIONS'].includes(request.method)){const auth=await session;
              const headers=new Headers(request.headers);headers.set('X-CSRF-Token',auth.csrf_token||'');
              return original(new Request(request,{headers,credentials:'same-origin'}));}
              return original(request);};})();</script>"""
            handler._send(200, page.replace("</head>", bridge + "</head>", 1).encode(), "text/html; charset=utf-8")
            return True
        return False
    except (HttpError, ValueError, PermissionError) as exc:
        handler._json({"error": str(exc)}, getattr(exc, "status", 400))
        return True
    except Exception as exc:
        status = getattr(exc, "status", None)
        if isinstance(status, int) and 400 <= status <= 599:
            handler._json({"error": str(exc)}, status)
        else:
            LOG.exception("Library request failed: %s", path)
            handler._json({"error": "服务暂时不可用，请稍后重试"}, 500)
        return True


_LOGIN_LOCK = threading.Lock()
_LOGIN_ATTEMPTS = {}
_REGISTER_LOCK = threading.Lock()
_REGISTER_ATTEMPTS = {}


def _registration_open():
    return os.environ.get("ZLQL_PUBLIC_REGISTRATION", "1").strip().lower() in ("1", "true", "on", "yes")


def _check_register_rate(handler):
    now = time.monotonic()
    key = handler.client_address[0]
    with _REGISTER_LOCK:
        for address in list(_REGISTER_ATTEMPTS):
            recent = [t for t in _REGISTER_ATTEMPTS[address] if t > now - 900]
            if recent:
                _REGISTER_ATTEMPTS[address] = recent
            else:
                _REGISTER_ATTEMPTS.pop(address, None)
        recent = _REGISTER_ATTEMPTS.get(key, [])
        if len(recent) >= 10 or (key not in _REGISTER_ATTEMPTS and len(_REGISTER_ATTEMPTS) >= 1000):
            raise HttpError("注册尝试过多，请十五分钟后重试", 429)
        _REGISTER_ATTEMPTS.setdefault(key, []).append(now)


def _check_login_rate(handler):
    now = time.monotonic()
    with _LOGIN_LOCK:
        key = handler.client_address[0]
        _LOGIN_ATTEMPTS[key] = [t for t in _LOGIN_ATTEMPTS.get(key, []) if t > now - 300]
        if len(_LOGIN_ATTEMPTS[key]) >= 20:
            raise HttpError("登录尝试过多，请五分钟后重试", 429)
        if len(_LOGIN_ATTEMPTS) > 1000:
            for address in list(_LOGIN_ATTEMPTS):
                if not _LOGIN_ATTEMPTS[address] or _LOGIN_ATTEMPTS[address][-1] < now - 300:
                    _LOGIN_ATTEMPTS.pop(address, None)


def _login_failure(handler):
    with _LOGIN_LOCK:
        _LOGIN_ATTEMPTS.setdefault(handler.client_address[0], []).append(time.monotonic())


def _handle_library(handler, method, parsed, body, files, store, actor):
    from .law_library import LawLibrary
    path = parsed.path
    q = _query(parsed)
    arg = lambda k, default="": (q.get(k) or [default])[0]
    limit = _integer(q, "limit", 50)
    offset = _integer(q, "offset", 0, 1000000)
    laws = LawLibrary(library_paths(_config())[1])
    is_admin = actor["role"] == "admin"
    if path == "/api/library/contracts":
        if method == "GET":
            return store.list_contracts(actor, query=arg("q"), limit=limit, offset=offset)
        if method == "POST":
            if len(files) > 1:
                raise HttpError("每次上传一份合同")
            if files:
                filename, content = next(iter(files.values()))
            else:
                filename = str(body.get("filename") or "合同.txt")
                if not isinstance(body.get("text"), str):
                    raise HttpError("请选择文件或提供合同文本")
                content = body["text"].encode("utf-8")
            metadata = {k: str(body[k]) for k in ("title", "domain", "region") if k in body}
            return store.upload(actor, filename, content, metadata=metadata), 201
    match = re.fullmatch(r"/api/library/contracts/([0-9a-f-]+)(?:/(download|update|delete|review))?", path)
    if match:
        cid, action = match.groups()
        if not action and method == "GET":
            return store.get_contract(actor, cid)
        if action == "download" and method == "GET":
            version = store.get_version(actor, cid, arg("version_id") or None)
            _binary(handler, version["content"], version["filename"])
            return None
        if action == "update" and method == "POST":
            _admin(actor)
            if len(files) > 1:
                raise HttpError("每次只能替换一个文件")
            filename, content = next(iter(files.values())) if files else (None, None)
            if "text" in body:
                if not isinstance(body["text"], str):
                    raise HttpError("修订正文必须为文本")
                content = body["text"].encode("utf-8")
                filename = Path(store.get_contract(actor, cid).get("filename", "合同")).stem + ".txt"
            return store.update_contract(actor, cid, metadata={k: str(body[k]) for k in
                ("title", "domain", "region") if k in body}, filename=filename, content=content,
                note=str(body.get("note") or ""), expected_version_id=body.get("expected_version_id") or None)
        if action == "delete" and method == "POST":
            _admin(actor)
            return store.delete_contract(actor, cid)
        if action == "review" and method == "POST":
            if not store.initialized():
                raise HttpError("请先初始化账户", 409)
            store.get_version(actor, cid, body.get("version_id") or None)
            from .review_context import normalize_review_options, REVIEW_OPTION_KEYS
            review_options = {key: body[key] for key in REVIEW_OPTION_KEYS if key in body}
            for key in REVIEW_OPTION_KEYS:
                if key in q:
                    review_options[key] = q[key][0]
            options = {k: body[k] for k in ("domain", "region", "as_of", "llm") if k in body}
            options.update(normalize_review_options(review_options))
            if options.get("as_of"):
                date.fromisoformat(str(options["as_of"]))
            if len(str(options.get("region") or "")) > 128:
                raise HttpError("地区名称过长")
            if len([t for t in store.list_tasks(actor, limit=100)["items"]
                    if t["status"] in ("queued", "running")]) >= 10:
                raise HttpError("待处理任务已达10项，请等待完成后再提交", 429)
            if "llm" in options and not isinstance(options["llm"], bool):
                raise HttpError("llm须为布尔值")
            scope = {"domain": str(options.get("domain") or ""),
                "region": str(options.get("region") or "全国"), "as_of": options.get("as_of") or None}
            snapshot = laws.create_snapshot(**scope, require_articles=True)
            options["law_snapshot"] = snapshot
            result = store.enqueue_review(actor, cid, body.get("version_id") or None, options=options)
            ensure_library_worker(handler.server, _config())
            return result, 202
    if path == "/api/library/reports" and method == "GET":
        return store.list_reports(actor, arg("contract_id") or None, limit=limit, offset=offset)
    match = re.fullmatch(r"/api/library/reports/([0-9a-f-]+)(?:/(feedback))?", path)
    if match:
        report_id, action = match.groups()
        if method == "GET" and not action:
            return store.get_report(actor, report_id)
        if method == "POST" and action == "feedback":
            _admin(actor)
            # Feedback is an append-only review record, never a rewritten model report.
            store.get_report(actor, report_id)
            return _feedback(store, actor, report_id, body)
    if path == "/api/library/tasks" and method == "GET":
        return store.list_tasks(actor, limit=limit)
    match = re.fullmatch(r"/api/library/tasks/([0-9a-f-]+)", path)
    if match and method == "GET":
        return store.get_task(actor, match[1])
    if path == "/api/library/users":
        _admin(actor)
        if method == "GET":
            return store.list_users(actor)
        if method == "POST":
            return store.create_user(actor, str(body.get("username") or ""),
                str(body.get("password") or ""), role=str(body.get("role") or "user")), 201
    match = re.fullmatch(r"/api/library/users/([0-9a-f-]+)", path)
    if match and method == "POST":
        _admin(actor)
        changes = {k: body[k] for k in ("password", "active", "role") if k in body}
        if "active" in changes and not isinstance(changes["active"], bool):
            raise HttpError("active须为布尔值")
        return store.update_user(actor, match[1], **changes)
    if path == "/api/library/events" and method == "GET":
        return store.events(actor, limit=limit)
    if path == "/api/library/laws/domains" and method == "GET":
        return {"items": laws.domains()}
    if path == "/api/library/laws/stats" and method == "GET":
        return laws.stats(include_unpublished=is_admin)
    if path == "/api/library/laws/search" and method == "GET":
        return laws.search(arg("q"), domain=arg("domain"), region=arg("region", "全国"),
            as_of=arg("as_of") or None, limit=min(limit, 50))
    if path == "/api/library/laws" and method == "GET":
        return laws.list_documents(query=arg("q"), domain=arg("domain"),
            include_unpublished=is_admin, limit=limit, offset=offset)
    match = re.fullmatch(r"/api/library/laws/([0-9a-z_-]+)", path)
    if match and method == "GET":
        return laws.get_document(match[1], include_unpublished=is_admin)
    match = re.fullmatch(r"/api/library/law-versions/([0-9a-z_-]+)(?:/(source|metadata|publish))?", path)
    if match:
        vid, action = match.groups()
        if not action and method == "GET":
            return laws.get_version(vid, include_unpublished=is_admin)
        if action == "source" and method == "GET":
            source = laws.raw_source(vid, include_unpublished=is_admin)
            _binary(handler, source["content"], source["filename"], source["content_type"])
            return None
        if action in ("metadata", "publish") and method == "POST":
            _admin(actor)
            if action == "metadata":
                return laws.set_version_metadata(actor, vid, body)
            return laws.publish_version(actor, vid, note=str(body.get("note") or ""))
    if path == "/api/library/law-sources" and method == "GET":
        _admin(actor)
        from .law_crawler import source_catalog
        return {"items": source_catalog()}
    if path == "/api/library/law-crawls":
        _admin(actor)
        if method == "GET":
            return laws.list_crawl_runs(limit=limit)
        if method == "POST":
            from .law_crawler import crawl_sources
            ids = body.get("source_ids")
            if ids is not None and (not isinstance(ids, list) or not all(isinstance(i, str) for i in ids)):
                raise HttpError("采集来源须为来源ID列表")
            maximum = int(body.get("max_documents", 5))
            if not 1 <= maximum <= 50:
                raise HttpError("单次采集须为1到50份")
            job_id = _start_library_job(handler.server, actor, lambda: crawl_sources(laws, actor,
                source_ids=ids, max_documents=maximum))
            return {"task_id": job_id, "job_id": job_id, "status": "running"}, 202
    match = re.fullmatch(r"/api/library/jobs/([0-9a-f]+)", path)
    if match and method == "GET":
        _admin(actor)
        job = getattr(handler.server, "library_jobs", {}).get(match[1])
        if not job:
            raise HttpError("任务不存在", 404)
        return dict(job)
    raise HttpError("接口不存在", 404)


def _feedback(store, actor, report_id, body):
    status = body.get("status")
    if status not in ("confirmed", "dismissed", "pending"):
        raise HttpError("复核状态不正确")
    rid = str(body.get("risk_id") or "")
    report = store.get_report(actor, report_id)
    if rid not in {str(r.get("risk_id") or r.get("id") or "") for r in report["payload"].get("risks", [])}:
        raise HttpError("报告中没有该风险", 404)
    return store.save_feedback(actor, report_id, rid, status, str(body.get("note") or ""))


def _start_library_job(server, actor, callback):
    with _WORKER_LOCK:
        jobs = getattr(server, "library_jobs", None)
        if jobs is None:
            server.library_jobs = jobs = {}
        if any(j["status"] == "running" for j in jobs.values()):
            raise HttpError("已有采集任务在执行，请稍后重试", 409)
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"job_id": job_id, "task_id": job_id, "status": "running",
            "created_by": actor["user_id"], "created_at": time.time(), "result": None, "error": None}

    def run():
        try:
            jobs[job_id].update(status="done", result=callback())
        except Exception:
            LOG.exception("Law crawl failed")
            jobs[job_id].update(status="failed", error="采集失败，详情见采集记录")
        finally:
            jobs[job_id]["finished_at"] = time.time()

    threading.Thread(target=run, name="law-crawl-" + job_id[:8], daemon=True).start()
    return job_id


class LibraryWorker:
    def __init__(self, server, config):
        self.server = server
        self.config = config
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="contract-library-worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def run(self):
        from .contract_library import ContractLibrary
        store = ContractLibrary(library_paths(self.config)[0], users_path=user_library_path(self.config))
        while not self.stop_event.is_set():
            try:
                worker_id = uuid.uuid4().hex
                task = store.claim_task(worker_id, lease_seconds=300)
                if not task:
                    self.stop_event.wait(1)
                    continue
                self._execute(store, task, worker_id)
            except Exception:
                LOG.exception("Contract worker error")
                self.stop_event.wait(2)

    def _execute(self, store, task, worker_id):
        heartbeat_stop = threading.Event()

        def heartbeat():
            while not heartbeat_stop.wait(30):
                if not store.renew_task(task["task_id"], worker_id, lease_seconds=300):
                    break

        thread = threading.Thread(target=heartbeat, name="contract-lease", daemon=True)
        thread.start()
        try:
            payload = run_library_review(task, self.config)
            store.finish_task(task["task_id"], worker_id, payload=payload)
        except Exception:
            LOG.exception("Contract review failed")
            try:
                store.finish_task(task["task_id"], worker_id,
                    error="审查未完成，请管理员检查模型配置、文档格式及运行日志后重试")
            except Exception:
                LOG.exception("Could not finish review task")
        finally:
            heartbeat_stop.set()
            thread.join(2)


def ensure_library_worker(server, config):
    with _WORKER_LOCK:
        if getattr(server, "library_worker", None) is None:
            server.library_worker = LibraryWorker(server, config)
            server.library_worker.start()
    return server.library_worker


def run_library_review(task, config):
    from . import api
    from .pipeline import prepare
    from .review_context import apply_review_options
    options = task.get("options") or {}
    if isinstance(options, str):
        options = json.loads(options)
    snapshot = options.get("law_snapshot") or {}
    path = Path(snapshot.get("path", ""))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != snapshot.get("sha256"):
        raise ValueError("法条快照缺失或完整性校验失败")
    # No shared private subject evidence enters an ordinary user's review.
    actor = task["actor"]
    evidence = library_paths(config)[0].parent / "private_evidence" / task["owner"] / "facts.json"
    cfg = replace(apply_review_options(config, options), statutes_path=str(path),
        subject_facts_path=str(evidence), enable_llm=bool(options.get("llm", config.enable_llm)))
    if cfg.enable_llm:
        api._ensure_llm_request_configured("1", cfg)
    version = task["version"]
    started = time.perf_counter()
    document = prepare(version["filename"], version["content"], config=cfg)
    if not document.full_text.strip():
        raise ValueError("合同没有可读取的正文")
    payload = api._review_document(document, cfg, uuid.uuid4().hex, started,
        time.perf_counter() - started, save=False)
    payload["law_snapshot"] = {k: v for k, v in snapshot.items() if k != "path"}
    payload["contract_version_id"] = version["version_id"]
    payload["library_contract_id"] = version["contract_id"]
    return payload
