"""Versioned legal texts. No application imports, legacy fallback, or import-time I/O."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

DOMAINS = {"general": "通用", "construction": "建筑工程", "finance": "金融", "real_estate": "房地产", "procurement": "政府采购", "trade": "贸易", "corporate": "公司商事", "labor": "劳动用工", "ip": "知识产权", "data": "数据与隐私"}
VALIDITIES = {"unknown", "effective", "abolished", "expired", "not_yet_effective"}
PROFILE = "published_raw_bm25_v1"


class LawError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _admin(actor):
    if not isinstance(actor, dict) or actor.get("role") != "admin" or not str(actor.get("user_id") or "").strip():
        raise LawError("需要已认证管理员", 403)


def _date(value, field):
    if value in (None, "", "unknown"):
        return ""
    try:
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError()
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise LawError(f"{field} 必须为有效的 YYYY-MM-DD") from exc


def _metadata(document, actor=None):
    out = dict(document)
    title = str(out.get("title") or "").strip()
    if not title or len(title) > 500:
        raise LawError("法规标题为空或过长")
    out["title"] = title
    domains = out.get("domains") or ["general"]
    if not isinstance(domains, list) or any(x not in DOMAINS for x in domains):
        raise LawError("未知法律领域")
    out["domains"] = list(dict.fromkeys(domains))
    validity = out.get("validity") or "unknown"
    if validity not in VALIDITIES:
        raise LawError("未知效力状态")
    out["validity"] = validity
    for name in ("published_at", "effective_from", "effective_to"):
        out[name] = _date(out.get(name), name)
    if out["effective_from"] and out["effective_to"] and out["effective_to"] < out["effective_from"]:
        raise LawError("失效日早于施行日")
    for key in ("authority_level", "issuing_authority", "document_number", "official_status"):
        out[key] = str(out.get(key) or "unknown")
    out["extraction_quality"] = out.get("extraction_quality") or "unknown"
    out["region"] = str(out.get("region") or "全国").strip()
    if not out["region"] or len(out["region"]) > 128:
        raise LawError("地域无效")
    if "verification" in out:
        review = out["verification"]
        if not isinstance(review, dict):
            raise LawError("核验信息必须为对象")
        if actor is not None:
            review = {"verified": review.get("verified") is True,
                      "evidence": str(review.get("evidence") or "").strip()[:4000],
                      "verified_by": str(actor["user_id"]), "verified_at": _now()}
            if review["verified"] and len(review["evidence"]) < 4:
                raise LawError("请填写可核对的效力与日期依据")
            out["verification"] = review
    return out


def _articles(articles):
    if not isinstance(articles, list) or not articles:
        raise LawError("缺少有效法条")
    result = []
    for number, record in enumerate(articles, 1):
        if not isinstance(record, dict):
            raise LawError("法条结构错误")
        item = dict(record)
        text = str(item.get("content") or "").strip()
        if len(text) < 12 or len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text)) < 8 or "<html" in text.lower():
            raise LawError("空白或无效法条内容")
        if re.search(r"验证码|访问被拒绝|Access Denied|安全验证|页面不存在", text, re.I):
            raise LawError("网页错误或验证码不能入库")
        item["content"] = text
        item["article_no"] = str(item.get("article_no") or f"第{number}项")
        item.pop("article_uid", None)
        result.append(item)
    return result


def _verified(metadata):
    check = metadata.get("verification") or {}
    return (isinstance(check, dict) and check.get("verified") is True and bool(check.get("verified_by"))
            and bool(check.get("verified_at")) and bool(str(check.get("evidence") or "").strip()))


class LawLibrary:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.path.with_name(self.path.stem + "_raw")
        self.snapshot_dir = self.path.with_name(self.path.stem + "_snapshots")
        for directory in (self.raw_dir, self.snapshot_dir):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,canonical_key TEXT UNIQUE NOT NULL,title TEXT NOT NULL,created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id),fingerprint TEXT UNIQUE NOT NULL,status TEXT NOT NULL,metadata TEXT NOT NULL,source_sha256 TEXT NOT NULL,source_url TEXT NOT NULL,created_at TEXT NOT NULL,published_at TEXT,publish_note TEXT);
              CREATE TABLE IF NOT EXISTS articles(version_id TEXT NOT NULL REFERENCES versions(id),ordinal INTEGER NOT NULL,article_uid TEXT UNIQUE NOT NULL,record TEXT NOT NULL,PRIMARY KEY(version_id,ordinal));
              CREATE TABLE IF NOT EXISTS sources(version_id TEXT NOT NULL REFERENCES versions(id),url TEXT NOT NULL,sha256 TEXT NOT NULL,path TEXT NOT NULL,content_type TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(version_id,url,sha256));
              CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY,path TEXT NOT NULL,sha256 TEXT NOT NULL,article_count INTEGER NOT NULL,scope TEXT NOT NULL,created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS crawl_runs(id TEXT PRIMARY KEY,status TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS versions_document ON versions(document_id,status);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def domains(self):
        return [{"id": key, "name": name} for key, name in DOMAINS.items()]

    def _write_raw(self, raw):
        digest = _sha(raw)
        target = self.raw_dir / (digest + ".bin")
        try:
            with target.open("xb") as handle:
                handle.write(raw)
        except FileExistsError:
            if _sha(target.read_bytes()) != digest:
                raise LawError("已有原件校验失败", 409)
        return digest, str(target)

    def ingest_document(self, actor, document, articles, raw, source_url, *, content_type="text/html"):
        _admin(actor)
        metadata = _metadata(document, actor)
        articles = _articles(articles)
        if not isinstance(raw, bytes) or not raw or len(raw) > 20 * 1024 * 1024:
            raise LawError("原件为空或超过20MiB")
        if str(source_url).startswith("legacy:"):
            if metadata.get("source_id") != "legacy":
                raise LawError("迁移来源标记缺失")
        else:
            # Only validate the authority here; crawler validates DNS on each hop.
            from .law_crawler import _validate_url
            _validate_url(source_url, resolve=False)
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", metadata["title"]))
        canonical = str(metadata.get("canonical_id") or _json([normalized, metadata["region"], metadata["issuing_authority"]]))
        document_id = "law_" + _sha(canonical.encode())[:24]
        identity = {"title": normalized, "published_at": metadata["published_at"], "effective_from": metadata["effective_from"],
                    "effective_to": metadata["effective_to"], "document_number": metadata["document_number"],
                    "version": metadata.get("version", metadata.get("version_label", "")),
                    "articles": [(x["article_no"], x["content"]) for x in articles]}
        fingerprint = _sha((document_id + _json(identity)).encode())
        digest, raw_path = self._write_raw(raw)
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)", (document_id, canonical, metadata["title"], _now()))
            existing = db.execute("SELECT id,document_id,status FROM versions WHERE fingerprint=?", (fingerprint,)).fetchone()
            duplicate = existing is not None
            if duplicate:
                version_id = existing["id"]
                document_id = existing["document_id"]
                status = existing["status"]
            else:
                version_id = "lv_" + uuid.uuid4().hex
                status = "draft"
                db.execute("INSERT INTO versions(id,document_id,fingerprint,status,metadata,source_sha256,source_url,created_at) VALUES(?,?,?,?,?,?,?,?)", (version_id, document_id, fingerprint, status, _json(metadata), digest, str(source_url), _now()))
                for ordinal, record in enumerate(articles):
                    uid = version_id + ":" + str(ordinal + 1)
                    db.execute("INSERT INTO articles VALUES(?,?,?,?)", (version_id, ordinal, uid, _json(record)))
            db.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?,?,?,?)", (version_id, str(source_url), digest, raw_path, str(content_type), _now()))
            count = db.execute("SELECT count(*) FROM articles WHERE version_id=?", (version_id,)).fetchone()[0]
        return {"document_id": document_id, "version_id": version_id, "status": status, "article_count": count, "duplicate": duplicate}

    def get_version(self, version_id, *, include_unpublished=False):
        with self._db() as db:
            version = db.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
            if version is None or (version["status"] != "published" and not include_unpublished):
                raise LawError("法条版本不存在", 404)
            metadata = json.loads(version["metadata"])
            articles = []
            for record in db.execute("SELECT * FROM articles WHERE version_id=? ORDER BY ordinal", (version_id,)):
                articles.append({**json.loads(record["record"]), "article_uid": record["article_uid"]})
            sources = [dict(row) for row in db.execute("SELECT * FROM sources WHERE version_id=? ORDER BY created_at,url", (version_id,))]
        return {**metadata, "version_id": version["id"], "id": version["id"], "document_id": version["document_id"], "status": version["status"],
                "source_url": version["source_url"], "source_sha256": version["source_sha256"], "created_at": version["created_at"],
                "publication_at": version["published_at"], "metadata": metadata, "articles": articles, "article_count": len(articles), "sources": sources}

    def get_document(self, document_id, *, include_unpublished=False):
        with self._db() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            ids = [x[0] for x in db.execute("SELECT id FROM versions WHERE document_id=?" + ("" if include_unpublished else " AND status='published'") + " ORDER BY created_at DESC,id DESC", (document_id,))]
        if row is None or not ids:
            raise LawError("法规不存在", 404)
        versions = [self.get_version(v, include_unpublished=include_unpublished) for v in ids]
        # Display the latest published version when available; drafts are separate history.
        current = next((v for v in versions if v["status"] == "published"), versions[0])
        summary = {key: current[key] for key in ("version_id", "id", "status", "validity", "article_count", "source_url")}
        return {"document_id": document_id, "title": current["title"], "domains": current["domains"], "current_version": summary,
                "versions_count": len(versions), "versions": versions}

    def list_documents(self, *, query="", domain="", include_unpublished=False, limit=50, offset=0):
        if domain and domain not in DOMAINS:
            raise LawError("未知法律领域")
        with self._db() as db:
            ids = [row[0] for row in db.execute("SELECT DISTINCT document_id FROM versions" + ("" if include_unpublished else " WHERE status='published'") + " ORDER BY document_id")]
        items = []
        for document_id in ids:
            document = self.get_document(document_id, include_unpublished=include_unpublished)
            if query and str(query).casefold() not in document["title"].casefold():
                continue
            if domain and domain not in document["domains"]:
                continue
            document.pop("versions")
            items.append(document)
        return {"items": items[max(0, int(offset)):max(0, int(offset)) + max(0, min(500, int(limit)))], "total": len(items)}

    def stats(self, include_unpublished=False):
        versions = self._all_versions(include_unpublished)
        by_domain = {key: {"documents": 0, "versions": 0, "articles": 0} for key in DOMAINS}
        for domain in DOMAINS:
            rows = [v for v in versions if domain in v["domains"]]
            by_domain[domain] = {"documents": len({v["document_id"] for v in rows}), "versions": len(rows), "articles": sum(v["article_count"] for v in rows)}
        return {"documents": len({v["document_id"] for v in versions}), "versions": len(versions), "articles": sum(v["article_count"] for v in versions),
                "published_versions": sum(v["status"] == "published" for v in versions), "draft_versions": sum(v["status"] == "draft" for v in versions), "by_domain": by_domain}

    def _all_versions(self, include_unpublished=False):
        with self._db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM versions" + ("" if include_unpublished else " WHERE status='published'") + " ORDER BY created_at,id")]
        return [self.get_version(v, include_unpublished=include_unpublished) for v in ids]

    def raw_source(self, version_id, *, include_unpublished=False):
        version = self.get_version(version_id, include_unpublished=include_unpublished)
        source = next((s for s in version["sources"] if s["sha256"] == version["source_sha256"] and s["url"] == version["source_url"]), None)
        if not source:
            raise LawError("原件缺失", 409)
        path = Path(source["path"])
        if path.resolve().parent != self.raw_dir.resolve():
            raise LawError("原件路径不合法", 409)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise LawError("原件缺失", 409) from exc
        if _sha(content) != version["source_sha256"]:
            raise LawError("原件哈希校验失败", 409)
        return {"content": content, "content_type": source["content_type"], "filename": path.name, "sha256": source["sha256"]}

    def publish_version(self, actor, version_id, *, note=""):
        _admin(actor)
        version = self.get_version(version_id, include_unpublished=True)
        self.raw_source(version_id, include_unpublished=True)
        _articles(version["articles"])
        metadata = version["metadata"]
        if metadata["validity"] == "effective":
            if not (metadata["effective_from"] and _verified(metadata)):
                raise LawError("现行有效版本须由管理员核对效力、施行日期并填写依据")
        with self._db() as db:
            # Check the validated row after obtaining the write reservation. Other writers
            # cannot change its metadata while the candidate snapshot is being prepared.
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT metadata,status,source_sha256,source_url FROM versions WHERE id=?", (version_id,)).fetchone()
            if (current is None or current["metadata"] != _json(metadata) or current["status"] != version["status"]
                    or current["source_sha256"] != version["source_sha256"] or current["source_url"] != version["source_url"]):
                raise LawError("版本在核验期间发生变化，请重新核验后发布", 409)
            records, scope = self._eligible("", "全国", None, _pending_version={**version, "status": "published"})
            # Failure here rolls back the transaction and leaves the version a draft.
            # A filesystem artifact from a failed commit remains private and unreferenced.
            snapshot = self._store_snapshot(records, scope, db=db)
            changed = db.execute("UPDATE versions SET status='published',published_at=COALESCE(published_at,?),publish_note=CASE WHEN status='draft' THEN ? ELSE publish_note END WHERE id=? AND metadata=? AND status=? AND source_sha256=?", (_now(), str(note)[:4000], version_id, _json(metadata), version["status"], version["source_sha256"])).rowcount
            if changed != 1:
                raise LawError("版本在核验期间发生变化", 409)
        warning = "" if metadata["validity"] == "effective" else "效力未确认为现行有效；仅供展示，不进入默认检索。"
        return {"version_id": version_id, "status": "published", "snapshot": snapshot, "warning": warning}

    def set_version_metadata(self, actor, version_id, changes):
        _admin(actor)
        version = self.get_version(version_id, include_unpublished=True)
        allowed = {"domains", "region", "validity", "effective_from", "effective_to", "published_at", "authority_level", "issuing_authority", "document_number", "official_status", "verification", "extraction_quality"}
        if not isinstance(changes, dict) or set(changes) - allowed:
            raise LawError("不能修改原文或不支持的元数据")
        old = version["metadata"]
        data = {**old, **changes}
        if any(key != "verification" and old.get(key) != value for key, value in changes.items()):
            data.pop("verification", None)
            if "verification" in changes:
                data["verification"] = changes["verification"]
        data = _metadata(data, actor if "verification" in changes else None)
        if data == old:
            return version
        new_id = version_id
        with self._db() as db:
            if version["status"] == "published":
                new_id = "lv_" + uuid.uuid4().hex
                data["supersedes_version_id"] = version_id
                data["legal_version_id"] = old.get("legal_version_id", version_id)
                db.execute("INSERT INTO versions(id,document_id,fingerprint,status,metadata,source_sha256,source_url,created_at) VALUES(?,?,?,?,?,?,?,?)", (new_id, version["document_id"], "metadata:" + uuid.uuid4().hex, "draft", _json(data), version["source_sha256"], version["source_url"], _now()))
                for ordinal, record in enumerate(version["articles"]):
                    record = {k: v for k, v in record.items() if k != "article_uid"}
                    db.execute("INSERT INTO articles VALUES(?,?,?,?)", (new_id, ordinal, new_id + ":" + str(ordinal + 1), _json(record)))
                db.execute("INSERT INTO sources SELECT ?,url,sha256,path,content_type,created_at FROM sources WHERE version_id=?", (new_id, version_id))
            else:
                changed = db.execute("UPDATE versions SET metadata=? WHERE id=? AND status='draft' AND metadata=?", (_json(data), version_id, _json(old))).rowcount
                if changed != 1:
                    raise LawError("版本已被修改或发布，请重新读取", 409)
        return self.get_version(new_id, include_unpublished=True)

    def _eligible(self, domain, region, as_of, *, _pending_version=None):
        if domain and domain not in DOMAINS:
            raise LawError("未知法律领域")
        day = _date(as_of, "as_of") if as_of else date.today().isoformat()
        if not isinstance(region, str) or not region.strip() or len(region) > 128:
            raise LawError("地域无效")
        versions = {version["version_id"]: version for version in self._all_versions()}
        if _pending_version is not None:
            versions[_pending_version["version_id"]] = _pending_version
        # Metadata corrections are editions of one legal version, not competing legal
        # amendments. Use its latest published correction even if a date was moved back.
        editions = {}
        for version in versions.values():
            legal_id = version["metadata"].get("legal_version_id", version["version_id"])
            previous = editions.get(legal_id)
            if previous is None or (version["created_at"], version["version_id"]) > (previous["created_at"], previous["version_id"]):
                editions[legal_id] = version
        latest = {}
        for version in editions.values():
            if version["region"] not in {"全国", region}:
                continue
            # Future revisions must not hide a currently applicable version.
            if version["effective_from"] and version["effective_from"] > day:
                continue
            key = version["document_id"]
            rank = (version["effective_from"], version["created_at"], version["version_id"])
            previous = latest.get(key)
            if previous is None or rank > (previous["effective_from"], previous["created_at"], previous["version_id"]):
                latest[key] = version
        rows = []
        for version in latest.values():
            historical = (bool(as_of) and version["validity"] in {"abolished", "expired"}
                          and bool(version["effective_to"]) and _verified(version["metadata"]))
            if (version["validity"] != "effective" and not historical) or not version["effective_from"]:
                continue
            if not _verified(version["metadata"]):
                continue
            if version["effective_to"] and day >= version["effective_to"]:
                continue
            if domain and domain not in version["domains"]:
                continue
            if version["region"] not in {"全国", region or "全国"}:
                continue
            for article in version["articles"]:
                migration = version["metadata"].get("source_kind") == "derived_migration_excerpt"
                official_hash = str(article.get("original_sha256") or "") if migration else ""
                provenance = {}
                if migration:
                    provenance = {"source_kind": "derived_migration_excerpt", "migration_source_sha256": version["source_sha256"],
                                  "official_original_sha256": official_hash or "unknown", "official_original_verification": "legacy_claim_unverified"}
                rows.append({**article, "law_name": version["title"], "title": version["title"], "document_id": version["document_id"], "version_id": version["version_id"],
                             "source_url": (article.get("source_url") or version["source_url"]) if migration else version["source_url"],
                             "source_sha256": official_hash or version["source_sha256"], "original_sha256": official_hash or version["source_sha256"],
                             "source_id": version["metadata"].get("source_id", "unknown"), "domains": version["domains"], "region": version["region"],
                             "effective_from": version["effective_from"], "effective_to": version["effective_to"], "published_at": version["metadata"]["published_at"],
                             "validity": version["validity"], "status": version["status"], "authority_level": version["authority_level"], "issuing_authority": version["issuing_authority"],
                             "document_number": version["document_number"], "extraction_quality": version["extraction_quality"], "metadata": version["metadata"], "retrieval_profile": PROFILE, **provenance})
        return rows, {"domain": domain, "region": region or "全国", "as_of": day, "published_only": True,
                      "validity": "effective_at_as_of" if as_of else "effective", "historical": bool(as_of), "retrieval_profile": PROFILE}

    def search(self, query, *, domain="", region="全国", as_of=None, limit=10):
        records, scope = self._eligible(domain, region, as_of)
        words = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", str(query).lower())
        terms = list(dict.fromkeys(term for word in words for term in ([word] if len(word) <= 2 or word.isascii() else [word] + [word[i:i + 2] for i in range(len(word) - 1)])))
        scored = []
        for record in records:
            text = (record["law_name"] + " " + record["article_no"] + " " + record["content"]).lower()
            score = sum(text.count(term) * len(term) for term in terms)
            if score:
                scored.append({**record, "score": score})
        scored.sort(key=lambda item: (-item["score"], item["article_uid"]))
        return {"items": scored[:max(0, min(100, int(limit)))], "total": len(scored), "scope": scope}

    def create_snapshot(self, *, domain="", region="全国", as_of=None, require_articles=False):
        records, scope = self._eligible(domain, region, as_of)
        if require_articles and not records:
            raise LawError("所选领域、地域及时点没有已核验的已发布法条", 409)
        return self._store_snapshot(records, scope)

    def _store_snapshot(self, records, scope, *, db=None):
        records.sort(key=lambda item: item["article_uid"])
        identity_raw = ("".join(_json(record) + "\n" for record in records)).encode("utf-8")
        snapshot_id = "ls_" + _sha(_json(scope).encode() + b"\n" + identity_raw)[:32]
        raw = ("".join(_json({**record, "snapshot_id": snapshot_id}) + "\n" for record in records)).encode("utf-8")
        digest = _sha(raw)
        target = self.snapshot_dir / (snapshot_id + ".jsonl")
        # link is an atomic no-clobber publish; a competing writer can only create identical bytes.
        fd, temporary = tempfile.mkstemp(dir=self.snapshot_dir, prefix=".snapshot-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _sha(target.read_bytes()) != digest:
                    raise LawError("不可变快照校验失败", 409)
        finally:
            os.unlink(temporary)
        values = (snapshot_id, str(target), digest, len(records), _json(scope), _now())
        if db is not None:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?)", values)
        else:
            with self._db() as connection:
                connection.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?)", values)
        return {"snapshot_id": snapshot_id, "path": str(target), "sha256": digest, "article_count": len(records), "scope": scope, "retrieval_profile": PROFILE}

    def import_jsonl(self, actor, path, *, source_label="legacy"):
        _admin(actor)
        raw = Path(path).read_bytes()
        if len(raw) > 20 * 1024 * 1024:
            raise LawError("迁移原件超过20MiB")
        archive_sha, _archive_path = self._write_raw(raw)
        groups, group_lines, errors = {}, {}, []
        for number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
            try:
                line = raw_line.decode("utf-8-sig")
            except UnicodeError:
                errors.append({"line": number, "error": "无效UTF-8 JSONL行"})
                continue
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not object")
                title = record.get("law_name") or record.get("title")
                if not title:
                    raise ValueError("missing law title")
                key = (title, record.get("version_id") or record.get("version") or "", record.get("published_at") or "", record.get("effective_from") or "", record.get("effective_to") or "", record.get("region") or "全国", record.get("issuing_authority") or "unknown")
                groups.setdefault(key, []).append(record)
                group_lines.setdefault(key, []).append(raw_line)
            except (ValueError, TypeError) as exc:
                errors.append({"line": number, "error": str(exc)[:200]})
        result = {"imported": 0, "skipped": 0, "versions": 0, "articles": 0, "skipped_structural_units": 0, "errors": errors[:100]}
        for key, records in groups.items():
            metadata = {**records[0], "title": key[0], "source_id": "legacy", "source_label": str(source_label), "source_root": str(Path(path).resolve()),
                        "source_kind": "derived_migration_excerpt", "source_statement": "原迁移载体中该法规版本的JSONL摘录，非独立取得的官方原件", "original_archive_sha256": archive_sha,
                        "extraction_quality": records[0].get("extraction_quality", "legacy_unverified")}
            metadata.pop("verification", None)
            metadata.pop("content", None)
            metadata.pop("article_uid", None)
            units, heading = [], ""
            for record in records:
                record = dict(record)
                text = str(record.get("content") or "").strip()
                location = dict(record.get("location") or {})
                if (location.get("unit_kind") == "section" and len(text) <= 100 and "\n" not in text
                        and not re.search(r"[。；;！？!?]", text)):
                    heading = text
                    result["skipped_structural_units"] += 1
                    continue
                if heading:
                    location["section_heading"] = heading
                    record["location"] = location
                units.append(record)
            try:
                excerpt = b"".join(group_lines[key])
                row = self.ingest_document(actor, metadata, units, excerpt, "legacy:" + _sha(excerpt), content_type="application/x-ndjson")
                result["skipped" if row["duplicate"] else "imported"] += 1
                if not row["duplicate"]:
                    result["versions"] += 1
                    result["articles"] += row["article_count"]
            except LawError as exc:
                if len(result["errors"]) < 100:
                    result["errors"].append({"title": key[0], "error": str(exc)[:200]})
        return result

    def _record_crawl_run(self, run):
        now = _now()
        with self._db() as db:
            db.execute("INSERT INTO crawl_runs VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload,updated_at=excluded.updated_at", (run["run_id"], run["status"], _json(run), run.get("started_at", now), now))

    def list_crawl_runs(self, *, limit=30):
        with self._db() as db:
            return {"items": [json.loads(row[0]) for row in db.execute("SELECT payload FROM crawl_runs ORDER BY created_at DESC LIMIT ?", (max(0, min(100, int(limit))),))]}
