"""Durable upload queue. Raw uploads and progress survive browser/server restarts.

Claims use a short renewable lease; expired work is retried, never silently lost.
The runner archives with report_id=task_id, making a completed archive idempotent.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Callable, Iterable


class QueueError(ValueError):
    pass


class PermanentTaskError(QueueError):
    pass


class QueueStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workspace_batches (
                    batch_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    options TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workspace_tasks (
                    task_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL
                        REFERENCES workspace_batches(batch_id),
                    source_name TEXT NOT NULL, file_type TEXT NOT NULL,
                    content BLOB NOT NULL, content_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3, worker_id TEXT,
                    lease_until REAL, available_at REAL NOT NULL DEFAULT 0,
                    report_id TEXT, result_summary TEXT, error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    started_at REAL, finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS workspace_tasks_claim
                    ON workspace_tasks(status,available_at,created_at);
                CREATE INDEX IF NOT EXISTS workspace_tasks_fifo
                    ON workspace_tasks(status,created_at);
                CREATE INDEX IF NOT EXISTS workspace_tasks_lease
                    ON workspace_tasks(status,lease_until);
                CREATE INDEX IF NOT EXISTS workspace_tasks_batch
                    ON workspace_tasks(batch_id,status,created_at);
                CREATE INDEX IF NOT EXISTS workspace_batches_created
                    ON workspace_batches(created_at DESC);
            """)

    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def create_batch(self, options: dict | None = None) -> dict:
        options = options or {}
        safe = {"engine": "single" if options.get("engine") == "single" else "multi",
                "llm": str(options.get("llm", "0")).lower() in ("1", "true", "on")}
        key, now = uuid.uuid4().hex, time.time()
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO workspace_batches VALUES (?,?,?,?,?)",
                       (key, "draft", json.dumps(safe), now, now))
        return {"batch_id": key, "status": "draft", "options": safe}

    def add_task(self, batch_id: str, source_name: str, file_type: str, content: bytes) -> dict:
        return self.add_many(batch_id, [(source_name, file_type, content)])[0]

    def add_many(self, batch_id: str, items: Iterable[tuple[str, str, bytes]]) -> list[dict]:
        rows, result, now = [], [], time.time()
        for name, kind, content in items:
            name = str(name).replace("\\", "/").rsplit("/", 1)[-1][:240]
            kind = str(kind).lower().lstrip(".")
            if kind not in {"docx", "pdf", "txt", "md", "contract_package"}:
                raise QueueError("批量仅支持 DOCX、PDF、TXT、MD")
            if kind == "contract_package":
                try:
                    descriptor = json.loads(content)
                    if (not isinstance(descriptor, dict)
                            or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(descriptor.get("package_id", "")))
                            or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("manifest_sha256", "")))
                            or not isinstance(descriptor.get("include_tender", False), bool)):
                        raise ValueError("invalid descriptor")
                except (ValueError, TypeError) as exc:
                    raise QueueError("合同包任务必须绑定包编号和证据清单哈希") from exc
            if not content or len(content) > 25 * 1024 * 1024:
                raise QueueError("单份上传必须非空且不超过25MB")
            key = uuid.uuid4().hex
            rows.append((key, batch_id, name, kind, content, len(content), now, now))
            result.append({"task_id": key, "source_name": name, "status": "pending"})
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute("SELECT status FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if not batch:
                raise QueueError("批次不存在")
            if batch["status"] != "draft":
                raise QueueError("批次已提交，不能继续添加文件")
            count = db.execute("SELECT COUNT(*) FROM workspace_tasks WHERE batch_id=?", (batch_id,)).fetchone()[0]
            if count + len(rows) > 100000:
                raise QueueError("每个服务端批次最多100000项")
            db.executemany("""INSERT INTO workspace_tasks
                (task_id,batch_id,source_name,file_type,content,content_bytes,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?)""", rows)
        return result

    def start_batch(self, batch_id: str):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute("SELECT status FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if not batch:
                raise QueueError("批次不存在")
            if batch["status"] != "draft":
                return
            if not db.execute("SELECT 1 FROM workspace_tasks WHERE batch_id=? LIMIT 1", (batch_id,)).fetchone():
                raise QueueError("请先上传合同")
            db.execute("UPDATE workspace_batches SET status='queued',updated_at=? WHERE batch_id=?",
                       (time.time(), batch_id))

    def _refresh(self, db, batch_id: str):
        batch = db.execute("SELECT status FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if not batch or batch["status"] == "draft":
            return
        counts = {status: db.execute("SELECT 1 FROM workspace_tasks WHERE batch_id=? AND status=? LIMIT 1",
            (batch_id, status)).fetchone() is not None for status in ("running", "pending", "cancelled", "failed")}
        if counts.get("running"):
            status = "cancelling" if batch["status"] in ("cancelling", "cancelled") else "running"
        elif counts.get("pending"):
            status = "queued"
        elif counts.get("cancelled"):
            status = "cancelled"
        elif counts.get("failed"):
            status = "failed"
        else:
            status = "completed"
        db.execute("UPDATE workspace_batches SET status=?,updated_at=? WHERE batch_id=?",
                   (status, time.time(), batch_id))

    def recover(self) -> int:
        """Only reclaim expired leases; a second process cannot steal live work."""
        now = time.time()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            stale = db.execute("SELECT task_id,batch_id,attempts,max_attempts FROM workspace_tasks WHERE status='running' AND lease_until<?", (now,)).fetchall()
            for task in stale:
                batch = db.execute("SELECT status FROM workspace_batches WHERE batch_id=?", (task["batch_id"],)).fetchone()
                cancelled = batch and batch["status"] in ("cancelling", "cancelled")
                exhausted = task["attempts"] >= task["max_attempts"]
                status = "cancelled" if cancelled else "failed" if exhausted else "pending"
                reason = "服务多次中断，已达到重试上限，请人工检查后重试" if exhausted else "服务中断后恢复任务"
                db.execute("""UPDATE workspace_tasks SET status=?,worker_id=NULL,lease_until=NULL,
                    error=?,updated_at=? WHERE task_id=?""", (status, reason, now, task["task_id"]))
                self._refresh(db, task["batch_id"])
            return len(stale)

    def claim_next(self, worker_id: str, lease_seconds: float = 30) -> dict | None:
        self.recover()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT t.*,b.options FROM workspace_tasks t INDEXED BY workspace_tasks_fifo
                JOIN workspace_batches b ON b.batch_id=t.batch_id
                WHERE t.status='pending' AND t.available_at<=?
                  AND b.status IN ('queued','running') ORDER BY t.created_at,t.rowid LIMIT 1""", (time.time(),)).fetchone()
            if not row:
                return None
            now = time.time()
            db.execute("""UPDATE workspace_tasks SET status='running',attempts=attempts+1,
                worker_id=?,lease_until=?,started_at=?,updated_at=? WHERE task_id=?""",
                (worker_id, now + lease_seconds, now, now, row["task_id"]))
            self._refresh(db, row["batch_id"])
            task = dict(row)
            task.update({"status": "running", "attempts": row["attempts"] + 1,
                         "worker_id": worker_id,
                         "options": json.loads(row["options"])})
            return task

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: float = 30):
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE workspace_tasks SET lease_until=?
                WHERE task_id=? AND worker_id=? AND status='running'""",
                (time.time() + lease_seconds, task_id, worker_id))

    def complete(self, task_id: str, report_id: str, summary: dict | None = None, worker_id: str | None = None):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT batch_id,worker_id,status FROM workspace_tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row or row["status"] != "running" or (worker_id and row["worker_id"] != worker_id):
                return False
            now = time.time()
            db.execute("""UPDATE workspace_tasks SET status='completed',report_id=?,result_summary=?,
                error='',lease_until=NULL,finished_at=?,updated_at=? WHERE task_id=?""",
                (report_id, json.dumps(summary or {}, ensure_ascii=False), now, now, task_id))
            self._refresh(db, row["batch_id"])
            return True

    def fail(self, task_id: str, error: str, *, permanent=False, worker_id: str | None = None):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM workspace_tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row or row["status"] != "running" or (worker_id and row["worker_id"] != worker_id):
                return False
            batch = db.execute("SELECT status FROM workspace_batches WHERE batch_id=?", (row["batch_id"],)).fetchone()
            retry = not permanent and row["attempts"] < row["max_attempts"] and batch["status"] not in ("cancelling", "cancelled")
            now = time.time()
            db.execute("""UPDATE workspace_tasks SET status=?,error=?,lease_until=NULL,available_at=?,
                finished_at=?,updated_at=? WHERE task_id=?""",
                ("pending" if retry else "failed", str(error)[:1000], now + min(30, 2 ** row["attempts"]), now, now, task_id))
            self._refresh(db, row["batch_id"])
            return True

    def cancel_batch(self, batch_id: str):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone():
                raise QueueError("批次不存在")
            db.execute("UPDATE workspace_tasks SET status='cancelled',updated_at=? WHERE batch_id=? AND status='pending'", (time.time(), batch_id))
            db.execute("UPDATE workspace_batches SET status='cancelling',updated_at=? WHERE batch_id=?", (time.time(), batch_id))
            self._refresh(db, batch_id)

    def retry_batch(self, batch_id: str):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone():
                raise QueueError("批次不存在")
            db.execute("""UPDATE workspace_tasks SET status='pending',attempts=0,error='',
                available_at=0,worker_id=NULL,lease_until=NULL,updated_at=?
                WHERE batch_id=? AND status IN ('failed','cancelled')""", (time.time(), batch_id))
            db.execute("UPDATE workspace_batches SET status='queued',updated_at=? WHERE batch_id=?", (time.time(), batch_id))
            self._refresh(db, batch_id)

    def get_batch(self, batch_id: str, page=1, size=100) -> dict:
        page, size = max(1, int(page)), max(1, min(1000, int(size)))
        with closing(self.connect()) as db:
            batch = db.execute("SELECT * FROM workspace_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if not batch:
                raise QueueError("批次不存在")
            counts = dict(db.execute("SELECT status,COUNT(*) FROM workspace_tasks WHERE batch_id=? GROUP BY status", (batch_id,)))
            rows = db.execute("""SELECT task_id,source_name,file_type,content_bytes,status,attempts,
                report_id,result_summary,error,created_at,started_at,finished_at FROM workspace_tasks
                WHERE batch_id=? ORDER BY created_at,rowid LIMIT ? OFFSET ?""", (batch_id, size, (page-1)*size)).fetchall()
            result = dict(batch)
            result.update({"options": json.loads(result["options"]), "counts": counts,
                           "total": sum(counts.values()), "page": page, "size": size,
                           "items": []})
            for row in rows:
                item = dict(row)
                item["result_summary"] = json.loads(item["result_summary"] or "{}")
                item["seconds"] = max(0, (item["finished_at"] or time.time()) - item["started_at"]) if item["started_at"] else 0
                result["items"].append(item)
            return result

    def list_batches(self, limit=20) -> list[dict]:
        with closing(self.connect()) as db:
            keys = [r[0] for r in db.execute("SELECT batch_id FROM workspace_batches ORDER BY created_at DESC LIMIT ?", (max(1, min(100, int(limit))),))]
        return [{k:v for k,v in self.get_batch(key, size=1).items() if k != "items"} for key in keys]


class QueueWorker:
    def __init__(self, store: QueueStore, runner: Callable[[dict], tuple[str, dict]], poll_interval=.5):
        self.store, self.runner, self.poll_interval = store, runner, poll_interval
        self.worker_id = uuid.uuid4().hex
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="workspace-review-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _loop(self):
        while not self.stopping.is_set():
            try:
                if self.run_once():
                    continue
            except (OSError, sqlite3.Error):
                pass  # Durable rows remain; reconnect after a temporary storage failure.
            self.stopping.wait(self.poll_interval)

    def run_once(self) -> bool:
        task = self.store.claim_next(self.worker_id)
        if task is None:
            return False
        done = threading.Event()

        def renew():
            while not done.wait(5):
                try:
                    self.store.heartbeat(task["task_id"], self.worker_id)
                except (OSError, sqlite3.Error):
                    pass

        pulse = threading.Thread(target=renew, daemon=True)
        pulse.start()
        try:
            report_id, summary = self.runner(task)
            self.store.complete(task["task_id"], report_id, summary, self.worker_id)
        except Exception as exc:
            self.store.fail(task["task_id"], f"{type(exc).__name__}: {exc}",
                            permanent=isinstance(exc, PermanentTaskError), worker_id=self.worker_id)
        finally:
            done.set()
            pulse.join(timeout=1)
        return True
