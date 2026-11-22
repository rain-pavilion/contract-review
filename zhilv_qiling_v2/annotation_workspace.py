"""Local blind manual-annotation workspace; never generates labels or identities.

The existing acceptance_annotations.py remains the authority for validation and
freeze. Accounts are self-declared local identities, not proof of real humans.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "zlql-human-gold-v1"
BLIND_SCHEMA = "zlql-blind-review-v1"
LIMITATION = ("本工作台不证明真实身份或实际独立性；账号、人工完成与盲标声明由实际人员负责。"
              "未标注、未双人提交、未仲裁或未通过防泄漏校验的数据不能称为冻结金标。"
              "合成来源即使人工完成标注也仍是合成测试材料。")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def jsonl(rows: list[dict]) -> str:
    return "".join(canonical(row) + "\n" for row in rows)


class AnnotationError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def require(condition: Any, message: str, status: int = 400) -> None:
    if not condition:
        raise AnnotationError(message, status)


def parse_jsonl(value: str) -> list[dict]:
    require(isinstance(value, str), "JSONL须为文本")
    try:
        rows = [json.loads(line) for line in value.lstrip("\ufeff").splitlines() if line.strip()]
    except (TypeError, ValueError) as exc:
        raise AnnotationError("JSONL格式错误") from exc
    require(rows and all(isinstance(row, dict) for row in rows), "需要非空的JSON对象行")
    return rows


def check_annotations(text: str, value: Any) -> list[dict]:
    require(isinstance(value, list) and len(value) <= 2000, "annotations须为至多2000项的数组")
    output = []
    for item in value:
        require(isinstance(item, dict), "每条标注须为对象")
        require(item.get("type") in ("entity", "risk", "law_reference"), "标注类型须为entity/risk/law_reference")
        a, b = item.get("start"), item.get("end")
        require(type(a) is int and type(b) is int and 0 <= a < b <= len(text), "标注范围必须是有效Unicode码点下标[start,end)")
        require(isinstance(item.get("label"), str) and 0 < len(item["label"].strip()) <= 200, "需要标签，最长200字")
        require(item.get("evidence") == text[a:b], "evidence与原文范围不一致")
        notes = item.get("notes", "")
        require(isinstance(notes, str) and len(notes) <= 4000, "标注备注最长4000字")
        output.append(dict(item))
    return output


class AnnotationWorkspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "annotations.sqlite3"
        with closing(self.connect()) as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS people(
                    reviewer_id TEXT PRIMARY KEY, role TEXT NOT NULL,
                    token_sha256 TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS datasets(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'open', guidelines TEXT NOT NULL,
                    guidelines_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    frozen_version TEXT);
                CREATE TABLE IF NOT EXISTS records(
                    dataset_id TEXT NOT NULL, id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    base_json TEXT NOT NULL, PRIMARY KEY(dataset_id,id));
                CREATE TABLE IF NOT EXISTS reviews(
                    dataset_id TEXT NOT NULL, record_id TEXT NOT NULL, slot TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL,
                    annotations_json TEXT NOT NULL, notes TEXT NOT NULL, claimed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, submitted_at TEXT, attestation_json TEXT,
                    PRIMARY KEY(dataset_id,record_id,slot));
                CREATE TABLE IF NOT EXISTS adjudications(
                    dataset_id TEXT NOT NULL, record_id TEXT NOT NULL, reviewer_id TEXT NOT NULL,
                    status TEXT NOT NULL, revision INTEGER NOT NULL, annotations_json TEXT NOT NULL,
                    rationale TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    adjudicated_at TEXT, attestation_json TEXT,
                    PRIMARY KEY(dataset_id,record_id));
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                    actor_id TEXT NOT NULL, actor_role TEXT NOT NULL, action TEXT NOT NULL,
                    detail_json TEXT NOT NULL, previous_sha256 TEXT NOT NULL, event_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS import_sources(
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, reviewer_id TEXT NOT NULL,
                    slot TEXT NOT NULL, imported_at TEXT NOT NULL, source_jsonl TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL);
            """)
            c.commit()

    def connect(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        return c

    @contextmanager
    def transaction(self):
        with closing(self.connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise

    def event(self, c, actor, action, detail):
        previous = c.execute("SELECT event_sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = previous[0] if previous else "0" * 64
        payload = {"created_at": now(), "actor_id": actor["reviewer_id"], "actor_role": actor["role"],
                   "action": action, "detail_json": canonical(detail), "previous_sha256": previous}
        c.execute("INSERT INTO events(created_at,actor_id,actor_role,action,detail_json,previous_sha256,event_sha256) VALUES(?,?,?,?,?,?,?)",
                  (*payload.values(), digest(canonical(payload))))

    def bootstrap(self, admin_id: str):
        require(isinstance(admin_id, str) and admin_id.strip(), "请由实际管理员填写admin_id")
        token = secrets.token_urlsafe(32)
        actor = {"reviewer_id": admin_id.strip(), "role": "admin"}
        with self.transaction() as c:
            require(not c.execute("SELECT 1 FROM people LIMIT 1").fetchone(), "工作台已初始化，不能重置管理员", 409)
            c.execute("INSERT INTO people VALUES(?,?,?,?)", (actor["reviewer_id"], "admin", digest(token), now()))
            self.event(c, actor, "bootstrap", {"identity_verification": "self_declared_not_verified"})
        return {**actor, "token": token, "identity_verification": "self_declared_not_verified"}

    def authenticate(self, token: str):
        require(isinstance(token, str) and token, "需要标注工作台访问令牌", 401)
        with closing(self.connect()) as c:
            row = c.execute("SELECT reviewer_id,role,created_at FROM people WHERE token_sha256=?", (digest(token),)).fetchone()
        require(row is not None, "无效的标注工作台令牌", 401)
        return dict(row)

    @staticmethod
    def roles(actor, *roles):
        require(actor["role"] in roles, "当前角色无权执行此操作", 403)

    @staticmethod
    def dataset(c, dataset_id, *, writable=False):
        row = c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        require(row is not None, "标注数据集不存在", 404)
        require(not writable or row["state"] == "open", "数据集已冻结，须创建新版本重新标注", 409)
        return dict(row)

    @staticmethod
    def base(c, dataset_id, record_id):
        row = c.execute("SELECT base_json FROM records WHERE dataset_id=? AND id=?", (dataset_id, record_id)).fetchone()
        require(row is not None, "原文记录不存在", 404)
        return json.loads(row[0])

    def standard(self, rows, *, training=None, freeze=False, seed=20261114):
        """Execute the existing validator/freeze CLI unchanged, in an isolated folder."""
        staging = self.root / "staging"
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="validation_", dir=staging) as folder:
            folder = Path(folder)
            data, result = folder / "source.jsonl", folder / "validation.json"
            data.write_text(jsonl(rows), encoding="utf-8")
            command = [sys.executable, "-X", "utf8", str(ROOT / "tools" / "acceptance_annotations.py"),
                       "freeze" if freeze else "validate", "--data", str(data), "--out", str(result)]
            if training is not None:
                inventory = folder / "training.jsonl"
                inventory.write_text(jsonl(training), encoding="utf-8")
                command.extend(["--training-inventory", str(inventory)])
            if freeze:
                command.extend(["--seed", str(seed)])
            try:
                run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AnnotationError("既有标注验证工具执行失败；未冻结金标", 503) from exc
            require(result.is_file(), "既有标注验证工具未产生结果；未冻结金标", 503)
            report = json.loads(result.read_text(encoding="utf-8"))
            frozen = result.with_suffix(".jsonl")
            data_bytes = frozen.read_bytes() if frozen.is_file() else None
            require(run.returncode in (0, 2), "既有标注验证工具异常退出", 503)
            return report, data_bytes

    def create_user(self, actor, body):
        self.roles(actor, "admin")
        reviewer_id, role = body.get("reviewer_id"), body.get("role")
        require(isinstance(reviewer_id, str) and 0 < len(reviewer_id.strip()) <= 120, "需要实际人员唯一编号，最长120字")
        require(role in ("A", "B", "adjudicator"), "角色须为A、B或adjudicator")
        require(body.get("identity_confirmed_by_admin") is True, "管理员须确认由对应实际人员独立使用；软件不核验身份")
        token = secrets.token_urlsafe(32)
        with self.transaction() as c:
            require(not c.execute("SELECT 1 FROM people WHERE reviewer_id=?", (reviewer_id.strip(),)).fetchone(),
                    "此人员编号已占用；A、B与仲裁员必须为不同人员编号", 409)
            c.execute("INSERT INTO people VALUES(?,?,?,?)", (reviewer_id.strip(), role, digest(token), now()))
            self.event(c, actor, "user_created", {"reviewer_id": reviewer_id.strip(), "role": role,
                                                 "identity_verification": "admin_attested_not_verified"})
        return {"reviewer_id": reviewer_id.strip(), "role": role, "token": token,
                "notice": "令牌仅本次返回；请私下分别交给对应真实人员，不放入公开URL或申报材料"}

    def create_dataset(self, actor, body):
        self.roles(actor, "admin")
        rows = body.get("rows") if "rows" in body else parse_jsonl(body.get("jsonl", ""))
        require(isinstance(rows, list) and 0 < len(rows) <= 1000 and all(isinstance(r, dict) for r in rows), "一次导入1至1000份候选原文")
        name, guidelines = body.get("name"), body.get("guidelines")
        require(isinstance(name, str) and 0 < len(name.strip()) <= 200, "需要数据集名称")
        require(isinstance(guidelines, str) and 20 <= len(guidelines.strip()) <= 100000,
                "请先确定标签手册/说明（至少20字）；创建后不可修改，修订须新建数据集")
        pristine = []
        for row in rows:
            require(row.get("status") == "unlabeled" and row.get("annotations") == [] and row.get("reviews") == [],
                    "候选导入只接受未标原文；独立结果请使用A/B导入")
            require(isinstance(row.get("id"), str) and 0 < len(row["id"]) <= 200, "候选id须为非空文本")
            require(isinstance(row.get("text"), str) and 0 < len(row["text"]) <= 500000, "候选原文长度须为1至500000码点")
            require(isinstance(row.get("source"), dict) and all(isinstance(row["source"].get(key), str)
                and row["source"][key].strip() for key in ("origin", "group_id", "license_or_permission", "collected_at")),
                "每份原文都需要来源、同源组、使用权限及采集日期")
            base = {key: row.get(key) for key in ("schema_version", "id", "text", "text_sha256", "source")}
            for key in ("synthetic_candidate", "contract_type"):
                if key in row:
                    base[key] = row[key]
            base.update(status="unlabeled", split="unassigned", annotations=[], reviews=[])
            pristine.append(base)
        report, _ = self.standard(pristine)
        require(report["status"] == "passed", "候选包验证失败：" + "; ".join(report["errors"][:8]), 422)
        dataset_id = uuid.uuid4().hex
        with self.transaction() as c:
            c.execute("INSERT INTO datasets(id,name,created_at,guidelines,guidelines_sha256,source_sha256) VALUES(?,?,?,?,?,?)",
                      (dataset_id, name.strip(), now(), guidelines.strip(), digest(guidelines.strip()), digest(jsonl(pristine))))
            for index, row in enumerate(pristine):
                c.execute("INSERT INTO records VALUES(?,?,?,?)", (dataset_id, row["id"], index, canonical(row)))
            self.event(c, actor, "dataset_created", {"dataset_id": dataset_id, "records": len(pristine),
                       "source_sha256": digest(jsonl(pristine)), "gold_created": False})
        return {"dataset_id": dataset_id, "records": len(pristine), "status_label": "pending_manual_annotation", "gold_ready": False}

    def list_datasets(self, actor):
        with closing(self.connect()) as c:
            output = []
            for row in c.execute("SELECT * FROM datasets ORDER BY created_at DESC"):
                item = dict(row)
                item.pop("guidelines")
                did = row["id"]
                item["records"] = c.execute("SELECT COUNT(*) FROM records WHERE dataset_id=?", (did,)).fetchone()[0]
                item["submitted_A"] = c.execute("SELECT COUNT(*) FROM reviews WHERE dataset_id=? AND slot='A' AND status='submitted'", (did,)).fetchone()[0]
                item["submitted_B"] = c.execute("SELECT COUNT(*) FROM reviews WHERE dataset_id=? AND slot='B' AND status='submitted'", (did,)).fetchone()[0]
                item["adjudicated"] = c.execute("SELECT COUNT(*) FROM adjudications WHERE dataset_id=? AND status='adjudicated'", (did,)).fetchone()[0]
                item["gold_ready"] = row["state"] == "frozen"
                output.append(item)
        return {"datasets": output, "limitation": LIMITATION}

    def task(self, c, actor, dataset_id, record_id):
        ds = self.dataset(c, dataset_id)
        base = self.base(c, dataset_id, record_id)
        review = c.execute("SELECT * FROM reviews WHERE dataset_id=? AND record_id=? AND slot=?",
                           (dataset_id, record_id, actor["role"])).fetchone()
        require(review is not None and review["reviewer_id"] == actor["reviewer_id"], "请先领取自己的任务；不可查看他人结果", 403)
        mine = dict(review)
        mine["annotations"] = json.loads(mine.pop("annotations_json"))
        mine["attestation"] = json.loads(mine.pop("attestation_json") or "null")
        return {"dataset_id": dataset_id, "record": base, "review": mine,
                "guidelines": ds["guidelines"], "guidelines_sha256": ds["guidelines_sha256"],
                "dataset_state": ds["state"], "blind": True}

    def claim(self, actor, body):
        self.roles(actor, "A", "B")
        did = body.get("dataset_id")
        with self.transaction() as c:
            self.dataset(c, did, writable=True)
            rid = body.get("record_id")
            if not rid:
                mine = c.execute("SELECT record_id FROM reviews WHERE dataset_id=? AND slot=? AND reviewer_id=? AND status='draft' ORDER BY claimed_at LIMIT 1",
                                 (did, actor["role"], actor["reviewer_id"])).fetchone()
                if mine:
                    rid = mine[0]
                else:
                    pending = c.execute("""SELECT r.id FROM records r LEFT JOIN reviews v ON
                        v.dataset_id=r.dataset_id AND v.record_id=r.id AND v.slot=?
                        WHERE r.dataset_id=? AND v.record_id IS NULL ORDER BY r.ordinal LIMIT 1""", (actor["role"], did)).fetchone()
                    require(pending is not None, "此槽位没有待领取任务", 409)
                    rid = pending[0]
            self.base(c, did, rid)
            existing = c.execute("SELECT reviewer_id FROM reviews WHERE dataset_id=? AND record_id=? AND slot=?", (did, rid, actor["role"])).fetchone()
            require(not existing or existing[0] == actor["reviewer_id"], "该槽位任务已被其他人员领取", 409)
            if not existing:
                timestamp = now()
                c.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (did, rid, actor["role"], actor["reviewer_id"], "draft", 0, "[]", "", timestamp, timestamp, None, None))
                self.event(c, actor, "review_claimed", {"dataset_id": did, "record_id": rid, "slot": actor["role"]})
            return self.task(c, actor, did, rid)

    def save_review(self, c, actor, body, *, submit=False, imported=False):
        did, rid = body.get("dataset_id"), body.get("record_id")
        ds = self.dataset(c, did, writable=True)
        current = self.task(c, actor, did, rid)
        review = current["review"]
        require(review["status"] == "draft", "已提交结果锁定，不能覆盖", 409)
        require(type(body.get("revision")) is int and body["revision"] == review["revision"], "草稿版本已变化，请重新载入后编辑", 409)
        annotations = check_annotations(current["record"]["text"], body.get("annotations", review["annotations"]))
        notes = body.get("notes", review["notes"])
        require(isinstance(notes, str) and len(notes) <= 4000, "备注最长4000字")
        attestation = body.get("attestation")
        reviewed_at = now()
        if submit:
            require(isinstance(attestation, dict) and all(attestation.get(k) is True for k in
                ("human_authored", "independent", "read_full_text")) and attestation.get("used_system_predictions") is False,
                "须由实际标注人声明人工独立通读，未使用系统预测")
            require(attestation.get("guidelines_sha256") == ds["guidelines_sha256"], "标注手册版本不匹配")
            require(annotations or attestation.get("empty_confirmed") is True, "空标注须人工确认确实无标注项")
            if imported:
                reviewed_at = body.get("reviewed_at")
                try:
                    require(isinstance(reviewed_at, str) and datetime.fromisoformat(reviewed_at.replace("Z", "+00:00")).tzinfo is not None,
                            "离线提交需要带时区的真实reviewed_at")
                except (TypeError, ValueError) as exc:
                    raise AnnotationError("离线提交reviewed_at格式错误") from exc
        c.execute("""UPDATE reviews SET status=?,revision=revision+1,annotations_json=?,notes=?,updated_at=?,submitted_at=?,attestation_json=?
            WHERE dataset_id=? AND record_id=? AND slot=?""", ("submitted" if submit else "draft", canonical(annotations), notes,
            now(), reviewed_at if submit else None, canonical(attestation) if submit else None, did, rid, actor["role"]))
        self.event(c, actor, "review_imported" if imported else "review_submitted" if submit else "draft_saved",
                   {"dataset_id": did, "record_id": rid, "slot": actor["role"], "revision": review["revision"] + 1,
                    "status": "submitted" if submit else "draft", "annotations_sha256": digest(canonical(annotations)), "annotations_count": len(annotations)})
        return self.task(c, actor, did, rid)

    def review_action(self, actor, body, *, submit=False):
        self.roles(actor, "A", "B")
        with self.transaction() as c:
            return self.save_review(c, actor, body, submit=submit)

    def import_reviews(self, actor, body):
        self.roles(actor, "A", "B")
        rows = body.get("rows") if "rows" in body else parse_jsonl(body.get("jsonl", ""))
        require(isinstance(rows, list) and 0 < len(rows) <= 1000 and all(isinstance(r, dict) for r in rows), "需要1至1000项独立结果")
        did, imported = body.get("dataset_id"), []
        with self.transaction() as c:
            self.dataset(c, did, writable=True)
            ids = set()
            for row in rows:
                rid = row.get("id")
                require(rid not in ids, "导入批次中任务id重复")
                ids.add(rid)
                base = self.base(c, did, rid)
                require(row.get("schema_version") == BLIND_SCHEMA and row.get("slot") == actor["role"] and
                        row.get("reviewer_id") == actor["reviewer_id"], "只能导入本人对应槽位的独立结果", 403)
                require(row.get("text_sha256") == base["text_sha256"], "导入原文hash不匹配", 422)
                require(row.get("status") in ("draft", "submitted"), "未标空模板不能当作已完成结果导入")
                if row["status"] == "submitted":
                    require(row.get("method") == "manual" and row.get("used_system_predictions") is False,
                            "离线提交必须明确method=manual且used_system_predictions=false")
                require(row.get("annotations_sha256") == digest(canonical(row.get("annotations"))), "独立标注hash不匹配", 422)
                existing = c.execute("SELECT reviewer_id FROM reviews WHERE dataset_id=? AND record_id=? AND slot=?", (did, rid, actor["role"])).fetchone()
                if not existing:
                    timestamp = now()
                    c.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (did, rid, actor["role"], actor["reviewer_id"], "draft", 0, "[]", "", timestamp, timestamp, None, None))
                payload = {**row, "dataset_id": did, "record_id": rid}
                result = self.save_review(c, actor, payload, submit=row["status"] == "submitted", imported=True)
                imported.append({"id": rid, "status": result["review"]["status"], "revision": result["review"]["revision"]})
            original = body.get("jsonl") if isinstance(body.get("jsonl"), str) else jsonl(rows)
            import_id = uuid.uuid4().hex
            c.execute("INSERT INTO import_sources VALUES(?,?,?,?,?,?,?)", (import_id, did, actor["reviewer_id"],
                actor["role"], now(), original, digest(original)))
            self.event(c, actor, "blind_import_batch", {"dataset_id": did, "import_id": import_id,
                "source_sha256": digest(original), "records": len(imported)})
        return {"imported": imported, "count": len(imported)}

    def adjudication_task(self, c, actor, did, rid):
        ds = self.dataset(c, did)
        base = self.base(c, did, rid)
        reviews = c.execute("SELECT * FROM reviews WHERE dataset_id=? AND record_id=? AND status='submitted' ORDER BY slot", (did, rid)).fetchall()
        require(len(reviews) == 2 and {r["slot"] for r in reviews} == {"A", "B"}, "两路独立提交尚未齐全，不能查看或仲裁", 409)
        require(len({r["reviewer_id"] for r in reviews}) == 2, "独立标注人编号不能相同", 409)
        item = c.execute("SELECT * FROM adjudications WHERE dataset_id=? AND record_id=?", (did, rid)).fetchone()
        require(item is not None and item["reviewer_id"] == actor["reviewer_id"], "请先领取自己的仲裁任务", 403)
        require(actor["reviewer_id"] not in {r["reviewer_id"] for r in reviews}, "仲裁员须独立于两名标注员", 403)
        independent = []
        for review in reviews:
            value = dict(review)
            value["annotations"] = json.loads(value.pop("annotations_json"))
            value["attestation"] = json.loads(value.pop("attestation_json"))
            independent.append(value)
        item = dict(item)
        item["annotations"] = json.loads(item.pop("annotations_json"))
        item["attestation"] = json.loads(item.pop("attestation_json") or "null")
        fingerprints = [sorted(canonical(a) for a in review["annotations"]) for review in independent]
        return {"dataset_id": did, "record": base, "reviews": independent, "adjudication": item,
                "disagreement": fingerprints[0] != fingerprints[1], "dataset_state": ds["state"],
                "guidelines": ds["guidelines"], "guidelines_sha256": ds["guidelines_sha256"]}

    def claim_adjudication(self, actor, body):
        self.roles(actor, "adjudicator")
        did, rid = body.get("dataset_id"), body.get("record_id")
        with self.transaction() as c:
            self.dataset(c, did, writable=True)
            if not rid:
                mine = c.execute("SELECT record_id FROM adjudications WHERE dataset_id=? AND reviewer_id=? AND status='draft' LIMIT 1", (did, actor["reviewer_id"])).fetchone()
                if mine:
                    rid = mine[0]
                else:
                    pending = c.execute("""SELECT r.id FROM records r WHERE r.dataset_id=?
                        AND (SELECT COUNT(*) FROM reviews v WHERE v.dataset_id=r.dataset_id AND v.record_id=r.id AND v.status='submitted')=2
                        AND NOT EXISTS(SELECT 1 FROM adjudications a WHERE a.dataset_id=r.dataset_id AND a.record_id=r.id)
                        ORDER BY r.ordinal LIMIT 1""", (did,)).fetchone()
                    require(pending is not None, "暂无双路提交齐全的待仲裁任务", 409)
                    rid = pending[0]
            self.base(c, did, rid)
            reviews = c.execute("SELECT reviewer_id,slot FROM reviews WHERE dataset_id=? AND record_id=? AND status='submitted'", (did, rid)).fetchall()
            require(len(reviews) == 2 and len({r["reviewer_id"] for r in reviews}) == 2, "需先完成两名不同标注员独立提交", 409)
            existing = c.execute("SELECT reviewer_id FROM adjudications WHERE dataset_id=? AND record_id=?", (did, rid)).fetchone()
            require(not existing or existing[0] == actor["reviewer_id"], "仲裁任务已由其他人员领取", 409)
            if not existing:
                timestamp = now()
                c.execute("INSERT INTO adjudications VALUES(?,?,?,?,?,?,?,?,?,?,?)", (did, rid, actor["reviewer_id"], "draft", 0, "[]", "", timestamp, timestamp, None, None))
                self.event(c, actor, "adjudication_claimed", {"dataset_id": did, "record_id": rid})
            return self.adjudication_task(c, actor, did, rid)

    def adjudicate(self, actor, body, *, submit=False):
        self.roles(actor, "adjudicator")
        did, rid = body.get("dataset_id"), body.get("record_id")
        with self.transaction() as c:
            ds = self.dataset(c, did, writable=True)
            task = self.adjudication_task(c, actor, did, rid)
            current = task["adjudication"]
            require(current["status"] == "draft", "仲裁已锁定，不能覆盖", 409)
            require(type(body.get("revision")) is int and body["revision"] == current["revision"], "仲裁版本已变化，请重新载入", 409)
            annotations = check_annotations(task["record"]["text"], body.get("annotations", current["annotations"]))
            rationale = body.get("rationale", current["rationale"])
            require(isinstance(rationale, str) and len(rationale) <= 8000, "仲裁理由最长8000字")
            attestation = body.get("attestation")
            if submit:
                require(rationale.strip(), "需要实际仲裁理由，包括一致或分歧的处理依据")
                require(isinstance(attestation, dict) and all(attestation.get(k) is True for k in
                    ("human_authored", "reviewed_both", "read_full_text")), "实际仲裁员须确认人工通读并核对两路独立结果")
                require(attestation.get("guidelines_sha256") == ds["guidelines_sha256"], "仲裁手册版本不匹配")
                require(annotations or attestation.get("empty_confirmed") is True, "空仲裁结果须人工确认")
            c.execute("""UPDATE adjudications SET status=?,revision=revision+1,annotations_json=?,rationale=?,updated_at=?,adjudicated_at=?,attestation_json=?
                WHERE dataset_id=? AND record_id=?""", ("adjudicated" if submit else "draft", canonical(annotations), rationale,
                now(), now() if submit else None, canonical(attestation) if submit else None, did, rid))
            self.event(c, actor, "adjudication_submitted" if submit else "adjudication_draft_saved",
                       {"dataset_id": did, "record_id": rid, "revision": current["revision"] + 1,
                        "annotations_sha256": digest(canonical(annotations)), "disagreement": task["disagreement"]})
            return self.adjudication_task(c, actor, did, rid)

    def snapshot(self, c, did):
        self.dataset(c, did)
        output = []
        for record in c.execute("SELECT base_json FROM records WHERE dataset_id=? ORDER BY ordinal", (did,)):
            row = json.loads(record[0])
            reviews = c.execute("SELECT * FROM reviews WHERE dataset_id=? AND record_id=? ORDER BY slot", (did, row["id"])).fetchall()
            submitted = [r for r in reviews if r["status"] == "submitted"]
            if reviews:
                row["status"] = "in_review"
            row["reviews"] = [{"reviewer_id": r["reviewer_id"], "reviewed_at": r["submitted_at"],
                "slot": r["slot"], "method": "manual", "decision": r["notes"] or "独立人工标注已提交",
                "used_system_predictions": False, "annotations": json.loads(r["annotations_json"]),
                "annotations_sha256": digest(r["annotations_json"]),
                "independence_attestation": json.loads(r["attestation_json"])} for r in submitted]
            adjudication = c.execute("SELECT * FROM adjudications WHERE dataset_id=? AND record_id=? AND status='adjudicated'", (did, row["id"])).fetchone()
            if adjudication:
                row["status"] = "adjudicated"
                row["annotations"] = json.loads(adjudication["annotations_json"])
                row["adjudication"] = {"adjudicator_id": adjudication["reviewer_id"],
                    "adjudicated_at": adjudication["adjudicated_at"], "rationale": adjudication["rationale"],
                    "annotations_sha256": digest(adjudication["annotations_json"]),
                    "attestation": json.loads(adjudication["attestation_json"])}
            output.append(row)
        return output

    def export(self, actor, did, kind):
        with closing(self.connect()) as c:
            ds = self.dataset(c, did)
            if kind == "blind":
                self.roles(actor, "A", "B")
                rows = []
                for record in c.execute("SELECT id,base_json FROM records WHERE dataset_id=? ORDER BY ordinal", (did,)):
                    base = json.loads(record["base_json"])
                    mine = c.execute("SELECT * FROM reviews WHERE dataset_id=? AND record_id=? AND slot=?", (did, record["id"], actor["role"])).fetchone()
                    if mine and mine["reviewer_id"] != actor["reviewer_id"]:
                        continue
                    annotations = json.loads(mine["annotations_json"]) if mine else []
                    rows.append({**base, "schema_version": BLIND_SCHEMA, "dataset_id": did,
                        "slot": actor["role"], "reviewer_id": actor["reviewer_id"],
                        "status": mine["status"] if mine else "pending", "revision": mine["revision"] if mine else 0,
                        "annotations": annotations, "annotations_sha256": digest(canonical(annotations)),
                        "notes": mine["notes"] if mine else "", "reviewed_at": mine["submitted_at"] if mine else None,
                        "method": "manual" if mine and mine["status"] == "submitted" else None,
                        "used_system_predictions": False if mine and mine["status"] == "submitted" else None,
                        "attestation": json.loads(mine["attestation_json"] or "null") if mine else None,
                        "guidelines_sha256": ds["guidelines_sha256"]})
                return {"filename": f"blind_{actor['role']}_{did}.jsonl", "jsonl": jsonl(rows),
                        "guidelines": ds["guidelines"], "guidelines_sha256": ds["guidelines_sha256"], "gold_ready": False}
            self.roles(actor, "admin")
            if kind == "snapshot":
                rows = self.snapshot(c, did)
                report, _ = self.standard(rows)
                return {"filename": f"annotation_snapshot_{did}.jsonl", "jsonl": jsonl(rows),
                        "validation": report, "gold_ready": False, "notice": "工作快照，不是冻结金标"}
            require(kind == "frozen" and ds["state"] == "frozen", "尚无冻结金标可导出", 409)
            folder = self.root / "frozen" / did / ds["frozen_version"]
            manifest_bytes = (folder / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            freeze_event = next((json.loads(row[0]) for row in c.execute(
                "SELECT detail_json FROM events WHERE action='dataset_frozen' ORDER BY seq DESC")
                if json.loads(row[0]).get("dataset_id") == did and json.loads(row[0]).get("version") == ds["frozen_version"]), None)
            require(freeze_event and digest(manifest_bytes) == freeze_event.get("manifest_sha256"), "冻结manifest与审计记录不一致，禁止导出", 409)
            data = (folder / "gold.jsonl").read_bytes()
            require(digest(data) == manifest["gold_sha256"], "冻结文件hash不一致，禁止导出", 409)
            for filename, key in (("training_inventory.jsonl", "training_inventory_sha256"),
                                  ("guidelines.txt", "guidelines_sha256"), ("validation.json", "validation_sha256")):
                require(digest((folder / filename).read_bytes()) == manifest[key], f"冻结{filename}的hash不一致，禁止导出", 409)
            return {"filename": f"gold_{did}_{ds['frozen_version']}.jsonl", "jsonl": data.decode("utf-8"),
                    "manifest": manifest, "gold_ready": True}

    def freeze(self, actor, body):
        self.roles(actor, "admin")
        did = body.get("dataset_id")
        require(all(body.get(k) is True for k in ("inventory_complete_attestation", "near_duplicate_review_completed",
                    "heldout_policy_acknowledged")), "须实际确认训练/开发来源清单完整、近重复人工复查和冻结测试集不用于调优")
        training = body.get("training_inventory")
        require(isinstance(training, list) and training and all(isinstance(row, dict) and isinstance(row.get("text"), str)
            and isinstance(row.get("source"), dict) and row["source"].get("group_id") for row in training), "需要非空、含text及source.group_id的完整训练/开发清单")
        seed = body.get("seed", 20261114)
        require(type(seed) is int and 0 <= seed <= 2**32 - 1, "seed须为32位非负整数")
        version = body.get("version") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        require(isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", version), "版本名称只允许字母数字下划线连字符，最长64字")
        with self.transaction() as c:
            ds = self.dataset(c, did, writable=True)
            rows = self.snapshot(c, did)
            require(rows and all(row["status"] == "adjudicated" for row in rows), "仍有未完成独立标注或仲裁的记录，保持pending，不能冻结", 409)
            report, data = self.standard(rows, training=training, freeze=True, seed=seed)
            require(report.get("gold_ready") is True and report.get("status") == "passed" and data,
                    "既有冻结工具拒绝：" + "; ".join(report.get("errors", [])[:8]), 422)
            destination = self.root / "frozen" / did / version
            try:
                destination.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise AnnotationError("冻结版本已存在，禁止覆盖", 409) from exc
            report["frozen_file"] = str(destination / "gold.jsonl")
            inventory = jsonl(training).encode("utf-8")
            manifest = {"schema_version": "zlql-manual-workspace-freeze-v1", "dataset_id": did,
                "version": version, "frozen_at": now(), "frozen_by": actor["reviewer_id"], "seed": seed,
                "records": len(rows), "gold_sha256": digest(data), "source_sha256": ds["source_sha256"],
                "training_inventory_sha256": digest(inventory), "guidelines_sha256": ds["guidelines_sha256"],
                "validation_sha256": digest(canonical(report)),
                "validator_sha256": digest((ROOT / "tools" / "acceptance_annotations.py").read_bytes()),
                "synthetic_records": sum(bool(row.get("synthetic_candidate")) for row in rows),
                "independently_sourced_real_contracts_verified": False,
                "source_independence": "not_independently_verified_by_workspace; inspect source provenance externally",
                "attestations": {key: True for key in ("inventory_complete_attestation", "near_duplicate_review_completed", "heldout_policy_acknowledged")},
                "identity_verification": "self_declared_and_admin_attested_not_independently_verified",
                "limitation": LIMITATION}
            for filename, content in (("gold.jsonl", data), ("training_inventory.jsonl", inventory),
                    ("guidelines.txt", ds["guidelines"].encode("utf-8")),
                    ("validation.json", canonical(report).encode("utf-8")), ("manifest.json", canonical(manifest).encode("utf-8"))):
                with (destination / filename).open("xb") as f:
                    f.write(content)
            c.execute("UPDATE datasets SET state='frozen',frozen_version=? WHERE id=?", (version, did))
            self.event(c, actor, "dataset_frozen", {"dataset_id": did, "version": version, "gold_sha256": digest(data),
                "manifest_sha256": digest(canonical(manifest))})
            return {"gold_ready": True, "version": version, "manifest": manifest, "validation": report}

    def tasks(self, actor, did):
        with closing(self.connect()) as c:
            ds = self.dataset(c, did)
            rows = []
            for record in c.execute("SELECT id,ordinal,base_json FROM records WHERE dataset_id=? ORDER BY ordinal", (did,)):
                item = {"id": record["id"], "ordinal": record["ordinal"]}
                if actor["role"] in ("A", "B"):
                    mine = c.execute("SELECT reviewer_id,status,revision FROM reviews WHERE dataset_id=? AND record_id=? AND slot=?", (did, record["id"], actor["role"])).fetchone()
                    item.update(status=mine["status"] if mine else "pending", owned=bool(mine and mine["reviewer_id"] == actor["reviewer_id"]))
                    if mine and not item["owned"]:
                        item["status"] = "claimed_by_other"
                else:
                    submitted = c.execute("SELECT COUNT(*) FROM reviews WHERE dataset_id=? AND record_id=? AND status='submitted'", (did, record["id"])).fetchone()[0]
                    adj = c.execute("SELECT reviewer_id,status FROM adjudications WHERE dataset_id=? AND record_id=?", (did, record["id"])).fetchone()
                    item.update(status=adj["status"] if adj else "ready_for_adjudication" if submitted == 2 else "awaiting_two_submissions",
                                owned=bool(adj and adj["reviewer_id"] == actor["reviewer_id"]))
                rows.append(item)
            return {"tasks": rows, "dataset_state": ds["state"], "guidelines": ds["guidelines"],
                    "guidelines_sha256": ds["guidelines_sha256"]}

    def audit(self, actor):
        self.roles(actor, "admin")
        previous, valid, items = "0" * 64, True, []
        with closing(self.connect()) as c:
            for row in c.execute("SELECT * FROM events ORDER BY seq"):
                value = dict(row)
                payload = {key: value[key] for key in ("created_at", "actor_id", "actor_role", "action", "detail_json", "previous_sha256")}
                valid = valid and value["previous_sha256"] == previous and value["event_sha256"] == digest(canonical(payload))
                previous = value["event_sha256"]
                value["detail"] = json.loads(value.pop("detail_json"))
                items.append(value)
        return {"chain_valid": valid, "events": items, "notice": "本地hash链不等于防恶意管理员篡改的独立可信时间戳"}

    def dispatch(self, method: str, path: str, body: dict | None, token: str, query=None):
        actor = self.authenticate(token)
        body = body if isinstance(body, dict) else {}
        query = query or {}
        path = path.rstrip("/")
        if path.startswith("/api/annotations"):
            path = path[len("/api/annotations"):]
        did = query.get("dataset_id") or body.get("dataset_id")
        rid = query.get("record_id") or body.get("record_id")
        if method == "GET" and path == "/session":
            return {"actor": actor, "limitation": LIMITATION}
        if path == "/datasets":
            return self.list_datasets(actor) if method == "GET" else self.create_dataset(actor, body)
        if path == "/users":
            if method == "POST":
                return self.create_user(actor, body)
            self.roles(actor, "admin")
            with closing(self.connect()) as c:
                return {"users": [dict(row) for row in c.execute("SELECT reviewer_id,role,created_at FROM people ORDER BY created_at")]}
        if method == "GET" and path == "/tasks":
            return self.tasks(actor, did)
        if method == "GET" and path == "/task":
            self.roles(actor, "A", "B")
            with closing(self.connect()) as c:
                return self.task(c, actor, did, rid)
        if method == "GET" and path == "/adjudication/task":
            self.roles(actor, "adjudicator")
            with closing(self.connect()) as c:
                return self.adjudication_task(c, actor, did, rid)
        if method == "GET" and path == "/export":
            return self.export(actor, did, query.get("kind", "blind"))
        if method == "GET" and path == "/audit":
            return self.audit(actor)
        if method == "POST":
            routes = {"/claim": self.claim, "/draft": self.review_action, "/import": self.import_reviews,
                "/adjudication/claim": self.claim_adjudication, "/adjudication/draft": self.adjudicate,
                "/freeze": self.freeze}
            if path in routes:
                return routes[path](actor, body)
            if path == "/submit":
                return self.review_action(actor, body, submit=True)
            if path == "/adjudication/submit":
                return self.adjudicate(actor, body, submit=True)
            if path == "/validate":
                self.roles(actor, "admin")
                with closing(self.connect()) as c:
                    rows = self.snapshot(c, did)
                result, _ = self.standard(rows, training=body.get("training_inventory"))
                return {"validation": result, "frozen_gold_ready": False, "notice": "仅校验当前工作快照，不表示已完成冻结"}
        raise AnnotationError("未知标注工作台接口或方法", 404)


def handle_request(workspace: AnnotationWorkspace, method: str, path: str, body=None, authorization="", query=None):
    """Mount adapter. Returns api.py-compatible dictionaries with numeric status."""
    token = authorization.removeprefix("Bearer ").strip() if isinstance(authorization, str) else ""
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        token = ""
    try:
        return workspace.dispatch(method, path, body, token, query)
    except AnnotationError as exc:
        return {"status": exc.status, "error": str(exc), "gold_created": False}
    except (TypeError, KeyError, ValueError) as exc:
        return {"status": 400, "error": "请求结构或字段格式错误", "gold_created": False}


def page_bytes():
    return Path(__file__).with_name("annotation.html").read_bytes()

