"""Private, versioned contract repository with database-enforced actor boundaries.

Only trusted server code creates actors. Every public operation reloads the user
and authenticated actors also revalidate their session inside its transaction.
Each call owns a separate SQLite connection; worker claims use BEGIN IMMEDIATE.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


from .user_library import LibraryError, UserLibrary


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id():
    return str(uuid.uuid4())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(user_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS contracts(
 contract_id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES users(user_id),
 title TEXT NOT NULL, domain TEXT NOT NULL DEFAULT '', region TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, current_version_id TEXT NOT NULL, deleted_at TEXT,
 source_metadata TEXT NOT NULL DEFAULT '{}', legacy_original_unavailable INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS contract_owner ON contracts(owner,created_at);
CREATE TABLE IF NOT EXISTS versions(
 version_id TEXT PRIMARY KEY, contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
 version_no INTEGER NOT NULL, filename TEXT NOT NULL, file_type TEXT NOT NULL,
 sha256 TEXT NOT NULL, size INTEGER NOT NULL, content BLOB NOT NULL,
 created_at TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
 UNIQUE(contract_id,version_no), UNIQUE(contract_id,version_id));
CREATE TRIGGER IF NOT EXISTS immutable_version_update BEFORE UPDATE ON versions
 BEGIN SELECT RAISE(ABORT, 'contract versions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_version_delete BEFORE DELETE ON versions
 BEGIN SELECT RAISE(ABORT, 'contract versions are immutable'); END;
CREATE TABLE IF NOT EXISTS reports(
 report_id TEXT PRIMARY KEY, contract_id TEXT NOT NULL, version_id TEXT NOT NULL,
 created_by TEXT NOT NULL REFERENCES users(user_id), created_at TEXT NOT NULL,
 payload TEXT NOT NULL, risk_count INTEGER NOT NULL, score REAL, source_name TEXT NOT NULL,
 FOREIGN KEY(contract_id,version_id) REFERENCES versions(contract_id,version_id));
CREATE INDEX IF NOT EXISTS report_contract ON reports(contract_id,created_at);
CREATE TABLE IF NOT EXISTS administrator_feedback(
 feedback_id TEXT PRIMARY KEY, report_id TEXT NOT NULL REFERENCES reports(report_id),
 risk_id TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('confirmed','dismissed','pending')),
 note TEXT NOT NULL, reviewer_id TEXT NOT NULL REFERENCES users(user_id),
 reviewer TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS feedback_report ON administrator_feedback(report_id,created_at);
CREATE TRIGGER IF NOT EXISTS immutable_feedback_update BEFORE UPDATE ON administrator_feedback
 BEGIN SELECT RAISE(ABORT, 'administrator feedback is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_feedback_delete BEFORE DELETE ON administrator_feedback
 BEGIN SELECT RAISE(ABORT, 'administrator feedback is append-only'); END;
CREATE TABLE IF NOT EXISTS review_tasks(
 task_id TEXT PRIMARY KEY, contract_id TEXT NOT NULL, version_id TEXT NOT NULL,
 owner TEXT NOT NULL REFERENCES users(user_id), created_by TEXT NOT NULL REFERENCES users(user_id),
 status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
 attempts INTEGER NOT NULL DEFAULT 0, options TEXT NOT NULL, created_at TEXT NOT NULL,
 started_at TEXT, updated_at TEXT NOT NULL, finished_at TEXT, lease_expires_at TEXT,
 worker_id TEXT, report_id TEXT REFERENCES reports(report_id), error TEXT,
 FOREIGN KEY(contract_id,version_id) REFERENCES versions(contract_id,version_id));
CREATE INDEX IF NOT EXISTS task_claim ON review_tasks(status,lease_expires_at,created_at);
CREATE TABLE IF NOT EXISTS task_leases(
 task_id TEXT NOT NULL REFERENCES review_tasks(task_id), worker_id TEXT NOT NULL,
 PRIMARY KEY(task_id,worker_id));
CREATE TABLE IF NOT EXISTS audit_events(
 event_id TEXT PRIMARY KEY, actor_id TEXT REFERENCES users(user_id), action TEXT NOT NULL,
 target_id TEXT, created_at TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS legacy_imports(
 source_path TEXT NOT NULL, source_table TEXT NOT NULL, source_id TEXT NOT NULL,
 contract_id TEXT NOT NULL REFERENCES contracts(contract_id), report_id TEXT NOT NULL REFERENCES reports(report_id),
 PRIMARY KEY(source_path,source_table,source_id));
"""


class _LibraryConnection(sqlite3.Connection):
    write_transaction = False


class ContractLibrary:
    MAX_UPLOAD = 40 * 1024 * 1024
    FILE_TYPES = frozenset('txt md pdf docx doc html htm png jpg jpeg tif tiff bmp webp'.split())

    def __init__(self, path, objects_dir=None, users_path=None):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.users_path = Path(users_path).expanduser().resolve() if users_path is not None else self.path.with_name('users.sqlite')
        if self.path == self.users_path:
            raise LibraryError('User and contract databases must be separate files', 409)
        self.users = UserLibrary(self.users_path)
        self._migrate_accounts()
        self.objects_dir = Path(objects_dir).expanduser().resolve() if objects_dir is not None else self.path.with_name(self.path.stem + '_objects')
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(_SCHEMA)
            if db.execute('PRAGMA main.foreign_key_check').fetchone():
                raise LibraryError('Contract principal references are inconsistent', 409)
            if db.execute('SELECT 1 FROM main.users r LEFT JOIN accounts.users u ON u.user_id=r.user_id WHERE u.user_id IS NULL LIMIT 1').fetchone():
                raise LibraryError('Referenced accounts are missing from the user database', 409)
        # Uploaded objects are SQLite BLOBs; no user-controlled filesystem path.
        if os.name != 'nt':
            self.objects_dir.chmod(0o700)
            self.path.chmod(0o600)

    def _migrate_accounts(self):
        """Split a legacy combined database, retaining all principal IDs.

        Source writes are reserved while the separate identity transaction
        commits its replay receipt. A retry after either commit cannot restore
        revoked sessions or overwrite a password already reset centrally.
        """
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in db.execute('PRAGMA table_info(users)')}
            if not columns or columns == {'user_id'}:
                return
            if 'password_hash' not in columns:
                raise LibraryError('Unsupported local account schema', 409)
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('PRAGMA secure_delete=ON')
            db.execute('BEGIN IMMEDIATE')
            # A concurrent opener may have finished while we waited for the lock.
            columns = {row[1] for row in db.execute('PRAGMA table_info(users)')}
            if columns == {'user_id'}:
                db.commit()
                return
            self.users.migrate_contract_accounts(self.path, _source_db=db)
            db.execute('DROP TABLE IF EXISTS sessions')
            db.execute('CREATE TABLE principal_references(user_id TEXT PRIMARY KEY)')
            db.execute('INSERT INTO principal_references SELECT user_id FROM users')
            db.execute('DROP TABLE users')
            db.execute('ALTER TABLE principal_references RENAME TO users')
            if db.execute('PRAGMA foreign_key_check').fetchone():
                raise LibraryError('Legacy principal references are inconsistent', 409)
            db.commit()
            # Do not leave removed credentials in a pre-migration WAL snapshot.
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(str(self.path), timeout=30, factory=_LibraryConnection)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        db.execute('ATTACH DATABASE ? AS accounts', (str(self.users_path),))
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _tx(self, write=False):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            db.write_transaction = write
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    _actor = staticmethod(UserLibrary._actor)

    def _require(self, db, actor, admin=False):
        actor = UserLibrary.require(db, actor, admin, schema='accounts')
        return self._index_actor(db, actor)

    @staticmethod
    def _index_actor(db, actor):
        if db.write_transaction:
            db.execute('INSERT OR IGNORE INTO main.users(user_id) VALUES (?)', (actor['user_id'],))
        return actor

    @staticmethod
    def _event(db, actor, action, target=None, details=None):
        db.execute('INSERT INTO audit_events VALUES (?,?,?,?,?,?)',
                   (_id(), actor['user_id'] if actor else None, action, target, _now(), json.dumps(details or {})))

    def bootstrap_admin(self, username, password):
        return self.users.bootstrap_admin(username, password)

    def initialized(self):
        return self.users.initialized()

    def register(self, username, password):
        return self.users.register(username, password)

    def create_user(self, actor, username, password, role='user'):
        return self.users.create_user(actor, username, password, role)

    def list_users(self, actor):
        return self.users.list_users(actor)

    def update_user(self, actor, user_id, *, password=None, active=None, role=None):
        return self.users.update_user(actor, user_id, password=password, active=active, role=role)

    def login(self, username, password):
        return self.users.login(username, password)

    def authenticate(self, token):
        return self.users.authenticate(token)

    def validate_csrf(self, token, csrf_token):
        return self.users.validate_csrf(token, csrf_token)

    def logout(self, token):
        return self.users.logout(token)

    @staticmethod
    def _metadata(metadata):
        if metadata is None:
            return {}
        if not isinstance(metadata, dict) or set(metadata) - {'title', 'domain', 'region'}:
            raise LibraryError('Metadata permits title, domain and region only')
        if any(not isinstance(v, str) or len(v) > 2000 for v in metadata.values()):
            raise LibraryError('Metadata values must be strings up to 2000 characters')
        return dict(metadata)

    def _file(self, filename, content):
        if not isinstance(filename, str) or any(ord(c) < 32 or ord(c) == 127 for c in filename):
            raise LibraryError('Invalid filename')
        filename = filename.replace('\\', '/').rsplit('/', 1)[-1]
        file_type = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if not filename or len(filename) > 255 or file_type not in self.FILE_TYPES:
            raise LibraryError('Unsupported file type')
        if not isinstance(content, bytes) or not content:
            raise LibraryError('Nonempty bytes are required')
        if len(content) > self.MAX_UPLOAD:
            raise LibraryError('Upload exceeds 40 MiB', 413)
        return filename, file_type

    @staticmethod
    def _summary(db, row):
        item = dict(row)
        version = db.execute('SELECT filename,file_type,sha256,size FROM versions WHERE version_id=? AND contract_id=?', (row['current_version_id'], row['contract_id'])).fetchone()
        item.update(dict(version))
        item['version_count'] = db.execute('SELECT COUNT(*) FROM versions WHERE contract_id=?', (row['contract_id'],)).fetchone()[0]
        item['source_metadata'] = json.loads(item['source_metadata'])
        item['legacy_original_unavailable'] = bool(item['legacy_original_unavailable'])
        item['owner_user_id'] = item['owner']
        item['owner_username'] = db.execute('SELECT username FROM accounts.users WHERE user_id=?', (item['owner'],)).fetchone()[0]
        return item

    @staticmethod
    def _contract(db, actor, contract_id, allow_deleted=False):
        row = db.execute('SELECT * FROM contracts WHERE contract_id=?', (contract_id,)).fetchone()
        if not row or (row['owner'] != actor['user_id'] and actor['role'] != 'admin'):
            raise LibraryError('Contract not found', 404)
        if row['legacy_original_unavailable'] and actor['role'] != 'admin':
            raise LibraryError('Contract not found', 404)
        if row['deleted_at'] and not (allow_deleted and actor['role'] == 'admin'):
            raise LibraryError('Contract not found', 404)
        return row

    @staticmethod
    def _version(db, contract_id, version_id):
        row = db.execute('SELECT * FROM versions WHERE contract_id=? AND version_id=?', (contract_id, version_id)).fetchone()
        if not row:
            raise LibraryError('Version not found', 404)
        return dict(row)

    def _insert_contract(self, db, actor, filename, content, metadata, *, legacy=False, source=None):
        filename, file_type = self._file(filename, content)
        contract_id, version_id, created_at = _id(), _id(), _now()
        db.execute('INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?)', (contract_id, actor['user_id'], metadata.get('title') or filename,
                   metadata.get('domain', ''), metadata.get('region', ''), created_at, version_id, None,
                   json.dumps(source or {}, ensure_ascii=False), int(legacy)))
        db.execute('INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?)', (version_id, contract_id, 1, filename, file_type,
                   hashlib.sha256(content).hexdigest(), len(content), content, created_at, 'Legacy text snapshot; original upload unavailable' if legacy else ''))
        self._event(db, actor, 'contract.import' if legacy else 'contract.upload', contract_id)
        return self._summary(db, self._contract(db, actor, contract_id))

    def upload(self, actor, filename, content, *, metadata=None):
        metadata = self._metadata(metadata)
        with self._tx(True) as db:
            actor = self._require(db, actor)
            return self._insert_contract(db, actor, filename, content, metadata)

    @staticmethod
    def _pagination(limit, offset=0):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise LibraryError('limit must be between 1 and 500')
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise LibraryError('offset must be nonnegative')
        return limit, offset

    def list_contracts(self, actor, *, query='', limit=50, offset=0, include_deleted=False):
        limit, offset = self._pagination(limit, offset)
        if not isinstance(query, str) or len(query) > 2000:
            raise LibraryError('Invalid query')
        with self._tx() as db:
            actor = self._require(db, actor)
            where, params = [], []
            if actor['role'] != 'admin':
                where.extend(['c.owner=?', 'c.legacy_original_unavailable=0'])
                params.append(actor['user_id'])
            if actor['role'] != 'admin' or not include_deleted:
                where.append('c.deleted_at IS NULL')
            if query:
                where.append("(c.title LIKE ? ESCAPE '\\' OR c.domain LIKE ? ESCAPE '\\' OR c.region LIKE ? ESCAPE '\\' OR v.filename LIKE ? ESCAPE '\\')")
                term = '%' + query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
                params.extend([term] * 4)
            clause = ' WHERE ' + ' AND '.join(where) if where else ''
            base = ' FROM contracts c JOIN versions v ON v.version_id=c.current_version_id' + clause
            total = db.execute('SELECT COUNT(*)' + base, params).fetchone()[0]
            rows = db.execute('SELECT c.*' + base + ' ORDER BY c.created_at DESC,c.contract_id DESC LIMIT ? OFFSET ?', (*params, limit, offset))
            return {'items': [self._summary(db, row) for row in rows], 'total': total}

    def get_contract(self, actor, contract_id):
        with self._tx() as db:
            actor = self._require(db, actor)
            row = self._contract(db, actor, contract_id, True)
            item = self._summary(db, row)
            item['versions'] = [dict(v) for v in db.execute('SELECT version_id,version_no,filename,sha256,size,created_at,note FROM versions WHERE contract_id=? ORDER BY version_no', (contract_id,))]
            return item

    def get_version(self, actor, contract_id, version_id=None):
        with self._tx() as db:
            actor = self._require(db, actor)
            row = self._contract(db, actor, contract_id, True)
            return self._version(db, contract_id, version_id or row['current_version_id'])

    def update_contract(self, actor, contract_id, *, metadata=None, filename=None, content=None, note='', expected_version_id=None):
        metadata = self._metadata(metadata)
        if not isinstance(note, str) or len(note) > 10000:
            raise LibraryError('Invalid version note')
        with self._tx(True) as db:
            actor = self._require(db, actor, True)
            row = self._contract(db, actor, contract_id)
            if expected_version_id is not None and row['current_version_id'] != expected_version_id:
                raise LibraryError('Contract changed; refresh before saving', 409)
            if metadata:
                db.execute('UPDATE contracts SET ' + ','.join(k + '=?' for k in metadata) + ' WHERE contract_id=?', (*metadata.values(), contract_id))
            if filename is not None or content is not None:
                current = self._version(db, contract_id, row['current_version_id'])
                content = current['content'] if content is None else content
                filename, file_type = self._file(current['filename'] if filename is None else filename, content)
                version_id = _id()
                db.execute('INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?)', (version_id, contract_id, current['version_no'] + 1, filename,
                           file_type, hashlib.sha256(content).hexdigest(), len(content), content, _now(), note))
                db.execute('UPDATE contracts SET current_version_id=? WHERE contract_id=?', (version_id, contract_id))
            if metadata or filename is not None or content is not None:
                self._event(db, actor, 'contract.update', contract_id, {'metadata_fields': sorted(metadata), 'new_version': filename is not None or content is not None})
            return self._summary(db, self._contract(db, actor, contract_id))

    def delete_contract(self, actor, contract_id):
        with self._tx(True) as db:
            actor = self._require(db, actor, True)
            self._contract(db, actor, contract_id, True)
            db.execute('UPDATE contracts SET deleted_at=COALESCE(deleted_at,?) WHERE contract_id=?', (_now(), contract_id))
            db.execute("UPDATE review_tasks SET status='failed',error='Contract deleted',finished_at=?,updated_at=?,lease_expires_at=NULL WHERE contract_id=? AND status IN ('queued','running')", (_now(), _now(), contract_id))
            self._event(db, actor, 'contract.delete', contract_id)
            return {'deleted': True, 'contract_id': contract_id}

    @staticmethod
    def _json_object(value, label):
        if not isinstance(value, dict):
            raise LibraryError(label + ' must be an object')
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (ValueError, TypeError, OverflowError):
            raise LibraryError(label + ' must contain JSON values') from None

    def _save_report(self, db, actor, contract_id, version_id, payload):
        self._contract(db, actor, contract_id)
        self._version(db, contract_id, version_id)
        payload = self._json_object(payload, 'Report payload')
        report_id, created_at = _id(), _now()
        payload.update(library_contract_id=contract_id, contract_version_id=version_id, report_id=report_id, saved=True)
        risks = payload.get('risks', [])
        risk_count = payload.get('risk_count', len(risks) if isinstance(risks, list) else 0)
        risk_count = risk_count if isinstance(risk_count, int) and risk_count >= 0 else 0
        score = payload.get('score')
        score = score if isinstance(score, (int, float)) and not isinstance(score, bool) else None
        source_name = payload.get('source_name', '')
        source_name = source_name if isinstance(source_name, str) else ''
        db.execute('INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)', (report_id, contract_id, version_id, actor['user_id'], created_at,
                   json.dumps(payload, ensure_ascii=False, allow_nan=False), risk_count, score, source_name))
        self._event(db, actor, 'report.create', report_id, {'contract_id': contract_id, 'version_id': version_id})
        return {'report_id': report_id, 'contract_id': contract_id, 'version_id': version_id, 'created_at': created_at, 'payload': payload}

    def save_report(self, actor, contract_id, version_id, payload):
        with self._tx(True) as db:
            actor = self._require(db, actor)
            return self._save_report(db, actor, contract_id, version_id, payload)

    def list_reports(self, actor, contract_id=None, *, limit=50, offset=0):
        limit, offset = self._pagination(limit, offset)
        with self._tx() as db:
            actor = self._require(db, actor)
            where, params = [], []
            if actor['role'] != 'admin':
                where.extend(['c.owner=?', 'c.deleted_at IS NULL', 'c.legacy_original_unavailable=0'])
                params.append(actor['user_id'])
            if contract_id is not None:
                self._contract(db, actor, contract_id, True)
                where.append('r.contract_id=?')
                params.append(contract_id)
            base = ' FROM reports r JOIN contracts c ON c.contract_id=r.contract_id'
            if where:
                base += ' WHERE ' + ' AND '.join(where)
            total = db.execute('SELECT COUNT(*)' + base, params).fetchone()[0]
            items = [dict(row) for row in db.execute('SELECT r.report_id,r.contract_id,r.version_id,r.created_at,r.risk_count,r.score,r.source_name' + base + ' ORDER BY r.created_at DESC,r.report_id LIMIT ? OFFSET ?', (*params, limit, offset))]
            return {'items': items, 'total': total}

    def get_report(self, actor, report_id):
        with self._tx() as db:
            actor = self._require(db, actor)
            row = db.execute('SELECT report_id,contract_id,version_id,created_at,payload FROM reports WHERE report_id=?', (report_id,)).fetchone()
            if not row:
                raise LibraryError('Report not found', 404)
            self._contract(db, actor, row['contract_id'], True)
            item = dict(row)
            item['payload'] = json.loads(item['payload'])
            item['feedback'] = [dict(feedback) for feedback in db.execute(
                'SELECT *,created_at AS updated_at FROM administrator_feedback WHERE report_id=? ORDER BY rowid', (report_id,))]
            return item

    def save_feedback(self, actor, report_id, risk_id, status, note=''):
        """Append an administrator finding without mutating the report snapshot."""
        with self._tx(True) as db:
            actor = self._require(db, actor, True)
            if status not in ('confirmed', 'dismissed', 'pending'):
                raise LibraryError('Invalid feedback status')
            if not isinstance(note, str) or len(note) > 4000:
                raise LibraryError('Feedback note must be text up to 4000 characters')
            if not isinstance(risk_id, str) or not risk_id or len(risk_id) > 500:
                raise LibraryError('Risk not found', 404)
            report = db.execute('SELECT contract_id,payload FROM reports WHERE report_id=?', (report_id,)).fetchone()
            if not report:
                raise LibraryError('Report not found', 404)
            self._contract(db, actor, report['contract_id'], True)
            risks = json.loads(report['payload']).get('risks', [])
            if not isinstance(risks, list) or risk_id not in {
                    str(r.get('risk_id') or r.get('id') or '') for r in risks if isinstance(r, dict)}:
                raise LibraryError('Risk not found', 404)
            feedback_id = _id()
            db.execute('INSERT INTO administrator_feedback VALUES (?,?,?,?,?,?,?,?)',
                       (feedback_id, report_id, risk_id, status, note, actor['user_id'], actor['username'], _now()))
            self._event(db, actor, 'feedback.create', feedback_id, {'report_id': report_id, 'status': status})
            return {'saved': True, 'status': status}

    @staticmethod
    def _task(row):
        item = dict(row)
        item['options'] = json.loads(item['options'])
        return item

    def enqueue_review(self, actor, contract_id, version_id=None, *, options=None):
        options = self._json_object({} if options is None else options, 'Task options')
        with self._tx(True) as db:
            actor = self._require(db, actor)
            contract = self._contract(db, actor, contract_id)
            version_id = version_id or contract['current_version_id']
            self._version(db, contract_id, version_id)
            task_id, now = _id(), _now()
            db.execute("INSERT INTO review_tasks(task_id,contract_id,version_id,owner,created_by,status,options,created_at,updated_at) VALUES (?,?,?,?,?,'queued',?,?,?)",
                       (task_id, contract_id, version_id, contract['owner'], actor['user_id'], json.dumps(options, ensure_ascii=False), now, now))
            self._event(db, actor, 'task.enqueue', task_id, {'contract_id': contract_id, 'version_id': version_id})
            return self._task(db.execute('SELECT * FROM review_tasks WHERE task_id=?', (task_id,)).fetchone())

    def get_task(self, actor, task_id):
        with self._tx() as db:
            actor = self._require(db, actor)
            row = db.execute('SELECT * FROM review_tasks WHERE task_id=?', (task_id,)).fetchone()
            if not row:
                raise LibraryError('Task not found', 404)
            self._contract(db, actor, row['contract_id'], True)
            return self._task(row)

    def list_tasks(self, actor, *, limit=50):
        self._pagination(limit)
        with self._tx() as db:
            actor = self._require(db, actor)
            base, params = ' FROM review_tasks t JOIN contracts c ON c.contract_id=t.contract_id', []
            if actor['role'] != 'admin':
                base += ' WHERE c.owner=? AND c.deleted_at IS NULL AND c.legacy_original_unavailable=0'
                params.append(actor['user_id'])
            total = db.execute('SELECT COUNT(*)' + base, params).fetchone()[0]
            rows = db.execute('SELECT t.*' + base + ' ORDER BY t.created_at DESC,t.task_id LIMIT ?', (*params, limit))
            return {'items': [self._task(row) for row in rows], 'total': total}

    @staticmethod
    def _worker(worker_id, lease_seconds=300):
        if not isinstance(worker_id, str) or not worker_id or len(worker_id) > 200:
            raise LibraryError('A worker identity is required')
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or not 0 < lease_seconds <= 86400:
            raise LibraryError('Invalid task lease')

    @staticmethod
    def _task_actor(db, row):
        creator = db.execute('SELECT * FROM accounts.users WHERE user_id=?', (row['created_by'],)).fetchone()
        contract = db.execute('SELECT * FROM contracts WHERE contract_id=?', (row['contract_id'],)).fetchone()
        if not creator or not creator['active'] or not contract or contract['deleted_at'] or contract['owner'] != row['owner']:
            return None
        if creator['role'] != 'admin' and creator['user_id'] != contract['owner']:
            return None
        if creator['role'] != 'admin' and contract['legacy_original_unavailable']:
            return None
        return ContractLibrary._index_actor(db, ContractLibrary._actor(creator))

    def claim_task(self, worker_id, *, lease_seconds=300):
        self._worker(worker_id, lease_seconds)
        with self._tx(True) as db:
            while True:
                # A worker identity may own a given task only once. Reusing an
                # identity after expiry would make old and new completions
                # indistinguishable with this API's worker_id fence.
                row = db.execute("SELECT * FROM review_tasks t WHERE (status='queued' OR (status='running' AND lease_expires_at<=?)) AND NOT EXISTS (SELECT 1 FROM task_leases l WHERE l.task_id=t.task_id AND l.worker_id=?) ORDER BY created_at,task_id LIMIT 1", (_now(), worker_id)).fetchone()
                if not row:
                    return None
                actor = self._task_actor(db, row)
                if row['attempts'] >= 3 or actor is None:
                    db.execute("UPDATE review_tasks SET status='failed',error=?,finished_at=?,updated_at=?,lease_expires_at=NULL WHERE task_id=?",
                               ('Task attempts exhausted' if row['attempts'] >= 3 else 'Task authorization is no longer valid', _now(), _now(), row['task_id']))
                    self._event(db, None, 'task.failed', row['task_id'])
                    continue
                now = _now()
                expires = (datetime.fromisoformat(now) + timedelta(seconds=lease_seconds)).isoformat()
                db.execute('INSERT INTO task_leases VALUES (?,?)', (row['task_id'], worker_id))
                db.execute("UPDATE review_tasks SET status='running',worker_id=?,lease_expires_at=?,attempts=attempts+1,started_at=COALESCE(started_at,?),updated_at=?,error=NULL WHERE task_id=?",
                           (worker_id, expires, now, now, row['task_id']))
                result = self._task(db.execute('SELECT * FROM review_tasks WHERE task_id=?', (row['task_id'],)).fetchone())
                result['actor'] = actor
                result['version'] = self._version(db, row['contract_id'], row['version_id'])
                self._event(db, actor, 'task.claim', row['task_id'], {'attempts': result['attempts']})
                return result

    def renew_task(self, task_id, worker_id, *, lease_seconds=300):
        self._worker(worker_id, lease_seconds)
        with self._tx(True) as db:
            row = db.execute('SELECT * FROM review_tasks WHERE task_id=?', (task_id,)).fetchone()
            now = _now()
            if not row or row['status'] != 'running' or row['worker_id'] != worker_id or not row['lease_expires_at'] or row['lease_expires_at'] <= now or self._task_actor(db, row) is None:
                return False
            expires = (datetime.fromisoformat(now) + timedelta(seconds=lease_seconds)).isoformat()
            db.execute('UPDATE review_tasks SET lease_expires_at=?,updated_at=? WHERE task_id=?', (expires, now, task_id))
            return True

    def finish_task(self, task_id, worker_id, *, payload=None, error=None):
        self._worker(worker_id)
        with self._tx(True) as db:
            row = db.execute('SELECT * FROM review_tasks WHERE task_id=?', (task_id,)).fetchone()
            if not row or row['status'] != 'running' or row['worker_id'] != worker_id or not row['lease_expires_at'] or row['lease_expires_at'] <= _now():
                raise LibraryError('Task lease is no longer owned by this worker', 409)
            actor = self._task_actor(db, row)
            if actor is None:
                raise LibraryError('Task authorization is no longer valid', 409)
            if error is None:
                report = self._save_report(db, actor, row['contract_id'], row['version_id'], payload)
                status, report_id, error_text = 'completed', report['report_id'], None
            else:
                if not isinstance(error, str):
                    raise LibraryError('Task error must be text')
                status, report_id, error_text = 'failed', None, error[:2000]
            now = _now()
            db.execute('UPDATE review_tasks SET status=?,report_id=?,error=?,finished_at=?,updated_at=?,lease_expires_at=NULL WHERE task_id=?', (status, report_id, error_text, now, now, task_id))
            self._event(db, actor, 'task.' + status, task_id)
            return self._task(db.execute('SELECT * FROM review_tasks WHERE task_id=?', (task_id,)).fetchone())

    def events(self, actor, *, limit=100):
        self._pagination(limit)
        with self._tx() as db:
            self._require(db, actor, True)
            items = []
            for row in db.execute('''WITH merged AS (
                SELECT a.* FROM accounts.audit_events a
                UNION ALL
                SELECT m.* FROM main.audit_events m WHERE NOT EXISTS (
                    SELECT 1 FROM accounts.audit_events a WHERE a.event_id=m.event_id)
                ) SELECT * FROM merged ORDER BY created_at DESC,event_id LIMIT ?''', (limit,)):
                item = dict(row)
                item['details'] = json.loads(item['details'])
                items.append(item)
            return {'items': items}

    def import_legacy(self, actor, source_path):
        with self._tx() as db:
            self._require(db, actor, True)
        source = Path(source_path).expanduser().resolve()
        if not source.is_file() or source == self.path:
            raise LibraryError('A separate existing legacy SQLite file is required')
        try:
            legacy = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=30)
            legacy.row_factory = sqlite3.Row
            legacy.execute('PRAGMA query_only=ON')
            legacy.execute('BEGIN')
            tables = {r[0] for r in legacy.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
            sources = [(table, identity) for table, identity in (
                ('workspace_reports', 'report_id'), ('contracts', 'contract_id')) if table in tables]
            if not sources:
                raise LibraryError('Unsupported legacy database schema')
            for table, identity in sources:
                columns = {r[1] for r in legacy.execute('PRAGMA table_info(' + table + ')')}
                if identity not in columns or (table == 'workspace_reports' and 'payload' not in columns):
                    raise LibraryError('Unsupported legacy database schema')
            result = {'imported': 0, 'skipped': 0, 'errors': []}
            source_key = os.path.normcase(str(source))
            for table, identity in sources:
                for record in legacy.execute('SELECT * FROM ' + table):
                    source_id = str(record[identity])
                    try:
                        with self._tx(True) as db:
                            current_actor = self._require(db, actor, True)
                            if db.execute('SELECT 1 FROM legacy_imports WHERE source_path=? AND source_table=? AND source_id=?', (source_key, table, source_id)).fetchone():
                                result['skipped'] += 1
                                continue
                            if table == 'workspace_reports':
                                payload = json.loads(record['payload'])
                                if not isinstance(payload, dict):
                                    raise ValueError('Report payload is not an object')
                            else:
                                payload = dict(record)
                                payload['risks'] = [dict(r) for r in legacy.execute('SELECT * FROM risks WHERE contract_id=?', (record[identity],))] if 'risks' in tables else []
                                payload['citations'] = [dict(r) for r in legacy.execute('SELECT c.* FROM citations c JOIN risks r ON r.risk_id=c.risk_id WHERE r.contract_id=?', (record[identity],))] if {'risks', 'citations'} <= tables else []
                                payload['clauses'] = [dict(r) for r in legacy.execute('SELECT * FROM clause_fts WHERE contract_id=?', (record[identity],))] if 'clause_fts' in tables else []
                            # Keep a complete unmodified source snapshot even when
                            # the library adds its own report_id/saved bindings.
                            original_payload = self._json_object(payload, 'Legacy report payload')
                            document = payload.get('document')
                            full_text = document.get('full_text') if isinstance(document, dict) else None
                            if not isinstance(full_text, str) or not full_text:
                                full_text = payload.get('full_text')
                            if not isinstance(full_text, str) or not full_text:
                                full_text = '\n'.join(str(c.get('body', '')) for c in payload.get('clauses', []) if isinstance(c, dict))
                            content = (full_text or '[Legacy original upload and full text unavailable]').encode('utf-8')
                            original_name = payload.get('source_name')
                            original_name = original_name if isinstance(original_name, str) and original_name else 'Legacy contract'
                            source_meta = {'source_path': str(source), 'source_table': table, 'source_record_id': source_id,
                                           'source_name': original_name, 'original_upload_available': False, 'snapshot_full_text_available': bool(full_text)}
                            item = self._insert_contract(db, current_actor, 'legacy-snapshot.txt', content, {'title': original_name}, legacy=True, source=source_meta)
                            payload = dict(payload, legacy_source_payload=original_payload)
                            report = self._save_report(db, current_actor, item['contract_id'], item['current_version_id'], payload)
                            db.execute('INSERT INTO legacy_imports VALUES (?,?,?,?,?)', (source_key, table, source_id, item['contract_id'], report['report_id']))
                        result['imported'] += 1
                    except (LibraryError, ValueError, TypeError, sqlite3.DatabaseError) as exc:
                        # Record identifiers only; malformed payloads can contain secrets.
                        result['errors'].append({'source_id': source_id, 'error': type(exc).__name__})
            return result
        except sqlite3.DatabaseError:
            raise LibraryError('Legacy SQLite database could not be read') from None
        finally:
            if 'legacy' in locals():
                legacy.close()
