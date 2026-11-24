"""Central accounts, revocable sessions, registration and migration receipts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import re
import sqlite3
import uuid


class LibraryError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id():
    return str(uuid.uuid4())


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
 user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
 password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','admin')),
 active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
 token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
 csrf_token TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS session_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS audit_events(
 event_id TEXT PRIMARY KEY, actor_id TEXT REFERENCES users(user_id), action TEXT NOT NULL,
 target_id TEXT, created_at TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS account_migrations(
 source_path TEXT PRIMARY KEY, principals TEXT NOT NULL, imported_at TEXT NOT NULL);
"""


class UserLibrary:
    PASSWORD_ITERATIONS = 260_000

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(_SCHEMA)
        if os.name != 'nt':
            self.path.chmod(0o600)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _tx(self, write=False):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _actor(row):
        return {key: (bool(row[key]) if key == 'active' else row[key])
                for key in ('user_id', 'username', 'role', 'active')}

    @classmethod
    def require(cls, db, actor, admin=False, *, schema="main"):
        if schema not in ("main", "accounts"):
            raise ValueError("Invalid account schema")
        if not isinstance(actor, dict) or not isinstance(actor.get('user_id'), str):
            raise LibraryError('Authentication required', 401)
        row = db.execute('SELECT * FROM ' + schema + '.users WHERE user_id=?', (actor['user_id'],)).fetchone()
        if not row or not row['active']:
            raise LibraryError('Authentication required', 401)
        if '_session_hash' in actor:
            session = db.execute('SELECT user_id,expires_at FROM ' + schema + '.sessions WHERE token_hash=?', (actor['_session_hash'],)).fetchone()
            if not session or session['user_id'] != row['user_id'] or session['expires_at'] <= _now():
                raise LibraryError('Authentication required', 401)
        if admin and row['role'] != 'admin':
            raise LibraryError('Administrator required', 403)
        return cls._actor(row)

    def _require(self, db, actor, admin=False):
        return self.require(db, actor, admin)

    @classmethod
    def _password(cls, password):
        if not isinstance(password, str) or not 12 <= len(password) <= 4096:
            raise LibraryError('Password must contain at least 12 characters')
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, cls.PASSWORD_ITERATIONS)
        return f'{cls.PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}'

    @staticmethod
    def _verify(password, stored):
        if not isinstance(password, str) or len(password) > 4096:
            return False
        iterations, salt, expected = stored.split('$')
        actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual, bytes.fromhex(expected))

    @staticmethod
    def _username(username):
        if not isinstance(username, str) or re.fullmatch(r'[A-Za-z0-9_\u4e00-\u9fff.\-]{2,64}', username) is None:
            raise LibraryError('Username must contain 2-64 Chinese letters, letters, digits, dots, underscores or hyphens')
        return username

    @staticmethod
    def _role(role):
        if role not in ('user', 'admin'):
            raise LibraryError('Invalid role')
        return role

    @staticmethod
    def _event(db, actor, action, target=None, details=None):
        db.execute('INSERT INTO audit_events VALUES (?,?,?,?,?,?)',
                   (_id(), actor['user_id'] if actor else None, action, target, _now(), json.dumps(details or {})))

    def _create_user(self, db, username, password, role):
        username, role = self._username(username), self._role(role)
        password_hash = self._password(password)
        user_id = _id()
        try:
            db.execute('INSERT INTO users VALUES (?,?,?,?,?,?)', (user_id, username, password_hash, role, 1, _now()))
        except sqlite3.IntegrityError:
            raise LibraryError('Username already exists', 409) from None
        return self._actor(db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone())

    def bootstrap_admin(self, username, password):
        with self._tx(True) as db:
            if db.execute('SELECT 1 FROM users LIMIT 1').fetchone():
                raise LibraryError('Administrator already initialized', 409)
            user = self._create_user(db, username, password, 'admin')
            self._event(db, user, 'user.bootstrap', user['user_id'])
            return user

    def initialized(self):
        """Public bootstrap status; contains no account or credential data."""
        with self._tx() as db:
            return db.execute('SELECT 1 FROM users LIMIT 1').fetchone() is not None

    def create_user(self, actor, username, password, role='user'):
        with self._tx(True) as db:
            actor = self._require(db, actor, True)
            user = self._create_user(db, username, password, role)
            self._event(db, actor, 'user.create', user['user_id'])
            return user

    def list_users(self, actor):
        with self._tx() as db:
            self._require(db, actor, True)
            return {'items': [self._actor(row) for row in db.execute('SELECT * FROM users ORDER BY created_at,user_id')]}

    def update_user(self, actor, user_id, *, password=None, active=None, role=None):
        with self._tx(True) as db:
            actor = self._require(db, actor, True)
            row = db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone()
            if not row:
                raise LibraryError('User not found', 404)
            if active is not None and not isinstance(active, bool):
                raise LibraryError('active must be boolean')
            new_role = row['role'] if role is None else self._role(role)
            new_active = row['active'] if active is None else int(active)
            if row['active'] and row['role'] == 'admin' and (not new_active or new_role != 'admin'):
                if db.execute("SELECT COUNT(*) FROM users WHERE active=1 AND role='admin'").fetchone()[0] <= 1:
                    raise LibraryError('Cannot remove the last active administrator', 409)
            password_hash = row['password_hash'] if password is None else self._password(password)
            changed = password is not None or new_role != row['role'] or new_active != row['active']
            db.execute('UPDATE users SET password_hash=?,active=?,role=? WHERE user_id=?', (password_hash, new_active, new_role, user_id))
            if changed:
                db.execute('DELETE FROM sessions WHERE user_id=?', (user_id,))
                self._event(db, actor, 'user.update', user_id, {'password_changed': password is not None, 'active': bool(new_active), 'role': new_role})
            return self._actor(db.execute('SELECT * FROM users WHERE user_id=?', (user_id,)).fetchone())

    def login(self, username, password):
        with self._tx(True) as db:
            row = db.execute('SELECT * FROM users WHERE username=? COLLATE NOCASE', (username if isinstance(username, str) else '',)).fetchone()
            # Missing users still incur PBKDF2 work and receive the same response.
            stored = row['password_hash'] if row else f"{self.PASSWORD_ITERATIONS}${'00' * 16}${'00' * 32}"
            valid = self._verify(password, stored)
            if not row or not row['active'] or not valid:
                raise LibraryError('Invalid username or password', 401)
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            created = datetime.now(timezone.utc)
            expires = (created + timedelta(hours=24)).isoformat()
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)', (_hash(token), row['user_id'], csrf, created.isoformat(), expires))
            actor = self._actor(row)
            self._event(db, actor, 'session.login', row['user_id'])
            return {'token': token, 'csrf_token': csrf, 'user': actor, 'expires_at': expires}

    def _authenticate(self, db, token):
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise LibraryError('Authentication required', 401)
        hashed = _hash(token)
        row = db.execute('SELECT u.*,s.csrf_token,s.expires_at FROM sessions s JOIN users u ON u.user_id=s.user_id WHERE s.token_hash=?', (hashed,)).fetchone()
        if not row or not row['active'] or row['expires_at'] <= _now():
            raise LibraryError('Authentication required', 401)
        return dict(self._actor(row), csrf_token=row['csrf_token'], _session_hash=hashed)

    def authenticate(self, token):
        with self._tx() as db:
            return self._authenticate(db, token)

    def validate_csrf(self, token, csrf_token):
        with self._tx() as db:
            actor = self._authenticate(db, token)
            return isinstance(csrf_token, str) and csrf_token.isascii() and hmac.compare_digest(actor['csrf_token'], csrf_token)

    def logout(self, token):
        if not isinstance(token, str):
            return
        with self._tx(True) as db:
            row = db.execute('SELECT user_id FROM sessions WHERE token_hash=?', (_hash(token),)).fetchone()
            db.execute('DELETE FROM sessions WHERE token_hash=?', (_hash(token),))
            if row:
                self._event(db, {'user_id': row['user_id']}, 'session.logout', row['user_id'])

    def register(self, username, password):
        with self._tx(True) as db:
            if not db.execute("SELECT 1 FROM users WHERE role='admin' AND active=1 LIMIT 1").fetchone():
                raise LibraryError('Administrator initialization required', 409)
            user = self._create_user(db, username, password, 'user')
            self._event(db, user, 'user.register', user['user_id'])
            return user

    def migrate_contract_accounts(self, source_path, *, _source_db=None):
        """Commit credentials and their one-time receipt before source cleanup.

        ContractLibrary supplies its locked source connection so a source writer
        cannot change credentials between this commit and credential removal.
        With no supplied connection this imports a readonly source snapshot.
        """
        source = Path(source_path).expanduser().resolve()
        if source == self.path or not source.is_file():
            raise LibraryError('A separate existing account source is required', 409)
        if _source_db is None:
            db = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                db.execute('PRAGMA query_only=ON')
                db.execute('BEGIN')
                return self.migrate_contract_accounts(source, _source_db=db)
            finally:
                db.close()
        source_db = _source_db
        required = {'user_id', 'username', 'password_hash', 'role', 'active', 'created_at'}
        if not required <= {r[1] for r in source_db.execute('PRAGMA table_info(users)')}:
            raise LibraryError('Unsupported legacy account schema', 409)
        users = [dict(row) for row in source_db.execute('SELECT user_id,username,password_hash,role,active,created_at FROM users')]
        principals = {row['user_id']: row['username'] for row in users}
        source_key = os.path.normcase(str(source))
        with self._tx(True) as db:
            receipt = db.execute('SELECT principals FROM account_migrations WHERE source_path=?', (source_key,)).fetchone()
            if receipt:
                if json.loads(receipt['principals']) != principals:
                    raise LibraryError('Account source changed after migration', 409)
                for user_id, username in principals.items():
                    current = db.execute('SELECT username FROM users WHERE user_id=?', (user_id,)).fetchone()
                    if not current or current['username'] != username:
                        raise LibraryError('Migrated account identity is missing or conflicting', 409)
                return {'imported': 0, 'skipped': len(users)}
            # Preflight the whole set so no collision silently maps old ownership
            # to an unrelated account. Existing credentials are never overwritten.
            for user in users:
                current = db.execute('SELECT * FROM users WHERE user_id=? OR username=? COLLATE NOCASE',
                                     (user['user_id'], user['username'])).fetchall()
                if current:
                    raise LibraryError('Account identity collision; legacy source was retained', 409)
                if user['role'] not in ('admin', 'user') or user['active'] not in (0, 1):
                    raise LibraryError('Invalid legacy account state', 409)
            for user in users:
                db.execute('INSERT INTO users VALUES (?,?,?,?,?,?)', tuple(user[k] for k in
                           ('user_id', 'username', 'password_hash', 'role', 'active', 'created_at')))
            if source_db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone():
                for session in source_db.execute('SELECT token_hash,user_id,csrf_token,created_at,expires_at FROM sessions'):
                    if session['user_id'] not in principals or db.execute('SELECT 1 FROM sessions WHERE token_hash=?', (session['token_hash'],)).fetchone():
                        raise LibraryError('Legacy session identity collision', 409)
                    db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)', tuple(session))
            db.execute('INSERT INTO account_migrations VALUES (?,?,?)',
                       (source_key, json.dumps(principals, ensure_ascii=False, sort_keys=True), _now()))
            self._event(db, None, 'accounts.migrate', details={'user_count': len(users)})
            return {'imported': len(users), 'skipped': 0}

