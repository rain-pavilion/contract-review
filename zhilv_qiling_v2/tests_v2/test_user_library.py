"""Isolated identity storage and restart-safe migration using real SQLite files."""
from contextlib import closing
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


class UserLibraryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('zhilv_qiling_v2.user_library'),
                             'Identity must have its own repository')
        module = importlib.import_module('zhilv_qiling_v2.user_library')
        self.Library, self.Error = module.UserLibrary, module.LibraryError
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'users.sqlite'
        self.users = self.Library(self.path)

    def denied(self, status, operation, *args, **kwargs):
        with self.assertRaises(self.Error) as caught:
            operation(*args, **kwargs)
        self.assertEqual(status, caught.exception.status)

    def bootstrap(self):
        return self.users.bootstrap_admin('admin', 'admin-test-password')

    def legacy(self, path, username='legacy-admin', user_id='original-user-id'):
        salt = b'legacy-test-salt!'
        password_hash = '1000$' + salt.hex() + '$' + hashlib.pbkdf2_hmac(
            'sha256', b'legacy-test-password', salt, 1000).hex()
        token = 'legacy-session-test-token'
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript('CREATE TABLE users(user_id TEXT PRIMARY KEY,username TEXT UNIQUE COLLATE NOCASE,'
                             'password_hash TEXT,role TEXT,active INTEGER,created_at TEXT);'
                             'CREATE TABLE sessions(token_hash TEXT PRIMARY KEY,user_id TEXT REFERENCES users(user_id),'
                             'csrf_token TEXT,created_at TEXT,expires_at TEXT);')
            db.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',
                       (user_id, username, password_hash, 'admin', 1, '2026-01-01'))
            db.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(), user_id, 'legacy-csrf', '2026-01-01', '2099-01-01'))
        return token

    def test_registration_requires_initialization_and_has_no_session_or_privileges(self):
        self.denied(409, self.users.register, 'first-user', 'ordinary-user-password')
        admin = self.bootstrap()
        user = self.users.register('注册用户_01', 'ordinary-user-password')
        self.assertEqual({'username': '注册用户_01', 'role': 'user', 'active': True},
                         {k: user[k] for k in ('username', 'role', 'active')})
        self.assertNotIn('token', user)
        self.denied(403, self.users.list_users, user)
        self.denied(403, self.users.create_user, dict(user, role='admin'), 'intruder', 'intruder-password')
        self.assertEqual(2, len(self.users.list_users(admin)['items']))
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(0, db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0])
            self.assertNotIn('ordinary-user-password', str(db.execute('SELECT * FROM users').fetchall()))
            self.assertTrue(db.execute("SELECT 1 FROM audit_events WHERE action='user.register'").fetchone())

    def test_registration_rejects_duplicates_weak_passwords_and_invalid_usernames(self):
        self.bootstrap()
        self.users.register('new-user', 'ordinary-user-password')
        self.denied(409, self.users.register, 'NEW-USER', 'another-user-password')
        self.denied(400, self.users.register, 'short-password', 'short')
        for username in ('a', 'x' * 65, ' spaced ', 'bad/name', 'bad@name', ''):
            self.denied(400, self.users.register, username, 'ordinary-user-password')
        self.assertEqual('user', self.users.login('new-user', 'ordinary-user-password')['user']['role'])

    def test_account_updates_revoke_sessions_and_protect_final_admin(self):
        admin = self.bootstrap()
        user = self.users.register('member', 'ordinary-user-password')
        session = self.users.login('member', 'ordinary-user-password')
        self.assertTrue(self.users.validate_csrf(session['token'], session['csrf_token']))
        self.users.update_user(admin, user['user_id'], password='replacement-password')
        self.denied(401, self.users.authenticate, session['token'])
        self.denied(401, self.users.login, 'member', 'ordinary-user-password')
        self.users.update_user(admin, user['user_id'], active=False)
        self.denied(401, self.users.login, 'member', 'replacement-password')
        self.denied(409, self.users.update_user, admin, admin['user_id'], active=False)
        self.denied(409, self.users.update_user, admin, admin['user_id'], role='user')

    def test_contracts_store_only_principal_ids_and_share_configured_users(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary, LibraryError
        self.assertIs(self.Error, LibraryError)
        contracts = ContractLibrary(self.root / 'contracts.sqlite', users_path=self.path)
        admin = contracts.bootstrap_admin('admin', 'admin-test-password')
        user = contracts.register('member', 'ordinary-user-password')
        item = contracts.upload(user, 'member.txt', b'private original')
        another = ContractLibrary(self.root / 'second.sqlite', users_path=self.path)
        self.assertTrue(another.initialized())
        self.assertEqual(user['user_id'], another.login('member', 'ordinary-user-password')['user']['user_id'])
        self.assertEqual('member', contracts.get_contract(admin, item['contract_id'])['owner_username'])
        with closing(sqlite3.connect(contracts.path)) as db:
            self.assertEqual(['user_id'], [r[1] for r in db.execute('PRAGMA table_info(users)')])
            self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone())
            self.assertIn((user['user_id'],), db.execute('SELECT * FROM users').fetchall())
            self.assertEqual([], db.execute('PRAGMA foreign_key_check').fetchall())

    def test_unused_accounts_are_not_copied_on_empty_worker_poll_or_another_users_write(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        store = ContractLibrary(self.root / 'lazy-references.sqlite', users_path=self.path)
        admin = store.bootstrap_admin('admin', 'admin-test-password')
        store.register('unused-member', 'ordinary-user-password')
        self.assertIsNone(store.claim_task('idle-worker'))
        self.assertEqual(0, store.list_contracts(admin)['total'])
        with closing(sqlite3.connect(store.path)) as db:
            self.assertEqual([], db.execute('SELECT * FROM users').fetchall())
        store.upload(admin, 'admin.txt', b'Only admin participates')
        with closing(sqlite3.connect(store.path)) as db:
            self.assertEqual([(admin['user_id'],)], db.execute('SELECT * FROM users').fetchall())

    def test_admin_events_merge_account_and_contract_audits_without_duplicate_ids(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        store = ContractLibrary(self.root / 'audited.sqlite', users_path=self.path)
        admin = store.bootstrap_admin('admin', 'admin-test-password')
        member = store.register('member', 'ordinary-user-password')
        store.update_user(admin, member['user_id'], password='replacement-password')
        store.upload(admin, 'audit.txt', b'Private original')
        events = store.events(admin)['items']
        self.assertTrue({'user.bootstrap', 'user.register', 'user.update', 'contract.upload'} <= {e['action'] for e in events})
        self.assertEqual('contract.upload', store.events(admin, limit=1)['items'][0]['action'])
        self.assertNotIn('replacement-password', json.dumps(events))
        self.denied(403, store.events, member)
        # A legacy audit copy with the same event ID must not duplicate the
        # canonical account event in the administrator's merged history.
        with closing(sqlite3.connect(store.path)) as db, db:
            db.execute('ATTACH DATABASE ? AS accounts', (str(self.path),))
            db.execute("INSERT INTO audit_events SELECT event_id,actor_id,action,target_id,created_at,'{}' FROM accounts.audit_events WHERE action='user.bootstrap'")
        merged = store.events(admin)['items']
        self.assertEqual(len(merged), len({e['event_id'] for e in merged}))
        self.assertEqual(1, sum(e['action'] == 'user.bootstrap' for e in merged))

    def test_legacy_accounts_sessions_and_ids_migrate_without_changing_ownership(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        source = self.root / 'old-contracts.sqlite'
        token = self.legacy(source)
        store = ContractLibrary(source, users_path=self.path)
        actor = store.authenticate(token)
        self.assertEqual('original-user-id', actor['user_id'])
        item = store.upload(actor, 'history.txt', b'Original ownership')
        self.assertEqual('original-user-id', item['owner'])
        self.assertEqual('original-user-id', store.login('legacy-admin', 'legacy-test-password')['user']['user_id'])
        with closing(sqlite3.connect(source)) as db:
            self.assertEqual(['user_id'], [r[1] for r in db.execute('PRAGMA table_info(users)')])
            self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone())
        self.assertNotIn(b'password_hash', source.read_bytes())
        self.assertNotIn(b'legacy-csrf', source.read_bytes())

    def test_completed_central_import_is_not_replayed_after_source_cleanup_interruption(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        source = self.root / 'interrupted.sqlite'
        token = self.legacy(source)
        # Simulate a committed central migration with the old source still intact.
        self.users.migrate_contract_accounts(source)
        actor = self.users.authenticate(token)
        self.users.update_user(actor, actor['user_id'], password='replacement-password')
        store = ContractLibrary(source, users_path=self.path)
        self.denied(401, store.login, 'legacy-admin', 'legacy-test-password')
        self.denied(401, store.authenticate, token)
        fresh = store.login('legacy-admin', 'replacement-password')
        self.assertEqual('original-user-id', fresh['user']['user_id'])
        self.assertEqual(1, len(store.list_users(store.authenticate(fresh['token']))['items']))

    def test_missing_or_incomplete_identity_database_fails_closed_for_existing_contracts(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        source = self.root / 'contracts.sqlite'
        store = ContractLibrary(source, users_path=self.path)
        admin = store.bootstrap_admin('admin', 'admin-test-password')
        user = store.register('member', 'ordinary-user-password')
        item = store.upload(user, 'owned.txt', b'Member private contract')
        missing = self.root / 'missing' / 'users.sqlite'
        self.denied(409, ContractLibrary, source, users_path=missing)
        incomplete = self.root / 'incomplete.sqlite'
        self.Library(incomplete).bootstrap_admin('other-admin', 'other-admin-password')
        self.denied(409, ContractLibrary, source, users_path=incomplete)
        self.assertEqual(user['user_id'], store.get_contract(admin, item['contract_id'])['owner'])

    def test_missing_local_principal_reference_cannot_hide_an_orphaned_contract(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        source = self.root / 'orphan.sqlite'
        store = ContractLibrary(source, users_path=self.path)
        admin = store.bootstrap_admin('admin', 'admin-test-password')
        store.upload(admin, 'original.txt', b'Old private contract')
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('DELETE FROM users')
        self.denied(409, ContractLibrary, source, users_path=self.root / 'wrong.sqlite')

    def test_business_history_and_all_foreign_keys_survive_legacy_account_split(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        source = self.root / 'history.sqlite'
        seed_path = self.root / 'seed-users.sqlite'
        old = ContractLibrary(source, users_path=seed_path)
        admin = old.bootstrap_admin('admin', 'admin-test-password')
        member = old.register('member', 'ordinary-user-password')
        session = old.login('member', 'ordinary-user-password')
        contract = old.upload(member, 'original.txt', b'Full original bytes')
        revised = old.update_contract(admin, contract['contract_id'], content=b'Revised bytes')
        report = old.save_report(member, contract['contract_id'], contract['current_version_id'], {'risks': [{'risk_id': 'r1'}]})
        old.save_feedback(admin, report['report_id'], 'r1', 'confirmed', 'Preserve this feedback')
        task = old.enqueue_review(member, contract['contract_id'], contract['current_version_id'])
        # Construct the actual previous combined schema without changing any IDs
        # or business tables. This is test-only simulated upgrade input.
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('ATTACH DATABASE ? AS seeded', (str(seed_path),))
            db.execute('CREATE TABLE legacy_accounts AS SELECT * FROM seeded.users')
            db.execute('DROP TABLE users')
            db.execute('ALTER TABLE legacy_accounts RENAME TO users')
            db.execute('CREATE TABLE sessions AS SELECT * FROM seeded.sessions')
        migrated = ContractLibrary(source, users_path=self.path)
        actor = migrated.authenticate(session['token'])
        self.assertEqual(member['user_id'], actor['user_id'])
        self.assertEqual(b'Full original bytes', migrated.get_version(actor, contract['contract_id'], contract['current_version_id'])['content'])
        self.assertEqual(revised['current_version_id'], migrated.get_contract(actor, contract['contract_id'])['current_version_id'])
        saved = migrated.get_report(actor, report['report_id'])
        self.assertEqual(report['payload'], saved['payload'])
        self.assertEqual('Preserve this feedback', saved['feedback'][0]['note'])
        claimed = migrated.claim_task('upgrade-worker')
        self.assertEqual(task['task_id'], claimed['task_id'])
        self.assertEqual(member['user_id'], claimed['actor']['user_id'])
        self.assertEqual(b'Full original bytes', claimed['version']['content'])
        with closing(sqlite3.connect(source)) as db:
            self.assertEqual([], db.execute('PRAGMA foreign_key_check').fetchall())
        self.users.update_user(admin, member['user_id'], active=False)
        self.denied(401, migrated.get_contract, actor, contract['contract_id'])
        self.denied(409, migrated.finish_task, task['task_id'], 'upgrade-worker', payload={})

    def test_identity_collisions_refuse_migration_and_preserve_source_credentials(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        admin = self.bootstrap()
        source = self.root / 'collision.sqlite'
        self.legacy(source, username='ADMIN')
        before = source.read_bytes()
        self.denied(409, ContractLibrary, source, users_path=self.path)
        self.assertEqual(before, source.read_bytes())
        self.assertEqual(admin['user_id'], self.users.login('admin', 'admin-test-password')['user']['user_id'])
        with closing(sqlite3.connect(source)) as db:
            self.assertIn('password_hash', [r[1] for r in db.execute('PRAGMA table_info(users)')])
        by_id = self.root / 'id-collision.sqlite'
        self.legacy(by_id, username='different-name', user_id=admin['user_id'])
        before = by_id.read_bytes()
        self.denied(409, ContractLibrary, by_id, users_path=self.path)
        self.assertEqual(before, by_id.read_bytes())


if __name__ == '__main__':
    unittest.main()
