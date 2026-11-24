"""Behavioral security and persistence tests using real temporary SQLite files."""
import concurrent.futures
from contextlib import closing
import importlib
import importlib.util
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


class ContractLibraryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('zhilv_qiling_v2.contract_library'),
                             'The contract repository must exist')
        module = importlib.import_module('zhilv_qiling_v2.contract_library')
        self.Error = module.LibraryError
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'private' / 'contracts.sqlite'
        self.Library = module.ContractLibrary
        self.lib = self.Library(self.path)
        self.admin = self.lib.bootstrap_admin('admin', 'admin-test-password')
        self.alice = self.lib.create_user(self.admin, 'alice', 'alice-test-password')
        self.bob = self.lib.create_user(self.admin, 'bob', 'bobby-test-password')

    def denied(self, status, operation, *args, **kwargs):
        with self.assertRaises(self.Error) as caught:
            operation(*args, **kwargs)
        self.assertEqual(status, caught.exception.status)

    def upload(self, actor=None):
        return self.lib.upload(actor or self.alice, 'sample.txt', b'original contract',
                               metadata={'title': 'Sample', 'domain': 'supply', 'region': 'CN'})

    def expire_task(self, task_id):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE review_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE task_id=?", (task_id,))

    def test_bootstrap_password_policy_and_last_admin_protection(self):
        self.denied(409, self.lib.bootstrap_admin, 'other', 'another-password')
        self.denied(400, self.lib.create_user, self.admin, 'weak', 'short')
        self.denied(409, self.lib.update_user, self.admin, self.admin['user_id'], active=False)
        self.denied(409, self.lib.update_user, self.admin, self.admin['user_id'], role='user')
        self.denied(403, self.lib.create_user, self.alice, 'intruder', 'intruder-password')
        self.denied(403, self.lib.list_users, self.alice)
        self.assertEqual(3, len(self.lib.list_users(self.admin)['items']))

    def test_initialization_status_does_not_expose_user_data(self):
        empty = self.Library(Path(self.temp.name) / 'empty' / 'contracts.sqlite')
        self.assertIs(False, empty.initialized())
        self.assertIs(True, self.lib.initialized())

    def test_login_generic_errors_hashed_tokens_csrf_and_logout(self):
        self.denied(401, self.lib.login, 'missing', 'wrong-password')
        self.denied(401, self.lib.login, 'alice', 'wrong-password')
        session = self.lib.login('alice', 'alice-test-password')
        actor = self.lib.authenticate(session['token'])
        self.assertEqual(self.alice['user_id'], actor['user_id'])
        self.assertEqual(session['csrf_token'], actor['csrf_token'])
        self.assertTrue(self.lib.validate_csrf(session['token'], session['csrf_token']))
        self.assertFalse(self.lib.validate_csrf(session['token'], 'wrong'))
        with closing(sqlite3.connect(self.lib.users_path)) as db, db:
            text = '\n'.join(str(tuple(row)) for row in db.execute('SELECT * FROM sessions'))
        self.assertNotIn(session['token'], text)
        self.assertNotIn('alice-test-password', self.path.read_bytes().decode('latin1'))
        self.lib.logout(session['token'])
        self.denied(401, self.lib.authenticate, session['token'])
        self.denied(401, self.lib.list_contracts, actor)

    def test_expired_sessions_and_arbitrary_csrf_are_rejected(self):
        session = self.lib.login('alice', 'alice-test-password')
        self.assertFalse(self.lib.validate_csrf(session['token'], '\u4e0d\u6b63\u786e'))
        with closing(sqlite3.connect(self.lib.users_path)) as db, db:
            db.execute("UPDATE sessions SET expires_at='2000-01-01T00:00:00+00:00'")
        self.denied(401, self.lib.authenticate, session['token'])
        self.denied(401, self.lib.validate_csrf, session['token'], session['csrf_token'])

    def test_password_role_and_disabled_users_revoke_sessions(self):
        for changes in ({'password': 'new-alice-password'}, {'role': 'admin'}, {'active': False}):
            password = 'new-alice-password' if changes != {'password': 'new-alice-password'} else 'alice-test-password'
            session = self.lib.login('alice', password)
            self.lib.update_user(self.admin, self.alice['user_id'], **changes)
            self.denied(401, self.lib.authenticate, session['token'])
        self.denied(401, self.lib.list_contracts, self.alice)

    def test_owners_are_isolated_and_database_role_overrides_actor_claim(self):
        a = self.upload()
        b = self.upload(self.bob)
        forged = dict(self.alice, role='admin')
        self.assertEqual([a['contract_id']], [x['contract_id'] for x in self.lib.list_contracts(forged)['items']])
        for operation, args in ((self.lib.get_contract, (b['contract_id'],)),
                                (self.lib.get_version, (b['contract_id'],)),
                                (self.lib.enqueue_review, (b['contract_id'],))):
            self.denied(404, operation, self.alice, *args)
        self.denied(403, self.lib.update_contract, forged, a['contract_id'], metadata={'title': 'changed'})
        self.denied(403, self.lib.delete_contract, self.alice, a['contract_id'])
        self.denied(403, self.lib.events, self.alice)
        self.assertEqual(2, self.lib.list_contracts(self.admin)['total'])

    def test_upload_limits_safe_filename_no_owner_override_and_no_blob_summaries(self):
        self.denied(400, self.lib.upload, self.alice, 'empty.txt', b'')
        self.denied(400, self.lib.upload, self.alice, 'run.exe', b'a')
        self.denied(400, self.lib.upload, self.alice, 'bad\r\nheader.txt', b'a')
        self.denied(413, self.lib.upload, self.alice, 'huge.txt', b'a' * (40 * 1024 * 1024 + 1))
        self.denied(400, self.lib.upload, self.alice, 'good.txt', b'a', metadata={'owner': self.bob['user_id']})
        item = self.lib.upload(self.alice, '../../escape.txt', b'ok')
        version = self.lib.get_version(self.alice, item['contract_id'])
        self.assertEqual('escape.txt', version['filename'])
        self.assertEqual(self.alice['user_id'], item['owner'])
        self.assertEqual(self.alice['user_id'], item['owner_user_id'])
        self.assertEqual('alice', item['owner_username'])
        self.assertNotIn('content', item)
        self.assertEqual(1, item['version_count'])
        self.assertEqual(1, self.Library(self.path).list_contracts(self.alice, query='escape')['total'])

    def test_admin_versions_optimistic_conflict_and_audit_preserve_original(self):
        item = self.upload()
        original = item['current_version_id']
        changed = self.lib.update_contract(self.admin, item['contract_id'], filename='revision.md',
                    content=b'revised contract', note='Reviewed', expected_version_id=original)
        self.assertEqual(2, changed['version_count'])
        self.assertEqual(b'original contract', self.lib.get_version(self.alice, item['contract_id'], original)['content'])
        self.assertEqual(b'revised contract', self.lib.get_version(self.alice, item['contract_id'])['content'])
        self.denied(409, self.lib.update_contract, self.admin, item['contract_id'], metadata={'title': 'lost'}, expected_version_id=original)
        updated = self.lib.update_contract(self.admin, item['contract_id'], metadata={'title': 'Renamed'})
        self.assertEqual(2, updated['version_count'])
        self.assertEqual(2, len(self.lib.get_contract(self.alice, item['contract_id'])['versions']))
        self.denied(400, self.lib.update_contract, self.admin, item['contract_id'], metadata={'owner': self.bob['user_id']})
        events = self.lib.events(self.admin)['items']
        self.assertTrue(any(e['action'] == 'contract.update' for e in events))
        self.assertNotIn('revised contract', json.dumps(events))

    def test_reports_are_create_only_bound_to_version_and_hidden_after_deletion(self):
        item = self.upload()
        payload = {'contract_id': 'engine-stable-id', 'source_name': 'sample.txt', 'score': 72, 'risks': [{'severity': 'high'}]}
        saved = self.lib.save_report(self.alice, item['contract_id'], item['current_version_id'], payload)
        self.assertNotIn('saved', payload)
        self.assertEqual('engine-stable-id', saved['payload']['contract_id'])
        self.assertEqual(item['contract_id'], saved['payload']['library_contract_id'])
        self.assertEqual(item['current_version_id'], saved['payload']['contract_version_id'])
        second = self.lib.save_report(self.alice, item['contract_id'], item['current_version_id'], payload)
        self.assertNotEqual(saved['report_id'], second['report_id'])
        self.assertNotIn('payload', self.lib.list_reports(self.alice)['items'][0])
        self.denied(404, self.lib.get_report, self.bob, saved['report_id'])
        self.denied(404, self.lib.save_report, self.bob, item['contract_id'], item['current_version_id'], {})
        self.lib.delete_contract(self.admin, item['contract_id'])
        self.denied(404, self.lib.get_contract, self.alice, item['contract_id'])
        self.denied(404, self.lib.get_version, self.alice, item['contract_id'])
        self.denied(404, self.lib.get_report, self.alice, saved['report_id'])
        self.assertEqual(0, self.lib.list_contracts(self.alice, include_deleted=True)['total'])
        self.assertEqual(1, self.lib.list_contracts(self.admin, include_deleted=True)['total'])
        self.assertEqual(2, self.lib.list_reports(self.admin)['total'])
        self.assertEqual('engine-stable-id', self.lib.get_report(self.admin, saved['report_id'])['payload']['contract_id'])

    def test_durable_tasks_lease_transfer_and_atomic_report_finish(self):
        item = self.upload()
        task = self.lib.enqueue_review(self.alice, item['contract_id'], options={'mode': 'local'})
        self.denied(404, self.lib.get_task, self.bob, task['task_id'])
        self.assertNotIn('content', task)
        restarted = self.Library(self.path)
        first = restarted.claim_task('worker-one')
        self.assertEqual(b'original contract', first['version']['content'])
        self.assertEqual(self.alice['user_id'], first['actor']['user_id'])
        self.assertEqual(1, first['attempts'])
        self.assertIsNone(self.lib.claim_task('worker-two'))
        self.assertFalse(self.lib.renew_task(task['task_id'], 'worker-two'))
        self.assertTrue(self.lib.renew_task(task['task_id'], 'worker-one'))
        self.expire_task(task['task_id'])
        second = self.lib.claim_task('worker-two')
        self.assertEqual(2, second['attempts'])
        self.denied(409, self.lib.finish_task, task['task_id'], 'worker-one', payload={'score': 0})
        finished = self.lib.finish_task(task['task_id'], 'worker-two', payload={'score': 81, 'contract_id': 'engine-id'})
        self.assertEqual('completed', finished['status'])
        self.assertEqual(1, self.lib.list_reports(self.alice)['total'])
        self.assertEqual(finished['report_id'], self.lib.list_reports(self.alice)['items'][0]['report_id'])
        self.denied(409, self.lib.finish_task, task['task_id'], 'worker-two', payload={})

    def test_task_attempt_limit_failure_and_invalidated_creator_or_contract(self):
        item = self.upload()
        self.denied(400, self.lib.enqueue_review, self.alice, item['contract_id'], options=[])
        task = self.lib.enqueue_review(self.alice, item['contract_id'])
        for n in range(3):
            claimed = self.lib.claim_task('worker-' + str(n))
            self.assertEqual(n + 1, claimed['attempts'])
            self.expire_task(task['task_id'])
        self.assertIsNone(self.lib.claim_task('worker-last'))
        self.assertEqual('failed', self.lib.get_task(self.alice, task['task_id'])['status'])
        task2 = self.lib.enqueue_review(self.alice, item['contract_id'])
        self.lib.update_user(self.admin, self.alice['user_id'], active=False)
        self.assertIsNone(self.lib.claim_task('disabled-user-worker'))
        self.assertEqual('failed', self.lib.get_task(self.admin, task2['task_id'])['status'])
        other = self.upload(self.bob)
        task3 = self.lib.enqueue_review(self.bob, other['contract_id'])
        self.lib.claim_task('delete-worker')
        self.lib.delete_contract(self.admin, other['contract_id'])
        self.denied(409, self.lib.finish_task, task3['task_id'], 'delete-worker', payload={})

    def test_concurrent_claims_have_single_winner(self):
        item = self.upload()
        self.lib.enqueue_review(self.alice, item['contract_id'])
        def claim(n):
            return self.Library(self.path).claim_task('worker-' + str(n))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            winners = [x for x in pool.map(claim, range(4)) if x is not None]
        self.assertEqual(1, len(winners))

    def test_claim_and_renew_lease_start_after_database_lock_wait(self):
        item = self.upload()
        task = self.lib.enqueue_review(self.alice, item['contract_id'])
        for operation in ('claim', 'renew'):
            with self.subTest(operation=operation):
                with closing(sqlite3.connect(self.path)) as blocker:
                    blocker.execute('BEGIN IMMEDIATE')
                    ready = threading.Event()
                    def wait_for_lock():
                        ready.set()
                        if operation == 'claim':
                            return self.lib.claim_task('blocked-worker', lease_seconds=0.2)
                        return self.lib.renew_task(task['task_id'], 'blocked-worker', lease_seconds=0.2)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(wait_for_lock)
                        self.assertTrue(ready.wait(1))
                        time.sleep(0.35)
                        blocker.commit()
                        result = future.result(timeout=3)
                self.assertTrue(result)
                current = self.lib.get_task(self.alice, task['task_id'])
                expires = datetime.fromisoformat(current['lease_expires_at'])
                self.assertGreater((expires - datetime.now(timezone.utc)).total_seconds(), 0.08)
                # Keep the old lease valid while the following renewal waits.
                if operation == 'claim':
                    self.assertTrue(self.lib.renew_task(task['task_id'], 'blocked-worker', lease_seconds=5))

    def test_feedback_is_admin_only_append_only_and_visible_to_report_owner(self):
        item = self.upload()
        report = self.lib.save_report(self.alice, item['contract_id'], item['current_version_id'],
                                      {'risks': [{'risk_id': 'r-one', 'message': 'Review this'}]})
        original_payload = report['payload']
        self.denied(403, self.lib.save_feedback, self.alice, report['report_id'], 'r-one', 'confirmed')
        self.denied(400, self.lib.save_feedback, self.admin, report['report_id'], 'r-one', 'invalid')
        self.denied(400, self.lib.save_feedback, self.admin, report['report_id'], 'r-one', 'confirmed', 'x' * 4001)
        self.denied(404, self.lib.save_feedback, self.admin, report['report_id'], 'missing', 'confirmed')
        self.denied(404, self.lib.save_feedback, self.admin, 'missing-report', 'r-one', 'confirmed')
        self.assertEqual({'saved': True, 'status': 'confirmed'}, self.lib.save_feedback(
            self.admin, report['report_id'], 'r-one', 'confirmed', 'Administrator finding'))
        self.lib.save_feedback(self.admin, report['report_id'], 'r-one', 'dismissed', 'Reconsidered')
        visible = self.lib.get_report(self.alice, report['report_id'])
        self.assertEqual(original_payload, visible['payload'])
        self.assertEqual(['confirmed', 'dismissed'], [f['status'] for f in visible['feedback']])
        self.assertEqual('Administrator finding', visible['feedback'][0]['note'])
        self.denied(404, self.lib.get_report, self.bob, report['report_id'])
        with closing(sqlite3.connect(self.path)) as db, db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE administrator_feedback SET status='pending'")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('DELETE FROM administrator_feedback')

    def test_feedback_rechecks_session_and_admin_after_write_lock_wait(self):
        item = self.upload()
        report = self.lib.save_report(self.alice, item['contract_id'], item['current_version_id'],
                                      {'risks': [{'risk_id': 'r-one'}]})
        self.lib.create_user(self.admin, 'second-admin', 'second-admin-password', role='admin')
        session = self.lib.login('admin', 'admin-test-password')
        authenticated = self.lib.authenticate(session['token'])
        with closing(sqlite3.connect(self.lib.users_path)) as blocker:
            blocker.execute('BEGIN IMMEDIATE')
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.lib.save_feedback, authenticated, report['report_id'], 'r-one', 'confirmed')
                time.sleep(0.1)
                blocker.execute("UPDATE users SET role='user' WHERE user_id=?", (self.admin['user_id'],))
                blocker.execute('DELETE FROM sessions WHERE user_id=?', (self.admin['user_id'],))
                blocker.commit()
                with self.assertRaises(self.Error) as caught:
                    future.result(timeout=3)
                self.assertEqual(401, caught.exception.status)
        self.assertEqual([], self.lib.get_report(self.alice, report['report_id'])['feedback'])
        self.denied(403, self.lib.save_feedback, self.admin, report['report_id'], 'r-one', 'confirmed')

    def test_expired_worker_identity_cannot_reacquire_its_own_task(self):
        item = self.upload()
        task = self.lib.enqueue_review(self.alice, item['contract_id'])
        self.lib.claim_task('same-worker')
        self.expire_task(task['task_id'])
        self.assertIsNone(self.lib.claim_task('same-worker'))
        self.assertIsNotNone(self.lib.claim_task('new-worker'))
        self.denied(409, self.lib.finish_task, task['task_id'], 'same-worker', payload={})

    def test_task_version_is_pinned_and_failed_or_invalid_payload_never_writes_report(self):
        item = self.upload()
        task = self.lib.enqueue_review(self.alice, item['contract_id'])
        self.lib.update_contract(self.admin, item['contract_id'], content=b'new contract')
        claimed = self.lib.claim_task('pinned-worker')
        self.assertEqual(item['current_version_id'], claimed['version']['version_id'])
        self.denied(400, self.lib.finish_task, task['task_id'], 'pinned-worker', payload={'bad': object()})
        self.assertEqual('running', self.lib.get_task(self.alice, task['task_id'])['status'])
        self.assertEqual(0, self.lib.list_reports(self.alice)['total'])
        failed = self.lib.finish_task(task['task_id'], 'pinned-worker', error='review unavailable')
        self.assertEqual('failed', failed['status'])
        self.assertEqual(0, self.lib.list_reports(self.alice)['total'])

    def test_finish_report_is_rolled_back_if_task_completion_write_fails(self):
        item = self.upload()
        task = self.lib.enqueue_review(self.alice, item['contract_id'])
        self.lib.claim_task('atomic-worker')
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("CREATE TRIGGER test_abort_completion BEFORE UPDATE ON review_tasks WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT,'simulated disk write failure'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            self.lib.finish_task(task['task_id'], 'atomic-worker', payload={'score': 91})
        self.assertEqual(0, self.lib.list_reports(self.alice)['total'])
        self.assertEqual('running', self.lib.get_task(self.alice, task['task_id'])['status'])

    def test_legacy_workspace_import_is_readonly_admin_owned_and_idempotent(self):
        source = Path(self.temp.name) / 'legacy.sqlite'
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('CREATE TABLE workspace_reports(report_id TEXT PRIMARY KEY, payload TEXT)')
            db.execute('INSERT INTO workspace_reports VALUES (?,?)', ('old-report', json.dumps({
                'contract_id': 'old-engine', 'report_id': 'source-payload-id', 'full_text': 'Historic contract text', 'score': 33, 'source_name': 'old.pdf'})))
        before = source.read_bytes()
        self.denied(403, self.lib.import_legacy, self.alice, source)
        first = self.lib.import_legacy(self.admin, source)
        self.assertEqual({'imported': 1, 'skipped': 0, 'errors': []}, first)
        self.assertEqual({'imported': 0, 'skipped': 1, 'errors': []}, self.lib.import_legacy(self.admin, source))
        self.assertEqual(before, source.read_bytes())
        self.assertEqual(0, self.lib.list_contracts(self.alice)['total'])
        migrated = self.lib.list_contracts(self.admin)['items'][0]
        self.assertEqual(self.admin['user_id'], migrated['owner'])
        self.assertTrue(migrated['legacy_original_unavailable'])
        self.assertEqual(b'Historic contract text', self.lib.get_version(self.admin, migrated['contract_id'])['content'])
        saved = self.lib.get_report(self.admin, self.lib.list_reports(self.admin)['items'][0]['report_id'])
        self.assertEqual('old-engine', saved['payload']['contract_id'])
        self.assertEqual('source-payload-id', saved['payload']['legacy_source_payload']['report_id'])

    def test_legacy_relations_are_preserved_and_unknown_schema_rejected(self):
        source = Path(self.temp.name) / 'old-store.sqlite'
        with closing(sqlite3.connect(source)) as db, db:
            db.executescript('CREATE TABLE contracts(contract_id TEXT PRIMARY KEY,source_name TEXT,score INTEGER);'
                             'CREATE TABLE risks(risk_id TEXT,contract_id TEXT,message TEXT);'
                             'CREATE TABLE citations(risk_id TEXT,law_name TEXT);'
                             'CREATE TABLE clause_fts(contract_id TEXT,clause_id TEXT,body TEXT);')
            db.execute("INSERT INTO contracts VALUES ('old','old.docx',44)")
            db.execute("INSERT INTO risks VALUES ('r1','old','risk message')")
            db.execute("INSERT INTO citations VALUES ('r1','law title')")
            db.execute("INSERT INTO clause_fts VALUES ('old','c1','Clause original')")
        self.assertEqual(1, self.lib.import_legacy(self.admin, source)['imported'])
        report = self.lib.get_report(self.admin, self.lib.list_reports(self.admin)['items'][0]['report_id'])['payload']
        self.assertEqual('risk message', report['risks'][0]['message'])
        self.assertEqual('law title', report['citations'][0]['law_name'])
        self.assertEqual('Clause original', report['clauses'][0]['body'])
        unknown = Path(self.temp.name) / 'unknown.sqlite'
        with closing(sqlite3.connect(unknown)) as db, db:
            db.execute('CREATE TABLE unrelated(value TEXT)')
        self.denied(400, self.lib.import_legacy, self.admin, unknown)

    def test_legacy_unknown_ownership_remains_admin_only_after_importer_demoted(self):
        source = Path(self.temp.name) / 'unknown-owner.sqlite'
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('CREATE TABLE workspace_reports(report_id TEXT PRIMARY KEY,payload TEXT)')
            db.execute('INSERT INTO workspace_reports VALUES (?,?)', ('old', '{"source_name":"old.txt"}'))
        self.lib.import_legacy(self.admin, source)
        item = self.lib.list_contracts(self.admin)['items'][0]
        report = self.lib.list_reports(self.admin)['items'][0]
        self.lib.create_user(self.admin, 'other-admin', 'second-admin-password', role='admin')
        self.lib.update_user(self.admin, self.admin['user_id'], role='user')
        self.assertEqual(0, self.lib.list_contracts(self.admin)['total'])
        self.assertEqual(0, self.lib.list_reports(self.admin)['total'])
        self.denied(404, self.lib.get_contract, self.admin, item['contract_id'])
        self.denied(404, self.lib.get_report, self.admin, report['report_id'])

    def test_mixed_legacy_tables_both_import_and_are_idempotent(self):
        source = Path(self.temp.name) / 'mixed.sqlite'
        with closing(sqlite3.connect(source)) as db, db:
            db.executescript('CREATE TABLE workspace_reports(report_id TEXT,payload TEXT);'
                             'CREATE TABLE contracts(contract_id TEXT,source_name TEXT,full_text TEXT);')
            db.execute('INSERT INTO workspace_reports VALUES (?,?)', ('same-id', json.dumps({
                'document': {'full_text': 'Nested source'}, 'full_text': 'Fallback source'})))
            db.execute("INSERT INTO contracts VALUES ('same-id','compact.txt','Compact source')")
        before = source.read_bytes()
        self.assertEqual({'imported': 2, 'skipped': 0, 'errors': []}, self.lib.import_legacy(self.admin, source))
        self.assertEqual({'imported': 0, 'skipped': 2, 'errors': []}, self.lib.import_legacy(self.admin, source))
        self.assertEqual(before, source.read_bytes())
        items = self.lib.list_contracts(self.admin)['items']
        texts = {self.lib.get_version(self.admin, i['contract_id'])['content'] for i in items}
        self.assertEqual({b'Nested source', b'Compact source'}, texts)
        self.assertTrue(all(i['legacy_original_unavailable'] for i in items))


if __name__ == '__main__':
    unittest.main()
