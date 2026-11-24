"""Real HTTP permission regressions for the dual business libraries."""
import http.cookiejar
import gc
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config


class LibraryApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"ZLQL_LIBRARY_ROOT": str(self.root),
            "ZLQL_MODEL_AUDIT_PATH": str(self.root / "model-audit.sqlite"),
            "ZLQL_PUBLIC_REGISTRATION": "1"})
        self.env.start()
        from zhilv_qiling_v2 import library_api
        self.registration_rate = patch.dict(library_api._REGISTER_ATTEMPTS, {}, clear=True)
        self.registration_rate.start()
        self.old = api._SERVER_CONFIG
        api._SERVER_CONFIG = Config(store_path=str(self.root / "old.sqlite"),
            subject_facts_path=str(self.root / "facts.json"), enable_llm=False,
            ner_enable_model=False, ner_use_bert=False, use_lsa=False,
            entailment_backend="rules")
        self.server = api.WorkspaceServer(("127.0.0.1", 0), api.Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        api._SERVER_CONFIG = self.old
        self.env.stop()
        self.registration_rate.stop()
        gc.collect()
        self.tmp.cleanup()

    def request(self, path, body=None, *, client=None, csrf=None, headers=None):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        h = {"Content-Type": "application/json", **(headers or {})}
        if csrf:
            h["X-CSRF-Token"] = csrf
        req = urllib.request.Request(self.base + path, data=data, headers=h)
        try:
            response = (client or urllib.request.build_opener()).open(req, timeout=20)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            raw = response.read()
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = raw
            return response.status, payload, dict(response.headers)

    def bootstrap(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        self.store = ContractLibrary(self.root / "contracts.sqlite")
        self.admin = self.store.bootstrap_admin("administrator", "Strong-admin-93825!")
        self.alice = self.store.create_user(self.admin, "alice", "Strong-alice-93825!")
        self.bob = self.store.create_user(self.admin, "bob", "Strong-bobby-93825!")

    def login(self, username="alice", password="Strong-alice-93825!"):
        client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        code, payload, headers = self.request("/api/auth/login", {"username": username,
            "password": password}, client=client)
        self.assertEqual(code, 200, payload)
        self.assertIn("HttpOnly", headers.get("Set-Cookie", ""))
        self.assertIn("SameSite=Strict", headers.get("Set-Cookie", ""))
        self.assertNotIn("token", payload)
        return client, payload["csrf_token"]

    def upload(self, client, csrf, text="第一条 甲方支付价款一万元。", name="合同.txt"):
        code, result, _ = self.request("/api/library/contracts", {"filename": name,
            "text": text, "title": name, "owner_user_id": "spoofed"}, client=client, csrf=csrf)
        self.assertEqual(code, 201, result)
        return result

    def test_anonymous_legacy_history_is_rejected(self):
        code, _, _ = self.request("/api/history")
        self.assertEqual(code, 401)

    def test_registration_cannot_claim_an_uninitialized_server(self):
        code, status, _ = self.request('/api/auth/status')
        self.assertEqual(code, 200)
        self.assertFalse(status.get('registration_enabled', True))
        code, _, _ = self.request('/api/auth/register', {'username': 'newperson',
            'password': 'Registered-person-123!', 'password_confirm': 'Registered-person-123!'})
        self.assertEqual(code, 409)
        self.assertFalse(self.request('/api/auth/status')[1]['initialized'])

    def test_stale_cookie_still_allows_login_and_registration_status(self):
        self.bootstrap()
        code, status, _ = self.request('/api/auth/status',
            headers={'Cookie': 'zlql_session=expired-or-unknown-cookie'})
        self.assertEqual(code, 200, status)
        self.assertFalse(status['authenticated'])
        self.assertTrue(status['registration_enabled'])

    def test_public_registration_is_ordinary_and_isolated(self):
        self.bootstrap()
        self.assertTrue(self.request('/api/auth/status')[1].get('registration_enabled'))
        body = {'username': '新用户_01', 'password': 'Registered-person-123!',
            'password_confirm': 'Registered-person-123!'}
        self.assertEqual(self.request('/api/auth/register', {**body, 'password_confirm': 'different'})[0], 400)
        self.assertEqual(self.request('/api/auth/register', {**body, 'password': 'short',
            'password_confirm': 'short'})[0], 400)
        for field, value in (('role', 'admin'), ('active', True), ('user_id', self.admin['user_id'])):
            self.assertEqual(self.request('/api/auth/register', {**body, field: value})[0], 403)
        code, registered, headers = self.request('/api/auth/register', body)
        self.assertEqual(code, 201, registered)
        self.assertTrue(registered['registered'])
        self.assertNotIn('Set-Cookie', headers)
        self.assertNotIn('password', json.dumps(registered))
        self.assertEqual(self.request('/api/auth/register', body)[0], 409)
        client, csrf = self.login(body['username'], body['password'])
        me = self.request('/api/auth/me', client=client)[1]['user']
        self.assertEqual(me['role'], 'user')
        self.assertTrue(me['active'])
        self.assertEqual(self.request('/api/library/users', client=client)[0], 403)
        own = self.upload(client, csrf, name='registered-owner.txt')
        alice, _ = self.login()
        self.assertEqual(self.request('/api/library/contracts/' + own['contract_id'], client=alice)[0], 404)
        admin, admin_csrf = self.login('administrator', 'Strong-admin-93825!')
        self.assertTrue(any(u['username'] == body['username'] for u in
            self.request('/api/library/users', client=admin)[1]['items']))
        self.assertEqual(self.request('/api/library/users/' + me['user_id'], {'active': False},
            client=admin, csrf=admin_csrf)[0], 200)
        self.assertEqual(self.request('/api/library/contracts', client=client)[0], 401)

    def test_registration_origin_disable_and_rate_limit(self):
        self.bootstrap()
        body = {'username': 'limiteduser', 'password': 'Registered-person-123!',
            'password_confirm': 'Registered-person-123!'}
        self.assertEqual(self.request('/api/auth/register', body,
            headers={'Origin': 'https://attacker.example'})[0], 403)
        with patch.dict(os.environ, {'ZLQL_PUBLIC_REGISTRATION': '0'}):
            self.assertFalse(self.request('/api/auth/status')[1]['registration_enabled'])
            self.assertEqual(self.request('/api/auth/register', body)[0], 403)
        from zhilv_qiling_v2 import library_api
        with patch.dict(library_api._REGISTER_ATTEMPTS,
            {'127.0.0.1': [time.monotonic()] * 10}, clear=True):
            self.assertEqual(self.request('/api/auth/register', body)[0], 429)

    def test_registration_uses_configured_separate_user_library(self):
        from zhilv_qiling_v2.contract_library import ContractLibrary
        import sqlite3
        user_path = self.root / 'account-data' / 'custom-users.sqlite'
        api._SERVER_CONFIG.user_library_path = str(user_path)
        store = ContractLibrary(self.root / 'contracts.sqlite', users_path=user_path)
        store.bootstrap_admin('administrator', 'Strong-admin-93825!')
        code, result, _ = self.request('/api/auth/register', {'username': 'customuser',
            'password': 'Registered-person-123!', 'password_confirm': 'Registered-person-123!'})
        self.assertEqual(code, 201, result)
        client, csrf = self.login('customuser', 'Registered-person-123!')
        self.upload(client, csrf)
        self.assertFalse((self.root / 'users.sqlite').exists())
        with sqlite3.connect(user_path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM users').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 1)
        with sqlite3.connect(self.root / 'contracts.sqlite') as db:
            self.assertEqual([r[1] for r in db.execute('PRAGMA table_info(users)')], ['user_id'])
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='sessions'").fetchone())

    def test_registration_rejects_invalid_types_and_oversized_body(self):
        self.bootstrap()
        body = {'username': 'typeduser', 'password': 'Registered-person-123!',
            'password_confirm': 'Registered-person-123!'}
        for changes in ({'username': ['typeduser']}, {'username': 'x'}, {'username': '<admin>'},
            {'password': 123, 'password_confirm': 123}, {'password_confirm': None}):
            self.assertEqual(self.request('/api/auth/register', {**body, **changes})[0], 400)
        self.assertEqual(self.request('/api/auth/register', {**body, 'password': 'x' * 9000})[0], 413)

    def test_session_csrf_logout_and_no_cross_origin_login(self):
        self.bootstrap()
        client, csrf = self.login()
        code, me, _ = self.request("/api/auth/me", client=client)
        self.assertEqual(me["user"]["username"], "alice")
        self.assertEqual(code, 200)
        self.assertEqual(self.request("/api/library/contracts", {"text": "合同"}, client=client)[0], 403)
        self.assertEqual(self.request("/api/auth/login", {"username": "alice", "password":
            "Strong-alice-93825!"}, headers={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.request("/api/auth/logout", {}, client=client, csrf=csrf)[0], 200)
        self.assertEqual(self.request("/api/library/contracts", client=client)[0], 401)

    def test_own_contract_download_and_admin_revision(self):
        self.bootstrap()
        alice, a_csrf = self.login()
        bob, b_csrf = self.login("bob", "Strong-bobby-93825!")
        admin, admin_csrf = self.login("administrator", "Strong-admin-93825!")
        c = self.upload(alice, a_csrf)
        cid = c["contract_id"]
        self.assertEqual(c["owner_user_id"], self.alice["user_id"])
        self.assertEqual(self.request("/api/library/contracts", client=bob)[1]["total"], 0)
        for suffix in ("", "/download"):
            self.assertEqual(self.request("/api/library/contracts/" + cid + suffix, client=bob)[0], 404)
        for action in ("update", "delete"):
            self.assertEqual(self.request("/api/library/contracts/" + cid + "/" + action,
                {"text": "bad"}, client=alice, csrf=a_csrf)[0], 403)
        code, revised, _ = self.request("/api/library/contracts/" + cid + "/update",
            {"text": "第一条 修订价款两万元。", "expected_version_id": c["current_version_id"]},
            client=admin, csrf=admin_csrf)
        self.assertEqual(code, 200, revised)
        self.assertNotEqual(revised["current_version_id"], c["current_version_id"])
        code, original, headers = self.request("/api/library/contracts/" + cid +
            "/download?version_id=" + c["current_version_id"], client=alice)
        self.assertEqual(code, 200)
        self.assertIn("一万元".encode(), original)
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertEqual(self.request("/api/library/contracts", client=admin)[1]["total"], 1)

    def test_legacy_and_global_private_routes_are_admin_only(self):
        self.bootstrap()
        alice, csrf = self.login()
        for path in ("/api/history", "/api/stats", "/api/workspace/reports",
            "/api/workspace/batches", "/api/workspace/subject-facts", "/api/runtime/events",
            "/api/workspace/contract-packages", "/workspace", "/annotations"):
            self.assertEqual(self.request(path, client=alice)[0], 403, path)
        for path in ("/api/review", "/api/analyze", "/api/batch", "/api/train", "/api/crawl"):
            self.assertEqual(self.request(path, {"path": str(self.root / "secret.txt")},
                client=alice, csrf=csrf)[0], 403, path)

    def test_reports_and_task_details_cannot_cross_users(self):
        self.bootstrap()
        alice, csrf = self.login()
        bob, _ = self.login("bob", "Strong-bobby-93825!")
        c = self.upload(alice, csrf)
        v = c["current_version_id"]
        report = self.store.save_report(self.alice, c["contract_id"], v,
            {"score": 89, "risk_count": 1, "risks": [{"risk_id": "risk-1", "message": "private"}]})
        task = self.store.enqueue_review(self.alice, c["contract_id"], v)
        self.assertEqual(self.request("/api/library/reports/" + report["report_id"], client=bob)[0], 404)
        self.assertEqual(self.request("/api/library/tasks/" + task["task_id"], client=bob)[0], 404)
        self.assertEqual(self.request("/api/library/reports", client=bob)[1]["total"], 0)
        self.assertEqual(self.request("/api/library/reports/" + report["report_id"], client=alice)[0], 200)
        route = "/api/library/reports/" + report["report_id"] + "/feedback"
        self.assertEqual(self.request(route, {"risk_id": "risk-1", "status": "confirmed"},
            client=alice, csrf=csrf)[0], 403)
        admin, admin_csrf = self.login("administrator", "Strong-admin-93825!")
        self.assertEqual(self.request(route, {"risk_id": "risk-1", "status": "confirmed", "note": "人工复核"},
            client=admin, csrf=admin_csrf)[0], 200)
        self.assertEqual(self.request("/api/library/reports/" + report["report_id"],
            client=alice)[1]["feedback"][0]["note"], "人工复核")

    def test_no_law_match_does_not_create_an_empty_snapshot_on_review(self):
        self.bootstrap()
        alice, csrf = self.login()
        c = self.upload(alice, csrf)
        before = set(self.root.rglob("*.jsonl"))
        for region in ("不存在地区一", "不存在地区二"):
            code, result, _ = self.request("/api/library/contracts/" + c["contract_id"] + "/review",
                {"llm": False, "region": region}, client=alice, csrf=csrf)
            self.assertEqual(code, 409, result)
        self.assertEqual(set(self.root.rglob("*.jsonl")), before)

    def test_law_draft_publication_real_review_and_bound_snapshot(self):
        self.bootstrap()
        from zhilv_qiling_v2.law_library import LawLibrary
        laws = LawLibrary(self.root / "laws.sqlite")
        version = laws.ingest_document(self.admin, {"title": "合同价款测试规定", "domains": ["general"],
            "effective_from": "2020-01-01"}, [{"article_no": "第一条", "content":
            "当事人应当按照约定全面履行自己的义务，按合同约定支付价款。"}],
            b"<html>official-source-fixture</html>", "https://www.gov.cn/test-contract-rule.html")
        alice, csrf = self.login()
        bob, _ = self.login("bob", "Strong-bobby-93825!")
        admin, admin_csrf = self.login("administrator", "Strong-admin-93825!")
        vid = version["version_id"]
        self.assertEqual(self.request("/api/library/law-versions/" + vid, client=alice)[0], 404)
        self.assertEqual(self.request("/api/library/law-versions/" + vid + "/publish", {},
            client=alice, csrf=csrf)[0], 403)
        code, _, _ = self.request("/api/library/law-versions/" + vid + "/metadata", {
            "validity": "effective", "verification": {"verified": True, "evidence": "测试夹具核对记录"}},
            client=admin, csrf=admin_csrf)
        self.assertEqual(code, 200)
        code, published, _ = self.request("/api/library/law-versions/" + vid + "/publish", {},
            client=admin, csrf=admin_csrf)
        self.assertEqual(code, 200, published)
        self.assertNotIn("path", published["snapshot"])
        self.assertEqual(self.request("/api/library/laws/search?q=" + urllib.parse.quote("价款"),
            client=alice)[1]["total"], 1)
        c = self.upload(alice, csrf)
        for invalid in ({"contract_type": "invalid"}, {"review_position": "owner"}):
            code, rejected, _ = self.request("/api/library/contracts/" + c["contract_id"] + "/review",
                {"llm": False, **invalid}, client=alice, csrf=csrf)
            self.assertEqual(code, 400, rejected)
        code, task, _ = self.request("/api/library/contracts/" + c["contract_id"] + "/review",
            {"llm": False, "domain": "general", "contract_type": "委托", "review_position": "party_b"},
            client=alice, csrf=csrf)
        self.assertEqual(code, 202, task)
        self.assertEqual(task["options"]["contract_type"], "委托")
        self.assertEqual(task["options"]["review_position"], "party_b")
        self.assertNotIn("path", task["options"]["law_snapshot"])
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            code, task, _ = self.request("/api/library/tasks/" + task["task_id"], client=alice)
            if task["status"] in ("completed", "failed"):
                break
            time.sleep(.1)
        self.assertEqual(task["status"], "completed", task)
        code, report, _ = self.request("/api/library/reports/" + task["report_id"], client=alice)
        self.assertEqual(code, 200)
        payload = report["payload"]
        self.assertEqual(payload["contract_version_id"], c["current_version_id"])
        self.assertEqual(payload["law_snapshot"]["article_count"], 1)
        self.assertEqual(payload["document"]["full_text"].strip(), "第一条 甲方支付价款一万元。")
        self.assertEqual(payload["review_context"]["contract_type"], "委托")
        self.assertEqual(payload["review_context"]["review_position"], "party_b")
        self.assertEqual(payload["review_framework"], payload["stats"]["review_framework"])
        self.assertEqual(payload["review_context"], payload["stats"]["review_context"])
        self.assertEqual(payload["review_framework"]["schema_version"], "contract-review-three-layers-v1")
        self.assertTrue(all(key in payload["review_framework"] for key in ("workflow", "review_points", "assessments")))
        report_route = "/api/library/reports/" + task["report_id"]
        self.assertEqual(self.request(report_route)[0], 401)
        second_code, second, _ = self.request(report_route, client=alice)
        self.assertEqual(second_code, 200, second)
        self.assertEqual(second["payload"]["review_context"], payload["review_context"])
        self.assertEqual(second["payload"]["review_framework"], payload["review_framework"])
        self.assertEqual(self.request("/api/library/reports/" + task["report_id"], client=bob)[0], 404)


if __name__ == "__main__":
    unittest.main()
