"""Exercise real HTTP, package bytes, queue persistence and source-bound citations."""
import hashlib
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.contract_packages import import_contract_package, package_evidence_summary
from zhilv_qiling_v2.tests_v2.test_contract_packages import ContractPackageTests
from zhilv_qiling_v2.tests_v2.http_fixtures import admin_session, prepare_admin
from zhilv_qiling_v2.workspace_queue import QueueStore, PermanentTaskError


class ContractPackageApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ContractPackageTests()
        self.fixture.setUp()
        self.fixture.fixture()
        self.summary = import_contract_package(self.fixture.root, self.fixture.store, "fixture")
        self.config = Config(store_path=str(Path(self.fixture.temp.name) / "reviews.sqlite"),
            contract_packages_path=str(self.fixture.store),
            subject_facts_path=str(Path(self.fixture.temp.name) / "facts.json"),
            ner_enable_model=False, ner_use_bert=False, use_lsa=False,
            enable_multi_agent=True, enable_llm=False, entailment_backend="rules")
        self.old = api._SERVER_CONFIG
        prepare_admin(self.config, self.fixture.temp.name)
        api._SERVER_CONFIG = self.config
        self.server = api.serve("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.client, self.csrf = admin_session(self.base)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        api._SERVER_CONFIG = self.old
        self.fixture.tearDown()

    def request(self, path, body=None, raw=False):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data,
            headers={"Content-Type": "application/json", "X-CSRF-Token": self.csrf})
        try:
            response = self.client.open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            return response.status, content if raw else json.loads(content), dict(response.headers)

    def test_list_analyze_source_and_background_are_distinct(self):
        status, packages, _ = self.request("/api/workspace/contract-packages")
        self.assertEqual(status, 200)
        self.assertEqual(packages["items"][0]["package_id"], "fixture")
        _, detail, _ = self.request("/api/workspace/contract-packages/fixture")
        self.assertEqual(detail["evidence_summary"]["manifest_sha256"], self.summary["manifest_sha256"])
        _, parsed, _ = self.request("/api/analyze", {"package_id": "fixture"})
        self.assertNotIn("虚构结论", parsed["document"]["full_text"])
        self.assertNotIn("TEMPLATE", parsed["document"]["full_text"])
        self.assertEqual(parsed["stats"]["pages"], 2)
        self.assertEqual(self.config.statutes_path, "")
        source = detail["manifest"]["sources"][0]
        code, original, headers = self.request(
            "/api/workspace/contract-packages/fixture/sources/" + source["source_id"], raw=True)
        self.assertEqual(code, 200)
        self.assertEqual(hashlib.sha256(original).hexdigest(), source["sha256"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        _, background, _ = self.request("/api/workspace/contract-packages/fixture/background?q=AUTHORIZATION")
        self.assertFalse(background["items"][0]["is_signed_evidence"])

    def test_package_retrieval_retains_original_and_quality(self):
        code, result, _ = self.request("/api/retrieve", {"package_id": "fixture", "text": "保留原始证据"})
        self.assertEqual(code, 200)
        self.assertTrue(result["hits"])
        self.assertEqual(result["hits"][0]["law"], "测试法")
        provenance = result["hits"][0]["provenance"]
        self.assertEqual(provenance["package_id"], "fixture")
        self.assertIn("original_sha256", provenance)
        self.assertIn("location", provenance)
        self.assertEqual(self.config.statutes_path, "")

    def test_package_queue_archives_and_reads_back_pending_privacy(self):
        code, queued, _ = self.request("/api/workspace/contract-packages/fixture/review",
                                     {"engine": "multi", "llm": "0"})
        self.assertEqual(code, 200)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, batch, _ = self.request("/api/workspace/batches/" + queued["batch_id"])
            if batch["status"] in ("completed", "failed"):
                break
            time.sleep(.1)
        self.assertEqual(batch["status"], "completed", batch)
        _, report, _ = self.request("/api/workspace/reports/" + queued["task_id"])
        self.assertTrue(report["saved"])
        self.assertEqual(report["document"]["file_type"], "contract_package")
        self.assertEqual(report["contract_package"]["manifest_sha256"], queued["manifest_sha256"])
        self.assertTrue(report["contract_package"]["missing_documents"])
        self.assertTrue(all(row["status"] == "unknown" for row in report["contract_package"]["missing_documents"]))
        self.assertIn("model_calls", report["stats"])
        self.assertIn("timings", report["stats"])
        self.assertTrue(report["review_explanations"])

    def test_changed_package_manifest_fails_before_processing(self):
        store = QueueStore(self.config.store_path)
        batch = store.create_batch({"llm": "0"})
        descriptor = {"package_id": "fixture", "manifest_sha256": self.summary["manifest_sha256"], "include_tender": False}
        store.add_task(batch["batch_id"], "fixture.contract_package", "contract_package", json.dumps(descriptor).encode())
        store.start_batch(batch["batch_id"])
        claimed = store.claim_next("owned-test-worker")
        path = self.fixture.store / "fixture/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["title"] += " changed"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(PermanentTaskError, "发生变化"):
            api._run_workspace_task(claimed, self.config)


if __name__ == "__main__":
    unittest.main()
