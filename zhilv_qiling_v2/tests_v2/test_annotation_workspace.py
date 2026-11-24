"""AUTOMATED FIXTURES ONLY. These tests are not real independent human gold."""
from __future__ import annotations
import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from zhilv_qiling_v2.annotation_workspace import AnnotationWorkspace, canonical, digest, handle_request
from tools.annotation_workspace import make_server


def candidate(i=1):
    text = f"甲方😀应按约支付第{i}笔价款。"
    return {"schema_version": "zlql-human-gold-v1", "id": f"FIXTURE-{i}", "text": text,
        "text_sha256": digest(text), "source": {"origin": "AUTOMATED_WORKFLOW_FIXTURE_NOT_HUMAN_GOLD",
        "group_id": f"fixture-group-{i}", "license_or_permission": "automated local test fixture only",
        "collected_at": "2026-11-14T00:00:00+00:00"}, "synthetic_candidate": True,
        "status": "unlabeled", "split": "unassigned", "annotations": [], "reviews": []}


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="annotation_fixture_")
        self.addCleanup(self.tmp.cleanup)
        self.store = AnnotationWorkspace(self.tmp.name)
        self.credentials = {"admin": self.store.bootstrap("FIXTURE_ADMIN_NOT_A_PERSON")}
        self.admin = self.store.authenticate(self.credentials["admin"]["token"])
        for role in ("A", "B", "adjudicator"):
            self.credentials[role] = self.store.create_user(self.admin, {"reviewer_id": f"FIXTURE_{role}_NOT_A_PERSON",
                "role": role, "identity_confirmed_by_admin": True})
        result = self.call("admin", "/datasets", {"name": "AUTOMATED_FIXTURE_NOT_HUMAN_GOLD",
            "rows": [candidate()], "guidelines": "仅用于自动化流程测试的虚拟手册，所有结果都是fixture，不能当作真人标注。"})
        self.did = result["dataset_id"]

    def call(self, role, path, body=None, query=None, status=200):
        result = handle_request(self.store, "GET" if body is None else "POST", "/api/annotations" + path,
            body, "Bearer " + self.credentials[role]["token"], query)
        self.assertEqual(result.get("status", 200), status, result)
        return result

    def claim(self, role):
        return self.call(role, "/claim", {"dataset_id": self.did})

    def ann(self, label="FIXTURE_LABEL"):
        return {"type": "entity", "start": 2, "end": 3, "label": label, "evidence": "😀"}

    def attest(self, task, **changes):
        return {"human_authored": True, "independent": True, "read_full_text": True,
            "used_system_predictions": False, "empty_confirmed": True,
            "guidelines_sha256": task["guidelines_sha256"], **changes}

    def submit(self, role, annotations=None):
        task = self.claim(role)
        return self.call(role, "/submit", {"dataset_id": self.did, "record_id": task["record"]["id"],
            "revision": task["review"]["revision"], "annotations": [self.ann()] if annotations is None else annotations,
            "attestation": self.attest(task), "notes": "AUTOMATED_FIXTURE_ONLY"})

    def complete(self):
        self.submit("A")
        self.submit("B", [])
        task = self.call("adjudicator", "/adjudication/claim", {"dataset_id": self.did})
        self.assertEqual(task["adjudication"]["annotations"], [])
        self.assertTrue(task["disagreement"])
        return self.call("adjudicator", "/adjudication/submit", {"dataset_id": self.did,
            "record_id": task["record"]["id"], "revision": 0, "annotations": [self.ann()],
            "rationale": "AUTOMATED_FIXTURE_ONLY: exercises disagreement flow, not actual legal/human adjudication",
            "attestation": self.attest(task, reviewed_both=True)})

    def freeze_body(self):
        return {"dataset_id": self.did, "version": "fixture_v1", "seed": 20261114,
            "training_inventory": [{"text": "不同来源的自动化训练清单fixture", "source": {"group_id": "training-fixture"}}],
            "inventory_complete_attestation": True, "near_duplicate_review_completed": True,
            "heldout_policy_acknowledged": True}

    def test_no_auth_or_cross_role_access(self):
        self.assertEqual(handle_request(self.store, "GET", "/api/annotations/datasets")["status"], 401)
        self.assertEqual(handle_request(self.store, "GET", "/api/annotations/datasets", authorization="bad")["status"], 401)
        self.call("A", "/users", status=403)
        self.call("B", "/export", query={"dataset_id": self.did, "kind": "snapshot"}, status=403)
        self.call("A", "/adjudication/claim", {"dataset_id": self.did}, status=403)

    def test_unlabeled_pack_cannot_freeze_and_never_autolabels(self):
        result = self.call("admin", "/freeze", self.freeze_body(), status=409)
        self.assertFalse(result["gold_created"])
        snapshot = self.call("admin", "/export", query={"dataset_id": self.did, "kind": "snapshot"})
        row = json.loads(snapshot["jsonl"])
        self.assertEqual(row["status"], "unlabeled")
        self.assertEqual(row["annotations"], [])
        self.assertEqual(row["reviews"], [])
        self.assertFalse(snapshot["gold_ready"])

    def test_unique_people_roles_and_no_bootstrap_reset(self):
        self.call("admin", "/users", {"reviewer_id": self.credentials["A"]["reviewer_id"], "role": "B",
            "identity_confirmed_by_admin": True}, status=409)
        with self.assertRaisesRegex(ValueError, "已初始化"):
            self.store.bootstrap("another-fixture")

    def test_blind_task_and_export_never_reveal_other_lane(self):
        task = self.claim("A")
        self.call("A", "/draft", {"dataset_id": self.did, "record_id": "FIXTURE-1", "revision": 0,
            "annotations": [self.ann("SECRET_A_FIXTURE")]})
        other = self.claim("B")
        exported = self.call("B", "/export", query={"dataset_id": self.did, "kind": "blind"})
        self.assertNotIn("SECRET_A_FIXTURE", json.dumps(other) + json.dumps(exported))
        self.assertEqual(other["review"]["annotations"], [])
        self.assertNotIn("reviews", other["review"])
        self.call("adjudicator", "/adjudication/claim", {"dataset_id": self.did}, status=409)
        self.assertNotIn("reviews", task["review"])

    def test_revision_and_exact_unicode_spans(self):
        self.claim("A")
        base = {"dataset_id": self.did, "record_id": "FIXTURE-1", "revision": 0, "annotations": [self.ann()]}
        result = self.call("A", "/draft", base)
        self.assertEqual(result["review"]["revision"], 1)
        self.call("A", "/draft", base, status=409)
        for item in ({**self.ann(), "start": True}, {**self.ann(), "end": 4}, {**self.ann(), "evidence": "错误"}):
            self.call("A", "/draft", {**base, "revision": 1, "annotations": [item]}, status=400)

    def test_submit_requires_actual_attestation_and_locks_result(self):
        task = self.claim("A")
        body = {"dataset_id": self.did, "record_id": "FIXTURE-1", "revision": 0, "annotations": []}
        self.call("A", "/submit", body, status=400)
        self.call("A", "/submit", {**body, "attestation": self.attest(task, used_system_predictions=True)}, status=400)
        self.call("A", "/submit", {**body, "attestation": self.attest(task, empty_confirmed=False)}, status=400)
        self.call("A", "/submit", {**body, "attestation": self.attest(task)})
        self.call("A", "/draft", {**body, "revision": 1}, status=409)

    def test_concurrent_claim_has_single_owner(self):
        self.credentials["A2"] = self.store.create_user(self.admin, {"reviewer_id": "FIXTURE_A2_NOT_A_PERSON",
            "role": "A", "identity_confirmed_by_admin": True})
        def take(role):
            return handle_request(self.store, "POST", "/api/annotations/claim", {"dataset_id": self.did},
                                  "Bearer " + self.credentials[role]["token"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(take, ("A", "A2")))
        self.assertEqual(sorted(r.get("status", 200) for r in results), [200, 409])

    def test_offline_import_preserves_original_hash_and_identity(self):
        exported = self.call("A", "/export", query={"dataset_id": self.did, "kind": "blind"})
        row = json.loads(exported["jsonl"])
        self.call("A", "/import", {"dataset_id": self.did, "rows": [row]}, status=400)
        task = self.claim("A")
        row.update(status="submitted", annotations=[self.ann()], notes="AUTOMATED_OFFLINE_FIXTURE",
            method="manual", used_system_predictions=False, reviewed_at="2026-11-14T01:00:00+00:00",
            attestation=self.attest(task), annotations_sha256=digest(canonical([self.ann()])))
        self.call("B", "/import", {"dataset_id": self.did, "rows": [row]}, status=403)
        self.call("A", "/import", {"dataset_id": self.did, "rows": [{**row, "text_sha256": "0"*64}]}, status=422)
        result = self.call("A", "/import", {"dataset_id": self.did, "jsonl": canonical(row) + "\n"})
        self.assertEqual(result["count"], 1)
        with closing(self.store.connect()) as c:
            raw = c.execute("SELECT source_jsonl,source_sha256 FROM import_sources").fetchone()
        self.assertEqual(raw[0], canonical(row) + "\n")
        self.assertEqual(raw[1], digest(raw[0]))

    def test_import_batch_rolls_back_if_any_record_invalid(self):
        row = json.loads(self.call("A", "/export", query={"dataset_id": self.did, "kind": "blind"})["jsonl"])
        row.update(status="draft", annotations=[self.ann()], annotations_sha256=digest(canonical([self.ann()])))
        other = {**row, "id": "missing"}
        self.call("A", "/import", {"dataset_id": self.did, "rows": [row, other]}, status=404)
        task = self.claim("A")
        self.assertEqual(task["review"]["annotations"], [])
        self.assertEqual(task["review"]["revision"], 0)

    def test_adjudication_requires_two_then_freeze_uses_existing_validator(self):
        task = self.complete()
        self.assertEqual(task["adjudication"]["status"], "adjudicated")
        result = self.call("admin", "/freeze", self.freeze_body())
        self.assertTrue(result["gold_ready"])
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(result["manifest"]["synthetic_records"], 1)
        self.assertFalse(result["manifest"]["independently_sourced_real_contracts_verified"])
        exported = self.call("admin", "/export", query={"dataset_id": self.did, "kind": "frozen"})
        row = json.loads(exported["jsonl"])
        self.assertEqual(row["status"], "adjudicated")
        self.assertEqual(row["split"], "test")
        self.assertEqual(len(row["reviews"]), 2)
        self.assertEqual(row["reviews"][0]["annotations_sha256"], digest(canonical(row["reviews"][0]["annotations"])))
        self.assertEqual(row["adjudication"]["annotations_sha256"], digest(canonical(row["annotations"])))
        self.call("admin", "/freeze", self.freeze_body(), status=409)
        self.call("A", "/claim", {"dataset_id": self.did}, status=409)
        self.assertTrue(self.call("admin", "/audit")["chain_valid"])

    def test_freeze_rejects_empty_inventory_and_training_leakage(self):
        self.complete()
        self.call("admin", "/freeze", {**self.freeze_body(), "training_inventory": []}, status=400)
        base = candidate()
        self.call("admin", "/freeze", {**self.freeze_body(), "training_inventory": [base]}, status=422)
        self.call("admin", "/export", query={"dataset_id": self.did, "kind": "frozen"}, status=409)

    def test_frozen_hash_tampering_is_rejected(self):
        self.complete()
        self.call("admin", "/freeze", self.freeze_body())
        path = Path(self.tmp.name) / "frozen" / self.did / "fixture_v1" / "gold.jsonl"
        path.write_text("tampered fixture", encoding="utf-8")
        self.call("admin", "/export", query={"dataset_id": self.did, "kind": "frozen"}, status=409)

    def test_candidate_text_changed_or_labeled_import_rejected(self):
        row = candidate(2)
        row["text"] += "changed"
        self.call("admin", "/datasets", {"name": "bad", "rows": [row], "guidelines": "固定标注手册及一致性边界，测试不得当作真实人工金标。"}, status=422)
        row = candidate(2)
        row["status"] = "adjudicated"
        self.call("admin", "/datasets", {"name": "bad", "rows": [row], "guidelines": "固定标注手册及一致性边界，测试不得当作真实人工金标。"}, status=400)


class HTTPFixtureTest(unittest.TestCase):
    setUp = WorkspaceTest.setUp
    call = WorkspaceTest.call
    ann = WorkspaceTest.ann

    def test_real_local_http_page_auth_and_draft_roundtrip(self):
        server = make_server(self.store, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/annotations") as response:
            page = response.read().decode("utf-8")
            self.assertIn("独立人工标注工作台", page)
            self.assertIn("Array.from(before.toString()).length", page)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertNotIn("sessionStorage", page)
            self.assertNotIn("localStorage", page)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/api/annotations/datasets")
        self.assertEqual(error.exception.code, 401)
        def post(path, data):
            request = urllib.request.Request(base + "/api/annotations" + path,
                json.dumps(data).encode("utf-8"), {"Content-Type": "application/json",
                "Authorization": "Bearer " + self.credentials["A"]["token"]})
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        task = post("/claim", {"dataset_id": self.did})
        result = post("/draft", {"dataset_id": self.did, "record_id": task["record"]["id"],
                                "revision": 0, "annotations": [self.ann()]})
        self.assertEqual(result["review"]["revision"], 1)
        self.assertEqual(result["review"]["annotations"][0]["evidence"], "😀")

    def test_real_product_api_mount_page_auth_and_unlabeled_dataset(self):
        from zhilv_qiling_v2 import api
        from zhilv_qiling_v2.config import Config
        from zhilv_qiling_v2.tests_v2.http_fixtures import admin_session, prepare_admin
        config = Config(store_path=str(Path(self.tmp.name) / "product_archive.sqlite"),
            ner_use_bert=False, ner_enable_model=False, use_lsa=False,
            entailment_backend="rules", enable_llm=False)
        prepare_admin(config, Path(self.tmp.name) / "product-libraries")
        with patch.dict("os.environ", {"ZLQL_ANNOTATION_WORKSPACE": self.tmp.name}), \
             patch.object(api, "_SERVER_CONFIG", config), \
             patch("zhilv_qiling_v2.model_runtime.initialize_backend", return_value={"active_backend": "rules"}):
            server = api.serve("127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                client, csrf = admin_session(base)
                for path in ("/annotations", "/annotation.html"):
                    with client.open(base + path, timeout=10) as response:
                        self.assertIn("独立人工标注工作台", response.read().decode("utf-8"))
                        self.assertIn("text/html", response.headers["Content-Type"])
                with client.open(base + "/workspace", timeout=10) as response:
                    self.assertIn('/annotations', response.read().decode("utf-8"))
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base + "/api/annotations/datasets", timeout=10)
                self.assertEqual(error.exception.code, 401)
                headers = {"Authorization": "Bearer " + self.credentials["admin"]["token"],
                    "Content-Type": "application/json", "X-CSRF-Token": csrf}
                request = urllib.request.Request(base + "/api/annotations/datasets", headers=headers)
                with client.open(request, timeout=10) as response:
                    result = json.load(response)
                self.assertEqual(result["datasets"][0]["id"], self.did)
                self.assertFalse(result["datasets"][0]["gold_ready"])
                request = urllib.request.Request(base + "/api/annotations/datasets", headers=headers,
                    data=json.dumps({"name": "ACTUAL_HTTP_FIXTURE_NOT_HUMAN_LABELS", "rows": [candidate(2)],
                        "guidelines": "这是实际HTTP挂载测试的fixture手册，不是独立真人标注或正式金标。"}).encode("utf-8"))
                with client.open(request, timeout=10) as response:
                    created = json.load(response)
                self.assertFalse(created["gold_ready"])
                self.assertEqual(created["status_label"], "pending_manual_annotation")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(5)


if __name__ == "__main__":
    unittest.main()
