"""Request perspective stays validated, isolated and bound to persisted reviews."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import ReviewReport
from zhilv_qiling_v2.workspace_queue import QueueStore


class ReviewContextApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.previous = api._SERVER_CONFIG
        self.config = Config(store_path=str(self.root / "workspace.sqlite"),
            subject_facts_path=str(self.root / "facts.json"), enable_llm=False,
            ner_use_bert=False, ner_enable_model=False, use_lsa=False,
            entailment_backend="rules")
        api._SERVER_CONFIG = self.config

    def tearDown(self):
        api._SERVER_CONFIG = self.previous
        self.tmp.cleanup()

    def context(self, body=None, *, query=None, raw=b"", content_type="application/json"):
        return api.RequestContext(handler=SimpleNamespace(headers={"Content-Type": content_type}, server=SimpleNamespace()),
            method="POST", path="/api/review", query=query or {}, body=raw,
            json_body=body or {}, trace_id="perspective-test")

    def test_invalid_review_options_are_rejected_before_document_parsing(self):
        # Missing request validation would otherwise accept these as a review.
        for options in ({"contract_type": "未知合同"}, {"contract_type": None},
                        {"review_position": "owner"}, {"review_position": ["party_a"]}):
            with self.subTest(options=options):
                response = api.handle_review(self.context({"text": "第一条 双方约定开发项目。", **options}))
                self.assertEqual(response.get("status"), 400)
                self.assertTrue(response.get("error"))

    def test_json_query_and_multipart_options_reach_an_isolated_review(self):
        seen = []

        def review(document, config, *args, **kwargs):
            seen.append(config)
            return {"contract_type": config.review_contract_type,
                    "review_position": config.review_position}

        with patch.object(api, "_review_document", side_effect=review):
            result = api.handle_review(self.context({"text": "第一条 开发项目。",
                "contract_type": "技术开发", "review_position": "party_b"}))
            self.assertEqual(result, {"contract_type": "技术开发", "review_position": "party_b"})
            result = api.handle_review(self.context({"text": "第一条 支付价款。"},
                query={"contract_type": ["委托"], "review_position": ["party_a"]}))
            self.assertEqual(result, {"contract_type": "委托", "review_position": "party_a"})
            raw = ("--Perspective\r\nContent-Disposition: form-data; name=\"contract_type\"\r\n\r\n房屋租赁\r\n"
                "--Perspective\r\nContent-Disposition: form-data; name=\"review_position\"\r\n\r\nparty_b\r\n"
                "--Perspective\r\nContent-Disposition: form-data; name=\"file\"; filename=\"contract.txt\"\r\n"
                "Content-Type: text/plain\r\n\r\n第一条 双方租赁房屋。\r\n--Perspective--\r\n").encode()
            result = api.handle_review(self.context(raw=raw,
                content_type="multipart/form-data; boundary=Perspective"))
            self.assertEqual(result, {"contract_type": "房屋租赁", "review_position": "party_b"})
        self.assertTrue(all(config is not self.config for config in seen))
        self.assertEqual(self.config.review_contract_type, "auto")
        self.assertEqual(self.config.review_position, "neutral")

    def test_queue_persists_review_perspective_after_reopen(self):
        store = QueueStore(self.root / "queue.sqlite")
        batch = store.create_batch({"engine": "multi", "llm": False,
            "contract_type": "技术开发", "review_position": "party_b"})
        store.add_task(batch["batch_id"], "合同.txt", "txt", "第一条 项目开发。".encode())
        restored = QueueStore(store.path)
        restored.start_batch(batch["batch_id"])
        task = restored.claim_next("review-context-worker")
        self.assertEqual(task["options"].get("contract_type"), "技术开发")
        self.assertEqual(task["options"].get("review_position"), "party_b")

    def test_queue_rejects_invalid_review_perspective(self):
        store = QueueStore(self.root / "queue.sqlite")
        with self.assertRaises(ValueError):
            store.create_batch({"review_position": "everyone"})

    def test_queue_rejects_nonobject_options(self):
        store = QueueStore(self.root / "queue.sqlite")
        with self.assertRaises(ValueError):
            store.create_batch([])

    def test_batch_api_validates_options_and_persists_query_selection(self):
        store = QueueStore(self.config.store_path)
        with patch.object(api, "_ensure_queue_worker", return_value=SimpleNamespace(store=store)):
            invalid = api.handle_workspace_batches(self.context({"review_position": "owner"}))
            self.assertEqual(invalid.get("status"), 400)
            result = api.handle_workspace_batches(self.context({"contract_type": "委托"},
                query={"review_position": ["party_b"]}))
        self.assertEqual(result["options"]["contract_type"], "委托")
        self.assertEqual(result["options"]["review_position"], "party_b")

    def test_legacy_batch_rejects_invalid_context_before_review(self):
        source = self.root / "contract.txt"
        source.write_text("第一条 甲方向乙方支付服务价款。", encoding="utf-8")
        response = api.handle_batch(self.context({"paths": [str(source)], "review_position": "owner"}))
        self.assertEqual(response.get("status"), 400)

    def test_report_payload_exposes_only_recorded_review_framework(self):
        framework = {"version": "test", "workflow": [], "review_points": [], "assessments": []}
        context = {"contract_type": "技术开发", "review_position": "party_b"}
        report = ReviewReport("contract-1", "合同.txt", stats={
            "review_framework": framework, "review_context": context})
        payload = api._report_payload(report)
        self.assertEqual(payload.get("review_framework"), framework)
        self.assertEqual(payload.get("review_context"), context)
        historical = api._report_payload(ReviewReport("old-1", "旧合同.txt"))
        self.assertNotIn("review_framework", historical)
        self.assertNotIn("review_context", historical)

    def test_real_review_archives_selected_context_and_actual_framework(self):
        result = api.handle_review(self.context({"text": "房屋租赁合同\n第一条 甲方出租房屋，乙方每月支付租金。",
            "source_name": "perspective.txt", "contract_type": "委托",
            "review_position": "party_b", "store": True, "llm": "0"}))
        self.assertTrue(result["saved"])
        context = result.get("review_context", {})
        self.assertEqual(context.get("contract_type"), "委托")
        self.assertEqual(context.get("review_position"), "party_b")
        self.assertEqual(context.get("contract_type_source"), "user")
        self.assertEqual(context.get("detected_contract_type"), "房屋租赁")
        self.assertIn("review_framework", result)
        ctx = self.context()
        ctx.params = {"report_id": result["report_id"]}
        restored = api.handle_workspace_report(ctx)
        self.assertEqual(restored["review_context"], context)
        self.assertEqual(restored["review_framework"], result["review_framework"])
        self.assertEqual(self.config.review_contract_type, "auto")
        self.assertEqual(self.config.review_position, "neutral")

    def test_auto_government_upgrade_uses_effective_report_type(self):
        text = "合作协议\n甲方：北京市财政局\n乙方：北京远山科技有限公司\n第一条 双方开展项目合作，甲方向乙方支付服务费。"
        for selected, expected in (("auto", "政府采购"), ("通用", "通用")):
            with self.subTest(selected=selected):
                result = api.handle_review(self.context({"text": text, "source_name": "cooperation.txt",
                    "contract_type": selected, "llm": "0"}))
                self.assertEqual(result["review_context"]["contract_type"], expected)
                self.assertEqual(result["stats"]["contract_type"], expected)
                self.assertEqual(result["contract_type"], expected)
                self.assertEqual(result["review_context"]["detected_contract_type"], "通用")
                self.assertEqual(result["document"]["stats"]["detected_contract_type"], "通用")
                self.assertEqual(result["document"]["stats"]["contract_type"], "通用")

    def test_final_report_and_archive_preserve_manual_framework_obligations(self):
        from zhilv_qiling_v2.tests_v2.test_web_workspace import SAMPLE
        api._SERVER_CONFIG = replace(self.config, enable_multi_agent=True)
        result = api.handle_review(self.context({"text": SAMPLE, "source_name": "manual.txt", "store": True}))
        framework = result["review_framework"]
        self.assertEqual(framework, result["stats"]["review_framework"])
        self.assertEqual(result["review_context"], result["stats"]["review_context"])
        manual_ids = {item["item_id"] for item in result["review_explanations"]["items"] if item.get("manual_required")}
        self.assertTrue(manual_ids)
        self.assertTrue(manual_ids.issubset(framework["workflow"]["pending_item_ids"]))
        self.assertEqual(framework["workflow"]["status"], "needs_manual")
        ctx = self.context()
        ctx.params = {"report_id": result["report_id"]}
        restored = api.handle_workspace_report(ctx)
        self.assertEqual(restored["review_framework"], framework)
        self.assertEqual(restored["stats"]["review_framework"], framework)
        self.assertEqual(restored["review_framework"]["workflow"]["pending_item_ids"],
            framework["workflow"]["pending_item_ids"])

    def test_context_fingerprint_changes_with_stance_but_has_no_identity_claim(self):
        from zhilv_qiling_v2.review_context import apply_review_options, build_review_context
        cfg = apply_review_options(self.config, {"contract_type": "委托", "review_position": "party_a"})
        context = build_review_context("委托", cfg, "房屋租赁")
        self.assertEqual(context["contract_type"], "委托")
        self.assertEqual(context["detected_contract_type"], "房屋租赁")
        self.assertFalse(context["requires_confirmation"])
        self.assertEqual(context["scope"], "user_selected_review_perspective_not_verified_identity")
        self.assertEqual(context, build_review_context("委托", cfg, "房屋租赁"))
        other = apply_review_options(self.config, {"contract_type": "委托", "review_position": "party_b"})
        self.assertNotEqual(context["context_fingerprint"],
            build_review_context("委托", other, "房屋租赁")["context_fingerprint"])
        defaults = build_review_context("通用", self.config)
        self.assertTrue(defaults["requires_confirmation"])
        self.assertEqual(defaults["contract_type_source"], "auto")

    def test_normalization_rejects_explicit_empty_and_nonstring_fields(self):
        from zhilv_qiling_v2.review_context import normalize_review_options
        self.assertEqual(normalize_review_options({}), {"contract_type": "auto", "review_position": "neutral"})
        for options in ({"contract_type": ""}, {"review_position": ""},
                        {"contract_type": 1}, {"review_position": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                normalize_review_options(options)

    def test_restarted_queue_task_archives_its_own_context(self):
        store = QueueStore(self.config.store_path)
        batch = store.create_batch({"contract_type": "技术开发", "review_position": "party_a"})
        store.add_task(batch["batch_id"], "合同.txt", "txt", "第一条 甲方委托乙方开发平台。".encode())
        restored = QueueStore(store.path)
        restored.start_batch(batch["batch_id"])
        task = restored.claim_next("context-archive-worker")
        report_id, _ = api._run_workspace_task(task, self.config)
        ctx = self.context()
        ctx.params = {"report_id": report_id}
        report = api.handle_workspace_report(ctx)
        self.assertEqual(report.get("review_context", {}).get("contract_type"), "技术开发")
        self.assertEqual(report.get("review_context", {}).get("review_position"), "party_a")
        self.assertIn("review_framework", report)
        self.assertEqual(self.config.review_contract_type, "auto")
        self.assertEqual(self.config.review_position, "neutral")

    def test_preview_options_never_change_default_followup_review(self):
        preview = api.handle_analyze(self.context({"text": "第一条 甲方支付价款。",
            "contract_type": "委托", "review_position": "party_b"}))
        self.assertIn("document", preview)
        self.assertNotIn("review_framework", preview)
        self.assertEqual(self.config.review_contract_type, "auto")
        self.assertEqual(self.config.review_position, "neutral")

        def review(document, config, *args, **kwargs):
            return {"contract_type": config.review_contract_type, "review_position": config.review_position}

        with patch.object(api, "_review_document", side_effect=review):
            result = api.handle_review(self.context({"text": "第一条 乙方提供服务。"}))
        self.assertEqual(result, {"contract_type": "auto", "review_position": "neutral"})

    def test_contract_package_queue_keeps_selected_context(self):
        from zhilv_qiling_v2.contract_packages import import_contract_package
        from zhilv_qiling_v2.tests_v2.test_contract_packages import ContractPackageTests
        fixture = ContractPackageTests()
        fixture.setUp()
        try:
            fixture.fixture()
            import_contract_package(fixture.root, fixture.store, "fixture")
            config = replace(self.config, contract_packages_path=str(fixture.store))
            api._SERVER_CONFIG = config
            store = QueueStore(config.store_path)
            ctx = self.context({"contract_type": "承揽", "review_position": "party_a", "llm": "0"})
            ctx.params = {"package_id": "fixture", "action": "review"}
            with patch.object(api, "_ensure_queue_worker", return_value=SimpleNamespace(store=store)):
                queued = api.handle_contract_packages(ctx)
            self.assertEqual(queued["status"], "queued")
            task = QueueStore(config.store_path).claim_next("package-context-worker")
            self.assertEqual(task["options"].get("contract_type"), "承揽")
            self.assertEqual(task["options"].get("review_position"), "party_a")
            report_id, _ = api._run_workspace_task(task, config)
            ctx.params = {"report_id": report_id}
            report = api.handle_workspace_report(ctx)
            self.assertEqual(report.get("review_context", {}).get("contract_type"), "承揽")
            self.assertEqual(report.get("review_context", {}).get("review_position"), "party_a")
            self.assertIn("review_framework", report)
        finally:
            fixture.tearDown()

    def test_library_request_rejects_invalid_context_and_runs_persisted_selection(self):
        from zhilv_qiling_v2 import library_api
        from zhilv_qiling_v2.contract_library import ContractLibrary
        from zhilv_qiling_v2.law_library import LawLibrary
        with patch.dict(os.environ, {"ZLQL_LIBRARY_ROOT": str(self.root / "libraries")}):
            contract_path, law_path = library_api.library_paths(self.config)
            store = ContractLibrary(contract_path)
            admin = store.bootstrap_admin("context-admin", "Context-admin-93825!")
            contract = store.upload(admin, "合同.txt", "第一条 甲方委托乙方开发平台。".encode())
            handler = SimpleNamespace(server=SimpleNamespace())
            route = "/api/library/contracts/" + contract["contract_id"] + "/review"
            for options in ({"review_position": "all"}, {"contract_type": "invalid"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    library_api._handle_library(handler, "POST", urlsplit(route), options, {}, store, admin)
            self.assertEqual(store.list_tasks(admin)["items"], [])
            laws = LawLibrary(law_path)
            version = laws.ingest_document(admin, {"title": "合同测试规定", "domains": ["general"],
                "effective_from": "2020-01-01"}, [{"article_no": "第一条",
                "content": "当事人应当按照约定全面履行义务，按照合同约定支付价款。"}],
                b"official-source-fixture", "https://www.gov.cn/context-test.html")
            laws.set_version_metadata(admin, version["version_id"], {"validity": "effective",
                "verification": {"verified": True, "evidence": "离线测试夹具核对记录"}})
            laws.publish_version(admin, version["version_id"])
            with patch.object(library_api, "ensure_library_worker", return_value=None):
                queued, status = library_api._handle_library(handler, "POST",
                    urlsplit(route + "?review_position=party_b"), {"contract_type": "委托", "llm": False},
                    {}, store, admin)
            self.assertEqual(status, 202)
            self.assertEqual(queued["options"].get("contract_type"), "委托")
            self.assertEqual(queued["options"].get("review_position"), "party_b")
            restored = ContractLibrary(contract_path)
            task = restored.claim_task("context-library-worker")
            result = library_api.run_library_review(task, self.config)
            self.assertEqual(result.get("review_context", {}).get("contract_type"), "委托")
            self.assertEqual(result.get("review_context", {}).get("review_position"), "party_b")
            self.assertIn("review_framework", result)
            finished = restored.finish_task(task["task_id"], "context-library-worker", payload=result)
            report = restored.get_report(admin, finished["report_id"])["payload"]
            self.assertEqual(report["review_context"], result["review_context"])
            self.assertEqual(report["review_framework"], result["review_framework"])
        self.assertEqual(self.config.review_contract_type, "auto")
        self.assertEqual(self.config.review_position, "neutral")


if __name__ == "__main__":
    unittest.main()
