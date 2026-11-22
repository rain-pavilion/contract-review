"""HTTP integration checks for source-linked reports and durable human review."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import DocumentParse


SAMPLE = """房屋租赁合同
甲方（出租人）：北京星河科技有限公司
乙方（承租人）：上海远山服务有限公司
第一条 标的
甲方将北京市海淀区一号院房屋出租给乙方。
第二条 租金及期限
租金每月5000元，押金10000元。
租赁期限为30年，自2026年1月1日至2025年1月1日。
第三条 解除
甲方有权随时单方解除合同，无需承担任何责任。
第四条 违约责任
乙方违约需支付100000元违约金。
"""


class WorkspaceApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = api.serve("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="zlql-workspace-")
        self.config = Config(store_path=str(Path(self.temp.name) / "archive.sqlite"),
                             ner_enable_model=False, ner_use_bert=False,
                             enable_llm=False, use_lsa=False)
        self.previous_config = api._SERVER_CONFIG
        api._SERVER_CONFIG = self.config

    def tearDown(self):
        api._SERVER_CONFIG = self.previous_config
        self.temp.cleanup()

    def request(self, path, payload=None, *, raw=None, content_type=None):
        if raw is None and payload is not None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=raw,
            headers={"Content-Type": content_type or "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def review(self, *, name="示例租赁合同.txt", store=True, text=SAMPLE):
        status, report = self.request("/api/review", {
            "text": text, "source_name": name, "store": store, "llm": "off",
        })
        self.assertEqual(status, 200, report)
        return report

    def test_item_opinion_persists_without_resolving_machine_pending(self):
        report = self.review()
        item = report["review_explanations"]["items"][0]
        original = report["review_explanations"]
        path = "/api/workspace/reports/" + report["report_id"]
        status, result = self.request(path + "/item-feedback", {
            "item_id": item["item_id"], "status": "supported",
            "reviewer": "OFFLINE-TEST", "note": "仅用于持久化测试，不是人工金标。"})
        self.assertEqual(status, 200, result)
        _, fetched = self.request(path)
        self.assertEqual(fetched["review_explanations"], original)
        self.assertEqual(fetched["score"], report["score"])
        self.assertEqual(fetched["item_feedback"][item["item_id"]]["status"], "supported")
        bad, _ = self.request(path + "/item-feedback", {
            "item_id": "other-report-item", "status": "supported", "reviewer": "TEST", "note": "wrong"})
        self.assertEqual(bad, 404)

    def test_item_opinion_requires_named_reviewer_and_explanation(self):
        report = self.review()
        status, _ = self.request("/api/workspace/reports/" + report["report_id"] + "/item-feedback", {
            "item_id": report["review_explanations"]["items"][0]["item_id"],
            "status": "supported", "reviewer": "", "note": ""})
        self.assertEqual(status, 400)

    def multipart(self, filename, content, *, store=None):
        boundary = "WorkspaceTestBoundary"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        body += content + b"\r\n"
        if store is not None:
            body += (f'--{boundary}\r\nContent-Disposition: form-data; name="store"\r\n\r\n'
                     f'{store}\r\n').encode()
        body += f"--{boundary}--\r\n".encode()
        return body, f"multipart/form-data; boundary={boundary}"

    def test_analyze_parses_without_review_or_archive(self):
        with patch("zhilv_qiling_v2.engine.ReviewEngine.run", side_effect=AssertionError("review called")):
            status, payload = self.request("/api/analyze", {
                "text": SAMPLE, "source_name": "粘贴的合同.docx", "store": True,
            })
        self.assertEqual(status, 200, payload)
        document = payload["document"]
        self.assertEqual(document["source_name"], "粘贴的合同.docx")
        self.assertEqual(document["file_type"], "txt")
        self.assertIn("租赁期限为30年", document["full_text"])
        self.assertTrue(document["clauses"])
        self.assertTrue(document["blocks"])
        self.assertNotIn("score", payload)
        self.assertFalse(Path(self.config.store_path).exists())

    def test_analyze_accepts_real_docx_upload(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as document:
            document.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>采购合同</w:t></w:r></w:p><w:p><w:r><w:t>第一条 合同金额10000元。</w:t></w:r></w:p></w:body></w:document>''')
        raw, content_type = self.multipart("真实上传.docx", stream.getvalue())
        status, payload = self.request("/api/analyze", raw=raw, content_type=content_type)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["document"]["source_name"], "真实上传.docx")
        self.assertIn("10000元", payload["document"]["full_text"])

    def test_review_contains_source_elements_and_text_scoped_subject_checks(self):
        report = self.review(store=False)
        self.assertFalse(report["saved"])
        self.assertIsNone(report["report_id"])
        self.assertEqual(report["document"]["source_name"], report["source_name"])
        self.assertTrue(report["document"]["clauses"])
        self.assertTrue(report["elements"]["amounts"])
        self.assertTrue(report["graph"]["nodes"])
        self.assertIn("未连接外部", report["subject_checks"]["verification_scope"])
        self.assertTrue(report["risks"])
        self.assertIn("trace", report)
        self.assertFalse(Path(self.config.store_path).exists())

    def test_saved_report_round_trips_full_source_and_trace(self):
        report = self.review()
        status, loaded = self.request("/api/workspace/reports/" + report["report_id"])
        self.assertEqual(status, 200, loaded)
        for field in ("document", "elements", "subject_checks", "graph", "risks", "trace", "score"):
            self.assertEqual(loaded[field], report[field], field)
        self.assertTrue(loaded["saved"])
        self.assertEqual(loaded["feedback"], {})

    def test_saved_multipart_review(self):
        raw, content_type = self.multipart("多部分合同.txt", SAMPLE.encode(), store="true")
        status, report = self.request("/api/review", raw=raw, content_type=content_type)
        self.assertEqual(status, 200, report)
        self.assertTrue(report["saved"])
        self.assertEqual(report["source_name"], "多部分合同.txt")

    def test_storage_failure_preserves_finished_review(self):
        for failure in (sqlite3.OperationalError("database is locked"), OSError("read only")):
            with self.subTest(error=type(failure).__name__), patch.object(api, "_workspace_db", side_effect=failure):
                report = self.review()
            self.assertFalse(report["saved"])
            self.assertIsNone(report["report_id"])
            self.assertIn("档案保存失败", report["storage_warning"])
            self.assertTrue(report["risks"])
            self.assertTrue(report["document"]["clauses"])

    def test_reports_paginate_search_and_escape_wildcards(self):
        first = self.review(name="合同100%.txt")
        self.review(name="第二份合同.txt")
        status, payload = self.request("/api/workspace/reports?size=1&page=2")
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["total"], 2)
        self.assertEqual(len(payload["items"]), 1)
        status, found = self.request("/api/workspace/reports?q=100%25")
        self.assertEqual(status, 200, found)
        self.assertEqual(found["total"], 1)
        self.assertEqual(found["items"][0]["source_name"], "合同100%.txt")
        status, by_id = self.request("/api/workspace/reports?q=" + first["contract_id"])
        self.assertEqual(status, 200, by_id)
        self.assertEqual(by_id["total"], 1)
        self.assertEqual(by_id["items"][0]["report_id"], first["report_id"])

    def test_equal_length_documents_are_distinct_snapshots(self):
        first = self.review()
        second = self.review(text=SAMPLE.replace("5000元", "6000元"))
        self.assertNotEqual(first["contract_id"], second["contract_id"])
        self.assertNotEqual(first["report_id"], second["report_id"])
        _, original = self.request("/api/workspace/reports/" + first["report_id"])
        self.assertIn("5000元", original["document"]["full_text"])
        self.assertNotIn("6000元", original["document"]["full_text"])

    def test_feedback_persists_and_never_changes_machine_score(self):
        report = self.review()
        risk_id = report["risks"][0]["risk_id"]
        url = "/api/workspace/reports/" + report["report_id"]
        status, feedback = self.request(url + "/feedback", {
            "risk_id": risk_id, "status": "dismissed", "note": "已核对补充协议。",
        })
        self.assertEqual(status, 200, feedback)
        self.assertEqual(feedback["feedback"][risk_id]["status"], "dismissed")
        _, loaded = self.request(url)
        self.assertEqual(loaded["feedback"][risk_id]["note"], "已核对补充协议。")
        self.assertEqual(loaded["score"], report["score"])
        self.assertEqual(loaded["risks"], report["risks"])
        _, listing = self.request("/api/workspace/reports")
        self.assertEqual(listing["items"][0]["reviewed_count"], 1)
        self.assertEqual(listing["items"][0]["pending_count"], len(report["risks"]) - 1)
        self.request(url + "/feedback", {"risk_id": risk_id, "status": "pending", "note": "待补充材料"})
        _, listing = self.request("/api/workspace/reports")
        self.assertEqual(listing["items"][0]["reviewed_count"], 0)

    def test_feedback_rejects_invalid_status_and_wrong_risk(self):
        report = self.review()
        url = "/api/workspace/reports/" + report["report_id"] + "/feedback"
        status, _ = self.request(url, {"risk_id": report["risks"][0]["risk_id"], "status": "delete"})
        self.assertEqual(status, 400)
        status, _ = self.request(url, {"risk_id": "other-report-risk", "status": "confirmed"})
        self.assertEqual(status, 404)
        status, _ = self.request(url, {"risk_id": report["risks"][0]["risk_id"], "status": "confirmed", "note": "x" * 4001})
        self.assertEqual(status, 400)

    def test_missing_report_returns_404(self):
        status, _ = self.request("/api/workspace/reports/" + "0" * 32)
        self.assertEqual(status, 404)

    def test_invalid_pagination_returns_400(self):
        status, _ = self.request("/api/workspace/reports?page=nope")
        self.assertEqual(status, 400)

    def test_empty_input_and_corrupt_upload_return_400(self):
        status, _ = self.request("/api/analyze", {})
        self.assertEqual(status, 400)
        raw, content_type = self.multipart("broken.docx", b"not a zip file")
        status, _ = self.request("/api/analyze", raw=raw, content_type=content_type)
        self.assertEqual(status, 400)

    def test_empty_uploaded_document_never_receives_a_score(self):
        raw, content_type = self.multipart("empty.txt", b"\r\n  \r\n")
        for path in ("/api/analyze", "/api/review"):
            with self.subTest(path=path):
                status, payload = self.request(path, raw=raw, content_type=content_type)
                self.assertEqual(status, 400, payload)
                self.assertNotIn("score", payload)
                self.assertIn("为空", payload["error"])

    def test_unreadable_pdf_reports_ocr_need(self):
        scanned = DocumentParse(source_name="scanned.pdf", file_type="pdf", full_text="",
                                stats={"needs_ocr_pages": [1, 2]})
        raw, content_type = self.multipart("scanned.pdf", b"%PDF-placeholder")
        with patch("zhilv_qiling_v2.pipeline.prepare", return_value=scanned), patch(
            "zhilv_qiling_v2.engine.ReviewEngine.run", side_effect=AssertionError("review called")
        ):
            status, payload = self.request("/api/review", raw=raw, content_type=content_type)
        self.assertEqual(status, 422, payload)
        self.assertEqual(payload["needs_ocr_pages"], [1, 2])
        self.assertNotIn("score", payload)

    def test_unsupported_upload_is_not_interpreted_as_contract_text(self):
        raw, content_type = self.multipart("program.exe", b"MZ binary content")
        status, payload = self.request("/api/review", raw=raw, content_type=content_type)
        self.assertEqual(status, 415, payload)
        self.assertNotIn("score", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
