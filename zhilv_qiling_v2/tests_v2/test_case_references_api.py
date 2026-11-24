"""Real HTTP checks of published metadata and read-only case lookup behavior."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse
import urllib.request

from zhilv_qiling_v2 import api, case_references as C
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.tests_v2.http_fixtures import admin_session, prepare_admin


class CaseReferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.previous_config = api._SERVER_CONFIG
        config = Config(enable_llm=False, ner_enable_model=False, ner_use_bert=False,
                        use_lsa=False, entailment_backend="rules")
        prepare_admin(config, cls.temp.name)
        api._SERVER_CONFIG = config
        # Use the real handler and HTTP stack without initializing any model or
        # job worker; this resource has no runtime dependencies on either.
        cls.server = api.WorkspaceServer(("127.0.0.1", 0), api.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.client, cls.csrf = admin_session(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        api._SERVER_CONFIG = cls.previous_config
        cls.temp.cleanup()

    def request(self, query="", *, method="GET", path="/api/references/cases"):
        request = urllib.request.Request(self.base + path + "?q=" + urllib.parse.quote(query),
            method=method, headers={"X-CSRF-Token": self.csrf})
        try:
            response = self.client.open(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_actual_http_returns_only_short_historical_references_with_original_dates(self):
        status, result = self.request()
        self.assertEqual(status, 200)
        self.assertTrue(result["available"])
        self.assertEqual((result["count"], result["total"]), (2, 2))
        self.assertEqual([item["guiding_case_number"] for item in result["items"]], [15, 17])
        for item, original_date, final_judgment in zip(result["items"],
                ("2013-01-31", "2013-11-08"), ("2011-10-19", "2008-03-13")):
            self.assertEqual(item["original_published_date"], original_date)
            self.assertEqual(item["judgment_dates"][-1]["date"], final_judgment)
            self.assertTrue(item["source_url"].startswith("https://www.chinacourt.cn/article/detail/"))
            self.assertLess(len(item["summary"]), 300)
            self.assertNotIn("full_text", item)
            self.assertNotIn("article_text", item)
            self.assertFalse(item["usage"]["automatic_training_allowed"])
            self.assertFalse(item["usage"]["independent_human_gold"])
            self.assertFalse(item["usage"]["subject_current_facts_verified"])
            self.assertFalse(item["usage"]["public_fulltext_republication_allowed"])
        self.assertEqual(result["items"][1]["judgment_dates"][0]["date"], "2007-10")

    def test_case_lookup_by_guiding_number_keyword_and_absent_term(self):
        for query, ids in (("指导案例15号", [15]), ("人格混同", [15]),
                           ("欺诈", [17]), ("（2008）二中民终字第00453号", [17]),
                           ("不存在的案例123", [])):
            with self.subTest(query=query):
                status, result = self.request(query)
                self.assertEqual(status, 200)
                self.assertEqual([item["guiding_case_number"] for item in result["items"]], ids)
                self.assertEqual(result["count"], len(ids))
                self.assertEqual(result["total"], 2)

    def test_metadata_matches_verified_origin_and_endpoint_has_no_write_operation(self):
        package = json.loads(C.INDEX_PATH.read_text(encoding="utf-8"))
        origin = Path(__file__).resolve().parents[2] / package["origin_index"]
        with self.subTest(check="retained_origin_hash"):
            if not origin.is_file():
                self.skipTest("Historical source index is not distributed in this checkout")
            self.assertEqual(package["origin_index_sha256"], hashlib.sha256(origin.read_bytes()).hexdigest())
        before = C.INDEX_PATH.read_bytes()
        with patch("zhilv_qiling_v2.model_audit.record", side_effect=AssertionError("case lookup must not invoke models")):
            self.assertEqual(self.request()[0], 200)
            self.assertEqual(self.request(method="POST")[0], 404)
        self.assertEqual(C.INDEX_PATH.read_bytes(), before)

    def test_missing_metadata_returns_explicit_unavailable_state(self):
        with TemporaryDirectory() as directory, patch.object(C, "INDEX_PATH", Path(directory) / "absent.json"):
            status, result = self.request()
        self.assertEqual(status, 200)
        self.assertFalse(result["available"])
        self.assertEqual(result["items"], [])


if __name__ == "__main__":
    unittest.main()
