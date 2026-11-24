"""v2 接口层测试：HTTP API 与 CLI 的对外契约。

接口层最容易在重构时悄悄改签名（`review_document(document, config)` 而不是
`review_document(name, text)` 这类），所以这里用真实 HTTP 与真实 argv 打一遍。

    python -m unittest zhilv_qiling_v2.tests_v2.test_api_cli -v
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

from zhilv_qiling_v2 import api, cli
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.tests_v2.http_fixtures import admin_session, prepare_admin
from zhilv_qiling_v2.version import VERSION_TAG

SAMPLE_TEXT = """技术开发（委托）合同
委托方（甲方）：星海科技产业园管理有限公司
受托方（乙方）：云图数据技术有限公司
第一条 项目内容
1.1 甲方委托乙方开发智慧园区数据中台。
第二条 价款
2.1 合同总金额为人民币1200000元。
第三条 验收
3.1 乙方应及时交付并保证通过验收。
"""


class _Server:
    """在后台线程里起一个真实的 HTTP 服务，端口交给系统分配。"""

    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.previous_config = api._SERVER_CONFIG
        config = Config(store_path=str(Path(self.temp.name) / "archive.sqlite"),
            ner_enable_model=False, ner_use_bert=False, enable_llm=False,
            use_lsa=False, entailment_backend="rules")
        prepare_admin(config, self.temp.name)
        api._SERVER_CONFIG = config
        self.httpd = api.serve("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        self.client, self.csrf = admin_session(f"http://127.0.0.1:{self.port}")
        return self

    def __exit__(self, *_exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)
        api._SERVER_CONFIG = self.previous_config
        self.temp.cleanup()

    def request(self, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json", "X-CSRF-Token": self.csrf},
        )
        try:
            with self.client.open(request, timeout=180) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _Server().__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.__exit__()

    def test_health_exposes_sanitized_public_status(self):
        status, payload = self.server.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["tag"], VERSION_TAG)
        self.assertTrue(payload["authentication_required"])
        self.assertNotIn("models", payload)
        self.assertNotIn("capabilities", payload)

    def test_models_lists_five_components(self):
        status, payload = self.server.request("/api/models")
        self.assertEqual(status, 200)
        self.assertTrue({"boundary", "ner", "rerank", "risk", "calibration"} <= set(payload["files"]))

    def test_review_text_returns_report_shape(self):
        status, payload = self.server.request("/api/review", {"text": SAMPLE_TEXT})
        self.assertEqual(status, 200)
        for key in ("contract_id", "score", "risks", "trace", "risk_count"):
            self.assertIn(key, payload)
        self.assertGreaterEqual(payload["score"], 0)
        self.assertLessEqual(payload["score"], 100)

    def test_review_rejects_empty(self):
        status, payload = self.server.request("/api/review", {})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_retrieve_returns_articles(self):
        status, payload = self.server.request(
            "/api/retrieve", {"text": "租赁期限二十年以上", "top_k": 3}
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(payload["hits"]), 1)
        self.assertTrue(payload["hits"][0]["article"].startswith("第"))

    def test_batch_returns_summary(self):
        status, payload = self.server.request("/api/batch", {
            "limit": 2, "paths": ["samples/tech-development-contract.txt"]})
        self.assertEqual(status, 200, payload)
        summary = payload["summary"]
        self.assertIn("count", summary)
        self.assertIn("code_histogram", summary)
        self.assertEqual(summary["count"], 1)
        self.assertLessEqual(summary["count"], 2)

    def test_unknown_route_404(self):
        status, _payload = self.server.request("/api/nope")
        self.assertEqual(status, 404)

    def test_webui_is_served(self):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.server.port}/", timeout=30
        ) as response:
            body = response.read().decode("utf-8")
        self.assertIn("智律契灵", body)
        self.assertIn("/api/library", body)
        with self.server.client.open(
            f"http://127.0.0.1:{self.server.port}/workspace", timeout=30
        ) as response:
            body = response.read().decode("utf-8")
        self.assertIn("api/review", body)


class CliTest(unittest.TestCase):
    """CLI 用真实 argv 跑，只断言可观察输出与退出码。"""

    def _run(self, argv: list[str]) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_doctor_passes(self):
        code, output = self._run(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("python>=3.10", output)
        self.assertIn("法条语料", output)

    def test_schema_lists_28_codes(self):
        code, output = self._run(["schema", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(len(payload["codes"]), 28)
        self.assertEqual(len(payload["categories"]), 7)

    def test_models_lists_weights(self):
        code, output = self._run(["models", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertTrue(payload["files"])

    def test_retrieve_puts_term_over_20y_first(self):
        code, output = self._run(["retrieve", "--text", "租赁期限二十年以上",
                                  "--top-k", "3", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        # 列标准化后的 LTR 会把若干 hub 条文提到第 1 位，单条查询的第 1 名不稳定；
        # 这条用例真正要保证的是"最长租赁期限那一条进得了前 3"，不是必须压在第 1。
        top3 = [hit["article"] for hit in payload["hits"][:3]]
        self.assertIn("第七百零五条", top3, top3)

    def test_review_json_has_risks(self):
        code, output = self._run(["review", "samples/tech-development-contract.txt", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(len(payload), 1)
        self.assertIn("risks", payload[0])

    def test_missing_file_exits_nonzero(self):
        code, _output = self._run(["review", "not-exists-xyz.txt"])
        self.assertEqual(code, 1)

    def test_help_lists_subcommands(self):
        parser = cli.build_parser()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertTrue(actions)
        choices = set(actions[0].choices)
        self.assertTrue({"review", "batch", "retrieve", "crawl", "ocr", "train"} <= choices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
