"""Independent parser and native-mode boundaries; no actual model calls."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import llm_client as C


class Response:
    def __init__(self, message):
        self.value = {"model": "legalone-8b", "choices": [{"message": message}]}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def read(self):
        return json.dumps(self.value).encode()


class LegalOneOutputBoundaryTests(unittest.TestCase):
    def test_quoted_think_tags_in_valid_json_are_data(self):
        for value in ("合同原文包含 </think> 标记", "<think>字段引用</think>", "引号\"与\\及</think>"):
            with self.subTest(value=value):
                expected = {"quote": value, "ok": True}
                self.assertEqual(C._extract_json(json.dumps(expected, ensure_ascii=False)), expected)

    def test_fenced_json_preserves_quoted_think_tags(self):
        expected = {"quote": "<think>合同引用</think>", "ok": True}
        self.assertEqual(C._extract_json("```json\n" + json.dumps(expected, ensure_ascii=False) + "\n```"), expected)

    def test_separate_analysis_is_not_promoted_to_business_content(self):
        with patch.object(C.urllib.request, "urlopen", return_value=Response({"content": None, "reasoning_content": '{"ok":true}'})):
            raw, envelope, _ = C._complete(C.LLMConfig(endpoint="http://unused", model="legalone-8b"), [], None)
        self.assertEqual(raw, "")
        self.assertIn("reasoning_content", envelope["choices"][0]["message"])
        with self.assertRaises(C.LLMClientError):
            C._extract_json(raw)

    def test_explicit_native_modes_do_not_change_messages(self):
        messages = [{"role": "user", "content": "原始问题"}]
        captured = []
        def complete(request, timeout):
            captured.append(json.loads(request.data))
            return Response({"content": '{"ok":true}'})
        with patch.object(C.urllib.request, "urlopen", side_effect=complete):
            C._complete(C.LLMConfig(endpoint="http://unused", model="legalone-8b"), messages, 12)
            C._complete(C.LLMConfig(endpoint="http://unused", model="legalone-8b", native_thinking="off", native_reasoning_format="deepseek"), messages, 12)
        self.assertNotIn("chat_template_kwargs", captured[0])
        self.assertNotIn("reasoning_format", captured[0])
        self.assertEqual(captured[1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(captured[1]["reasoning_format"], "deepseek")
        self.assertEqual(captured[0]["messages"], captured[1]["messages"])
        self.assertEqual(captured[1]["messages"], messages)
        self.assertEqual(captured[1]["max_tokens"], 12)

    def test_invalid_native_modes_rejected_before_network(self):
        for kwargs in ({"native_thinking": "invalid"}, {"native_reasoning_format": "invalid"}):
            with self.subTest(kwargs=kwargs), patch.object(C.urllib.request, "urlopen") as urlopen:
                with self.assertRaises(C.LLMClientError):
                    C._complete(C.LLMConfig(endpoint="http://unused", model="legalone-8b", **kwargs), [], None)
                urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
