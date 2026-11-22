"""Output boundary guards, independent of model quality or human labels."""
import unittest
from zhilv_qiling_v2 import llm_client as C


class LegalOneOutputTests(unittest.TestCase):
    def test_prefilled_open_tag_discards_analysis_examples(self):
        raw = '思考中示例 {"defects":[{"message":"示例"}]}\n</think>\n{"defects":[]}'
        self.assertEqual(C._extract_json(raw), {"defects": []})

    def test_documented_final_marker_uses_only_answer(self):
        raw = '思考过程\n{"ok":false}\n[最终回答]\n{"ok":true}'
        self.assertEqual(C._extract_json(raw), {"ok": True})

    def test_incomplete_reasoning_without_final_answer_is_rejected(self):
        with self.assertRaises(C.LLMClientError):
            C._extract_json('思考过程：可以举例 {"defects":[]}')

    def test_marker_inside_json_string_is_preserved(self):
        self.assertEqual(C._extract_json('{"quote":"[最终回答]是文本"}'), {"quote": "[最终回答]是文本"})

    def test_think_tags_inside_complete_json_quotes_are_literal(self):
        for raw in ('{"quote":"合同有 </think> 标记"}', '{"quote":"<think>是合同原文</think>"}'):
            import json
            self.assertEqual(C._extract_json(raw), json.loads(raw))
            self.assertEqual(C._extract_json('```json\n' + raw + '\n```'), json.loads(raw))

    def test_plain_json_and_complete_think_remain_supported(self):
        self.assertEqual(C._extract_json('<think>{"ok":false}</think>{"ok":true}'), {"ok": True})
        self.assertEqual(C._extract_json('{"ok":true}'), {"ok": True})

    def test_unclosed_think_is_never_repaired_into_an_answer(self):
        with self.assertRaises(C.LLMClientError):
            C._extract_json('<think>例子 {"ok":true}')
