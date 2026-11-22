"""Native extractive quotation constrains source location, never legal truth."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import long_context as C, nodes as N
from zhilv_qiling_v2.tests_v2.test_core_completion import clause, contract, configuration


class NativeSourceExtractTests(unittest.TestCase):
    def setUp(self):
        self.a = clause("a", "甲方负责交付设备。")
        self.b = clause("b", "乙方应在验收后十日内付款。")
        self.contract = contract([self.a, self.b])
        self.config = configuration(llm_cross_clause_pairs=0, llm_native_json_schema=True)
        self.item = {"feedback_item_id": "f-b", "candidate_id": "candidate-b", "article_uid": "law-b",
            "clause_ids": ["b"], "article_quote": "这是法条引文，不是合同原文。"}

    def quote_schema(self, items=None, sources=None):
        schema = N._feedback_json_schema(items or [self.item], clause_sources=sources)
        return schema["properties"]["verifications"]["items"]["properties"]

    def run_reply(self, *, items=None, wrong_quote=False, subject=False, native=True):
        self.config.llm_native_json_schema = native
        items = [self.item] if items is None else items
        captured = []
        def reply(config, messages, **options):
            payload = json.loads(messages[-1]["content"])
            captured.append((payload, messages[0]["content"], options))
            values = payload.get("subject_fact_feedback") if subject else payload["verification_feedback"]
            rows = []
            for item in values:
                root = "subject_fact_verifications" if subject else "verifications"
                fields = ("feedback_item_id", "fact_id", "source_id", "subject_name", "fact_type", "clause_id") if subject else ("feedback_item_id", "candidate_id", "article_uid")
                row = {key: item[key] for key in fields}
                text = payload["clause_sources"][item["bound_clause_ids"][0]][0]["text"]
                row.update(verdict="confirmed", reason="只作来源协议测试。", clause_quote=item.get("article_quote") if wrong_quote else text)
                if subject:
                    row.update(fact_quote=item["fact_evidence_text"], source_quote=item["source_evidence_text"])
                else:
                    row["article_quote"] = item["article_quote"]
                rows.append(row)
            return SimpleNamespace(content={root: rows}, model="fixture")
        kwargs = {"subject_fact_feedback" if subject else "verification_feedback": items}
        with patch.object(N.L, "request_json", side_effect=reply):
            _, meta, degraded = N._llm_logic(self.contract, self.config, [self.a, self.b], **kwargs)
        return captured, meta, degraded

    def test_only_current_bound_contract_segments_enter_enum_without_normalization_or_joining(self):
        sources = {"a": [self.a.text], "b": ["\n 绑定原文。\n", "\n 绑定原文。\n", "另一独立片段。"], "unseen": ["不在对应条款。"]}
        props = self.quote_schema(sources=sources)
        self.assertEqual(props["clause_quote"]["enum"], ["\n 绑定原文。\n", "另一独立片段。"])
        self.assertNotIn(" 绑定原文。另一独立片段。", props["clause_quote"]["enum"])
        self.assertNotIn(self.item["article_quote"], props["clause_quote"]["enum"])
        base = self.quote_schema()
        for key in ("feedback_item_id", "candidate_id", "article_uid", "verdict", "reason", "article_quote"):
            self.assertEqual(props[key], base[key])

    def test_long_segments_use_overlapping_80_character_slices_covering_the_tail(self):
        for length, expected_count in ((1, 1), (80, 1), (81, 2), (160, 3), (161, 4), (241, 6)):
            with self.subTest(length=length):
                text = "".join(chr(0x4e00 + index) for index in range(length))
                options = self.quote_schema(sources={"b": [text]})["clause_quote"]["enum"]
                self.assertEqual(len(options), expected_count)
                covered = set()
                for option in options:
                    self.assertGreater(len(option), 0)
                    self.assertLessEqual(len(option), 80)
                    start = text.index(option)
                    covered.update(range(start, start + len(option)))
                self.assertEqual(covered, set(range(length)))
                self.assertTrue(options[-1].endswith(text[-1]))

    def test_actual_native_job_passes_real_window_sources_and_declares_its_narrow_scope(self):
        captured, meta, _ = self.run_reply()
        payload, system, options = captured[0]
        props = options["json_schema"]["properties"]["verifications"]["items"]["properties"]
        self.assertEqual(props["clause_quote"]["enum"], [self.b.text])
        self.assertEqual(payload["clause_quote_mode"], "native_source_extract_selection")
        self.assertIn("不证明相关性或法律真值", system)
        self.assertIn("不是自由引用能力验收", system)
        self.assertEqual(meta["clause_quote_mode"], "native_source_extract_selection")
        self.assertEqual(meta["calls"][0]["clause_quote_mode"], "native_source_extract_selection")
        self.assertEqual(meta["calls"][0]["clause_quote_candidates"], 1)
        self.assertTrue(meta["feedback_complete"])

    def test_model_return_is_not_replaced_and_wrong_article_quote_remains_pending(self):
        _, meta, _ = self.run_reply(wrong_quote=True)
        call = meta["calls"][0]
        response = json.loads(call["feedback_response"]["json"])
        self.assertEqual(response["verifications"][0]["clause_quote"], self.item["article_quote"])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f-b"])
        self.assertEqual(call["verification_errors"][0]["reason"], "clause_quote_not_in_feedback_window")

    def test_known_batch_quote_from_wrong_item_still_fails_item_binding(self):
        other = {**self.item, "feedback_item_id": "f-a", "candidate_id": "candidate-a", "article_uid": "law-a", "clause_ids": ["a"]}
        feedback = [other, self.item]
        schema = N._feedback_json_schema(feedback, clause_sources={"a": [self.a.text], "b": [self.b.text]})
        row = {"feedback_item_id": "f-b", "candidate_id": "candidate-b", "article_uid": "law-b",
            "verdict": "confirmed", "reason": "来源可定位但配对错误。", "clause_quote": self.a.text,
            "article_quote": self.item["article_quote"]}
        content = {"verifications": [row]}
        N.L._validate_json_schema(content, schema)
        _, errors = N._validate_feedback_verifications(content, feedback, {"a": [self.a.text], "b": [self.b.text]})
        self.assertIn({"feedback_item_id": "f-b", "reason": "clause_quote_not_in_feedback_window"}, errors)
        self.assertIn({"feedback_item_id": "f-a", "reason": "missing_verification"}, errors)

    def test_empty_sources_have_no_fabricated_option_and_skip_unsatisfiable_model_call(self):
        self.assertEqual(self.quote_schema(sources={"a": [self.a.text], "b": ["", " \n"]})["clause_quote"]["enum"], [])
        empty = C.ReviewWindow("empty-source", "text", (C.Segment("b", 0, 1, " "),))
        with patch.object(C, "build_review_plan", return_value=([empty], {"cross_windows": 0})), \
             patch.object(N.L, "request_json") as request:
            _, meta, degraded = N._llm_logic(self.contract, self.config, [self.b], verification_feedback=[self.item])
        request.assert_not_called()
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["f-b"])
        self.assertEqual(meta["calls"][0]["status"], "feedback_source_unavailable")
        self.assertIn("llm_feedback_clause_source_unavailable", degraded)

    def test_gateway_off_and_plain_review_do_not_gain_native_selection_behavior(self):
        captured, meta, _ = self.run_reply(native=False)
        self.assertNotIn("json_schema", captured[0][2])
        self.assertNotIn("clause_quote_mode", captured[0][0])
        self.assertNotIn("native_source_extract_selection", captured[0][1])
        self.assertNotIn("clause_quote_mode", meta)
        self.config.llm_native_json_schema = True
        with patch.object(N.L, "request_json", return_value=SimpleNamespace(
                content={"defects": [], "more_findings_possible": False}, model="fixture",
                call_id="offline-plain-source-test")) as request:
            _, plain, _ = N._llm_logic(self.contract, self.config, [self.a, self.b])
        schema = request.call_args.kwargs["json_schema"]
        self.assertEqual(set(schema["properties"]), {"defects", "more_findings_possible"})
        self.assertNotIn("enum", schema["properties"]["defects"]["items"]["properties"]["quote"])
        self.assertTrue(request.call_args.args[1][0]["content"].startswith(N.LOGIC_SYSTEM_PROMPT))
        self.assertNotIn("clause_quote_mode", plain)

    def test_unknown_subject_is_still_pending_after_locatable_quote_selection(self):
        item = {"feedback_item_id": "sf-b", "fact_id": "missing-evidence:sf", "source_id": "",
            "subject_name": "乙方", "fact_type": "license", "clause_id": "b", "clause_ids": ["b"],
            "source_complete": False, "effective_status": "unknown", "fact_evidence_text": "资质状态unknown。",
            "source_evidence_text": ""}
        captured, meta, _ = self.run_reply(items=[item], subject=True)
        props = captured[0][2]["json_schema"]["properties"]["subject_fact_verifications"]["items"]["properties"]
        self.assertEqual(props["clause_quote"]["enum"], [self.b.text])
        self.assertEqual(set(props["verdict"]["enum"]), {"confirmed", "rejected", "needs_manual"})
        self.assertNotIn("enum", props["fact_quote"])
        self.assertNotIn("enum", props["source_quote"])
        self.assertFalse(meta["feedback_complete"])
        self.assertEqual(meta["uncompleted_feedback_item_ids"], ["sf-b"])


if __name__ == "__main__":
    unittest.main()
