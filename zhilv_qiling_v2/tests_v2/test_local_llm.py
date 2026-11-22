"""local_llm 与 lawbench 的契约测试（不需要 torch / 真实模型，可在 CI 里跑）。"""
from __future__ import annotations

import json
import pathlib
import unittest

from zhilv_qiling_v2 import lawbench as LB
from zhilv_qiling_v2 import local_llm as LL


class LocalLlmContractTest(unittest.TestCase):
    def test_strip_think_removes_reasoning_block(self) -> None:
        raw = "<think>先看条款有没有期限。</think>属于履约风险。"
        self.assertEqual(LL.strip_think(raw), "属于履约风险。")

    def test_strip_think_keeps_plain_text(self) -> None:
        self.assertEqual(LL.strip_think("  没有思考段  "), "没有思考段")

    def test_extract_json_plain(self) -> None:
        payload = LL.extract_json('{"defects": [{"code": "zero_rent"}]}')
        self.assertEqual(payload["defects"][0]["code"], "zero_rent")

    def test_extract_json_with_fence_and_prose(self) -> None:
        raw = '好的，结果如下：\n```json\n{"defects": []}\n```\n以上。'
        self.assertEqual(LL.extract_json(raw), {"defects": []})

    def test_extract_json_ignores_think_block(self) -> None:
        raw = '<think>{"defects": 1}</think>{"defects": []}'
        self.assertEqual(LL.extract_json(raw), {"defects": []})

    def test_extract_json_returns_empty_on_garbage(self) -> None:
        self.assertEqual(LL.extract_json("完全不是 JSON"), {})
        self.assertEqual(LL.extract_json("[1, 2, 3]"), {})

    def test_find_model_dir_rejects_missing_path(self) -> None:
        missing = pathlib.Path("__definitely_not_a_model__")
        # 传一个不存在的目录时不应抛异常；有回退目录就返回它，否则 None
        found = LL.find_model_dir(missing)
        self.assertTrue(found is None or found.is_dir())

    def test_describe_before_load(self) -> None:
        info = LL.LocalLLM(model_dir=None).describe()
        for key in ("model_dir", "device", "ready", "calls", "max_new_tokens"):
            self.assertIn(key, info)
        self.assertFalse(info["ready"])


class LawbenchContractTest(unittest.TestCase):
    def test_task_table_ids_unique_and_typed(self) -> None:
        ids = [row["lawbench"] for row in LB.TASK_TABLE]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 6)
        for row in LB.TASK_TABLE:
            self.assertTrue(row["type"])
            self.assertTrue(row["ours"])
            self.assertTrue(row["metric"])

    def test_tasks_cover_table_entries(self) -> None:
        table_types = {row["lawbench"] for row in LB.TASK_TABLE}
        produced = {func.__doc__ is not None for func in LB.TASKS.values()}
        self.assertTrue(all(produced))
        self.assertIn("2-6 Extraction", table_types)
        self.assertEqual(set(LB.TASKS), {"extraction", "detection", "retrieval",
                                         "entailment", "subject", "generation"})

    def test_safe_swallows_failure(self) -> None:
        def boom() -> LB.TaskResult:
            raise RuntimeError("模拟失败")

        self.assertIsNone(LB._safe("boom", boom))

    def test_run_single_task_without_save(self) -> None:
        payload = LB.run(tasks=("generation",), save=False)
        self.assertIn("caveat", payload)
        self.assertEqual(len(payload["results"]), 1)
        row = payload["results"][0]
        self.assertEqual(row["lawbench_id"], "generation")
        self.assertIn("local_model_available", row["metrics"])
        json.dumps(payload, ensure_ascii=False)  # 必须可序列化（会被写进 reports/）

    def test_format_table_mentions_ids(self) -> None:
        payload = LB.run(tasks=("generation",), save=False)
        text = LB.format_table(payload)
        self.assertIn("generation", text)
        self.assertIn("LawBench", text)


if __name__ == "__main__":
    unittest.main()
