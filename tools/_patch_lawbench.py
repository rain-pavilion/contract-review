"""一次性补丁：修 lawbench.py 的 n 键名与 subject 任务的金标缺失保护。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

target = ROOT / "zhilv_qiling_v2" / "lawbench.py"
text = target.read_text(encoding="utf-8")

pairs = [
    (
        'f"强金标准 {row.get(\'n_queries\', 0)} 条查询（按标签族定义）"',
        'f"强金标准 {row.get(\'n\', 0)} 条查询（按标签族定义）"',
    ),
    ('n=int(row.get("n_queries", 0) or 0),', 'n=int(row.get("n", 0) or 0),'),
]
for old, new in pairs:
    if old in text:
        text = text.replace(old, new)
        print(f"ok  {old[:48]}")

guard = (
    "    if tp + fn == 0:\n"
    "        # 该批样本的金标里没有 need_license，单码代理指标无从计算，如实报空表而不是报 0\n"
    "        return TaskResult(\n"
    '            lawbench_id="1-2 SLC (4 类)",\n'
    '            task_type="single-label",\n'
    '            ours="节点③ 主体资格审查（资质知识图谱 + RAG）",\n'
    '            dataset="合成对抗集 10 份",\n'
    "            n=0,\n"
    "            metrics={},\n"
    '            notes=f"金标无 need_license 标注（检出 {fp} 条，均无金标）；该任务需人工标注子集",\n'
    "        )\n"
    "    total = max(tp + fp + fn, 1)"
)
if "if tp + fn == 0:" not in text:
    assert "    total = max(tp + fp + fn, 1)" in text
    text = text.replace("    total = max(tp + fp + fn, 1)", guard, 1)
    print("ok  subject guard")

target.write_text(text, encoding="utf-8")
import importlib

import zhilv_qiling_v2.lawbench as W  # noqa: E402

importlib.reload(W)
print("compile ok, tasks =", sorted(W.TASKS))
