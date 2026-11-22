"""把 NER 的最新实测数字同步进 README 与打磨记录（不改代码）。

背景：`bert_ner.predict()` 加了金额两道后处理闸后，严格口径微平均 F1 从
0.8776 升到 0.9505（2026-11-13 重算，见 `reports/gpu_upgrade_bert_ner.json`
的 `new.valid.micro`）。README 里还残留 0.5348 / 0.7514 等更早的数字。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "zhilv_qiling_v2" / "README.md"
NOTE = ROOT / "docs" / "NER打磨记录.md"

# (老文本, 新文本)；只在命中时才替换，命中数会打印出来
RULES_README = [
    (
        "| NER 宏 F1（GPU） | 字符感知机 0.0179 → **BERT-CRF 0.5348**（规则通道 0.4204） |",
        "| NER 微平均 F1（GPU） | 字符感知机 0.0179 → 规则 0.518 → **BERT-CRF 0.9505**（严格口径） |",
    ),
    (
        "| 节点① 要素抽取 | NER F1 ≥ 90% | `bert_ner.py`：bert-base-chinese + BERT-CRF（`pytorch-crf`），313 个训练窗口，10 epoch / 192 s | 宏 F1 **0.5348**（旧字符级感知机 0.0179、规则通道 0.4204）；`AMOUNT` 0.786、`DATE` 0.718、`PARTY` 0.635（偏召回 P 0.46）、`SUBJECT` 0 |",
        "| 节点① 要素抽取 | NER F1 ≥ 90% | `bert_ner.py`：bert-base-chinese + BERT-CRF（`pytorch-crf`），金标 3 倍过采样、1082 个训练窗口；推理侧加大写重复/单价两道金额闸 | **微平均 F1 0.9505**（P 0.9058 / R 1.0000）、补全口径 0.9238、宏 0.9508；逐类 `PARTY` 0.9767、`AMOUNT` 0.8831、`DATE` 0.9434、`SUBJECT` 1.0000 |",
    ),
    (
        "| 2-6 Extraction | 节点① 要素抽取 | micro-F1 **0.7514**（补全金标口径 **0.8812**）、macro 0.7300 |",
        "| 2-6 Extraction | 节点① 要素抽取 | micro-F1 **0.9505**（补全金标口径 0.9238）、macro 0.9508 |",
    ),
    (
        "| NER 模型权重 | —— | **默认关闭**（实测 CPU 感知机 F1 仅 0.0179，见下） |",
        "| NER 模型权重 | —— | **默认开启**（`ner_use_bert=True`；权重缺席时自动降级到规则通道） |",
    ),
]

RULES_NOTE = [
    (
        "| 补全（`micro_complete`） | 金标 = 清单字段 + 正文全部金额/日期（规则补全，须含数字，PARTY/SUBJECT 不补） | 0.9519 | 0.8241 | **0.8834** |",
        "| 补全（`micro_complete`） | 金标 = 清单字段 + 正文全部金额/日期（规则补全，须含数字，PARTY/SUBJECT 不补） | 0.9843 | 0.8704 | **0.9238** |",
    ),
    (
        "| **BERT-CRF（现在）** | **0.7535** | 严格口径；补全口径 0.8834 |",
        "| **BERT-CRF（金标过采样后）** | **0.8782** | 严格口径；补全 0.9862 |\n| **BERT-CRF（+金额两道闸，现在）** | **0.9505** | 严格口径 P0.9058/R1.0000；补全 0.9238；宏 0.9508 |",
    ),
]


def apply(path: pathlib.Path, rules: list[tuple[str, str]]) -> int:
    text = path.read_text(encoding="utf-8")
    total = 0
    for old, new in rules:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            total += count
            print(f"  ✓ {count} 处：{old[:46]}…")
        else:
            print(f"  · 未命中：{old[:60]}…")
    path.write_text(text, encoding="utf-8")
    return total


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("README：")
    n1 = apply(README, RULES_README)
    print("NER打磨记录：")
    n2 = apply(NOTE, RULES_NOTE)
    print(f"合计替换 {n1 + n2} 处")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
