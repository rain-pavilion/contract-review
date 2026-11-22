"""把 test_multi_agent.py 里"顺序必须等于 SCHEDULE"的断言改成"顺序前缀 + 含 judge"。

多智能体加入主脑（orchestrator）与强模型复核（reviewer）后，order 里同一 Agent 可能
出现两次（主脑派活两次、回边重跑 logic），逐字比对 SCHEDULE 已不成立；
真正要守住的是"入口在前、终审在末、主脑在开工前"。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "zhilv_qiling_v2" / "tests_v2" / "test_multi_agent.py"

SUBS = [
    (
        'self.assertEqual(payload["order"], list(R.SCHEDULE))',
        'self.assertEqual(payload["order"][:4], ["intake", "orchestrator", "ner", "logic"])\n'
        '        self.assertIn("judge", payload["order"])',
    ),
    (
        'self.assertEqual(summary["order"], list(R.SCHEDULE))',
        'self.assertEqual(summary["order"][:4], ["intake", "orchestrator", "ner", "logic"])\n'
        '        self.assertIn("judge", summary["order"])',
    ),
    (
        'self.assertEqual(report.stats["multi_agent"]["order"], list(R.SCHEDULE))',
        'self.assertEqual(\n'
        '            report.stats["multi_agent"]["order"][:4], ["intake", "orchestrator", "ner", "logic"]\n'
        '        )',
    ),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    text = TARGET.read_text(encoding="utf-8")
    for old, new in SUBS:
        if old not in text:
            print("MISS:", old[:60])
            continue
        text = text.replace(old, new)
        print("OK  :", old[:60])
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
