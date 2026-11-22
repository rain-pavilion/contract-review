"""把 refs.bib 摘要成"作者 | 标题 | venue | year | vol | no | pages | doi"清单，
用于在论文参考文献列表里做**逐条核对**（避免手抄出错）。

    python scripts/_tmp_bib_dump.py
"""

from __future__ import annotations

import re
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "zhilv_qiling_v2" / "docs" / "refs" / "refs.bib"


def field(body: str, name: str) -> str:
    match = re.search(r"\n\s*" + name + r"\s*=\s*\{(.*?)\},?\n", body, re.S)
    return " ".join(match.group(1).split()) if match else ""


def main() -> int:
    text = BIB.read_text(encoding="utf-8")
    entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S)
    print(f"# entries: {len(entries)}")
    for kind, key, body in entries:
        print(" | ".join([
            key,
            field(body, "author")[:70],
            field(body, "title")[:90],
            field(body, "journal") or field(body, "booktitle"),
            field(body, "year"),
            field(body, "volume"),
            field(body, "number"),
            field(body, "pages"),
            field(body, "doi"),
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
