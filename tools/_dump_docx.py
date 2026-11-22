# -*- coding: utf-8 -*-
"""把一个 docx 按 body 顺序导出成纯文本（段落 + 表格），用于逐段核对内容。

用法：
    python tools/_dump_docx.py <docx 路径> [输出 txt 路径]
不传输出路径时写到 %TEMP% 下同名 txt，并打印路径与行数。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_body(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def dump(path: pathlib.Path) -> tuple[str, int]:
    doc = Document(str(path))
    lines: list[str] = []
    table_no = 0
    for block in iter_body(doc):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style is not None else ""
            text = block.text.strip()
            if not text:
                continue
            lines.append(f"[{style}] {text}")
        else:
            table_no += 1
            lines.append(f"=== TABLE {table_no} ===")
            for row in block.rows:
                cells = [c.text.strip().replace("\n", " / ") for c in row.cells]
                if any(cells):
                    lines.append("  | " + " | ".join(cells))
    return "\n".join(lines), table_no


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: python tools/_dump_docx.py <docx> [out.txt]")
    src = pathlib.Path(sys.argv[1])
    text, tables = dump(src)
    out = (
        pathlib.Path(sys.argv[2])
        if len(sys.argv) > 2
        else pathlib.Path(tempfile.gettempdir()) / (src.stem + "_dump.txt")
    )
    out.write_text(text, encoding="utf-8")
    print(f"源文件: {src}")
    print(f"导出:   {out}")
    print(f"行数:   {len(text.splitlines())}   表格: {tables}   字符: {len(text)}")
