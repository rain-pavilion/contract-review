# -*- coding: utf-8 -*-
"""把 docx 的兼容模式改到 Word 2013+（compatibilityMode=15），并清掉只读标记。

python-docx 用的是内置默认模板，settings.xml 里 compatSetting 常为旧值，
Word 打开就进“兼容性模式”，行距与表格渲染会按 2007 规则走。这里直接改包。
"""
from __future__ import annotations

import pathlib
import re
import shutil
import zipfile

COMPAT_RE = re.compile(
    r'(<w:compatSetting[^>]*w:name="compatibilityMode"[^>]*w:val=")(\d+)(")'
)
COMPAT_NEW = (
    '<w:compat><w:compatSetting w:name="compatibilityMode" '
    'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
)


def _patch_settings(xml: bytes) -> tuple[bytes, str]:
    text = xml.decode("utf-8")
    match = COMPAT_RE.search(text)
    if match:
        if match.group(2) == "15":
            return xml, "already-15"
        old = match.group(2)
        text = text[: match.start(2)] + "15" + text[match.end(2):]
        return text.encode("utf-8"), f"{old}->15"
    if "<w:compat>" in text:
        text = text.replace(
            "<w:compat>",
            '<w:compat><w:compatSetting w:name="compatibilityMode" '
            'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/>',
            1,
        )
        return text.encode("utf-8"), "patched-compat-block"
    if "</w:settings>" in text:
        text = text.replace("</w:settings>", COMPAT_NEW + "</w:settings>", 1)
        return text.encode("utf-8"), "inserted-compat"
    return xml, "no-settings-anchor"


def _patch_app(xml: bytes) -> tuple[bytes, str]:
    text = xml.decode("utf-8")
    match = re.search(r"<DocSecurity>(\d+)</DocSecurity>", text)
    if match:
        if match.group(1) == "0":
            return xml, "already-0"
        text = text.replace(match.group(0), "<DocSecurity>0</DocSecurity>", 1)
        return text.encode("utf-8"), f"{match.group(1)}->0"
    return xml, "no-docsecurity"


def apply(path: pathlib.Path) -> dict:
    path = pathlib.Path(path)
    tmp = path.with_name(path.stem + ".__tmp__.docx")
    report: dict[str, str] = {}
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/settings.xml":
                data, note = _patch_settings(data)
                report["settings"] = note
            elif item.filename == "docProps/app.xml":
                data, note = _patch_app(data)
                report["app"] = note
            dst.writestr(item, data)
    shutil.move(str(tmp), str(path))
    report["bytes"] = str(path.stat().st_size)
    return report


if __name__ == "__main__":
    import json
    import sys

    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs/期刊论文-第一页-马宇轩.docx")
    print(json.dumps(apply(target), ensure_ascii=False, indent=2))
