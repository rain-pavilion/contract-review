"""把毕业论文 docx 的目录页码刷成真实页码，并顺手修两处会让 Word 弹「兼容性模式」的元数据。

为什么需要这个脚本：
    生成器 `tools/build_thesis_v2.py` 用 python-docx 写文档，它算不出分页结果，
    所以插入的 `TOC` 域里带的是**按字数估算**的静态页码（实测偏差 1~3 页）。
    模板又是从 `.doc` 转来的，`settings.xml` 里 `compatibilityMode=11`（Word 2003），
    打开时标题栏会挂「兼容性模式」，行距与表格自适应也会按旧规则渲染。

本脚本做三件事（全部幂等，可反复跑）：
    1. 把 `compatibilityMode` 改成 15（Word 2013+）；
    2. 把 `docProps/app.xml` 的 `DocSecurity` 归零，避免「建议以只读方式打开」；
    3. 借 Word COM 更新 `TablesOfContents(1)` 并保存，页码变成真实分页结果。

用法：
    python tools/fix_thesis_toc.py                       # 处理默认那份论文
    python tools/fix_thesis_toc.py path/to/xxx.docx      # 处理指定文件
    python tools/fix_thesis_toc.py --no-word             # 只改 XML，不更新目录

注意：需要本机装有 Word（COM）。没有 Word 时脚本只做第 1、2 步并在末尾提示手工 F9。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
_DOCS = WORKSPACE / "docs"
# 论文 docx 可能被用户改名（去掉"毕业论文-"前缀），按存在性自动发现，避免死路径
DEFAULT_DOCX = next(
    (_DOCS / _n for _n in ("智律契灵合同智能审查系统v2.docx",
                           "毕业论文-智律契灵合同智能审查系统v2.docx")
     if (_DOCS / _n).exists()),
    _DOCS / "智律契灵合同智能审查系统v2.docx",
)

COMPAT_FROM = 'w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="11"'
COMPAT_TO = 'w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"'


def patch_xml(path: Path) -> dict[str, bool]:
    """原位重写 zip 里的 settings.xml 与 app.xml。返回每处是否真的改了。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = {"compat": False, "docsecurity": False}
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/settings.xml":
                text = data.decode("utf-8")
                if COMPAT_FROM in text:
                    text = text.replace(COMPAT_FROM, COMPAT_TO)
                    changed["compat"] = True
                data = text.encode("utf-8")
            elif item.filename == "docProps/app.xml":
                text = data.decode("utf-8")
                new = re.sub(r"<DocSecurity>\d+</DocSecurity>", "<DocSecurity>0</DocSecurity>", text)
                if new != text:
                    changed["docsecurity"] = True
                data = new.encode("utf-8")
            zout.writestr(item, data)
    os.replace(tmp, path)
    return changed


def update_toc_with_word(path: Path) -> bool:
    """用 Word COM 更新目录域并存盘。成功返回 True。"""
    try:
        import comtypes.client  # type: ignore
    except ImportError:
        pass
    try:
        import win32com.client  # type: ignore
        word = win32com.client.Dispatch("Word.Application")
    except Exception:
        # 退回 PowerShell 侧的 COM 调用，仍失败就返回 False
        return _update_toc_via_powershell(path)

    word.Visible = False
    doc = None
    try:
        doc = word.Documents.Open(str(path))
        if doc.TablesOfContents.Count == 0:
            return False
        doc.TablesOfContents.Item(1).Update()
        doc.Save()
        return True
    finally:
        if doc is not None:
            doc.Close(0)
        word.Quit()


def _update_toc_via_powershell(path: Path) -> bool:
    import subprocess

    script = (
        "$w=[Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application');"
        f"$d=$w.Documents.Open('{path}');"
        "if($d.TablesOfContents.Count -gt 0){$d.TablesOfContents.Item(1).Update();$d.Save();Write-Output 'OK'}"
        "$d.Close(0)"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=180,
        )
        return "OK" in (out.stdout or "")
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新论文目录页码并修兼容性元数据")
    parser.add_argument("docx", nargs="?", default=str(DEFAULT_DOCX))
    parser.add_argument("--no-word", action="store_true", help="只改 XML，不更新目录域")
    parser.add_argument("--no-backup", action="store_true", help="不留 .bak 备份")
    args = parser.parse_args(argv)

    path = Path(args.docx)
    if not path.is_file():
        print(f"[×] 文件不存在：{path}")
        return 1

    if not args.no_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"[·] 备份 {backup.name}（{backup.stat().st_size} B）")

    changed = patch_xml(path)
    print(f"[·] compatibilityMode 11→15：{'已改' if changed['compat'] else '本来就是 15'}")
    print(f"[·] DocSecurity → 0：{'已改' if changed['docsecurity'] else '本来就是 0'}")

    if args.no_word:
        print("[!] 跳过目录更新；请在 Word 里 Ctrl+A 再按 F9 手动更新域")
        return 0

    ok = update_toc_with_word(path)
    if ok:
        print("[✓] 目录页码已刷新为真实分页结果并存盘")
        return 0

    print("[!] 未能通过 Word 更新目录（可能没装 Word 或 COM 被占用）")
    print("    打开文档后 Ctrl+A → F9 → 只更新目录，再保存即可")
    return 0


if __name__ == "__main__":
    sys.exit(main())
