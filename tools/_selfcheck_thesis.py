# -*- coding: utf-8 -*-
"""论文生成器自检脚本。

做四件事：
1. 调用 ``thesis_builder.build_demo()`` 生成 ``docs/thesis_demo.docx``；
2. 用 ``python-docx`` 重新打开成品，逐段打印 ``style.name`` 与文本前 30 字；
3. 检查是否含 ``TOC`` 域、``sectPr`` 页边距、页数与样式使用统计；
4. 在控制台打印 ASCII 摘要，同时把完整报告写成 UTF-8 文本文件
   （Windows 控制台默认 GBK，直接输出中文会乱码，所以落盘一份便于核对）。

用法::

    python tools/_selfcheck_thesis.py [输出目录]

不带参数时输出到 ``<仓库根>/docs/thesis_demo.docx``。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# 让脚本能直接 import 同目录的 thesis_builder（不依赖安装）
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docx import Document  # noqa: E402  (要在 sys.path 调整之后导入)
from docx.oxml.ns import qn  # noqa: E402

from thesis_builder import DEFAULT_TEMPLATE, build_demo  # noqa: E402

TEXT_PREVIEW = 30


def _instruction_texts(document: Document) -> list[str]:
    """收集文档中所有域代码（instrText）。

    页脚的页码域在独立的 footer part 里，不在 ``document.element.body`` 下，
    所以这里同时遍历每一节的页眉与页脚，否则页码域会被漏计。
    """
    texts = [el.text or "" for el in document.element.body.iter(qn("w:instrText"))]
    for section in document.sections:
        for part in (section.header, section.footer,
                     section.first_page_header, section.first_page_footer,
                     section.even_page_header, section.even_page_footer):
            if part is None or part.is_linked_to_previous:
                continue
            try:
                element = part._element  # 页眉/页脚部分的根元素
            except AttributeError:  # pragma: no cover
                continue
            texts.extend(el.text or "" for el in element.iter(qn("w:instrText")))
    return texts


def _style_chain(style: object) -> list[object]:
    """从样式沿 basedOn 链往回走，直到无父样式（用于解算继承后的有效值）。"""
    chain: list[object] = []
    current = style
    while current is not None and current not in chain:
        chain.append(current)
        current = getattr(current, "base_style", None)
    return chain


def _first_in_chain(style: object, getter: object) -> tuple[object, str | None]:
    """沿样式链找第一个非 None 的属性值，返回 (值, 定义它的样式名)。"""
    for item in _style_chain(style):
        try:
            value = getter(item)  # type: ignore[operator]
        except (AttributeError, KeyError, ValueError):
            value = None
        if value is not None:
            return value, getattr(item, "name", None)
    return None, None


def _eastasia_font(style: object) -> str | None:
    """读样式 XML 里的 rFonts/@w:eastAsia（python-docx 不暴露该属性）。"""
    element = getattr(style, "element", None)
    rpr = None if element is None else element.find(qn("w:rPr"))
    rfonts = None if rpr is None else rpr.find(qn("w:rFonts"))
    return None if rfonts is None else rfonts.get(qn("w:eastAsia"))


def _audit_style(document: Document, style_name: str) -> list[str]:
    """抽查某样式的首个段落：段落级直接格式 + 沿样式链解算出的有效字体字号。"""
    for para in document.paragraphs:
        if para.style is not None and para.style.name == style_name and para.text.strip():
            style = para.style
            east, east_src = _first_in_chain(style, _eastasia_font)
            latin, latin_src = _first_in_chain(style, lambda s: s.font.name)
            size, size_src = _first_in_chain(style, lambda s: s.font.size)
            bold, bold_src = _first_in_chain(style, lambda s: s.font.bold)
            align, align_src = _first_in_chain(style, lambda s: s.paragraph_format.alignment)
            spacing, spacing_src = _first_in_chain(style, lambda s: s.paragraph_format.line_spacing)
            before, before_src = _first_in_chain(style, lambda s: s.paragraph_format.space_before)
            fmt = para.paragraph_format
            return [
                f"样式={style_name}（继承链 {len(_style_chain(style))} 级）",
                f"  文本: {para.text[:24]}",
                f"  中文字体={east}（来自「{east_src}」） 西文字体={latin}（来自「{latin_src}」）",
                f"  字号={None if size is None else round(size.pt, 2)}pt"
                f"（来自「{size_src}」） 加粗={bold}（来自「{bold_src}」）",
                f"  对齐={align}（来自「{align_src}」） "
                f"行距={spacing}（来自「{spacing_src}」）",
                f"  样式段前={_tw(before)}（来自「{before_src}」） "
                f"段落直接段前={_tw(fmt.space_before)} 首行缩进={_tw(fmt.first_line_indent)}",
                f"  段落直接段前分页={fmt.page_break_before} "
                f"与下段同页={fmt.keep_with_next} 左缩进={_tw(fmt.left_indent)}",
            ]
    return [f"样式={style_name}: 未找到含文本的段落"]


def _tw(length: object) -> str:
    """把 Length 转成「Ntwips(Npt)」形式，便于与样本实测值对照。"""
    if length is None:
        return "None"
    twips = round(length.twips, 1)
    return f"{twips}twips({round(length.pt, 2)}pt)"


def main(argv: list[str]) -> int:
    """执行自检，返回进程退出码（0 = 通过）。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_path = (Path(argv[1]) if len(argv) > 1 else REPO_ROOT / "docs" / "thesis_demo.docx")
    if out_path.is_dir():
        out_path = out_path / "thesis_demo.docx"

    lines: list[str] = []
    def emit(text: str = "") -> None:
        lines.append(text + "\n")
        print(text)

    emit("=" * 78)
    emit("论文格式生成器自检 / thesis builder self-check")
    emit("=" * 78)
    emit(f"模板: {DEFAULT_TEMPLATE}")
    emit(f"模板存在: {DEFAULT_TEMPLATE.exists()}")
    emit(f"输出: {out_path}")

    report = build_demo(out_path)
    emit("")
    emit(f"生成完成: {report['file']}")
    emit(f"文件大小: {report['size_kb']} KB ({report['size_bytes']} 字节)")
    emit(f"段落数: {report['paragraph_count']}  表格数: {report['table_count']}")
    emit(f"含 TOC 域: {report['has_toc_field']}  域代码: {report['toc_instr']}")
    setup = report["page_setup"]
    emit("页面设置: " + ", ".join(f"{k}={v}" for k, v in setup.items()))

    # 回读校验：逐段样式 + 文本前 30 字
    document = Document(str(out_path))
    emit("")
    emit("-" * 78)
    emit(f"{'idx':>3}  {'样式':<14} 文本前 {TEXT_PREVIEW} 字")
    emit("-" * 78)
    style_counter: Counter[str] = Counter()
    for idx, para in enumerate(document.paragraphs):
        style_name = para.style.name if para.style is not None else "<无样式>"
        style_counter[style_name] += 1
        preview = para.text.replace("\n", " ").replace("\t", "→")[:TEXT_PREVIEW]
        emit(f"{idx:>3}  {style_name:<14} {preview}")

    emit("-" * 78)
    emit("样式使用统计:")
    for name, count in style_counter.most_common():
        emit(f"  {count:>3}  {name}")

    emit("")
    emit("表格:")
    for idx, table in enumerate(document.tables):
        header = " | ".join(cell.text for cell in table.rows[0].cells)
        emit(f"  [{idx}] {len(table.rows)}x{len(table.columns)}  表头: {header}")

    instr = _instruction_texts(document)
    toc_found = [i for i in instr if "TOC" in i]
    page_found = [i for i in instr if "PAGE" in i]
    emit("")
    emit(f"TOC 域数量: {len(toc_found)}  页码域数量: {len(page_found)}")

    emit("")
    emit("-" * 78)
    emit("关键样式实测抽查（与样本对照）")
    emit("-" * 78)
    for style_name in ("论文1级标题", "论文2级标题", "论文3级标题", "论文正文", "Caption"):
        for line in _audit_style(document, style_name):
            emit(line)
    emit(f"sections 数量: {len(document.sections)}")
    for idx, section in enumerate(document.sections):
        emit(f"  section[{idx}] {section.page_width.cm:.2f}x{section.page_height.cm:.2f}cm "
             f"边距 上{section.top_margin.cm:.2f}/下{section.bottom_margin.cm:.2f}/"
             f"左{section.left_margin.cm:.2f}/右{section.right_margin.cm:.2f}cm")

    ok = (
        report["paragraph_count"] > 20
        and toc_found
        and page_found
        and abs(document.sections[0].left_margin.cm - 3.18) < 0.02
    )
    emit("")
    emit(f"自检结论: {'通过 / PASS' if ok else '需要检查 / FAIL'}")

    # 完整报告落盘（UTF-8），便于在控制台编码不支持中文时核对
    import tempfile

    report_file = Path(tempfile.gettempdir()) / "thesis_selfcheck_report.txt"
    report_file.write_text("".join(lines), encoding="utf-8")
    json_file = Path(tempfile.gettempdir()) / "thesis_selfcheck_report.json"
    json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整报告: {report_file}")
    print(f"校验 JSON: {json_file}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
