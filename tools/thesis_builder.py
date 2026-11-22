# -*- coding: utf-8 -*-
"""南通大学计算机学院本科毕业设计论文 .docx 生成器（只用 python-docx，无网络依赖）。

设计要点
--------
1. **样式 100% 继承自样本**：``ThesisBuilder`` 以学校样本 docx 为模板
   （``Document(template_path)``），随后清空 body 正文内容，但**保留**
   ``styles.xml``、``sectPr``（页面设置）、``numbering.xml`` 与页眉页脚关系。
   这样 论文1级标题 / 论文2级标题 / 论文3级标题 / 论文正文 / Caption /
   目录标题格式 / toc 1~3 等样式的字体字号行距段距全部来自样本本身，
   不需要在代码里重复抄写一遍数值。

2. **标题不自动编号**：样本里每个标题段落都带
   ``<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>``，
   即显式关闭样式自带的 ``第%1章`` 多级列表；章号是正文里的字面文本。
   本生成器同样写 ``numId=0`` 覆盖，所以 ``h1('第1章  绪论')`` 不会出现双重编号。

3. **目录兜底**：``toc_field()`` 写入真正的 ``TOC \\o "1-3" \\h \\z \\u`` 域，
   同时把已登记的各级标题写成静态目录条目（toc 1/2/3 样式 + 右对齐点状制表位 +
   页码占位），这样在"打开不更新域"的机器上也能看到目录。

典型用法
--------
>>> b = ThesisBuilder(TEMPLATE, 'out.docx')
>>> b.cover(title='…', student='…', advisor='…', student_id='…',
...         ref_count=28, major='计算机科学与技术', word_count=21000,
...         date='2025年5月18日')
>>> b.section_break(page_number_format='upperRoman')   # 目录/摘要用罗马页码
>>> b.toc_field()
>>> b.abstract_cn('中文摘要正文…', ['水产品', '追溯系统'])
>>> b.abstract_en('English abstract …', ['Aquatic Products', 'Traceability'])
>>> b.section_break(page_number_format='decimal')      # 正文用阿拉伯页码
>>> b.h1('第1章  绪论')
>>> b.h2('1.1  研究背景及意义')
>>> b.body('正文段落……')
>>> report = b.save()
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from docx import Document
from docx.enum.section import WD_SECTION, WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips
from docx.table import Table
from docx.text.paragraph import Paragraph

__all__ = ["ThesisBuilder", "build_demo", "DEFAULT_TEMPLATE", "SAMPLE_WIDTH_TWIPS"]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 学校样本（只读，不要改动它）
DEFAULT_TEMPLATE = Path(
    r"D:\xwechat_files\wxid_gsy7l1w31fm522_d68f\msg\file\2026-10\计科4班彭雅玲_conv.docx"
)

#: 段落样式名（样本实测，见 docs/thesis_format_spec.md）
STYLE_H1 = "论文1级标题"
STYLE_H2 = "论文2级标题"
STYLE_H3 = "论文3级标题"
STYLE_BODY = "论文正文"
STYLE_CAPTION = "Caption"
STYLE_TOC_TITLE = "目录标题格式"
STYLE_TOC = ("toc 1", "toc 2", "toc 3")

#: 正文可用宽度（twips）：21.0cm - 3.18cm - 3.18cm = 14.64cm = 8290 twips
SAMPLE_WIDTH_TWIPS = 8290

#: 表格/插图实际可用宽度（cm）：版心 14.64 留 0.04 余量，避免 Word 因取整把内容顶出右边界
BODY_TABLE_CM = 14.60

#: 占位符（封面缺值时使用）
PLACEHOLDER = "【待填】"

#: 标题段前距（twips，样本实测：1 级 326、2/3 级 163）
_SPACE_BEFORE_H1 = 326
_SPACE_BEFORE_H23 = 163

#: 正文首行缩进（twips，样本正文段落直接格式 firstLine=480 = 2 字符 @12pt）
_BODY_FIRST_LINE = 480

#: 自动目录域代码（样本实测：1~3 级、超链接、隐藏空级别、使用大纲级别）
_TOC_INSTR = 'TOC \\o "1-3" \\h \\z \\u'

#: 目录域尚未刷新时的提示文字（静态目录会在 save() 阶段按实际标题补全并替换它）
_TOC_PLACEHOLDER = "右键“更新域”生成目录"

# pPr / rPr 子元素顺序（ECMA-376 CT_PPrBase / CT_RPr 的序列，插错顺序 Word 会报文档损坏）
_PPR_ORDER: tuple[str, ...] = (
    "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
    "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
    "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
    "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind",
    "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
    "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
    "sectPr", "pPrChange",
)

_RPR_ORDER: tuple[str, ...] = (
    "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike",
    "dstrike", "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid",
    "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz",
    "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign",
    "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath",
)


# ---------------------------------------------------------------------------
# 底层 XML 小工具
# ---------------------------------------------------------------------------

def _local(tag: Any) -> str:
    """取 lxml 标签的本地名（去掉 ``{namespace}``）。"""
    return str(tag).split("}")[-1]


def _w_el(tag: str, val: str | None = None) -> Any:
    """新建一个 w: 命名空间元素，可选设置 ``w:val``。"""
    el = OxmlElement(tag)
    if val is not None:
        el.set(qn("w:val"), val)
    return el


def _insert_ordered(parent: Any, child: Any, order: Sequence[str]) -> Any:
    """按 schema 顺序把 ``child`` 插入 ``parent``，避免 Word 判定文档损坏。"""
    name = _local(child.tag)
    if name not in order:  # 未知元素直接追加
        parent.append(child)
        return child
    idx = order.index(name)
    for existing in list(parent):
        if not isinstance(existing.tag, str):
            continue
        other = _local(existing.tag)
        if other in order and order.index(other) > idx:
            existing.addprevious(child)
            return child
    parent.append(child)
    return child


def _get_or_add(parent: Any, tag: str) -> Any:
    """取父元素下的直接子元素，没有就新建（不带 w:val）。"""
    found = parent.find(qn(tag))
    if found is None:
        found = OxmlElement(tag)
        parent.append(found)
    return found


def _set_pPr_child(paragraph: Paragraph, tag: str, **attrs: str) -> Any:
    """按 schema 顺序写入/更新段落 pPr 的某个子元素。"""
    pPr = paragraph._p.get_or_add_pPr()
    holder = pPr.find(qn(tag))
    if holder is None:
        holder = OxmlElement(tag)
        _insert_ordered(pPr, holder, _PPR_ORDER)
    for key, value in attrs.items():
        holder.set(qn("w:" + key), value)
    return holder


def _remove_pPr_child(paragraph: Paragraph, tag: str) -> None:
    """删除段落 pPr 下的某个子元素（用于清理样本残留）。"""
    pPr = paragraph._p.pPr
    if pPr is None:
        return
    for el in pPr.findall(qn(tag)):
        pPr.remove(el)


def _suppress_numbering(paragraph: Paragraph) -> None:
    """写 ``<w:numPr><w:ilvl 0/><w:numId 0/></w:numPr>`` 关闭样式自带自动编号。

    样本里每个标题段落都是这么干的：章节号「第1章」「1.1」由正文文本提供，
    样式自带的 ``第%1章  `` 多级列表被 numId=0 显式压掉。
    """
    pPr = paragraph._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:numPr")):
        pPr.remove(old)
    num_pr = OxmlElement("w:numPr")
    num_pr.append(_w_el("w:ilvl", "0"))
    num_pr.append(_w_el("w:numId", "0"))
    _insert_ordered(pPr, num_pr, _PPR_ORDER)


def _display_width(text: str) -> int:
    """文字的显示宽度（以半角字符为单位）：CJK/全角算 2，其余算 1。"""
    total = 0
    for ch in str(text):
        total += 2 if ord(ch) > 0x2E7F else 1
    return total


def auto_column_widths(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                       total_cm: float, font_pt: float = 10.5,
                       *, min_cm: float = 1.5, max_ratio: float = 0.45) -> list[float]:
    """按内容自动分配列宽，避免等分导致"编号"折成两行、宽列大片留白。

    依据：``font_pt`` 下半个汉字宽约 ``font_pt * 0.0353`` cm，单元格左右边距按
    0.35 cm 计；先求每列"不折行所需宽度"，再把剩余宽度按需求比例分配，
    并夹在 ``[min_cm, total_cm*max_ratio]`` 之间（防止某一列独吞整张表）。
    """
    unit = max(font_pt, 9.0) * 0.0353 / 2.0      # 半角字符宽度（cm）
    pad = 0.35                                   # 单元格左右内边距
    need: list[float] = []
    for ci, header in enumerate(headers):
        cells = [header] + [row[ci] if ci < len(row) else "" for row in rows]
        widest = max(_display_width(c) for c in cells)
        need.append(max(widest * unit + pad, min_cm))
    total_need = sum(need)
    if total_need <= total_cm:
        # 需求总和还有剩：按需求比例把余量摊掉（宽列多分，视觉更平衡）
        scale = total_cm / total_need if total_need else 1.0
        widths = [n * scale for n in need]
    else:
        # 需求超了：窄列先拿到自己"不折行"所需的宽度（上限 2.0cm），
        # 让长文本列去承担折行——短列（编号/结论/角色）折行最伤观感
        floors = [min(n, max(min_cm, 2.0)) for n in need]
        rest = max(total_cm - sum(floors), 0.0)
        extra_need = [max(n - f, 0.0) for n, f in zip(need, floors)]
        extra_total = sum(extra_need) or 1.0
        widths = [f + rest * e / extra_total for f, e in zip(floors, extra_need)]
    cap = total_cm * max_ratio
    overflow = 0.0
    for i, w in enumerate(widths):
        if w > cap:
            overflow += w - cap
            widths[i] = cap
    if overflow > 0:
        donors = [i for i, w in enumerate(widths) if w < cap]
        pool = sum(widths[i] for i in donors) or 1.0
        for i in donors:
            widths[i] += overflow * widths[i] / pool
    scale = total_cm / sum(widths)
    return [round(w * scale, 2) for w in widths]


def _paragraph_shading(paragraph: Paragraph, fill: str) -> None:
    """给段落加底纹（``w:shd``），用于代码块浅灰底。"""
    shd = _set_pPr_child(paragraph, "w:shd", val="clear", color="auto", fill=fill)
    shd.set(qn("w:val"), "clear")


def _set_run_font(
    run: Any,
    *,
    ascii_font: str | None = None,
    eastasia_font: str | None = None,
    size_pt: float | None = None,
    bold: bool | None = None,
    color: str | None = None,
) -> None:
    """设置某个 run 的字体（含中文 eastAsia）、字号、加粗、颜色。"""
    rPr = run._r.get_or_add_rPr()
    if ascii_font is not None or eastasia_font is not None:
        rfonts = rPr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            _insert_ordered(rPr, rfonts, _RPR_ORDER)
        if ascii_font is not None:
            rfonts.set(qn("w:ascii"), ascii_font)
            rfonts.set(qn("w:hAnsi"), ascii_font)
        if eastasia_font is not None:
            rfonts.set(qn("w:eastAsia"), eastasia_font)
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def _make_field(paragraph: Paragraph, instr: str, cached: str = "") -> None:
    """在段落末尾插入一个完整的 Word 域（begin / instrText / separate / 缓存值 / end）。"""
    for kind in ("begin",):
        run = paragraph.add_run()
        fld = OxmlElement("w:fldChar")
        fld.set(qn("w:fldCharType"), kind)
        run._r.append(fld)

    run = paragraph.add_run()
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    run._r.append(instr_el)

    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "separate")
    run._r.append(fld)

    if cached:
        paragraph.add_run(cached)

    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "end")
    run._r.append(fld)


def _start_field(paragraph: Paragraph, instr: str) -> None:
    """只写域的 begin + instrText + separate（域结果跨多个段落时用）。"""
    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    run._r.append(fld)

    run = paragraph.add_run()
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    run._r.append(instr_el)

    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "separate")
    run._r.append(fld)


def _end_field(paragraph: Paragraph) -> None:
    """写域的 end 标记（放在域结果的最后一个段落）。"""
    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "end")
    run._r.append(fld)


def _iter_paragraph_text(paragraph: Paragraph) -> str:
    """段落纯文本（python-docx 的 ``paragraph.text`` 即可，这里保留给未来扩展）。"""
    return paragraph.text


def _all_instr_text(document: Any) -> list[str]:
    """收集文档里所有域代码（instrText），用于回读校验是否含 TOC 域。"""
    return [el.text or "" for el in document.element.body.iter(qn("w:instrText"))]


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class ThesisBuilder:
    """以学校样本 docx 为模板、按规范产出论文各章节的生成器。

    Parameters
    ----------
    template_path:
        学校格式样本 .docx 的路径（只读打开，不会被改动）。
    out_path:
        目标 .docx 路径；目录不存在会自动创建。
    page_number:
        是否给正文 section 自动加页脚页码域，默认 ``True``。

    Attributes
    ----------
    doc:
        内部的 ``python-docx`` ``Document`` 对象。
    out_path:
        输出路径（``pathlib.Path``）。
    last_report:
        最近一次 ``save()`` 的回读校验结果。
    """

    def __init__(self, template_path: str | Path, out_path: str | Path,
                 *, page_number: bool = True) -> None:
        tpl = Path(template_path)
        if not tpl.exists():
            raise FileNotFoundError(f"模板 docx 不存在：{tpl}")
        if tpl.suffix.lower() != ".docx":
            raise ValueError(f"模板必须是 .docx（不要用老式 .doc）：{tpl}")

        target = Path(out_path)
        if target.suffix.lower() != ".docx":
            raise ValueError(f"输出文件必须以 .docx 结尾：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)

        self.template_path: Path = tpl
        self.out_path: Path = target
        self.page_number: bool = page_number
        self.doc = Document(str(tpl))
        self.last_report: dict[str, Any] = {}

        self._headings: list[tuple[int, str]] = []       # [(级别, 文本)]
        self._chapter: str = "0"                          # 当前章号（表/图编号用）
        self._caption_seq: Counter[str] = Counter()       # 每章的图/表序号
        self._suppress_next_page_break: bool = False      # 分节后抑制多余的段前分页
        self._cover_done: bool = False
        self._after_table: bool = False                   # 上一块是表格 → 下一正文段留 6pt
        self._toc_done: bool = False
        self._toc_anchor: Paragraph | None = None        # 待补全静态目录的锚点段
        self._toc_pages: Mapping[str, int | str] | None = None

        self._clear_body()
        self._prepare_first_section()

    # -- 初始化 ----------------------------------------------------------

    def _clear_body(self) -> None:
        """清空 body 上所有正文块，但保留 styles / sectPr / numbering / 页眉页脚关系。"""
        body = self.doc.element.body
        sect_pr = body.find(qn("w:sectPr"))
        for child in list(body):
            if child is sect_pr:
                continue
            body.remove(child)
        self._prune_unused_rels()

    def _prune_unused_rels(self) -> None:
        """清掉已随正文一起删掉的图片/页眉页脚/超链接关系，防止成品体积被样本撑大。

        样本里含十几张截图与 4 组页眉页脚；只清 body 元素的话这些部件仍会被
        打进 zip（实测会让 3 页的演示文件变成 ~10MB）。这里按 body 里实际引用的
        ``r:id`` 反查，把无引用的关系移除。
        """
        from lxml import etree  # 局部导入，避免顶层多一个显式依赖

        body_xml = etree.tostring(self.doc.element.body, encoding="unicode")
        used = set(re.findall(r'r:(?:id|embed|link)="([^"]+)"', body_xml))
        rels = self.doc.part.rels
        drop_types = ("/image", "/header", "/footer", "/hyperlink", "/oleObject")
        for r_id in list(rels.keys()):
            rel = rels[r_id]
            if r_id in used:
                continue
            if any(rel.reltype.endswith(suffix) for suffix in drop_types):
                try:
                    del rels[r_id]
                except (KeyError, NotImplementedError):  # pragma: no cover
                    pass

    def _prepare_first_section(self) -> None:
        """封面 section：不显示页码；页边距等页面设置沿用样本 sectPr。"""
        section = self.doc.sections[0]
        section.different_first_page_header_footer = False
        section.footer.is_linked_to_previous = False
        for para in list(section.footer.paragraphs):
            for run in list(para.runs):
                run._r.getparent().remove(run._r)

    # -- 内部通用方法 ----------------------------------------------------

    def _add_para(self, text: str = "", style: str | None = None,
                  *, allow_empty: bool = False) -> Paragraph:
        """新建段落并套样式；若上一动作刚分节，则压掉样式自带的段前分页。

        Parameters
        ----------
        text:
            段落文本。``allow_empty=False`` 时空白文本会被当成空段落（仍创建）。
        style:
            样式名；``None`` 表示用模板默认（Normal）。
        """
        para = self.doc.add_paragraph()
        if style is not None:
            para.style = self._style(style)
        if self._suppress_next_page_break:
            para.paragraph_format.page_break_before = False
            self._suppress_next_page_break = False
        if text:
            para.add_run(text)
        return para

    def _style(self, name: str) -> Any:
        """按名字取样式，取不到时给出可读的报错（提示模板不对）。"""
        try:
            return self.doc.styles[name]
        except KeyError as exc:  # pragma: no cover - 只有模板不对才会走到
            raise KeyError(
                f"模板里没有段落样式「{name}」。请确认 template_path 指向学校格式样本 docx。"
            ) from exc

    @staticmethod
    def _require_text(value: Any, name: str) -> str:
        """校验必填文本参数：必须是 str 且非空。"""
        if not isinstance(value, str):
            raise TypeError(f"{name} 必须是 str，收到 {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"{name} 不能为空字符串")
        return value

    @staticmethod
    def _as_sequence(value: Any, name: str) -> list[str]:
        """把字符串/可迭代对象统一成字符串列表，并做类型校验。"""
        if isinstance(value, str):
            return [value]
        if not isinstance(value, Iterable):
            raise TypeError(f"{name} 必须是字符串或字符串序列")
        items = list(value)
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"{name} 的元素必须是 str，收到 {type(item).__name__}")
        return items

    def _split_paragraphs(self, text: str) -> list[str]:
        """按换行切段，丢掉空行（论文正文里空行没有意义）。"""
        return [seg.strip() for seg in re.split(r"\r?\n", text) if seg.strip()]

    # -- 封面 ------------------------------------------------------------

    def cover(self, title: str, student: str, advisor: str, student_id: str,
              ref_count: int | str, major: str, word_count: int | str,
              date: str | None = None) -> None:
        """生成封面：标题 → 题目表 → 信息表 → 日期，并把页面切到前置部分。

        版式对齐样本：标题「毕业论文」42pt 居中；论文题目两行 22pt 加粗；
        学生姓名/指导老师/学号/参考文献/专业/论文字数 14pt；日期 14pt 居中。
        任何值为 ``None`` 或空串时写入占位符 ``【待填】``。
        """
        def val(v: Any) -> str:
            if v is None:
                return PLACEHOLDER
            if isinstance(v, int):
                return str(v)
            s = str(v).strip()
            return s if s else PLACEHOLDER

        title_text = val(title)
        for field_name, field_value in (("student", student), ("advisor", advisor),
                                        ("student_id", student_id), ("major", major)):
            if field_value is None:
                sys.stderr.write(f"提示：cover 参数 {field_name} 为空，已填占位符。\n")

        # 1) 顶部空段（样本 para0，1.5 倍行距）
        top = self._add_para("")
        top.paragraph_format.line_spacing = 1.5
        top.paragraph_format.first_line_indent = Twips(0)

        # 2) 「毕业论文」42pt 居中
        para = self._add_para(style=None)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing = 1.5
        para.paragraph_format.first_line_indent = Twips(0)
        run = para.add_run("毕业论文")
        _set_run_font(run, size_pt=42.0, bold=False)

        # 3) 28pt 行高的空段（撑开标题与题目表的间距）
        gap = self._add_para("")
        gap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        gap.paragraph_format.line_spacing = 1.5
        gap.paragraph_format.first_line_indent = Twips(0)
        gap_run = gap.add_run("")
        _set_run_font(gap_run, size_pt=28.0)

        # 4) 题目表 2×2（无边框；标签右对齐、正文两端对齐）
        # 左列要放得下 22pt 的「论文题目：」（5 字≈3.9cm），右列留足题目折行空间
        title_tbl = self._cover_table(rows=2, cols=2, widths_twips=(2400, 6260))
        title_lines = self._split_title(title_text, 2)
        for row_idx, line in enumerate(title_lines):
            # 标签只在第一行写一次：两行都写「论文题目：」既重复又把题目挤成三行
            label = "论文题目：" if row_idx == 0 else ""
            self._fill_cover_cell(title_tbl.cell(row_idx, 0), label,
                                  size_pt=16.0, bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)
            self._fill_cover_cell(title_tbl.cell(row_idx, 1), line,
                                  size_pt=16.0, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

        # 5) 题目与信息表之间的间距（样本原有 4 空格下划线，实印像一条残线，改为纯空段）
        spacer = self._add_para(style=None)
        spacer.paragraph_format.line_spacing = 1.5
        spacer.paragraph_format.first_line_indent = Twips(0)
        spacer_run = spacer.add_run("")
        _set_run_font(spacer_run, size_pt=14.0)

        # 6) 信息表 3×4（标签用全角空格对齐，值居中）
        # 标签列须容纳「学生姓名：」「参考文献：」（14pt 5 字≈2.5cm）与「学    号：」
        info = self._cover_table(rows=3, cols=4,
                                 widths_twips=(1900, 2350, 1900, 2224))
        cells = [
            ("学生姓名：", val(student), "指导老师：", val(advisor)),
            ("学    号：", val(student_id), "参考文献：", val(ref_count)),
            ("专    业：", val(major), "论文字数：", val(word_count)),
        ]
        for row_idx, row_values in enumerate(cells):
            for col_idx, text in enumerate(row_values):
                is_label = col_idx % 2 == 0
                self._fill_cover_cell(
                    info.cell(row_idx, col_idx), text,
                    size_pt=14.0, bold=is_label,
                    align=WD_ALIGN_PARAGRAPH.RIGHT if is_label else WD_ALIGN_PARAGRAPH.CENTER,
                )

        # 7) 日期（14pt 居中）
        date_text = date if (isinstance(date, str) and date.strip()) else (
            f"{_date.today().year}年{_date.today().month}月{_date.today().day}日"
        )
        date_para = self._add_para(style=None)
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_para.paragraph_format.line_spacing = 1.5
        date_para.paragraph_format.first_line_indent = Twips(0)
        date_run = date_para.add_run(date_text)
        _set_run_font(date_run, size_pt=14.0)

        self._cover_done = True
        # 封面结束 → 新 section，前置部分（目录/摘要）用罗马页码
        self.section_break(page_number_format="upperRoman", restart=1)

    @staticmethod
    def _split_title(title: str, lines: int) -> list[str]:
        """把论文题目折成 ``lines`` 行：优先在标点/空格处断，否则按字数均分。"""
        clean = title.strip()
        if lines <= 1 or len(clean) <= 1:
            return [clean] + [""] * max(0, lines - 1)
        # 先按常见断点尝试
        for sep in ("：", ":", "，", ",", " "):
            if sep in clean:
                head, _, tail = clean.partition(sep)
                if head and tail:
                    return [head + sep, tail]
        # 再按结构助词断：避免把「政府采购合同」这类专名从中间劈开
        for sep in ("的", "与", "和", "及"):
            idx = clean.find(sep, max(6, len(clean) // 3))
            if 0 < idx < len(clean) - 4:
                cut = idx + len(sep)
                return [clean[:cut], clean[cut:]]
        # 按字符均分，尽量让第二行短一点
        cut = max(1, len(clean) * 2 // 3)
        return [clean[:cut], clean[cut:]]

    def _cover_table(self, rows: int, cols: int,
                     widths_twips: Sequence[int]) -> Table:
        """封面用的无边框固定列宽表格（列宽单位 twips，与样本 tblGrid 一致）。"""
        if rows <= 0 or cols <= 0:
            raise ValueError("cover 表格行列数必须为正整数")
        if len(widths_twips) != cols:
            raise ValueError(f"列宽数量 {len(widths_twips)} 与列数 {cols} 不匹配")

        table = self.doc.add_table(rows=rows, cols=cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        # 固定布局 + 总宽
        tbl_pr = table._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        _insert_ordered(tbl_pr, layout, ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
                                        "tblStyleRowBandSize", "tblStyleColBandSize",
                                        "tblW", "jc", "tblCellSpacing", "tblInd",
                                        "tblBorders", "shd", "tblLayout", "tblCellMar",
                                        "tblLook", "tblCaption", "tblDescription"))
        total = Twips(sum(widths_twips))
        table.columns  # 触发网格建立
        for idx, width in enumerate(widths_twips):
            table.columns[idx].width = Twips(width)
            for cell in table.columns[idx].cells:
                cell.width = Twips(width)
        tbl_w = tbl_pr.find(qn("w:tblW"))
        if tbl_w is None:
            tbl_w = OxmlElement("w:tblW")
            _insert_ordered(tbl_pr, tbl_w, ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
                                           "tblStyleRowBandSize", "tblStyleColBandSize",
                                           "tblW", "jc", "tblCellSpacing", "tblInd",
                                           "tblBorders", "shd", "tblLayout", "tblCellMar",
                                           "tblLook", "tblCaption", "tblDescription"))
        tbl_w.set(qn("w:type"), "dxa")
        tbl_w.set(qn("w:w"), str(int(total.twips) if hasattr(total, "twips") else sum(widths_twips)))
        self._set_table_borders(table, {"top": None, "left": None, "bottom": None,
                                        "right": None, "insideH": None, "insideV": None})
        # 左右内边距归零：Word 默认 0.19cm×2 会把「论文题目：」「参考文献：」挤成两行
        mar = OxmlElement("w:tblCellMar")
        for edge, size in (("top", 0), ("left", 0), ("bottom", 0), ("right", 0)):
            item = OxmlElement("w:" + edge)
            item.set(qn("w:w"), str(size))
            item.set(qn("w:type"), "dxa")
            mar.append(item)
        _insert_ordered(tbl_pr, mar, ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
                                      "tblStyleRowBandSize", "tblStyleColBandSize",
                                      "tblW", "jc", "tblCellSpacing", "tblInd",
                                      "tblBorders", "shd", "tblLayout", "tblCellMar",
                                      "tblLook", "tblCaption", "tblDescription"))
        return table

    @staticmethod
    def _fill_cover_cell(cell: Any, text: str, *, size_pt: float, bold: bool,
                         align: WD_ALIGN_PARAGRAPH) -> None:
        """往封面表格单元格里写一段文字（首行缩进清零、居中/右对齐）。"""
        para = cell.paragraphs[0]
        para.alignment = align
        # 关键：模板样式里带 firstLineChars（"2 字符"），Word 会按字号换算成 twips
        # （18pt → 723tw≈1.27cm），把「论文题目：」「学生姓名：」硬挤出单元格而折行。
        # 只设 firstLine=0 不生效——必须连 firstLineChars 一起清掉。
        pPr = para._p.get_or_add_pPr()
        old_ind = pPr.find(qn("w:ind"))
        if old_ind is not None:
            pPr.remove(old_ind)
        _set_pPr_child(para, "w:ind", firstLine="0", firstLineChars="0",
                       left="0", leftChars="0", right="0", rightChars="0")
        para.paragraph_format.line_spacing = 1.0
        run = para.add_run(text)
        _set_run_font(run, size_pt=size_pt, bold=bold)

    @staticmethod
    def _set_table_borders(table: Table,
                           spec: Mapping[str, tuple[str, int] | None]) -> None:
        """设置表格边框：值为 ``None`` 表示无边框，``('single', 12)`` 表示 1.5pt 单线。"""
        tbl_pr = table._tbl.tblPr
        old = tbl_pr.find(qn("w:tblBorders"))
        if old is not None:
            tbl_pr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            item = OxmlElement("w:" + edge)
            rule = spec.get(edge)
            if rule is None:
                item.set(qn("w:val"), "none")
                item.set(qn("w:sz"), "0")
            else:
                val, size = rule
                item.set(qn("w:val"), val)
                item.set(qn("w:sz"), str(size))
            item.set(qn("w:space"), "0")
            item.set(qn("w:color"), "auto")
            borders.append(item)
        _insert_ordered(tbl_pr, borders, ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
                                         "tblStyleRowBandSize", "tblStyleColBandSize",
                                         "tblW", "jc", "tblCellSpacing", "tblInd",
                                         "tblBorders", "shd", "tblLayout", "tblCellMar",
                                         "tblLook", "tblCaption", "tblDescription"))

    # -- 目录 ------------------------------------------------------------

    def toc_field(self, static_toc: bool = True,
                  pages: Mapping[str, int | str] | None = None,
                  heading: str = "目录") -> None:
        """插入 Word 自动目录域，并（可选）写一份静态目录作为兜底。

        Parameters
        ----------
        static_toc:
            ``True`` 时把已登记的标题写成 toc 1/2/3 样式的静态条目，
            并用右对齐点状制表位 + 页码占位；这些条目同时是 ``TOC`` 域的域结果。
        pages:
            标题文本 → 页码 的映射；缺省时按条目顺序生成占位页码 1,2,3…
        heading:
            目录标题文字，默认「目录」（套 ``目录标题格式``，段前分页）。

        Notes
        -----
        目录域代码为 ``TOC \\o "1-3" \\h \\z \\u``；在 Word 里 Ctrl+A → F9
        即可刷新成真实页码。
        """
        if not isinstance(static_toc, bool):
            raise TypeError("static_toc 必须是 bool")
        if pages is not None and not isinstance(pages, Mapping):
            raise TypeError("pages 必须是 标题→页码 的映射")

        title_para = self._add_para(style=STYLE_TOC_TITLE)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Twips(_SPACE_BEFORE_H1)
        title_para.paragraph_format.space_after = Twips(_SPACE_BEFORE_H1)
        title_para.add_run(self._require_text(heading, "heading"))

        instr = _TOC_INSTR
        entries: list[tuple[int, str]] = list(self._headings)
        self._toc_pages = pages

        if not static_toc or not entries:
            para = self.doc.add_paragraph()
            para.style = self._style(STYLE_TOC[0])
            para.paragraph_format.first_line_indent = Twips(0)
            _make_field(para, instr, cached=_TOC_PLACEHOLDER)
            if static_toc and not entries:
                # 目录在正文之前、此时还没有任何标题：先留锚点，等 save() 时补全
                self._toc_anchor = para
            self._toc_done = True
            return

        self._write_toc_entries(entries, pages, anchor=None)
        self._toc_done = True

    def _toc_paragraph(self, anchor: Paragraph | None, text: str, level: int,
                       page_holder: str, instr: str, *,
                       begin: bool, end: bool) -> Paragraph:
        """写一条目录条目（toc 1/2/3 + 右对齐点状制表位 + 11pt）。"""
        lvl = min(max(int(level), 1), 3)
        if anchor is None:
            para = self.doc.add_paragraph()
        else:
            para = anchor.insert_paragraph_before()
        para.style = self._style(STYLE_TOC[lvl - 1])
        fmt = para.paragraph_format
        fmt.first_line_indent = Twips(0)
        fmt.line_spacing = 1.5
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        fmt.left_indent = Twips(0 if lvl == 1 else (480 if lvl == 2 else 960))
        fmt.tab_stops.add_tab_stop(Twips(SAMPLE_WIDTH_TWIPS),
                                   WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if begin:
            _start_field(para, instr)
        run = para.add_run(text)
        _set_run_font(run, size_pt=11.0)
        tab_run = para.add_run()
        _set_run_font(tab_run, size_pt=11.0)
        tab_run.add_tab()
        page_run = para.add_run(page_holder)
        _set_run_font(page_run, size_pt=11.0)
        if end:
            _end_field(para)
        return para

    def _write_toc_entries(self, entries: Sequence[tuple[int, str]],
                           pages: Mapping[str, int | str] | None,
                           anchor: Paragraph | None = None) -> None:
        """按登记顺序把条目写成域结果（第一段 begin、最后一段 end）。"""
        for idx, (level, text) in enumerate(entries):
            page_holder = "1"
            if pages and text in pages:
                page_holder = str(pages[text])
            elif not pages:
                page_holder = str(idx + 1)
            self._toc_paragraph(anchor, text, level, page_holder, _TOC_INSTR,
                                begin=(idx == 0), end=(idx == len(entries) - 1))

    def _expand_static_toc(self) -> None:
        """把 toc_field() 留下的锚点段替换成按实际标题生成的静态目录（save() 调用）。"""
        anchor = self._toc_anchor
        entries: list[tuple[int, str]] = list(self._headings)
        if anchor is None or not entries:
            return
        self._write_toc_entries(entries, self._toc_pages, anchor=anchor)
        parent = anchor._p.getparent()
        if parent is not None:
            parent.remove(anchor._p)
        self._toc_anchor = None

    # -- 摘要 ------------------------------------------------------------

    def _abstract_heading(self, text: str, *, page_break: bool = False) -> Paragraph:
        """摘要 / Abstract 标题：样本实测是 Normal 段 + 黑体加粗 12pt 居中 + outlineLvl=0。

        （样本没有用「论文1级标题」，而是直接格式；这里按样本复刻，
        outlineLvl=0 保证它仍能进 TOC 的一级条目。）
        """
        para = self._add_para(style=None)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt = para.paragraph_format
        fmt.space_before = Twips(_SPACE_BEFORE_H1)
        fmt.space_after = Twips(_SPACE_BEFORE_H1)
        fmt.line_spacing = 1.0
        fmt.first_line_indent = Twips(0)
        fmt.keep_with_next = True
        if page_break:
            fmt.page_break_before = True
        _set_pPr_child(para, "w:outlineLvl", val="0")
        run = para.add_run(self._require_text(text, "摘要标题"))
        _set_run_font(run, ascii_font="黑体", eastasia_font="黑体",
                      size_pt=12.0, bold=True)
        self._headings.append((1, para.text))
        return para

    def _keyword_paragraph(self, label: str, label_sep: str, keywords: Sequence[str],
                           kw_sep: str, page_break: bool = False) -> None:
        """关键词段：标签黑体加粗 12pt，内容 10.5pt，首行不缩进（与样本一致）。

        ``label_sep`` 是标签后面的分隔（中文「：」、英文「: 」），
        ``kw_sep`` 是关键词之间的分隔（中文全角「；」、英文「; 」）。
        """
        para = self._add_para(style=None)
        fmt = para.paragraph_format
        fmt.first_line_indent = Twips(0)
        fmt.line_spacing = 1.5
        if page_break:
            fmt.page_break_before = True
        head = para.add_run(label)
        _set_run_font(head, ascii_font="黑体", eastasia_font="黑体", size_pt=12.0, bold=True)
        body_run = para.add_run(label_sep + kw_sep.join(keywords))
        _set_run_font(body_run, size_pt=10.5)

    def abstract_cn(self, text: str, keywords: Sequence[str]) -> None:
        """中文摘要：标题「摘要」+ 若干正文段 + 关键词。

        Parameters
        ----------
        text:
            摘要正文；含换行时按行拆成多段。
        keywords:
            关键词列表，用全角分号「；」连接。
        """
        body_text = self._require_text(text, "摘要正文")
        kws = self._as_sequence(keywords, "keywords")
        if not kws:
            raise ValueError("abstract_cn 至少需要一个关键词")
        self._abstract_heading("摘要")
        for seg in self._split_paragraphs(body_text):
            self.body(seg)
        self._keyword_paragraph("关键词", "：", kws, "；")

    def abstract_en(self, text: str, keywords: Sequence[str]) -> None:
        """英文摘要：标题「Abstract」（段前分页）+ 正文 + ``Keywords:``。"""
        body_text = self._require_text(text, "abstract 正文")
        kws = self._as_sequence(keywords, "keywords")
        if not kws:
            raise ValueError("abstract_en 至少需要一个关键词")
        self._abstract_heading("Abstract", page_break=True)
        for seg in self._split_paragraphs(body_text):
            self.body(seg)
        self._keyword_paragraph("Keywords", ": ", kws, "; ")

    # -- 标题 ------------------------------------------------------------

    def h1(self, text: str) -> Paragraph:
        """一级标题（章）：套 ``论文1级标题``，14pt 黑体加粗居中，段前分页。

        传完整字面文本，例如 ``h1('第1章  绪论')``、``h1('结论')``、
        ``h1('参考文献')``。会自动登记进目录，并更新当前章号（表/图编号用）。
        """
        title = self._require_text(text, "h1 文本")
        para = self._add_para(style=STYLE_H1)
        para.paragraph_format.space_before = Twips(_SPACE_BEFORE_H1)
        para.paragraph_format.keep_with_next = True
        _suppress_numbering(para)
        para.add_run(title)
        self._headings.append((1, title))
        match = re.search(r"第\s*(\d+)\s*章", title)
        if match:
            self._chapter = match.group(1)
            self._caption_seq.clear()
        return para

    def h2(self, text: str) -> Paragraph:
        """二级标题（节）：套 ``论文2级标题``，12pt 黑体加粗，段前 0.5 行。"""
        title = self._require_text(text, "h2 文本")
        para = self._add_para(style=STYLE_H2)
        para.paragraph_format.space_before = Twips(_SPACE_BEFORE_H23)
        para.paragraph_format.keep_with_next = True
        _suppress_numbering(para)
        para.add_run(title)
        self._headings.append((2, title))
        return para

    def h3(self, text: str) -> Paragraph:
        """三级标题（小节）：套 ``论文3级标题``，12pt 加粗，段前 0.5 行。"""
        title = self._require_text(text, "h3 文本")
        para = self._add_para(style=STYLE_H3)
        para.paragraph_format.space_before = Twips(_SPACE_BEFORE_H23)
        para.paragraph_format.keep_with_next = True
        _suppress_numbering(para)
        para.add_run(title)
        self._headings.append((3, title))
        return para

    # -- 正文与列表 ------------------------------------------------------

    def body(self, text: str) -> None:
        """正文段：套 ``论文正文``（宋体/Times New Roman 12pt、1.5 倍行距、两端对齐）。

        首行缩进 2 字符（twips 480，与样本正文段直接格式一致）。
        ``text`` 含换行时按行拆成多段。**不改动** ``[1]`` 之类的引用标记——
        参考文献角标该不该上标由人工决定，这里只当普通文本输出。
        """
        content = self._require_text(text, "body 文本")
        for seg in self._split_paragraphs(content):
            para = self._add_para(style=STYLE_BODY)
            para.paragraph_format.first_line_indent = Twips(_BODY_FIRST_LINE)
            if self._after_table:
                # 表格下边框与紧随的正文贴太近，视觉上是"糊在一起"
                para.paragraph_format.space_before = Pt(6)
                self._after_table = False
            para.add_run(seg)

    def bullets(self, items: Sequence[str], marker: str = "· ") -> None:
        """无序列表：正文样式 + 项目符号，悬挂缩进 2 字符。"""
        seq = self._as_sequence(items, "items")
        if not seq:
            raise ValueError("bullets 至少需要一项")
        for item in seq:
            para = self._add_para(style=STYLE_BODY)
            fmt = para.paragraph_format
            fmt.left_indent = Twips(_BODY_FIRST_LINE)
            fmt.first_line_indent = Twips(-_BODY_FIRST_LINE)
            para.add_run(marker + item)

    def numbered(self, items: Sequence[str]) -> None:
        """有序列表：中文括号编号「（1）（2）」，悬挂缩进 2 字符。"""
        seq = self._as_sequence(items, "items")
        if not seq:
            raise ValueError("numbered 至少需要一项")
        for idx, item in enumerate(seq, start=1):
            para = self._add_para(style=STYLE_BODY)
            fmt = para.paragraph_format
            fmt.left_indent = Twips(_BODY_FIRST_LINE)
            fmt.first_line_indent = Twips(-_BODY_FIRST_LINE)
            para.add_run(f"（{idx}）{item}")

    # -- 图表 ------------------------------------------------------------

    def table_caption(self, text: str) -> Paragraph:
        """表题（表上方，套 ``Caption``，居中，与表同页）。"""
        return self._caption("表", text, keep_next=True)

    def _caption(self, prefix: str, text: str, *, keep_next: bool,
                 number: int | None = None) -> Paragraph:
        """图/表题注通用实现：居中、黑体 10.5pt、1.5 倍行距、清掉样式自带首行缩进。

        ``number`` 由调用方指定时不再自增（``figure()`` 用它保证占位提示与图题同号）。
        """
        raw = self._require_text(text, "题注文本")
        if raw.startswith(prefix):
            caption_text = raw
        else:
            if number is None:
                self._caption_seq[prefix] += 1
                number = self._caption_seq[prefix]
            caption_text = f"{prefix}{self._chapter}-{number} {raw}"
        para = self._add_para(style=STYLE_CAPTION)
        fmt = para.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.first_line_indent = Twips(0)
        fmt.line_spacing = 1.5
        fmt.keep_with_next = keep_next
        # 图题贴在图下要留呼吸感；表题在表上方，段后留白由 keep_with_next 的表格自带
        if prefix == "图":
            fmt.space_before = Pt(6)
            fmt.space_after = Pt(10)
        else:
            fmt.space_before = Pt(6)
            fmt.space_after = Pt(4)
        para.add_run(caption_text)
        return para

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]],
              col_widths: Sequence[float] | None = None, *,
              caption: str | None = None, three_line: bool = False,
              font_pt: float = 10.5, header_bold: bool = False) -> Table:
        """插入表格：默认 ``Table Grid``，单元格 10.5pt 居中、首行缩进清零。

        Parameters
        ----------
        headers:
            表头文字序列。
        rows:
            数据行；每行元素会 ``str()``，长度不足自动补空串。
        col_widths:
            列宽（cm）。缺省时按可用宽度 14.64cm 等分。
        caption:
            表题文字；给了就先在表上方写一条 ``Caption`` 段（形如「表4-1 xxx」）。
        three_line:
            ``True`` 时改成样本实测的三线表（上下 1.5pt 粗线、表头下 0.75pt 细线、无竖线）。
        """
        head = [str(h) for h in self._as_sequence(headers, "headers")]
        if not head:
            raise ValueError("table 至少需要一列表头")
        row_list = [list(r) for r in rows]
        for idx, row in enumerate(row_list):
            if not isinstance(row, list):
                raise TypeError(f"rows[{idx}] 必须是序列")

        cols = len(head)
        if col_widths is None:
            widths = auto_column_widths(head, row_list, BODY_TABLE_CM, font_pt)
        else:
            if len(col_widths) != cols:
                raise ValueError(f"col_widths 数量 {len(col_widths)} 与列数 {cols} 不匹配")
            widths = [float(w) for w in col_widths]
            if any(w <= 0 for w in widths):
                raise ValueError("col_widths 必须都是正数（单位 cm）")
            # 总宽超过版心时按比例收窄，否则表格会伸出右侧页边距
            total = sum(widths)
            if total > BODY_TABLE_CM + 1e-6:
                widths = [w * BODY_TABLE_CM / total for w in widths]

        if caption is not None:
            self.table_caption(caption)

        table = self.doc.add_table(rows=1 + len(row_list), cols=cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        if three_line:
            self._set_table_borders(table, {
                "top": ("single", 12), "bottom": ("single", 12),
                "left": None, "right": None, "insideH": None, "insideV": None,
            })
        else:
            table.style = self._style("Table Grid")
            table.autofit = False

        for idx, width_cm in enumerate(widths):
            for cell in table.columns[idx].cells:
                cell.width = Cm(width_cm)

        # 逐列判定对齐方式：长文本列左对齐（居中会让折行文字读起来很碎），
        # 短标签与数字仍居中
        col_align: list[int] = []
        for col_idx in range(cols):
            cells = [head[col_idx]] + [str(r[col_idx]) if col_idx < len(r) else ""
                                       for r in row_list]
            widest = max(_display_width(c) for c in cells)
            col_align.append(WD_ALIGN_PARAGRAPH.LEFT if widest > 12
                             else WD_ALIGN_PARAGRAPH.CENTER)

        for col_idx, text in enumerate(head):
            self._format_cell(table.cell(0, col_idx), text, font_pt, header_bold,
                              align=WD_ALIGN_PARAGRAPH.CENTER)
        for row_idx, row in enumerate(row_list, start=1):
            padded = [str(v) for v in row] + [""] * (cols - len(row))
            for col_idx in range(cols):
                self._format_cell(table.cell(row_idx, col_idx), padded[col_idx],
                                  font_pt, False, align=col_align[col_idx])
        if three_line:
            # 表头行下边线（三线表的中间那条细线）
            for cell in table.rows[0].cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "6")
                bottom.set(qn("w:space"), "0")
                bottom.set(qn("w:color"), "auto")
                borders.append(bottom)
                _insert_ordered(tc_pr, borders, ("cnfStyle", "tcW", "gridSpan", "hMerge",
                                                 "vMerge", "tcBorders", "shd", "noWrap",
                                                 "tcMar", "textDirection", "tcFitText",
                                                 "vAlign", "hideMark"))
        # 逐行加固：跨页时表头重复、单行不被页边界劈开、单元格垂直居中
        self._harden_table_rows(table, repeat_header=three_line)
        self._after_table = True
        return table

    @staticmethod
    def _harden_table_rows(table: Table, *, repeat_header: bool) -> None:
        """表头跨页重复 + 行内不断页 + 单元格垂直居中（三线表长表的必备项）。"""
        for row_idx, row in enumerate(table.rows):
            tr_pr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            _insert_ordered(tr_pr, cant, ("cnfStyle", "divId", "gridBefore", "gridAfter",
                                          "wBefore", "wAfter", "cantSplit", "trHeight",
                                          "tblHeader", "tblCellSpacing", "jc", "hidden"))
            if repeat_header and row_idx == 0:
                header = OxmlElement("w:tblHeader")
                _insert_ordered(tr_pr, header, ("cnfStyle", "divId", "gridBefore", "gridAfter",
                                                "wBefore", "wAfter", "cantSplit", "trHeight",
                                                "tblHeader", "tblCellSpacing", "jc", "hidden"))
            for cell in row.cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                valign = OxmlElement("w:vAlign")
                valign.set(qn("w:val"), "center")
                _insert_ordered(tc_pr, valign, ("cnfStyle", "tcW", "gridSpan", "hMerge",
                                                "vMerge", "tcBorders", "shd", "noWrap",
                                                "tcMar", "textDirection", "tcFitText",
                                                "vAlign", "hideMark"))

    @staticmethod
    def _format_cell(cell: Any, text: str, font_pt: float, bold: bool,
                     align: int = WD_ALIGN_PARAGRAPH.CENTER) -> None:
        """单元格统一排版：默认居中、首行缩进 0、单倍行距（长文本列可传 LEFT）。"""
        para = cell.paragraphs[0]
        para.alignment = align
        fmt = para.paragraph_format
        fmt.first_line_indent = Twips(0)
        fmt.line_spacing = 1.0
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        run = para.add_run(text)
        _set_run_font(run, ascii_font="Times New Roman", eastasia_font="宋体",
                      size_pt=font_pt, bold=bold)

    def figure_caption(self, text: str) -> Paragraph:
        """图题（图下方，套 ``Caption``，居中）。"""
        return self._caption("图", text, keep_next=False)

    def figure(self, path: str | Path | None, width_cm: float = 12.0,
               *, caption: str | None = None, placeholder_text: str | None = None) -> None:
        """插入图片并居中；文件缺失时插入灰色占位段「【此处插入图 x-x】」。

        Parameters
        ----------
        path:
            图片路径；``None`` 或文件不存在时走占位流程（方便先排版后补图）。
        width_cm:
            图片宽度（cm），高度按原比例。
        caption:
            图题文字；给了就在图**下方**写一条 ``Caption`` 段。
        """
        if width_cm <= 0:
            raise ValueError("width_cm 必须为正数")
        # 超过版心的图会被 Word 折行或顶出版心，统一收到可用宽度
        width_cm = min(float(width_cm), BODY_TABLE_CM)

        # 图号只在这里算一次：图片 / 占位提示 / 图题三处共用，正文引用才不会错位
        num = self._caption_seq["图"] + 1

        image_path = Path(path) if path else None
        inserted = False
        if image_path is not None and image_path.exists():
            para = self._add_para(style=None)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt = para.paragraph_format
            fmt.first_line_indent = Twips(0)
            fmt.keep_with_next = caption is not None
            para.add_run().add_picture(str(image_path), width=Cm(width_cm))
            inserted = True

        if not inserted:
            label = placeholder_text
            if label is None:
                label = f"【此处插入图 {self._chapter}-{num}】"
            para = self._add_para(style=None)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt = para.paragraph_format
            fmt.first_line_indent = Twips(0)
            fmt.line_spacing = 1.5
            fmt.keep_with_next = caption is not None
            run = para.add_run(label)
            _set_run_font(run, ascii_font="宋体", eastasia_font="宋体",
                          size_pt=10.5, color="808080")
        if caption is not None:
            self._caption("图", caption, keep_next=False, number=num)
        if not inserted or caption is not None:
            self._caption_seq["图"] = num

    # -- 代码 / 公式 -----------------------------------------------------

    def code_block(self, text: str, lang: str = "python") -> None:
        """代码块：等宽字体 9pt、单倍行距、浅灰底纹、无首行缩进。

        Parameters
        ----------
        text:
            代码原文；按行用软换行（``w:br``）连成一段，保证底纹是连续的一整块。
        lang:
            语言标记，仅写进注释首行便于人工核对（不参与排版）。
        """
        content = self._require_text(text, "code_block 文本")
        para = self._add_para(style=STYLE_BODY)
        fmt = para.paragraph_format
        fmt.first_line_indent = Twips(0)
        fmt.left_indent = Cm(0.5)
        fmt.right_indent = Cm(0.2)
        fmt.line_spacing = 1.0
        fmt.space_before = Pt(4)
        fmt.space_after = Pt(4)
        fmt.keep_together = True
        _paragraph_shading(para, "F5F5F5")

        lines = content.splitlines() or [""]
        for idx, line in enumerate(lines):
            if idx:
                para.add_run().add_break(WD_BREAK.LINE)
            run = para.add_run(line if line else " ")
            _set_run_font(run, ascii_font="Consolas", eastasia_font="宋体",
                          size_pt=9.0)
        _ = lang  # 语言标记暂不参与排版，保留参数以便后续高亮

    def equation_like(self, text: str) -> None:
        """公式行占位：居中、1.5 倍行距、无首行缩进（真实公式请人工用公式编辑器替换）。"""
        content = self._require_text(text, "equation_like 文本")
        para = self._add_para(style=STYLE_BODY)
        fmt = para.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.first_line_indent = Twips(0)
        fmt.line_spacing = 1.5
        fmt.space_before = Pt(6)
        fmt.space_after = Pt(6)
        run = para.add_run(content)
        _set_run_font(run, ascii_font="Times New Roman", eastasia_font="宋体", size_pt=12.0)

    # -- 参考文献 / 致谢 / 附录 ------------------------------------------

    def refs(self, entries: Sequence[str]) -> None:
        """参考文献：顺序编码制 ``[1] ...``，悬挂缩进（回行对齐到 2 字符）。

        Parameters
        ----------
        entries:
            GB/T 7714 格式的文献条目（**不含**编号，编号由本方法生成）。
        """
        seq = self._as_sequence(entries, "entries")
        if not seq:
            raise ValueError("refs 至少需要一条文献")
        self.h1("参考文献")
        for idx, entry in enumerate(seq, start=1):
            para = self._add_para(style=STYLE_BODY)
            fmt = para.paragraph_format
            fmt.left_indent = Twips(_BODY_FIRST_LINE)
            fmt.first_line_indent = Twips(-_BODY_FIRST_LINE)
            para.add_run(f"[{idx}] {entry}")

    def ack(self, text: str) -> None:
        """致谢：``论文1级标题`` 的「致谢」+ 正文段落。"""
        content = self._require_text(text, "ack 文本")
        self.h1("致谢")
        for seg in self._split_paragraphs(content):
            self.body(seg)

    def appendix(self, title: str, text: str) -> None:
        """附录：标题套 ``论文1级标题``（默认「附录A xxx」），正文照排。"""
        heading = self._require_text(title, "appendix 标题")
        content = self._require_text(text, "appendix 正文")
        self.h1(heading)
        for seg in self._split_paragraphs(content):
            self.body(seg)

    # -- 分页 / 分节 -----------------------------------------------------

    def page_break(self) -> None:
        """插入一个分页符段（用于封面与目录之间等需要硬分页的位置）。"""
        para = self._add_para(style=None)
        para.paragraph_format.first_line_indent = Twips(0)
        para.add_run().add_break(WD_BREAK.PAGE)

    def section_break(self, start_type: Any = WD_SECTION_START.NEW_PAGE,
                      *, page_number_format: str | None = None,
                      restart: int = 1) -> Any:
        """插入分节符并（可选）设定该节的页码格式，返回新的 ``Section``。

        Parameters
        ----------
        start_type:
            分节方式，默认下一页（``WD_SECTION_START.NEW_PAGE``）。
        page_number_format:
            页码格式：``'decimal'`` / ``'upperRoman'`` / ``'lowerRoman'`` 等；
            ``None`` 表示沿用上一节。
        restart:
            页码起始值，默认 1（前置部分罗马页码 I 开始，正文阿拉伯页码 1 开始）。
        """
        section = self.doc.add_section(start_type)
        # 新节继承上一节的页边距/纸张；分节后紧跟的段不需要再加段前分页
        self._suppress_next_page_break = True

        if page_number_format is not None:
            sect_pr = section._sectPr
            pg_num = sect_pr.find(qn("w:pgNumType"))
            if pg_num is None:
                pg_num = OxmlElement("w:pgNumType")
                sect_pr.append(pg_num)
            pg_num.set(qn("w:fmt"), page_number_format)
            pg_num.set(qn("w:start"), str(restart))

        if self.page_number:
            footer = section.footer
            footer.is_linked_to_previous = False
            base = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
            for run in list(base.runs):
                run._r.getparent().remove(run._r)
            base.alignment = WD_ALIGN_PARAGRAPH.CENTER
            base.paragraph_format.first_line_indent = Twips(0)
            _make_field(base, " PAGE  \\* MERGEFORMAT ", cached="1")
            for run in base.runs:
                _set_run_font(run, ascii_font="Times New Roman",
                              eastasia_font="宋体", size_pt=10.5)
        return section

    # -- 保存与回读校验 --------------------------------------------------

    def save(self) -> dict[str, Any]:
        """保存并**回读校验**：重新用 ``Document(out_path)`` 打开，统计格式落点。

        Returns
        -------
        dict
            含 ``file`` / ``size_bytes`` / ``paragraph_count`` / ``table_count`` /
            ``style_usage`` / ``has_toc_field`` / ``toc_instr`` / ``page_setup`` 等。
        """
        self._expand_static_toc()          # 目录若先于正文写入，这里补全静态条目
        self.doc.save(str(self.out_path))
        if self._toc_anchor is not None:
            # 兜底：没有任何标题时锚点仍在，删掉它而不是把占位段落留在成品里
            parent = self._toc_anchor._p.getparent()
            if parent is not None:
                parent.remove(self._toc_anchor._p)
            self._toc_anchor = None
            self.doc.save(str(self.out_path))

        reopened = Document(str(self.out_path))
        style_usage = Counter(p.style.name if p.style is not None else "<无>" for p in reopened.paragraphs)
        instr_list = _all_instr_text(reopened)
        toc_instr = [i for i in instr_list if "TOC" in i]
        section = reopened.sections[0]
        page_setup = {
            "page_width_cm": round(section.page_width.cm, 3),
            "page_height_cm": round(section.page_height.cm, 3),
            "margin_top_cm": round(section.top_margin.cm, 3),
            "margin_bottom_cm": round(section.bottom_margin.cm, 3),
            "margin_left_cm": round(section.left_margin.cm, 3),
            "margin_right_cm": round(section.right_margin.cm, 3),
            "header_distance_cm": round(section.header_distance.cm, 3),
            "footer_distance_cm": round(section.footer_distance.cm, 3),
            "sections": len(reopened.sections),
        }
        report = {
            "file": str(self.out_path),
            "size_bytes": self.out_path.stat().st_size,
            "size_kb": round(self.out_path.stat().st_size / 1024, 1),
            "paragraph_count": len(reopened.paragraphs),
            "table_count": len(reopened.tables),
            "style_usage": dict(style_usage.most_common()),
            "has_toc_field": bool(toc_instr),
            "toc_instr": toc_instr,
            "page_setup": page_setup,
            "headings": list(self._headings),
        }
        self.last_report = report
        return report


# ---------------------------------------------------------------------------
# 演示论文
# ---------------------------------------------------------------------------

def build_demo(out_path: str | Path, template_path: str | Path | None = None) -> dict[str, Any]:
    """生成一份 3~5 页的演示论文，用于验证样式是否落对。

    内容：封面 + 目录（域 + 静态兜底）+ 中英文摘要 + 第 1 章两节
    + 一个表 + 一个图占位 + 代码块 + 公式占位 + 三条参考文献 + 致谢 + 附录。

    Returns
    -------
    dict
        ``ThesisBuilder.save()`` 的回读校验报告。
    """
    tpl = Path(template_path) if template_path is not None else DEFAULT_TEMPLATE
    builder = ThesisBuilder(tpl, out_path)

    builder.cover(
        title="基于Python的毕业设计论文格式自动化生成工具设计与实现",
        student="张三",
        advisor="李四 副教授",
        student_id="2021001",
        ref_count=28,
        major="计算机科学与技术",
        word_count=21500,
        date="2025年5月18日",
    )

    # 前置部分（目录 / 摘要）：罗马页码
    builder.toc_field()

    builder.abstract_cn(
        "本文面向本科毕业设计论文的排版痛点，设计并实现了一个以学校格式样本为模板的"
        "文档生成工具。工具直接复用样本 docx 中的样式表与页面设置，通过 python-docx "
        "在模板骨架上填充各章节内容，从而在不引入任何第三方排版软件的前提下，"
        "产出格式完全合规的论文文件。测试表明，生成结果的标题层级、行距段距、"
        "图表题注与参考文献格式均与样本一致。\n"
        "本文的主要工作包括：抽取样本样式规范并量化为可复用的样式清单；"
        "实现封面、摘要、正文、图表、参考文献等十余种内容块的生成接口；"
        "并在保存后自动回读校验样式落点。",
        ["毕业设计", "论文格式", "python-docx", "自动化生成"],
    )

    builder.abstract_en(
        "This paper presents a document generator for undergraduate graduation theses. "
        "The tool takes the school-provided sample .docx as a template, reuses its style "
        "definitions and page setup directly, and fills in chapters via python-docx.",
        ["Graduation Thesis", "Document Format", "python-docx", "Automation"],
    )

    # 正文：阿拉伯页码从 1 开始
    builder.section_break(page_number_format="decimal", restart=1)

    builder.h1("第1章  绪论")
    builder.h2("1.1  研究背景及意义")
    builder.body(
        "毕业设计论文的格式规范由学院统一规定，涉及页面设置、标题层级、正文行距、"
        "图表题注与参考文献著录等数十项细节。学生在实际排版中往往反复调整，"
        "仍难以完全对齐模板，造成大量重复劳动。"
    )
    builder.body(
        "将格式规范沉淀为代码，可以把“人工对格式”转化为“程序守格式”，"
        "使内容写作与版面合规解耦，显著降低返工成本。"
    )
    builder.bullets([
        "格式一致性：所有段落由样式驱动，不依赖手工微调；",
        "可复用性：样式清单与生成器分离，便于其他院校替换模板；",
        "可校验性：生成后自动回读，报告样式使用统计。",
    ])

    builder.h2("1.2  国内外研究现状")
    builder.h3("1.2.1  国内研究现状")
    builder.body(
        "国内高校多采用 Word 模板加人工排版的方式，少数单位提供了基于 VBA 或"
        "LaTeX 的辅助工具，但通用性与可维护性有限。"
    )
    builder.numbered([
        "模板法：下发 .dotx 模板，靠人工套用样式；",
        "脚本法：用 VBA 宏批量调整段落格式；",
        "排版系统法：改用 LaTeX 模板，但与学院提交要求存在格式差异。",
    ])
    builder.h3("1.2.2  国外研究现状")
    builder.body(
        "国外高校普遍采用 LaTeX 或机构模板，配合版本管理工具维护格式规范，"
        "在工程化程度上相对成熟。"
    )

    builder.table_caption("样式继承与直接格式的关系")
    builder.table(
        headers=["格式项", "来源", "是否可被段落覆盖"],
        rows=[
            ["字体字号", "样式 rPr", "是（run 级）"],
            ["行距段距", "样式 pPr/spacing", "是（段落级）"],
            ["页面设置", "sectPr", "否（按节生效）"],
        ],
        caption=None,
    )

    builder.body("系统总体结构如图1-1所示。")
    builder.figure(None, width_cm=12.0, caption="系统总体结构图")

    builder.body("关键代码如下：")
    builder.code_block(
        "from docx import Document\n"
        "doc = Document(template_path)\n"
        "body = doc.element.body\n"
        "for child in list(body):\n"
        "    if child.tag.endswith('}sectPr'):\n"
        "        continue\n"
        "    body.remove(child)",
        lang="python",
    )

    builder.body("样式继承的量化关系可表示为：")
    builder.equation_like("S_effective = S_style ∪ S_direct      (1-1)")

    builder.refs([
        "彭雅玲. 基于Spring Boot的水产品质量安全追溯系统的设计与开发[D]. 南通: 南通大学, 2025.",
        "GB/T 7714—2015 信息与文献 参考文献著录规则[S]. 北京: 中国标准出版社, 2015.",
        "Python Software Foundation. python-docx 1.2.0 documentation[EB/OL]. "
        "(2024-06-01)[2025-05-01]. https://python-docx.readthedocs.io/.",
    ])

    builder.ack("感谢指导老师在选题、设计与写作过程中给予的悉心指导，感谢同学在测试阶段的帮助。")
    builder.appendix("附录A  样式清单节选", "STYLE_H1 = 论文1级标题；STYLE_BODY = 论文正文。")

    return builder.save()


if __name__ == "__main__":  # pragma: no cover
    target = Path(__file__).resolve().parent.parent / "docs" / "thesis_demo.docx"
    info = build_demo(target)
    print("paragraphs:", info["paragraph_count"], "tables:", info["table_count"],
          "toc:", info["has_toc_field"], "size:", info["size_kb"], "KB")
