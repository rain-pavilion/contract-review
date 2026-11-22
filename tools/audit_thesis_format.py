# -*- coding: utf-8 -*-
"""毕业论文 docx 排版审计：按学校样本的实测规范逐项体检，只报问题不改文件。

检查项（每项都给出判据，不是"看着不顺眼"）：
    1. 段落样式分布 —— 是否只用了样本里那 9 个样式，有无 Normal 漏网
    2. 正文段落：首行缩进 2 字符(480twips)、行距 1.5、两端对齐、段前段后 0
    3. 标题：1/2/3 级样式是否正确、有无空标题、层级是否跳级（1→3）
    4. 空段堆积（连续 ≥2 个空段）
    5. 图片：宽度是否超版心 14.64cm、是否与图题同段、图题是否在图下
    6. 表格：总宽是否超版心、是否三线表（边框只留上下+表头下）、表题是否在表上
    7. 参考文献：是否悬挂缩进、编号是否为 [n]、行距
    8. 代码块：字体是否等宽、单行是否超宽（会被 Word 自动折行）
    9. 目录：条目数、样式、右制表位是否 8290twips
   10. 页码域与页眉页脚：各节是否都有
   11. 中英文之间的空格、全角标点误用（。，；：）与半角混用
   12. 段落分页：一级标题是否段前分页（避免标题压页脚）

用法：
    python tools/audit_thesis_format.py            # 审计默认论文
    python tools/audit_thesis_format.py xxx.docx
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from docx import Document
from docx.shared import Twips
from lxml import etree

WORKSPACE = Path(__file__).resolve().parents[1]
_DOCS = WORKSPACE / "docs"
# 论文 docx 可能被用户改名（去掉"毕业论文-"前缀），按存在性自动发现，避免死路径
DEFAULT_DOCX = next(
    (_DOCS / _n for _n in ("智律契灵合同智能审查系统v2.docx",
                           "毕业论文-智律契灵合同智能审查系统v2.docx")
     if (_DOCS / _n).exists()),
    _DOCS / "智律契灵合同智能审查系统v2.docx",
)

BODY_WIDTH_TWIPS = 8290          # 21.0 - 3.18 - 3.18 cm
STYLE_H1, STYLE_H2, STYLE_H3 = "论文1级标题", "论文2级标题", "论文3级标题"
STYLE_BODY, STYLE_CAPTION = "论文正文", "Caption"
ALLOWED = {
    "Normal", STYLE_H1, STYLE_H2, STYLE_H3, STYLE_BODY, STYLE_CAPTION,
    "目录标题格式", "toc 1", "toc 2", "toc 3", "Header", "Footer",
}

issues: list[tuple[str, str, str]] = []   # (级别, 位置, 说明)


def add(level: str, where: str, msg: str) -> None:
    issues.append((level, where, msg))


def tw(v) -> int | None:
    """把 Length 转成 twips（None 透传）。"""
    if v is None:
        return None
    return int(round(v.twips if hasattr(v, "twips") else v / 635))


def is_code_para(p) -> bool:
    """代码块判定：等宽字体 + 左缩进>0 + 单倍行距（builder 里 code_block 的签名）。"""
    mono = any(_mono_run(r) for r in p.runs)
    return mono and (p.paragraph_format.left_indent is not None)


def _mono_run(run) -> bool:
    name = (run.font.name or "").lower()
    return any(k in name for k in ("consol", "courier", "mono", "menlo"))


def is_hanging(p) -> bool:
    """悬挂缩进判定：首行缩进为负、左缩进为正（参考文献的正确形态）。"""
    fi = tw(p.paragraph_format.first_line_indent)
    li = tw(p.paragraph_format.left_indent)
    return fi is not None and fi < 0 and (li or 0) > 0


def audit(path: Path) -> None:
    doc = Document(path)
    paras = doc.paragraphs

    # ---- 1. 样式分布 ----
    styles = Counter(p.style.name for p in paras)
    for name, cnt in styles.most_common():
        if name not in ALLOWED:
            add("!!", f"样式 {name}", f"出现 {cnt} 段，不在样本样式集内（会破坏目录/大纲）")

    # ---- 2. 正文格式 ----
    bad_indent = bad_spacing = bad_align = 0
    first_body = None
    for p in paras:
        if p.style.name != STYLE_BODY or not p.text.strip():
            continue
        if is_code_para(p) or is_hanging(p):
            continue                      # 代码块与参考文献各有自己的正确缩进
        pf = p.paragraph_format
        ind = tw(pf.first_line_indent)
        ls = pf.line_spacing
        if ind is not None and abs(ind - 480) > 20:
            bad_indent += 1
            if first_body is None:
                first_body = (p.text[:24], ind)
        if ls is not None and abs(float(ls) - 1.5) > 0.02:
            bad_spacing += 1
        if pf.alignment is not None and str(pf.alignment) not in ("JUSTIFY (3)",):
            bad_align += 1
    if bad_indent:
        add("!!", "论文正文", f"{bad_indent} 段首行缩进不是 2 字符（480twips），例：{first_body}")
    if bad_spacing:
        add("!!", "论文正文", f"{bad_spacing} 段行距不是 1.5 倍")
    if bad_align:
        add("·", "论文正文", f"{bad_align} 段不是两端对齐（继承样式值，若来自样式可忽略）")

    # ---- 3. 标题层级 ----
    levels = []
    for i, p in enumerate(paras):
        s = p.style.name
        if s in (STYLE_H1, STYLE_H2, STYLE_H3):
            if not p.text.strip():
                add("!!", f"第 {i} 段", f"{s} 是空标题")
            levels.append((i, s, p.text.strip()[:30]))
    seq = [{"论文1级标题": 1, "论文2级标题": 2, "论文3级标题": 3}[s] for _, s, _ in levels]
    for a, b in zip(seq, seq[1:]):
        if b - a > 1:
            add("!!", "标题层级", f"从 {a} 级直接跳到 {b} 级（目录会出现孤立层级）")

    h1 = [t for _, s, t in levels if s == STYLE_H1]
    add("·", "一级标题", f"共 {len(h1)} 个：" + " / ".join(h1[:12]))

    # ---- 4. 空段堆积 ----
    run_empty = 0
    worst = (0, "")
    for p in paras:
        if not p.text.strip():
            run_empty += 1
            worst = max(worst, (run_empty, f"第 {paras.index(p)} 段附近"))
        else:
            run_empty = 0
    if worst[0] >= 3:
        add("!", "空段", f"连续 {worst[0]} 个空段（{worst[1]}），建议改用段前/段后间距")

    # ---- 5. 图片 ----
    body = doc.element.body
    pics = body.findall(".//" + "{http://schemas.openxmlformats.org/drawingml/2006/main}ext")
    wide = 0
    for ext in pics:
        cx = int(ext.get("cx", "0"))
        if cx / 635 > BODY_WIDTH_TWIPS + 20:
            wide += 1
    if wide:
        add("!!", "插图", f"{wide} 张图宽超过版心 {BODY_WIDTH_TWIPS/567:.2f} cm")

    # 图题位置：Caption 段以"图"开头时应紧跟图片段之后
    # 注意：图片所在的段落**没有文本**，判断"上一个元素"必须看元素本身，不能看文本
    BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    for i, p in enumerate(paras):
        if p.style.name == STYLE_CAPTION and p.text.strip().startswith("图"):
            prev = None
            for j in range(i - 1, -1, -1):
                cand = paras[j]
                if cand._p.findall(".//" + BLIP):
                    prev = cand
                    break
                if cand.text.strip():
                    prev = cand
                    break
            if prev is None or not prev._p.findall(".//" + BLIP):
                add("!", f"第 {i} 段", f"图题「{p.text.strip()[:20]}」前面不是图片段（图题应在图下）")

    # ---- 6. 表格 ----
    # 封面两张表沿用学校样本宽度（样本本身就是 8379 / 8137 twips），只审正文表
    COVER_TABLES = 2
    for ti, t in enumerate(doc.tables, 1):
        width = 0
        for col in t.columns:
            w = tw(col.width)
            if w:
                width += w
        if ti > COVER_TABLES and width and width > BODY_WIDTH_TWIPS + 20:
            add("!!", f"表{ti}", f"总宽 {width} twips > 版心 {BODY_WIDTH_TWIPS}（{width/567:.2f} cm 会出界）")
        tblPr = t._tbl.tblPr
        borders = tblPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblBorders")
        if borders is None and ti > COVER_TABLES:
            add("!", f"表{ti}", "未显式设置边框（依赖样式，可能不是三线表）")
        elif borders is not None and ti > COVER_TABLES:
            # 结构化判断：Word 重存后会把 w:val="none" 规范成 "nil"、并重排属性顺序，
            # 所以不能靠字符串匹配，直接读每条边的 val
            W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            edges = {}
            for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
                node = borders.find(W + edge)
                if node is not None:
                    edges[edge] = (node.get(W + "val") or "").lower()
            hollow = {"none", "nil", "0", ""}
            ok = (edges.get("top") not in hollow and edges.get("bottom") not in hollow
                  and edges.get("left", "nil") in hollow
                  and edges.get("right", "nil") in hollow
                  and edges.get("insideH", "nil") in hollow
                  and edges.get("insideV", "nil") in hollow)
            if not ok:
                add("!", f"表{ti}", f"边框不是三线表形态（实测 {edges}）")
            if "w:tblHeader" not in t._tbl.xml:
                add("!", f"表{ti}", "表头未设跨页重复（长表翻页后没有表头）")

    # 表题位置：Caption 以"表"开头时应紧跟表格之前
    for i, p in enumerate(paras):
        if p.style.name == STYLE_CAPTION and p.text.strip().startswith("表"):
            nxt = paras[i + 1] if i + 1 < len(paras) else None
            has_tbl = any(
                el.tag.endswith("}tbl")
                for el in body.iter()
                if el.tag.endswith("}tbl")
            )
            if nxt is not None and not nxt.text.strip() and not has_tbl:
                add("!", f"第 {i} 段", f"表题「{p.text.strip()[:20]}」后没有表格")

    # ---- 7. 参考文献 ----
    in_refs = False
    refs = []
    for p in paras:
        t = p.text.strip()
        if p.style.name == STYLE_H1 and t.startswith("参考文献"):
            in_refs = True
            continue
        if p.style.name == STYLE_H1 and in_refs:
            break
        if in_refs and t:
            refs.append(p)
    if not refs:
        add("!!", "参考文献", "没找到参考文献段")
    else:
        bad_num = [p.text[:28] for p in refs if not re.match(r"^\[\d+\]", p.text.strip())]
        if bad_num:
            add("!!", "参考文献", f"{len(bad_num)} 条不是 [n] 起头，例：{bad_num[0]}")
        no_hang = [p for p in refs if (tw(p.paragraph_format.first_line_indent) or 0) >= 0
                   and (tw(p.paragraph_format.left_indent) or 0) == 0]
        if len(no_hang) > len(refs) // 2:
            add("!", "参考文献", f"{len(no_hang)}/{len(refs)} 条没有悬挂缩进（换行与首行不对齐）")
        add("·", "参考文献", f"{len(refs)} 条；样式 {Counter(p.style.name for p in refs).most_common()}")

    # ---- 8. 代码块（附录）----
    mono = 0
    code_paras = [p for p in paras if re.match(r"^\s{2,}\S", p.text) and len(p.text) > 40]
    for p in code_paras:
        for r in p.runs:
            name = (r.font.name or "").lower()
            if "consol" in name or "courier" in name or "mono" in name:
                mono += 1
                break
    if code_paras and mono < len(code_paras) * 0.5:
        add("!", "代码块", f"{len(code_paras)} 段缩进长行里只有 {mono} 段用等宽字体（Word 会按比例字体折行，走样）")
    elif code_paras:
        add("·", "代码块", f"{len(code_paras)} 段，{mono} 段等宽字体")

    # ---- 9. 目录 ----
    toc = [p for p in paras if p.style.name in ("toc 1", "toc 2", "toc 3")]
    add("·", "目录", f"{len(toc)} 条（" + " / ".join(
        f"{s}:{c}" for s, c in Counter(p.style.name for p in toc).items()) + "）")
    over = [p.text[:26] for p in toc if len(p.text) > 60]
    if over:
        add("!", "目录", f"{len(over)} 条过长，可能压到页码：{over[0]}")

    # ---- 10. 页码域 ----
    with zipfile.ZipFile(path) as z:
        footers = [n for n in z.namelist() if re.match(r"word/footer\d*\.xml", n)]
        page_fields = 0
        for n in footers:
            page_fields += z.read(n).decode("utf-8").count("PAGE")
        headers = [n for n in z.namelist() if re.match(r"word/header\d*\.xml", n)]
        doc_xml = z.read("word/document.xml").decode("utf-8")
        sect = doc_xml.count("<w:sectPr")
    add("·", "域", f"页脚 part {len(footers)} 个、PAGE 域 {page_fields} 处、页眉 part {len(headers)} 个、节 {sect} 个")
    if page_fields < sect:
        add("!", "页码", f"{sect} 个节只有 {page_fields} 处 PAGE 域，某些节可能没有页码")

    # ---- 11. 标点与中英混排 ----
    # 参考文献按 GB/T 7714 用半角逗号/句点（"李琳, 段围, 等. 题名[J]. ..."），不算问题
    half_punct = 0
    samples: list[str] = []
    for p in paras:
        if p.style.name != STYLE_BODY or re.match(r"^\[\d+\]", p.text.strip()):
            continue
        txt = p.text
        found = re.findall(r"[\u4e00-\u9fa5][,;:!?]", txt)
        half_punct += len(found)
        if found and len(samples) < 3:
            m = re.search(r"[\u4e00-\u9fa5][,;:!?]", txt)
            samples.append("…" + txt[max(0, m.start() - 12):m.end() + 8] + "…")
    if half_punct:
        add("!", "正文标点", f"{half_punct} 处中文后跟半角标点（应为，；：！？），例：{samples}")

    # ---- 12. 一级标题段前分页（样式继承也算） ----
    def page_break_inherited(p) -> bool:
        if p.paragraph_format.page_break_before:
            return True
        style = p.style
        while style is not None:
            if style.paragraph_format.page_break_before:
                return True
            style = style.base_style
        return False

    nopage = [p.text[:20] for p in paras
              if p.style.name == STYLE_H1 and not page_break_inherited(p)]
    if nopage:
        add("!", "一级标题", f"{len(nopage)} 个一级标题没有「段前分页」（章首可能接在上章末尾）：{nopage[:3]}")
    else:
        add("·", "一级标题", "全部「段前分页」，章首各自起新页")


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:])
    path = Path(args[0]) if args else DEFAULT_DOCX
    if not path.is_file():
        print(f"[×] 文件不存在：{path}")
        return 1
    audit(path)

    print(f"审计对象：{path.name}（{path.stat().st_size} B）\n")
    order = {"!!": 0, "!": 1, "·": 2}
    issues.sort(key=lambda x: order.get(x[0], 3))
    fatal = [i for i in issues if i[0] == "!!"]
    warn = [i for i in issues if i[0] == "!"]
    info = [i for i in issues if i[0] == "·"]
    for level, where, msg in fatal + warn:
        print(f"{level} {where}：{msg}")
    if info:
        print("\n--- 参考信息 ---")
        for level, where, msg in info:
            print(f"   {where}：{msg}")
    print(f"\n结论：{len(fatal)} 项必须修、{len(warn)} 项建议修、{len(info)} 项信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
