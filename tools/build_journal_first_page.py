# -*- coding: utf-8 -*-
"""生成《计算机应用》期刊投稿第一页（马宇轩 / 南通大学）。

只用 python-docx，不读模板文件：按期刊常见版式手工排版，
字号与对齐方式写在 SPEC 里，便于按实际模板微调后重跑。
"""
from __future__ import annotations

import pathlib
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fix_docx_compat import apply as fix_compat  # noqa: E402

OUT = pathlib.Path("docs/期刊论文-第一页-马宇轩.docx")

TITLE_CN = "图结构引导的多智能体合同智能审查系统"
AUTHOR_CN = "马宇轩"
AFFIL_CN = "（南通大学 人工智能与计算机学院，江苏 南通 226019）"
EMAIL = "（通信作者电子邮箱 339861519@qq.com）"

ABSTRACT_CN = (
    "政府采购合同数量大、条款专业，人工审查成本高且结论难以复核。针对现有方法依赖大模型算力、"
    "版面还原粗糙、风险判定不可追溯的问题，设计并实现了一个图结构引导的多智能体合同智能审查系统。"
    "系统对 DOCX 与 PDF 做版面感知解析，文字层薄弱的页面转入光学字符识别通道；以规则证据与动态"
    "规划求全局最优的方式切分条款，构建“源文本块—条款—合同”三层契约；抽取当事人、金额、日期与"
    "标的四类要素并同步生成条款结构图与法条标签图；检索侧以字级稀疏检索、潜在语义检索与标签图"
    "传播三路召回，经加权倒数排名融合与学习型重排输出可溯源法条；审查侧由理解、条款审查、交叉"
    "一致性、法条适用与综合裁量五个智能体协同，经证据强度仲裁与置信度校准输出七类风险与评分。"
    "在 1260 条《中华人民共和国民法典》条文与 113 份合同上测试：法条检索强金标准召回率@5 由"
    " 0.628 升至 0.694，平均倒数排名@5 为 1.000；10 份对抗样本的注入缺陷全部检出；扫描件池化"
    "字符错误率 0.79%；校准后 Brier 分数由 0.2768 降至 0.0977；单份端到端耗时 0.02 至 0.36 秒。"
    "系统无需图形处理器与外部大模型，可作为律师审查前的辅助工具。"
)

KEYWORDS_CN = "合同智能审查；版面感知解析；要素抽取；图结构引导；置信度校准"
CLC = "中图分类号：TP391.1　　　文献标志码：A"

TITLE_EN = (
    "Graph-Guided Multi-Agent System\nfor Intelligent Contract Review"
)
AUTHOR_EN = "MA Yuxuan"
AFFIL_EN = (
    "(College of Artificial Intelligence and Computer Science, Nantong University, "
    "Nantong 226019, Jiangsu, China)"
)

ABSTRACT_EN = (
    "Government procurement contracts are large in number and highly specialized in wording, "
    "which makes manual review costly and its conclusions hard to verify. Existing approaches "
    "either depend on large-model computing power, restore document layout coarsely, or produce "
    "risk judgements that cannot be traced back to evidence. This paper designs and implements a "
    "graph-guided multi-agent system for intelligent contract review. Layout-aware parsing is "
    "performed on DOCX and PDF files to recognize columns, running heads and table rows, and "
    "pages with a thin text layer are routed to an optical character recognition channel. Clauses "
    "are segmented by combining rule evidence with dynamic programming that optimizes the whole "
    "document, yielding a three-level contract of source blocks, clauses and contract. Parties, "
    "amounts, dates and subject matter are extracted, and a clause-level structural graph together "
    "with a statute-label graph is built in the same pass. Sparse character-level retrieval, "
    "latent semantic retrieval and label-graph propagation are fused by weighted reciprocal rank "
    "fusion and re-ranked by a learned model, so that every cited provision is traceable to the "
    "local knowledge base. Five agents for understanding, clause review, cross-clause consistency, "
    "statute application and overall adjudication cooperate under evidence-strength arbitration "
    "and confidence calibration. Experiments on 1260 articles of the Civil Code of the People's "
    "Republic of China and 113 contracts show that recall@5 on the strict gold standard rises from "
    "0.628 to 0.694 with a mean reciprocal rank@5 of 1.000; all injected defects in ten "
    "adversarial samples are detected, with a recall of 1.000 and a precision of 0.571; the pooled "
    "character error rate on scanned pages is 0.79%; the Brier score drops from 0.2768 to 0.0977 "
    "after calibration; and the end-to-end cost per contract ranges from 0.02 to 0.36 seconds, with "
    "all 118 unit tests passing. The system requires neither a graphics processing unit nor an "
    "external large model, and serves as an auxiliary tool before lawyer review."
)

KEYWORDS_EN = (
    "contract review; layout-aware parsing; element extraction; graph-guided reasoning; "
    "confidence calibration"
)

INTRO = [
    "政府采购合同是财政资金支出的主要载体。以公开渠道可获取的合同公告为例，单一省级平台每周新增"
    "公告即达数十份，单份合同普遍在十余页以上，扫描件占比接近九成。审查这类合同需要逐条比对"
    "法条、核对当事人与金额、检查日期与条款之间是否自相矛盾，人工完成一份往往以小时计，且结论"
    "高度依赖审查人员的经验，不同人给出的意见难以对齐。常见的缺陷集中在四处：金额的小写数值与"
    "大写文本不一致、履行日期区间颠倒、当事人名称前后不统一、正文引用的条款号在实际文本中并不"
    "存在。这些问题单看每一条都不复杂，但分布在数十页文本中，人工极易漏检。",

    "已有研究大致分为三条路线。第一条以静态规则或模板匹配为主，把审查要点写成判定条件，优点是"
    "结论可解释，缺点是规则由人工维护，遇到非标准条款即失效。第二条把要素抽取与条款分类交给"
    "深度学习模型，效果依赖标注数据规模，且模型给出的判断往往缺少可追溯的依据。第三条借助大"
    "语言模型直接阅读合同并给出意见，通用性强，但推理成本高、结果不稳定，在离线终端上难以部署，"
    "更重要的是无法保证所引法条确实存在。三条路线的共同短板在于：文档解析被当成前置的黑盒，"
    "版面错乱导致条款被切碎，后续无论用多强的模型都无法补救。",

    "本文的工作包括四个方面。第一，把版面还原纳入正式流程，解析阶段保留字号、加粗、页面坐标与"
    "表格单元格，使跨栏文本按阅读顺序重组、页眉页脚与断行被正确剥离，并把文字层薄弱的页面显式"
    "标记后转入光学字符识别通道，而不是静默返回空文本。第二，条款切分不采用逐块贪心判定，而是"
    "把边界决策写成整份文档上的动态规划问题，以条款长度先验约束切分粒度，避免长条款被切碎。"
    "第三，检索侧不依赖单一通道，字级稀疏检索负责术语精确匹配，潜在语义检索负责同义改写，标签图"
    "传播负责补齐同一风险家族的条文，三路结果按倒数排名融合后再由学习型重排按候选集内标准化特征"
    "定序，并强制只返回本地知识库中真实存在的条文。第四，把审查拆成五个职责明确的智能体，条款级"
    "判定与跨条款一致性判定并行执行，冲突由证据强度仲裁，最终置信度经保序回归校准后输出，使"
    "“高置信”与“需人工复核”在数值上可分。",

    "全文结构如下：第 1 节回顾相关工作；第 2 节介绍版面感知、多智能体协同、多路召回融合与概率"
    "校准四项关键技术；第 3 节给出系统需求与总体目标；第 4 节说明体系结构、执行图与数据结构设计；"
    "第 5 节按模块描述实现；第 6 节报告测试环境、功能测试、性能测试与消融结果；最后总结全文并"
    "指出局限。",
]

FOOTNOTE = [
    "收稿日期：2026-xx-xx；修回日期：2026-xx-xx；录用日期：2026-xx-xx。",
    "基金项目：无（如获资助请按“项目名称（编号）”格式补充）。",
    "作者简介：马宇轩（2006—），男，江苏常州人，本科在读，主要研究方向：自然语言处理、法律智能。"
    "E-mail: 339861519@qq.com",
]

# 版式：字号按期刊常见值，实测模板后可在此微调
SPEC = {
    "title_cn": (18, "黑体", WD_ALIGN_PARAGRAPH.CENTER, 12, 6),
    "author_cn": (12, "宋体", WD_ALIGN_PARAGRAPH.CENTER, 6, 2),
    "affil_cn": (9, "宋体", WD_ALIGN_PARAGRAPH.CENTER, 2, 2),
    "email": (9, "宋体", WD_ALIGN_PARAGRAPH.CENTER, 0, 10),
    "abstract": (9, "宋体", WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 4),
    "keywords": (9, "宋体", WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 4),
    "clc": (9, "宋体", WD_ALIGN_PARAGRAPH.LEFT, 0, 14),
    "title_en": (16, "Times New Roman", WD_ALIGN_PARAGRAPH.CENTER, 10, 6),
    "author_en": (11, "Times New Roman", WD_ALIGN_PARAGRAPH.CENTER, 4, 2),
    "affil_en": (9, "Times New Roman", WD_ALIGN_PARAGRAPH.CENTER, 2, 2),
    "abstract_en": (9, "Times New Roman", WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 4),
    "h1": (10.5, "黑体", WD_ALIGN_PARAGRAPH.LEFT, 10, 4),
    "body": (10.5, "宋体", WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 0),
    "foot": (7.5, "宋体", WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 0),
}


def _set_font(run, size_pt: float, name: str) -> None:
    run.font.size = Pt(size_pt)
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), name if name != "Times New Roman" else "宋体")
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def para(doc, key: str, text: str, *, bold: bool = False, indent_chars: int = 0,
         first_line: bool = False):
    size, font, align, before, after = SPEC[key]
    p = doc.add_paragraph()
    p.alignment = align
    fmt = p.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if indent_chars:
        fmt.left_indent = Pt(size * indent_chars)
    if first_line:
        fmt.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    run.bold = bold
    _set_font(run, size, font)
    return p


def labeled(doc, key: str, label: str, text: str):
    """“摘要：”这类带标签的段落：标签加粗，正文常规。"""
    size, font, align, before, after = SPEC[key]
    p = doc.add_paragraph()
    p.alignment = align
    fmt = p.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r1 = p.add_run(label)
    r1.bold = True
    _set_font(r1, size, font)
    r2 = p.add_run(text)
    _set_font(r2, size, font)
    return p


def _footer_block(section) -> None:
    """把收稿日期/基金项目/作者简介写进“首页页脚”，只出现在第一页底部。"""
    section.different_first_page_header_footer = True
    footer = section.first_page_footer
    footer.is_linked_to_previous = False

    first = footer.paragraphs[0]
    first.text = ""
    first.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = first.paragraph_format
    fmt.space_before = Pt(4)
    fmt.space_after = Pt(0)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    # 页脚上方一条细线，和正文分开
    ppr = first._element.get_or_add_pPr()
    borders = ppr.makeelement(qn("w:pBdr"), {})
    top = borders.makeelement(
        qn("w:top"),
        {qn("w:val"): "single", qn("w:sz"): "4", qn("w:space"): "1", qn("w:color"): "808080"},
    )
    borders.append(top)
    ppr.append(borders)

    for index, line in enumerate(FOOTNOTE):
        para_obj = first if index == 0 else footer.add_paragraph()
        if index:
            para_obj.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            f = para_obj.paragraph_format
            f.space_before = Pt(0)
            f.space_after = Pt(0)
            f.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = para_obj.add_run(line)
        _set_font(run, 7.5, "宋体")


def build(out: pathlib.Path) -> dict:
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.6)

    # 页眉信息栏（投稿时按模板填写，录用后删除）
    para(doc, "foot", "您的稿件编号：（新投稿无编号时暂不填写）")
    para(doc, "foot", "本稿件需要修改时请联系 作者：")
    para(doc, "foot", "手机号：　　　　　　　邮箱：　　　　　　　doi: 10.11772/j.issn.1001-9081.yyyymmnnn")
    para(doc, "foot", "请务必确认好作者、单位、基金等相关信息，投稿成功后不得随意修改")

    para(doc, "title_cn", TITLE_CN, bold=True)
    para(doc, "author_cn", AUTHOR_CN)
    para(doc, "affil_cn", AFFIL_CN)
    para(doc, "email", EMAIL)

    labeled(doc, "abstract", "摘要：", ABSTRACT_CN)
    labeled(doc, "keywords", "关键词：", KEYWORDS_CN)
    para(doc, "clc", CLC)

    para(doc, "title_en", TITLE_EN, bold=True)
    para(doc, "author_en", AUTHOR_EN)
    para(doc, "affil_en", AFFIL_EN)
    labeled(doc, "abstract_en", "Abstract: ", ABSTRACT_EN)
    labeled(doc, "keywords_en" if "keywords_en" in SPEC else "keywords",
            "Keywords: ", KEYWORDS_EN)

    para(doc, "h1", "0  引言", bold=True)
    for chunk in INTRO:
        para(doc, "body", chunk, first_line=True)

    _footer_block(section)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    # python-docx 默认模板是旧 compatMode，Word 会以“兼容性模式”打开，这里统一修到 15
    compat = fix_compat(out)

    text = "\n".join(t for t in (dd.text for dd in doc.paragraphs) if t.strip())
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "compat": compat,
        "paragraphs": len([d for d in doc.paragraphs if d.text.strip()]),
        "cjk_chars": sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"),
        "abstract_cn_chars": len(ABSTRACT_CN),
        "abstract_en_words": len(ABSTRACT_EN.split()),
        "title_cn_chars": len(TITLE_CN),
        "keywords_cn": KEYWORDS_CN.count("；") + 1,
        "intro_chars": sum(len(s) for s in INTRO),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build(OUT), ensure_ascii=False, indent=2))
