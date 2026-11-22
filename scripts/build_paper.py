"""构建毕业论文 .docx（图片取自 paper_figures/）。"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_figures"
OUT = ROOT / "基于多智能体协同与知识图谱的政府采购合同智能审查系统设计与实现.docx"

doc = Document()

# ------------------------------------------------------------------ 基础样式
normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)

section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.top_margin = Cm(2.54)
section.bottom_margin = Cm(2.54)
section.left_margin = Cm(2.8)
section.right_margin = Cm(2.6)


def font(run, size=12, bold=False, name="宋体", color=(0, 0, 0)):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def P(text="", indent=True, size=12, bold=False, align=None, space_before=0, space_after=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    if text:
        font(p.add_run(text), size, bold)
    return p


def H(text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level == 1 else 10)
    pf.space_after = Pt(10 if level == 1 else 6)
    pf.line_spacing = 1.5
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    font(p.add_run(text), 16 if level == 1 else (14 if level == 2 else 12), True)
    return p


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    font(p.add_run(text), 10.5, True)


def figure(name, width_cm, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    caption(cap)


def shade(cell, color="EDEDED"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def table(headers, rows, cap, widths=None, size=10.5, align_cols=None):
    caption(cap)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头跨页重复，且每一行不在页间断开
    _tr = t.rows[0]._tr.get_or_add_trPr()
    _th = OxmlElement("w:tblHeader")
    _th.set(qn("w:val"), "true")
    _tr.append(_th)
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        font(p.add_run(h), size, True)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            mode = (align_cols[i] if align_cols else WD_ALIGN_PARAGRAPH.LEFT)
            p.alignment = mode
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            font(p.add_run(str(value)), size, False)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    for row in t.rows:
        _trPr = row._tr.get_or_add_trPr()
        _cs = OxmlElement("w:cantSplit")
        _trPr.insert(0, _cs)
    if len(rows) <= 6:
        # 小表整体不跨页：除最后一行外，其余行与下一行同页
        for row in t.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    P("", indent=False, space_after=6)


def page_break():
    doc.add_page_break()


def code_block(lines):
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.left_indent = Cm(0.75)
        pf.line_spacing = 1.0
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.keep_with_next = i < len(lines) - 1
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0, 0, 0)
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    P("", indent=False, space_after=6)


def footer_page_number():
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    font(run, 10.5)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    node = OxmlElement("w:r")
    txt = OxmlElement("w:t")
    txt.text = "1"
    node.append(txt)
    fld.append(node)
    p._p.append(fld)


def toc_field():
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-3" \h \z \u')
    node = OxmlElement("w:r")
    txt = OxmlElement("w:t")
    txt.text = "（在 Word / WPS 中右键选择“更新域”即可生成带页码的目录）"
    node.append(txt)
    fld.append(node)
    p._p.append(fld)


footer_page_number()

# ------------------------------------------------------------------ 封面
for _ in range(4):
    P("", indent=False)
P("毕业论文", indent=False, size=26, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3):
    P("", indent=False)
P("论文题目：", indent=False, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
P("基于多智能体协同与知识图谱的", indent=False, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
P("政府采购合同智能审查系统设计与实现", indent=False, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
for _ in range(3):
    P("", indent=False)
for _line, _tail in (
    ("学生姓名：", "____________________"),
    ("指导老师：", "____________________"),
    ("学　　号：", "____________________"),
    ("专　　业：", "计算机科学与技术"),
):
    _p = P(_line + _tail, indent=False, size=14, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=10)
    _p.paragraph_format.left_indent = Cm(3.6)
for _ in range(4):
    P("", indent=False)
P("2026 年 5 月", indent=False, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ------------------------------------------------------------------ 目录
P("目　　录", indent=False, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=12)
toc_field()
page_break()

# ------------------------------------------------------------------ 摘要
H("摘  要", 1)
P("政府采购合同数量大、专业性强，风险往往隐藏在表述细节与跨条款关系之中。人工审查依赖审查人员经验，"
  "普遍存在效率低、尺度不一致、结论难以追溯到原文与法律依据等问题；同时政府采购公告中的合同多以扫描件形式发布，"
  "进一步抬高了自动审查的门槛。本文以房屋租赁、买卖、委托、中介与保管类合同为对象，设计并实现了一套可追溯、"
  "可解释、可复核的合同智能审查系统。")
P("论文的主要工作包括四个方面。其一，建立合同统一数据结构，把解析得到的原子区块组织为条款树，"
  "条款、要素与风险条目均可回溯到页码、区块与原文片段。其二，提出规则与模型双引擎的条款切分方法，"
  "以结构一致性校验识别区块遗漏、层级断裂与重复编号，并把受影响的局部窗口回流给模型修正，"
  "回流结果仅在可修正问题数严格减少时被采纳。其三，构建覆盖《中华人民共和国民法典》1260 条的法律知识库，"
  "采用稠密向量与稀疏 BM25 双路召回并经倒数排序融合，法条引用被严格限制在召回池内。其四，"
  "以 LangGraph 把合同理解、逐条审查、跨条款审查、法律适用、裁判与仲裁组织为一张统一状态图，"
  "配合七分类评估、三条硬护栏与四条有界回环，实现“不违法不等于合理”的可解释输出，并在模型失效时保守降级。")
P("为支撑上述工作，本文还实现了政府采购合同扫描件的采集程序，从北京市政府采购网获取 63 份合同公告附件，"
  "整理为 1265 页的页级图像数据集，其中 1152 页（91.1%）无有效文字层，须经 OCR 与版面分析后方可进入审查流程。"
  "系统现有 123 个单元与集成用例全部通过；样例合同的 36 个原子区块被切分为 23 条条款，"
  "结构知识图谱生成 25 个节点、19 条边，并检出 2 处悬空条款引用。")
P("关键词：合同智能审查；多智能体协同；知识图谱；检索增强生成；LangGraph", indent=False, bold=True, space_before=12)
page_break()

# ------------------------------------------------------------------ Abstract
H("Abstract", 1)
P("Government procurement contracts are numerous and highly specialised, and their risks are often hidden in "
  "wording details and cross-clause relations. Manual review depends on the experience of reviewers and suffers from "
  "low efficiency, inconsistent standards, and conclusions that cannot be traced back to the original text or legal "
  "basis. Meanwhile, a large number of published contracts are scanned images, which further raises the threshold for "
  "automatic review. Taking lease, sale, entrustment, brokerage and custody contracts as the object, this thesis "
  "designs and implements a traceable, explainable and reviewable contract review system.", indent=True)
P("The work consists of four parts. First, a unified contract data structure is established, in which atomic blocks "
  "are organised into a clause tree, so that clauses, elements and risk items can be traced back to page numbers, "
  "blocks and source snippets. Second, a dual-engine clause segmentation method combining rules and a large language "
  "model is proposed; structural validation detects missing blocks, broken hierarchy and duplicated numbering, and "
  "only the affected local windows are fed back to the model, with the revision accepted solely when the number of "
  "fixable problems strictly decreases. Third, a legal knowledge base covering 1260 articles of the Civil Code of "
  "China is built, retrieved through dense embeddings and sparse BM25 with reciprocal rank fusion, and citations are "
  "restricted to the retrieved pool. Fourth, LangGraph organises contract understanding, clause-level review, "
  "cross-clause review, legal application, adjudication and arbitration into a single state graph, together with a "
  "seven-class evaluation, three hard guardrails and four bounded loops, so that the output is explainable and "
  "conservatively degrades when models are unavailable.", indent=True)
P("To support the above work, a crawler is implemented for scanned government procurement contracts. Sixty-three "
  "contract attachments are collected from the Beijing government procurement portal and organised into a page-level "
  "image dataset of 1265 pages, 1152 of which (91.1%) contain no usable text layer and therefore require OCR and "
  "layout analysis before entering the review pipeline. All 123 unit and integration tests pass; a sample contract "
  "with 36 atomic blocks is segmented into 23 clauses, and its structural knowledge graph contains 25 nodes and 19 "
  "edges with 2 dangling clause references detected.", indent=True)
P("Keywords: contract review; multi-agent collaboration; knowledge graph; retrieval-augmented generation; LangGraph",
  indent=False, bold=True, space_before=12)
page_break()

# ------------------------------------------------------------------ 第1章
H("第1章  绪论", 1)
H("1.1  研究背景及意义", 2)
P("政府采购是公共财政支出的重要环节，合同则是采购结果的法律载体。随着电子化采购的普及，合同公告数量持续增长，"
  "一份合同动辄数十页，条款之间还存在金额口径、履约期限、权利义务分配等相互牵连的关系。合同审查因此成为一项"
  "高频且高强度的工作：审查人员既要判断条款表述是否与法律规定相符，也要判断双方权利义务是否失衡、"
  "商业安排是否合理，而这些判断往往取决于个别措辞与跨条款的呼应关系。")
P("人工审查的局限主要体现在三个方面。第一是专业壁垒，《中华人民共和国民法典》合同编条文众多，"
  "合同成立、效力、履行与违约责任的判断需要成体系的法律知识，合同成立的认定本身在司法实践中即有诸多争议[1]。"
  "第二是规模压力，审查人员难以对每一份合同的每一条款逐一核对，容易遗漏隐藏在高风险表述中的问题。"
  "第三是可追溯性不足，审查意见常以自然语言给出，缺少“结论—条款—原文片段—法律依据”的完整链条，"
  "复核与追责都缺少抓手。")
P("另一重困难来自文档形态。政府采购公告中的合同附件大量以扫描件发布，正文没有文字层，"
  "或者仅在一整页图片上叠加少量不可靠的文字。此类文档无法直接进入自然语言处理流程，"
  "必须先完成版面分析与光学字符识别，才能得到可切分、可定位的文本。因此，一个可用的合同智能审查系统，"
  "既要解决“怎么审”的语义问题，也要解决“怎么读”的文档问题。")
P("政府采购合同按标的可分为货物、服务与工程三类，三类合同的审查重点并不相同。货物类合同关注交付标准、"
  "验收方式与质保期；服务类合同关注服务范围、人员配置与考核方式；工程类合同则涉及工期、变更、"
  "结算与分包等复杂安排。审查人员需要在同一套流程下识别不同类型的风险，这对系统的条款角色识别能力"
  "提出了较高要求：同一条款在不同合同类型中的功能可能完全不同，例如“验收”在货物合同中指向交付确认，"
  "在服务合同中则可能指向阶段考核。系统因此把合同类型判断放在审查流程的最前端，"
  "并把类型信息作为条款角色识别与法条排序的弱特征，而不是硬性过滤条件。")
P("本文的研究意义在于：一方面，把多智能体协同、检索增强生成与知识图谱引入合同审查，"
  "将审查过程拆解为可独立验证的环节，使每一条风险结论都有明确的产生路径；另一方面，"
  "通过统一数据结构与硬性护栏，把大语言模型的不确定性约束在可接受范围内，"
  "为政府采购合同的辅助审查提供一套可落地、可复核的技术方案，也为后续接入 OCR 与更大规模真实合同数据奠定基础。")

H("1.2  国内外研究综述", 2)
H("1.2.1  国内研究现状", 3)
P("国内对合同智能处理的研究大致沿两条路线展开。一条是合同分类与要素识别，"
  "周红等针对建设工程合同提出智能分类方法，通过文本特征与类别体系把合同归入相应类型，"
  "为后续的模板化审查提供了前置条件[2]。另一条是面向具体业务场景的审查系统，"
  "蔡颢婷等构建了基于大语言模型的医院智能合同审查系统，把模型能力嵌入医院合同审核流程，"
  "在真实业务中验证了自动审查的可行性[3]。")
P("在方法层面，范雨晴等提出的 CTML 模型把对比学习引入合同文本表示，"
  "缓解了合同语料类别不均衡与表述差异大的问题，提升了相似条款的判别能力[4]。"
  "法律基础研究方面，刘贵祥对合同成立的要件、要约与承诺的界分以及格式条款的效力进行了系统论述[1]，"
  "这些法理判断正是自动审查中“是否成立、是否生效、是否显失公平”等规则的知识来源。"
  "总体来看，国内研究已从单一分类任务走向面向业务的审查系统，但仍普遍存在两个短板："
  "一是审查结论与原文的定位关系较弱，难以支撑逐条复核；二是法律依据多依赖模型记忆，缺少可验证的检索与引用机制。")
P("国内研究的另一类工作集中在合同要素抽取与风险提示。受限于标注数据稀缺，早期系统多依赖模板与规则，"
  "在排版规整的示范文本上表现尚可，一旦遇到编号混排、表格嵌套或扫描件便难以维持精度。"
  "近年来随着通用大模型能力提升，研究者开始把抽取任务改写为结构化生成任务，"
  "以字段约束代替特征工程。但这类方法带来新的问题：模型可能改写原文、遗漏长合同中的条款，"
  "且难以对每一条抽取结果给出可核验的依据。这正是本文强调“原文一致性”与“来源定位”的原因。")
H("1.2.2  国外研究现状", 3)
P("国外研究以公开数据集和基准任务为牵引。Hendrycks 等发布的 CUAD 数据集标注了 510 份商业合同中的 41 类关键条款，"
  "成为合同要素抽取事实上的标准评测集[5]；Koreeda 与 Manning 构建的 ContractNLI 把合同条款间的蕴含关系转化为"
  "自然语言推理任务，强调合同判断本质上是“给定条款能否支持某结论”的证据推理[6]；"
  "Tuggener 等构建的 LEDGAR 则提供了大规模合同条款类型标注，用于条款分类与分割研究[7]。")
P("在系统层面，Zheng 等提出以知识图谱增强大语言模型的施工合同审查方法，"
  "把领域概念与条款关系显式建模，弥补模型对专业术语理解的不足[8]；"
  "Liu 等发布的 ContractEval 从条款覆盖率与风险识别两个维度系统评估模型审查能力，"
  "指出模型在长合同上的召回明显下降[9]；Raptopoulos 等提出的 PAKTON 采用多智能体框架处理长篇合同，"
  "通过分工与迭代把长文档拆解为可处理的片段[10]。"
  "此外，面向监管合规的多智能体知识图谱方案[11]与基于图强化学习的法律推理方法[12]也表明，"
  "把结构化知识与智能体协作结合，是提升法律文本处理可靠性的有效途径。")
H("1.2.3  研究小结", 3)
P("近年来的研究呈现出三个趋势：从单任务走向端到端审查、从纯模型走向知识与检索增强、"
  "从单次推理走向多智能体协作。Yun 等提出的 Graph-of-Agents 把多个智能体的推理过程组织为图结构，"
  "在长文档问答上显著优于单智能体串联[13]；Li 等的 LegalOne 在法律任务上系统比较了不同智能体编排策略，"
  "指出角色分工与结果融合比单纯增大模型规模更有效[14]。"
  "评测方面，LawBench 从法律知识记忆、理解与应用三个层次评估模型能力[15]，"
  "ProvBench 则强调结论必须可追溯到证据片段[16]，这两点恰好对应合同审查最需要的“懂法”与“可追溯”。")
P("综合来看，现有工作为本研究提供了数据集、方法与评测框架，但仍存在可改进之处："
  "多数系统把审查视为一次性的模型调用，缺少对失败情形的兜底；风险结论常以二元“违法/不违法”输出，"
  "忽略了“不违法但不合理”的广泛中间地带；法律引用缺少召回池约束，存在模型自造法条的风险。"
  "本文针对上述问题，以统一数据结构、混合检索与有界多智能体图为核心，构建可追溯的合同审查系统。")

# ------------------------------------------------------------------ 第2章
page_break()
H("第2章  关键技术简介", 1)
H("2.1  大语言模型与结构化输出", 2)
P("本文以 Qwen3 系列模型作为语义理解与推理的基础模型[17]。该系列模型在中文理解、长文本处理与指令遵循方面表现稳定，"
  "且提供不同参数规模以适配本地部署与接口调用两种形态。合同审查对输出的结构要求很高："
  "模型不能只给出一段评述，而要按预先约定的字段返回风险码、风险类别、严重程度、涉及条款与判断理由。"
  "因此系统对所有模型调用都采用 JSON Schema 约束的结构化输出，并在解析失败时按“字段缺失即视为无效”处理，"
  "避免把不完整的模型输出直接写入报告。")
P("提示工程方面，系统把每个智能体的职责、可用输入、禁止事项写入系统提示，"
  "例如逐条审查智能体只允许依据所给条款提出风险，不得自行引用法条；法律适用智能体只允许从检索结果中选择条文。"
  "这种“职责边界先于能力”的设计，使模型即便能力很强，也不会越权产生无法验证的结论。")
H("2.2  检索增强生成与混合检索", 2)
P("检索增强生成通过在生成前引入外部知识，缓解模型知识陈旧与幻觉问题[18]。"
  "合同审查的知识来源是法条，而法条检索有两个特点：条文表述高度书面化，同一含义存在多种措辞；"
  "同时条文编号、术语又是精确匹配的强信号。单一检索方式难以同时满足这两点需求，因此系统采用混合检索。")
P("稠密通道使用 bge-m3 嵌入模型[19]，把条款文本与法条文本映射到同一向量空间，"
  "以 1024 维余弦相似度衡量语义接近程度；稀疏通道使用 BM25[20]，"
  "在字级 unigram 与 bigram 上统计词频，参数取 k1=1.5、b=0.75，负责捕捉条文中的专业术语与编号等精确线索。"
  "两路结果通过倒数排序融合（Reciprocal Rank Fusion，RRF）合并[21]，"
  "其优势是不依赖两路分值的量纲，只利用排名信息，因而在稠密与稀疏分数不可比时依然稳健。")
H("2.3  LangGraph 状态图与多智能体编排", 2)
P("LangGraph 以状态图的方式组织多个处理节点：节点读取共享状态、返回状态增量，条件边根据状态决定下一节点。"
  "与线性链式调用相比，状态图天然支持并行分支与有界回环，适合把“理解—审查—检索—裁判—仲裁—融合”这类"
  "既有顺序又有反馈的流程显式建模。系统中所有节点共享同一份审查状态，"
  "包括合同对象、要素、候选风险、法条适用结果、裁判结论、冲突记录与各类计数器；"
  "每个节点只写入自己负责的字段，避免相互覆盖。")
P("多智能体协作的关键不在于智能体数量，而在于职责划分与结果收敛方式。"
  "本文把审查拆为语义理解、条款级审查、跨条款审查、法律适用、裁判与仲裁六类角色，"
  "并规定候选风险必须经过“合并—评估—裁决—仲裁—融合”才能进入最终报告，"
  "任何单个智能体都不能单独决定删除一条风险。")
H("2.4  文档版面分析与光学字符识别", 2)
P("扫描件合同先要转化为带位置信息的文本。版面分析方面，LayoutLMv3 把文本、布局与图像统一建模，"
  "在表单与文档理解任务上表现良好[22]，其思路可用于识别合同中的标题、条款编号、签署区与表格区域。"
  "文字识别方面，PP-OCR 系列以检测与识别两阶段流水线在中文场景下取得较好的精度与速度平衡[23]，"
  "适合批量处理公告附件。")
P("本文在数据侧完成了扫描件的采集与页级图像整理，并把 PDF 中可提取的文字层保存为逐页文本，"
  "作为 OCR 的弱标注：对同一页，图像用于识别，文字层用于计算字符错误率，"
  "从而在没有人工标注的情况下也能粗略评估识别质量。OCR 与版面分析的接入属于后续工作，"
  "本文的系统在接口层面已按“页图像 + 页码 + 坐标”的形式预留。")
P("需要说明的是，扫描件的识别难度并不均匀。公告附件中的合同通常由 Word 排版后打印再扫描，"
  "正文以宋体为主、版式规整，识别难度较低；但签署页、骑缝章、手写批注与加盖的红色印章会明显干扰识别，"
  "印章覆盖处常出现字符粘连或漏识。因此版面分析不仅要定位条款，还要识别印章与手写区域，"
  "并在这些区域降低识别置信度，避免把噪声当作条款正文送入审查流程。")
H("2.5  知识图谱与结构化审查", 2)
P("知识图谱在本系统中承担两类职责。一类是合同结构图：以条款与要素为节点，"
  "以父子关系、要素定位关系与条款引用关系为边，把合同的层次结构与交叉引用显式表达出来，"
  "从而可以机械地检出“引用了不存在的条款”“金额缺少角色”等结构性问题。"
  "另一类是法条图谱：以法条为节点、以风险标签为连接，形成风险与法条的二部图，"
  "在向量召回之后按标签扩展相邻法条，弥补纯语义检索对同义风险表述的遗漏。")

# ------------------------------------------------------------------ 第3章
page_break()
H("第3章  系统分析", 1)
H("3.1  系统目标", 2)
P("系统的总体目标是：把一份合同转化为一份可追溯、可解释、可复核的结构化审查报告，"
  "使审查人员能够在数分钟内定位到需要关注的风险条款，而不是通读全文。具体目标包括四项。")
P("（1）读得进。支持 DOCX、文本型 PDF 与 TXT 三类文件，保留页码、区块顺序与原文映射；"
  "对扫描件，保留页图像与页码，为后续 OCR 预留接口。")
P("（2）切得准。把合同切分为层级清晰的条款树，切分结果必须覆盖全部原文区块，"
  "且切分后拼接的文本与原文完全一致，不允许模型“改写”原文。")
P("（3）审得全。既检查单条条款的显失公平、要素缺失、表述歧义，也检查跨条款的金额口径冲突、"
  "日期矛盾、主体不一致与引用错误；并对每条候选风险给出法律依据或明确标注“事实不足”。")
P("（4）说得清。每条结论都要能回指条款与区块；当模型不可用时，系统必须明示“未执行实质审查”，"
  "而不是给出看似正常的空报告。")
H("3.2  系统功能需求分析", 2)
P("系统的使用者分为业务人员、审查人员与管理员三类，各自的关注点不同，功能需求也存在差异。")
H("3.2.1  业务人员", 3)
P("业务人员负责上传合同与初筛。其核心需求是快速得到合同的基本画像：合同类型、当事人、标的、金额、"
  "履行期限与主要义务，并能在页面上看到条款原文与所在页码。业务人员不需要理解全部法律判断，"
  "但需要在提交审查前确认系统正确识别了合同的关键要素，避免因解析错误导致后续审查偏离。"
  "因此系统提供要素回填与结构预览功能，允许在审查前人工确认要素是否正确。")
H("3.2.2  审查人员", 3)
P("审查人员是系统的核心用户。其需求包括：查看按严重程度排序的风险清单；对每条风险查看涉及的条款、"
  "原文片段、法律依据与判断理由；对存疑结论进行人工复核并记录复核意见；对系统标记为“需补充材料”的条目"
  "上传补充材料后重新审查。审查人员还需要一份整体结论，包括风险等级、合理性评分与概述，"
  "以便快速判断该合同是否可以进入下一环节。")
H("3.2.3  管理员", 3)
P("管理员负责系统运行所依赖的知识与配置：维护法条库及其风险标签、触发索引重建、配置模型与检索参数、"
  "查看运行日志与失败记录。管理员不直接参与具体合同审查，但需要保证法条版本与检索索引的一致性，"
  "并在模型服务异常时能够快速定位问题。")
H("3.2.4  系统业务流程", 3)
P("系统的业务流程分为六个环节。第一是上传与解析，用户上传合同后，系统在数秒内完成解析并给出页数、"
  "区块数与识别到的合同类型，同时对超限文件给出明确提示。第二是结构预览，系统展示条款树与每一条款的"
  "页码范围，用户可折叠或展开，确认切分是否符合预期。第三是要素确认，系统列出抽取到的当事人、金额、"
  "日期与标的，用户可就明显错误的值进行修正，修正结果会写回合同对象并影响后续审查。")
P("第四是智能审查，用户点击开始审查后，系统按统一图依次执行理解、要素、逐条审查、跨条款审查、"
  "合并、法律适用、裁判、仲裁与融合，界面显示各节点的执行状态与回环次数。第五是结果复核，"
  "用户按严重程度浏览风险清单，对每条风险查看条款原文与法条引用，勾选“已确认”或“有异议”并填写意见。"
  "第六是报告导出，系统把复核结果与原始报告合并，输出可归档的结构化报告与可读的清单版本。")
P("六个环节中，人工介入集中在第三与第五环节，其余环节自动完成。这样的分工使系统既保持了自动化效率，"
  "又把最终判断权留给审查人员，符合辅助审查而非替代审查的定位。")
H("3.3  系统非功能性需求分析", 2)
H("3.3.1  实用性", 3)
P("合同审查是专业工作，系统输出必须符合审查人员的阅读习惯：风险按严重程度排序、结论在前、证据在后、"
  "引用格式统一。界面应避免堆砌技术细节，把模型内部字段折叠为“展开查看证据”的形式。"
  "同时，系统应容忍不规范的输入，例如编号风格混用、条款标题缺失、表格与正文混排，"
  "在这些情况下仍应给出可用的切分结果，而不是直接失败。")
H("3.3.2  可靠性", 3)
P("可靠性体现在三个方面。第一是结果可靠：法条引用只能来自召回结果，模型不得自造条文；"
  "风险条目在缺乏裁决时不得被删除。第二是过程可靠：所有模型调用都有超时与重试，"
  "重试次数有上限，避免流程因单点失败而挂起。第三是降级可靠：模型服务不可用时，"
  "系统输出 unknown 并明确提示“未执行实质审查，结果不可作为合同结论”，同时把疑点转入人工队列。"
  "系统还实现了全离线测试，123 个用例不依赖外部模型服务即可运行，保证核心逻辑可回归。")
H("3.3.3  可扩展性", 3)
P("系统在四个层面预留扩展能力：解析层可新增文件格式；切分层可替换切分引擎而不影响下游；"
  "知识层可扩充法条库与风险标签并重建索引；编排层可在状态图中新增节点或调整条件边。"
  "数据结构以 schema_version 标识版本，新增字段不破坏既有消费者。")

# ------------------------------------------------------------------ 第4章
page_break()
H("第4章  系统设计", 1)
H("4.1  系统架构设计", 2)
P("系统采用分层架构，自上而下分为表现层、编排层、智能体层、知识层与基础设施层，如图 4-1 所示。"
  "分层的目的在于把“文档怎么读”“条款怎么切”“风险怎么判”“结论怎么合”四类问题彼此隔离，"
  "任一层的变化都不会穿透到其他层。")
figure("fig4_1_architecture.png", 14.5, "图 4-1  系统总体架构图")
P("表现层负责合同上传、报告预览、风险条目定位与人工复核队列，是审查人员唯一直接接触的界面。"
  "编排层是 M6 统一状态图，负责节点调度、并行分支与有界回环，本身不含业务判断。"
  "智能体层由合同理解、逐条审查、跨条款审查、法律适用与裁判仲裁五个智能体构成，各自只处理一类判断。"
  "知识层提供文档解析、条款切分、要素抽取、结构图谱与法条检索五项能力，是智能体的证据来源。"
  "基础设施层包括 PyMuPDF、LangGraph、Qdrant、Ollama 与模型接口，为上层提供解析、编排、向量检索与推理服务。")
P("分层带来的直接好处是替换成本可控。例如把规则切分替换为另一套切分模型，只需保证输出的条款树满足同一契约，"
  "下游的要素、图谱与审查节点无需改动；把 bge-m3 换成其他嵌入模型，只需重建索引并保持向量维度声明一致。"
  "代价是层间契约需要显式维护，因此系统以 schema_version 标识数据结构版本，"
  "并在测试中对契约字段做完整性断言，避免某一层悄悄改变字段含义。")
H("4.2  系统功能设计", 2)
H("4.2.1  文档解析与条款切分设计", 3)
P("解析阶段的输入是一份合同文件，输出是若干原子区块。DOCX 直接解析 OOXML，保留段落与表格顺序、"
  "Word 自动编号、标题样式与加粗信息；文本型 PDF 由 PyMuPDF 按页解析，保留页码与文本块坐标；"
  "TXT 按行组织并自动识别编码。为避免异常文件拖垮服务，解析层设置了文件大小、解压体积、页数与字符数上限。")
P("切分阶段把原子区块组织为条款树，流程如图 4-2 所示。规则引擎依据编号形式（第X章、第X条、1.1、一、（一）等）"
  "与样式特征判定相邻区块的关系，输出四类边界决策：续接、同一条款、新子条款与新条款；"
  "模型引擎则把全部区块一次性交给大语言模型，由模型给出条款标签、标题、层级与区块编号列表。"
  "两种结果都要经过结构一致性校验，检查区块遗漏、重复引用、父子层级异常、编号重复与低置信度边界。")
figure("fig4_4_split.png", 15.0, "图 4-2  条款切分与结构回流流程")
P("校验发现的问题并不会整体重跑，而是把受影响的局部窗口连同问题说明交回模型修正，"
  "回流默认一轮，并且只有当可修正问题数严格减少时才采纳新结果，否则保留上一轮输出。"
  "这一取舍保证了回流只会让结果变好或不变，不会因为模型随机性而退化。")
H("4.2.2  统一审查图设计", 3)
P("统一审查图是系统的核心，如图 4-3 所示。合同理解节点只执行一次，输出合同类型、条款角色与主要义务，"
  "不重复抽取当事人与标的，避免与要素节点产生口径冲突。随后三路并行：要素节点回填要素槽位并计算数学信号，"
  "逐条审查节点按条款提出风险候选，跨条款审查节点检查金额、日期、主体与引用的一致性。")
P("三路结果在合并节点按“风险码 + 条款重叠”去重，保留最强证据并记录全部来源。"
  "法律适用节点对每条候选执行混合检索，把候选归入七类之一并给出引用；"
  "裁判节点逐条裁决是否保留、严重程度如何、是否需要转人工，并给出整体合理性结论；"
  "仲裁节点处理跨模块冲突，融合节点执行确定性去重、交叉印证、排序与证据链组装。")
P("图中四条虚线为有界回环：检索重试用于候选没有评估结论的情形，以更宽的查询重新检索；"
  "裁判重试用于模型调用失败；证据回环把“事实不足/需补充材料”的结论退回条款审查，补充证据后重新检索；"
  "仲裁重裁用于仲裁发现实质冲突时，把仲裁意见带回裁判节点重新裁决。四条回环的计数上限均为 1，"
  "超限即终止并转人工，从根本上避免流程在模型反复自我否定时无限循环。")
P("状态字段的设计遵循单一写入者原则：合同对象由要素节点改写，条款候选只由逐条审查节点写入，"
  "跨条款候选只由跨条款审查节点写入，法条适用结果只在法律适用节点与证据回环的清理动作中变化，"
  "裁判结论与回环计数由裁判节点负责。单一写入者使并行分支不会互相覆盖，"
  "也使“某条结论从何而来”在状态层面即可回答。合并节点只读取两路候选并做去重，不修改候选本身，"
  "从而保留了每一路智能体的原始判断，为后续的冲突仲裁提供依据。")
figure("fig4_2_graph.png", 14.0, "图 4-3  多智能体统一审查图与四条有界回环")
H("4.2.3  报告与复核设计", 3)
P("报告以统一结构输出，包含合同理解、要素、要素信号、风险清单、冲突记录、整体结论、复核队列与统计信息。"
  "风险条目携带风险码、类别、严重程度、消息、法条引用、条款与区块定位、置信度、是否交叉印证、"
  "判断理由与来源智能体。复核队列记录所有未能自动得出结论的条目及其原因，保证“不确定”不被静默丢弃。")
H("4.3  系统模块设计", 2)
H("4.3.1  解析与切分模块", 3)
P("该模块由解析器、边界判定器、结构校验器与回流控制器四部分组成。解析器把文件转成区块；"
  "边界判定器对每一对相邻区块给出关系与置信度；结构校验器以条款树为对象检查结构完整性；"
  "回流控制器负责挑选局部窗口、调用模型并比较修正前后的可修正问题数。"
  "模块的输出是条款树以及每个条款的页码范围与来源区块列表。")
H("4.3.2  要素抽取与结构图谱模块", 3)
P("要素抽取采用规则与模型双通道。规则通道覆盖当事人、金额、日期与标的四类要素，"
  "通过角色词、货币前缀、日期归一化与地址模式实现精确抽取；模型通道负责规则无法表达的语义型标的与措辞变体。"
  "两通道结果按归一化键合并，同值要素保留置信度更高的记录。")
P("结构图谱模块在条款树之上构建合同结构图：条款与要素为节点，父子关系、要素定位关系与条款引用关系为边。"
  "图谱用于两类结构信号：一是悬空引用，即某条款引用的条款编号在图中不存在；"
  "二是金额角色缺失，即金额节点没有关联到合同总价、定金、租金等角色。"
  "这些信号不直接生成风险，而是作为独立佐证或人工复核线索。")
H("4.3.3  法律检索模块", 3)
P("法律检索模块负责把风险候选映射到法条，流程如图 4-4 所示。查询由风险码、条款原文与要素摘要组成，"
  "分别进入稠密与稀疏两路召回，再经 RRF 融合得到 top-k 法条池。"
  "法条图谱在召回之后按风险标签扩展相邻法条，以弥补同义表述造成的漏召回。"
  "评估环节把候选归入七类，并执行引用白名单校验：若模型给出的条文不在召回池中，该引用被置空，"
  "只保留模型对风险的定性判断。")
figure("fig4_3_retrieval.png", 15.0, "图 4-4  混合检索与法条图谱增强流程")
P("检索默认不对合同类型做硬过滤。原因是元数据标注覆盖不全，硬过滤会漏掉跨类型适用的条文，"
  "例如“无处分权”条款虽标注为买卖专用，但租赁标的权属不清时同样适用。类型信息仅作为排序时的弱特征。")
H("4.3.4  统一图编排模块", 3)
P("编排模块以共享状态为中心。状态包含合同对象、要素、要素信号、条款候选、跨条款候选、合并候选、"
  "法条适用结果、裁判结论、冲突记录、复核队列、回环计数与错误列表。"
  "每个节点只返回自己负责的字段增量，图在编译时绑定检索器与模型传输对象，"
  "并在运行结束后统一关闭检索器，避免本地向量库的目录锁被长期占用。")
H("4.4  数据结构设计", 2)
P("系统以 Contract 为全流程接口契约，内部结构见表 4-1。所有结构都以标识符关联，"
  "不复制原文，保证同一段文本在条款、要素与风险三处引用时指向同一来源。")
table(
    ["结构", "关键字段", "说明"],
    [
        ["SourceBlock", "block_id、order、kind、text、raw_text、page、metadata",
         "段落级原子区块，规范化文本与原文分开保存"],
        ["Clause", "clause_id、label、title、level、heading、body、parent_id、block_ids、page_start、page_end、source_spans",
         "条款树节点，可回溯到区块与页码"],
        ["BoundaryDecision", "boundary_id、左右区块、relation、confidence、decision_source、evidence、review_required",
         "相邻区块关系与判断依据，含模型复核字段"],
        ["Element", "element_id、kind、value、normalized、role、clause_id、block_ids、confidence、source",
         "要素记录，source 区分规则与模型来源"],
        ["RiskItem", "risk_id、code、category、severity、citations、clause_ids、block_ids、confidence、review_required",
         "风险条目与证据链，citations 受召回池约束"],
    ],
    "表 4-1  合同统一数据结构的主要字段",
    widths=[3.0, 7.4, 5.0],
)
P("审查环节采用七分类评估，取代传统的二元“违法/不违法”判断，类别与处理方式见表 4-2。"
  "这一设计的出发点是：一份合同完全合法，仍可能因权利义务失衡或商业安排不合理而不宜签署，"
  "若只用二元判断，这类问题会被系统性忽略。")
table(
    ["类别", "含义", "进入报告"],
    [
        ["legal_risk", "法律风险，条款可能违反强制性规定", "是"],
        ["commercial_unreasonable", "商业不合理，对价或安排明显偏离市场常规", "是"],
        ["rights_imbalance", "权利义务失衡，单方权利过大或义务过重", "是"],
        ["performance_risk", "履约风险，履行条件不具备或难以执行", "是"],
        ["insufficient_facts", "事实不足，现有条款无法支撑判断", "是，并触发证据回环"],
        ["need_more_material", "需要补充材料，缺少附件或前置文件", "是，并转人工"],
        ["no_substantive_risk", "未发现实质风险", "否，不进入最终报告"],
    ],
    "表 4-2  七分类评估及其处理方式",
    widths=[4.4, 8.2, 2.8],
)

# ------------------------------------------------------------------ 第5章
page_break()
H("第5章  系统实现", 1)
H("5.1  文档解析与条款切分的实现", 2)
P("解析器对 DOCX 采用直接读取 OOXML 的方式，不依赖第三方文档库，从而能够读取 Word 自动编号与标题样式。"
  "段落与表格按文档顺序展开为区块，表格按行拆分为区块并保留单元格边界；"
  "PDF 逐页读取文本块，记录每块的页码与包围盒坐标；TXT 按行切分并自动尝试 UTF-8 与 GB18030 解码。"
  "解析结果统一落到 SourceBlock 列表，后续所有环节只依赖该列表与页码信息。")
P("规则切分器先识别编号形式，再结合样式与缩进判定层级。对每一对相邻区块，"
  "切分器给出四类关系之一：续接表示同一段落被换行拆开；同一条款表示属于当前条款正文；"
  "新子条款表示进入下一层级；新条款表示开启新的条款。判定的置信度由编号匹配强度、"
  "样式差异与文本长度综合得到，低置信度边界被标记为待复核。")
P("模型切分器把区块列表连同编号提示交给大语言模型，要求返回条款的标签、标题、层级与区块编号。"
  "由于模型可能漏填或重复，系统在本地做三项兜底：未被任何条款引用的区块归入最近条款；"
  "重复或越界的编号被丢弃；超长合同按规则识别的条款起点分批提交，每批附带上一批尾部区块作为上下文。"
  "压力测试中强制分为 6 批，区块覆盖仍为 36/36，顺序一致，切分后拼接文本与原文完全相同。")
P("结构校验器以条款树为对象，检查区块遗漏与重复、父子层级异常、编号重复以及小数编号缺少父级。"
  "对已通过 parent_id 正确建树的情况不做误报。校验通过后输出条款树；未通过时，"
  "只把受影响的局部窗口交回模型，并按“可修正问题数严格减少”的规则决定是否采纳。")

H("5.2  要素抽取与结构信号的实现", 2)
P("规则通道按要素类型分别实现。当事人抽取先以“甲方/乙方/委托人/受托人/出租人/承租人”等角色词定位，"
  "再在句子内截断，避免把签署区的地址与联系方式误识别为名称；金额抽取识别货币前缀、大写与小写双写、"
  "零金额与负数，并按就近角色词判定金额角色，如合同总价、首付款、定金、押金、租金、违约金；"
  "日期抽取把多种写法归一化为 YYYY-MM-DD，并同样绑定角色；标的抽取以坐落地址与建筑面积为主要模式。")
P("模型通道在规则结果之上补充语义型标的与措辞变体，输出附原文片段与置信度。"
  "合并阶段以归一化值为键去重，例如 38500 与 38500.00 视为同一金额，保留置信度更高的记录。"
  "模型不可用时保留规则结果，不阻塞流程。")
P("要素节点还计算数学信号，包括金额非正数、金额极端偏大或偏小、履行期限倒置、关键要素缺失等。"
  "这些信号不生成风险，因为“数值异常”不等于“条款不合理”，例如零租金可能对应无偿借用。"
  "信号只有两种用途：当同一条款已有模型风险命中时，标记该风险得到交叉印证并提升置信度；"
  "当某信号未被任何风险覆盖时，以要素信号的形式进入人工复核队列，避免被静默忽略。")

H("5.3  法律知识库与混合检索的实现", 2)
P("法条库以 JSONL 存储，覆盖《中华人民共和国民法典》第 1 至 1260 条，每条包含法名、条号、正文、"
  "效力层级、生效时间、适用合同类型、风险标签、来源地址与版本号。其中 272 条由人工整理了风险标签，"
  "作为法条图谱的连接依据。索引构建脚本把法条写入本地向量库，同时生成稠密向量与稀疏词表，"
  "索引目录可随法条库更新而重建。")
P("稀疏通道在字级 unigram 与 bigram 上实现 BM25：文档侧计算词频饱和与逆文档频率，"
  "查询侧权重取 1，点积在向量库内复现 BM25 分值，因此无需额外部署检索服务。"
  "稠密通道由本地嵌入服务提供 1024 维向量。查询时两路各取若干候选，"
  "再由融合查询按 RRF 合并，取 top-k 作为法条池。")
P("法律适用智能体拿到法条池后，对每条候选风险做七分类评估，要求在给定池内选择最相关的条文并给出引用。"
  "系统对引用执行白名单校验：若 article_uid 不在召回结果中，该引用被清空，只保留定性结论。"
  "这一约束把“模型背法条”变成“模型选法条”，从机制上消除了自造条文的空间。")

H("5.4  多智能体统一审查的实现", 2)
P("统一图在编译时绑定检索器、模型传输对象与法条图谱。运行时按拓扑执行："
  "理解节点读取合同全文，输出类型、角色与义务摘要；要素节点改写合同对象以回填要素槽位；"
  "逐条审查与跨条款审查共享同一份合同，但提示词与输出模式不同；"
  "合并节点按风险码与条款重叠度去重；法律适用节点对每条候选检索并评估；"
  "裁判节点逐条裁决并给出整体结论；仲裁节点比较不同模块对同一风险的类别与严重程度；"
  "融合节点执行确定性规则，不做任何模型调用。")
P("三条硬护栏在代码层实现，模型无法绕过。第一，评估类别非“未发现实质风险”的候选永不被删除，"
  "即使裁判模型要求删除也保留；第二，裁判不可用时保守保留全部候选并转人工；"
  "第三，裁判未返回某条候选的裁决时，该条同样保留并转人工。"
  "三条护栏的共同效果是：系统的错误只会表现为“多报”，不会表现为“漏报”。")
P("护栏的核心判断位于裁判节点，其逻辑可简化为如下形式：")
code_block([
    "if decision is None:                       # 裁判未返回该项裁决",
    "    keep, review_required = True, True     # 保守保留并转人工",
    "elif not decision.get(\"keep\", True):",
    "    if assessment != \"no_substantive_risk\":  # 只要不是“无实质风险”",
    "        keep, review_required = True, True  # 删除请求被护栏否决",
    "        rationale += \"（护栏：不因未违法而删除。）\"",
])
P("这段逻辑的关键在于把“删除”权限限定在一个极窄的条件上：只有当评估结论明确为“未发现实质风险”时，"
  "候选才允许被丢弃。这样一来，模型即便因为提示词理解偏差而要求删除一条“商业不合理”的风险，"
  "系统也会保留该条并标记为需人工复核，理由字段同时记录护栏的介入痕迹，便于回溯。")
P("四条有界回环由状态中的计数器控制，每次触发前检查额度，触发后自增。"
  "检索重试只对没有评估结论的候选发起，并把查询放宽为风险码加合同类型；"
  "裁判重试仅在模型调用失败时发生；证据回环把“事实不足/需补充材料”的结论带回条款审查与跨条款审查，"
  "并要求在补充证据后重新执行法律适用，因此该轮会清空已有的法条适用结果，避免新旧结论混用；"
  "仲裁重裁在仲裁发现实质冲突时把意见写回裁判节点，随后重新仲裁一次。")
P("四条回环的额度与检索参数集中在同一份运行配置中，便于在实验中统一调整：")
code_block([
    "top_k: int = 6                     # 每轮召回法条数量",
    "tag_expansion_limit: int = 2       # 风险标签扩展的额外法条数",
    "max_legal_retries: int = 1         # ④→④ 检索重试",
    "max_judge_retries: int = 1         # ⑤→⑤ 裁判重试",
    "max_evidence_rounds: int = 1       # ⑤→②③ 证据回环",
    "max_arbitration_rounds: int = 1    # arbitrate→⑤ 仲裁重裁",
])
P("把额度集中配置的好处是：流程的终止性不依赖各节点的自觉，而由统一配置在编译期注入；"
  "同时，回环次数也便于在消融实验中作为自变量加以控制。")
P("融合阶段完全由确定性规则实现：按风险码与条款集合去重；对同时被条款审查与跨条款审查命中的风险标记交叉印证；"
  "按严重程度、置信度与条款位置排序；组装证据链，把条款、区块、原文片段与法条引用串成一条可点击的路径；"
  "最后把未决条目与要素信号写入复核队列。整体合理性评分由裁判节点给出，取值 0 至 100，越高表示越合理。")

H("5.5  审查报告与界面实现", 2)
P("报告以 m6 版本的统一结构输出，字段包括合同标识、理解结果、要素、要素信号、风险清单、冲突记录、"
  "整体结论、复核队列、统计信息与错误列表。统计信息中记录四条回环的实际触发次数，"
  "使审查过程本身可被审计。当模型服务不可用时，报告输出 unknown，并在错误列表与整体结论中明示"
  "“未执行实质审查，结果不可作为合同结论”，同时把所有疑点转入人工队列。")
P("界面以网页形式提供，包含合同上传、解析预览、条款树浏览、风险清单与报告导出五个区域。"
  "风险条目支持按严重程度与类别筛选，点击后跳转到对应条款并高亮原文片段；"
  "法条引用以《法名》第X条的形式展示，点击可查看条文全文。复核队列单独成页，供审查人员逐条处理。")
P("为了保证长合同在页面上的可用性，条款树采用按需展开的懒加载方式，只在用户展开某个章节时才渲染其子条款；"
  "风险清单则一次性载入全部条目，但把证据详情折叠，避免首屏出现大量原文。"
  "报告导出提供两种形态：结构化 JSON 供归档与二次分析，"
  "可读清单以“结论—依据—原文”三段式排版供打印归档。两种形态由同一份报告生成，不会出现口径不一致。")

# ------------------------------------------------------------------ 第6章
page_break()
H("第6章  系统测试与数据集构建", 1)
H("6.1  政府采购合同扫描件数据集的构建", 2)
P("为让系统面向真实文档而非仅面向排版规整的模板合同，本文首先尝试从政府开放数据平台与政府采购网站采集合同。"
  "天津开放数据平台的政府采购合同数据集提供网页预览，但接口调用需要授权令牌，公开页面仅展示少量记录，"
  "无法批量获取附件；天津市政府采购网的附件下载环节设置了图形验证码，不宜以脚本绕过；"
  "中国政府采购网对脚本请求返回 403，存在访问控制。")
P("最终选择北京市政府采购网作为数据来源：其合同公告栏目提供稳定的列表分页，详情页附件为可直接下载的 PDF，"
  "且多数 PDF 的每一页是一整张扫描图，正符合“图片形式合同”的采集目标。需要注意的是，该站点仅 HTTP 可访问，"
  "HTTPS 连接超时，脚本据此把协议相对链接补全为 HTTP。")
P("采集脚本按“列表页—详情页—附件”三级抓取，解析合同名称、项目编号、合同编号、采购人、供应商、合同金额、"
  "签订日期与公告日期，并按合同标识建立目录。对 PDF，脚本优先判断页面是否被一整张图覆盖，"
  "若覆盖面积超过页面面积的 60%，直接抽取嵌入原图，避免二次重采样；否则按 200 DPI 渲染页面。"
  "每页同时保存从 PDF 提取的文字层，用于后续计算 OCR 的字符错误率。")
P("采集结果见表 6-1，页数分布见图 6-1。共获得 63 份合同、1265 页，单份合同页数中位数为 16 页，"
  "最多 128 页，最少 2 页；其中 1147 页为整页扫描图且无有效文字层，82 页为整页扫描图但带文字层；"
  "另有 36 页页面本身不含整页图，按 200 DPI 渲染，其中 5 页无文字层、31 页带文字层。"
  "合计 1152 页（91.1%）无有效文字层，分布在 59 份合同之中。"
  "数据集总占用约 697 MB，因体量较大未纳入代码仓库，由脚本按需重新采集。")
table(
    ["统计项", "数值"],
    [
        ["合同份数", "63 份"],
        ["总页数", "1265 页"],
        ["单份页数（最少 / 中位 / 最多）", "2 / 16 / 128"],
        ["整页扫描图、无有效文字层", "1147 页"],
        ["整页扫描图、含文字层", "82 页"],
        ["无整页图、渲染且无文字层", "5 页"],
        ["无整页图、渲染且含文字层", "31 页"],
        ["无有效文字层页合计", "1152 页（91.1%）"],
        ["含无文字层页的合同", "59 份"],
        ["数据集占用空间", "约 697 MB"],
    ],
    "表 6-1  政府采购合同扫描件数据集统计",
    widths=[7.2, 7.6],
    align_cols=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER],
)
figure("fig6_1_dataset.png", 15.0, "图 6-1  数据集页数分布与页面构成")
P("图像规格方面，1265 页的平均尺寸为 2265×2765 像素，中位数为 2305×2338 像素，最小 1201×1619 像素，"
  "最大 4962×7094 像素。按 A4 纸面折算，多数页面接近 300 DPI，足以支撑中文小字号正文的识别；"
  "少数页面为整页大幅扫描，单页像素量约为平均值的四倍，需要在 OCR 前做等比例缩放，"
  "否则会显著增加识别耗时。文件以 JPEG 为主，渲染页保存为 PNG，"
  "前者有利于控制体积，后者避免文字边缘出现压缩伪影。")
P("该数据集的价值在于三点：一是文档形态真实，绝大多数页面为扫描图，可直接用于 OCR 与版面分析研究；"
  "二是带有弱标注，82 页扫描图自带文字层，可作为无人工标注条件下的对照；"
  "三是字段完整，公告页面提供了合同编号、金额与当事人，可用于校验要素抽取结果。"
  "后续工作将在此数据集上接入版面分析与文字识别，把页图像转为带坐标的文本，再送入本文的审查流程。")
H("6.2  测试环境与功能测试", 2)
P("测试环境见表 6-2。为避免外部模型服务的波动影响结果，功能测试以桩模型替代真实调用，"
  "桩模型按预置脚本返回结构化输出，因而同一用例可反复执行且结果一致。"
  "真实模型仅在人工体验环节使用，不参与回归测试。")
table(
    ["项目", "配置"],
    [
        ["操作系统", "Windows 11"],
        ["文档解析", "PyMuPDF"],
        ["流程编排", "LangGraph"],
        ["向量检索", "Qdrant 本地模式"],
        ["嵌入模型", "bge-m3（1024 维）"],
        ["法条库", "《民法典》1260 条，其中 272 条带风险标签"],
        ["测试方式", "离线桩模型，123 个用例可重复执行"],
    ],
    "表 6-2  测试环境",
    widths=[4.6, 10.0],
    align_cols=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
)
P("功能测试针对系统各模块的核心行为设计用例，全部用例可在无外部模型服务的条件下运行，"
  "以固定输入与桩模型替代真实调用，保证结果可重复。测试结果见表 6-3。")
table(
    ["测试项", "测试内容", "结果"],
    [
        ["文档解析", "DOCX、PDF、TXT 三类文件解析，页码与坐标保留，超限文件被拒绝", "通过"],
        ["条款切分", "规则与模型两路切分，区块覆盖完整，切分文本与原文一致", "通过"],
        ["结构校验", "区块遗漏、重复引用、层级断裂、重复编号、小数编号缺父级", "通过"],
        ["切分回流", "仅当可修正问题数严格减少时采纳，否则保留上一轮", "通过"],
        ["要素抽取", "四类要素抽取、同值合并、模型不可用时保留规则结果", "通过"],
        ["结构图谱", "条款父子边、要素定位边、悬空引用与金额角色缺失检测", "通过"],
        ["法条检索", "双路召回、RRF 融合、引用白名单校验", "通过"],
        ["统一审查", "节点串联、并行分支、候选合并、裁判与仲裁、四条回环计数", "通过"],
        ["降级处理", "模型不可用时输出 unknown 并明示未执行实质审查", "通过"],
        ["报告输出", "字段完整性、证据链定位、复核队列生成", "通过"],
    ],
    "表 6-3  功能测试结果",
    widths=[3.0, 10.6, 2.4],
    align_cols=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER],
)
H("6.3  处理链路实测", 2)
P("在功能测试之外，本文对处理链路的关键环节做了实测，结果见表 6-4。测试使用仓库自带的样例合同，"
  "不依赖外部模型即可完成解析、切分、结构校验与图谱构建；需要模型参与的环节以桩模型替代，"
  "仅验证流程连通性与字段传递，不用于评价模型效果。")
table(
    ["实测项目", "结果"],
    [
        ["单元与集成用例", "123 个，全部通过（全离线）"],
        ["样例合同区块与条款", "36 个原子区块切分为 23 条条款，覆盖 36/36"],
        ["模型分批切分压力测试", "强制 6 批，区块覆盖 36/36，顺序一致，拼接文本与原文相同"],
        ["样例合同结构图谱", "25 个节点、19 条边、检出 2 处悬空条款引用"],
        ["法条索引", "1260 条法条，其中 272 条带风险标签"],
        ["采集脚本默认参数", "列表 3 页、200 DPI、请求间隔 1.2 秒"],
    ],
    "表 6-4  处理链路实测结果",
    widths=[5.6, 10.4],
    align_cols=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
)
P("需要说明的是，本文未对要素抽取与法条检索的精确率、召回率做独立复现。仓库中曾记录的一组抽取与检索指标"
  "依赖合成合同数据集，该数据集体积较大且未纳入版本库，在当前环境下无法完整复现，"
  "因此不作为本文的实验结论。相应的评测脚本与金标准构造方式已经实现，"
  "后续可在本文构建的真实扫描件数据集完成 OCR 之后，按同一协议重新评测。")
H("6.4  测试总结", 2)
P("本章完成了两项工作：一是构建了面向扫描件的政府采购合同数据集，验证了从公告列表到页级图像的完整采集路径，"
  "并量化了文字层缺失的比例；二是对系统各模块进行了功能测试与链路实测，"
  "结果表明解析、切分、校验、图谱与编排各环节均按设计工作，且在模型缺失时能够保守降级。"
  "测试也暴露出当前阶段的边界：审查质量依赖模型能力，尚未在真实扫描件上做端到端验证，"
  "这构成后续工作的主要方向。")

# ------------------------------------------------------------------ 结论
page_break()
H("结  论", 1)
P("本文围绕政府采购合同的自动审查，设计并实现了一套以多智能体协同与知识图谱为核心的系统。"
  "系统把合同解析为可回溯的条款树，以规则与模型双引擎完成切分并用结构校验与局部回流保证切分质量；"
  "以规则与模型双通道抽取当事人、金额、日期与标的，并把数学信号限定为佐证与复核线索；"
  "构建覆盖《民法典》1260 条的法条库，以稠密与稀疏混合检索并经 RRF 融合，"
  "通过引用白名单把模型输出约束在可验证的范围内；"
  "以 LangGraph 把理解、逐条审查、跨条款审查、法律适用、裁判与仲裁组织为统一状态图，"
  "用七分类评估表达“不违法但不合理”的中间地带，用三条硬护栏保证不漏报，用四条有界回环保证流程可终止。")
P("在数据侧，本文实现了政府采购合同扫描件的采集程序，获得 63 份合同、1265 页页级图像，"
  "其中 91.1% 的页面没有有效文字层，为后续 OCR 与版面分析提供了真实语料；"
  "系统现有 123 个用例全部通过，样例合同的切分、图谱与报告链路均可复现。")
P("从方法上看，本文的贡献不在于引入某个新模型，而在于把既有能力组织成一条可验证的链路："
  "切分环节用结构校验与局部回流保证“读得对”，抽取环节用双通道与原文定位保证“抓得准”，"
  "检索环节用混合召回与引用白名单保证“引得住”，编排环节用硬护栏与有界回环保证“判得稳”。"
  "这四个环节各自都有明确的失败模式与对应的兜底策略，系统因此可以在模型能力波动时保持行为可预期。")
P("系统仍存在不足。其一，OCR 与版面分析尚未接入，扫描件目前只能停在图像与页码层面，"
  "端到端审查仍以文本型合同为主。其二，真实合同的评测规模有限，要素抽取与法条检索的指标"
  "尚未在本文数据集上完成复现。其三，法条库目前仅覆盖《民法典》，"
  "尚未纳入行政法规、部门规章与政府采购专门规范。其四，系统以单机服务形式运行，"
  "尚未做并发与多租户方面的工程优化。")
P("后续工作将按三步推进：先接入版面分析与文字识别，把页图像转为带坐标的文本并完成端到端评测；"
  "再扩充法条与规范性文件，完善风险标签体系与检索评测；"
  "最后在真实业务场景中引入审查人员反馈，形成“机器初筛—人工复核—反馈回流”的闭环，"
  "持续提升系统的实用价值。")

# ------------------------------------------------------------------ 参考文献
page_break()
H("参考文献", 1)
refs = [
    "刘贵祥. 关于合同成立的几个问题[J]. 法律适用, 2022(4): 3-17.",
    "周红, 等. 建设工程合同智能分类方法研究[J]. 科技管理研究, 2023(8): 165-172.",
    "蔡颢婷, 等. 基于大语言模型的医院智能合同审查系统构建[J]. 中国卫生信息管理杂志, 2025, 22(3): 450-455+470.",
    "范雨晴, 等. CTML: 基于对比学习的合同文本表示模型[J]. 工程科学学报, 2024, 46(1): 109-119. "
    "DOI: 10.13374/j.issn2095-9389.2023.01.13.003.",
    "HENDRYCKS D, BURNS C, BASART S, et al. CUAD: an expert-annotated NLP dataset for legal contract review[C]//"
    "Advances in Neural Information Processing Systems 34 (NeurIPS 2021), Datasets and Benchmarks Track. 2021. "
    "arXiv: 2103.06268.",
    "KOREEDA Y, MANNING C D. ContractNLI: a dataset for document-level natural language inference for contracts[C]//"
    "Findings of the Association for Computational Linguistics: EMNLP 2021. 2021: 1907-1919. "
    "DOI: 10.18653/v1/2021.findings-emnlp.164.",
    "TUGGENER D, POLUTTA A, SVENSSON M, et al. LEDGAR: a large-scale multi-label corpus for text classification of "
    "legal provisions in contracts[C]//Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020). "
    "2020: 1235-1241.",
    "ZHENG C, ZHOU X, LU Z, et al. Automating construction contract review using knowledge graph-enhanced large "
    "language models[EB/OL]. arXiv: 2309.12132, 2023.",
    "LIU S, ZHENG Y, WANG Y, et al. ContractEval: benchmarking large language models on contract review[EB/OL]. "
    "arXiv: 2508.03080, 2025.",
    "RAPTOPOULOS P, KARAMITSOS I, et al. PAKTON: a multi-agent framework for question-answering on long contracts[C]//"
    "Proceedings of EMNLP 2025. 2025: 7948-7984. DOI: 10.18653/v1/2025.emnlp-main.403.",
    "RAGulating compliance: a multi-agent knowledge graph framework for regulatory compliance[EB/OL]. "
    "arXiv: 2508.09893, 2025.",
    "GRAPH-GRPO-LEX: graph-based reinforcement learning for legal reasoning[EB/OL]. arXiv: 2511.06618, 2025.",
    "YUN S, et al. Graph-of-Agents: a graph-based multi-agent framework for long-context understanding[C]//"
    "International Conference on Learning Representations (ICLR 2026). arXiv: 2604.17148.",
    "LI H, et al. LegalOne: empowering large language models for legal tasks with multi-agent collaboration[C]//"
    "Findings of EMNLP 2026. 2026: 19566-19588. DOI: 10.18653/v1/2026.findings-emnlp.927. arXiv: 2602.00642.",
    "FEI Z, et al. LawBench: benchmarking large language models' legal knowledge, reasoning and application[C]//"
    "Proceedings of EMNLP 2024. 2024: 7933-7962. DOI: 10.18653/v1/2024.emnlp-main.452.",
    "SHEN X, et al. ProvBench: benchmarking provenance-grounded legal reasoning[C]//Proceedings of ACL 2025. "
    "2025: 6240-6254. DOI: 10.18653/v1/2025.acl-long.312.",
    "Qwen Team. Qwen3 technical report[EB/OL]. arXiv: 2505.09388, 2025.",
    "LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks[C]//"
    "Advances in Neural Information Processing Systems 33 (NeurIPS 2020). 2020: 9459-9474.",
    "CHEN J, XIAO S, ZHANG P, et al. BGE M3-Embedding: multi-lingual, multi-functionality, multi-granularity text "
    "embeddings through self-knowledge distillation[EB/OL]. arXiv: 2402.03216, 2024.",
    "ROBERTSON S, ZARAGOZA H. The probabilistic relevance framework: BM25 and beyond[J]. Foundations and Trends in "
    "Information Retrieval, 2009, 3(4): 333-389. DOI: 10.1561/1500000019.",
    "CORMACK G V, CLARKE C L A, BUETTCHER S. Reciprocal rank fusion outperforms Condorcet and individual rank learning "
    "methods[C]//Proceedings of the 32nd International ACM SIGIR Conference. 2009: 758-759. "
    "DOI: 10.1145/1571941.1572114.",
    "HUANG Y, LYU T, XU X, et al. LayoutLMv3: pre-training for document AI with unified text and image masking[C]//"
    "Proceedings of the 30th ACM International Conference on Multimedia. 2022: 4083-4091. "
    "DOI: 10.1145/3503161.3548112.",
    "DU Y, LI C, GUO R, et al. PP-OCR: a practical ultra lightweight OCR system[EB/OL]. arXiv: 2009.09941, 2020.",
]
for i, ref in enumerate(refs, start=1):
    P(f"[{i}]  {ref}", indent=False, size=10.5, space_after=4)

page_break()
# ------------------------------------------------------------------ 致谢
H("致  谢", 1)
P("论文完成之际，首先要感谢我的指导老师。从选题的确定、技术路线的论证到论文结构的调整，"
  "老师都给予了耐心细致的指导，并在系统遇到瓶颈时提供了关键的建议。老师严谨的治学态度与务实的工程风格，"
  "让我认识到一个系统能否落地，往往取决于对边界条件的敬畏。")
P("感谢在数据采集阶段给予帮助的同学们。政府网站的可访问性、附件形态与验证码机制差异很大，"
  "多次尝试才能确定可用的数据源，这段经历让我对真实数据的复杂性有了更具体的理解。"
  "感谢实验室提供的计算资源，使本地模型推理与索引构建得以完成。")
P("感谢家人一直以来的支持与理解，让我能够安心完成学业。最后，向在百忙之中评阅本文的各位老师表示诚挚的谢意。")

# ------------------------------------------------------------------ 附录
page_break()
H("附  录", 1)
P("附录 A  项目目录结构", indent=False, bold=True, space_before=6)
P("contract_review/        解析、切分、要素、图谱、检索、智能体与编排的核心代码", indent=False, size=10.5)
P("  agents/               统一图节点：understand、elements、clause_review、cross_review、", indent=False, size=10.5)
P("                        merge、legal_apply、judge、arbitrate、fuse", indent=False, size=10.5)
P("  models.py             合同统一数据结构与序列化", indent=False, size=10.5)
P("  parser.py             三类文件解析", indent=False, size=10.5)
P("  splitter.py           规则与模型切分、结构回流", indent=False, size=10.5)
P("  validation.py         结构一致性校验", indent=False, size=10.5)
P("  extractor.py          要素抽取双通道", indent=False, size=10.5)
P("  kg.py                 合同结构图与法条图谱", indent=False, size=10.5)
P("  legal_kb.py           法条库与混合检索", indent=False, size=10.5)
P("scripts/                数据采集与索引构建脚本", indent=False, size=10.5)
P("  crawl_contracts.py    政府采购合同扫描件采集", indent=False, size=10.5)
P("  build_legal_index.py  法条索引重建", indent=False, size=10.5)
P("data/                   法条库、索引与采集数据（体量较大的目录不入库）", indent=False, size=10.5)
P("checks/                 单元与集成测试", indent=False, size=10.5)
P("附录 B  采集与索引构建命令", indent=False, bold=True, space_before=10)
P("python scripts/crawl_contracts.py --pages 3 --limit 20", indent=False, size=10.5)
P("python scripts/build_legal_index.py", indent=False, size=10.5)
P()
P("（采集数据默认写入 data/gov_contracts/，按合同标识分目录存放页图像、文字层与元数据；"
  "索引默认写入 data/legal_index/。）", indent=False, size=10.5)

doc.save(OUT)
print("已生成", OUT)
