"""生成《实测数据说明》——不是改申报书，是给答辩用的补充材料。

为什么不做成"修订稿"：作品名、GoA 派发、LegalOne-8B、F1 90%、PROVBENCH、LawBench
这六处是申报书的核心表述，改掉等于改掉立意。正确做法是正文一字不动，
另附一份实测数据与口径说明，把"承诺 / 实测 / 差异原因"摆在一起，主动交代。
"""

from __future__ import annotations

import pathlib

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "申报书-实测数据说明.docx"

CN_FONT = "宋体"
CN_HEI = "黑体"


def set_font(run, name=CN_FONT, size=10.5, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def heading(doc, text, size=13):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run(text), CN_HEI, size, True)
    return p


def body(doc, text, size=10.5, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.4
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    set_font(p.add_run(text), CN_FONT, size)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.35
    set_font(p.add_run("· " + text), CN_FONT, size)
    return p


def three_line_table(doc, header, rows, widths=None, size=9):
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, text in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(text), CN_HEI, size, True)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == len(row) - 1 else WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(str(text)), CN_FONT, size)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # 三线表：只留顶线、表头下线、底线
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            borders = tcPr.find(qn("w:tcBorders"))
            if borders is None:
                from docx.oxml import OxmlElement

                borders = OxmlElement("w:tcBorders")
                tcPr.append(borders)
            for edge in ("top", "left", "bottom", "right"):
                from docx.oxml import OxmlElement

                el = borders.find(qn(f"w:{edge}"))
                if el is None:
                    el = OxmlElement(f"w:{edge}")
                    borders.append(el)
                if edge == "top" and row_index == 0:
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "12")
                elif edge == "bottom" and row_index == 0:
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "6")
                elif edge == "bottom" and row_index == len(table.rows) - 1:
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "12")
                else:
                    el.set(qn("w:val"), "nil")
    return table


def main() -> int:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.6)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    set_font(title.add_run("智律契灵 · 实测数据与口径说明"), CN_HEI, 16, True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(14)
    set_font(sub.add_run("附件：与《作品申报书》配套，正文承诺不变，逐条给出实测结果"), CN_FONT, 9)

    body(
        doc,
        "本说明不修改申报书正文的任何表述，仅就申报书中列出的技术指标与关键技术，"
        "逐条给出当前原型的实测结果、测量方法与差异原因。凡未达到承诺数值的项，"
        "如实标注差距及其来源，不以外推值代替实测值。",
    )

    heading(doc, "一、主要技术指标的三条承诺")
    three_line_table(
        doc,
        ["申报书承诺", "实测结果", "测量口径"],
        [
            ("关键实体识别 F1 ≥ 90%", "严格 87.5% / 补全 98.4%", "留出集 28 份、174 个实体；两套金标定义并列报告"),
            ("单份端到端 ≤ 3 分钟", "P50 2.97 s，最慢 8.93 s", "13 份真实政府采购合同（含 10 份扫描件 OCR）"),
            ("十万份级存储/检索/调度", "100000 份 / 80 万风险行 / 301.9 MB", "写入 459.5 份每秒；分页 5000 页 80 ms；中文检索 19 ms"),
        ],
        widths=[5.0, 4.4, 6.4],
    )
    body(
        doc,
        "第一项未达 90%，差距全部集中在金额字段：严格口径精确率 0.4789，"
        "误报多为正文真实存在、但金标清单只收录“合同总额”一项而未标注的分项报价、单价与押金。"
        "将同类金额回填进金标后（补全口径），精确率升至 0.9817、F1 达 0.9839。"
        "两套口径同时报告，不做单口径挑选。",
    )

    heading(doc, "二、五个功能节点的实现对应")
    three_line_table(
        doc,
        ["节点", "申报书表述", "实现"],
        [
            ("① 要素抽取", "NER 抽取当事人/金额/日期/标的", "BERT-CRF（9 类 BIO）+ 规则双通道，要素经图边同时送入③④"),
            ("② 逻辑错误检测", "调用 LegalOne-8B 遍历推理", "通道已通（OpenAI 兼容 + 引文校验 + 只读评分）；本机跑 1.7B，8B 需云端"),
            ("③ 主体资格审查", "法律知识图谱 + RAG", "15 条资质法规图谱（45 节点 30 边）+ BM25 检索 + 五类核验"),
            ("④ 法条匹配", "BERT、Mamba 编码 + 余弦 + 多头注意力", "双塔 + Mamba 分支 + 8 头交叉注意力，输出“一致/冲突”"),
            ("⑤ 综合评估", "LawBench 评估 + 人工复核", "按 LawBench 3 大类 20 子任务口径组织评测；复核队列落库"),
        ],
        widths=[3.2, 5.0, 7.6],
    )

    heading(doc, "三、关键技术的实测支撑")
    bullet(doc, "检索：三路召回（BM25 / LSA / 标签图 PPR）+ 加权 RRF + LTR 重排，强金标准 Recall@5 0.6944、MRR@5 1.0000。")
    bullet(doc, "风险检出：27 个风险码 / 7 类，合成对抗集 32 个注入缺陷召回 1.000，合同级命中 10/10。")
    bullet(doc, "条款—条文一致性：BERT 双塔 + 8 头交叉注意力，macro-F1 0.5771、准确率 0.9231；规则残差分支保证不劣于规则基线。")
    bullet(doc, "主体资格：资质知识图谱含 2 条已废止资质（物业服务企业资质、计算机信息系统集成资质），按状态抑制误报。")
    bullet(doc, "存储与调度：SQLite（WAL）+ FTS5 trigram 中文检索，支持断点续跑，10 万份压测通过。")

    heading(doc, "四、LawBench / PROVBENCH 的引用口径")
    body(
        doc,
        "LawBench 的官方划分与标注未公开发布，PROVBENCH 亦然。本系统按其任务类型定义"
        "（extraction / single-label classification / multi-label classification / generation）"
        "组织自建语料上的评测，报告中的数值是“按该口径的自家评测结果”，"
        "并非官方榜单分数，亦未使用其训练划分，特此说明。",
    )

    heading(doc, "五、大模型分支的部署边界")
    body(
        doc,
        "申报书所列 LegalOne-8B 权重为 16.42 GB（4 分片），本机显卡为 RTX 5070 Laptop（显存 8151 MiB），"
        "无法在本地以 fp16 加载；已实测 1.7B 版本可在本机运行（单条款生成 5.7 s），"
        "并已预留云端接入路径（vLLM 起 OpenAI 兼容服务，改环境变量即可切换）。"
        "模型分支的三条护栏始终生效：引文必须能在条款原文中定位、风险码必须在码表内、评分只读不改。",
    )

    heading(doc, "六、可复现命令")
    for cmd in [
        "python -m zhilv_qiling_v2.api --port 8611          # Web 服务",
        "python tools\\_web_e2e.py http://127.0.0.1:8611     # 七项端到端自检",
        "python -m zhilv_qiling_v2.lawbench                  # 六任务评测表",
        "python tools\\_store_scale.py --n 100000 --risks 8   # 十万份压测",
        "python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade ner  # NER 重训与评测",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(cmd), "Consolas", 9)

    doc.save(str(OUT))
    print("written:", OUT, OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
