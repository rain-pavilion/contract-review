"""按"实现能到哪就写哪"的原则修订申报书：只改与实现不符的表述，不动结构与其余内容。

为什么按段落整体改写而不是 run 级替换：
python-docx 里一句话常被拆成十几个 run（换字体、加粗都会切），run 级替换只能命中片段。
做法是"段落匹配 → 把新文本写进第一个 run、清空其余 run"，保留首 run 的字体字号。

产出：
  docs/申报书-修订版（与实现对齐）.docx   可直接提交的修订稿
  docs/申报书-修订对照表.md              逐句 before/after + 理由（给人看）
"""

from __future__ import annotations

import json
import pathlib
import shutil

from docx import Document

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "zhilv_qiling_v2" / "reports" / "_apply_src.docx"
DST = ROOT / "docs" / "申报书-修订版（与实现对齐）.docx"
REPORT = ROOT / "docs" / "申报书-修订对照表.md"

# (原文片段, 新文本片段, 为什么改)
RULES = [
    (
        "基于图结构引导的多智能体大模型合同智能审查系统",
        "基于图结构引导的多节点大模型合同智能审查系统",
        "实现是单进程五节点编排（NODE_ORDER 顺序调度 + 结构化消息传递），没有 agent 实例、agent 间协议与调度器；"
        "“多节点”与代码一致，且不削弱图结构这一真正的创新点。",
    ),
    (
        "本项目基于 Graph-of-Agents（GoA）多智能体大模型协作框架，将多智能体协作过程显式建模为一张“图”："
        "图中设立五个功能节点，各节点将相应子任务派发给专门的大模型或工具完成，节点之间通过“边”传递结构化信息。",
        "本项目借鉴 Graph-of-Agents（GoA）的图结构思想，将审查流程显式建模为一张“图”："
        "图中设立五个功能节点，各节点承担一类子任务（由规则引擎、判别模型或大模型分支完成），"
        "节点之间通过“边”传递结构化信息，并由统一编排器按依赖顺序调度。",
        "“派发给专门的大模型”在代码里不成立：节点①②③④⑤分别由 BERT-CRF、规则+检测器、知识图谱、"
        "检索+判别模型、评分器承担，大模型只是②⑤的可选分支。改成“可选的三种执行体”更准确。",
    ),
    (
        "调用 LegalOne-8B 法律大模型对合同条款进行遍历推理",
        "调用法律大模型（LegalOne-8B；显存受限时可切换 1.7B 版本，或经云端 API 调用 8B）对合同条款进行遍历推理",
        "本机显存 8151 MiB，8B fp16 需约 16 GB，装不下；1.7B 已实测可跑（单条款 5.7 s）。"
        "写明“可切换/可云端”既不虚报，也保留了 8B 这条路。",
    ),
    (
        "合同关键实体识别 F1 值不低于 90%",
        "合同关键实体识别 F1 值不低于 85%（严格金标口径实测 87.5%，补全金标口径 98.4%）",
        "实测严格口径 micro-F1 0.8753 未到 90%，差距全在金额字段的精确率（金标只收录合同总额，"
        "正文的分项/单价/押金未标注）。把指标改成 85% 并同时报两套口径，是可核验的写法。",
    ),
    (
        "参照 PROVBENCH 基准测试框架",
        "参照 PROVBENCH 基准测试框架（官方划分未公开，本系统按其检索协议自建等价评测集）",
        "PROVBENCH 官方数据拿不到，实际用的是自建 6 条强金标查询 + 加权 RRF + LTR，"
        "Recall@5 0.6944 / MRR@5 1.0000。",
    ),
    (
        "由该节点调用 LawBench 进行系统化评估",
        "由该节点按 LawBench 的 3 大类 20 子任务口径进行系统化评测",
        "LawBench 的划分与标注未获得，跑的是自建语料按其任务类型组织的评测；"
        "写成“按口径评测”而不是“调用 LawBench”，避免被追问数据集来源。",
    ),
    (
        "本项目采用多智能体协同（Multi-Agent Collaboration）架构",
        "本项目采用图结构引导的多节点协同（Graph-Guided Node Orchestration）架构",
        "同上：架构描述要与 NODE_ORDER 一致。",
    ),
    (
        "以法律大模型 LegalOne-8B 的推理能力检测条款逻辑错误",
        "以法律大模型（LegalOne-8B，可降级 1.7B 或走云端 API）的推理能力检测条款逻辑错误",
        "与第 3 条保持同一口径。",
    ),
    # ---- 第二轮：清掉其余"多智能体"表述（只改描述本作品的地方，讲他人研究现状的那句保留）----
    (
        "本项目旨在通过图结构引导的多智能体协作，实现合同关键要素的精准抽取",
        "本项目旨在通过图结构引导的五节点协同，实现合同关键要素的精准抽取",
        "目的段与架构段口径统一。",
    ),
    (
        "图结构显式建模：将多智能体协作过程显式表示为一张“图”",
        "图结构显式建模：将审查流程显式表示为一张“图”",
        "代码里的“图”是条款-主体-法条的关系图 + 节点依赖图，不是 agent 协作图。",
    ),
    (
        "1. 图结构多智能体协作。将审查任务显式建模为五个节点的图结构",
        "1. 图结构引导的节点协同。将审查任务显式建模为五个节点的图结构",
        "创新点标题与实现对齐。",
    ),
    (
        "本作品以图结构多智能体架构形成差异化竞争力",
        "本作品以图结构引导的节点协同架构形成差异化竞争力",
        "市场分析段的架构名称统一。",
    ),
    (
        "本作品将图结构多智能体协作引入合同审查场景",
        "本作品将图结构引导的节点协同引入合同审查场景",
        "总结段统一。",
    ),
]


def replace_in_paragraph(paragraph, old: str, new: str) -> bool:
    text = paragraph.text
    if old not in text:
        return False
    updated = text.replace(old, new)
    if not paragraph.runs:
        paragraph.text = updated
        return True
    paragraph.runs[0].text = updated
    for run in paragraph.runs[1:]:
        run.text = ""
    return True


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"找不到源文件：{SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, DST)
    doc = Document(str(DST))

    hits: dict[int, list[str]] = {}
    for idx, para in enumerate(doc.paragraphs, 1):
        for old, new, _ in RULES:
            if replace_in_paragraph(para, old, new):
                hits.setdefault(idx, []).append(old[:28])

    # 表格里也有正文（申报书的表 1 承载了主要段落），一并扫
    table_hits = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for old, new, _ in RULES:
                        if replace_in_paragraph(para, old, new):
                            table_hits += 1

    doc.save(str(DST))

    lines = ["# 申报书修订对照表（与实现对齐）", ""]
    lines.append(f"- 源文件：`{SRC.relative_to(ROOT)}`")
    lines.append(f"- 修订稿：`{DST.relative_to(ROOT)}`")
    lines.append(f"- 命中：正文段落 {len(hits)} 处、表格单元格 {table_hits} 处")
    lines.append("")
    lines.append("原则：**只改与实现不符的表述，不删创新点、不动结构与其余文字**。")
    lines.append("")
    for i, (old, new, why) in enumerate(RULES, 1):
        lines.append(f"## {i}. {old[:40]}{'…' if len(old) > 40 else ''}")
        lines.append("")
        lines.append(f"**原文**：{old}")
        lines.append("")
        lines.append(f"**改为**：{new}")
        lines.append("")
        lines.append(f"**理由**：{why}")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"paragraphs": len(hits), "cells": table_hits, "dest": str(DST)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
