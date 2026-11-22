# -*- coding: utf-8 -*-
"""申报书修订 v2：保留“多智能体”真实实现，只改与实测不符的数据/基准/实现细节。"""
from __future__ import annotations
import hashlib, json, pathlib, shutil
from docx import Document

ROOT = pathlib.Path(r"C:\Users\ROG\OneDrive\Desktop\LLM")
SRC = pathlib.Path(r"C:\Users\ROG\OneDrive\Desktop\学习\计252马宇轩2542110170挑战杯材料\马宇轩 申报书.docx")
DST = ROOT / "docs" / "申报书-修订版（与实测对齐）_20261121.docx"
REPORT = ROOT / "docs" / "申报书-修订对照表_20261121.md"

RULES = [
    (
        "最后，回环验证结果汇入第五节点，由该节点调用 LawBench 进行系统化评估，形成最终审查结论并交由人工纠错复核。系统主要技术指标为：合同关键实体识别 F1 值不低于 90%；单份常规合同端到端审查耗时不超过 3 分钟；架构支持十万份级合同的存储、检索与批量审查调度。",
        "最后，回环验证结果汇入第五节点，由该节点按 LawBench 任务口径组织自建评测，形成最终审查结论并交人工复核。主要技术指标为：关键实体识别在统一标注集上报告严格口径 F1（目标 85%，需独立标注验证）；规则通道单份合同审查实测 P50 2.97 s（≤3 分钟），大模型档按算力分档；支持十万份级合成档案存储调度，真实业务待测。",
        "90%改为85%目标并注明需独立标注；3分钟限定为规则通道并给实测；十万份限定为合成档案存储/调度。",
    ),
    (
        "调用 LegalOne-8B 法律大模型对合同条款进行遍历推理",
        "调用法律大模型（LegalOne-8B，4-bit 量化；显存不足时可切换 1.7B 版本或云端 API）对候选条款进行推理",
        "本机8GB显存无法fp16加载8B；实际为4-bit量化、两段式候选复核，不是逐条遍历。",
    ),
    (
        "集成法律知识图谱与检索增强生成（RAG）技术，对合同主体的资格、权限与信用等事项进行核验",
        "集成轻量资质知识图谱（15 条法规、45 个节点、30 条边）与 BM25 检索，对合同主体的资格、权限与信用等事项进行核验；外部事实以人工导入的权威证据为准，缺证保持待核验",
        "实际为轻量图谱+BM25，无生成式RAG；不直连工商/信用数据库，外部事实需人工证据。",
    ),
    (
        "参照 PROVBENCH 基准测试框架，采用 BERT、Mamba 等编码器将合同条款与法律条文映射为向量表示，计算条款与候选法条之间的余弦相似度，并利用多头注意力机制（Multi-head Attention）深入分析二者之间的逻辑关系，输出“一致”或“冲突”标签",
        "参照 PROVBENCH 的检索与冲突检测协议（官方数据与评分器不可得，采用自建等价评测集），采用 BERT 双塔、Mamba 选择性状态空间分支与 8 头交叉注意力进行条款—法条关系判别，计算余弦相似度并输出“一致”“冲突”或“中性”标签",
        "PROVBENCH官方数据/评分器不可得；实际为自建等价评测；判别器输出三分类且为弱监督训练。",
    ),
    (
        "以法律大模型 LegalOne-8B 的推理能力检测条款逻辑错误；以知识图谱与检索增强生成（RAG）核验主体资格，增强事实判断的准确性；以向量相似度计算与多头注意力机制实现条款与法律条文之间的深度语义匹配和逻辑关系分析。",
        "以法律大模型 LegalOne-8B（4-bit 量化）的推理能力检测条款逻辑错误；以轻量知识图谱与 BM25 检索核验主体资格，增强事实判断的准确性；以向量相似度计算与 8 头交叉注意力实现条款与法律条文之间的语义匹配和逻辑关系分析。",
        "同步4-bit量化、轻量图谱/BM25、8头交叉注意力的实际实现。",
    ),
    (
        "标准化评估：引入 PROVBENCH 基准测试与 LawBench 系统化评估，使审查效果可量化、可验证。",
        "标准化评估：参照 PROVBENCH 与 LawBench 的任务口径组织自建评测，使审查结果可复核；官方全量基准成绩待补齐。",
        "未跑官方全量LawBench/ProvBench，只做协议核验与自建等价评测。",
    ),
    (
        "集成法律知识图谱与 LegalOne-8B 专业法律大模型",
        "集成轻量法律知识图谱与 LegalOne-8B（4-bit 量化）法律大模型",
        "轻量图谱+4-bit量化，避免暗示全量知识图谱/fp16 8B。",
    ),
    (
        "部署需配备 GPU 算力服务器，可调用大模型 API 或本地部署 LegalOne-8B 等模型。",
        "推荐使用 GPU 算力服务器；也可调用大模型 API、云端部署，或在 8 GB 显存级设备/CPU 上以 4-bit 量化运行 LegalOne-8B，时延相应增加。",
        "本机已实际在8GB显存/CPU上4-bit运行，不是必须GPU服务器。",
    ),
    (
        "融合法律知识图谱、RAG 检索增强与法律专用大模型",
        "融合轻量法律知识图谱、BM25 检索与法律专用大模型",
        "无生成式RAG；实际为轻量图谱+BM25。",
    ),
    (
        "4. 结果可量化。采用 PROVBENCH 与 LawBench 基准评测，审查效果可验证。",
        "4. 结果可复核。参照 PROVBENCH 与 LawBench 的任务口径组织自建评测，并保留真实运行、失败与边界记录。",
        "未取得官方全量成绩；改为可复核的自建评测+失败记录。",
    ),
    (
        "本作品将图结构多智能体协作引入合同审查场景，以五个功能节点分别承担要素抽取、逻辑检测、主体审查、法条匹配与综合评估，融合知识图谱、RAG 与回环验证机制，并以 ProvBench、LawBench 作为量化评估手段。",
        "本作品将图结构多智能体协作引入合同审查场景，以五个功能节点分别承担要素抽取、逻辑检测、主体审查、法条匹配与综合评估，融合轻量知识图谱、BM25 检索与回环验证机制，并参照 ProvBench、LawBench 的任务口径组织自建评测。",
        "保留多智能体（真实实现），仅同步轻量图谱/BM25与自建评测口径。",
    ),
    (
        "3. 回环验证。各节点输出互相校验并辅以人工纠错，有效降低幻觉风险。",
        "3. 回环验证。各节点输出在有界轮次内互相校验；未解决事项保留人工核验，不自动改判。",
        "回环是有界协议，pending不自动改判；避免“有效降低”的绝对化表述。",
    ),
]

MULTI_AGENT_NOTE = """## 关于“多智能体”的保留说明

本次修订**保留**申报书中的“多智能体”表述，因为代码中有真实的多智能体层，且 2026-11-19 的真实归档运行实际使用了它：

- zhilv_qiling_v2/agents_multi.py：Agent / Message / MessageBus / TOPOLOGY；agents_multi_run.py：intake、orchestrator、ner、logic、reviewer、subject、statute、judge 等 Agent，按拓扑投递，超边拒收。
- start-core-web.cmd 设置 ZLQL_MULTI_AGENT=1；tools/core_delivery_20261119/latest_task.json 的任务 options.engine 为 multi。
- 该次归档报告 stats.multi_agent.enabled=true：8 个 Agent 类型、15 个执行步骤、42 条消息、19 条实际通信边、0 条拓扑拒收；五节点（ner/logic/subject/statute/judge）均由独立 Agent 承载。

因此上一版把“多智能体”改成“多节点”属于过度保守，本版已撤回；只保留数据、基准、模型与部署口径的修正。作品名称保持不变。
"""

def iter_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    seen = set()
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                key = cell._tc
                if key in seen:
                    continue
                seen.add(key)
                for p in cell.paragraphs:
                    yield p

def replace_in_paragraph(p, old, new):
    text = p.text
    if old not in text:
        return False
    updated = text.replace(old, new)
    if p.runs:
        p.runs[0].text = updated
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.text = updated
    return True

def main():
    src_hash = hashlib.sha256(SRC.read_bytes()).hexdigest()
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, DST)
    doc = Document(str(DST))

    hits = {i: 0 for i in range(len(RULES))}
    for p in iter_paragraphs(doc):
        for i, (old, new, _) in enumerate(RULES):
            if old in p.text and replace_in_paragraph(p, old, new):
                hits[i] += 1
    doc.save(str(DST))

    doc2 = Document(str(DST))
    after_text = "\n".join(p.text for p in iter_paragraphs(doc2))
    remaining = [old[:40] for old, _, _ in RULES if old in after_text]
    missed = [i for i, c in hits.items() if c == 0]

    txt_path = DST.with_suffix(".txt")
    txt_path.write_text(after_text, encoding="utf-8")

    lines = ["# 《马宇轩 申报书》修订对照表（与实测对齐·v2）", ""]
    lines.append(f"- 源文件：{SRC}")
    lines.append(f"- 源文件 SHA256：{src_hash}")
    lines.append(f"- 修订稿：{DST}")
    lines.append(f"- 修订稿 SHA256：{hashlib.sha256(DST.read_bytes()).hexdigest()}")
    lines.append(f"- 规则数：{len(RULES)}；命中规则：{sum(1 for c in hits.values() if c)}；未命中规则：{len(missed)}")
    lines.append("")
    lines.append(MULTI_AGENT_NOTE)
    lines.append("")
    for i, (old, new, why) in enumerate(RULES, 1):
        lines.append(f"## {i}. 命中 {hits[i-1]} 处")
        lines.append("")
        lines.append(f"**原文**：{old}")
        lines.append("")
        lines.append(f"**改为**：{new}")
        lines.append("")
        lines.append(f"**理由**：{why}")
        lines.append("")
    if remaining:
        lines.append("## 警告：仍残留的原文")
        for r in remaining:
            lines.append(f"- {r}…")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "src": str(SRC), "dst": str(DST), "report": str(REPORT), "txt": str(txt_path),
        "src_sha256": src_hash,
        "dst_sha256": hashlib.sha256(DST.read_bytes()).hexdigest(),
        "rules": len(RULES),
        "hit_rules": sum(1 for c in hits.values() if c),
        "missed_rules": missed,
        "hits": hits,
        "remaining": remaining,
        "multi_agent_mentions": after_text.count("多智能体"),
        "multi_node_mentions": after_text.count("多节点"),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
