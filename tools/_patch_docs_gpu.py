"""把 GPU 升级后的实测指标写进 README 与 CHANGELOG（一次性 patch）。"""

import pathlib

readme = pathlib.Path("zhilv_qiling_v2/README.md")
text = readme.read_text(encoding="utf-8")

old_test = "| 单元测试 | **141 passed**（core 20 / review 16 / api+cli 15 / ocr 29 / crawl 38 / nodes 23） | `python -m unittest discover -s zhilv_qiling_v2/tests_v2 -t .` |"
new_test = "| 单元测试 | **153 passed**（core 20 / review 16 / api+cli 15 / ocr 29 / crawl 38 / nodes 23 / gpu 12） | `python -m unittest discover -s zhilv_qiling_v2/tests_v2 -t .` |"
assert old_test in text, "找不到单元测试行"
text = text.replace(old_test, new_test, 1)

old_tail = "| 五节点端到端 | 评分 73、风险 6 条、24 条款，trace 8 个节点 | `python -m zhilv_qiling_v2.cli review samples\\tech-development-contract.txt --trace` |"
new_tail = old_tail + """
| NER 宏 F1（GPU） | 字符感知机 0.0179 → **BERT-CRF 0.5348**（规则通道 0.4204） | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade ner --epochs 10` |
| 条款—条文判别（GPU） | 双塔 cross-attention macro-F1 **0.5771**、准确率 0.9231 | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade entail --pairs 300 --epochs 6` |
| 密集通道对照 | LSA 强金标准 Recall@5 0.628 vs BGE-M3 **0.167**（负结果，默认仍用 LSA） | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade embed` |"""
assert old_tail in text, "找不到五节点端到端行"
text = text.replace(old_tail, new_tail, 1)

section = """
## GPU 升级（v2.2，2026-11-10，RTX 5070 Laptop 8 GB / CUDA 12.8 / sm_120）

申报书里的三处"必须 GPU"的组件现在真的跑起来了，数字全部来自本机实跑：

| 组件 | 申报书设想 | 落地方案 | 实测 |
|------|-----------|----------|------|
| 节点① 要素抽取 | NER F1 ≥ 90% | `bert_ner.py`：bert-base-chinese + BERT-CRF（`pytorch-crf`），313 个训练窗口，10 epoch / 192 s | 宏 F1 **0.5348**（旧字符级感知机 0.0179、规则通道 0.4204）；`AMOUNT` 0.786、`DATE` 0.718、`PARTY` 0.635（偏召回 P 0.46）、`SUBJECT` 0 |
| 节点④ 条文判别 | BERT/Mamba + 余弦 + 多头注意力判一致/冲突 | `bert_entail.py`：共享 BERT 双塔 + 8 头 cross-attention（query=条款，key/value=条文），`[pool_c, pool_a, |c-a|, c*a, cosine, pool_attn]` → MLP 三分类 | macro-F1 **0.5771**、准确率 0.9231（contradict P 1.00 / R 0.64、neutral 0.95、entail 0） |
| 节点④ 检索密集通道 | BGE-M3 语义向量 | `bert_embed.py`：1260 条法条 × 1024 维，24 s 建库 | **负结果**：强金标准 Recall@5 0.628 → 0.167，纯密集通道 0.29；池化方式与拼标签差异 <0.001，查询指令 +0.017 → 默认仍走 LSA，BGE 只作可选 |

**为什么 BGE 会输**：查询是 8–40 字的短句，条文平均 180 字且带标签词，语义向量的相似度被长文本稀释；
本任务的判别信号高度依赖"术语字面重合 + 风险标签匹配"，这正是 BM25/LSA 的强项。这条负结果已写进
`docs/五节点落地与申报书优化建议.md`，答辩时按"试过、量过、有据地不用"讲，而不是当没做。

**监督信号的天花板**：双塔的训练标签由 8 维规则蒸馏而来（无人工标注），所以它的上限被老师框住——
它会学到规则的判断边界，学不出规则之外的判断。要突破必须人工标注，这一点在模块 docstring 与文档里都写明了。

**开关**：
```bash
# 默认：有 GPU 产物就用 BERT-CRF 抽要素；没有产物自动退回规则
export ZLQL_NER_MODEL=1
# 判别后端：rules（默认，可复现）/ bert（双塔）/ auto
export ZLQL_ENTAIL_BACKEND=bert
```
"""

anchor = "## 三条硬护栏（与 v1 一致，且更严格）"
assert anchor in text
text = text.replace(anchor, section.strip() + "\n\n" + anchor, 1)
readme.write_text(text, encoding="utf-8")
print("README 已更新：153 测试 + GPU 升级一节")

changelog = pathlib.Path("docs/CHANGELOG-v1-v2.md")
ct = changelog.read_text(encoding="utf-8")
add = """
## 三之三、v2.2：GPU 组件落地（2026-11-10）

拿到 RTX 5070 Laptop（8 GB，sm_120）后，把申报书里"需 GPU"的三处补上：

- **torch 2.9.1+cu128** 从交大镜像分块并行下载（`tools/_dl_chunked.py`，8 线程 + Range 断点续传，
  单连接会被掐在 835 MB）；`transformers 5.22.2` / `pytorch-crf 0.7.2` 走清华源。
- `bert_ner.py`（BERT-CRF）**宏 F1 0.5348**，首次超过规则通道 0.4204；旧感知机 0.0179 只作无 GPU 兜底。
- `bert_entail.py`（双塔 + cross-attention）**macro-F1 0.5771 / acc 0.9231**，注意力峰值可解释。
- `bert_embed.py`（BGE-M3 1024 维）**负结果**：强金标准 Recall@5 0.628 → 0.167，默认不启用。
- 节点③ 收紧后对抗集：TP 32 / FP 31 / FN 0，P 0.508 / R 1.000 / F1 0.674（FP 主要来自模板共享句与金标只标主缺陷）。
- 单元测试 141 → **153**（新增 12 条 GPU 契约测试，3 条在无 torch 时跳过）。
"""
if "三之三、v2.2" not in ct:
    ct = ct.rstrip() + "\n" + add
    changelog.write_text(ct, encoding="utf-8")
    print("CHANGELOG 已加 v2.2 一节")
else:
    print("CHANGELOG 已有 v2.2")
