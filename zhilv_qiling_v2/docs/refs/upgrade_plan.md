# 智律契灵 v2 排期清单（由 literature.md 34 条文献收敛）

> 基线：要素抽取宏平均 F1 0.992（规则单通道 0.880）；法条检索 **Recall@5 = 0.600 / MRR@5 = 0.806**；5 Agent + LOOP 仲裁 + 有界回环（上限 3 轮）；七分类风险 + 27 标签 + 0~100 评分。
> 约束：CPU only、无 GPU 训练、无线上 LLM、纯离线；可用样本 50~2000 份；依赖限 numpy / pandas / scikit-learn / PyMuPDF / rapidocr / python-docx（新增仅允许 `xgboost` 或 `lightgbm`，二者均为 CPU 轮子，二选一）。
> 排序原则：**单位工作量收益**（主指标改善 ÷ 小时数）降序；主指标锁定 Recall@5（0.600 为全系统唯一显著短板）。

## 一、升级清单（14 条，按收益/工作量排序）

| 编号 | 升级点 | v2 模块 | 依据文献 | 预期收益（可量化） | 工作量(h) | CPU可行 |
|---|---|---|---|---|---|---|
| U01 | 加权三路 RRF：`w_b/(k+r_b) + w_d/(k+r_d) + w_kg/(k+r_kg)`，网格 `k∈{5,10,20,30,60,100}` × 3 组权重 | `zhilv_qiling_v2/retrieval/rrf_fuse.py` | C2 RRF(k=60)、C3 BGE-M3 稀疏通道、G2 低置信拒识 | **Recall@5 0.600→0.63~0.65**；MRR@5 0.806→0.815~0.82 | 6 | 是（纯公式 + 18~36 组网格，秒级） |
| U02 | 分数校准：七分类 temperature scaling（拟合 T）+ isotonic 单调对齐 0~100 分，输出可靠性曲线 | `zhilv_qiling_v2/risk/calibration.py` | F1 Guo 温度缩放/ECE、F2 Platt 小样本平滑 | **ECE 0.15→0.05**（估）；"92 分 ↔ 高风险概率 0.87"可解释映射 | 5 | 是（单参数 `minimize_scalar`，需 100~500 条标注） |
| U03 | 条款→法条检索 query 改写为 `(行为要件, 主体, 标的, 法律后果)` 四元组模板，槽位由要素抽取 + 功能段填充 | `zhilv_qiling_v2/retrieval/query_rewrite.py` | C4 LegalSearchLM 要素生成、C5 CEKLE 分段定位 | **Recall@5 0.600→0.70+**（主目标，需 50~100 条"条款→正确法条"标注） | 16 | 是（纯规则槽位 + 模板，无训练） |
| U04 | BM25F 三流加权（标题 3.0 / 要件 2.0 / 正文 1.0）+ `(k1,b)` 两轮坐标搜索 | `zhilv_qiling_v2/retrieval/bm25_index.py` | C1 BM25 概率相关性框架 §3.6/§5 | Recall@5 +0.02~0.04；MRR@5 +0.02 | 8 | 是（倒排重建 <1 s） |
| U05 | 法律概念粒度倒排（300~800 概念词典 → 法条/条款双向倒排），作 RRF 第 4 路 | `zhilv_qiling_v2/retrieval/concept_index.py` | G4 无监督多粒度（概念/词/词组） | **Recall@5 +0.02~0.03**，**零标注**；命中理由可解释 | 10 | 是（无训练，纯倒排） |
| U06 | 共形预测：27 标签各出预测集（α=0.1），集合大小=不确定性，超阈转人工 | `zhilv_qiling_v2/risk/conformal.py` | F3 split conformal、H2 三态输出 | 覆盖率保证 ≥90%（分布无关）；漏检率↓；新增"不确定性"指标 | 8 | 是（numpy ≈60 行，需 ≥100 条校准集） |
| U07 | 两阶段精排：三路 RRF 取 top-50 → 10 维特征（BM25/dense/字面/标题命中/要件覆盖/KG 路径/段一致…）→ 树模型选 top-5 | `zhilv_qiling_v2/retrieval/rerank_ltr.py` | G1 DeepLawRec 双塔+XGBoost 再选择、G3 多任务+阈值 | **Recall@5 +0.05**；**MRR@5 0.806→0.85+** | 20 | 是（`HistGradientBoosting`/`xgboost`，训练秒级） |
| U08 | 双 agent 辩论 + 第 3 轮多数表决（触发：条款审查与法条适用等级差 ≥2 档；轮数 = 2） | `zhilv_qiling_v2/agents/consistency.py` | E3 多智能体辩论（算术 67.0→81.8%）、E2 Reflexion 情节记忆 | **风险等级不一致率 −20%**；最长回环 3→2 轮 | 10 | 是（编排改造，离线走规则化证据比对） |
| U09 | 条款功能段五分类标签（主体/权利义务/价款/违约解除/争议解决），仅对"权利义务+违约解除"建向量 | `zhilv_qiling_v2/splitter/clause_section.py` | C5 CEKLE 五段切分、H3 CUAD 41 类映射 | 检索噪声↓；Recall@5 +0.02；索引体积 −40% | 12 | 是（规则分段 + 重建索引） |
| U10 | TransE(k=128) 缺项链接预测：对"应有但缺失"的关系打分，低分转"缺项风险" | `zhilv_qiling_v2/kg/contract_graph.py` | D2 TransE、H4 ContractNLI `not mentioned` | **新增"缺项类"风险，27→31 标签**；缺项召回 ≥0.6 | 12 | 是（numpy SGD，<1000 边训练 <30 s） |
| U11 | 强类型 artifact 交接契约 + schema 校验 + 缺字段回退重跑 | `zhilv_qiling_v2/agents/state.py`、`agents/graph.py` | E1 AutoGen 会话协议、E4 MetaGPT SOP/结构化产物 | 无效结论（缺 evidence）自动回退；级联幻觉↓；平均回环 3→2 轮 | 12 | 是（schema + 校验器） |
| U12 | 合同 KG 的 Louvain 社区划分 + 模板化社区摘要，前置注入跨条款审查 | `zhilv_qiling_v2/kg/graph_rag.py`、`agents/cross_review.py` | D1 GraphRAG（社区摘要 + map-reduce） | **跨条款漏检 −15%**；长合同（>30 条款）结论覆盖率↑ | 14 | 是（louvain 毫秒级；LLM 摘要降级为模板） |
| U13 | 评测口径对齐：LawBench 任务定义 + CUAD 27↔41 映射 + ContractNLI 三态框架 | `zhilv_qiling_v2/eval/bench.py` | H1 LawBench、H3 CUAD、H4 ContractNLI | **标签映射覆盖率 ≥60%**；对外可复现基准，替代自造口径 | 14 | 是（仅借定义与脚本） |
| U14 | 要素抽取规则增强：FMM 词社区特征 + CRF 合法转移掩码 | `zhilv_qiling_v2/extract/legal_lexicon.py`、`extract/rules_elements.py` | B4 FMM-CAM 社区注意力、B3 BiLSTM-CRF 转移约束 | **规则通道 F1 0.880→0.93**；非法标签序列率 →0 | 16 | 是（词典 + 掩码；可选 `sklearn-crfsuite` 训练） |
| | **合计** | | | **Recall@5 0.600→0.70+；MRR@5 0.806→0.85+；规则通道 F1 +0.05；风险标签 27→31** | **163 h** | 全部 CPU 可行 |

## 二、阶段排期（按依赖关系切三段，可并行）

| 阶段 | 周次 | 条目 | 里程碑（验收口径） |
|---|---|---|---|
| **P0 快赢** | 第 1–2 周（约 27 h） | U01、U02、U04 | 检索侧先出数：Recall@5 报出 0.63+，MRR@5 报出 0.82；校准报告（ECE + 可靠性曲线图）落 `docs/refs/calib_report.md` |
| **P1 主攻** | 第 3–6 周（约 74 h） | U03、U05、U06、U07、U08 | **Recall@5 ≥0.70、MRR@5 ≥0.85** 双指标达成；conformal 覆盖率实测 ≥0.88；辩论前后不一致率对照表 |
| **P2 差异化** | 第 7–10 周（约 62 h） | U09、U10、U11、U12、U13、U14 | 风险标签扩到 31（含缺项类）；跨条款漏检对照；CUAD 映射覆盖率 ≥60%；规则通道 F1 ≥0.93；三套评测可一键复跑 |

依赖链（不可颠倒）：
- `U04 → U01 → U07`（BM25F 是融合的一路；融合分是精排特征）
- `U09 → U03`（四元组模板要用功能段标签决定"填哪一类槽位"）
- `U02 → U06`（conformal 的非一致性分数用校准后的概率）
- `U10 ↔ U13`（缺项标签需 ContractNLI 三态口径来评）
- `U11` 是 `U08/U12` 的前置（辩论与社区摘要都以结构化 artifact 为输入）

## 三、验收方式（每个升级点必须有对照实验，不允许只写"已实现"）

1. **检索类（U01/U03/U04/U05/U07/U09）**：固定 60~100 条"条款 → 正确法条"金标（人工标注，双人交叉），报 Recall@1/3/5/10、MRR@5、nDCG@10；每项改动单独消融（baseline / +单项 / 全量），表放 `docs/refs/ablation.md`。
2. **抽取类（U14）**：沿用现有 200 份标注合同，报 macro-P/R/F1 + 非法序列率；规则单通道与双通道各报一行，证明 0.880→0.93 与宏平均不回退。
3. **风险/校准类（U02/U06/U10）**：报 ECE、Brier、覆盖率（实测 vs 名义 1−α）、平均预测集大小、缺项标签的 precision/recall。
4. **多智能体类（U08/U11/U12）**：报"仲裁前后等级不一致率"、"平均回环轮数"、"跨条款漏检数（人工 30 份长合同复核）"。
5. **可复现性**：所有脚本入口 `python -m zhilv_qiling_v2.eval.bench --suite {retrieval,extract,risk,agents}`，随机种子固定 42，结果 JSON + Markdown 双写。

## 四、风险与回退方案

| 风险 | 触发信号 | 回退方案 |
|---|---|---|
| 金标标注量不足（<50 条条款-法条对） | U03/U07 无法训练/调参 | 降级为"弱监督"：用现有规则命中 + 法条标题字面匹配自动生成 300 条银标，仅用于**排序**不用于**阈值**；U03 退回纯模板不改写 |
| 树模型过拟合（样本 <200） | 验证集 Recall 波动 >0.05 | 降到 3 特征（BM25、dense、要件覆盖）+ `LogisticRegression`，并加 5 折交叉验证报告均值±std |
| 无 xgboost/lightgbm 轮子可装 | pip 离线失败 | 用 `sklearn.ensemble.HistGradientBoostingClassifier`（原生，无额外依赖），效果差异在 1~2 pt 内 |
| 概念词典构建质量差（U05） | 概念通道 Recall 低于 BM25 | 词典只保留在 ≥5 条法条中出现且非停用词的概念，规模从 800 砍到 300；不行则关闭该路（RRF 权重置 0） |
| 辩论不收敛（U08） | 3 轮后仍不一致率 >15% | 硬性 2 轮封顶 + 裁判按"证据引用完整度 > 法条匹配度 > 上游置信度"的字典序裁决，保证有限终止 |
| conformal 覆盖率不达标（U06） | 实测覆盖 <0.85（α=0.1） | 检查可交换性：按合同类型分层校准（每类各留校准集）；仍不行则报"按类型条件覆盖"而非边际覆盖 |
| 特征泄露（U07 用到了金标字段） | 验证集 Recall 异常高（>0.95） | 强制特征白名单（不得使用 `legal_ref` 任何字段），并加"时间切分"（旧合同训练/新合同测试）复核 |

## 五、与文献的对应索引

| 方向 | 文献条目 | 落到本表的升级点 |
|---|---|---|
| A 文档解析/OCR | A1 LayoutLMv3、A2 Donut、A3 PP-OCR、A4 MinerU | 未单列（作为 U09/U14 的前置数据质量保障；条件触发 OCR + 表格还原由 `parsers/` 承接） |
| B 要素抽取 | B1 W2NER、B2 UIE、B3 BiLSTM-CRF、B4 FMM-CAM | U14（B3/B4），B1/B2 作为后续 2D 融合与 schema 化的储备 |
| C 检索增强 | C1 BM25F、C2 RRF、C3 BGE-M3、C4 LegalSearchLM、C5 CEKLE | U01、U03、U04、U09（C3 稀疏通道并入 U01 第三路） |
| D 知识图谱 | D1 GraphRAG、D2 TransE、D3 R-GCN、D4 FU-TransH、D5 KG 相似度特征 | U10、U12（D3 的路径特征、D5 的相似度特征作 U07 特征列储备） |
| E 多智能体 | E1 AutoGen、E2 Reflexion、E3 辩论、E4 MetaGPT | U08、U11 |
| F 校准/不确定性 | F1 温度缩放、F2 Platt、F3 conformal、F4 CLAUDETTE | U02、U06（F4 的标签映射并入 U13） |
| G 中文核心 | G1 李琳 DeepLawRec、G2 张虎、G3 余连玮、G4 周献杭 | U07（G1 核心）、U01（G2 拒识）、U05（G4 核心） |
| H 基准数据集 | H1 LawBench、H2 LegalBench、H3 CUAD、H4 ContractNLI | U13（全部）、U06（H2 三态）、U10（H4 not-mentioned） |
