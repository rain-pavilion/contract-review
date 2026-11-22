# 智律契灵 v2（zhilv_qiling_v2）

> 面向政府采购合同的**图结构引导多智能体**智能审查系统 —— v2 重写迭代版。
>
> 本目录是**独立实现**：不 `import` 原 `contract_review/` 包，只把 `data/` 下的语料与法条
> 当作只读资产复用。原程序的任何一个文件都没有被修改。

```
zhilv_qiling_v2/
├── __init__.py      包定位与 9 条设计升级清单
├── version.py       版本与基线标签（v1-langgraph-qwen25）
├── config.py        路径/开关/降级说明 + 能力探测 capabilities()
├── models.py        共享数据类型（SourceBlock→Clause→Contract…）与 27 风险码表
├── text.py          中文归一化：金额（含大写）、日期（含中文数字）、编号、分句
├── layout.py        版面感知：分栏、阅读顺序、页眉页脚、断行合并、表格行
├── parsers.py       DOCX/PDF/TXT/HTML/图片 → SourceBlock（PyMuPDF 可选）
├── ocr.py           OCR 通道（RapidOCR/Paddle/Tesseract 三后端 + 预算控制 + CER）
├── features.py      27 维边界特征 / 18 维风险特征 / 15 维重排特征
├── learned.py       纯 JSON 权重学习的四个组件（边界/风险/重排/校准/NER）
├── splitter.py      规则+分类器融合边界 → DP 全局切分（Viterbi）→ 条款树
├── ner.py           三通道要素抽取（规则 + 感知机 + 可选 LLM）与一致性信号
├── graph.py         合同结构图 + 义务链 + 法条标签图
├── retrieve.py      四元组查询改写 + BM25/LSA/标签图三路召回 + 加权 RRF + LTR
├── detectors.py     31 类注入式缺陷专用检测器（零/负金额、期限倒置、无限授权…）
├── rules.py         27 风险码确定性规则引擎（要素层/条款层/文档层）
├── agents.py        五智能体 + 证据仲裁 + 融合
├── engine.py        自研 DAG 执行引擎（并行分支 + 四路有界回环 + 轨迹）
├── pipeline.py      端到端编排（prepare → extract → build → review → batch）
├── crawl.py         多源政采合同爬虫（北京/全国/天津开放数据 + 断点续爬 + OCR）
├── train.py         五组件训练与评测（真实指标写 reports/training_report.md）
├── api.py           标准库 HTTP 服务（/api/review、/api/batch、/api/retrieve…）
├── cli.py           命令行（review/batch/explain/retrieve/crawl/ocr/train/models/doctor/schema）
├── webui.html       深色单页前端（原生 JS，零依赖）
├── models/          训练产物（JSON，可 diff、可手改、可解释）
├── reports/         训练报告与实验记录
├── docs/refs/       文献综述、升级清单、BibTeX
└── tests_v2/        118 个单元测试
```

## 快速开始

```bash
# 环境自检（缺什么、降级成什么，一眼看清）
python -m zhilv_qiling_v2.cli doctor

# 审查一份合同（Markdown 报告 + 证据 + 引用）
python -m zhilv_qiling_v2.cli review samples/tech-development-contract.txt --verbose

# 看节点轨迹与耗时
python -m zhilv_qiling_v2.cli explain samples/tech-development-contract.txt

# 法条检索（可直接观察三路通道与重排后的位次）
python -m zhilv_qiling_v2.cli retrieve --text "租赁期限二十年以上" --top-k 5

# 批量评测 + CSV
python -m zhilv_qiling_v2.cli batch --limit 20 --csv reports/batch.csv

# 训练五个组件（CPU 约 3 分钟）
python -m zhilv_qiling_v2.train

# 起 Web 界面（默认 http://127.0.0.1:8610）
python -m zhilv_qiling_v2.cli  # 或 python -c "from zhilv_qiling_v2.api import main; main([])"

# 抓合同（无令牌时天津源会写 degraded 而不是崩）
python -m zhilv_qiling_v2.cli crawl --sources ccgp_beijing --limit 3 --use-ocr

# 跑全部测试
python -m unittest discover -s zhilv_qiling_v2/tests_v2 -t .
```

## v1 → v2 的九处实质性升级

| # | 升级 | v1 的做法 | v2 的做法 | 落点文件 |
|---|------|-----------|-----------|----------|
| 1 | 依赖面 | streamlit + qdrant + langgraph + reportlab | 标准库优先，PyMuPDF/sklearn/OCR/requests 全部可选，缺了就降级并写 `degraded` | `config.py` |
| 2 | 版面 | 直接按段落切 | 分栏投影、阅读顺序、跨页页眉页脚剥离、断行合并、表格行识别 | `layout.py` |
| 3 | OCR | 无 | 三后端 + 预算（页数/秒数）+ 与文字层对比 CER | `ocr.py` |
| 4 | 切分 | 规则边界 + 贪心 | 27 维特征融合 → **DP 全局最优（Viterbi）**，带长度先验正则项 | `splitter.py` |
| 5 | 要素抽取 | 规则 + LLM | 规则 + 字符级感知机 + LLM 三通道融合，含大写金额/中文日期/零负金额 | `ner.py` |
| 6 | 检索 | BM25 + 向量 + RRF | 四元组改写 + 三路召回 + **可训练融合权重** + 15 维列标准化 LTR + 标签扩展 | `retrieve.py` |
| 7 | 编排 | LangGraph | 自研 DAG 引擎，分支并行、四路有界回环、每节点轨迹与摘要 | `engine.py` |
| 8 | 置信度 | 原始分数 | isotonic 分段线性校准（Brier 0.2768 → 0.0977） | `train.py` |
| 9 | 数据 | 手工 50 份 | 多源爬虫 1265 页 + 生成集 + 对抗集，共 113 份可训练样本 | `crawl.py` |

## 申报书五节点的落地（v2.1 新增）

申报书写的五个节点，在这一版里全部有对应代码，且模型不可用时都能降级：

| 节点 | 申报书 | 落点 | 模型不可用时 |
|------|--------|------|--------------|
| ① NER | 当事人/金额/日期/标的 | `ner.extract_elements` → `nodes.node_ner` | 规则通道照跑 |
| ② 逻辑错误检测 | LegalOne-8B 遍历推理 | `nodes.node_logic` = 规则 27 码 + 15 检测器 + `_llm_logic`（**引文必须命中原文才收**） | 规则通道照跑，记 `llm_off` |
| ③ 主体资格审查 | 法律知识图谱 + RAG | `subject_graph.check_subjects` → `nodes.node_subject`（11 类资质 / 5 类权限 / 7 类信用线索 + 图上路径作证据） | 纯规则，无依赖 |
| ④ 法条匹配 | BERT/Mamba + 余弦 + 多头注意力判"一致/冲突" | `retrieve`（三路召回+RRF+LTR） + `entailment.judge_pair`（8 维判别特征，输出 entail/contradict/neutral） | 判别器在本地，照跑 |
| ⑤ 综合评估 | LawBench 评估 → 结论 → 人工复核 | `agents.score_risks`（确定性 0~100） + `nodes.node_judge`（模型只写复核意见） | 只输出分数与规则摘要 |

②③④ 互为输入：② 的 `focus_clause_ids` → ③ 优先核验相关主体；③ 的 `query_hint`（主体类型/所需资质/是否机关）→ ④ 收窄检索；④ 判为 contradict 的条文不进引用，改为给候选挂 `entailment_conflict` 证据并 +0.05 置信，回灌 ②。

三条新护栏：**模型不能造证据**（`_locate_quote` 原文子串校验，未命中计 `rejected`）、**模型不能改分数**（分数永远由 `score_risks` 算）、**降级必须留账**（写进 `report.stats["degraded"]`）。

模型通道用环境变量打开，不写进代码：

```powershell
$env:ZLQL_LLM_ENDPOINT="http://127.0.0.1:11434/v1"   # Ollama / vLLM / 云 API 均可
$env:ZLQL_LLM_MODEL="qwen2.5:7b"
$env:ZLQL_ENABLE_LLM="1"
python zhilv_qiling_v2\reports\experiments\check_llm_channel.py   # 假端点自检，不需要真模型
```

## 实测指标（本机 CPU，2026-11-04，全部可复现）

| 指标 | 数值 | 复现命令 |
|------|------|----------|
| 单元测试 | **153 passed**（core 20 / review 16 / api+cli 15 / ocr 29 / crawl 38 / nodes 23 / gpu 12） | `python -m unittest discover -s zhilv_qiling_v2/tests_v2 -t .` |
| 法条检索 Recall@5（强金标准） | **0.694**（纯 RRF 0.628，v1 报告 0.600） | `python -m zhilv_qiling_v2.train --only rerank` |
| 法条检索 MRR@5（强金标准） | **1.000**（v1 0.806） | 同上 |
| 对抗集注入缺陷检出 | 10/10 合同、32/32 缺陷，R=1.000 | `python reports/experiments/eval_adversarial.py` |
| OCR 字符错误率 CER | **0.79%**（池化，113 页有文字层做参照） | `python -m zhilv_qiling_v2.cli ocr X.pdf --cer` |
| OCR 速度 | 7.56 s/页（CPU，200 dpi，平均置信 0.986） | 同上 |
| 置信度校准 | Brier 0.2768 → **0.0977**（267 个样本，11 箱） | `python -m zhilv_qiling_v2.train --only calibration` |
| 单份端到端 | 首份 2.5 s（含法条索引 2.4 s），后续 0.02–0.36 s/份 | `cli review` 里的 `elapsed_seconds` |
| 五节点端到端 | 评分 73、风险 6 条、24 条款，trace 8 个节点 | `python -m zhilv_qiling_v2.cli review samples\tech-development-contract.txt --trace` |
| NER 微平均 F1（GPU） | 字符感知机 0.0179 → 规则 0.518 → **BERT-CRF 0.9505**（严格口径） | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade ner --epochs 10` |
| 条款—条文判别（GPU） | 双塔 cross-attention macro-F1 **0.5771**、准确率 0.9231 | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade entail --pairs 300 --epochs 6` |
| 密集通道对照 | LSA 强金标准 Recall@5 0.628 vs BGE-M3 **0.167**（负结果，默认仍用 LSA） | `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade embed` |

## GPU 升级（v2.2，2026-11-10，RTX 5070 Laptop 8 GB / CUDA 12.8 / sm_120）

申报书里的三处"必须 GPU"的组件现在真的跑起来了，数字全部来自本机实跑：

| 组件 | 申报书设想 | 落地方案 | 实测 |
|------|-----------|----------|------|
| 节点① 要素抽取 | NER F1 ≥ 90% | `bert_ner.py`：bert-base-chinese + BERT-CRF（`pytorch-crf`），金标 3 倍过采样、1082 个训练窗口；推理侧加大写重复/单价两道金额闸 | **微平均 F1 0.9505**（P 0.9058 / R 1.0000）、补全口径 0.9238、宏 0.9508；逐类 `PARTY` 0.9767、`AMOUNT` 0.8831、`DATE` 0.9434、`SUBJECT` 1.0000 |
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

## 三条硬护栏（与 v1 一致，且更严格）

1. **引用可溯源**：`legal` 只接受本地知识库里真实存在的 `article_uid`，模型生成的条文号一律丢弃。
2. **证据不足不外推**：证据强度不足时归入"事实不足/需补充材料"分类，并进人工复核队列。
3. **结论为辅助**：报告首行明确"审查辅助，不构成法律意见"，评分只用于排序与筛选。

## 存储与统一评测台（v2.3）

申报书指标里"支持十万份级存储 / 检索 / 批量审查调度"与"系统化评估"此前是空白：
v1 用 qdrant 做向量检索，但审查结果本身没有归档层，评测也只有零散实验脚本。
v2.3 补上这两块，都是纯标准库，笔记本可直接跑。

### `store.py`：SQLite 审查结果库

| 表 | 内容 | 用途 |
|---|---|---|
| `contracts` | 来源、类型、评分、风险数、耗时、文本指纹 | 列表页与统计 |
| `risks` | 风险码、分类、等级、置信、是否待复核、标签 | 按码/按级筛选 |
| `citations` | 法条名、条号、`article_uid` | 引用溯源（只存索引里真实存在的条） |
| `runs` | 批量任务起止、处理数、失败数 | 调度账本 |
| `clause_fts` | 条款正文（FTS5 `trigram` 分词） | "哪份合同写过这句话" |

三个设计取舍：

1. **单文件 SQLite**：零依赖，`PRAGMA journal_mode=WAL` + 三个索引，百万行级分页仍是毫秒。
2. **断点续跑**：`batch_to_store(..., resume=True)` 按 `contract_id` 主键跳过已完成项，
   重跑只补新文件（实测第二次 `processed=0 skipped=20`）。
3. **中文全文检索必须用 trigram**：默认 `unicode61` 把整句当一个 token，
   "违约责任"永远匹配不上（实测返回 0 条）；换 `trigram` 后子串可检索，不可用时自动退化 LIKE。

### `bench.py`：三层统一评测台

| 任务 | 指标 | 口径来源 |
|---|---|---|
| `retrieval` | 强/弱金标准分层 R@1..10、MRR@5 | **复用 `train.evaluate_retrieval`**（与 v1 脚本同协议） |
| `detection` | 逐码 TP/FP/FN、合同级命中 | 对抗集 `issue_codes` 为金标准 |
| `latency` | P50/P95/最大、份每分钟 | 固定样本集端到端 |

指标只允许一处实现：bench 第一版自己重算过一遍，得出 rrf 0.875 / ltr 0.125 这种与论文
（0.6278 / 0.6944）冲突的数字 —— 同一指标两个口径是答辩硬伤，已改为直接调用训练脚本里的那套。

### 实测（2026-11-11，`python -m zhilv_qiling_v2.cli bench`）

```
检索 rrf_only     strong R@5=0.6278 R@10=0.6556 MRR@5=1.0000
检索 rrf_plus_ltr strong R@5=0.6944 R@10=0.8889 MRR@5=1.0000
检出 P=0.508 R=1.000 F1=0.674 合同 10/10
延迟 P50=1.52s P95=3.09s 首份=0.38s 吞吐=49 份/分
```

批量入库实测：20 份对抗合同 → 151 条风险 / 293 条引用，评分区间 0~51、均值 11.95，
平均 468 ms/份；`store stats` 与 `store list` 分页、`store search 押金` 条款检索均正常。

### 命令行

```bash
python -m zhilv_qiling_v2.cli review samples/tech-development-contract.txt --store   # 审查并入库
python -m zhilv_qiling_v2.cli batch --root data/generated_contracts --limit 20 --store
python -m zhilv_qiling_v2.cli store stats
python -m zhilv_qiling_v2.cli store list --size 10 --severity high
python -m zhilv_qiling_v2.cli store risks ct_xxxxxxxxxx
python -m zhilv_qiling_v2.cli store search 押金
python -m zhilv_qiling_v2.cli bench --tasks retrieval,detection,latency
```

### 同批修掉的两个"静默漂移"

1. **BGE-M3 改为 opt-in**：接上 BGE 当密集通道后强金标准 R@5 从 0.628 掉到 0.167，
   但代码原是无条件启用，导致之后所有评测都在 BGE 上跑、复现不出论文数字。
   现在 `Config.use_bge=False`（`ZLQL_USE_BGE=1` 显式开启）。
2. **`rerank_ltr.json` 补 `normalize` / `fusion_weights` / `rrf_k`**：
   训练脚本手写 JSON 时漏了这三个键，推理端读不到 `normalize` 就退回逐条打分（量纲差两个
   数量级），+LTR 的 R@5 从 0.6944 掉到 0.2222。补上后 `bench` 与论文数字一致。

单元测试：**153 → 162**（新增 `tests_v2/test_store.py` 9 例：落库读回、幂等、分页过滤、
中文子串检索、统计一致性、断点续跑、评测台形状与可序列化）。

## 降级矩阵（缺依赖时的真实行为）

| 缺失 | 影响 | 降级行为 |
|------|------|----------|
| PyMuPDF | 无法抽 PDF 文字层/版面 | 走最小 PDF 解析（zlib+正则），或整页进 OCR 队列 |
| sklearn | 无 LSA 稠密通道 | 自动退回 BM25 + 标签图两路融合（`stats.degraded` 记录） |
| RapidOCR/Paddle/Tesseract | 扫描件无法识别 | `needs_ocr_pages` 记账，返回空文本并写警告，不抛异常 |
| requests | 无法在线爬取 | 爬虫返回 `degraded`，本地语料与审查链路不受影响 |
| LLM 端点 | 无大模型通道 | 全部走规则 + 训练权重；`enable_llm=False` 是默认值 |
| NER 模型权重 | —— | **默认开启**（`ner_use_bert=True`；权重缺席时自动降级到规则通道） |

## 关于 NER 模型通道

`train.py` 会训练并落盘 `models/ner_perceptron.json`，但 `Config.ner_enable_model`
默认是 `False`：在纯 CPU、无预训练字符表示、仅 88 个训练窗口的条件下，感知机的留出集
宏平均 F1 只有 **0.0179**（同一留出集上规则通道 0.4204），会把非实体串当当事人灌进要素表。这个负结果被
保留在 `reports/training_report.md` 里作为消融记录，换 GPU + BERT-CRF 后置
`ZLQL_NER_MODEL=1` 即可打开。

## v2.4：申报书剩余缺口的四项落地（2026-11-12）

上一版对照申报书列了九处差异，其中三处是真缺口（节点③"图谱+RAG"名不副实、
"十万份级存储"未压测、"调 LawBench 评估"未做），加上"名字含大模型但链路无 LLM"。
这一版逐项补：

| 差异 | 落地文件 | 实测 |
|---|---|---|
| 节点③ 只有关键词表，没有图谱与依据 | `legal_kg.py` + `data/qualification_kg.jsonl` | 15 条资质条目（13 生效 / 2 已废止）、四类节点三类边 45 节点 30 边；检索首位正确率见 `test_legal_kg.py` 12 例 |
| "十万份级存储"只在 20 份上验过 | `tools/_store_scale.py` | **10 万份写入 177.6 s（563 份/秒）**；库 57.4 MB；深分页 page=5000 75.8 ms；FTS 0.1~17.6 ms；`stats()` 35.5 ms |
| "调 LawBench 评估" 没做 | `lawbench.py` | 六项能力映射到 LawBench 任务类型与指标类型，5 项出数（见下表） |
| 名字含"大模型"但默认链路没有任何模型调用 | `local_llm.py` | 本机 Qwen2.5-1.5B-Instruct（3.10 GB）以 **OpenAI 兼容服务**接入，`ZLQL_LLM_ENDPOINT` 一填即生效；端到端 trace 新增 `logic_llm` 节点 |

### 模型通道怎么接（两种都不改代码）

```bash
# A. 本地小模型（本机已验证：加载 4.5 s，生成 1.17 s / 64 token）
python -m zhilv_qiling_v2.local_llm --serve --port 8620
export ZLQL_LLM_ENDPOINT=http://127.0.0.1:8620/v1
export ZLQL_LLM_MODEL=qwen2.5-1.5b-instruct

# B. 换成申报书里的 LegalOne-8B（vLLM / Ollama / 云网关任一，只要 OpenAI 兼容）
export ZLQL_LLM_ENDPOINT=http://<host>:<port>/v1
export ZLQL_LLM_MODEL=LegalOne-8B
```

配了端点就自动开通道（`ZLQL_ENABLE_LLM=0` 可显式关掉）。**硬护栏不变**：模型给的每条
结论必须能在条款原文里定位到引文，否则丢弃并计入 `rejected`。

### LawBench 式评测结果（`python -m zhilv_qiling_v2.lawbench`）

| LawBench 任务 | 本项目落点 | 指标 |
|---|---|---|
| 2-6 Extraction | 节点① 要素抽取 | micro-F1 **0.9505**（补全金标口径 0.9238）、macro 0.9508 |
| 2-3 MLC | 27 风险码检出（对抗集） | P 0.5079 / R **1.0000** / F1 0.6737 |
| 3-1 MLC（排序） | 节点④ 法条推荐 | Recall@5 **0.6944** / @10 0.8889 / MRR@5 **1.0000** |
| 2-4 SLC | 条款—条文一致性（双塔） | Accuracy 0.8750 / macro-F1 0.3994（弱监督） |
| 1-2 SLC | 节点③ 主体资格审查 | 暂无金标，如实报空表 |
| generation | 节点⑤ 复核意见 | 无参考答案，只报可用性 |

**这不是 LawBench 官方分数**：数据是自建 113 份语料 + 对抗集 + 真实政采合同，没有用
LawBench 的划分与标注；对齐的是"任务类型 + 指标类型"，便于横向比较，不冒充榜单成绩。


## 主脑调度与分级复核（v2.5）

申报书里"各节点将子任务派发给专门的大模型或工具"这句，此前只是把五个函数按顺序调一遍。
v2.5 补了两个真实 Agent，并把调用改成**分级**——昂贵的 8B 只碰最该碰的条款。

### 图结构（拓扑即白名单）

```
intake ──→ orchestrator ──→ ner / logic / subject / statute
              ↑                    │
              │  defects.detected  │
              └───── logic ←────────┘
orchestrator ──→ reviewer ──→ judge
```

- `MessageBus(strict=True)` 只放行 `TOPOLOGY` 里声明过的边，越界投递直接拒收并计数；
- 轨迹可导出（`MultiAgentRun.dump_trace()`），"谁在什么时候给谁发了什么"落在 JSON 里。

### 主脑（orchestrator）跑两趟

| 轮次 | 触发 | 产出 |
| --- | --- | --- |
| 第一趟 | `intake` 交来解析结果 | 给四个功能节点发 `task.assign`，带**档位**（`rules` / `llm` / `local`）与模型名 |
| 第二趟 | `logic` 回报 `defects.detected` | 按「严重度 × 100 + 置信度」排序取前 `llm_review_top_k` 条条款，发 `review.assign` 给 `reviewer` |

没配 `ZLQL_LLM_STRONG_ENDPOINT` 时第二趟直接跳过并记一条 note——不会让每条合同都去等一个不存在的服务。

### 分级复核（reviewer）

`reviewer` 只跑主脑指派的那几条条款，配置独立：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZLQL_LLM_STRONG_ENDPOINT` | 空 | 强模型端点（如 4bit 的 LegalOne-8B）；空则关闭分级复核 |
| `ZLQL_LLM_STRONG_MODEL` | `legalone-8b` | 强模型名 |
| `ZLQL_LLM_REVIEW_TOP_K` | `3` | 只复核风险最高的前 K 条条款 |
| `ZLQL_LLM_STRONG_TIMEOUT` | `600` | 强模型超时（8B 4bit 在本机约 2.25 token/s，90 s 一定不够） |

### 实测（`samples/tech-development-contract.txt`，24 条款）

| 档位 | 耗时 | 评分 | 风险 | 说明 |
| --- | --- | --- | --- | --- |
| 规则全量 | 0.55 s | 83 | 3 | 无模型 |
| 1.7B 全量（6 条款） | 49.0 s | 58 | 6 | 3 条模型产出，误套 `unverified_property` |
| 8B 全量（6 条款） | 921.5 s | 59 | 7 | 4 条模型产出 |
| **规则全量 + 8B 复核 top-1** | **99.8 s** | **77** | **4** | 8B 补出 `jurisdiction_ambiguous`；比全量快 **9.2 倍** |

### 插件位

外部模块可以往图里加节点：

```python
from zhilv_qiling_v2 import agents_multi as M

def handle(ctx):
    ctx.emit("auditor", "judge", "audit.done", {"ok": True})
    return {"notes": ["插件节点跑过了"]}

M.register_agent(
    "auditor",
    {"role": "合规复核", "executor": "custom", "model": "plugin"},
    handle,
    upstream=("statute",),
    downstream=("judge",),
)
```

注册后 `agent_specs()` 与 `TOPOLOGY` 同步更新，`build_agents()` 会自动带上它；
拓扑外投递照样拒收，所以插件也必须守图。

### 测试

`zhilv_qiling_v2/tests_v2/test_agents_dispatch.py`（5 例）守住三件事：主脑开工前确实派活给四个节点、
没配强模型端点时不派复核、插件注册后拓扑与调度同步。全量 **206 tests OK**。
