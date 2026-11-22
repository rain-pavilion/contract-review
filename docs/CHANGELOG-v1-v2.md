# 智律契灵 v1 → v2 变更说明（升级对照）

> 用途：挑战杯申报材料 / 答辩讲解时的"我们做了什么升级"清单。
> 原则：**原程序（`contract_review/`、`checks/`、`app.py`、`scripts/` 下 v1 的 10 个脚本）一个字节都没有改动**，
> v2 是 `zhilv_qiling_v2/` 下的独立实现，只把 `data/` 里的语料与法条当只读资产。

## 一、为什么重写而不是打补丁

v1 的约束是"能用"：LangGraph 组织五个 Agent、Qwen2.5-7B 走 Ollama、BGE-M3 出向量、qdrant 存索引。
这套技术栈在参赛演示机上能跑，但有三个硬伤：

1. **依赖面太重**：streamlit + qdrant + langgraph + reportlab + Ollama，任一缺失整条链路就断。评审机器上装不全，
   演示就要靠运气。
2. **能力不可降级**：没有大模型时 `legal` 节点直接空转，报告里出现"0 条法条引用"，而这恰恰是最该有引用的部分。
3. **不可度量**：切分靠规则贪心、检索靠固定权重、置信度是原始分，任何一处想调都缺少"改动前后差多少"的对照。

v2 因此重写为**标准库优先、能力可探测可降级、每个组件都有可训练参数与可复现指标**的形态。

## 二、九处实质性升级

| # | 升级 | v1 | v2 | 落点 |
|---|------|----|----|------|
| 1 | 依赖面 | streamlit/qdrant/langgraph/reportlab 全必需 | 标准库优先；PyMuPDF、scikit-learn、RapidOCR、requests 全部可选，缺失即降级并在 `stats.degraded` 记账 | `config.py` |
| 2 | 版面感知 | 段落顺序即阅读顺序 | 分栏投影切栏、栏内 y 吸附排序、跨页页眉页脚剥离（重复率 ≥40%）、断行合并（带元信息行与编号保护）、表格行识别 | `layout.py` |
| 3 | OCR | 无 | RapidOCR / PaddleOCR / Tesseract 三后端统一接口，页数与秒数预算，与文字层对比算 CER | `ocr.py` |
| 4 | 切分 | 规则边界 + 贪心串接 | 融合 27 维特征（编号/层级/长度/终止符/几何 gap/字号跳变/加粗/数字密度…）→ **DP 全局最优（Viterbi）**，目标函数含长度先验罚项 | `splitter.py`、`features.py` |
| 5 | 要素抽取 | 规则 + LLM 两通道 | 规则 + 字符级平均感知机（Viterbi 解码）+ LLM 三通道；新增大写金额、中文数字日期、零/负金额；输出 9 类一致性信号 | `ner.py`、`text.py` |
| 6 | 法条检索 | BM25 + 向量 + 固定 RRF | 四元组查询改写（行为·主体·标的·法律后果）+ 三路召回（BM25 / LSA / 标签图 PPR）+ **网格搜索的可训练融合权重** + 15 维列标准化 LTR 重排 + 标签扩展补位 | `retrieve.py`、`train.py` |
| 7 | 编排 | LangGraph 图 | 自研轻量 DAG 引擎：分支并行（线程池）、四路有界回环、逐节点轨迹与摘要、无外部依赖 | `engine.py` |
| 8 | 置信度 | 原始分数直接输出 | isotonic 分段线性校准（强制单调），风险分级与"是否进人工复核队列"都由校准后概率决定 | `learned.py`、`train.py` |
| 9 | 数据 | 手工 50 份样本 | 多源政采爬虫（北京/全国/天津开放数据，含断点续爬与 OCR 回写）+ 生成集 + 对抗集，共 113 份可训练样本 | `crawl.py`、`pipeline.py` |

## 三、指标对照（同一台机器、同一份语料、同一套金标准）

| 指标 | v1（原程序报告值） | v2（实测） | 说明 |
|------|------------------|-----------|------|
| 法条检索 Recall@5（强金标准） | 0.600 | **0.694** | 协议完全复刻 v1（按"样本×风险码"计次，strong/weak 分级） |
| 法条检索 MRR@5 | 0.806 | **1.000** | 前 5 位内必中至少一条强金标准 |
| 单份端到端耗时 | 132.78 s（含 7B 模型） | **0.02–0.36 s**（首份 2.5 s 含索引） | v1 走本地大模型，v2 全 CPU 无模型 |
| 引用条数 | 4 条风险仅 1 条引用 | 每条风险平均 2 条可溯源引用 | 硬护栏：不在知识库里的条文一律丢弃 |
| 对抗集缺陷检出 | v1 未做该项批量评测 | 32/32 缺陷、合同级 10/10、R=1.000 | 金标准 = 生成清单里的 `issue_codes` |
| 置信度可靠性 | 未校准 | Brier 0.2768 → **0.0977** | 267 个样本、11 箱 |
| 单元测试 | 无（只有 12 个手工 check 脚本） | **141 个 unittest** | core 20 / review 16 / api+cli 15 / ocr 29 / crawl 38 / nodes 23 |

## 三之二、v2.1：把申报书的五节点补齐（2026-11-10）

v2.0 按"能跑、能度量"重写时，申报书写的节点③（主体资格审查）和节点④的"一致/冲突判别"是缺的，LLM 通道也是关的。v2.1 把它们补回来，全部带降级：

| 节点 | 新增/改动 | 文件 |
|------|-----------|------|
| ① NER | `node_ner` 封装，要素登记 + 降级标记 | `nodes.py` |
| ② 逻辑错误检测 | `node_logic` = 规则 27 码 + 15 检测器 + `_llm_logic`；模型每条缺陷必须带 `quote` 且在原文可定位，否则丢弃并计 `rejected` | `nodes.py` |
| ③ 主体资格审查 | **新增**小知识图谱（主体-类型-资质-权限-信用）+ 5 条核验 + 图上路径作证据 | `subject_graph.py` |
| ④ 法条匹配 | **新增** 8 维判别特征 + 三分类判别器，输出 entail/contradict/neutral；contradict 不进引用、改为挂证据并 +0.05 置信 | `entailment.py` |
| ⑤ 综合评估 | `node_judge` 让模型只写复核意见与 `focus`，**分数永远由 `score_risks` 算** | `nodes.py` |
| 编排 | 插入 `subject_check` 与 `judge_llm` 两个 trace 节点，②③④ 通过 `focus_clause_ids` / `query_hint` / contradict 回灌互为输入 | `engine.py` |
| 模型通道 | 标准库 OpenAI 兼容客户端（无 SDK），`ZLQL_LLM_ENDPOINT/MODEL/API_KEY` 环境变量开启 | `llm_client.py` |

实测：`Ran 141 tests in 39.6s OK`；五节点端到端评分 73、风险 6 条、trace 8 节点；假端点自检下节点② 接受 21 条、拒绝 21 条（编造引文被挡），节点⑤ 只写意见不改分。


## 四、v2 新增的"能说得出口"的东西

- **可解释的权重**：五个组件的参数全部是 JSON（`models/*.json`），可以 `git diff`、可以手改、可以在报告里逐项列出；不使用 pickle，避免"权重里藏了什么看不见"。
- **可复现的评测**：`train.py` 跑完会把指标写进 `reports/training_report.md`，论文里的每个数字都能顺着这个文件倒查到生成它的那次训练。
- **负结果也留档**：字符级感知机在 88 个窗口上的宏平均 F1 只有 0.0179（同留出集上规则通道 0.4204），因此 v2 **默认关闭**模型通道并把这条写进 README 与论文 6.4 节 —— 比"假装它能用"更可信。
- **降级有账本**：缺 sklearn 少一条召回通道、缺 OCR 把页记进 `needs_ocr_pages`、缺 LLM 就全走规则，每种情况都在返回体的 `degraded` 里出现，界面会显式提示"当前为降级运行"。
- **数据链路可续跑**：爬虫按 `contract_id` 去重、支持 `limit/pages/rate` 与预算化 OCR，中断重跑不会重复下载。

## 五、仍然存在的不足（答辩时要主动说）

1. 法条语料只有《民法典》1260 条（`laws=1`），跨法域的检索能力没有被真正检验；扩到《招标投标法》《政府采购法》需要补语料与标签。
2. 检索金标准规模小（强金标准每类 1~6 条），`Recall@1` 的理论上限被锁死，只能报 `@5` 与 `MRR@5`。
3. 对抗集 FP 偏高（P=0.571）主要来自"模板共享句 + 金标准只标主缺陷"的口径差异，需要一个"未标注但确实存在"的二级标注集才能给出公平的精确率。
4. 无 GPU，字符级 NER 学不出跨度边界；上 BERT-CRF 或换成 span-based 的 W2NER 才是正解。
5. 单页 OCR 7.56 s（CPU），整份 20 页扫描件约 2.5 min，批量场景需要异步队列与多进程。

## 三之三、v2.2：GPU 组件落地（2026-11-10）

拿到 RTX 5070 Laptop（8 GB，sm_120）后，把申报书里"需 GPU"的三处补上：

- **torch 2.9.1+cu128** 从交大镜像分块并行下载（`tools/_dl_chunked.py`，8 线程 + Range 断点续传，
  单连接会被掐在 835 MB）；`transformers 5.22.2` / `pytorch-crf 0.7.2` 走清华源。
- `bert_ner.py`（BERT-CRF）**宏 F1 0.5348**，首次超过规则通道 0.4204；旧感知机 0.0179 只作无 GPU 兜底。
- `bert_entail.py`（双塔 + cross-attention）**macro-F1 0.5771 / acc 0.9231**，注意力峰值可解释。
- `bert_embed.py`（BGE-M3 1024 维）**负结果**：强金标准 Recall@5 0.628 → 0.167，默认不启用。
- 节点③ 收紧后对抗集：TP 32 / FP 31 / FN 0，P 0.508 / R 1.000 / F1 0.674（FP 主要来自模板共享句与金标只标主缺陷）。
- 单元测试 141 → **153**（新增 12 条 GPU 契约测试，3 条在无 torch 时跳过）。

## 三之四、v2.3：存储层与统一评测台（2026-11-11）

**为什么补**：申报书写的"十万份级存储/检索/批量审查调度"和"系统化评估"，在 v2.2 之前是空白 ——
v1 的 qdrant 是向量检索，不是审查结果归档；评测散在若干实验脚本里，没有统一协议。

**做了什么**

- `store.py`：SQLite 单文件库（`contracts` / `risks` / `citations` / `runs` / `clause_fts`），
  WAL + 三索引，分页、按码/按级过滤、断点续跑、条款全文检索（FTS5 `trigram`，中文子串必需）。
- `bench.py`：三层基准（检索 / 检出 / 延迟），检索指标**复用** `train.evaluate_retrieval`，
  结果落 `reports/bench_*.json`。
- CLI：`store stats|list|risks|search`、`bench`，以及 `review/batch/explain --store`。
- 测试：153 → 162（`test_store.py` 9 例）。

**同批修的两个静默漂移**

1. BGE-M3 从"无条件当密集通道"改为 opt-in（`use_bge=False`）：它是负结果（R@5 0.628→0.167），
   默认开着会让后续所有评测复现不出论文数字。
2. `rerank_ltr.json` 补 `normalize`/`fusion_weights`/`rrf_k`：缺失时推理端退回逐条打分，
   +LTR 的 R@5 从 0.6944 掉到 0.2222。

**实测（可复现）**

| 项目 | 数字 |
|---|---|
| 检索 rrf_only strong | R@5 0.6278 / R@10 0.6556 / MRR@5 1.0000 |
| 检索 rrf_plus_ltr strong | R@5 0.6944 / R@10 0.8889 / MRR@5 1.0000 |
| 检出（对抗集） | P 0.508 / R 1.000 / F1 0.674，合同级 10/10 |
| 延迟 | P50 1.52 s / P95 3.09 s / 首份 0.38 s / 49 份每分 |
| 批量入库 | 20 份 → 151 风险 / 293 引用，均值 468 ms/份 |

**仍未做**：真实 LawBench / PROVDATA 数据集不可得（需申请或需 GPU 微调），
`bench` 的检索任务只对齐其**指标形态**，不冒充数据集复现。

## 三之五、v2.4：补齐申报书剩余三处缺口（2026-11-12）

**做了什么**

1. `legal_kg.py`（430 行）+ `data/qualification_kg.jsonl`（15 条）：把节点③从"11 组关键词"
   升级为「主体→行业→资质→法规」知识图谱 + BM25 检索，结论必须带 `basis_law` /
   `basis_article` / `source_url`；两条已废止资质（物业服务企业资质、计算机信息系统集成资质）
   显式标 `abolished` 并抑制报警——这类"拿废止资质报缺证"是真实项目里最常见的误报来源。
2. `tools/_store_scale.py`：10 万份压测。写入 177.6 s（563 份/秒、1.78 ms/份）、库 57.4 MB
   （573.5 B/份）、page=5000 深分页 75.8 ms、FTS trigram 0.1~17.6 ms、`stats()` 35.5 ms。
   结论三项全过：十万份可写 / 深分页可接受 / FTS 可用。
3. `local_llm.py`（约 320 行）：本地因果语言模型包成 OpenAI 兼容服务，让 `llm_client` 那条
   链路一行不改就能跑真模型；`<think>` 段按 LegalOne 契约剥离；支持 `--serve` 与 `--prompt`。
4. `lawbench.py`（约 330 行）：六项能力映射 LawBench 任务类型与指标类型，单任务失败不影响
   整表（`_safe`），结果落 `reports/lawbench.json`。
5. `engine.py` 新增 `logic_llm` 节点（节点② 的模型分支真正接进主链）；`config_from_env()`
   支持 `ZLQL_LLM_*` / `ZLQL_USE_BGE` / `ZLQL_NER_MODEL`，且**配了端点自动开通道**——
   "填了地址却没生效"是最难查的一类静默问题。
6. 测试 162 → **175**（新增 `test_local_llm.py` 13 例：strip_think / extract_json 四种形态 /
   describe / LawBench 任务表完整性与单点失败隔离）。

**实测（2026-11-12）**

- 真实本地模型端到端：加载 4.47 s（cuda），5 条条款遍历 19.2 s，**采纳 4 条 / 引文未通过 6 条**，
  `judge_llm` 生成复核意见但**分数仍由规则算（45）**——模型只读，不改分。
- 小模型的真实短板：它把"合同中的某些术语未经核实"套上了 `unverified_property`（该码本义是
  "房产权属未核实"）。**1.5B 只能验证通道可用，达不到可用质量**；要可用得换 7B/8B，
  这也是申报书写 LegalOne-8B 的原因。

**仍未做**

- LegalOne-8B 权重本机取不到（HF 直连不通、ModelScope 无镜像），现用 Qwen2.5-1.5B 验证协议；
- 节点③ 的资质条目只有 15 条，要做成"可查"至少要覆盖常见 30 类行业；
- LawBench 官方划分未使用（需按其格式重训，属下一阶段）。


### 三之六、v2.5：主脑调度与分级复核（2026-11-13）

**为什么补**：申报书写"各节点将子任务派发给专门的大模型或工具"，实际是单进程顺序调用五个函数；
且 8B 全量跑一份合同要 921 秒，不可能按"每份都上 8B"落地。

**做了什么**：
1. `agents_multi.py` 加 `orchestrator`（主脑）与 `reviewer`（强模型复核）两个 Agent，拓扑重写为
   `intake → orchestrator → 四节点`，`logic → orchestrator` 作为回报边，`reviewer → judge`；
2. 主脑跑两趟：先按合同规模派活（带档位），再按 logic 回报的缺陷挑 top-K 派强模型复核；
3. `nodes.node_logic` 增 `clauses=` 关键字——只对指定条款跑模型分支，规则与检测器仍走全文；
4. 四个新配置：`llm_strong_endpoint` / `llm_strong_model` / `llm_review_top_k` / `llm_strong_timeout`
   （对应 `ZLQL_LLM_STRONG_*`、`ZLQL_LLM_REVIEW_TOP_K`）；
5. `api.py` 的 `/api/health` 增 `llm.strong` 段；`webui.html` 系统页展示分级复核是否开启、强模型名与 top-K；
6. 插件位 `register_agent(name, spec, handler, upstream, downstream)`。

**实测**：规则全量 0.55 s / 83 分 / 3 风险 → 规则全量 + 8B 复核 top-1 **99.8 s / 77 分 / 4 风险**，
比 8B 全量（921 s）快 9.2 倍，且多检出一条 `jurisdiction_ambiguous`。

**测试**：新增 `test_agents_dispatch.py`（5 例），全量 201 → **206**。
