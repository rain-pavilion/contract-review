# 智律契灵 v2 升级文献综述（检索日期 2026-11）

> 项目：《智律契灵——基于图结构引导的多智能体大模型合同智能审查系统》（南通大学，挑战杯科技发明制作 B 类）
> 升级基线：要素抽取宏平均 F1 0.992（规则单通道 0.880）；法条检索 BM25(字 unigram+bigram)+BGE-M3(1024d)+RRF，**Recall@5=0.600 / MRR@5=0.806（明确短板）**；5 Agent（LangGraph）编排 + LOOP 仲裁 + 有界回环；七分类风险 + 27 标签 + 0~100 评分。
> 硬约束：CPU only、无 GPU 训练、无线上 LLM（Ollama 未运行）、纯离线；可用 numpy / pandas / scikit-learn / PyMuPDF / rapidocr / python-docx；可用样本 50~2000 份。
> 模块路径约定（v2 新建，不改动原 contract_review/）：`zhilv_qiling_v2/{parsers,extract,retrieval,kg,agents,risk,eval}/*.py`。

## 收录统计与约束说明

- **共 34 条**，A~H 八方向各 ≥4 条；**中文文献 8 条**（郭力华、田萱、黄治纲、李紫阳、李琳、张虎、余连玮、周献杭）。
- 约束冲突说明：任务要求"每方向 ≥4 条"（8×4=32）与"总计 20~28 条"在数学上不可同时满足。本稿取 **32 条为下限**，另加 2 条（田萱 CEKLE、李紫阳 KG 特征增强）以满足"中文 ≥8"，故总数 34。若评审强制压到 28 条，建议按文末《附录：可裁剪条目》删除 6 条。
- 核验方式：每条均给出 arXiv abs 页 / 出版商 DOI 页 / 期刊官网"引用本文"块 / 官方 GitHub bibtex 之一。**未核验到页码者明确标注"（页码待核）"**；凡标"（未核验到全文，仅见题录）"者仅可引用题录，不得引用其数据。
- DOI 与卷期一律照抄核验页面，未做任何推测构造。

---

## A. 文档解析 / 版面分析 / OCR

### A1 LayoutLMv3：统一文本与图像掩码的文档预训练
- 文献：HUANG Y, LV T, CUI L, et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking[C]//Proceedings of the 30th ACM International Conference on Multimedia (MM '22). 2022: 4083-4091. DOI: 10.1145/3503161.3548112；预印本 arXiv:2204.08387. https://arxiv.org/abs/2204.08387
- 核心思想：把 OCR 文本 token 与图像 patch 放进同一 Transformer，用 **word-patch alignment 损失**（判断 patch 是否被词覆盖）替代昂贵的区域级目标检测，文本掩码与图像掩码统一训练；推理时输出 token 级 (text, bbox) 表示，天然携带版面位置。
- 可迁移点：v2 不要训练新模型，只借两件事——(1) `parsers/pdf_layout.py` 的阅读顺序重建：先按 y 聚类成行、行内按 x 排序、再用"跨栏 x 区间重叠"判定双栏并做 XY-cut，等价于 LayoutLMv3 依赖的 bbox 结构信号；(2) 给每个 SourceBlock 增加 `bbox` 字段（PyMuPDF 的 `page.get_text("dict")` 直接给），使后续条款切分可用"位置邻近性"作为 C→L 回流的第 3 个特征（当前只有规则 + LLM 双引擎）。
- CPU 小样本可行性：**高**（只借结构思想与 bbox 特征，零训练；若真跑 LayoutLMv3 推理，CPU 单页约数秒且需 500MB 权重，不建议作为主路径）。
- 对应 v2 模块：`zhilv_qiling_v2/parsers/pdf_layout.py`、`zhilv_qiling_v2/parsers/docx_ooxml.py`

### A2 Donut：OCR-free 文档理解，直接输出结构化 JSON
- 文献：KIM G, HONG T, YIM M, et al. OCR-free Document Understanding Transformer[C]//Computer Vision – ECCV 2022. Cham: Springer, 2022. arXiv:2111.15664. https://arxiv.org/abs/2111.15664（官方实现 https://github.com/clovaai/donut ，MIT）
- 核心思想：彻底去掉 OCR 环节——Swin-B 视觉编码器 + BART 文本解码器，图像直接映射为 JSON 序列；配套 SynthDoG 合成文档生成器支持多语言域适配。论文报告 RVL-CDIP 95.30%（对比 LayoutLMv2 95.25%），推理 752 ms vs 1489 ms，参数 143M。
- 可迁移点：v2 的要素抽取输出改为**声明式 JSON schema**（`{"当事人":{"买方":str,"卖方":str},"金额":{"总额":num,"币种":str,"含税":bool}}`），使 (a) 规则通道与 LLM 通道都可按同一 schema 校验完整性（缺键即降级到另一通道）；(b) 借鉴 SynthDoG 思路，用 reportlab/Pillow 合成 300~800 张"合同首页 + 价款表"图像，训练 sklearn 版式分类器（三分类：纯文本页/表格页/扫描页），决定是否触发 OCR 与表格还原。**不训练 Donut 本身**。
- CPU 小样本可行性：**低**（Donut 端到端训练需 GPU；推理也要数百 ms/页且中文合同域未微调）→ **降级实现方案**：只迁移 JSON schema 契约 + 合成数据做版式分类器（sklearn RandomForest，特征=文本层字符数/页面积、行高方差、图块占比），工作量约 6 h。
- 对应 v2 模块：`zhilv_qiling_v2/parsers/ocr_pipeline.py`、`zhilv_qiling_v2/extract/elements_schema.py`

### A3 PP-OCR：可落地的超轻量 OCR 体系（检测+方向+识别+蒸馏）
- 文献：DU Y, LI C, YIN R, et al. PP-OCR: A Practical Ultra Lightweight OCR System[EB/OL]. arXiv:2009.09941, 2020. DOI: 10.48550/arXiv.2009.09941. https://arxiv.org/abs/2009.09941 （后续版本 PP-OCRv2: arXiv:2109.03144；PP-OCRv3: arXiv:2206.03001）
- 核心思想：DB 可微二值化文本检测 + 轻量方向分类器（处理 180° 翻转）+ CRNN 识别，配合"骨干裁剪-蒸馏-量化-数据增广"四步压缩策略，在 CPU 上跑到可用帧率；识别模型仅数 MB。
- 核心思想补充（工程要点）：文本检测阈值 `box_thresh`/`unclip_ratio` 直接决定小字号条款（合同脚注、附表）是否漏检，是精度的主要旋钮。
- 可迁移点：v2 现有 rapidocr 后端固定跑全页 OCR，代价高且有误识别污染文本层。改为**条件触发**：`parsers/ocr_pipeline.py` 先算 `text_density = len(page.get_text())/page_area`，仅当 `text_density < 0.002`（扫描页）或该页含表格线（`page.get_drawings()` 命中数 > 20）时启动 OCR；并把 `unclip_ratio` 从默认 1.6 调到 1.8 以救回被裁掉的小字号条款编号（"第 8.3 条"常被检测框切断）。
- CPU 小样本可行性：**高**（现成轮子，参数级改动，2~4 h）。
- 对应 v2 模块：`zhilv_qiling_v2/parsers/ocr_pipeline.py`

### A4 MinerU：版面检测→阅读顺序→内容列表的精确抽取流水线
- 文献：WANG B, XU C, ZHAO X, et al. MinerU: An Open-Source Solution for Precise Document Content Extraction[EB/OL]. arXiv:2409.18839, 2024. https://arxiv.org/abs/2409.18839 （官方仓库 https://github.com/opendatalab/MinerU ）
- 核心思想：把 PDF 抽取拆成"版面区域检测 → 阅读顺序排序 → 按区域类型（标题/正文/表格/公式）分别过专用模型 → 统一为中间内容列表 → 输出 Markdown/JSON"，实现"先定结构、再定内容"，避免整页线性切词的跨栏错序。
- 可迁移点：v2 的 `parsers/pdf_layout.py` 采用其**四阶段顺序**而非一次性 extract_text：区域分类只用启发式（字号 > 正文 1.2 倍→标题；y 间距 > 1.8 行高→段落边界；有框线→表格），表格区域用 PyMuPDF `find_tables()` 还原单元格再拼成"价款表"行文本（保留"单价/数量/税率"列头），使"金额"要素抽取从"跨栏乱序文本"变为"表格行结构化"，直接支撑 A2 的 schema 契约。
- CPU 小样本可行性：**中**（MinerU 原版模型推理 CPU 慢，一页 3~8 s；只借流水线设计与启发式规则则 8~12 h 可落地）。
- 对应 v2 模块：`zhilv_qiling_v2/parsers/pdf_layout.py`、`zhilv_qiling_v2/parsers/table_restore.py`

---

## B. 要素抽取 / NER / 信息抽取

### B1 W2NER：把 NER 统一为"词-词关系分类"
- 文献：LI J, FEI H, LIU J, et al. Unified Named Entity Recognition as Word-Word Relation Classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-22). 2022, 36(10): 10965-10973. arXiv:2112.10070. https://arxiv.org/abs/2112.10070
- 核心思想：构造 n×n 的**词对关系矩阵**，单元取值 ∈ {None, NNW（相邻词同实体）, THW-{B,M,E,S}（头尾关系）}；一次前向即统一处理扁平/重叠/不连续实体，无需复杂解码。
- 可迁移点：v2 现有"规则通道 + Qwen 通道"是**结果级加权**（谁的分高信谁），误差不可解释。改为**结构级融合**：把两条通道的 BIES 序列各自转成词对关系矩阵 M_rule、M_llm，按 `M = α·M_rule + (1-α)·M_llm`（α 由验证集搜索）逐单元融合再解码。合同里"甲方：A 公司（以下简称甲方）"这类**别名重叠**、"金额（含税 13%）"这类**嵌套**正好落在 W2NER 的表达能力内，而当前的线性 BIES 标注会丢一层。预期把规则单通道 0.880 的误差结构变成可分析项。
- CPU 小样本可行性：**中**（不训练 W2NER，只借 2D 标注表示做融合，纯 numpy，4~6 h；若训练则需 GPU）。
- 对应 v2 模块：`zhilv_qiling_v2/extract/ner_ensemble.py`

### B2 UIE：结构化抽取语言与 schema 驱动范式
- 文献：LU Y, LIU Q, DAI D, et al. Unified Structure Generation for Universal Information Extraction[C]//Proceedings of the 60th Annual Meeting of the ACL (ACL 2022). 2022: 5755-5772. arXiv:2203.12277. https://arxiv.org/abs/2203.12277 （项目页 https://universal-ie.github.io/ ）
- 核心思想：用 SEL（Structured Extraction Language）把抽取目标写成 `spot`（实体）/`assoc`（实体-实体关系）/`asoc`（实体-记录关系）三类结构化指令，把 IE 统一为"按 schema 生成结构化文本"，换 schema 即换任务。
- 可迁移点：v2 要素抽取的 schema 从硬编码字典升级为 `elements_schema.yaml`（`{字段名, 类型, spot/asoc, 正则提示, 单位归一}`），由 `extract/elements_schema.py` 编译成三样东西：(1) 规则通道的正则束；(2) Qwen 通道的 SEL 风格 prompt（离线时自动跳过）；(3) 校验器（未命中 spot 的必填字段 → 触发跨条款 asoc 搜索，如"总价"缺失时回查"价款"条款中的"合计"）。这是当前"契约驱动"设计的自然延伸，且不引入新依赖。
- CPU 小样本可行性：**高**（纯 schema 工程，8~10 h）。
- 对应 v2 模块：`zhilv_qiling_v2/extract/elements_schema.py`、`zhilv_qiling_v2/extract/elements_ner.py`

### B3 BiLSTM-CRF：序列标注的强基线
- 文献：HUANG Z, XU W, YU K. Bidirectional LSTM-CRF Models for Sequence Tagging[EB/OL]. arXiv:1508.01991, 2015. https://arxiv.org/abs/1508.01991
- 核心思想：BiLSTM 提取上下文特征，CRF 层建模**标签转移约束**（全局最优路径，而非逐位独立 softmax），显著改善 BIES 边界一致性；文中对比 LSTM / BI-LSTM / CRF / LSTM-CRF 四种组合。
- 可迁移点：v2 规则通道的偶发非法序列（如 `B-金额 → I-金额 → I-日期`）本质是缺转移约束。落地两条：**（1）不训练也能用**——在 `extract/rules_elements.py` 后加一层"合法转移掩码"（用 CRF 的转移矩阵形式写死：`I-X` 只能接 `I-X`/`E-X`/`O`，`E-X` 不可接 `I-X`），非法序列率→0；**（2）小样本训练**——字符级 BiLSTM-CRF（emb 64 / hidden 128 / 训练 50 epoch，torch 不可用时用 numpy+sklearn 的 `CRF` 等价实现或 `sklearn-crfsuite`），50~2000 份合同可训，作为第三通道补规则盲区（"不可抗力致合同目的不能实现"这类长实体）。
- CPU 小样本可行性：**高**（掩码版 3 h；训练版 12~16 h，CPU 上 2000 份×2000 字约 20~40 min/epoch，可控）。
- 对应 v2 模块：`zhilv_qiling_v2/extract/rules_elements.py`、`zhilv_qiling_v2/extract/crf_channel.py`

### B4 基于匹配策略与社区注意力机制的法律文书命名实体识别（中文核心）
- 文献：郭力华, 李旸, 王素格, 陈鑫, 符玉杰, 裴文生. 基于匹配策略和社区注意力机制的法律文书命名实体识别[J]. 中文信息学报, 2022, 36(2): 85-92. （山西大学；题录经中文信息学报官网/中国知网检索确认，PDF：http://jcip.cipsc.org.cn/ ）
- 核心思想：先用**最大正向匹配（FMM）**把法律术语词表对齐到字符序列并映射为 B/M/E/S 四种"词社区"，再用社区自注意力把词边界信息注入字符级编码器——即"词典词边界 = 先验结构"，缓解法律复合术语（"取保候审""不可抗力免责"）切分歧义。
- 可迁移点：v2 规则通道 F1 0.880 的主要误差源是**未登录法律复合词**导致边界抖动。落地：`extract/legal_lexicon.py` + FMM 生成字符级"社区标签"（4 类），作为 CRF/BiLSTM 通道的额外特征列（one-hot 4 维），或在纯规则路径上用"社区标签优先"覆盖字符规则（命中词表则整词成实体，不再按单字启发式）。这一步零 GPU，且与 B3 的转移掩码互补。
- CPU 小样本可行性：**高**（词典 + FMM，6~8 h；词典可从法条标题与合同模板高频词自动构建 2000~5000 词）。
- 对应 v2 模块：`zhilv_qiling_v2/extract/legal_lexicon.py`、`zhilv_qiling_v2/extract/rules_elements.py`

---

## C. 检索增强 / 混合检索 / 重排 / 查询改写

### C1 BM25 的概率相关性框架（含 BM25F 多流加权）
- 文献：ROBERTSON S, ZARAGOZA H. The Probabilistic Relevance Framework: BM25 and Beyond[J]. Foundations and Trends in Information Retrieval, 2009, 3(4): 333-389. DOI: 10.1561/1500000019. https://dl.acm.org/doi/10.1561/1500000019
- 核心思想：从二分独立模型推出 RSJ 权重，再由 2-Poisson 模型导出**饱和函数** `tf/(k1+tf)` 与**长度归一** `k1((1-b)+b·dl/avdl)`；第 3.6 节给出 BM25F 的**多流（multi-stream）线性加权**：对各字段流分别饱和后 `score = Σ_s w_s · tf_s/(k1+tf_s)`；第 5 节给出参数优化（k1,b 的 Robust Line Search、网格缓存）。
- 可迁移点：v2 现行 BM25 把"字 unigram + bigram"当两路词项混在一个索引里，信息流混同。改为 **BM25F 三流**：流 1=法条标题（权重 3.0）、流 2=要件/适用条件描述（2.0）、流 3=条文正文（1.0），各流独立饱和后线性加权；k1 取 1.2（标题流）/1.5（正文流），b=0.75，用 RLS 思路对 (k1, b, w_s) 做 2 轮坐标搜索，目标函数 Recall@5。法条是短文本，`b` 应偏小（0.5~0.75）以免长度归一过度惩罚长条文。
- CPU 小样本可行性：**高**（纯统计，倒排一次构建 < 1 s；参数搜索 6~15 组，8 h 内完成）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/bm25_index.py`

### C2 RRF：基于排名的融合优于 Condorcet 与单模型
- 文献：CORMACK G V, CLARKE C L A, BÜTTCHER S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods[C]//Proceedings of the 32nd International ACM SIGIR Conference (SIGIR '09). 2009: 158-159. DOI: 10.1145/1571941.1572114
- 核心思想：`RRFscore(d) = Σ_r 1/(k + rank_r(d))`，**只用排名不用分数**，故无需跨系统分数归一；作者以 k=60 为常数，在 TREC 与试点实验中 6/7 次优于 Condorcet/CombMNZ 与最佳单系统（p≈0.04）。
- 可迁移点：v2 已有 RRF，但 (1) k 未调（默认 60 未必适配"法条库仅千级、候选仅 top-50"的小规模场景，k 过大导致排名差异被压平）；(2) 两路等权。改为**加权 RRF**：`score = w_bm25/(k+r_bm25) + w_dense/(k+r_dense) + w_kg/(k+r_kg)`，网格 `k∈{5,10,20,30,60,100}`、`(w_bm25,w_dense,w_kg)∈{(1,1,0),(1,1.2,0.5),(0.8,1,0.5)}`，第三路来自 D4 的法条标签图兜底。目标 Recall@5 0.600→0.63~0.65，MRR@5 0.806→0.82。
- CPU 小样本可行性：**高**（只改融合公式 + 18~36 组网格，6 h）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/rrf_fuse.py`

### C3 BGE M3-Embedding：稠密+稀疏+多向量三合一
- 文献：CHEN J, XIAO S, ZHANG P, et al. BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation[EB/OL]. arXiv:2402.03216, 2024. https://arxiv.org/abs/2402.03216
- 核心思想：一个模型同时输出三种检索信号——**稠密向量**（CLS，语义）、**稀疏词权重**（lexical weight，可解释的 term 加权）、**多向量**（ColBERT 式 token 级 late interaction）；用自知识蒸馏把三路能力合并进同一编码器，支持 100+ 语言与最长 8192 token。
- 可迁移点：v2 目前只用了 dense 1024 维，等于扔掉模型自带的稀疏通道。落地：`retrieval/dense_index.py` 里把 BGE-M3 的 sparse 输出（`{token_id: weight}`）单独存成倒排，作为**介于 BM25 与 dense 之间的第三路**，与 BM25 做项级加权（`w = λ·bm25_idf + (1-λ)·m3_sparse`）后进 C2 的加权 RRF。稀疏通道在 CPU 上是查表 + 点积，成本≈0，却能显著缓解法条短文本的 dense 漂移（法条内容高度同质，"第五百七十七条"与"第五百七十八条"向量极近）。预期 Recall@5 +0.01~0.02，但**对 MRR 更有利**（把正确法条从 rank 3~5 提到 1~2）。
- CPU 小样本可行性：**高**（模型已在用，多存一份稀疏倒排，6 h）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/dense_index.py`、`zhilv_qiling_v2/retrieval/sparse_index.py`

### C4 LegalSearchLM：把法律检索重构为"法律要素生成"
- 文献：KIM C, LEE J, HWANG W. LegalSearchLM: Rethinking Legal Case Retrieval as Legal Elements Generation[C]//Proceedings of EMNLP 2025 (Main). arXiv:2505.23832. https://arxiv.org/abs/2505.23832
- 核心思想：不再用"查询-文档"双塔匹配，而是让模型**生成查询所对应的法律要素**（罪名/构成要件/行为/结果），再用 constrained decoding 保证生成合法，最后以要素为桥检索；在 LEGAR BENCH（411 个罪名、约 1.2M 候选）上较基线提升 6~20%，分布外（OOD）提升约 15%。
- 可迁移点：这是 v2 **Recall@5=0.600 的主攻方向**。等价改造（无需 LLM）：把"条款原文 → 法条检索 query"改写为**四元组模板** `(行为要件, 主体, 标的, 法律后果)`，槽位由 B 方向要素抽取（当事人/金额/标的）+ 条款功能段（C5）填充，再拼成受控 query 检索。例：`"乙方逾期交付货物超过30日，甲方有权解除合同并请求赔偿损失"` → `(逾期交付, 卖方, 货物/价款, 解除权+损害赔偿)` → 检索到《民法典》第 563 条（法定解除）、第 577 条（违约责任）、第 585 条（违约金）。因为 query 从"长句噪声"变成"要件信号"，BM25 与 dense 两路都能受益；同时 OOD（罕见合同类型）提升正对应挑战杯答辩时的"泛化"提问。
- CPU 小样本可行性：**中**（纯规则槽位填充 + 模板，16~20 h；不需要训练，但需要 50~100 条人工标注的"条款→正确法条"对来调模板，验证集留 30%）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/query_rewrite.py`

### C5 CEKLE：因果事件抽取驱动的关键法律要素感知检索（中文核心，EI）
- 文献：田萱, 谢格云, 吴志超. 基于因果事件抽取驱动关键法律要素感知的林法类案检索模型构建[J]. 农业机械学报, 2025, 56(8): 411-418, 446. DOI: 10.6041/j.issn.1000-1298.2025.08.038
- 核心思想：把法律案文先切成**"前言 / 事实 / 分析 / 判决 / 尾文"五段**，只对"事实"与"分析"两段做因果事件抽取，从而**定位关键法律要素出现的位置**，避免整篇等权编码导致要素信号被程序性套话淹没；在两个数据集上优于 SOTA 基线。
- 可迁移点：v2 已有 Clause 层，但所有条款等权进向量库。落地为**条款功能段标签**：`splitter/clause_section.py` 给每个 Clause 打 5 类标签（`主体信息 / 权利义务 / 价款支付 / 违约与解除 / 争议解决`，规则=标题关键词 + 首句动词模式），检索时只对 `权利义务+违约与解除` 段建向量（其余段落只进 BM25 作弱信号），并在 query 侧按"当前审查的条款功能"路由到对应段。这是"降噪"而非"改模型"，与 C4 的四元组改写叠加（改写提精度、分段降噪声）。
- CPU 小样本可行性：**高**（纯规则分段 + 索引重建，10~12 h）。
- 对应 v2 模块：`zhilv_qiling_v2/splitter/clause_section.py`、`zhilv_qiling_v2/retrieval/legal_search.py`

---

## D. 知识图谱 / 图推理 / 法规本体

### D1 GraphRAG：从局部到全局的图检索增强
- 文献：EDGE D, TRINH H, CHENG N, et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization[EB/OL]. arXiv:2404.16130, 2024. https://arxiv.org/abs/2404.16130
- 核心思想：先用 LLM 从语料抽实体-关系图 → 做**社区检测（Leiden）**得到层次化社区 → 为每个社区预生成"社区摘要" → 查询时按社区摘要做 **map-reduce** 汇总，从而回答"整体讲了什么"这类全局问题（传统向量 RAG 只能回答局部事实）。
- 可迁移点：v2 的跨条款审查（cross_review）本质上就是"全局问题"（"这份合同整体风险如何？哪些条款互相冲突？"），但当前只喂局部条款窗口。落地：`kg/graph_rag.py` 用 networkx 的 `louvain_communities` 对合同 KG（节点=Clause/Element/Party/LegalTag，边=C→L 回流边 + 要素共享边）做社区划分（CPU 上 200 节点毫秒级），每个社区用**模板摘要**（不调 LLM）：`"社区 #3：5 个条款均涉及交付义务，主体=乙方，缺'验收标准'条款，引用法条=第 596/598 条"`，作为 cross_review agent 的全局上下文前置注入。这既补了长合同"看不见全局"的短板，又完全离线可跑。
- CPU 小样本可行性：**高**（louvain + 模板摘要，12~16 h；LLM 摘要那一步降级为模板）。
- 对应 v2 模块：`zhilv_qiling_v2/kg/graph_rag.py`、`zhilv_qiling_v2/agents/cross_review.py`

### D2 TransE：多关系数据的平移嵌入
- 文献：BORDES A, USUNIER N, GARCIA-DURAN A, et al. Translating Embeddings for Modeling Multi-relational Data[C]//Advances in Neural Information Processing Systems 26 (NIPS 2013). 2013: 2787-2795. https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
- 核心思想：对三元组 (h, r, t) 施加 `h + r ≈ t`，用 margin ranking loss + **负采样（随机替换头/尾实体）** 训练，参数量极小（每实体/关系一个 k 维向量），可扩展到 1M 实体 / 17M 样本；缺点是只能表达 1-1 关系（后续 TransH/RotatE 解决）。
- 可迁移点：v2 合同 KG 规模小（单合同 50~200 节点、边 < 1000），numpy 手写 TransE（k=128，SGD，1000 epoch）在 CPU 上 < 30 s。用途不是"通用补全"，而是**缺失条款链接预测**：把"合同必须有但本条缺失"的关系（如 `(本合同, 应有, 争议解决)`）作为候选三元组打分，低分即"缺项风险"，直接扩充当前 27 标签体系（新增"缺项类"风险，例如缺"不可抗力"、缺"保密"、缺"通知送达"）。这是七分类风险之外最容易做出差异化的点。
- CPU 小样本可行性：**高**（numpy 实现 ~120 行，训练秒级，12 h 含工程与评测）。
- 对应 v2 模块：`zhilv_qiling_v2/kg/contract_graph.py`、`zhilv_qiling_v2/risk/risk_labels.py`

### D3 R-GCN：关系特定权重的图卷积与链接预测
- 文献：SCHLICHTKRULL M, KIPF T N, BLOEM P, et al. Modeling Relational Data with Graph Convolutional Networks[C]//The Semantic Web – ESWC 2018. Cham: Springer, 2018: 593-607. arXiv:1703.06103. https://arxiv.org/abs/1703.06103
- 核心思想：为每种关系 r 配一个独立的权重矩阵 W_r，并用**基分解（basis decomposition）**把 W_r 表示成少量基矩阵的线性组合以控制参数；用编码器累积多跳邻域证据、再与 DistMult 解码器联合训练，链接预测在 FB15k-237 上比纯解码器基线提升 **29.8%**。
- 可迁移点：v2 不训练 GNN（无 GPU、样本少）。落地为**关系路径特征 + 线性链接预测器**：对每个候选 (条款, 关系, 法条标签)，抽取 R-GCN 会聚合的等价特征——1 跳邻居类型计数、2 跳关系路径类型（`(条款)-[引用]->(法条)-[属于]->(章)`）、邻居度数、路径上的要素重叠度——共 20~40 维，喂 `sklearn.linear_model.LogisticRegression` 或 `HistGradientBoostingClassifier` 做链接存在性判别。这在数学上是 R-GCN 的"一阶近似 + 可解释特征"，CPU 上秒级训练，且能直接给出"为什么判为存在链接"（系数符号），对答辩友好。
- CPU 小样本可行性：**高**（特征工程 40 维，训练秒级，10~12 h）。
- 对应 v2 模块：`zhilv_qiling_v2/kg/legal_tag_graph.py`、`zhilv_qiling_v2/kg/path_features.py`

### D4 基于司法案例知识图谱的类案推荐（中文核心，FU-TransH）
- 文献：黄治纲, 谢新强, 邢铁军, 葛东, 蔡晨秋, 窦丽莉, 王天翊. 基于司法案例知识图谱的类案推荐[J]. 南京大学学报(自然科学版), 2021, 57(6): 1053-1063. DOI: 10.13232/j.cnki.jnju.2021.06.014
- 核心思想：构建刑事案例 KG（文本分类 + 信息抽取建图），用**实体对齐**把"当事人陈述三元组"与"裁判文书 KG"对齐（余弦相似度 + 人工干预），再用改进的 **FU-TransH**（针对 TransH 负采样不足，改进为过滤假阴性 + 按相似度采样）做案件向量化，最后以欧氏距离取 Top-m 相似案例；KG 作为辅助信息缓解冷启动与矩阵稀疏，准确率与召回率均优于代表算法。
- 可迁移点：两处直接可用。(1) **负采样策略**：v2 法条标签图做向量化时的负采样改为"同章/相邻条法条优先"的难度负采样（避免把"第 563 条"当"第 577 条"的负例而假阴性）；(2) **流程复用**：KGCR 的"陈词→三元组→实体对齐→向量检索"整条链路，正好对应 v2 的"条款→要素三元组→与法条标签图对齐→检索"，可作为 v2 检索链路的**第二意见通道**（与 BM25/dense 并列进 RRF），在 dense 冷启动（新合同类型）时提供兜底。
- CPU 小样本可行性：**高**（TransH/FU-TransH 手写 numpy，法条库千级规模训练分钟级，12~16 h）。
- 对应 v2 模块：`zhilv_qiling_v2/kg/legal_tag_graph.py`、`zhilv_qiling_v2/retrieval/kg_channel.py`

### D5 基于知识图谱的案件特征增强法律判决预测（中文核心）
- 文献：李紫阳, 张亚娟, 黄义雄, 王云鹤. 基于知识图谱的案件特征增强法律判决预测[J]. 计算机应用研究, 2024, 41(7): 2153-2159. DOI: 10.19734/j.issn.1001-3695.2023.11.0533
- 核心思想：BiGRU 编码全文语义 + 在 KG 向量空间中计算**类案相似度注意力**（用相似案例的特征加权当前表示）+ 融合结构化知识做链路预测；报告 MRR 与 Hit@1 相对基线提升约 1.5%。
- 可迁移点：v2 的 0~100 评分目前是规则加权 + LLM 打分的混合，缺"类案证据"。落地：在 `risk/scorer.py` 的特征表里增加 3 列 KG 相似度特征——(a) 本条款与"高风险的同类条款"（同功能段 + 同法条标签）的余弦相似度、(b) 该相似簇的历史风险等级均值、(c) 相似簇样本数（置信度权重）。用 `LogisticRegression` 或 `HistGradientBoostingRegressor` 在 100~500 条人工评分样本上校准，把"评分"从拍脑袋权重变成可解释线性模型。1.5% 这类小幅提升在 v2 这种小数据场景反而更可信。
- CPU 小样本可行性：**高**（特征 + sklearn 回归，10~14 h）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/scorer.py`

---

## E. 多智能体协同 / 仲裁 / 一致性

### E1 AutoGen：可对话多智能体与代码执行
- 文献：WU Q, BANSAL G, ZHANG J, et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation[EB/OL]. arXiv:2308.08155, 2023. https://arxiv.org/abs/2308.08155
- 核心思想：把 agent 抽象为"可定制、可对话、可执行代码"的实体，用**对话协议 + 群聊管理器（GroupChatManager）**编排多 agent 协作；支持人在回路（human-in-the-loop）与工具调用，强调"可组合、可回退"的会话式编排。
- 可迁移点：v2 用 LangGraph 固定了 5 节点 DAG + 有界回环，问题是**节点间只传自由文本**，回环时信息丢失。落地为 `agents/state.py` 的强类型消息契约：每个 agent 输出 `{artifact_type, payload, confidence, evidence_refs[]}`（`evidence_refs` 指向 SourceBlock id，可回溯到原文 bbox），编排层按 artifact_type 校验后才允许下一步；并在 `understanding` 节点后留一个 `interrupt` 位（人工意见注入），答辩演示"专家修正后重跑"很有说服力。
- CPU 小样本可行性：**高**（消息 schema + 校验器，8~12 h；纯结构化改造，不需 LLM 在线）。
- 对应 v2 模块：`zhilv_qiling_v2/agents/state.py`、`zhilv_qiling_v2/agents/graph.py`

### E2 Reflexion：语言智能体的言语强化学习
- 文献：SHINN N, CASSANO F, BERMAN E, et al. Reflexion: Language Agents with Verbal Reinforcement Learning[C]//Advances in Neural Information Processing Systems 36 (NeurIPS 2023). 2023. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
- 核心思想：Actor（执行）→ Evaluator（打分）→ Self-Reflection（生成自然语言反思）三模块；反思文本写入**情节记忆缓冲**（论文中记忆窗口 Ω 取 1~3），下一轮作为额外上下文注入，从而"用语言而非梯度"改进策略。
- 可迁移点：v2 的 LOOP 仲裁目前是"重跑一遍"，无学习。落地：在 `agents/arbitrate.py` 维护 `episode_buffer`（上限 3 条，对应 Ω=3），每次仲裁冲突后由 Evaluator 生成**1~2 句结构化反思**（模板：`"上轮冲突：条款审查判'高'、法条适用判'低'，原因是前者未核对第 585 条违约金上限 → 本轮强制核对"`），注入下一轮 prompt。回环上限仍为 3，但**同样轮数下收敛率提升**，且日志可解释（答辩可直接展示反思文本）。
- CPU 小样本可行性：**高**（缓冲 + 模板反思，8~10 h；离线时反思由模板生成，不依赖 LLM）。
- 对应 v2 模块：`zhilv_qiling_v2/agents/arbitrate.py`

### E3 多智能体辩论（Multiagent Debate）
- 文献：DU Y, LI S, TORRALBA A, et al. Improving Factuality and Reasoning in Language Models through Multiagent Debate[EB/OL]. arXiv:2305.14325, 2023（ICML 2024）. https://arxiv.org/abs/2305.14325
- 核心思想：多个模型实例对同一问题各自作答并**多轮互相阅读对方答案与理由**，最终收敛到共识；论文报告算术任务 67.0%→81.8%，GSM 77.0→85.0，三智能体辩论达 92.5%，且辩论收益随轮数增长后饱和（通常 2 轮足够）。
- 可迁移点：v2 的 LOOP 仲裁是"单一裁判"，容易继承上游偏差。落地为**双 agent 辩论 + 第 3 轮多数表决**：条款审查 agent 与法条适用 agent 对同一 Clause 给出不同风险等级时，各自输出"等级 + 依据法条 + 依据原文片段"，互相可见后重判 2 轮；若仍不一致，由裁判 agent 按"证据引用完整度 + 法条匹配度"投票。相关阈值（轮数=2、触发条件=等级差 ≥2 档）写进 `agents/consistency.py` 常量，便于消融。预期风险等级不一致率下降约 20%。
- CPU 小样本可行性：**高**（编排改造，10 h；离线时"辩论"由规则化证据比对实现，LLM 在线时自动切换为真实辩论）。
- 对应 v2 模块：`zhilv_qiling_v2/agents/consistency.py`、`zhilv_qiling_v2/agents/arbitrate.py`

### E4 MetaGPT：把 SOP 编码进多智能体协作
- 文献：HONG S, ZHUGE M, CHEN J, et al. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework[C]//International Conference on Learning Representations (ICLR 2024). arXiv:2308.00352. https://arxiv.org/abs/2308.00352
- 核心思想：把人类**标准作业程序（SOP）**编码为 prompt 序列，并用"流水线 + 角色专精 + 结构化产物（PRD/设计/接口定义）+ 发布-订阅消息池"约束协作，明确要求 agent 产出**结构化文档而非自由对话**，从而抑制级联幻觉（报告 MBPP 绝对提升 5.4%）。
- 可迁移点：v2 五 agent 当前交接物是自然语言段落，"条款审查"的结论到"裁判"手里已被改写两次。落地：为每条边定义**结构化 artifact 合约**（`ClauseRisk{clause_id, level∈{低,中,高}, tags[27], evidence_block_ids[], legal_refs[]}`、`ElementSet{party[], amount[], date[], subject[]}`、`LegalRef{article_no, title, score, channel}`），编排器做 schema 校验 + 缺失字段回退（缺 evidence_block_ids 直接判该结论无效并重跑）。这条与 E1 互补：E1 给消息层协议，E4 给产物层 SOP。
- CPU 小样本可行性：**高**（schema + 校验 + 回退，12 h）。
- 对应 v2 模块：`zhilv_qiling_v2/agents/graph.py`、`zhilv_qiling_v2/agents/fuse.py`

---

## F. 风险评估 / 置信度校准 / 不确定性

### F1 现代神经网络的校准（ECE/MCE + Temperature Scaling）
- 文献：GUO C, PLEISS G, SUN Y, et al. On Calibration of Modern Neural Networks[C]//Proceedings of the 34th International Conference on Machine Learning (ICML 2017). PMLR 70: 1321-1330. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
- 核心思想：定义 **ECE / MCE** 度量，发现现代网络"过度自信"；系统比较 histogram binning、isotonic regression、**temperature scaling**（单参数 T，`softmax(z/T)`）等方法，结论是 temperature scaling 在保持精度不变的前提下校准效果最好且最简单（仅 1 个参数，不易在小验证集上过拟合）。
- 可迁移点：v2 的 0~100 评分与七分类风险概率从未被校准，"85 分"到底对应多大概率不可解释。落地 `risk/calibration.py`：在 100~500 条人工标注上拟合七分类的 T（`scipy.optimize.minimize_scalar` 最小化 NLL，或网格 T∈[0.5,3.0] 步长 0.05），报告校准前后 ECE（分 10 箱）；评分与概率的映射用保序回归（isotonic）单调对齐，保证"分数越高风险越大"不被破坏。这是**答辩加分项**（"我们的 92.3 分对应的实际高风险概率是 0.87，ECE=0.04"）。
- CPU 小样本可行性：**高**（单参数拟合，秒级，5~6 h 含 ECE 报告与可靠性曲线图）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/calibration.py`

### F2 Platt scaling：SVM 输出的概率化
- 文献：PLATT J C. Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods[M]//SMOLA A J, BARTLETT P, SCHÖLKOPF B, et al. Advances in Large Margin Classifiers. Cambridge, MA: MIT Press, 1999: 61-74. （作者版 PDF：https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf ）
- 核心思想：用 `P(y=1|f) = 1/(1+exp(A·f+B))` 把任意判别分数 f 映射为概率，A、B 用**带平滑的目标**拟合（正负样本数先做 `(N+ +1)/(N+ +2)` 修正），从而避免小样本下 sigmoid 过拟合到 0/1。
- 可迁移点：v2 的 27 个风险标签是**多标签**，每个标签的判别分数（规则命中强度、dense 相似度）都需单独概率化才能做阈值决策与排序。落地：`sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv=3)` 包住现有多标签分类器（OneVsRest + LogisticRegression），输出每标签校准概率；对样本极少的标签（<20 正例）启用 Platt 的平滑修正，避免概率被压到 0/1 两端导致阈值失效。
- CPU 小样本可行性：**高**（sklearn 一行调用级，4~6 h）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/calibration.py`、`zhilv_qiling_v2/risk/risk_labels.py`

### F3 共形预测（Conformal Prediction）导论
- 文献：ANGELOPOULOS A N, BATES S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification[EB/OL]. arXiv:2107.07511, 2021. https://arxiv.org/abs/2107.07511
- 核心思想：**split conformal**——留出校准集算非一致性分数 s_i，取 `q̂ = ⌈(n+1)(1-α)⌉/n` 的经验分位数，对新样本输出**预测集合** `C(x) = {y : s(x,y) ≤ q̂}`，在可交换假设下严格保证 `P(Y ∈ C(X)) ≥ 1-α`，且对任何模型（哪怕是很差的规则系统）都成立。
- 可迁移点：v2 的风险标签输出是"单点判断"，漏检无法量化。落地 `risk/conformal.py`（约 60 行）：对每个 Clause 的 27 标签，用非一致性分数 `s = 1 - p_calibrated` 构造预测集，α=0.1 时输出"高风险标签集合 + 保证 90% 覆盖率"，集合大小本身即**不确定性指标**（集合大 = 模型犹豫 → 转人工复核）。这条最契合挑战杯"可信 AI"叙事，且**是唯一能在无 GPU、小样本下给出统计保证的方法**。
- CPU 小样本可行性：**高**（numpy 60 行，6~8 h；需 ≥100 条标注做校准集）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/conformal.py`

### F4 CLAUDETTE：在线服务条款中不公平条款的自动检测
- 文献：LIPPI M, PAŁKA P, CONTISSA G, et al. CLAUDETTE: An Automated Detector of Potentially Unfair Clauses in Online Terms of Service[J]. Artificial Intelligence and Law, 2019, 27(2): 117-139. DOI: 10.1007/s10506-019-09243-2 （题录经 LegalBench 官方仓库 bibtex 与 Springer 期刊页确认）
- 核心思想：把"不公平条款"检测建模为**从句级二分类 + 规则后处理**，标签体系直接来自欧盟不公平条款清单（单方变更权、单方终止权、责任限制、管辖法院指定、仲裁强制等），并给出可解释的"哪一句、属于哪一类不公平"输出。
- 可迁移点：v2 的七分类风险与 27 标签是自建的，缺外部效度。落地两件事：(1) **标签对齐表**——把 v2 的 27 标签映射到 CLAUDETTE 的不公平类别（`单方解除 ↔ unilateral termination`、`责任限制 ↔ limitation of liability`、`管辖约定 ↔ choice of law/jurisdiction`、`强制仲裁 ↔ arbitration`），映射不上的标签标为"中式合同特有"（如"盖章生效""发票条款"），反而成为差异化论证；(2) **句级判定粒度**——v2 目前按 Clause 判，改为"Clause 内句级打分 + 取最大句"作为条款级风险，可解释性更强（指出触发风险的原句）。
- CPU 小样本可行性：**高**（映射表 + 句级打分聚合，6~8 h）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/risk_labels.py`、`zhilv_qiling_v2/agents/clause_review.py`

---

## G. 中文核心期刊专项（法律文本 / 知识图谱 / 法条推荐方向）

> 本节 4 条为"中文核心专项"新增；另有 4 条中文核心已分布在前述方向：郭力华（中文信息学报 2022，B4）、田萱（农业机械学报 2025，C5）、黄治纲（南京大学学报 2021，D4）、李紫阳（计算机应用研究 2024，D5）。合计中文文献 8 条。

### G1 基于深度语义匹配的法律条文推荐方法（中文核心，软件学报）
- 文献：李琳, 段围, 周栋, 袁景凌. 基于深度语义匹配的法律条文推荐方法[J]. 软件学报, 2022, 33(7): 2618-2632. http://www.jos.org.cn/1000-9825/6287.htm
- 核心思想：DeepLawRec——BERT 对"案例描述（截断 270 词）"与"法条（截断 30 词）"分别做**句级编码**，用 **CNN 捕获局部语义匹配模式**（n-gram 级交互），再用 **XGBoost 回归树做二次选择**（超参 gamma=0.1、max_depth=8、min_child_weight=3）；在 CAIL2018（训练 114824 / 验证 14293 / 测试 23593，183 条法条）上 Precision@1 从 0.638 提升到 0.913、Accuracy 0.913。
- 可迁移点：这是 v2 **CPU-only 条件下最该照抄的架构**——"双塔召回 + 树模型精排"。落地 `retrieval/rerank_ltr.py`：第一层 BM25F + BGE-M3 + KG 三路 RRF 取 top-50；第二层构造 6~10 维特征（BM25 分、dense 余弦、字面 Jaccard、法条标题命中数、要件覆盖度、KG 路径分、条款功能段与法条章节一致性），用 `xgboost`/`sklearn.HistGradientBoosting` 的回归或 `LambdaMART`（`lightgbm` 的 `lambdarank` objective）训练 top-5 精排。预期 Recall@5 0.600→0.70+、MRR@5 0.806→0.85+，因为 0.600 的瓶颈在"top-50 内排序"而非"召回不到"。
- CPU 小样本可行性：**高**（树模型训练秒级；50~2000 份合同足以训 10 维特征的精排器，20 h 含标注）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/rerank_ltr.py`

### G2 面向法律裁判文书的法条推荐方法（中文核心，计算机科学）
- 文献：张虎, 王鑫, 王冲, 等. 面向法律裁判文书的法条推荐方法[J]. 计算机科学, 2019, 46(9): 211-215. （CNKI: https://www.cnki.com.cn/Article/CJFDTOTAL-JSJA201909033.htm ；题录经《软件学报》2022, 33(7) 参考文献 [4] 交叉确认）
- 核心思想：面向裁判文书的法条推荐，采用多模型/多特征融合思路提升推荐准确率（该文被后续法条推荐工作（李琳 2022、黄治纲 2021）反复引为多模型融合基线）。
- 可迁移点：v2 已有 RRF 融合，可加**置信度阈值拒识 + KG 兜底**：当 top-1 与 top-2 的融合分差 < δ（如 δ=0.02）时判为"低置信"，自动转 D4 的标签图通道重排，并降低该法条的引用权重（在裁判 agent 的输入里标注"待确认"）。这比硬取 top-1 更稳，且天然产出"不确定性"指标（配合 F3 的 conformal 集）。
- CPU 小样本可行性：**高**（阈值 + 路由，4~6 h）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/legal_search.py`

### G3 基于多任务学习的民事案件判决预测方法（中文核心，计算机应用与软件）
- 文献：余连玮, 马志柔, 刘杰, 叶丹. 基于多任务学习的民事案件判决预测方法[J]. 计算机应用与软件, 2024, 41(1). DOI: 10.3969/j.issn.1000-386x.2024.01.004 （页码待核；起始页 19 已见 PDF 版式页眉，全文预览：https://www.shcas.net/cn/article/pdf/preview/10.3969/j.issn.1000-386x.2024.01.004.pdf ）
- 核心思想：针对民事案件"法条组合多样化"，用**多任务学习**联合预测"案由纠纷"与"相关法条"，采用多 CNN 融合 + 阈值设定策略利用案由与法条间的依赖关系；在 10 万篇民事文书上验证优于传统单任务模型。
- 可迁移点：v2 的"条款类型分类"与"法条检索"目前是两条独立链路，但二者共享同一份条款表示。落地**共享编码 + 双头**：`retrieval/multitask.py` 用同一个 clause 特征（BM25 词项向量 + dense 向量拼接降维到 256 维，用 TruncatedSVD）同时喂两个 `LogisticRegression`（头 1：条款功能段五分类；头 2：法条 top-k 相关性），联合训练让功能段信息正则化检索头；并用**按标签频率设定的阈值**处理 27 标签的长尾不平衡（每标签阈值 = 使该标签在验证集 F1 最大的分位点）。
- CPU 小样本可行性：**高**（sklearn 多输出 + 阈值搜索，12~16 h）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/multitask.py`、`zhilv_qiling_v2/risk/risk_labels.py`

### G4 基于多粒度语义交互的无监督法律裁判文书检索（中文期刊，零标注方案）
- 文献：周献杭, 申妍燕. 基于多粒度语义交互的无监督法律裁判文书检索[J]. 集成技术, 2022, 11(2). （页码待核；起始页 57 已见 PDF 版式页眉，全文预览：https://jcjs.siat.ac.cn/cn/article/pdf/preview/10.12146/j.issn.2095-3135.20210413001.pdf ）
- 核心思想：针对法律文本"篇幅长、结构复杂、专业性强"且**标注昂贵**的问题，提出**无监督**多粒度匹配——分别从**法律概念、词语、词组**三个粒度做无监督文本匹配，规避训练数据依赖导致的冷启动；在裁判文书数据集上 MAP、MRR、NDCG@10 均显著优于基线（基座为 8000 万份文书库、人工标注 23196 份作评测）。
- 可迁移点：v2 完全没有标注数据的现实下，这是最匹配的一条。落地**第三粒度通道** `retrieval/concept_index.py`：从法条标题与条文首句抽取"法律概念词"（如"解除权""违约金""不可抗力""表见代理"，构建 300~800 概念词典），建立"概念 → 法条"倒排与"概念 → 条款"倒排，检索时做概念级匹配并作为 RRF 的第 4 路。三粒度（字面 BM25 / 概念倒排 / dense 语义）互补，且概念通道**零训练、可解释**（能直接告诉评委"因为条款含'解除权'概念，命中 563 条"）。
- CPU 小样本可行性：**高**（词典 + 倒排，10 h；无需任何标注）。
- 对应 v2 模块：`zhilv_qiling_v2/retrieval/concept_index.py`

---

## H. 数据集与评测基准

### H1 LawBench：中文法律知识的多层级基准
- 文献：FEI Z, SHEN X, ZHU D, et al. LawBench: Benchmarking Legal Knowledge of Large Language Models[C]//Proceedings of EMNLP 2024 (Main). arXiv:2309.16289. https://arxiv.org/abs/2309.16289 （ACL Anthology: 2024.emnlp-main.452；仓库 https://github.com/open-compass/LawBench ）
- 核心思想：按"记忆（法条背诵）→ 理解（要素识别/分类）→ 应用（判决预测/法条适用）"三层认知能力组织 20 个任务，覆盖 51 个中文大模型；提供统一 prompt 与评测脚本。
- 可迁移点：v2 的评测脚本 `eval/bench.py` 直接对齐 LawBench 的**任务定义与指标口径**（法条检索用 Recall@k/MRR，要素抽取用 macro-F1，条款分类用 macro-F1/micro-F1），使"F1 0.992、Recall@5 0.600"这些数字有外部可比口径；并挑 3 个子任务（法条检索、要素识别、法条适用）做 zero-shot 对照，答辩时"我们的离线小模型在 × 子任务上接近 7B 在线模型的 ×%"是可量化的加分项。
- CPU 小样本可行性：**高**（仅用其任务定义与脚本，14 h 含适配）。
- 对应 v2 模块：`zhilv_qiling_v2/eval/bench.py`

### H2 LegalBench：协作构建的法律推理基准
- 文献：GUHA N, NYARKO J, HO D E, et al. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models[C]//Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track. arXiv:2308.11462. https://arxiv.org/abs/2308.11462 （仓库 https://github.com/HazyResearch/legalbench ）
- 核心思想：由法律专家协作定义 162 个任务，按 IRAC 框架（Issue / Rule / Application / Conclusion）分类，强调"任务由法律人定义、指标单一化"；尤其包含 `rule_recall`（规则召回）、`rule_application`（规则适用）等直接对应合同审查的子任务。
- 可迁移点：v2 的条款审查实质是"规则适用"——从法条抽规则、把事实归入规则。落地：用 LegalBench 的 `rule_recall` / `rule_application` 任务措辞改造 v2 的审查 prompt 与评测样例（"该条款是否满足第 585 条违约金上限规则：是/否/信息不足"），把"三态输出（是/否/信息不足）"引入 27 标签判定，直接改善"证据不足却硬判"的过判问题（与 F3 的 conformal 集一致）。
- CPU 小样本可行性：**高**（只借任务定义与三态输出，8~10 h）。
- 对应 v2 模块：`zhilv_qiling_v2/eval/bench.py`、`zhilv_qiling_v2/agents/clause_review.py`

### H3 CUAD：合同审查的专家标注数据集（41 类条款 / 13k+ 标注）
- 文献：HENDRYCKS D, BURNS C, CHEN A, et al. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review[C]//Advances in Neural Information Processing Systems 35 (NeurIPS 2021) Datasets and Benchmarks Track. arXiv:2103.06268. https://arxiv.org/abs/2103.06268 （数据与代码 https://github.com/TheAtticusProject/cuad ，CC BY 4.0）
- 核心思想：由数十位律师对 **510 份合同**（SEC EDGAR 采集，25 种合同类型）做 **13,000+ 条**专家标注，覆盖 **41 个"审查关键条款"类别**（如 `Document Name`、`Effective Date`、`Cap On Liability`、`Non-Compete`、`Termination For Convenience`），任务=在合同中高亮需人工复核的关键片段；结论是 Transformer 表现尚浅，模型设计与数据规模影响巨大，仍有大幅提升空间。
- 可迁移点：v2 的 27 标签缺外部标注。落地：(1) 建 **27↔41 映射表**（`责任上限 ↔ Cap On Liability`、`竞业限制 ↔ Non-Compete`、`任意解除 ↔ Termination For Convenience`、`最惠国 ↔ Most Favored Nation`…），映射覆盖率作为"标签体系完备性"指标写进论文；(2) 用 CUAD 的 41 类做**弱监督预训练语料**，把 510 份英文合同经机器翻译/术语替换为中文模板句（仅用于扩增规则词表与负例，不训练大模型），把可用样本从 50~200 份扩到 2000 份量级的"条款模式多样性"。
- CPU 小样本可行性：**中**（映射表 6 h；扩增与词表挖掘 12~16 h；直接训练抽取模型不可行→降级为词表/模式挖掘）。
- 对应 v2 模块：`zhilv_qiling_v2/risk/risk_labels.py`、`zhilv_qiling_v2/extract/legal_lexicon.py`

### H4 ContractNLI：合同文档级自然语言推理
- 文献：KOREEDA Y, MANNING C D. ContractNLI: A Dataset for Document-Level Natural Language Inference for Contracts[C]//Findings of EMNLP 2021. arXiv:2110.01799. https://arxiv.org/abs/2110.01799 （数据 https://stanfordnlp.github.io/contract-nli/ ，CC BY 4.0）
- 核心思想：把合同审查表述为**文档级 NLI**：给定 17 条标准化"假设"（如"保密信息在披露后 5 年内仍受保护"），判断合同整体是 entail / contradict / **not mentioned**；not mentioned 这一态专门捕获"合同未约定"的风险，是抽取式方法做不到的。
- 可迁移点：v2 的跨条款审查与"缺失条款"检测可直接套这个框架：把 27 个风险标签改写成 27 条**假设句**（`"本合同约定了不可抗力免责"`、`"违约金不超过合同总额 30%"`），由跨条款 agent 对全合同输出三态；`not mentioned` 即"缺项风险"（与 D2 的 TransE 链接预测互为验证），`contradict` 即"条款冲突"（当前 27 标签体系里最难判的一类）。这给 v2 一个清晰的、可评测的跨条款审查形式化定义。
- CPU 小样本可行性：**高**（假设句模板 + 三态判定规则，10~12 h；离线时用"关键词存在性 + 要素匹配"近似三态，在线 LLM 时用真 NLI）。
- 对应 v2 模块：`zhilv_qiling_v2/agents/cross_review.py`、`zhilv_qiling_v2/risk/risk_labels.py`

---

## 附录：可裁剪条目（若须把 34 条压到 28 条）

按"对 v2 升级的直接收益 / 不可替代性"从低到高，建议依次删除以下 6 条（删除后仍满足 A~H 各 ≥3、中文 6 条，低于任务要求的中文 ≥8，故**只有在评审强制 28 条上限且放宽中文配额时**才裁剪）：

1. A2 Donut（OCR-free，CPU 训练不可行，仅借 schema 思想，可由 B2 UIE 的 schema 部分替代）
2. A4 MinerU（流水线思想可由 A1 的 bbox + 启发式覆盖）
3. E4 MetaGPT（与 E1 AutoGen 的消息契约部分重叠）
4. D3 R-GCN（与 D2 TransE 同属 KG 嵌入，路径特征可用 D2/D4 覆盖）
5. H2 LegalBench（三态输出可由 H4 ContractNLI 的 entail/contradict/not-mentioned 覆盖）
6. F2 Platt scaling（与 F1 的 temperature scaling 同属概率校准，二选一即可）

**不可删**（构成升级主线）：C1（BM25F）、C2（加权 RRF）、C4（要素四元组改写）、C5（条款功能段）、G1（两阶段树模型精排）、G4（概念粒度零标注通道）、F1（校准）、F3（conformal）、E3（辩论仲裁）、D1（社区摘要）、D5（KG 相似度特征）、H3（CUAD 标签映射）。
