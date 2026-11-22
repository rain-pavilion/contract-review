"""智律契灵 v2 —— 基于图结构引导的多智能体政府采购合同智能审查系统（迭代升级版）。

与 v1（`contract_review/`）的关系
--------------------------------
v1 是可运行的基线，本包是**独立目录下的重写迭代版**：不 import、不修改 v1 的任何
模块，仅复用其数据资产（`data/legal_index`、`data/laws`、`data/generated_contracts`、
`data/gov_contracts`）。两者可以并存、可以对比评测。

v2 的设计升级（每条都能在 docs/UPGRADE.md 里找到依据、实验与回退方案）
----------------------------------------------------------------------
1. 零重依赖内核：核心链路只用标准库 + numpy/pandas/lxml；`pymupdf`、`scikit-learn`、
   `requests`、`rapidocr` 均为**可选加速器**，缺失时自动降级并在 `capabilities()`
   里如实报告。v1 硬依赖 requests/pymupdf/langgraph/qdrant，换机器就跑不起来。
2. 版面感知解析：新增 XY-cut 阅读序、页眉页脚剔除、表格行识别、双栏合同处理。
3. OCR 通道：可插拔后端 + sqlite 缓存 + 置信度过滤 + 文字层弱监督校准。
4. 切分：规则引擎 + **全局动态规划分段**（替代局部贪心）+ **统计边界分类器兜底**
   （替代"LLM 不可用就只剩规则"）。
5. 检索（v1 明确短板 Recall@5=0.600）：BM25 + 字符 n-gram TF-IDF + LSA 伪稠密向量
   + RRF + **查询改写模板** + **学习型重排（LTR）** + 法条标签图 PPR 传播。
6. 编排：自研轻量 DAG 执行器（拓扑调度 + 线程并行 + 有界回环 + 可重放 trace），
   取代 langgraph 硬依赖。
7. 仲裁与置信度：证据加权投票 + isotonic/Platt 校准，输出可解释的 review_required。
8. 训练：四个 CPU 可训练的组件（NER 结构化感知机 / 风险分类器 / 检索 LTR /
   置信度校准），配套统一评测脚本与 v1-v2 对照报告。
9. 爬虫：多源（天津开放数据 API + 政府采购网）+ sqlite 断点续爬 + 附件解析 +
   OCR 队列，v1 只支持北京单源、无状态。

快速开始
--------
    python -m zhilv_qiling_v2.cli doctor              # 能力自检
    python -m zhilv_qiling_v2.cli review samples/tech-development-contract.txt
    python -m zhilv_qiling_v2.cli crawl --source tj --limit 20
    python -m zhilv_qiling_v2.cli train all
    python -m zhilv_qiling_v2.cli serve --port 8601
"""

from .version import __version__, VERSION_TAG

__all__ = ["__version__", "VERSION_TAG"]
