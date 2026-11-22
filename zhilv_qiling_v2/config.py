"""运行配置与**能力探测**。

设计原则
--------
v1 在 `settings.py` 里直连 `.env`，并且整条链路硬依赖 requests / pymupdf / langgraph /
qdrant —— 换一台机器（比如评审现场）就整套起不来。v2 反过来：

* 核心链路只用标准库 + numpy/pandas/lxml，**必然可跑**；
* 每个可选能力（PDF 版面、OCR、sklearn、LLM、HTTP 抓取）都在这里探测一次，探测结果
  写进 `Capabilities`，任何下游模块只允许问 `caps.has("ocr")`，不允许自己 try-import；
* 缺能力时的行为写死在 `DEGRADE_NOTES` 里，命令行 `cli doctor` 会原样打印，
  报告里也照这个口径写（不吹能力，不藏短板）。
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
WORKSPACE = PKG_DIR.parent
DATA_DIR = PKG_DIR / "data"
MODEL_DIR = PKG_DIR / "models"
REPORT_DIR = PKG_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
DOCS_DIR = PKG_DIR / "docs"

#: v1 的数据资产（**只读**）。v2 自己产生的数据一律写在自己的目录下。
V1_ROOT = WORKSPACE
V1_LEGAL_INDEX = V1_ROOT / "data" / "legal_index"
V1_LAWS = V1_ROOT / "data" / "laws"
V1_GENERATED = V1_ROOT / "data" / "generated_contracts"
V1_GOV = V1_ROOT / "data" / "gov_contracts"
V1_STATUTES = WORKSPACE / "data" / "法条结构化数据.jsonl"
"""法条结构化数据（v1 语料，v2 只读复用，不复制）。"""
V1_SAMPLES = V1_ROOT / "samples"


def ensure_dirs() -> None:
    """确保 v2 自己的写目录存在。幂等。"""
    for path in (DATA_DIR, MODEL_DIR, REPORT_DIR, CACHE_DIR, DOCS_DIR,
                 DATA_DIR / "ocr", DATA_DIR / "crawl", DATA_DIR / "index"):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 能力探测
# ---------------------------------------------------------------------------

#: 缺失时各模块的真实降级行为。cli doctor 与实验报告都引用这张表。
DEGRADE_NOTES: dict[str, str] = {
    "pymupdf": "PDF 走内置纯 Python 文本抽取（仅覆盖未加密、常见字体编码）；"
               "拿不到文字层的页面转入 OCR 队列，`--allow-ocr` 时可继续。",
    "sklearn": "训练器改用内置 numpy 实现（logistic 回归用 IRLS/梯度下降，"
               "LTR 用 pairwise logistic），指标口径不变。",
    "requests": "HTTP 用标准库 urllib.request，自动带重试与限速。",
    "ocr": "扫描件只登记进 OCR 队列并如实标注 `ocr_pending`，不臆造文本。",
    "llm": "多智能体节点的 LLM 通道整段跳过，改用规则 + 统计模型 + 图谱证据，"
           "报告里 `llm_available=false`。",
    "numpy": "向量检索退化为 BM25 + 字符重合度；LTR 退化为手写权重。",
}


@dataclass(frozen=True)
class Capabilities:
    """一次进程内只探测一次的能力快照。"""

    pymupdf: bool = False
    sklearn: bool = False
    requests: bool = False
    numpy: bool = False
    pandas: bool = False
    lxml: bool = False
    docx: bool = False
    ocr_rapid: bool = False
    ocr_paddle: bool = False
    ocr_tesseract: bool = False
    llm: bool = False
    python: str = ""
    notes: tuple[str, ...] = ()

    def has(self, name: str) -> bool:
        if name == "ocr":
            return self.ocr_rapid or self.ocr_paddle or self.ocr_tesseract
        if name == "llm":
            return self.llm
        return bool(getattr(self, name, False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def missing(self) -> list[str]:
        keys = ("pymupdf", "sklearn", "requests", "numpy", "ocr", "llm")
        return [k for k in keys if not self.has(k)]

    def degrade_report(self) -> list[str]:
        return [f"{k}: {DEGRADE_NOTES[k]}" for k in self.missing() if k in DEGRADE_NOTES]


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # pragma: no cover - 环境相关
        return False


_CAPS: Capabilities | None = None


def _llm_configured() -> bool:
    """LLM 通道是否可用：必须显式给 endpoint，且不做联网探测（探测太慢也不稳）。"""
    endpoint = os.environ.get("ZLQL_LLM_ENDPOINT", "").strip()
    return bool(endpoint)


def capabilities(refresh: bool = False) -> Capabilities:
    """探测并缓存能力快照。"""
    global _CAPS
    if _CAPS is not None and not refresh:
        return _CAPS

    ocr_rapid = _can_import("rapidocr_onnxruntime")
    ocr_paddle = _can_import("paddleocr")
    ocr_tess = bool(shutil.which("tesseract"))
    notes: list[str] = []
    caps = Capabilities(
        pymupdf=_can_import("pymupdf") or _can_import("fitz"),
        sklearn=_can_import("sklearn"),
        requests=_can_import("requests"),
        numpy=_can_import("numpy"),
        pandas=_can_import("pandas"),
        lxml=_can_import("lxml"),
        docx=_can_import("docx"),
        ocr_rapid=ocr_rapid,
        ocr_paddle=ocr_paddle,
        ocr_tesseract=ocr_tess,
        llm=_llm_configured(),
        python=sys.version.split()[0],
    )
    if not caps.lxml:
        notes.append("lxml 缺失：DOCX/HTML 解析退化为纯正则，表格结构可能丢失")
    _CAPS = Capabilities(**{**caps.to_dict(), "notes": tuple(notes)})
    return _CAPS


# ---------------------------------------------------------------------------
# 运行配置
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """一次审查运行的完整参数。

    默认值按"CPU、无网、无模型"这条最坏路径定，保证开箱能跑出真实结果。
    """

    # --- 切分 ---
    boundary_min_confidence: float = 0.55
    """低于该置信度的规则边界会被标 review_required，并交给统计边界分类器复核。"""

    use_dp_segmentation: bool = True
    """True = 全局动态规划分段（默认）；False = v1 的局部贪心（用于对照实验）。"""

    dp_lambda_len: float = 0.35
    """DP 目标里对"条款长度偏离先验"的惩罚系数。"""

    dp_len_prior: float = 260.0
    """条款正文字符数先验，来自 v1 数据集的统计中位数。"""

    # --- 要素抽取 ---
    ner_enable_rules: bool = True
    ner_enable_model: bool = True
    # 有 GPU 训练产物时优先用 BERT-CRF（宏 F1 0.5348，感知机只有 0.0179）
    ner_use_bert: bool = True
    """字符级感知机通道默认关闭。

    实测（`reports/training_report.md` §2）：88 个训练窗口、15 轮之后，留出集上
    PARTY 的 P=0.038 / R=0.727 / F1=0.072，四类宏平均 F1=0.027，而规则通道宏平均
    0.77。纯 CPU、无预训练字符表示的条件下跨度的首尾边界学不出来，直接启用会把
    大量非实体串当当事人灌进要素表。权重照常训练并落盘，用于论文里的消融对照；
    GPU 环境换 BERT-CRF 之后再打开（`ZLQL_NER_MODEL=1` 或本字段置 True）。"""
    ner_min_confidence: float = 0.45

    # --- 检索 ---
    statutes_path: str = ""
    """本次审查专属的法条库；空值保持原有默认库，不修改其他审查的配置。"""
    contract_packages_path: str = ""
    """真实合同包存储目录；空值使用合同包模块的默认路径。"""
    top_k: int = 8
    retrieve_candidate_limit: int = 120
    rrf_k: int = 60
    use_lsa: bool = True
    lsa_dim: int = 128
    use_query_rewrite: bool = True
    use_rerank: bool = True
    tag_ppr_alpha: float = 0.15
    tag_ppr_iters: int = 3

    # --- 图谱 ---
    use_contract_graph: bool = True
    graph_signal_weight: float = 0.25

    # --- 多智能体 ---
    max_legal_retries: int = 1
    max_judge_retries: int = 1
    max_evidence_rounds: int = 1
    max_arbitration_rounds: int = 1
    arbitration_material_gap: float = 0.22
    """两方置信度差超过该值才认为冲突"实质"，否则按一致处理（v1 是死阈值比较）。"""
    parallel_agents: bool = True
    trace_enabled: bool = True

    # --- 决策与校准 ---
    review_threshold: float = 0.60
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {"high": 18.0, "medium": 9.0, "low": 4.0}
    )
    calibration_model: str = "models/calibration.json"
    enable_llm: bool = False
    """是否尝试调用 LLM；默认关——本机无模型服务时必须能整段降级。"""
    max_citations_per_risk: int = 2
    """每条风险最多挂几条法条引用（护栏：引用必须能在本地索引回指）。"""
    use_conformal: bool = True
    """是否用 split conformal 给出风险码预测集（覆盖率保证）。"""

    # --- 申报书五节点开关（v2.1：补回节点③与大模型通道）---
    enable_subject_check: bool = True
    """节点③ 主体资格审查是否启用（法律知识小图 + 资格/权限/信用核验）。"""
    enable_entailment: bool = True
    """节点④ 是否对候选条文做“一致/冲突”判别（弱监督判别器）。"""
    llm_endpoint: str = ""
    """大模型端点；空则回落环境变量 ZLQL_LLM_ENDPOINT。"""
    llm_model: str = ""
    """模型名；空则回落环境变量 ZLQL_LLM_MODEL。"""
    llm_api_key: str = ""
    """API Key；空则回落环境变量 ZLQL_LLM_API_KEY（本地端点通常不需要）。"""
    llm_timeout: float = 90.0
    """单次请求超时（秒）。"""
    llm_max_clauses: int = 24
    llm_max_tokens: int = 512
    llm_native_thinking: str = ""
    llm_native_reasoning_format: str = ""
    llm_native_json_schema: bool = False
    llm_native_context_preflight: bool = False
    """保留旧版预算字段；完整覆盖由分块及显式调用预算控制。"""
    llm_chunk_chars: int = 1800
    llm_chunk_overlap: int = 160
    llm_max_calls: int = 0
    """0 表示不截断覆盖；非零预算耗尽时必须报告未审条款。"""
    llm_feedback_window_policy: str = "all_matching_windows"
    """cover_all_context_v1 avoids feedback calls whose entire source context is already present in one selected window."""
    llm_initial_max_attempts: int = 3
    # Versioned source-selection mode is explicit so historical free-quotation
    # receipts retain their original interpretation. The current website opts in.
    llm_initial_source_protocol: str = "legacy_free_quote_v1"
    """原生初审每个不变原文窗口的最大真实尝试次数，含首轮；未修正项仍待核验。"""
    llm_cross_clause_pairs: int = 12
    review_max_rounds: int = 3
    freeze_feedback_cycle: bool = False
    """显式启用不可变反馈周期；新发现/新证据版本延至下一周期并保持待办。"""
    subject_facts_path: str = str(REPORT_DIR / "subject_facts.json")
    """外部事实证据库；合同自述不作为已核验事实。"""
    entailment_backend: str = "auto"
    """auto 尝试本地深度模型，失败时记录原因并降级；rules 显式关闭。"""
    llm_strong_endpoint: str = ""
    """强模型端点（如 4bit 的 LegalOne-8B）；空则不做分级复核。"""
    llm_strong_model: str = ""
    """强模型名；空则用 legalone-8b。"""
    llm_review_top_k: int = 3
    """分级策略：规则/小模型全量过一遍后，只把风险最高的前 K 条条款交给强模型复核。"""
    llm_strong_timeout: float = 600.0
    """强模型单次超时。8B 4bit 在本机约 2.25 token/s，256 token 要 110 s，90 s 一定不够。"""

    # --- 存储与调度（申报书：十万份级存储/检索/批量审查）---
    use_bge: bool = False
    """密集通道是否用 BGE-M3 向量；默认关（实测强金标准 Recall@5 0.628 -> 0.167，属负结果）。"""

    store_path: str = ""
    """审查结果库路径；空则落 ``WORKSPACE/data/reviews.sqlite``。"""

    # --- 多智能体编排（申报书：基于图结构引导的多智能体协作）---
    enable_multi_agent: bool = False
    """是否走多智能体编排（六 Agent + 消息总线 + 图调度）。

    默认关：既有单进程五节点链路的产物结构、耗时、测试断言都按原样保留。
    打开后 ``ReviewEngine.run`` 转 ``agents_multi_run.run_multi_agent``，
    风险/评分/引用/复核队列的产物格式不变，只有 trace 变成 Agent 消息轨迹。
    """

    # --- 评测 ---
    seed: int = 20261104

    def resolve(self, relative: str) -> Path:
        """把 `models/xxx.json` 这类相对路径解析到包目录下。"""
        path = Path(relative)
        return path if path.is_absolute() else (PKG_DIR / path)

    def statute_file(self) -> Path:
        return Path(self.statutes_path).resolve() if self.statutes_path else V1_STATUTES


def default_config() -> Config:
    return Config()


def config_from_env(base: Config | None = None) -> Config:
    """从环境变量覆盖少量高频参数（供 CI / 现场演示切换）。"""
    cfg = base or Config()
    mapping: dict[str, tuple[str, type]] = {
        "ZLQL_STORE_PATH": ("store_path", str),
        "ZLQL_STATUTES_PATH": ("statutes_path", str),
        "ZLQL_CONTRACT_PACKAGES_PATH": ("contract_packages_path", str),
        "ZLQL_TOP_K": ("top_k", int),
        "ZLQL_REVIEW_THRESHOLD": ("review_threshold", float),
        "ZLQL_USE_DP": ("use_dp_segmentation", lambda v: v not in ("0", "false", "False")),
        "ZLQL_USE_RERANK": ("use_rerank", lambda v: v not in ("0", "false", "False")),
        "ZLQL_SEED": ("seed", int),
        "ZLQL_LLM_ENDPOINT": ("llm_endpoint", str),
        "ZLQL_LLM_MODEL": ("llm_model", str),
        "ZLQL_LLM_API_KEY": ("llm_api_key", str),
        "ZLQL_LLM_TIMEOUT": ("llm_timeout", float),
        "ZLQL_LLM_MAX_CLAUSES": ("llm_max_clauses", int),
        "ZLQL_LLM_MAX_TOKENS": ("llm_max_tokens", int),
        "ZLQL_LLM_NATIVE_THINKING": ("llm_native_thinking", str),
        "ZLQL_LLM_NATIVE_REASONING_FORMAT": ("llm_native_reasoning_format", str),
        "ZLQL_LLM_NATIVE_JSON_SCHEMA": ("llm_native_json_schema", lambda v: v.strip().lower() in ("1", "true", "on", "yes")),
        "ZLQL_LLM_NATIVE_CONTEXT_PREFLIGHT": ("llm_native_context_preflight", lambda v: v.strip().lower() in ("1", "true", "on", "yes")),
        "ZLQL_LLM_CHUNK_CHARS": ("llm_chunk_chars", int),
        "ZLQL_LLM_CHUNK_OVERLAP": ("llm_chunk_overlap", int),
        "ZLQL_LLM_MAX_CALLS": ("llm_max_calls", int),
        "ZLQL_LLM_FEEDBACK_WINDOW_POLICY": ("llm_feedback_window_policy", str),
        "ZLQL_LLM_INITIAL_MAX_ATTEMPTS": ("llm_initial_max_attempts", int),
        "ZLQL_LLM_INITIAL_SOURCE_PROTOCOL": ("llm_initial_source_protocol", str),
        "ZLQL_LLM_CROSS_CLAUSE_PAIRS": ("llm_cross_clause_pairs", int),
        "ZLQL_REVIEW_MAX_ROUNDS": ("review_max_rounds", int),
        "ZLQL_FREEZE_FEEDBACK_CYCLE": ("freeze_feedback_cycle", lambda v: v.lower() in ("1", "true", "on")),
        "ZLQL_SUBJECT_FACTS_PATH": ("subject_facts_path", str),
        "ZLQL_ENTAILMENT_BACKEND": ("entailment_backend", str),
        "ZLQL_LLM_STRONG_ENDPOINT": ("llm_strong_endpoint", str),
        "ZLQL_LLM_STRONG_MODEL": ("llm_strong_model", str),
        "ZLQL_LLM_REVIEW_TOP_K": ("llm_review_top_k", int),
        "ZLQL_LLM_STRONG_TIMEOUT": ("llm_strong_timeout", float),
        "ZLQL_USE_BGE": ("use_bge", lambda v: v not in ("0", "false", "False")),
        "ZLQL_NER_MODEL": ("ner_enable_model", lambda v: v not in ("0", "false", "False")),
        "ZLQL_MULTI_AGENT": ("enable_multi_agent", lambda v: v not in ("0", "false", "False")),
    }
    for env_name, (attr, cast) in mapping.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, attr, cast(raw))
        except (TypeError, ValueError):
            continue
    # 配了端点就默认开模型通道：否则"填了地址却没生效"是最难查的一类静默问题。
    # 想显式关掉用 ZLQL_ENABLE_LLM=0。
    if cfg.llm_endpoint and os.environ.get("ZLQL_ENABLE_LLM", "1") not in ("0", "false", "False"):
        cfg.enable_llm = True
    return cfg
