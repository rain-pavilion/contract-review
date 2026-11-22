# -*- coding: utf-8 -*-
"""节点④ 的第二半：判断"条款与法条是否一致/冲突"。

申报书写的做法是 BERT/Mamba 编码 + 余弦相似度召回 + 多头注意力二分类。真实约束是：
本机只有 CPU、没有预训练权重、也没有 PROVBENCH 那种带"蕴含/矛盾"标注的数据。
所以这里保留**同样的三段结构**，把中间的表示层换成可解释的判别特征：

```
条款文本 + 检索候选条文
   └─► [特征层] 8 维判别特征（覆盖、否定极性、强制强度、数字、动作、标签、层级、长度）
        └─► [判别层] 三分类 one-vs-rest 逻辑回归（entail / contradict / neutral）
             └─► [决策层] 置信度与门限，输出可写进报告的证据串
```

为什么值得这么绕：节点④ 的产出不只是"列出可能相关的法条"，而是要给出"这条法条与这个
条款是**支持**还是**冲突**"。冲突一旦成立，风险等级就能往上抬——这是申报书第五节点
"综合评估"最实在的输入。

训练方式与其它组件一致：**弱监督**。`weak_label()` 用规则造标签（条文与条款在义务强度、
否定极性、数字、标签上是否同向），逻辑回归在 CPU 上 2 秒收敛；权重落
`models/entailment.json`，可读可 diff。没有权重文件时用 `DEFAULT_WEIGHTS` 手写先验。

诚实标注：这不是 BERT 级别的语义判别，只在"条款与条文的关键要素同向/反向"这一层有效。
论文与答辩里要按"弱监督判别器"表述，不能宣称达到 PROVBENCH 的 F1。
"""
from __future__ import annotations

import json
import math
import re
import os
import contextlib
import contextvars
import copy
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import text as T

LABELS: tuple[str, ...] = ("entail", "contradict", "neutral")

#: 判别特征（顺序固定，训练与推理必须一致）
FEATURES: tuple[str, ...] = (
    "e_char_overlap",     # 条款与条文的字面覆盖率
    "e_keyword_hit",      # 条款命中条文里的高频法律词
    "e_tag_match",        # 风险标签与条文 risk_tags 的交集比
    "e_negation_flip",    # 否定极性是否相反（条款免责令 vs 条文应为令）
    "e_modality_gap",     # 义务强度落差（应当/必须 vs 可以/自行）
    "e_number_same",      # 数字是否同向出现
    "e_action_match",     # 行为词是否同类（支付/交付/解除/赔偿）
    "e_len_ratio",        # 长度比（过短条款对长条文时判别力下降）
)

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    # 手写先验：先让系统能跑，再由 train.py 用弱标注覆盖
    "entail": {
        "e_char_overlap": 2.6, "e_keyword_hit": 1.8, "e_tag_match": 2.2,
        "e_negation_flip": -2.4, "e_modality_gap": -0.6, "e_number_same": 0.9,
        "e_action_match": 1.4, "e_len_ratio": 0.5, "_bias": -1.6,
    },
    "contradict": {
        "e_char_overlap": 0.9, "e_keyword_hit": 0.8, "e_tag_match": 1.6,
        "e_negation_flip": 3.2, "e_modality_gap": 2.4, "e_number_same": -0.7,
        "e_action_match": 0.9, "e_len_ratio": 0.2, "_bias": -2.2,
    },
    "neutral": {
        "e_char_overlap": 0.2, "e_keyword_hit": 0.3, "e_tag_match": -0.5,
        "e_negation_flip": 0.6, "e_modality_gap": 0.4, "e_number_same": -0.2,
        "e_action_match": 0.1, "e_len_ratio": -0.1, "_bias": 0.4,
    },
}

WEIGHTS_FILE = "entailment.json"

#: 义务强度词（数值越大越强）
MODALITY = {
    "应当": 1.0, "应该": 1.0, "必须": 1.0, "须": 0.9, "不得": 1.0, "禁止": 1.0,
    "有权": 0.6, "可以": 0.4, "可": 0.35, "自行": 0.3, "无需": 0.2, "不需": 0.2,
}

#: 否定/免责词
NEGATION = (
    "不承担", "不负责", "免除", "免于", "无需", "不需", "概不", "放弃", "不得追究",
    "不视为", "不构成", "免责",
)

#: 行为词分组，同组视为同向
ACTION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("支付", "付款", "结算", "价款", "报酬"),
    ("交付", "交货", "提供", "送达", "提交"),
    ("解除", "终止", "撤销", "退订"),
    ("赔偿", "违约金", "损失", "补偿"),
    ("验收", "检验", "确认", "签收"),
    ("保密", "不得披露", "商业秘密", "个人信息"),
    ("资质", "许可", "资格", "备案"),
    ("保管", "妥善保管", "存储", "留存"),
)

#: 条文里常见的法律高频词，命中说明主题接近
LEGAL_TERMS: tuple[str, ...] = (
    "当事人", "合同", "履行", "违约", "损害", "赔偿", "解除", "撤销", "无效",
    "价款", "标的", "期限", "义务", "权利", "责任", "同意", "通知", "书面",
)


@dataclass
class EntailmentVerdict:
    """一个"条款 × 条文"的判别结果。"""

    article_uid: str
    article_no: str
    label: str
    score: float
    margin: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    note: str = ""
    source: str = "prior"
    neural_label: str = ""
    neural_score: float = 0.0
    rule_label: str = ""
    rule_score: float = 0.0
    disagreement: bool = False
    input_truncated: bool = False
    coverage_complete: bool = True
    token_coverage_complete: bool = True
    pair_coverage_complete: bool = True
    semantic_pending: bool = False
    manual_required: bool = False
    window_disagreement: bool = False
    decision_scope: str = "whole_input"
    window_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def is_conflict(self) -> bool:
        return (self.label == "contradict" and self.score >= 0.5
                and self.coverage_complete and not self.semantic_pending
                and not self.manual_required and not self.disagreement and not self.window_disagreement)

    def evidence_text(self) -> str:
        result = (
            f"{self.article_no} 判别为{self.label}（置信 {self.score:.2f}，"
            f"余量 {self.margin:+.2f}；{self.source}）"
        )
        return result + ("；仅为局部窗口信号，全文关系待核验" if self.semantic_pending else "")


def _modality(text: str) -> float:
    best = 0.0
    for token, value in MODALITY.items():
        if token in text:
            best = max(best, value)
    return best


def _action_group(text: str) -> int:
    for index, group in enumerate(ACTION_GROUPS):
        if any(token in text for token in group):
            return index
    return -1


def _has_negation(text: str) -> bool:
    return any(token in text for token in NEGATION)


def clause_features(
    clause_text: str,
    article_text: str,
    *,
    risk_tags: Sequence[str] = (),
    codes: Sequence[str] = (),
) -> dict[str, float]:
    """8 维判别特征。全部落在 [0,1]，避免量纲差异。"""
    clause = T.squash(clause_text or "")
    article = T.squash(article_text or "")
    feats = {name: 0.0 for name in FEATURES}

    tokens = {token for token in T.tokenize(clause) if len(token) >= 2}
    if tokens and article:
        hit = sum(1 for token in tokens if token in article)
        feats["e_char_overlap"] = round(hit / max(1, len(tokens)), 4)

    terms = [term for term in LEGAL_TERMS if term in clause and term in article]
    feats["e_keyword_hit"] = round(len(terms) / max(1, len(LEGAL_TERMS) * 0.25), 4)
    feats["e_keyword_hit"] = min(1.0, feats["e_keyword_hit"])

    tags = {str(tag) for tag in (risk_tags or [])}
    if tags:
        wanted = tags | {str(code) for code in (codes or [])}
        feats["e_tag_match"] = round(len(tags & wanted) / max(1, len(tags)), 4)

    clause_neg = _has_negation(clause)
    article_neg = _has_negation(article)
    feats["e_negation_flip"] = 1.0 if (clause_neg != article_neg and clause_neg) else 0.0

    gap = _modality(article) - _modality(clause)
    # 条文强、条款弱 → 落差为正且越大越可疑
    feats["e_modality_gap"] = round(max(0.0, min(1.0, gap)), 4)

    clause_nums = set(re.findall(r"\d+(?:\.\d+)?", clause))
    article_nums = set(re.findall(r"\d+(?:\.\d+)?", article))
    if clause_nums:
        feats["e_number_same"] = round(len(clause_nums & article_nums) / len(clause_nums), 4)
    else:
        feats["e_number_same"] = 0.5  # 无数字时不奖不罚

    clause_action, article_action = _action_group(clause), _action_group(article)
    if clause_action >= 0 and article_action >= 0:
        feats["e_action_match"] = 1.0 if clause_action == article_action else 0.0
    else:
        feats["e_action_match"] = 0.5

    ratio = len(clause) / max(1, len(article))
    feats["e_len_ratio"] = round(min(1.0, ratio), 4)
    return feats


def weak_label(features: dict[str, float]) -> str:
    """弱标注：给训练用的三类标签。

    规则：否定极性相反且主题接近 → contradict；字面与标签都高 → entail；其余 neutral。
    """
    if features["e_negation_flip"] >= 0.5 and features["e_char_overlap"] >= 0.12:
        return "contradict"
    if features["e_char_overlap"] >= 0.3 and features["e_tag_match"] >= 0.5:
        return "entail"
    if features["e_modality_gap"] >= 0.6 and features["e_action_match"] >= 1.0:
        return "contradict"
    return "neutral"


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class EntailmentClassifier:
    """三分类 one-vs-rest 逻辑回归，权重来自 JSON 或手写先验。"""

    weights: dict[str, dict[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()})
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.meta.get("source") or "prior")

    def scores(self, features: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for label in LABELS:
            row = self.weights.get(label, {})
            total = float(row.get("_bias", 0.0))
            for name in FEATURES:
                total += float(row.get(name, 0.0)) * float(features.get(name, 0.0))
            out[label] = _sigmoid(total)
        return out

    def predict(self, features: dict[str, float]) -> tuple[str, float, float]:
        """返回 (标签, 该标签置信, 与次优的余量)。"""
        scores = self.scores(features)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, score = ranked[0]
        margin = score - ranked[1][1]
        return label, round(score, 4), round(margin, 4)


_REGISTRY: dict[str, EntailmentClassifier] = {}


def _model_path(model_dir: Path | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir) / WEIGHTS_FILE
    from .config import MODEL_DIR

    return Path(MODEL_DIR) / WEIGHTS_FILE


def load_classifier(model_dir: Path | None = None) -> EntailmentClassifier:
    """读 `models/entailment.json`；不存在则用手写先验（并标注 source=prior）。"""
    path = _model_path(model_dir)
    key = str(path)
    if key in _REGISTRY:
        return _REGISTRY[key]
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            weights = {
                label: {k: float(v) for k, v in (payload.get("weights", {}).get(label) or {}).items()}
                for label in LABELS
            }
            if all(weights[label] for label in LABELS):
                meta = dict(payload.get("meta") or {})
                meta.setdefault("source", "trained")
                classifier = EntailmentClassifier(weights=weights, meta=meta)
                _REGISTRY[key] = classifier
                return classifier
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    classifier = EntailmentClassifier()
    _REGISTRY[key] = classifier
    return classifier


def save_weights(weights: dict[str, dict[str, float]], meta: dict[str, Any], model_dir: Path | None = None) -> Path:
    path = _model_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights, "meta": dict(meta, source=meta.get("source", "trained"))}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _REGISTRY.pop(str(path), None)
    return path


_BACKEND = {"name": "rules"}
_BERT = {"loaded": False, "predictor": None, "error": "", "reason": "", "load_seconds": 0.0}
_BERT_LOCK = threading.RLock()


@dataclass
class ReviewPredictionCache:
    """Ephemeral neural computations, never semantic or legal confirmations."""

    entries: dict[tuple, tuple[dict[str, Any], str]] = field(default_factory=dict)
    identities: dict[tuple[int, ...], tuple[Any, ...]] = field(default_factory=dict)
    prediction_calls: int = 0
    cache_reuses: int = 0
    prediction_failures: int = 0
    uncacheable_results: int = 0

    def summary(self) -> dict[str, Any]:
        return {"scope": "single_review_exact_neural_input", "prediction_calls": self.prediction_calls,
                "cache_reuses": self.cache_reuses, "cached_results": len(self.entries),
                "prediction_failures": self.prediction_failures,
                "uncacheable_results": self.uncacheable_results,
                "changes_legal_status": False}


_PREDICTION_CACHE: contextvars.ContextVar[ReviewPredictionCache | None] = contextvars.ContextVar(
    "entailment_review_prediction_cache", default=None)


@contextlib.contextmanager
def review_prediction_cache():
    """Create a fresh scope for every review and release all retained input."""
    cache = ReviewPredictionCache()
    token = _PREDICTION_CACHE.set(cache)
    try:
        yield cache
    finally:
        _PREDICTION_CACHE.reset(token)
        cache.entries.clear()
        cache.identities.clear()


def _prediction_key(predictor: Any, clause: str, article: str, cache: ReviewPredictionCache) -> tuple:
    method = predictor.predict
    function = getattr(method, "__func__", method)
    identities = (predictor, function, getattr(predictor, "model", None), getattr(predictor, "tokenizer", None))
    identity = tuple(id(value) for value in identities)
    # Strong references prevent a replaced predictor/model from reusing an id.
    cache.identities[identity] = identities
    namespace = getattr(function, "__globals__", {})
    options = json.dumps({
        "device": str(getattr(predictor, "device", "")),
        "max_window_pairs": getattr(predictor, "max_window_pairs", None),
        "env_max_window_pairs": os.environ.get("ZLQL_ENTAILMENT_MAX_WINDOW_PAIRS", "64"),
        "max_clause": namespace.get("MAX_CLAUSE"), "max_article": namespace.get("MAX_ARTICLE"),
        "metadata": getattr(predictor, "meta", None),
    }, ensure_ascii=False, sort_keys=True, default=repr)
    return identity, options, clause, article


def _cacheable_prediction(out: Any) -> bool:
    """Partial failed executions and malformed outputs must remain retryable."""
    try:
        if not isinstance(out, dict) or out.get("label") not in LABELS:
            return False
        if not math.isfinite(float(out["score"])):
            return False
        probabilities = out.get("probabilities") or {}
        if not isinstance(probabilities, dict) or not all(math.isfinite(float(v)) for v in probabilities.values()):
            return False
        audit = out.get("window_audit") or {}
        return (isinstance(audit, dict) and not audit.get("failed_pairs") and not audit.get("failures")
                and audit.get("input_valid", True) is not False
                and ("executed_pairs" not in audit or int(audit["executed_pairs"]) > 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _predict_pair(predictor: Any, clause: str, article: str) -> tuple[dict[str, Any], bool, str]:
    cache = _PREDICTION_CACHE.get()
    # The model lock makes lookup, actual forward work and insertion atomic for
    # copied review contexts. Distinct review scopes still have distinct stores.
    with _BERT_LOCK:
        key = None
        if cache is not None:
            try:
                key = _prediction_key(predictor, clause, article, cache)
            except (AttributeError, TypeError, ValueError, RecursionError):
                pass  # Optional cache metadata must never suppress inference.
        if cache is not None and key is not None and key in cache.entries:
            cached, call_id = cache.entries[key]
            cache.cache_reuses += 1
            return copy.deepcopy(cached), True, call_id
        if cache is not None:
            cache.prediction_calls += 1
        try:
            out = predictor.predict(clause, article)
        except Exception:
            if cache is not None:
                cache.prediction_failures += 1
            raise
        audit = (out.get("window_audit") or {}) if isinstance(out, dict) else {}
        call_id = str(audit.get("call_id") or uuid.uuid4().hex) if isinstance(audit, dict) else uuid.uuid4().hex
        if cache is not None:
            if key is not None and _cacheable_prediction(out):
                try:
                    snapshot = copy.deepcopy(out)
                    returned = copy.deepcopy(snapshot)
                except (TypeError, ValueError, RecursionError):
                    pass
                else:
                    cache.entries[key] = (snapshot, call_id)
                    return returned, False, call_id
            cache.uncacheable_results += 1
        return out, False, call_id


def set_backend(name: str) -> str:
    """切换判别后端：rules（8 维弱监督，默认）/ bert（双塔 cross-attention）/ auto。

    默认留在 rules 是为了可复现：BERT 后端依赖 models/bert_entail/model.pt，
    有产物才生效；没有产物时 auto 静默退回 rules（降级留账）。
    """
    choice = (name or "rules").strip().lower()
    if choice not in ("rules", "bert", "auto"):
        choice = "rules"
    _BACKEND["name"] = choice
    return choice


def backend() -> str:
    return str(_BACKEND["name"])


def _bert_predictor():
    """惰性加载双塔判别器；不可用时返回 None。"""
    with _BERT_LOCK:
        if _BERT["loaded"]:
            return _BERT["predictor"]
        _BERT["loaded"] = True
        started = time.perf_counter()
        try:
            from . import bert_entail
            if not bert_entail.available():
                raise FileNotFoundError("bert_entail_weights_missing")
            _BERT["predictor"] = bert_entail.load_predictor()
            if _BERT["predictor"] is None:
                raise RuntimeError("bert_entail_load_returned_none")
        except Exception as exc:
            _BERT["predictor"] = None
            _BERT["error"] = type(exc).__name__
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            _BERT["reason"] = ("dependency_missing:" + str(cause.name) if isinstance(cause, ModuleNotFoundError)
                               else "checkpoint_missing" if isinstance(exc, FileNotFoundError)
                               else "checkpoint_load_failed")
        _BERT["load_seconds"] = time.perf_counter() - started
        from . import model_audit
        model_audit.record("entailment", "bert_entail", "load", "succeeded" if _BERT["predictor"] else "degraded",
                           _BERT["load_seconds"], "" if _BERT["predictor"] else _BERT["reason"],
                           error_type=_BERT["error"], backend="bert" if _BERT["predictor"] else "rules")
    return _BERT["predictor"]


def runtime_status() -> dict[str, Any]:
    return {"requested_backend": backend(), "active_backend": "bert" if _BERT["predictor"] is not None and backend() != "rules" else "rules",
            "bert_loaded": _BERT["predictor"] is not None, "bert_error": _BERT["error"],
            "bert_error_reason": _BERT["reason"],
            "bert_load_seconds": round(float(_BERT["load_seconds"]), 3),
            "bert_device": os.environ.get("ZLQL_ENTAILMENT_DEVICE", os.environ.get("ZLQL_BERT_DEVICE", "cpu")),
            "bert_metadata": getattr(_BERT["predictor"], "meta", {}) or {}}


def judge_pair(
    clause_text: str,
    article: Any,
    *,
    codes: Sequence[str] = (),
    classifier: EntailmentClassifier | None = None,
) -> EntailmentVerdict:
    """判别一个条款与一条候选条文的关系。

    `article` 可以是 RetrievalHit / Statute / dict，只要能取到 content、article_no、risk_tags。
    """
    if isinstance(article, dict):
        content = str(article.get("content") or "")
        article_no = str(article.get("article_no") or "")
        uid = str(article.get("article_uid") or article_no)
        tags = article.get("risk_tags") or ()
    else:
        content = str(getattr(article, "content", "") or "")
        article_no = str(getattr(article, "article_no", "") or "")
        uid = str(getattr(article, "article_uid", article_no))
        tags = getattr(article, "risk_tags", ()) or ()

    feats = clause_features(clause_text, content, risk_tags=tags, codes=codes)
    label: str | None = None
    score = 0.0
    margin = 0.0
    source = "rules"
    coverage_note = ""
    truncated = False
    coverage_complete = token_coverage_complete = pair_coverage_complete = True
    semantic_pending = manual_required = window_disagreement = False
    decision_scope = "whole_input"
    window_audit: dict[str, Any] = {}
    neural_label, neural_score = "", 0.0
    rule_label, rule_score = "", 0.0
    disagreement = False
    choice = backend()
    if choice in ("bert", "auto") and classifier is None:
        predictor = _bert_predictor()
        if predictor is not None:
            started = time.perf_counter()
            try:
                out, cache_reused, prediction_call_id = _predict_pair(predictor, clause_text, content)
                label = str(out["label"])
                score = float(out["score"])
                probabilities = sorted(
                    (float(v) for v in (out.get("probabilities") or {}).values()), reverse=True
                )
                margin = score - (probabilities[1] if len(probabilities) > 1 else 0.0)
                source = "bert_entail"
                neural_label, neural_score = label, score
                rule_label, rule_score, _ = load_classifier().predict(feats)
                disagreement = rule_label != neural_label
                truncated = bool(out.get("input_truncated"))
                token_coverage_complete = bool(out.get("token_coverage_complete", not truncated))
                pair_coverage_complete = bool(out.get("pair_coverage_complete", not truncated))
                window_disagreement = bool(out.get("window_disagreement"))
                semantic_pending = bool(out.get("semantic_pending")) or not token_coverage_complete or not pair_coverage_complete
                coverage_complete = (bool(out.get("coverage_complete", not truncated))
                                     and token_coverage_complete and pair_coverage_complete and not semantic_pending)
                manual_required = bool(out.get("manual_required")) or disagreement or not coverage_complete or window_disagreement
                decision_scope = str(out.get("decision_scope") or ("local_window_only" if semantic_pending else "whole_input"))
                window_audit = dict(out.get("window_audit") or {})
                if window_audit and semantic_pending:
                    coverage_note = (f"实际执行{window_audit.get('executed_pairs', 0)}/{window_audit.get('planned_pairs', 0)}个窗口对；"
                                     f"输入token{'已全覆盖' if token_coverage_complete else '未全覆盖'}，"
                                     "局部预测不能证明全文关系，需跨窗口语义核验")
                elif truncated:
                    coverage_note = "BERT本项仍有未编码输入，当前结果覆盖不完整，需结合全文人工核验"
                elif semantic_pending:
                    coverage_note = "神经判别的全文语义关系尚未确认，需人工核验"
                from . import model_audit
                reason = ("bert_window_prediction_failed" if window_audit.get("failed_pairs") else
                          "bert_window_pair_budget_exhausted" if not pair_coverage_complete and window_audit else
                          "bert_input_window_limit" if truncated else
                          "bert_multi_window_semantics_pending" if semantic_pending else "")
                operation = "prediction_cache_reuse" if cache_reused else "predict_pair"
                status = "reused" if cache_reused else "degraded" if semantic_pending or truncated else "succeeded"
                model_audit.record("entailment", "bert_entail", operation, status,
                                   time.perf_counter() - started, "exact_input_same_review" if cache_reused else reason,
                                   backend="bert", pairs=0 if cache_reused else 1,
                                   call_id=prediction_call_id,
                                   input_sha256=hashlib.sha256(json.dumps([clause_text, content], ensure_ascii=False).encode("utf-8")).hexdigest(),
                                   planned_pairs=window_audit.get("planned_pairs", 1),
                                   executed_pairs=0 if cache_reused else window_audit.get("executed_pairs", 1),
                                   token_coverage_complete=token_coverage_complete,
                                   pair_coverage_complete=pair_coverage_complete,
                                   semantic_pending=semantic_pending, window_disagreement=window_disagreement)
                if disagreement:
                    model_audit.record("entailment", "bert_entail", "cross_check", "degraded", 0,
                                       "neural_rule_disagreement", backend="bert+rules")
                if window_disagreement:
                    model_audit.record("entailment", "bert_entail", "cross_check", "degraded", 0,
                                       "window_label_disagreement", backend="bert",
                                       call_id=window_audit.get("call_id"), window_disagreement=True)
            except Exception as exc:
                label = None
                source = "rules"
                window_audit = dict(getattr(exc, "window_audit", {}) or {})
                token_coverage_complete = bool(window_audit.get("token_coverage_complete", False))
                pair_coverage_complete = bool(window_audit.get("pair_coverage_complete", False))
                input_valid = bool(window_audit.get("input_valid", True))
                truncated = input_valid and not token_coverage_complete
                semantic_pending = manual_required = True
                coverage_complete = False
                decision_scope = "rules_fallback_pending" if input_valid else "invalid_input"
                coverage_note = ("神经判别执行失败，当前为规则降级结果，全文关系需人工核验" if input_valid else
                                 "条款或法条缺少有效token，未执行神经判别；规则降级结果不能确证全文关系，需补齐输入")
                from . import model_audit
                model_audit.record("entailment", "bert_entail", "predict_pair", "degraded", time.perf_counter() - started,
                                   str(getattr(exc, "reason_code", "bert_prediction_failed")),
                                   error_type=type(exc).__name__, backend="rules")
        else:
            coverage_complete = token_coverage_complete = pair_coverage_complete = False
            semantic_pending = manual_required = True
            decision_scope = "rules_fallback_pending"
            coverage_note = "神经判别不可用，当前为规则降级结果，全文关系需人工核验"
            from . import model_audit
            model_audit.record("entailment", "bert_entail", "predict_pair", "degraded", 0,
                               "bert_unavailable", error_type=_BERT["error"], backend="rules")
    if label is None:
        classifier = classifier or load_classifier()
        label, score, margin = classifier.predict(feats)
    # 主题完全不沾边时强制 neutral：避免"字面零重叠也判蕴含"的假阳性。
    # 只对 8 维弱监督生效 —— BERT 后端已看到原文语义，再加这层过滤会把它的冲突判断抹掉。
    if source == "rules" and feats["e_char_overlap"] < 0.05 and feats["e_tag_match"] < 0.5:
        label, score, margin = "neutral", min(score, 0.5), margin
    dispute_note = (f"{'局部窗口神经信号' if decision_scope == 'local_window_only' else '神经判定'}{neural_label}({neural_score:.4f})，"
                    f"全文规则判定{rule_label}({rule_score:.4f})；存在分歧，需人工复核"
                    if disagreement else "")
    return EntailmentVerdict(
        article_uid=uid,
        article_no=article_no,
        label=label,
        score=float(score),
        margin=float(margin),
        features=feats,
        note="；".join(filter(None, ["条款以否定/免责表述偏离条文要求" if label == "contradict" and not semantic_pending else "", coverage_note, dispute_note,
                                    "局部窗口预测存在分歧，需人工复核" if window_disagreement else ""])),
        source=source,
        neural_label=neural_label, neural_score=neural_score, rule_label=rule_label, rule_score=rule_score,
        disagreement=disagreement, input_truncated=truncated, coverage_complete=coverage_complete,
        token_coverage_complete=token_coverage_complete, pair_coverage_complete=pair_coverage_complete,
        semantic_pending=semantic_pending, manual_required=manual_required,
        window_disagreement=window_disagreement, decision_scope=decision_scope, window_audit=window_audit,
    )


def annotate(
    clause_text: str,
    articles: Sequence[Any],
    *,
    codes: Sequence[str] = (),
    classifier: EntailmentClassifier | None = None,
) -> list[EntailmentVerdict]:
    """批量判别，供 legal() 对一条条款的 top-k 引用做统一处理。"""
    # An explicit classifier is a deliberate rule-only override. Do not create one
    # here: passing it would bypass the neural backend for every production call.
    return [judge_pair(clause_text, article, codes=codes, classifier=classifier) for article in articles]


def summarize(verdicts: Iterable[EntailmentVerdict]) -> dict[str, Any]:
    """统计：给 trace 与报告用的一行摘要。"""
    items = list(verdicts)
    counts = {label: 0 for label in LABELS}
    for item in items:
        counts[item.label] = counts.get(item.label, 0) + 1
    conflicts = [item for item in items if item.is_conflict]
    return {
        "pairs": len(items),
        "counts": counts,
        "conflicts": [f"{item.article_no}:{item.score:.2f}" for item in conflicts[:4]],
        "source": items[0].source if items else "prior",
    }
