"""学习型组件的加载与推理（纯 numpy，权重落盘为 JSON）。

四个组件
--------
1. `BoundaryClassifier`：条款边界二分类（logistic）。训练见 `train.py::train_boundary`。
   作用：规则给不出的边界（没有编号、靠语义断句的地方）由它兜底，替代 v1 的
   "LLM 不可用就只剩规则"。
2. `RiskClassifier`：条款级多标签风险分类（one-vs-rest logistic）。
3. `RetrievalRanker`：法条检索的 LTR 重排（pairwise logistic 学到的线性打分）。
4. `Calibrator`：置信度校准（isotonic 分段线性 / Platt）。
5. `NerTagger`：字符级 BIO 标注器（平均感知机 + Viterbi）。

设计约束
--------
* 权重全部 JSON，**不 pickle**：可 diff、可审计、可手改，评审现场能解释每一项。
* 模型缺失时 `predict` 返回 `None` 而不是抛异常，调用方必须显式处理"没有模型"这条
  分支 —— 这是 v1 最常踩的坑（try/except 吞掉后静默降级，报告里看不出来）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import MODEL_DIR
from . import text as T


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(weights: Sequence[float], bias: float, vector: Sequence[float]) -> float:
    total = bias
    for w, v in zip(weights, vector):
        total += w * v
    return total


def load_json(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# 1. 边界分类器
# ---------------------------------------------------------------------------

@dataclass
class BoundaryClassifier:
    names: list[str]
    weights: list[float]
    bias: float = 0.0
    train_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "BoundaryClassifier | None":
        payload = load_json(path)
        if not payload or "weights" not in payload:
            return None
        return cls(
            names=list(payload.get("names", [])),
            weights=[float(w) for w in payload["weights"]],
            bias=float(payload.get("bias", 0.0)),
            train_meta=dict(payload.get("meta", {})),
        )

    def predict_proba(self, features: Mapping[str, float]) -> float | None:
        if not self.names:
            return None
        vector = [float(features.get(name, 0.0)) for name in self.names]
        return _sigmoid(_dot(self.weights, self.bias, vector))

    def contribution(self, features: Mapping[str, float], top: int = 5) -> list[tuple[str, float]]:
        """解释一个预测：返回贡献最大的特征（论文里画因子图的素材）。"""
        pairs = []
        for name, weight in zip(self.names, self.weights):
            value = float(features.get(name, 0.0))
            if value:
                pairs.append((name, weight * value))
        pairs.sort(key=lambda item: -abs(item[1]))
        return pairs[:top]


# ---------------------------------------------------------------------------
# 2. 风险分类器
# ---------------------------------------------------------------------------

@dataclass
class RiskClassifier:
    names: list[str]
    labels: list[str]
    weights: dict[str, list[float]]
    biases: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)
    train_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "RiskClassifier | None":
        payload = load_json(path)
        if not payload or "weights" not in payload:
            return None
        return cls(
            names=list(payload.get("names", [])),
            labels=list(payload.get("labels", [])),
            weights={k: [float(x) for x in v] for k, v in payload["weights"].items()},
            biases={k: float(v) for k, v in payload.get("biases", {}).items()},
            thresholds={k: float(v) for k, v in payload.get("thresholds", {}).items()},
            train_meta=dict(payload.get("meta", {})),
        )

    def predict_proba(self, features: Mapping[str, float]) -> dict[str, float]:
        vector = [float(features.get(name, 0.0)) for name in self.names]
        out: dict[str, float] = {}
        for label in self.labels:
            weights = self.weights.get(label)
            if not weights:
                continue
            out[label] = _sigmoid(_dot(weights, self.biases.get(label, 0.0), vector))
        return out

    def predict(self, features: Mapping[str, float]) -> list[tuple[str, float]]:
        probs = self.predict_proba(features)
        hits = [
            (label, prob) for label, prob in probs.items()
            if prob >= self.thresholds.get(label, 0.5)
        ]
        hits.sort(key=lambda item: -item[1])
        return hits


# ---------------------------------------------------------------------------
# 3. 检索重排（LTR）
# ---------------------------------------------------------------------------

RERANK_FEATURES: tuple[str, ...] = (
    "q_bm25",
    "q_bm25_rank",
    "q_lsa_cos",
    "q_lsa_rank",
    "q_rrf",
    "q_tag_hit",
    "q_tag_ratio",
    "q_law_match",
    "q_num_match",
    "q_len_penalty",
    "q_query_len",
    "q_authority",
    "q_expansion",
    "q_char_overlap",
    "q_idf_overlap",
)


@dataclass
class RetrievalRanker:
    names: list[str]
    weights: list[float]
    bias: float = 0.0
    train_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "RetrievalRanker | None":
        payload = load_json(path)
        if not payload or "weights" not in payload:
            return None
        return cls(
            names=list(payload.get("names", list(RERANK_FEATURES))),
            weights=[float(w) for w in payload["weights"]],
            bias=float(payload.get("bias", 0.0)),
            train_meta=dict(payload.get("meta", {})),
        )

    def score(self, features: Mapping[str, float]) -> float | None:
        if not self.names:
            return None
        vector = [float(features.get(name, 0.0)) for name in self.names]
        return _dot(self.weights, self.bias, vector)

    @property
    def normalized(self) -> bool:
        """是否按"同一查询内的候选集"做列标准化。

        14 维 `q_*` 的量纲差着两个数量级（`q_bm25` 可到 20，`q_rrf` 只有 0.03），
        直接线性相加会让 `q_tag_hit`（0/1）这类特征压过 `q_rrf`，重排反而把 RRF
        的好排序打乱（实测强金标准 Recall@5 从 0.628 掉到 0.389）。列标准化后各
        特征同量纲，训练与推理口径一致。
        """
        return bool(self.train_meta.get("normalize", False))

    def rank_normalized(
        self,
        items: Sequence[tuple[Any, Mapping[str, float]]],
    ) -> list[tuple[Any, float]]:
        """候选集内列标准化后再打分（训练与推理共用这一条路径）。"""
        rows = [[float(feats.get(name, 0.0)) for name in self.names] for _item, feats in items]
        if not rows:
            return []
        columns = len(self.names)
        means = [sum(row[col] for row in rows) / len(rows) for col in range(columns)]
        stds: list[float] = []
        for col in range(columns):
            var = sum((row[col] - means[col]) ** 2 for row in rows) / max(1, len(rows))
            stds.append(var ** 0.5 or 1.0)
        scored: list[tuple[Any, float]] = []
        for (item, _feats), row in zip(items, rows):
            vector = [(row[col] - means[col]) / stds[col] for col in range(columns)]
            scored.append((item, _dot(self.weights, self.bias, vector)))
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def rank(self, items: Sequence[tuple[Any, Mapping[str, float]]]) -> list[tuple[Any, float]]:
        scored: list[tuple[Any, float]] = []
        for item, feats in items:
            value = self.score(feats)
            scored.append((item, value if value is not None else 0.0))
        scored.sort(key=lambda pair: -pair[1])
        return scored


#: 没有训练权重时的手写初始权重（有文献依据的先验，作为 LTR 的冷启动）。
DEFAULT_RERANK_WEIGHTS: dict[str, float] = {
    "q_bm25": 1.2,
    "q_bm25_rank": -0.9,
    "q_lsa_cos": 2.0,
    "q_lsa_rank": -0.6,
    "q_rrf": 1.5,
    "q_tag_hit": 0.8,
    "q_tag_ratio": 0.6,
    "q_law_match": 0.7,
    "q_num_match": 0.5,
    "q_len_penalty": -0.4,
    "q_query_len": 0.05,
    "q_authority": 0.3,
    "q_expansion": 0.2,
    "q_char_overlap": 0.9,
    "q_idf_overlap": 1.6,
}


def default_ranker() -> RetrievalRanker:
    return RetrievalRanker(
        names=list(RERANK_FEATURES),
        weights=[DEFAULT_RERANK_WEIGHTS[name] for name in RERANK_FEATURES],
        bias=0.0,
        train_meta={"source": "handcrafted-prior"},
    )


# ---------------------------------------------------------------------------
# 4. 校准器
# ---------------------------------------------------------------------------

@dataclass
class Calibrator:
    """isotonic 分段线性校准（Platt 作为退化情形：只有一个区间）。"""

    xs: list[float] = field(default_factory=lambda: [0.0, 1.0])
    ys: list[float] = field(default_factory=lambda: [0.0, 1.0])
    kind: str = "identity"
    train_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator | None":
        payload = load_json(path)
        if not payload:
            return None
        xs = [float(x) for x in payload.get("xs", [])]
        ys = [float(y) for y in payload.get("ys", [])]
        if len(xs) < 2 or len(xs) != len(ys):
            return None
        return cls(xs=xs, ys=ys, kind=str(payload.get("kind", "isotonic")),
                   train_meta=dict(payload.get("meta", {})))

    def apply(self, value: float) -> float:
        if not self.xs:
            return value
        if value <= self.xs[0]:
            return max(0.0, min(1.0, self.ys[0]))
        if value >= self.xs[-1]:
            return max(0.0, min(1.0, self.ys[-1]))
        for index in range(1, len(self.xs)):
            left, right = self.xs[index - 1], self.xs[index]
            if value <= right:
                span = right - left
                if span <= 0:
                    return self.ys[index]
                ratio = (value - left) / span
                return max(0.0, min(1.0, self.ys[index - 1] + ratio * (self.ys[index] - self.ys[index - 1])))
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "xs": self.xs, "ys": self.ys, "meta": self.train_meta}


# ---------------------------------------------------------------------------
# 5. 字符级感知机 NER
# ---------------------------------------------------------------------------

NER_LABELS: tuple[str, ...] = (
    "O",
    "B-PARTY", "I-PARTY",
    "B-AMOUNT", "I-AMOUNT",
    "B-DATE", "I-DATE",
    "B-SUBJECT", "I-SUBJECT",
)

_DATE_CHARS = set("年月日〇零一二三四五六七八九十")
_AMOUNT_CHARS = set("元万亿圆角分整")
_PARTY_SUFFIX = ("公司", "中心", "学校", "医院", "大学", "研究院", "事务所", "局", "委员会", "政府", "银行")


def ner_char_features(chars: Sequence[str], index: int, length: int) -> list[str]:
    """字符级特征：字符本身 + 前后窗口 + 字符类型 + 词形标志。

    v1 没有这一层（只有正则），所以"甲方：江苏某某建设有限公司"换个前缀就抽不到。
    """
    def at(pos: int) -> str:
        return chars[pos] if 0 <= pos < length else "<PAD>"

    ch = at(index)
    feats = [
        f"c={ch}",
        f"c-1={at(index - 1)}",
        f"c-2={at(index - 2)}",
        f"c+1={at(index + 1)}",
        f"c+2={at(index + 2)}",
        f"b={at(index - 1)}{ch}",
        f"t={ch}{at(index + 1)}",
        f"c-1c+1={at(index - 1)}{at(index + 1)}",
    ]
    if ch.isdigit():
        feats.append("type=digit")
    elif ch in _DATE_CHARS:
        feats.append("type=datechar")
    elif ch in _AMOUNT_CHARS:
        feats.append("type=amountchar")
    elif ch in "，。；：、（）()0123456789":
        feats.append("type=punct")
    elif T.has_cjk(ch):
        feats.append("type=cjk")
    else:
        feats.append("type=other")

    left = "".join(at(i) for i in range(index - 6, index))
    right = "".join(at(i) for i in range(index + 1, index + 7))
    for suffix in _PARTY_SUFFIX:
        if right.startswith(suffix[:2]) or suffix in left[-4:] + ch:
            feats.append(f"party_suffix={suffix}")

    # Gazetteer 特征（命名实体识别里的标准做法）：把"机构后缀在前面/后面 20 字内"
    # 和"前面 10 字里有当事人角色标签＋冒号"显式喂给模型。没有这两条，字符级感知机
    # 只能靠单字记住名字的首尾，跨度边界几乎学不出来（实测跨度 F1 会掉到 0）。
    ahead = "".join(at(i) for i in range(index, index + 20))
    behind = "".join(at(i) for i in range(max(0, index - 10), index))
    gaz = next((suffix for suffix in _PARTY_SUFFIX if suffix in ahead), "")
    if gaz:
        feats.append(f"gaz_ahead={gaz}")
        # 名字结束位置：紧接着就是后缀尾部
        close = next((suffix for suffix in _PARTY_SUFFIX if ahead.startswith(suffix)), "")
        if close:
            feats.append("gaz_at_suffix_end")
    role_hit = next((role for role in ("甲方", "乙方", "丙方", "采购人", "供应商", "委托方",
                                       "受托方", "保管人", "寄存人", "出租人", "承租人")
                     if role in behind), "")
    if role_hit:
        feats.append(f"gaz_role_before={role_hit}")
        if "：" in behind or ":" in behind:
            feats.append("gaz_after_colon")
    if any(word in ahead for word in ("万元", "亿元", "元整", "元")):
        feats.append("gaz_amount_ahead")
    if any(word in ahead for word in ("年", "月", "日")):
        feats.append("gaz_date_ahead")
    if any(word in behind for word in ("项目名称", "标的名称", "服务名称", "工程名称", "合同名称")):
        feats.append("gaz_subject_before")
    if any(word in left for word in ("甲方", "乙方", "丙方", "名称", "当事人")):
        feats.append("role=near")
    if any(word in left[-6:] for word in ("人民币", "￥", "¥", "RMB")):
        feats.append("amount=near")
    if any(word in left[-6:] for word in ("年", "月")):
        feats.append("date=near")
    if index == 0:
        feats.append("pos=BOS")
    if index >= length - 1:
        feats.append("pos=EOS")
    return feats


@dataclass
class NerTagger:
    """平均感知机 BIO 标注器。权重以 `feature -> {label: weight}` 存 JSON。"""

    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    labels: list[str] = field(default_factory=lambda: list(NER_LABELS))
    train_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "NerTagger | None":
        payload = load_json(path)
        if not payload or "weights" not in payload:
            return None
        weights = {
            feat: {label: float(w) for label, w in table.items()}
            for feat, table in payload["weights"].items()
        }
        return cls(weights=weights, labels=list(payload.get("labels", NER_LABELS)),
                   train_meta=dict(payload.get("meta", {})))

    def to_dict(self) -> dict[str, Any]:
        return {"labels": self.labels, "weights": self.weights, "meta": self.train_meta}

    # -- 打分与解码 --------------------------------------------------------
    def _scores(self, feats: Sequence[str]) -> dict[str, float]:
        totals = {label: 0.0 for label in self.labels}
        for feat in feats:
            table = self.weights.get(feat)
            if not table:
                continue
            for label, weight in table.items():
                totals[label] = totals.get(label, 0.0) + weight
        return totals

    def _transition(self, prev: str, cur: str) -> float:
        """极简转移约束：I-X 只能跟在 B-X/I-X 后（用大负数实现，不学转移权重）。"""
        if not cur.startswith("I-"):
            return 0.0
        entity = cur[2:]
        if prev in (f"B-{entity}", f"I-{entity}"):
            return 0.0
        return -1000.0

    def decode(self, text: str) -> list[tuple[int, int, str, float]]:
        """返回 [(start, end, 实体类型, 平均得分)]。"""
        chars = list(text)
        length = len(chars)
        if length == 0:
            return []
        emissions: list[dict[str, float]] = []
        for index in range(length):
            emissions.append(self._scores(ner_char_features(chars, index, length)))

        # Viterbi
        labels = self.labels
        score: dict[str, float] = {label: emissions[0][label] for label in labels}
        back: list[dict[str, str]] = []
        for index in range(1, length):
            new_score: dict[str, float] = {}
            pointer: dict[str, str] = {}
            for cur in labels:
                best_label, best_value = labels[0], -1e18
                for prev in labels:
                    value = score[prev] + self._transition(prev, cur) + emissions[index][cur]
                    if value > best_value:
                        best_value, best_label = value, prev
                new_score[cur] = best_value
                pointer[cur] = best_label
            score = new_score
            back.append(pointer)

        last = max(score, key=lambda label: score[label])
        path = [last]
        for pointer in reversed(back):
            path.append(pointer[path[-1]])
        path.reverse()

        spans: list[tuple[int, int, str, float]] = []
        start = 0
        current = path[0]
        for index in range(1, length + 1):
            if index == length or path[index] != current:
                if current != "O":
                    entity = current[2:]
                    span_score = sum(emissions[i][path[i]] for i in range(start, index)) / max(1, index - start)
                    spans.append((start, index, entity, span_score))
                if index < length:
                    start, current = index, path[index]
        return spans


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

class ModelRegistry:
    """惰性加载所有模型；`available()` 给报告与 CLI 用。"""

    FILES = {
        "boundary": "boundary_clf.json",
        "risk": "risk_clf.json",
        "rerank": "rerank_ltr.json",
        "calibration": "calibration.json",
        "ner": "ner_perceptron.json",
    }

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or MODEL_DIR)
        self._cache: dict[str, Any] = {}
        self._loaded: dict[str, bool] = {}

    def path(self, name: str) -> Path:
        return self.model_dir / self.FILES[name]

    def boundary(self) -> BoundaryClassifier | None:
        return self._get("boundary", BoundaryClassifier.load)

    def risk(self) -> RiskClassifier | None:
        return self._get("risk", RiskClassifier.load)

    def rerank(self) -> RetrievalRanker | None:
        return self._get("rerank", RetrievalRanker.load)

    def calibration(self) -> Calibrator | None:
        return self._get("calibration", Calibrator.load)

    def ner(self) -> NerTagger | None:
        return self._get("ner", NerTagger.load)

    def _get(self, name: str, loader):
        if name in self._cache:
            return self._cache[name]
        value = loader(self.path(name))
        self._cache[name] = value
        self._loaded[name] = value is not None
        return value

    def available(self) -> dict[str, bool]:
        for name in self.FILES:
            if name not in self._loaded:
                getattr(self, name)()
        return dict(self._loaded)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model_dir": str(self.model_dir), "available": self.available()}
        for name in ("boundary", "risk", "rerank", "ner"):
            model = self._cache.get(name)
            if model is not None:
                out[f"{name}_meta"] = getattr(model, "train_meta", {})
        return out


_REGISTRY: ModelRegistry | None = None


def registry(model_dir: Path | None = None) -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None or model_dir is not None:
        _REGISTRY = ModelRegistry(model_dir)
    return _REGISTRY
