"""法条检索：三路召回 + 加权 RRF + 学习型重排（v1 的升级版）。

v1 的做法（`contract_review/legal_kb.py`）是 BM25 + BGE-M3 向量两路，用等权 RRF 融合，
向量通道依赖 Ollama 上的 BGE-M3。它的短板在报告里写得很清楚：**Recall@5 只有 0.600**。

v2 在这一层做四件事（每条都能对应到文献，见 `docs/refs/literature.md`）：

1. **查询改写**：把"待审条款"改写成 (行为要件, 主体, 标的, 法律后果) 四元组，
   再拼上风险标签词与合同类型词（依据 LegalSearchLM 的 query reformulation、田萱的
   分段定位思想）。这一步不训练任何模型，纯规则填槽，却是召回率提升最大的一步。
2. **三路召回**：稀疏（BM25，字 unigram+bigram）+ 语义（LSA/TF-IDF+SVD，无需 BGE-M3
   也能跑）+ **标签图**（法条自带 risk_tags，构标签-法条二部图，做 Personalized
   PageRank 扩展）。第三路是 v1 完全没有的通道。
3. **加权 RRF**：v1 是等权 `1/(k+rank)`；v2 用可网格搜索的通道权重 + 可调 k，
   并保留 `channel_scores` 便于消融。
4. **两阶段精排**：三路融合取 top-N 候选，再用 14 维特征 + LTR 权重重排取 top-k
   （依据李琳等 DeepLawRec 的"召回 + 再选择"两阶段结构；cross-encoder 在 CPU 上太贵，
   这里用线性 LTR 作为它的廉价替身）。

任何一步缺失都自动降级，降级路径写进 `RetrievalStats.degraded`，报告里可以逐条解释。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import text as T
from .config import MODEL_DIR, Config, ensure_dirs
from .learned import (
    RetrievalRanker,
    default_ranker,
    registry,
)
from .models import Clause, ContractElements, QueryPlan, RetrievalHit, Statute

# ---------------------------------------------------------------------------
# 查询改写：四元组槽位
# ---------------------------------------------------------------------------

#: 行为要件词（条款在"做什么"）
ACTION_TERMS: tuple[tuple[str, str], ...] = (
    ("支付", "价款支付"), ("付款", "价款支付"), ("结算", "价款支付"),
    ("交付", "标的交付"), ("验收", "标的验收"), ("发货", "标的交付"),
    ("保管", "保管义务"), ("储存", "保管义务"), ("寄存", "保管义务"),
    ("出租", "租赁"), ("承租", "租赁"), ("转租", "转租"),
    ("委托", "委托事务"), ("授权", "代理权限"), ("代理", "代理权限"),
    ("解除", "合同解除"), ("终止", "合同终止"), ("变更", "合同变更"),
    ("违约", "违约责任"), ("赔偿", "损害赔偿"), ("罚款", "违约责任"),
    ("担保", "担保"), ("抵押", "担保"), ("质押", "担保"), ("保证", "担保"),
    ("保密", "保密义务"), ("披露", "信息披露"), ("公开", "信息披露"),
    ("转让", "权利转让"), ("许可", "授权许可"), ("使用", "使用权"),
    ("中介", "中介服务"), ("居间", "中介服务"), ("促成", "中介服务"),
    ("预售", "商品房预售"), ("备案", "登记备案"), ("登记", "登记"),
    ("招标", "招标投标"), ("投标", "招标投标"), ("采购", "政府采购"),
)

#: 法律后果词（条款违反后会触发什么）
CONSEQUENCE_TERMS: tuple[tuple[str, str], ...] = (
    ("无效", "无效"), ("可撤销", "可撤销"), ("撤销", "可撤销"),
    ("解除", "解除权"), ("终止", "终止"),
    ("赔偿", "赔偿责任"), ("补偿", "补偿"),
    ("返还", "返还财产"), ("退还", "返还财产"),
    ("承担", "承担责任"), ("负责", "承担责任"),
    ("逾期", "迟延履行"), ("迟延", "迟延履行"), ("延期", "迟延履行"),
    ("减免", "责任减免"), ("免责", "责任减免"), ("不承担", "责任减免"),
    ("罚则", "行政处罚"), ("处罚", "行政处罚"),
)

#: 主体词
SUBJECT_TERMS: tuple[tuple[str, str], ...] = (
    ("甲方", "合同当事人"), ("乙方", "合同当事人"), ("丙方", "合同当事人"),
    ("出租人", "出租人"), ("承租人", "承租人"), ("委托人", "委托人"),
    ("受托人", "受托人"), ("寄存人", "寄存人"), ("保管人", "保管人"),
    ("中介人", "中介人"), ("采购人", "采购人"), ("供应商", "供应商"),
    ("出卖人", "出卖人"), ("买受人", "买受人"), ("劳动者", "劳动者"),
    ("用人单位", "用人单位"),
)

#: 条款里出现这些词时，把风险标签加进查询（标签词本身就是检索线索）
RISK_TAG_HINTS: dict[str, tuple[str, ...]] = {
    "zero_rent": ("租金", "数额", "无偿"),
    "zero_deposit": ("押金", "保证金"),
    "term_over_20y": ("租赁期限", "二十年"),
    "deposit_over_20pct": ("定金", "百分之二十"),
    "penalty_excessive": ("违约金", "过分高于"),
    "unverified_property": ("权属", "产权", "处分权"),
    "unilateral_entry": ("擅自进入", "单方进入", "占有"),
    "liability_exemption": ("免责", "格式条款", "不承担"),
    "no_permit": ("预售许可", "许可", "批准"),
    "date_reversed": ("履行期限", "起始", "届满"),
    "fee_mismatch": ("费用", "价款", "总价"),
    "impossible_deadline": ("履行期限", "交付时间"),
    "insecure_delivery": ("毁损", "灭失", "风险"),
    "personal_data_no_consent": ("个人信息", "同意"),
    "public_disclosure": ("保密", "披露"),
    "quantity_value_mismatch": ("数量", "质量", "价款"),
    "retrieval_before_storage": ("保管凭证", "交付保管物"),
    "zero_records": ("凭证", "收据"),
    "negative_fee": ("报酬", "费用"),
    "commission_100pct": ("中介报酬", "促成"),
    "success_guarantee": ("如实报告", "隐瞒"),
    "unlimited_authority": ("权限", "指示", "超越"),
    "no_report": ("报告义务", "报告"),
    "nonexistent_subject": ("标的", "成立"),
    "perpetual_license": ("使用期限",),
    "cash_only": ("支付方式",),
    "zero_quantity": ("数量",),
    "zero_fee": ("报酬",),
    "abnormal_price": ("价格", "市场价"),
    "illegal_terms": ("违法", "强制性规定"),
}

TEMPLATE_TYPE_WORDS: dict[str, str] = {
    "housing_rent": "房屋租赁",
    "presale": "商品房预售",
    "sale": "房屋买卖",
    "entrust": "委托",
    "intermediary": "中介",
    "custody": "保管",
    "data_provide": "数据提供",
    "purchase": "政府采购",
    "tech_develop": "技术开发",
    "通用": "合同",
}


def _hits_of(text: str, table: Sequence[tuple[str, str]]) -> list[str]:
    """按出现顺序返回命中词槽（去重、保序）。"""
    found: list[tuple[int, str]] = []
    for keyword, slot in table:
        index = text.find(keyword)
        if index >= 0:
            found.append((index, slot))
    found.sort()
    out: list[str] = []
    for _, slot in found:
        if slot not in out:
            out.append(slot)
    return out


def rewrite_query(
    clause_text: str,
    *,
    contract_type: str = "通用",
    elements: ContractElements | None = None,
    codes: Sequence[str] = (),
) -> QueryPlan:
    """把条款改写成 (行为要件, 主体, 标的, 法律后果) 四元组 + 标签词。

    这是"零训练但收益最大"的一步：原条款多是"甲方应当于验收后十五个工作日内支付"
    这种口语化长句，直接拿去检索法条会和"当事人应当按照约定支付价款"这种法条表述
    字面不重叠。拆成槽位后，`术语 + 标签词` 组合能显著提高字面与语义的匹配率。
    """
    action = _hits_of(clause_text, ACTION_TERMS)[:4]
    consequence = _hits_of(clause_text, CONSEQUENCE_TERMS)[:3]
    subject = _hits_of(clause_text, SUBJECT_TERMS)[:3]
    if elements is not None:
        for party in elements.parties[:2]:
            label = party.extra.get("role_label") or party.role
            if label and label not in subject:
                subject.append(str(label))
        for amount in elements.amounts[:1]:
            if amount.value and amount.value not in subject:
                subject.append(amount.value[:12])

    object_terms: list[str] = []
    subject_thing = re.sub(r"\s+", "", clause_text)
    for key in ("项目名称", "标的", "货物", "服务", "房屋", "保管物", "艺术品", "设备", "数据"):
        if key in subject_thing:
            object_terms.append(key)
    if elements is not None:
        for item in elements.subjects[:1]:
            fragment = item.normalized[:16]
            if fragment:
                object_terms.append(fragment)

    tags: list[str] = list(codes)
    for code, keywords in RISK_TAG_HINTS.items():
        if any(keyword in clause_text for keyword in keywords) and code not in tags:
            tags.append(code)

    type_word = TEMPLATE_TYPE_WORDS.get(contract_type, "合同")
    parts: list[str] = []
    parts.extend(action)
    parts.extend(subject)
    parts.extend(object_terms)
    parts.extend(consequence)
    parts.extend(tags[:6])
    if type_word and type_word != "合同":
        parts.append(type_word)
    rewritten = " ".join(dict.fromkeys(part for part in parts if part))
    if len(rewritten) < 6:  # 改写失败则退回原文（并标注）
        rewritten = T.truncate(clause_text, 120)
    return QueryPlan(
        raw=clause_text,
        rewritten=rewritten,
        templates=list(action),
        tags=tags,
        contract_types=[contract_type],
        slots={
            "action": action,
            "subject": subject,
            "object": object_terms,
            "consequence": consequence,
        },
    )


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

_BM25_K1 = 1.5
_BM25_B = 0.72
AUTHORITY_WEIGHT: dict[str, float] = {
    "法律": 1.0,
    "行政法规": 0.9,
    "司法解释": 0.85,
    "部门规章": 0.75,
    "地方性法规": 0.6,
    "地方规章": 0.55,
    "规范性文件": 0.5,
}


@dataclass
class CorpusStats:
    documents: int = 0
    tokens: int = 0
    avg_len: float = 0.0
    vocab: int = 0
    tags: int = 0
    laws: int = 0


class StatuteIndex:
    """法条索引：BM25 + LSA + 标签图，全部可在 CPU/无网环境构建。"""

    def __init__(self, statutes: Sequence[Statute | Mapping[str, Any]], config: Config) -> None:
        self.config = config
        self.statutes: list[dict[str, Any]] = []
        for item in statutes:
            data = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            if not data.get("content"):
                continue
            self.statutes.append(data)
        self.n = len(self.statutes)
        self._build_postings()
        self._build_lsa()
        if getattr(config, "use_bge", False):
            self._build_bge()
        else:
            self.bge_ready = False
            self.bge_matrix = None
            self.bge_meta = {}
        self._build_tag_graph()
        self.stats = CorpusStats(
            documents=self.n,
            tokens=self.total_len,
            avg_len=round(self.avgdl, 2),
            vocab=len(self.idf),
            tags=len(self.tag_postings),
            laws=len(self.law_postings),
        )

    # -- 稀疏通道 ---------------------------------------------------------
    def _doc_text(self, doc: Mapping[str, Any]) -> str:
        parts = [
            str(doc.get("law_name", "")),
            str(doc.get("article_no", "")),
            str(doc.get("content", "")),
            " ".join(doc.get("risk_tags", []) or []),
            " ".join(doc.get("contract_types", []) or []),
        ]
        return " ".join(parts)

    def _build_postings(self) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.total_len = 0
        for doc_id, doc in enumerate(self.statutes):
            tokens = T.tokenize(self._doc_text(doc))
            counts = Counter(tokens)
            self.doc_len.append(len(tokens))
            self.total_len += len(tokens)
            for token, freq in counts.items():
                self.postings[token].append((doc_id, freq))
        self.avgdl = (self.total_len / self.n) if self.n else 1.0
        self.idf: dict[str, float] = {}
        for token, posting in self.postings.items():
            df = len(posting)
            self.idf[token] = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def bm25(self, query: str, limit: int | None = None) -> list[tuple[int, float]]:
        tokens = T.tokenize(query)
        if not tokens:
            return []
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokens):
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf.get(token, 0.0)
            weight = 1.0 + 0.4 * (tokens.count(token) - 1)  # 词频饱和
            for doc_id, freq in posting:
                length = self.doc_len[doc_id] or 1
                denom = freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / self.avgdl)
                scores[doc_id] += weight * idf * (freq * (_BM25_K1 + 1) / denom)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit] if limit else ranked

    # -- 语义通道 ---------------------------------------------------------
    def _build_lsa(self) -> None:
        self.lsa_ready = False
        self.lsa_matrix = None
        self.lsa_vectorizer = None
        if not self.config.use_lsa:
            return
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
        except Exception:  # pragma: no cover - sklearn 缺失时走稀疏通道
            return
        texts = [self._doc_text(doc) for doc in self.statutes]
        if not texts:
            return
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 3), min_df=2, max_features=60000,
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return
        dim = max(8, min(self.config.lsa_dim, max(8, min(matrix.shape) - 1)))
        svd = TruncatedSVD(n_components=dim, random_state=self.config.seed)
        reduced = svd.fit_transform(matrix)
        self.lsa_matrix = normalize(reduced)
        self.lsa_vectorizer = vectorizer
        self.lsa_svd = svd
        self.lsa_ready = True

    def lsa(self, query: str, limit: int | None = None) -> list[tuple[int, float]]:
        if not self.lsa_ready or self.lsa_matrix is None:
            return []
        vector = self.lsa_vectorizer.transform([query])
        reduced = self.lsa_svd.transform(vector)
        norm = float((reduced ** 2).sum() ** 0.5) or 1.0
        scores = (self.lsa_matrix @ (reduced[0] / norm)).tolist()
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit] if limit else ranked

    # -- BGE-M3 密集通道（存在预计算向量时优先于 LSA）-----------------------
    def _build_bge(self) -> None:
        """读 `models/bge_m3/statutes.npy`；没有就退回 LSA（降级留账）。"""
        self.bge_ready = False
        self.bge_matrix = None
        try:
            from . import bert_embed
        except Exception:  # pragma: no cover
            return
        try:
            matrix, meta = bert_embed.load_vectors()
        except Exception:
            return
        if matrix is None or getattr(matrix, "shape", (0,))[0] != self.n:
            return
        self.bge_matrix = matrix
        self.bge_meta = meta
        self.bge_ready = True

    def dense(self, query: str, limit: int | None = None) -> list[tuple[int, float]]:
        """密集通道入口：有 BGE-M3 向量就走 BGE，否则用 LSA。"""
        if not self.bge_ready or self.bge_matrix is None:
            return self.lsa(query, limit)
        try:
            from . import bert_embed

            embedder = bert_embed.shared_embedder()
            vector = embedder.encode([query])[0]
        except Exception:
            return self.lsa(query, limit)
        scores = (self.bge_matrix @ vector).tolist()
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit] if limit else ranked

    # -- 标签图通道 -------------------------------------------------------
    def _build_tag_graph(self) -> None:
        self.tag_postings: dict[str, set[int]] = defaultdict(set)
        self.law_postings: dict[str, set[int]] = defaultdict(set)
        self.doc_tags: list[set[str]] = []
        self.adjacency: dict[int, set[int]] = defaultdict(set)
        for doc_id, doc in enumerate(self.statutes):
            tags = set(doc.get("risk_tags") or [])
            self.doc_tags.append(tags)
            for tag in tags:
                self.tag_postings[tag].add(doc_id)
            law = str(doc.get("law_name", ""))
            self.law_postings[law].add(doc_id)
            # 同法相邻条文互连（"相邻条文常同主题"）
            for neighbour in (doc_id - 1, doc_id + 1):
                if 0 <= neighbour < self.n and str(self.statutes[neighbour].get("law_name", "")) == law:
                    self.adjacency[doc_id].add(neighbour)
                    self.adjacency[neighbour].add(doc_id)

    def tag_channel(self, tags: Sequence[str], contract_type: str = "通用",
                    limit: int | None = 60) -> list[tuple[int, float]]:
        """标签命中 + PPR 扩展。v1 只有"命中即追加"，这里是带权扩散。"""
        seeds: dict[int, float] = defaultdict(float)
        for tag in tags:
            for doc_id in self.tag_postings.get(tag, ()):  # 标签直接命中
                seeds[doc_id] += 1.0
        for doc_id, doc in enumerate(self.statutes):
            types = doc.get("contract_types") or []
            if contract_type and contract_type in types:
                seeds[doc_id] += 0.35
        if not seeds:
            return []
        total = sum(seeds.values()) or 1.0
        rank = {doc_id: value / total for doc_id, value in seeds.items()}
        alpha = self.config.tag_ppr_alpha
        for _ in range(max(1, self.config.tag_ppr_iters)):
            nxt: dict[int, float] = defaultdict(float)
            for doc_id, weight in rank.items():
                neighbours = self.adjacency.get(doc_id)
                if not neighbours:
                    continue
                share = alpha * weight / len(neighbours)
                for neighbour in neighbours:
                    nxt[neighbour] += share
            # 保留种子分数（阻尼的另一半）
            for doc_id, weight in rank.items():
                nxt[doc_id] += (1 - alpha) * weight
            rank = dict(nxt)
        ranked = sorted(rank.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit] if limit else ranked

    def expand_by_tags(self, hits: Sequence[Any], limit: int = 2) -> list[dict[str, Any]]:
        """标签扩展：把与已召回条文**共享 risk_tags** 的条文追加进来。

        v1 的 `LegalTagGraph.expand` 就是干这个的（`--tag-expansion-limit` 默认 2），
        这一步对 Recall@k 影响很大 —— 强金标准常是"一组相邻条文"，只靠文本匹配
        很难全都进 top-k，但它们的标签相同。
        """
        if limit <= 0 or not hits:
            return []
        known = {str(getattr(hit, "article_uid", "")) for hit in hits}
        wanted: dict[str, float] = {}
        for hit in hits:
            for tag in getattr(hit, "risk_tags", ()) or ():
                wanted[tag] = wanted.get(tag, 0.0) + 1.0
        if not wanted:
            return []
        scored: list[tuple[float, int]] = []
        for doc_id, doc in enumerate(self.statutes):
            uid = str(doc.get("article_uid", ""))
            if uid in known:
                continue
            overlap = sum(wanted.get(tag, 0.0) for tag in (doc.get("risk_tags") or ()))
            if overlap:
                scored.append((overlap, doc_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [dict(self.statutes[doc_id]) for _score, doc_id in scored[:limit]]

    # -- 持久化 -----------------------------------------------------------
    def save(self, directory: Path) -> None:
        ensure_dirs()
        directory.mkdir(parents=True, exist_ok=True)
        import numpy as np

        payload = {
            "config": {"lsa_dim": self.config.lsa_dim, "seed": self.config.seed},
            "ids": [doc.get("article_uid") for doc in self.statutes],
            "stats": self.stats.__dict__,
            "lsa_ready": self.lsa_ready,
        }
        (directory / "v2_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.lsa_ready and self.lsa_matrix is not None:
            np.save(directory / "v2_lsa.npy", self.lsa_matrix)


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    hits: list[RetrievalHit] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


CHANNEL_WEIGHTS: dict[str, float] = {"sparse": 1.0, "dense": 0.9, "tag": 0.55}

#: 训练阶段网格搜索出来的融合参数（不存在时用手写先验 `CHANNEL_WEIGHTS`）
FUSION_WEIGHTS_FILE = "fusion_weights.json"


def trained_fusion() -> dict[str, Any] | None:
    """读取 `models/fusion_weights.json`（由 `train.py` 的网格搜索产出）。"""
    path = MODEL_DIR / FUSION_WEIGHTS_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload.get("weights"), dict):
        return None
    return payload


class Retriever:
    """一路：BM25；二路：LSA；三路：标签图 → 加权 RRF → LTR 重排。"""

    def __init__(
        self,
        index: StatuteIndex,
        config: Config,
        ranker: RetrievalRanker | None = None,
        *,
        channel_weights: Mapping[str, float] | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.index = index
        self.config = config
        package_ids = {str(row.get("package_id") or "") for row in index.statutes}
        self.source_package_id = (next(iter(package_ids)) if len(package_ids) == 1 and "" not in package_ids else "")
        # Package statutes have source provenance but no package-trained labels,
        # fusion or LTR validation. Preserve the complete user's terms and do not
        # transfer unrelated learned ranking or inferred risk tags into this corpus.
        self.ranker = None if self.source_package_id else (ranker or default_ranker())
        fusion = trained_fusion() if channel_weights is None and not self.source_package_id else None
        if self.source_package_id:
            self.weights = {"sparse": 1.0, "dense": 0.0, "tag": 0.0}
        elif channel_weights is not None:
            self.weights = dict(channel_weights)
        elif fusion:
            self.weights = {name: float(value) for name, value in fusion["weights"].items()}
        else:
            self.weights = dict(CHANNEL_WEIGHTS)
        if self.source_package_id:
            self.rrf_k = config.rrf_k
        elif rrf_k is not None:
            self.rrf_k = rrf_k
        elif fusion and fusion.get("rrf_k"):
            self.rrf_k = int(fusion["rrf_k"])
        else:
            self.rrf_k = config.rrf_k

    # -- 融合 -------------------------------------------------------------
    def _fuse(
        self,
        channels: Mapping[str, list[tuple[int, float]]],
    ) -> list[tuple[int, dict[str, float], float]]:
        rows: dict[int, dict[str, float]] = defaultdict(dict)
        for name, ranked in channels.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                rows[doc_id][name] = score
                rows[doc_id][f"{name}_rank"] = float(rank)
                rows[doc_id]["rrf"] = rows[doc_id].get("rrf", 0.0) + (
                    self.weights.get(name, 0.5) / (self.rrf_k + rank)
                )
        fused = sorted(rows.items(), key=lambda pair: (-pair[1].get("rrf", 0.0), pair[0]))
        return [(doc_id, feats, feats.get("rrf", 0.0)) for doc_id, feats in fused]

    # -- 重排特征 ---------------------------------------------------------
    def _features(
        self,
        doc: Mapping[str, Any],
        plan: QueryPlan,
        feats: Mapping[str, float],
        *,
        query_len: int,
    ) -> dict[str, float]:
        content = str(doc.get("content", ""))
        doc_tags = set(doc.get("risk_tags") or [])
        hit_tags = doc_tags & set(plan.tags)
        numbers = re.findall(r"\d+", content)
        query_numbers = re.findall(r"\d+", plan.rewritten)
        overlap = T.overlap_coefficient(plan.rewritten, content)
        authority = AUTHORITY_WEIGHT.get(str(doc.get("authority_level", "")), 0.5)
        length_penalty = abs(len(content) - 220) / 600.0
        law_match = 1.0 if str(doc.get("law_name", "")) and str(doc["law_name"]) in plan.rewritten else 0.0
        expansion = 1.0 if feats.get("tag") and not (feats.get("sparse") or feats.get("dense")) else 0.0
        # IDF 加权覆盖率：修"高频中心条文（hub）被误排第一"。单纯字符重合率会让
        # 出现过"租赁/押金"字样的条文都拿高分（如第七百一十七条在多个租赁查询里
        # 都排第一），但"二十年"这类罕见词只出现在一处。用 IDF 给"只被命中罕见词"
        # 的条文加权，能把真正的对应条文顶上去。
        query_tokens = [token for token in T.tokenize(plan.rewritten) if len(token) >= 2]
        idf_total = 0.0
        idf_hit = 0.0
        for token in query_tokens:
            weight = self.index.idf.get(token, 1.0)
            idf_total += weight
            if token in content:
                idf_hit += weight
        idf_overlap = (idf_hit / idf_total) if idf_total else 0.0
        return {
            "q_bm25": feats.get("sparse", 0.0),
            "q_bm25_rank": feats.get("sparse_rank", 0.0),
            "q_lsa_cos": feats.get("dense", 0.0),
            "q_lsa_rank": feats.get("dense_rank", 0.0),
            "q_rrf": feats.get("rrf", 0.0),
            "q_tag_hit": float(len(hit_tags)),
            "q_tag_ratio": (len(hit_tags) / len(plan.tags)) if plan.tags else 0.0,
            "q_law_match": law_match,
            "q_num_match": float(len(set(numbers) & set(query_numbers))),
            "q_len_penalty": length_penalty,
            "q_query_len": float(query_len),
            "q_authority": authority,
            "q_expansion": expansion,
            "q_char_overlap": overlap,
            "q_idf_overlap": idf_overlap,
        }

    # -- 对外接口 ---------------------------------------------------------
    def search(
        self,
        plan: QueryPlan,
        *,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.config.top_k
        requested_rerank = self.config.use_rerank if use_rerank is None else use_rerank
        use_rerank = False if self.source_package_id else requested_rerank
        query = plan.raw if self.source_package_id else (plan.rewritten if self.config.use_query_rewrite else plan.raw)
        limit = self.config.retrieve_candidate_limit

        channels: dict[str, list[tuple[int, float]]] = {
            "sparse": self.index.bm25(query, limit=limit),
            "dense": [] if self.source_package_id else self.index.dense(query, limit=limit),
            "tag": [] if self.source_package_id else self.index.tag_channel(plan.tags, (plan.contract_types[0] if plan.contract_types else "通用"), limit=limit),
        }
        fused = self._fuse(channels)
        candidates = fused[:limit]

        scored: list[tuple[RetrievalHit, float]] = []
        feature_rows: list[tuple[int, dict[str, float]]] = []
        feature_plan = replace(plan, rewritten=query, tags=[]) if self.source_package_id else plan
        for doc_id, feats, rrf in candidates:
            doc = self.index.statutes[doc_id]
            feature_map = self._features(doc, feature_plan, feats, query_len=len(query))
            feature_map["q_rrf"] = rrf
            feature_rows.append((doc_id, feature_map))

        # 重排：候选集内列标准化后线性打分（与训练口径一致）；无模型时退回 RRF 分值
        rerank_scores: dict[int, float] = {}
        if use_rerank and self.ranker is not None and self.ranker.weights:
            if self.ranker.normalized:
                ordered = self.ranker.rank_normalized(
                    [(doc_id, feats) for doc_id, feats in feature_rows]
                )
                rerank_scores = {doc_id: value for doc_id, value in ordered}
            else:
                for doc_id, feats in feature_rows:
                    value = self.ranker.score(feats)
                    if value is not None:
                        rerank_scores[doc_id] = value

        rrf_of = {doc_id: rrf for doc_id, _feats, rrf in candidates}
        for doc_id, feature_map in feature_rows:
            doc = self.index.statutes[doc_id]
            feats = {k: v for k, v in feature_map.items()}
            score = rerank_scores.get(doc_id)
            if score is None:
                score = rrf_of[doc_id]
            hit = RetrievalHit(
                article_uid=str(doc.get("article_uid", "")),
                law_name=str(doc.get("law_name", "")),
                article_no=str(doc.get("article_no", "")),
                content=str(doc.get("content", "")),
                score=float(score),
                channels={k: float(v) for k, v in feats.items()},
                risk_tags=list(doc.get("risk_tags") or []),
                source_url=str(doc.get("source_url", "")),
                authority_level=str(doc.get("authority_level", "")),
                provenance=statute_provenance(doc),
            )
            scored.append((hit, feature_map["q_char_overlap"]))

        scored.sort(key=lambda pair: (-pair[0].score, pair[0].article_uid))
        hits = []
        for position, (hit, _) in enumerate(scored[:top_k], start=1):
            hit.rank = position
            hits.append(hit)

        # 标签扩展（v1 `--tag-expansion-limit` 的等价实现）：把与已召回条文共享
        # risk_tags 的条文追加到尾部，扩展项的 score 置为最小分值以下，保证不抢占
        # 真正的文本命中位次。
        expanded: list[RetrievalHit] = []
        expansion_limit = int(getattr(self.config, "tag_expansion_limit", 2) or 0)
        if expansion_limit > 0 and hits and not self.source_package_id:
            floor = min(hit.score for hit in hits)
            for doc in self.index.expand_by_tags(hits, limit=expansion_limit):
                expanded.append(
                    RetrievalHit(
                        article_uid=str(doc.get("article_uid", "")),
                        law_name=str(doc.get("law_name", "")),
                        article_no=str(doc.get("article_no", "")),
                        content=str(doc.get("content", "")),
                        score=floor - 1e-6,
                        channels={"tag_expansion": 1.0},
                        risk_tags=list(doc.get("risk_tags") or []),
                        source_url=str(doc.get("source_url", "")),
                        authority_level=str(doc.get("authority_level", "")),
                        provenance=statute_provenance(doc),
                    )
                )
            for offset, hit in enumerate(expanded, start=len(hits) + 1):
                hit.rank = offset
            hits.extend(expanded)
        degraded: list[str] = []
        if not self.index.lsa_ready and not self.source_package_id:
            degraded.append("dense:disabled-by-config" if not self.config.use_lsa else "dense:unavailable")
        if not self.index.tag_postings and not self.source_package_id:
            degraded.append("tag:no-tags")
        if self.ranker is None and not self.source_package_id:
            degraded.append("rerank:no-model")
        stats = {
            "query": query,
            "slots": plan.slots,
            "tags": [] if self.source_package_id else plan.tags,
            "channels": {name: len(ranked) for name, ranked in channels.items()},
            "candidates": len(candidates),
            "returned": len(hits),
            "rerank": bool(use_rerank),
            "degraded": degraded,
            "weights": dict(self.weights),
            "rrf_k": self.rrf_k,
            "retrieval_profile": "source_package_raw_bm25_v1" if self.source_package_id else "legacy_multichannel",
            "package_id": self.source_package_id or None,
            "query_rewrite_applied": bool(self.config.use_query_rewrite and not self.source_package_id),
            "trained_rerank_applied": bool(use_rerank and self.ranker and self.ranker.weights),
        }
        if self.source_package_id:
            stats["source_package_policy"] = {
                "reason": "新来源法条库尚无适配的人工标签与重排验证；保留原查询，暂不迁移旧库训练排序。",
                "ranking": "bm25_rank_with_single_channel_rrf_score",
                "learned_rerank": False, "trained_fusion": False,
                "risk_tag_channel": False, "dense_channel": False,
                "disabled_by_policy": ["lossy_query_rewrite", "legacy_ltr", "legacy_trained_fusion", "dense", "risk_tag_expansion"],
                "ignored_query_tags": list(plan.tags),
                "requested_settings": {"use_query_rewrite": self.config.use_query_rewrite,
                                       "use_lsa": self.config.use_lsa, "use_bge": getattr(self.config, "use_bge", False),
                                       "use_rerank": bool(requested_rerank)},
                "meaning": "可审计的原文检索基线；不是新训练重排，也不证明法条适用或正式准确率。",
            }
        return RetrievalResult(hits=hits, stats=stats)


# ---------------------------------------------------------------------------
# 索引构建／加载
# ---------------------------------------------------------------------------

def statute_provenance(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the original file, extraction quality and locator on each hit."""
    fields = ("package_id", "source_id", "original_sha256", "text_sha256", "text_source_id",
              "source_status_statement", "applicability_status", "extraction_quality", "location")
    return {key: doc[key] for key in fields if key in doc}


def load_statutes(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("content"):
                out.append(record)
    return out


def build_index(
    statute_file: Path | str,
    config: Config | None = None,
) -> StatuteIndex:
    config = config or Config()
    statutes = load_statutes(statute_file)
    return StatuteIndex(statutes, config)


_CACHE: dict[str, StatuteIndex] = {}


def get_index(
    statute_file: Path | str,
    config: Config | None = None,
    *,
    refresh: bool = False,
) -> StatuteIndex:
    config = config or Config()
    path = Path(statute_file).resolve()
    stamp = path.stat()
    key = f"{path}|{stamp.st_size}|{stamp.st_mtime_ns}|{config.lsa_dim}|{config.seed}|{config.use_lsa}|{config.use_bge}"
    if refresh or key not in _CACHE:
        _CACHE[key] = build_index(statute_file, config)
    return _CACHE[key]


def get_retriever(
    statute_file: Path | str,
    config: Config | None = None,
    *,
    refresh: bool = False,
) -> Retriever:
    config = config or Config()
    index = get_index(statute_file, config, refresh=refresh)
    ranker = registry().rerank() or default_ranker()
    return Retriever(index, config, ranker)
