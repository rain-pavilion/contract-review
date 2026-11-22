"""特征工程：切分边界特征 + 条款风险特征 + 要素抽取上下文特征。

为什么要单独一个模块
--------------------
v1 的规则写在 `splitter.py` 里，训练脚本（`scripts/generate_*.py`）只生成数据、不做
特征，导致"规则阈值"和"数据分布"两套东西对不上，改一个阈值就只能重跑全量看效果。
v2 把特征抽出来做成**纯函数 + 稳定特征名**，训练/推理/消融共用同一份实现，模型权重
可以落盘（`models/boundary_clf.json`），阈值与权重都能单独评测。

特征命名约定
------------
`b_` 前缀 = 边界（boundary）特征，`r_` = 风险（risk）特征，`e_` = 要素（element）特征。
只放**可解释、可单独统计**的特征，不做 embedding（CPU 环境、小样本，embedding 特征
会过拟合且不可解释）。
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from . import text as T
from .models import Clause, SourceBlock

# ---------------------------------------------------------------------------
# 边界特征
# ---------------------------------------------------------------------------

#: 边界特征的固定顺序。改这个列表必须同步 +1 `FEATURE_SCHEMA_VERSION`。
BOUNDARY_FEATURES: tuple[str, ...] = (
    "b_cur_has_label",
    "b_cur_label_level",
    "b_cur_label_is_article",
    "b_cur_label_is_chapter",
    "b_cur_label_is_decimal",
    "b_cur_label_is_item",
    "b_cur_starts_cn_num",
    "b_cur_short",
    "b_cur_very_short",
    "b_cur_has_colon",
    "b_cur_has_verb",
    "b_prev_ends_terminal",
    "b_prev_short",
    "b_prev_very_short",
    "b_prev_is_heading_like",
    "b_cur_is_heading_like",
    "b_prev_len_log",
    "b_cur_len_log",
    "b_len_ratio",
    "b_gap_y_norm",
    "b_same_column",
    "b_font_delta",
    "b_bold_delta",
    "b_cur_numeric_density",
    "b_cur_has_party_kw",
    "b_cur_has_obligation_kw",
    "b_duplicate_ratio",
)

FEATURE_SCHEMA_VERSION = 2

_OBLIGATION_KW = (
    "应当", "必须", "不得", "有权", "负责", "应", "须", "承诺", "保证", "同意",
)
_PARTY_KW = (
    "甲方", "乙方", "丙方", "双方", "当事人", "承包人", "发包人", "供应商", "采购人",
)
_VERB_RE = re.compile(r"(支付|履行|交付|验收|违约|赔偿|解除|终止|签订|承担|提供|完成)")
_CN_NUM_HEAD = re.compile(r"^[一二三四五六七八九十]{1,3}[、.．]")


def _safe_log(value: float) -> float:
    return math.log1p(max(0.0, value))


def boundary_features(
    prev_block: SourceBlock,
    cur_block: SourceBlock,
    *,
    page_height: float | None = None,
) -> dict[str, float]:
    """抽取一处边界（prev | cur）的特征。

    全部返回 float，确保可以直接喂给 numpy 矩阵；布尔用 0/1。
    """
    prev_text = prev_block.text.strip()
    cur_text = cur_block.text.strip()
    label = T.detect_label(cur_text)
    label_text = str(label.get("label") or "")
    label_level = float(label.get("level") or 0)
    kind = str(label.get("kind") or "")

    prev_ends_terminal = 1.0 if prev_text.endswith(("。", "；", "!", "！", "？", "?", "”", "》")) else 0.0
    prev_is_heading = 1.0 if (0 < len(prev_text) <= 28 and not prev_text.endswith("。")) else 0.0
    cur_is_heading = 1.0 if (0 < len(cur_text) <= 30 and bool(label_text)) else 0.0
    prev_len = len(prev_text)
    cur_len = len(cur_text)

    gap_norm = 0.0
    if prev_block.bbox and cur_block.bbox and page_height:
        gap = cur_block.bbox[1] - prev_block.bbox[3]
        gap_norm = max(-0.5, min(1.0, gap / max(1.0, page_height)))

    font_delta = 0.0
    bold_delta = 0.0
    if prev_block.font_size and cur_block.font_size:
        font_delta = (cur_block.font_size - prev_block.font_size) / max(1.0, prev_block.font_size)
    bold_delta = float(int(cur_block.is_bold) - int(prev_block.is_bold))

    digits = len(re.findall(r"\d", cur_text))
    numeric_density = digits / max(1, cur_len)

    return {
        "b_cur_has_label": 1.0 if label_text else 0.0,
        "b_cur_label_level": label_level,
        "b_cur_label_is_article": 1.0 if kind == "article" else 0.0,
        "b_cur_label_is_chapter": 1.0 if kind == "chapter" else 0.0,
        "b_cur_label_is_decimal": 1.0 if kind == "decimal" else 0.0,
        "b_cur_label_is_item": 1.0 if kind == "item" else 0.0,
        "b_cur_starts_cn_num": 1.0 if _CN_NUM_HEAD.match(cur_text) else 0.0,
        "b_cur_short": 1.0 if cur_len <= 40 else 0.0,
        "b_cur_very_short": 1.0 if cur_len <= 14 else 0.0,
        "b_cur_has_colon": 1.0 if ("：" in cur_text[:30] or ":" in cur_text[:30]) else 0.0,
        "b_cur_has_verb": 1.0 if _VERB_RE.search(cur_text) else 0.0,
        "b_prev_ends_terminal": prev_ends_terminal,
        "b_prev_short": 1.0 if prev_len <= 40 else 0.0,
        "b_prev_very_short": 1.0 if prev_len <= 14 else 0.0,
        "b_prev_is_heading_like": prev_is_heading,
        "b_cur_is_heading_like": cur_is_heading,
        "b_prev_len_log": _safe_log(prev_len),
        "b_cur_len_log": _safe_log(cur_len),
        "b_len_ratio": min(8.0, _safe_log(cur_len) / max(0.1, _safe_log(prev_len))),
        "b_gap_y_norm": gap_norm,
        "b_same_column": 1.0 if prev_block.column == cur_block.column else 0.0,
        "b_font_delta": font_delta,
        "b_bold_delta": bold_delta,
        "b_cur_numeric_density": numeric_density,
        "b_cur_has_party_kw": 1.0 if T.contains_any(cur_text[:20], _PARTY_KW) else 0.0,
        "b_cur_has_obligation_kw": 1.0 if T.contains_any(cur_text, _OBLIGATION_KW) else 0.0,
        "b_duplicate_ratio": T.overlap_coefficient(prev_text, cur_text, n=4),
    }


def features_to_vector(features: dict[str, float], names: Sequence[str] = BOUNDARY_FEATURES) -> list[float]:
    """按固定顺序取特征向量（缺失补 0，保证维度恒定）。"""
    return [float(features.get(name, 0.0)) for name in names]


# ---------------------------------------------------------------------------
# 条款风险特征
# ---------------------------------------------------------------------------

RISK_FEATURES: tuple[str, ...] = (
    "r_len_log",
    "r_sentence_count",
    "r_digit_density",
    "r_punct_density",
    "r_has_amount",
    "r_has_date",
    "r_has_party",
    "r_ratio_density",
    "r_exemption_kw",
    "r_one_sided_kw",
    "r_waiver_kw",
    "r_unilateral_kw",
    "r_vague_kw",
    "r_penalty_kw",
    "r_dispute_kw",
    "r_confidential_kw",
    "r_obligation_balance",
)

_EXEMPTION = ("免责", "不承担任何责任", "概不负责", "免除全部", "不承担")
_ONE_SIDED = ("甲方有权单方", "单方面解除", "仅甲方有权", "乙方不得", "无权要求")
_WAIVER = ("放弃", "视为放弃", "不得主张", "放弃追究")
_UNILATERAL = ("单方变更", "单方调整", "有权随时变更", "自行决定")
_VAGUE = ("及时", "尽快", "合理期限", "适当", "必要时", "视情况", "另行约定", "协商确定")
_PENALTY = ("违约金", "赔偿", "罚", "滞纳")
_DISPUTE = ("争议", "仲裁", "诉讼", "管辖")
_CONFIDENTIAL = ("保密", "商业秘密", "不得泄露")


def risk_features(clause: Clause | str) -> dict[str, float]:
    text = clause.text if isinstance(clause, Clause) else str(clause)
    length = len(text)
    sentences = max(1, len(T.split_sentences(text)))
    digits = len(re.findall(r"\d", text))
    puncts = len(re.findall(r"[，。；：、,.;:]", text))
    a_party = text.count("甲方")
    b_party = text.count("乙方")
    balance = abs(a_party - b_party) / max(1, a_party + b_party)
    return {
        "r_len_log": _safe_log(length),
        "r_sentence_count": float(sentences),
        "r_digit_density": digits / max(1, length),
        "r_punct_density": puncts / max(1, length),
        "r_has_amount": 1.0 if re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:元|万|亿)", text) else 0.0,
        "r_has_date": 1.0 if T.parse_date(text) is not None else 0.0,
        "r_has_party": 1.0 if T.contains_any(text[:40], _PARTY_KW) else 0.0,
        "r_ratio_density": len(re.findall(r"\d+(?:\.\d+)?%", text)) / max(1, sentences),
        "r_exemption_kw": 1.0 if T.contains_any(text, _EXEMPTION) else 0.0,
        "r_one_sided_kw": 1.0 if T.contains_any(text, _ONE_SIDED) else 0.0,
        "r_waiver_kw": 1.0 if T.contains_any(text, _WAIVER) else 0.0,
        "r_unilateral_kw": 1.0 if T.contains_any(text, _UNILATERAL) else 0.0,
        "r_vague_kw": 1.0 if T.contains_any(text, _VAGUE) else 0.0,
        "r_penalty_kw": 1.0 if T.contains_any(text, _PENALTY) else 0.0,
        "r_dispute_kw": 1.0 if T.contains_any(text, _DISPUTE) else 0.0,
        "r_vague_kw_count": float(sum(text.count(k) for k in _VAGUE)),
        "r_confidential_kw": 1.0 if T.contains_any(text, _CONFIDENTIAL) else 0.0,
        "r_obligation_balance": balance,
    }


def risk_vector(text: Clause | str) -> list[float]:
    feats = risk_features(text)
    return [float(feats.get(name, 0.0)) for name in RISK_FEATURES]


# ---------------------------------------------------------------------------
# 要素上下文特征
# ---------------------------------------------------------------------------

def element_context_features(text: str, index: int) -> dict[str, float]:
    """要素候选位置的上下文特征（供感知机 NER 用）。"""
    window = text[max(0, index - 12): index + 12]
    left = text[max(0, index - 30): index]
    return {
        "e_has_role_left": 1.0 if re.search(r"(甲方|乙方|采购人|供应商|名称|地址|电话|账号)", left) else 0.0,
        "e_has_currency": 1.0 if re.search(r"[￥¥]|人民币|RMB", window) else 0.0,
        "e_has_date_kw": 1.0 if re.search(r"(年|月|日|日期|期限)", window) else 0.0,
        "e_has_amount_unit": 1.0 if re.search(r"(元|万元|亿元|角|分)", window) else 0.0,
        "e_window_len": float(len(window)),
    }
