"""条款切分：规则引擎 + **全局动态规划分段** + 统计边界分类器兜底。

与 v1 `splitter.py` + `llm_splitter.py` 的差异
---------------------------------------------
v1 的切分本质上是**局部贪心**：从左到右逐对判断边界关系，`confidence` 低于阈值就标
`review_required`；LLM 只用来"回头重标有问题的窗口"（`_reflow_round`），一旦 LLM 不可用
就退化成纯规则。两个结构性缺陷：

* **局部最优不等于全局最优**：某处误切会让后面整段结构错位（v1 用 `_reflow_round`
  打补丁，但要求"可回流问题严格减少"才采纳，很多情况下根本不触发）；
* **无编号条款（政府合同常见）完全靠标点**：`prev` 以"。"结尾就当新条，粗暴。

v2 的做法：
1. 先给每个候选边界算三路证据：规则关系 + 统计分类器概率 + 版面几何；
2. 把"选哪些边界切开"变成一个**全局优化**：最大化 Σ边界对数几率 − λ·Σ|段落长度偏离先验|
   （即带长度正则的 Viterbi 分段），O(n·L) 可解，CPU 毫秒级；
3. 保留 v1 的贪心实现作为 `use_dp_segmentation=False` 的对照基线（消融实验直接用）。

这条升级对应文献里的 text segmentation / 结构化预测思路（见 docs/refs/literature.md
的 C 组与 B 组），并且在 `reports/` 里给出 DP vs 贪心的实测差异。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import features as F
from . import text as T
from .config import Config
from .learned import BoundaryClassifier
from .models import (
    BoundaryDecision,
    Clause,
    DocumentParse,
    SourceBlock,
    StructureIssue,
    stable_id,
)

# ---------------------------------------------------------------------------
# 标题识别
# ---------------------------------------------------------------------------

_HEADING_STYLE_HINTS = ("heading", "标题", "title", "subtitle", "论文1级", "论文2级")
_ARTICLE_TITLE_MAX = 40
_CN_ORDINAL_HEAD = ("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、")


@dataclass
class HeadingInfo:
    is_heading: bool
    label: str = ""
    title: str = ""
    level: int = 0
    kind: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def heading_text(self) -> str:
        return f"{self.label} {self.title}".strip()


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def detect_heading(block: SourceBlock, *, median_font: float | None = None) -> HeadingInfo:
    """判定一个 block 是否是条款标题，并给出层级。

    与 v1 的差别：v1 只看正则；v2 加入字号/加粗/样式名三路证据，因此"三、付款方式"
    这种手写编号、以及 Word 里用样式排的标题都能识别。
    """
    text = block.text.strip()
    if not text:
        return HeadingInfo(False)

    label_info = T.detect_label(text)
    label = str(label_info.get("label") or "")
    level = int(label_info.get("level") or 0)
    kind = str(label_info.get("kind") or "")
    evidence: list[str] = []
    confidence = 0.0

    if label:
        confidence += 0.55
        evidence.append(f"label:{label}")

    if len(text) <= _ARTICLE_TITLE_MAX:
        confidence += 0.15
        evidence.append("short")

    if not text.endswith(("。", "；", "：")):
        confidence += 0.05

    style_name = (block.style or "").lower()
    if any(hint in style_name for hint in _HEADING_STYLE_HINTS):
        confidence += 0.2
        evidence.append(f"style:{block.style}")

    if block.is_bold:
        confidence += 0.1
        evidence.append("bold")

    if median_font and block.font_size:
        ratio = block.font_size / median_font
        if ratio >= 1.15:
            confidence += 0.15
            evidence.append(f"font x{ratio:.2f}")
        elif ratio <= 0.9:
            confidence -= 0.05

    if text in _CN_ORDINAL_HEAD or any(text.startswith(head) for head in _CN_ORDINAL_HEAD):
        confidence += 0.3
        evidence.append("cn_ordinal")
        if not label:
            label = text[:2]
            level = level or 3
            kind = kind or "cn_ordinal"

    # 层级推断
    if not level:
        if kind == "chapter":
            level = 1
        elif kind == "article":
            level = 2
        elif kind == "decimal":
            level = min(4, 2 + label.count("."))
        elif kind == "item":
            level = 3
        elif kind == "cn_ordinal":
            level = 3

    if not label:
        # 没有编号但版面证据强 → 仍按标题处理（章节名之类的自由标题）
        if confidence >= 0.45 and len(text) <= 24 and not T.has_cjk(T.squash(text)[:0] or ""):
            pass
        if confidence < 0.5:
            return HeadingInfo(False, confidence=confidence, evidence=evidence)

    _, rest = T.label_heading_split(text)
    if not rest and label:
        rest = text[len(label):].strip(" ：:、.　")
    title = T.truncate(" ".join(rest.split()), _ARTICLE_TITLE_MAX, ellipsis="")
    return HeadingInfo(
        is_heading=True,
        label=label or text[:8],
        title=title,
        level=level or 2,
        kind=kind or "plain_heading",
        confidence=min(1.0, confidence),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# 规则边界
# ---------------------------------------------------------------------------

_TERMINAL = ("。", "；", "!", "！", "？", "?", "：", ":")


def rule_boundary(prev: SourceBlock, cur: SourceBlock) -> tuple[str, float, list[str]]:
    """纯规则给出的边界关系与置信度（v1 的判定逻辑，整理后保留作为基线通道）。"""
    prev_text = prev.text.strip()
    cur_text = cur.text.strip()
    evidence: list[str] = []

    if not prev_text or not cur_text:
        return BoundaryDecision.CONTINUATION, 0.5, ["empty_block"]

    label_info = T.detect_label(cur_text)
    kind = str(label_info.get("kind") or "")
    level = int(label_info.get("level") or 0)

    if kind == "chapter":
        return BoundaryDecision.NEW_CLAUSE, 0.97, [f"chapter:{label_info['label']}"]
    if kind == "article":
        return BoundaryDecision.NEW_CLAUSE, 0.95, [f"article:{label_info['label']}"]
    if kind == "decimal":
        return BoundaryDecision.NEW_CLAUSE, 0.88 if level >= 3 else 0.8, [f"decimal:{label_info['label']}"]
    if kind == "item":
        return BoundaryDecision.NEW_SUBCLAUSE, 0.86, [f"item:{label_info['label']}"]

    if cur_text.startswith(_CN_ORDINAL_HEAD):
        return BoundaryDecision.NEW_SUBCLAUSE, 0.84, ["cn_ordinal_head"]

    prev_terminal = prev_text.endswith(_TERMINAL)
    if not prev_terminal:
        # 上一段没结束 → 几乎必然是续行
        if prev_text.endswith(("，", ",", "、", "的", "和")):
            return BoundaryDecision.CONTINUATION, 0.9, ["prev_open_comma"]
        return BoundaryDecision.CONTINUATION, 0.72, ["prev_no_terminal"]

    if len(prev_text) <= 20 and len(cur_text) >= 40:
        return BoundaryDecision.NEW_CLAUSE, 0.6, ["short_prev_long_cur"]
    if cur_text.startswith(("现", "兹", "鉴于", "根据", "依照", "经")):
        return BoundaryDecision.NEW_CLAUSE, 0.62, ["recital_head"]
    if cur_text.startswith(("甲方", "乙方", "丙方", "双方")) and "：" in cur_text[:12]:
        return BoundaryDecision.NEW_CLAUSE, 0.7, ["party_block"]
    return BoundaryDecision.SAME_CLAUSE, 0.55, ["default_same"]


# ---------------------------------------------------------------------------
# 边界融合
# ---------------------------------------------------------------------------

@dataclass
class BoundaryEvidence:
    index: int
    decision: BoundaryDecision
    split_prob: float


def fuse_boundaries(
    blocks: Sequence[SourceBlock],
    config: Config,
    classifier: BoundaryClassifier | None = None,
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[BoundaryEvidence]:
    """对每对相邻 block 融合三路证据，输出切分概率。

    `split_prob` 是"这里应当开始一个新条款"的概率，供 DP 使用。
    """
    page_sizes = page_sizes or {}
    results: list[BoundaryEvidence] = []
    for index in range(1, len(blocks)):
        prev, cur = blocks[index - 1], blocks[index]
        relation, rule_conf, rule_evidence = rule_boundary(prev, cur)

        page_height = page_sizes.get(cur.page or 0, (0.0, 0.0))[1] or prev.metadata.get("page_height")
        feats = F.boundary_features(prev, cur, page_height=page_height)
        clf_prob = classifier.predict_proba(feats) if classifier else None

        rule_split = 1.0 if relation in (BoundaryDecision.NEW_CLAUSE, BoundaryDecision.NEW_SUBCLAUSE) else 0.0
        rule_prob = rule_conf if rule_split else (1.0 - rule_conf)

        if clf_prob is None:
            split_prob = rule_prob
            source = "rule"
        else:
            # 规则给强信号时以规则为主（分类器在训练集外容易过度自信）
            weight = 0.72 if rule_conf >= 0.9 else 0.45
            split_prob = weight * rule_prob + (1.0 - weight) * clf_prob
            source = "rule+classifier"

        # 版面几何：跨页/跨栏且字号跳变的边界更可能是新条款
        if prev.page != cur.page and rule_split == 0.0:
            split_prob = max(split_prob, 0.45)
            rule_evidence.append("page_break")
        if prev.font_size and cur.font_size and cur.font_size > prev.font_size * 1.15:
            split_prob = max(split_prob, 0.6)
            rule_evidence.append("font_jump")

        split_prob = max(0.0, min(1.0, split_prob))
        decision = BoundaryDecision(
            boundary_id=stable_id("bd", prev.block_id, cur.block_id),
            left_block_id=prev.block_id,
            right_block_id=cur.block_id,
            relation=relation if split_prob >= config.boundary_min_confidence or split_prob < 0.5 else BoundaryDecision.NEW_CLAUSE,
            confidence=round(split_prob if split_prob >= 0.5 else 1.0 - split_prob, 4),
            source=source,
            channel_scores={
                "rule": round(rule_conf, 4),
                "rule_split": rule_split,
                **({"classifier": round(clf_prob, 4)} if clf_prob is not None else {}),
            },
            features={k: round(v, 4) for k, v in feats.items()},
            evidence=rule_evidence,
            review_required=0.35 <= split_prob <= 0.65,
        )
        results.append(BoundaryEvidence(index=index, decision=decision, split_prob=split_prob))
    return results


# ---------------------------------------------------------------------------
# 分段：全局 DP（默认） / 贪心（对照）
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: int
    end: int
    blocks: list[SourceBlock]

    @property
    def text_length(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def dp_segment(
    blocks: Sequence[SourceBlock],
    split_probs: Sequence[float],
    config: Config,
) -> tuple[list[Segment], dict[str, Any]]:
    """带长度正则的全局分段（Viterbi）。

    `split_probs[i]` 表示第 i 个 block（i>=1）开始新条款的概率。目标函数：

        max  Σ_{切点 i} logit(p_i)  −  λ · Σ_{段} |log(len / prior)|

    约束：段长 ∈ [min_len, max_len]（末段可短）。
    """
    n = len(blocks)
    if n == 0:
        return [], {"segments": 0}
    if n == 1:
        return [Segment(0, 1, list(blocks))], {"segments": 1, "mode": "single"}

    eps = 1e-4
    logit = [0.0] * n
    for index in range(1, n):
        prob = min(1 - eps, max(eps, float(split_probs[index - 1])))
        logit[index] = math.log(prob / (1 - prob))
        logit[index] = max(-8.0, min(8.0, logit[index]))

    prefix = [0]
    for block in blocks:
        prefix.append(prefix[-1] + len(block.text) + 1)

    min_len = 12
    max_len = 4000
    prior = max(40.0, config.dp_len_prior)
    lam = config.dp_lambda_len

    neg_inf = float("-inf")
    best = [neg_inf] * (n + 1)
    choice = [-1] * (n + 1)
    best[0] = 0.0

    for end in range(1, n + 1):
        start = end - 1
        while start >= 0:
            length = prefix[end] - prefix[start]
            if length > max_len:
                break
            if length < min_len and end < n:
                start -= 1
                continue
            if best[start] == neg_inf:
                start -= 1
                continue
            penalty = lam * abs(math.log(max(1, length) / prior))
            score = best[start] + logit[start] + (-penalty if start > 0 else -penalty * 0.3)
            if score > best[end]:
                best[end] = score
                choice[end] = start
            start -= 1

    segments: list[Segment] = []
    cursor = n
    guard = 0
    while cursor > 0 and guard < n + 2:
        start = choice[cursor]
        if start < 0:
            start = 0
        segments.append(Segment(start=start, end=cursor, blocks=list(blocks[start:cursor])))
        cursor = start
        guard += 1
    segments.reverse()

    stats = {
        "mode": "dp",
        "segments": len(segments),
        "objective": round(best[n], 3) if best[n] != neg_inf else None,
        "lambda": lam,
        "len_prior": prior,
        "avg_len": round(sum(s.text_length for s in segments) / max(1, len(segments)), 1),
    }
    return segments, stats


def greedy_segment(
    blocks: Sequence[SourceBlock],
    decisions: Sequence[BoundaryEvidence],
    config: Config,
) -> tuple[list[Segment], dict[str, Any]]:
    """v1 式局部贪心（对照基线）。"""
    segments: list[Segment] = []
    start = 0
    for item in decisions:
        decision = item.decision
        split = (
            decision.relation in (BoundaryDecision.NEW_CLAUSE, BoundaryDecision.NEW_SUBCLAUSE)
            and decision.channel_scores.get("rule_split", 1.0) >= 0.5
            and item.split_prob >= config.boundary_min_confidence
        )
        if split:
            segments.append(Segment(start=start, end=item.index, blocks=list(blocks[start:item.index])))
            start = item.index
    if start < len(blocks):
        segments.append(Segment(start=start, end=len(blocks), blocks=list(blocks[start:])))
    stats = {
        "mode": "greedy",
        "segments": len(segments),
        "avg_len": round(sum(s.text_length for s in segments) / max(1, len(segments)), 1),
    }
    return segments, stats


# ---------------------------------------------------------------------------
# 组装条款树
# ---------------------------------------------------------------------------

def assemble_clauses(
    segments: Sequence[Segment],
    median_font: float | None = None,
) -> list[Clause]:
    """把分段结果组装成带父子关系的条款树。"""
    clauses: list[Clause] = []
    stack: list[Clause] = []

    for segment in segments:
        if not segment.blocks:
            continue
        first = segment.blocks[0]
        heading = detect_heading(first, median_font=median_font)
        text = "\n".join(b.text for b in segment.blocks).strip()
        if not text:
            continue
        tail = "\n".join(b.text for b in segment.blocks[1:]).strip()

        if heading.is_heading:
            level = heading.level
            label = heading.label
            rest = heading.title
            if len(rest) > 30:
                # 编号后面的内容其实是一整段正文（"1.1 乙方应当按照……"），只取首句做标题
                first_sentence = T.split_sentences(rest)[0] if rest else ""
                title = T.truncate(" ".join(first_sentence.split()), 24, ellipsis="")
            else:
                title = rest
            body = "\n".join(part for part in (rest, tail) if part).strip()
            heading_line = f"{label} {title}".strip() if label else title
            source = "rule"
        elif not clauses and ("合同" in text or "协议" in text):
            # 首部/封面块：标题 + 项目名称/当事人信息，单独成"前言"节点，避免污染第一条
            level = 1
            label = ""
            title = T.truncate(" ".join(text.split()), 24, ellipsis="")
            body = text
            heading_line = title
            source = "cover"
        else:
            level = 3
            label = ""
            title = T.truncate(" ".join(first.text.split()), 24, ellipsis="")
            heading_line = ""
            body = text
            source = "inferred"
            level = 3
            label = ""
            title = T.truncate(" ".join(first.text.split()), 24, ellipsis="")
            heading_line = ""
            body = text
            source = "inferred"

        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None

        clause = Clause(
            clause_id=stable_id("cl", segment.start, segment.end, label, text[:40]),
            label=label,
            title=title,
            level=level,
            heading=heading_line,
            body=body,
            text=text,
            block_ids=[b.block_id for b in segment.blocks],
            parent_id=parent.clause_id if parent else None,
            page_start=segment.blocks[0].page,
            page_end=segment.blocks[-1].page,
            source=source,
        )
        clauses.append(clause)
        stack.append(clause)
    return clauses


def _clause_text(blocks: Sequence[SourceBlock]) -> str:
    return "\n".join(b.text for b in blocks)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def split_document(
    blocks: Sequence[SourceBlock],
    config: Config,
    classifier: BoundaryClassifier | None = None,
    page_sizes: dict[int, tuple[float, float]] | None = None,
    source_name: str = "",
) -> DocumentParse:
    """解析 + 切分一步到位（blocks 需已过 layout 处理）。"""
    blocks = list(blocks)
    if not blocks:
        return DocumentParse(
            source_name=source_name,
            file_type="",
            full_text="",
            stats={"clauses": 0},
            warnings=["没有可切分的文本块"],
        )

    fonts = [b.font_size for b in blocks if b.font_size]
    median_font = _median(fonts) if fonts else None

    evidences = fuse_boundaries(blocks, config, classifier, page_sizes)
    split_probs = [item.split_prob for item in evidences]

    if config.use_dp_segmentation:
        segments, seg_stats = dp_segment(blocks, split_probs, config)
    else:
        segments, seg_stats = greedy_segment(blocks, evidences, config)

    clauses = assemble_clauses(segments, median_font=median_font)
    full_text = "\n".join(b.text for b in blocks)

    # 覆盖率检查：每个 block 必须落在且仅落在一个条款里
    covered: dict[str, int] = {}
    for clause in clauses:
        for block_id in clause.block_ids:
            covered[block_id] = covered.get(block_id, 0) + 1
    issues: list[StructureIssue] = []
    missing = [b.block_id for b in blocks if b.block_id not in covered]
    if missing:
        issues.append(
            StructureIssue(
                code="block_uncovered",
                message=f"{len(missing)} 个文本块没有落到任何条款里",
                severity="high",
                block_ids=missing[:20],
            )
        )
    duplicated = [bid for bid, count in covered.items() if count > 1]
    if duplicated:
        issues.append(
            StructureIssue(
                code="block_duplicated",
                message=f"{len(duplicated)} 个文本块被重复归属",
                severity="high",
                block_ids=duplicated[:20],
            )
        )

    review_bounds = [e.decision for e in evidences if e.decision.review_required]
    stats = {
        "blocks": len(blocks),
        "clauses": len(clauses),
        "levels": {
            level: sum(1 for c in clauses if c.level == level)
            for level in sorted({c.level for c in clauses})
        },
        "boundaries": len(evidences),
        "review_required_boundaries": len(review_bounds),
        "avg_clause_len": round(
            sum(len(c.text) for c in clauses) / max(1, len(clauses)), 1
        ),
        "segmentation": seg_stats,
        "median_font": median_font,
        "law_like": bool(any(T.detect_label(b.text)["kind"] == "article" for b in blocks)),
    }

    return DocumentParse(
        source_name=source_name,
        file_type="",
        full_text=full_text,
        blocks=blocks,
        clauses=clauses,
        boundaries=[e.decision for e in evidences],
        issues=issues,
        stats=stats,
    )
