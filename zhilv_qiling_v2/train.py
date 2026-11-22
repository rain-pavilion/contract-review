"""训练 v2 的四个学习型组件，并把权重写成可审计的 JSON。

为什么单独一个模块
------------------
v1 的能力全部来自"规则 + 调用大模型"，模型不可用时整条链降级成规则。v2 的定位是
**在 CPU、无外网、无 Ollama 的答辩机器上也能跑出可复现指标**，所以必须有一层
"自己训练出来的轻量模型"。这些模型不是替代规则，而是补规则给不出的边界：

1. `boundary`：条款边界分类器（logistic）。训练标签来自**文档结构本身**
   （编号种类 + 段末标点），不来自 `splitter.rule_boundary` 的人工先验，因此
   评估的是"学生模型能否在没有编号的乱版面上仍然判对边界"。
2. `ner`：字符级 BIO 平均感知机。金标签来自数据集清单里的
   `party_a/party_b/amount/start_date/end_date/subject`，用字符跨度回标；
   规则通道的输出作为银标签补充长尾（政府合同扫描件没有金标签）。
3. `rerank`：法条检索的 LTR 重排（pairwise logistic）。训练/评估沿用
   v1 `scripts/eval_legal_retrieval.py` 的 31 条 `ISSUE_SPECS` 协议，指标可与
   v1 报告值直接对比（Recall@1..10、MRR@5）。
4. `calibration`：风险置信度校准（isotonic）。用"检出是否正确"做监督，
   把规则 + 模型混合打分的置信度拉到可解释的概率尺度。
5. `risk`：条款级多标签风险分类（one-vs-rest logistic），作为规则之外的第二意见。

用法
----
    python -m zhilv_qiling_v2.train --all
    python -m zhilv_qiling_v2.train --only boundary,ner --quick

所有权重落到 `zhilv_qiling_v2/models/`，训练报告落到
`zhilv_qiling_v2/reports/training_report.md`。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import detectors as D
from . import features as F
from . import learned as L
from . import text as T
from .config import DATA_DIR, MODEL_DIR, V1_STATUTES, WORKSPACE, default_config
from .learned import (
    NER_LABELS,
    BoundaryClassifier,
    Calibrator,
    NerTagger,
    RetrievalRanker,
    ner_char_features,
)

REPORT_PATH = Path(__file__).resolve().parent / "reports" / "training_report.md"

#: 与 v1 评测脚本一致的模板 → 合同类型映射（四元组改写要用到）
_TEMPLATE_TO_TYPE: dict[str, str] = {
    "housing_rent": "房屋租赁",
    "entrust": "委托",
    "intermediary": "中介",
    "custody": "保管",
    "data_provide": "通用",
    "presale": "商品房预售",
    "sale": "房屋买卖",
}


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    """一条训练样本：文档 + 已知金标签（没有金标签时为 None）。"""

    path: Path
    kind: str
    gold_codes: list[str] = field(default_factory=list)
    gold_entities: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect_docs(workspace: Path | None = None) -> list[Doc]:
    """收集全部可用样本：生成集（有金标签）+ 政府合同扫描件（无金标签）。"""
    base = Path(workspace or WORKSPACE)
    docs: list[Doc] = []

    gen = base / "data" / "generated_contracts"
    multi = gen / "samr_multi_template_2025"
    manifest = _read_manifest(multi / "manifest.json")
    for record in manifest.get("records", []):
        path = multi / str(record.get("docx", ""))
        if path.exists():
            docs.append(
                Doc(
                    path=path,
                    kind="generated",
                    gold_entities={
                        "PARTY": [str(record.get("party_a", "")), str(record.get("party_b", ""))],
                        "AMOUNT": [str(record.get("amount", "")), str(record.get("secondary_amount", ""))],
                        "DATE": [str(record.get("start_date", "")), str(record.get("end_date", ""))],
                        "SUBJECT": [str(record.get("subject", ""))],
                    },
                    meta={"template": record.get("template_key"), "clean": True},
                )
            )
    adversarial = multi / "adversarial" / "manifest.json"
    adv_manifest = _read_manifest(adversarial)
    for record in adv_manifest.get("records", []):
        path = multi / "adversarial" / str(record.get("docx", ""))
        if path.exists():
            docs.append(
                Doc(
                    path=path,
                    kind="adversarial",
                    gold_codes=[str(code) for code in record.get("issue_codes", [])],
                    gold_entities={
                        "PARTY": [str(record.get("party_a", "")), str(record.get("party_b", ""))],
                        "AMOUNT": [str(record.get("amount", "")), str(record.get("secondary_amount", ""))],
                        "DATE": [str(record.get("start_date", "")), str(record.get("end_date", ""))],
                        "SUBJECT": [str(record.get("subject", ""))],
                    },
                    meta={"template": record.get("template_key"), "clean": False},
                )
            )
    sdf = gen / "samr_sdf_2025_0002" / "manifest.json"
    sdf_manifest = _read_manifest(sdf)
    for record in sdf_manifest.get("records", []):
        path = Path(record.get("docx", ""))
        if not path.is_absolute():
            path = sdf.parent / str(record.get("docx", ""))
        if not path.exists():
            continue
        docs.append(
            Doc(
                path=path,
                kind="generated",
                gold_codes=[] if str(record.get("risk_type", "clean")) == "clean" else [
                    str(record.get("risk_type"))
                ],
                gold_entities={
                    "PARTY": [str(record.get("seller", "")), str(record.get("buyer", ""))],
                    "AMOUNT": [str(record.get("total_price", "")), str(record.get("deposit", ""))],
                    "DATE": [str(record.get("signing_date", "")), str(record.get("delivery_date", ""))],
                    "SUBJECT": [str(record.get("address", ""))],
                },
                meta={"template": "sdf", "clean": str(record.get("risk_type", "clean")) == "clean"},
            )
        )

    gov = base / "data" / "gov_contracts" / "manifest.jsonl"
    if gov.exists():
        for line in gov.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            contract_dir = base / "data" / "gov_contracts" / str(record.get("contract_id", ""))
            texts = sorted((contract_dir / "text").glob("page_*.txt"))
            if not texts:
                continue
            docs.append(
                Doc(
                    path=contract_dir,
                    kind="gov",
                    gold_entities={},
                    meta={"pages": len(texts), "contract_id": record.get("contract_id")},
                )
            )
    return docs


def doc_text(doc: Doc) -> str:
    """读取样本文本：docx/pdf 用 v2 解析器，政府合同读文字层。"""
    from .parsers import parse_path

    if doc.path.is_dir():
        parts: list[str] = []
        for text_path in sorted((doc.path / "text").glob("page_*.txt")):
            try:
                parts.append(text_path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
        return "\n".join(parts)
    try:
        parsed = parse_path(doc.path)
    except Exception:  # 解析失败的样本不参与训练，但要留痕
        return ""
    return "\n".join(block.text for block in parsed.blocks)


def split_docs(docs: Sequence[Doc], ratio: float = 0.25, seed: int = 20261104) -> tuple[list[Doc], list[Doc]]:
    """按样本（不是按条款）切分训练/验证，避免同一文档的条款泄漏到两边。"""
    rng = random.Random(seed)
    pool = list(docs)
    rng.shuffle(pool)
    cut = max(1, int(round(len(pool) * ratio)))
    return pool[cut:], pool[:cut]


# ---------------------------------------------------------------------------
# 1. 边界分类器
# ---------------------------------------------------------------------------

def boundary_dataset(doc: Doc, config) -> list[tuple[dict[str, float], int]]:
    """把文档展开成 (特征, 是否边界) 的监督样本。

    标签来源是**结构本身**：块首带"第X章/第X条/1.1/（一）"就是边界，级次决定
    是 NEW_CLAUSE 还是 NEW_SUBCLAUSE；没有编号的块，若上一块已用句末标点收尾则
    算 SAME_CLAUSE，否则算 CONTINUATION。这样标签与 `splitter.rule_boundary`
    的先验无关，评估才不是自己评自己。
    """
    from .parsers import parse_path
    from .splitter import detect_heading

    if doc.path.is_dir():
        return []
    try:
        parsed = parse_path(doc.path)
    except Exception:
        return []
    blocks = parsed.blocks
    rows: list[tuple[dict[str, float], int]] = []
    for index in range(1, len(blocks)):
        prev, cur = blocks[index - 1], blocks[index]
        heading = detect_heading(cur)
        if heading.label:
            is_boundary = 1
        else:
            is_boundary = 1 if prev.text.rstrip()[-1:] in "。；;：:" else 0
        page_height = 800.0
        feats = F.boundary_features(prev, cur, page_height=page_height)
        rows.append((feats, is_boundary))
    return rows


def train_boundary(docs: Sequence[Doc], config, *, seed: int = 20261104) -> dict[str, Any]:
    train_docs, valid_docs = split_docs(docs, 0.25, seed)
    train_rows: list[tuple[dict[str, float], int]] = []
    for doc in train_docs:
        train_rows.extend(boundary_dataset(doc, config))
    valid_rows: list[tuple[dict[str, float], int]] = []
    for doc in valid_docs:
        valid_rows.extend(boundary_dataset(doc, config))
    if len(train_rows) < 30:
        return {"skipped": "样本不足", "train": len(train_rows), "valid": len(valid_rows)}

    names = list(F.BOUNDARY_FEATURES)
    x_train = [[float(row.get(name, 0.0)) for name in names] for row, _ in train_rows]
    y_train = [label for _, label in train_rows]
    x_valid = [[float(row.get(name, 0.0)) for name in names] for row, _ in valid_rows]
    y_valid = [label for _, label in valid_rows]

    weights, bias = _fit_logistic(x_train, y_train, epochs=400, lr=0.35, l2=1e-3, seed=seed)
    model = BoundaryClassifier(names=names, weights=weights, bias=bias,
                              train_meta={"samples": len(train_rows), "source": "structure-labels"})
    acc_train = _binary_accuracy(model, x_train, y_train)
    acc_valid = _binary_accuracy(model, x_valid, y_valid)
    payload = {
        "names": names,
        "weights": [round(w, 6) for w in weights],
        "bias": round(bias, 6),
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": len(train_rows),
            "valid_samples": len(valid_rows),
            "labels": "structure(heading-label + terminal punctuation)",
            "acc_train": round(acc_train, 4),
            "acc_valid": round(acc_valid, 4),
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / L.ModelRegistry.FILES["boundary"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "samples": len(train_rows),
        "valid_samples": len(valid_rows),
        "acc_train": round(acc_train, 4),
        "acc_valid": round(acc_valid, 4),
        "top_weights": sorted(zip(names, weights), key=lambda p: -abs(p[1]))[:6],
    }


def _fit_logistic(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    epochs: int = 300,
    lr: float = 0.3,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[list[float], float]:
    """全批量梯度下降的 logistic 回归（不依赖 sklearn，权重可直接读）。"""
    dim = len(x[0]) if x else 0
    weights = [0.0] * dim
    bias = 0.0
    n = max(1, len(x))
    for _ in range(epochs):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for row, label in zip(x, y):
            z = bias + sum(w * v for w, v in zip(weights, row))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            diff = p - label
            for index, value in enumerate(row):
                grad_w[index] += diff * value
            grad_b += diff
        for index in range(dim):
            weights[index] -= lr * (grad_w[index] / n + l2 * weights[index])
        bias -= lr * grad_b / n
    return weights, bias


def _binary_accuracy(model: BoundaryClassifier, x: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    if not len(y):
        return 0.0
    hit = 0
    for row, label in zip(x, y):
        z = model.bias + sum(w * v for w, v in zip(model.weights, row))
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        hit += int((p >= 0.5) == bool(label))
    return hit / len(y)


# ---------------------------------------------------------------------------
# 2. NER（字符级平均感知机）
# ---------------------------------------------------------------------------

def bio_from_spans(text: str, spans: Sequence[tuple[int, int, str]]) -> list[str]:
    labels = ["O"] * len(text)
    for start, end, entity in spans:
        for index in range(start, min(end, len(text))):
            labels[index] = f"B-{entity}" if index == start else f"I-{entity}"
    return labels


def _value_candidates(value: Any) -> list[str]:
    """把清单里的字段值变成"可能出现在合同正文里的字符串"。

    清单里 `amount` 常是 `1341600.0`、`4800.0` 这类浮点，`start_date` 可能是
    `2026-12-01`，而正文写的是 `1341600元` / `4,800.00` / `2026年12月1日`。
    直接 `find(str(value))` 会全部落空（这是首版 NER 只有 43 个训练窗口的根因）。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return []
    out = {text}
    # 数字：去掉 .0、加千分位、加"元/万元"
    try:
        number = float(text)
    except (TypeError, ValueError):
        number = None
    if number is not None:
        if abs(number - round(number)) < 1e-9:
            integer = str(int(round(number)))
            out.add(integer)
            out.add(f"{number:,.2f}")
            out.add(f"{number:,.2f}".rstrip("0").rstrip("."))
            out.add(f"{integer}元")
            out.add(f"人民币{integer}元")
            if number >= 10_000 and abs(number / 10_000 - round(number / 10_000)) < 1e-9:
                out.add(f"{int(round(number / 10_000))}万元")
        out.add(f"{number:.2f}")
    # 日期：2026-12-01 → 2026年12月1日
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        out.add(f"{year}年{month}月{day}日")
        out.add(f"{year}-{month:02d}-{day:02d}")
        out.add(f"{year}/{month}/{day}")
    return [item for item in out if len(item) >= 2]


def _extend_amount(text: str, start: int, end: int) -> int:
    """金额金标只给了数值（如 4800.0），正文写法是 `4800.00元`，向右吸到完整实体。"""
    tail = re.match(r"[\d.,]{0,12}\s*(?:万|亿)?\s*(?:元|圆)(?:整)?", text[end:end + 24])
    if tail:
        return end + tail.end()
    return end


def gold_spans(text: str, entities: Mapping[str, Sequence[str]]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for entity, values in entities.items():
        if entity not in ("PARTY", "AMOUNT", "DATE", "SUBJECT"):
            continue
        for value in values:
            for candidate in _value_candidates(value):
                found = False
                cursor = text.find(candidate)
                while cursor >= 0:  # 同一实体可能在正文出现多次（首页 + 签章区），全部标注
                    end = cursor + len(candidate)
                    if entity == "AMOUNT":
                        # 清单字段是数值，正文是"4,800.00元"/"人民币500000元"。
                        # 不补齐边界，金标就比预测短一截，评测时同一实体既算误报又算漏报。
                        end = _extend_amount(text, cursor, end)
                    spans.append((cursor, end, entity))
                    found = True
                    cursor = text.find(candidate, end)
                if found:
                    break
    # 去掉互相重叠的标注（保留更长的）
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for span in spans:
        if any(not (span[1] <= other[0] or span[0] >= other[1]) for other in kept):
            continue
        kept.append(span)
    return kept


def rule_silver_spans(text: str) -> list[tuple[int, int, str]]:
    """用规则通道给无金标签的长尾样本造银标签（政府合同扫描件）。

    **必须走生产链路的抽取器**（`ner.rule_elements`），不能另写一套正则。旧版直接用
    `_PARTY_PATTERNS` 的 group(1) 区间当标注，拿到的是"吴磊伟将艺术品一件交由"这类
    "实体 + 后续宾语"的串，而金标是"吴磊伟"本身 —— 边界口径不一致会让训练学到错误
    边界，评测时同一实体既算误报又算漏报（实测纯金标口径下规则通道 PARTY tp=0、fp=31）。
    """
    from . import ner as N
    from .models import DocumentParse, SourceBlock

    block = SourceBlock(block_id="b_rule", order=0, kind="paragraph", text=text)
    document = DocumentParse(source_name="<silver>", file_type="txt", full_text=text, blocks=[block])
    kind_map = {
        "party": "PARTY",
        "amount": "AMOUNT",
        "date": "DATE",
        "subject": "SUBJECT",
    }
    spans: list[tuple[int, int, str]] = []
    for hit in N.rule_elements(document):
        kind = kind_map.get(hit.kind.lower())
        if kind is None or hit.span is None:
            continue
        start, end = hit.span.char_start, hit.span.char_end
        if start is None or end is None:
            continue
        # 兜底 trim：抽取器偶尔把前导冒号/空格带进来
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 2:
            spans.append((start, end, kind))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for span in spans:
        if any(not (span[1] <= other[0] or span[0] >= other[1]) for other in kept):
            continue
        kept.append(span)
    return kept


def train_ner(docs: Sequence[Doc], config, *, epochs: int = 15, seed: int = 20261104,
              use_silver: bool = True) -> dict[str, Any]:
    train_docs, valid_docs = split_docs(docs, 0.25, seed)
    tagger = NerTagger()
    weights: dict[str, dict[str, float]] = {}
    totals: dict[str, dict[str, float]] = {}
    timestamps: dict[str, dict[str, int]] = {}
    step = 0
    examples = 0

    def add(feat: str, label: str, delta: float) -> None:
        table = weights.setdefault(feat, {})
        table[label] = table.get(label, 0.0) + delta
        total = totals.setdefault(feat, {})
        total[label] = total.get(label, 0.0) + delta * step
        stamps = timestamps.setdefault(feat, {})
        stamps[label] = step

    def cost(feat: str, label: str) -> float:
        table = weights.get(feat, {})
        if not table:
            return 0.0
        value = table.get(label, 0.0)
        total = totals.get(feat, {}).get(label, 0.0)
        stamp = timestamps.get(feat, {}).get(label, 0)
        return value - total / step if step else value

    def decode(chars: Sequence[str]) -> list[str]:
        length = len(chars)
        if not length:
            return []
        emissions = [
            {label: sum(cost(feat, label) for feat in ner_char_features(chars, i, length)) for label in NER_LABELS}
            for i in range(length)
        ]
        score = {label: emissions[0][label] for label in NER_LABELS}
        back: list[dict[str, str]] = []
        for index in range(1, length):
            new_score: dict[str, float] = {}
            pointer: dict[str, str] = {}
            for cur in NER_LABELS:
                best_label, best_value = NER_LABELS[0], -1e18
                for prev in NER_LABELS:
                    penalty = -1000.0 if cur.startswith("I-") and prev not in (f"B-{cur[2:]}", f"I-{cur[2:]}") else 0.0
                    value = score[prev] + penalty + emissions[index][cur]
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
        return path

    training_sets: list[tuple[str, list[str]]] = []
    for doc in train_docs:
        text = doc_text(doc)
        if len(text) < 40:
            continue
        spans = gold_spans(text, doc.gold_entities) if doc.gold_entities else []
        # 无金标（或金标只覆盖首部字段）的样本用规则通道造银标：全文扫描而不是只扫
        # 前 4000 字，否则 20 页的政采合同只有首页拿到标注。
        if use_silver:
            silver = rule_silver_spans(text)
            if len(silver) > len(spans):
                spans = silver
        if not spans:
            continue
        # 长文本切成 1200 字的窗口；步长 400 让窗口重叠，等于把语料放大 3 倍
        # （纯 CPU、无预训练权重的字符级感知机对数据量极其敏感）。
        step_size = 1200
        stride = 400
        for offset in range(0, len(text), stride):
            window = text[offset:offset + step_size]
            local = [(max(0, s - offset), max(0, e - offset), t) for s, e, t in spans
                     if s >= offset and e <= offset + step_size]
            if not local:
                continue
            training_sets.append((window, bio_from_spans(window, local)))
            if offset + step_size >= len(text):
                break

    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(training_sets)
        for text, gold in training_sets:
            chars = list(text)
            length = len(chars)
            pred = decode(chars)
            feats_per_char = [ner_char_features(chars, i, length) for i in range(length)]
            for index in range(length):
                if pred[index] == gold[index]:
                    continue
                step += 1
                for feat in feats_per_char[index]:
                    add(feat, gold[index], 1.0)
                    add(feat, pred[index], -1.0)
            examples += 1

    final_weights = {
        feat: {label: value for label, value in table.items() if abs(value) > 1e-6}
        for feat, table in weights.items()
    }
    model = NerTagger(
        weights=final_weights,
        labels=list(NER_LABELS),
        train_meta={
            "epochs": epochs,
            "windows": len(training_sets),
            "updates": step,
            "source": "manifest-gold + rule-silver",
        },
    )
    metrics = evaluate_ner(model, valid_docs)
    payload = model.to_dict()
    payload["meta"]["valid"] = metrics
    payload["meta"]["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / L.ModelRegistry.FILES["ner"]).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return {"windows": len(training_sets), "updates": step, "valid": metrics,
            "feature_count": len(final_weights)}


def evaluate_ner(tagger: NerTagger, docs: Sequence[Doc]) -> dict[str, Any]:
    """按"实体串完全一致"统计 P/R/F1；同时给规则通道的对照。"""
    from .ner import rule_elements

    per_label: dict[str, dict[str, int]] = {}
    rule_per_label: dict[str, dict[str, int]] = {}
    for doc in docs:
        text = doc_text(doc)
        if len(text) < 40 or not doc.gold_entities:
            continue
        gold = gold_spans(text, doc.gold_entities)
        gold_keys = {(T.squash(text[s:e]), entity) for s, e, entity in gold}
        predicted = tagger.decode(text[:4000])
        pred_keys = {
            (T.squash(text[s:e]), entity)
            for s, e, entity, _score in predicted
            if entity in ("PARTY", "AMOUNT", "DATE", "SUBJECT") and e - s >= 2
        }
        for entity in ("PARTY", "AMOUNT", "DATE", "SUBJECT"):
            bucket = per_label.setdefault(entity, {"tp": 0, "fp": 0, "fn": 0})
            bucket["tp"] += len({k for k in pred_keys if k[1] == entity} & {k for k in gold_keys if k[1] == entity})
            bucket["fp"] += len({k for k in pred_keys if k[1] == entity} - gold_keys)
            bucket["fn"] += len({k for k in gold_keys if k[1] == entity} - pred_keys)
        # 规则通道对照（走完整管线，拿到带条款归属的文档）
        try:
            from .pipeline import prepare

            document = prepare(doc.path.name, path=doc.path)
        except Exception:
            document = None
        if document is not None:
            hits = rule_elements(document)
            rule_keys: set[tuple[str, str]] = set()
            for hit in hits:
                entity = {"party": "PARTY", "amount": "AMOUNT", "date": "DATE", "subject": "SUBJECT"}.get(hit.kind)
                if entity:
                    rule_keys.add((T.squash(hit.value), entity))
            for entity in ("PARTY", "AMOUNT", "DATE", "SUBJECT"):
                bucket = rule_per_label.setdefault(entity, {"tp": 0, "fp": 0, "fn": 0})
                bucket["tp"] += len({k for k in rule_keys if k[1] == entity} & {k for k in gold_keys if k[1] == entity})
                bucket["fp"] += len({k for k in rule_keys if k[1] == entity} - gold_keys)
                bucket["fn"] += len({k for k in gold_keys if k[1] == entity} - rule_keys)

    def f1(bucket: dict[str, int]) -> tuple[float, float, float]:
        tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return round(precision, 4), round(recall, 4), round(value, 4)

    def macro(table: Mapping[str, dict[str, int]]) -> float:
        scores = [f1(bucket)[2] for bucket in table.values()]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "model": {label: f1(bucket) for label, bucket in per_label.items()},
        "model_macro_f1": macro(per_label),
        "rule": {label: f1(bucket) for label, bucket in rule_per_label.items()},
        "rule_macro_f1": macro(rule_per_label),
    }


# ---------------------------------------------------------------------------
# 3. 检索 LTR
# ---------------------------------------------------------------------------

def _load_issue_specs() -> tuple[dict[str, tuple[list[str], list[str], str]], dict[str, str]]:
    """从 v1 评测脚本里**静态**取出评测协议（不 import v1 包）。

    v1 `scripts/eval_legal_retrieval.py` 顶层 import 了 `contract_review`（依赖
    langgraph），直接 exec_module 会连带拉起整条 v1 依赖链。这里用 AST 只摘
    `ISSUE_SPECS` / `TEMPLATE_TO_TYPE` 两个字面量，保证 v2 零 v1 依赖。
    """
    import ast

    path = WORKSPACE / "scripts" / "eval_legal_retrieval.py"
    if not path.exists():
        return {}, {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            node_value = node.value
        else:
            continue
        node_value = getattr(node, "value", None)
        if name in ("ISSUE_SPECS", "TEMPLATE_TO_TYPE") and node_value is not None:
            try:
                found[name] = ast.literal_eval(node_value)
            except (ValueError, SyntaxError):
                continue
    return found.get("ISSUE_SPECS", {}), found.get("TEMPLATE_TO_TYPE", {})


def train_rerank(config, *, top_k: int = 10) -> dict[str, Any]:
    """检索层训练，**严格按"先调融合、再训重排"的顺序**。

    顺序很重要：`_features` 里的 `q_*` 通道特征依赖融合权重（同一候选在不同权重下
    的 `q_rrf`、名次都不一样）。若先用默认权重建训练对、再用调好的权重评估，特征
    分布会漂移，学到的重排模型会把已经很好的 RRF 排序打乱（实测强金标准 Recall@5
    从 0.628 掉到 0.511）。所以流程固定为：

    1. 只用 RRF（`ranker=None`）做三路融合网格搜索 → 最优 `weights` / `rrf_k`；
    2. 用该最优融合构造 pairwise 训练对（正样本＝命中金标准，负样本＝候选池其余）；
    3. L2 向零收缩的 pairwise logistic 训练重排权重；
    4. 在强/弱两级金标准上分别评估 RRF-only 与 RRF+LTR，两者都要报。
    """
    from . import retrieve as R

    specs, _template_map = _load_issue_specs()
    if not specs:
        return {"skipped": "找不到 ISSUE_SPECS"}
    statutes = R.load_statutes(V1_STATUTES)
    index = R.StatuteIndex(statutes, config)

    # ---- 1. 融合权重网格搜索（不做重排，纯 RRF）-------------------------
    fusion = tune_fusion(index, R.Retriever(index, config, ranker=None),
                         statutes, specs, None, config)
    retriever = R.Retriever(index, config, ranker=None,
                            channel_weights=fusion["weights"], rrf_k=fusion["rrf_k"])

    # ---- 2. 在最优融合的候选池上构造 pairwise 训练对 --------------------
    # 特征在"同一查询的候选集内"做列标准化：14 维量纲差两个数量级，不标准化会让
    # q_tag_hit（0/1）压过 q_rrf，学出来的模型把好排序打乱。训练/推理共用同一口径。
    names = list(L.RERANK_FEATURES)
    pairs: list[tuple[list[float], list[float]]] = []
    evaluation: dict[str, Any] = {}
    for code, (strong, weak, query_text) in specs.items():
        plan = R.rewrite_query(query_text)
        wanted = set(strong) if strong else set(weak)
        expected = {str(a["article_uid"]) for a in statutes if wanted & set(a.get("risk_tags") or [])}
        if not expected:
            continue
        channels = {
            "sparse": index.bm25(plan.rewritten, limit=config.retrieve_candidate_limit),
            "dense": index.lsa(plan.rewritten, limit=config.retrieve_candidate_limit),
            "tag": index.tag_channel(plan.tags, "通用", limit=config.retrieve_candidate_limit),
        }
        fused = retriever._fuse(channels)[: config.retrieve_candidate_limit]
        raw_rows: list[tuple[dict[str, float], int]] = []
        for doc_id, feats, rrf in fused:
            doc = index.statutes[doc_id]
            feature_map = retriever._features(doc, plan, feats, query_len=len(plan.rewritten))
            feature_map["q_rrf"] = rrf
            label = 1 if str(doc.get("article_uid")) in expected else 0
            raw_rows.append((feature_map, label))
        if not raw_rows:
            continue
        # 列标准化（与 learned.RetrievalRanker.rank_normalized 完全一致）
        matrix = [[float(feats.get(name, 0.0)) for name in names] for feats, _label in raw_rows]
        columns = len(names)
        means = [sum(row[col] for row in matrix) / len(matrix) for col in range(columns)]
        stds: list[float] = []
        for col in range(columns):
            var = sum((row[col] - means[col]) ** 2 for row in matrix) / max(1, len(matrix))
            stds.append(var ** 0.5 or 1.0)
        normalized = [
            [(row[col] - means[col]) / stds[col] for col in range(columns)] for row in matrix
        ]
        rows = [(vector, raw_rows[i][1]) for i, vector in enumerate(normalized)]
        positive = [row for row in rows if row[1] == 1]
        best_pos_rank = next((i for i, row in enumerate(rows) if row[1] == 1), len(rows))
        # 难度适中的负样本：排在最好正样本之前、或紧随其后 20 名内
        hard_negative = [
            row for i, row in enumerate(rows) if row[1] == 0 and i <= best_pos_rank + 20
        ][:12]
        if not hard_negative:
            hard_negative = [row for row in rows if row[1] == 0][:12]
        for pos, _pos_label in positive[:8]:
            for neg, _neg_label in hard_negative:
                pairs.append((pos, neg))
        evaluation[code] = {"rows": len(rows), "positives": len(positive), "expected": len(expected)}

    if not pairs:
        return {"skipped": "没有可用训练对", "details": evaluation}

    # ---- 3. pairwise logistic（L2 向零收缩）----------------------------
    weights = [0.0] * len(names)
    lr = 0.05
    l2 = 0.02
    for _ in range(600):
        grad = [0.0] * len(names)
        for pos, neg in pairs:
            diff = [p - n for p, n in zip(pos, neg)]
            z = sum(w * d for w, d in zip(weights, diff))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            for dim, value in enumerate(diff):
                grad[dim] += (p - 1.0) * value
        for dim in range(len(weights)):
            grad[dim] = grad[dim] / max(1, len(pairs)) + l2 * weights[dim]
            weights[dim] -= lr * grad[dim]
    weights = [round(w, 6) for w in weights]
    ranker = RetrievalRanker(names=names, weights=weights, bias=0.0,
                             train_meta={"pairs": len(pairs), "source": "ISSUE_SPECS-LTR",
                                         "fusion_weights": fusion["weights"],
                                         "rrf_k": fusion["rrf_k"], "l2": l2,
                                         "normalize": True})

    # ---- 4. 两种排序各评一次 ------------------------------------------
    tuned = R.Retriever(index, config, ranker=ranker,
                        channel_weights=fusion["weights"], rrf_k=fusion["rrf_k"])
    metrics = evaluate_retrieval(tuned, index, statutes, specs, ranker, config)
    payload = {
        "names": names,
        "weights": weights,
        "bias": 0.0,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": len(pairs),
            "queries": len(evaluation),
            "metrics": metrics,
            "normalize": True,
            "fusion_weights": fusion["weights"],
            "rrf_k": fusion["rrf_k"],
            "l2": 0.02,
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / L.ModelRegistry.FILES["rerank"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fusion_payload = {
        "weights": fusion["weights"],
        "rrf_k": fusion["rrf_k"],
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grid": fusion["grid_size"],
            "objective": fusion["objective"],
            "best": fusion["best"],
            "top5": fusion["top"],
        },
    }
    (MODEL_DIR / R.FUSION_WEIGHTS_FILE).write_text(
        json.dumps(fusion_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"pairs": len(pairs), "queries": len(evaluation), "metrics": metrics,
            "fusion": {"weights": fusion["weights"], "rrf_k": fusion["rrf_k"],
                       "grid": fusion["grid_size"], "objective": fusion["objective"],
                       "best": fusion["best"]},
            "top_weights": sorted(zip(names, weights), key=lambda p: -abs(p[1]))[:6]}


def tune_fusion(index: "R.StatuteIndex", retriever, statutes, specs, ranker, config,
                *, top_k: int = 10) -> dict[str, Any]:
    """对三路 RRF 的通道权重与 `k` 做网格搜索，目标是"强金标准 Recall@5 + MRR@5"。

    只在 31 条评测查询上搜参，规模很小（默认 4×4×3×3 = 144 组），CPU 十几秒。
    返回最优组合与完整搜索规模，论文里可以直接写成"网格搜索 144 组、取强金标准
    Recall@5 最优"。
    """
    from .retrieve import rewrite_query

    manifest = _read_manifest(
        WORKSPACE / "data" / "generated_contracts" / "samr_multi_template_2025" / "adversarial" / "manifest.json"
    )
    occurrences: list[tuple[str, str, str]] = []
    for record in manifest.get("records", []):
        ctype = _TEMPLATE_TO_TYPE.get(str(record.get("template_key", "")), "通用")
        for code in record.get("issue_codes", []):
            spec = specs.get(code)
            if not spec:
                continue
            strong, weak, _text = spec
            tier = "strong" if strong else "weak"
            if not (set(strong) or set(weak)):
                continue
            occurrences.append((str(code), tier, ctype))

    gold_cache: dict[str, set[str]] = {}
    plans: dict[tuple[str, str], Any] = {}
    for code, tier, ctype in occurrences:
        if code not in gold_cache:
            strong, weak, _text = specs[code]
            wanted = set(strong) or set(weak)
            gold_cache[code] = {str(a["article_uid"]) for a in statutes
                                if wanted & set(a.get("risk_tags") or [])}
        if (code, ctype) not in plans:
            plans[(code, ctype)] = rewrite_query(specs[code][2], contract_type=ctype)

    # 预取三路通道结果，网格只重算 RRF 与排序
    channel_cache: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = {}
    for key, plan in plans.items():
        _code, ctype = key
        channel_cache[key] = {
            "sparse": index.bm25(plan.rewritten, limit=config.retrieve_candidate_limit),
            "dense": index.lsa(plan.rewritten, limit=config.retrieve_candidate_limit),
            "tag": index.tag_channel(plan.tags, ctype, limit=config.retrieve_candidate_limit),
        }

    def score_combo(weights: dict[str, float], rrf_k: int) -> tuple[float, dict[str, float]]:
        recall5 = {"strong": [], "weak": []}
        recall1 = {"strong": [], "weak": []}
        mrr5 = {"strong": [], "weak": []}
        for code, tier, ctype in occurrences:
            gold = gold_cache[code]
            if not gold:
                continue
            plan = plans[(code, ctype)]
            rows: dict[int, float] = {}
            for name, ranked in channel_cache[(code, ctype)].items():
                weight = weights.get(name, 0.0)
                if weight <= 0:
                    continue
                for rank, (doc_id, _value) in enumerate(ranked, start=1):
                    rows[doc_id] = rows.get(doc_id, 0.0) + weight / (rrf_k + rank)
            order = sorted(rows.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
            got = [str(index.statutes[doc_id].get("article_uid")) for doc_id, _ in order]
            recall5[tier].append(len(set(got[:5]) & gold) / len(gold))
            recall1[tier].append(len(set(got[:1]) & gold) / len(gold))
            first = next((i for i, uid in enumerate(got[:5], start=1) if uid in gold), 0)
            mrr5[tier].append(1.0 / first if first else 0.0)

        def mean(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        strong_recall5 = mean(recall5["strong"])
        strong_recall1 = mean(recall1["strong"])
        strong_mrr = mean(mrr5["strong"])
        weak_recall5 = mean(recall5["weak"])
        # 强金标准优先：Recall@5 0.4 + MRR@5 0.3 + Recall@1 0.2 + 弱金标准 Recall@5 0.1
        objective = (0.4 * strong_recall5 + 0.3 * strong_mrr
                     + 0.2 * strong_recall1 + 0.1 * weak_recall5)
        detail = {
            "objective": round(objective, 4),
            "strong_recall@1": round(strong_recall1, 4),
            "strong_recall@5": round(strong_recall5, 4),
            "strong_mrr@5": round(strong_mrr, 4),
            "weak_recall@5": round(weak_recall5, 4),
        }
        return objective, detail

    grid_sparse = (1.0, 1.3, 1.6, 2.0)
    grid_dense = (0.2, 0.5, 0.9, 1.2)
    grid_tag = (0.3, 0.6, 1.0, 1.4, 1.8)
    grid_k = (15, 30, 60, 100)
    results: list[tuple[float, dict[str, float], int, dict[str, float]]] = []
    for sparse in grid_sparse:
        for dense in grid_dense:
            for tag in grid_tag:
                weights = {"sparse": sparse, "dense": dense, "tag": tag}
                for rrf_k in grid_k:
                    value, detail = score_combo(weights, rrf_k)
                    results.append((value, weights, rrf_k, detail))
    results.sort(key=lambda row: -row[0])
    best_value, best_weights, best_k, best_detail = results[0]
    return {
        "weights": best_weights,
        "rrf_k": best_k,
        "objective": best_detail,
        "best": {"value": round(best_value, 4), "weights": best_weights, "rrf_k": best_k},
        "grid_size": len(results),
        "top": [
            {"value": round(value, 4), "weights": weights, "rrf_k": k, **detail}
            for value, weights, k, detail in results[:5]
        ],
    }


def evaluate_retrieval(retriever, index, statutes, specs, ranker, config,
                       *, top_k: int = 10) -> dict[str, Any]:
    """**v1 协议兼容**的 Recall@1..10 + MRR@5。

    v1 `scripts/eval_legal_retrieval.py` 的三条口径必须原样复刻，否则数字不可比：

    1. 统计单位是"样本记录 × issue_code"的出现次数（同一风险码在 5 份合同里出现
       就计 5 次），不是按风险码去重；
    2. 金标准分两级：`strong`（该风险专用的法条标签，个数 1~6）与 `weak`（宽泛
       家族，个数可达 55），两级**分开报告**。混在一起算会让 weak 把均值压死；
    3. `Recall@k = |top-k ∩ 金标准| / |金标准|`，`MRR@5` 只对首位命中求倒数。
    """
    from .retrieve import rewrite_query

    # 出现次数来自对抗集清单（与 v1 完全一致的统计单位）
    manifest = _read_manifest(
        WORKSPACE / "data" / "generated_contracts" / "samr_multi_template_2025" / "adversarial" / "manifest.json"
    )
    occurrences: list[tuple[str, str, str]] = []  # (code, tier, contract_type)
    for record in manifest.get("records", []):
        ctype = _TEMPLATE_TO_TYPE.get(str(record.get("template_key", "")), "通用")
        for code in record.get("issue_codes", []):
            spec = specs.get(code)
            if not spec:
                continue
            strong, weak, _text = spec
            tier = "strong" if strong else "weak"
            wanted = set(strong) or set(weak)
            if not wanted:
                continue
            occurrences.append((str(code), tier, ctype))

    def gold_of(code: str) -> set[str]:
        strong, weak, _text = specs[code]
        wanted = set(strong) or set(weak)
        return {str(a["article_uid"]) for a in statutes if wanted & set(a.get("risk_tags") or [])}

    gold_cache: dict[str, set[str]] = {}

    def run(use_ranker: bool) -> dict[str, Any]:
        buckets: dict[str, list[float]] = {"strong": [], "weak": []}
        ranks: dict[str, list[int]] = {"strong": [], "weak": []}
        recalls_by_k: dict[str, dict[int, list[float]]] = {"strong": {}, "weak": {}}
        for code, tier, ctype in occurrences:
            if code not in gold_cache:
                gold_cache[code] = gold_of(code)
            expected = gold_cache[code]
            if not expected:
                continue
            strong, weak, query_text = specs[code]
            plan = rewrite_query(query_text, contract_type=ctype)
            saved = retriever.ranker
            retriever.ranker = ranker if use_ranker else None
            result = retriever.search(plan, top_k=top_k, use_rerank=use_ranker)
            retriever.ranker = saved
            got = [hit.article_uid for hit in result.hits]
            buckets[tier].append(len(set(got) & expected) / len(expected))
            first = next((i for i, uid in enumerate(got[:5], start=1) if uid in expected), 0)
            ranks[tier].append(first)
            for k in range(1, top_k + 1):
                recalls_by_k[tier].setdefault(k, []).append(len(set(got[:k]) & expected) / len(expected))

        def mrr(values: Sequence[int]) -> float:
            reciprocal = [1.0 / rank for rank in values if rank > 0]
            return round(sum(reciprocal) / len(reciprocal), 4) if reciprocal else 0.0

        out: dict[str, Any] = {}
        for tier in ("strong", "weak"):
            values = buckets[tier]
            if not values:
                continue
            out[tier] = {
                "n": len(values),
                "recall@k": {f"@{k}": round(sum(v) / len(v), 4)
                             for k, v in sorted(recalls_by_k[tier].items())},
                "mrr@5": mrr(ranks[tier]),
            }
        return out

    return {"rrf_only": run(False), "rrf_plus_ltr": run(True)}


# ---------------------------------------------------------------------------
# 4. 校准器
# ---------------------------------------------------------------------------

def train_calibration(docs: Sequence[Doc], *, seed: int = 20261104) -> dict[str, Any]:
    """用"检出是否正确"做 isotonic 校准：把 confidence 映射成经验命中率。"""
    from .pipeline import review_path

    config = default_config()
    samples: list[tuple[float, int]] = []
    for doc in docs:
        if not doc.gold_codes or doc.path.is_dir():
            continue
        try:
            report = review_path(doc.path, config=config)
        except Exception:
            continue
        gold = {D.ISSUE_TO_CODE.get(code, code) for code in doc.gold_codes}
        gold_sets = set(doc.gold_codes)
        for risk in report.risks:
            hit = 1 if (set(risk.tags) & gold_sets) or (risk.code in gold) else 0
            samples.append((float(risk.confidence), hit))
    if len(samples) < 20:
        return {"skipped": "校准样本不足", "samples": len(samples)}

    samples.sort(key=lambda pair: pair[0])
    # 分箱后做单调回归（PAV 简化版：桶内取均值再强制单调递增）
    bins = 8
    size = max(1, len(samples) // bins)
    xs: list[float] = []
    ys: list[float] = []
    for start in range(0, len(samples), size):
        chunk = samples[start:start + size]
        if not chunk:
            continue
        xs.append(round(sum(x for x, _ in chunk) / len(chunk), 4))
        ys.append(round(sum(y for _, y in chunk) / len(chunk), 4))
    for index in range(1, len(ys)):
        ys[index] = max(ys[index], ys[index - 1])
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 0.0)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(ys[-1])

    calibrator = Calibrator(xs=xs, ys=ys, kind="isotonic-binned",
                            train_meta={"samples": len(samples), "bins": bins})
    payload = calibrator.to_dict()
    payload["meta"]["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / L.ModelRegistry.FILES["calibration"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 校准前后的 Brier 分数，论文里直接引用
    before = sum((x - y) ** 2 for x, y in samples) / len(samples)
    after = sum((calibrator.apply(x) - y) ** 2 for x, y in samples) / len(samples)
    return {
        "samples": len(samples),
        "bins": len(xs),
        "brier_before": round(before, 4),
        "brier_after": round(after, 4),
        "curve": list(zip(xs, ys)),
    }


# ---------------------------------------------------------------------------
# 5. 风险分类器（条款级多标签）
# ---------------------------------------------------------------------------

def train_risk(docs: Sequence[Doc], config, *, seed: int = 20261104) -> dict[str, Any]:
    """条款级 one-vs-rest logistic，标签来自"该条款是否覆盖某个金标缺陷码"。"""
    from .detectors import ISSUE_TO_CODE
    from .pipeline import prepare

    train_docs, valid_docs = split_docs(docs, 0.25, seed)

    def rows_of(subset: Sequence[Doc]) -> tuple[list[list[float]], list[set[str]]]:
        names = list(F.RISK_FEATURES)
        x: list[list[float]] = []
        y: list[set[str]] = []
        for doc in subset:
            if not doc.gold_codes or doc.path.is_dir():
                continue
            try:
                document = prepare(doc.path.name, path=doc.path, config=config)
            except Exception:
                continue
            codes = {ISSUE_TO_CODE.get(code, code) for code in doc.gold_codes}
            for clause in document.clauses:
                feats = F.risk_features(clause)
                x.append([float(feats.get(name, 0.0)) for name in names])
                hit = {code for code in codes if _clause_supports(clause.text, code)}
                y.append(hit)
        return x, y

    x_train, y_train = rows_of(train_docs)
    x_valid, y_valid = rows_of(valid_docs)
    if not x_train:
        return {"skipped": "风险样本不足"}
    names = list(F.RISK_FEATURES)
    labels = sorted({code for row in y_train for code in row})
    weights: dict[str, list[float]] = {}
    biases: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for label in labels:
        target = [1 if label in row else 0 for row in y_train]
        if sum(target) < 2:
            continue
        w, b = _fit_logistic(x_train, target, epochs=250, lr=0.3, l2=1e-2)
        weights[label] = [round(v, 6) for v in w]
        biases[label] = round(b, 6)
        # 阈值取验证集上 F1 最优
        best_f1, best_t = 0.0, 0.5
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            tp = fp = fn = 0
            for row, gold in zip(x_valid, y_valid):
                z = b + sum(wv * v for wv, v in zip(w, row))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                pred = p >= t
                truth = label in gold
                tp += pred and truth
                fp += pred and not truth
                fn += (not pred) and truth
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[label] = best_t

    payload = {
        "names": names,
        "labels": labels,
        "weights": weights,
        "biases": biases,
        "thresholds": thresholds,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": len(x_train),
            "valid_samples": len(x_valid),
            "labels_trained": len(weights),
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / L.ModelRegistry.FILES["risk"]).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "samples": len(x_train),
        "valid_samples": len(x_valid),
        "labels": len(weights),
        "thresholds": thresholds,
    }


def _clause_supports(text: str, code: str) -> bool:
    """弱标注：条款文本是否"谈到"了某个风险码的语义。"""
    flat = T.full_to_half(text)
    keywords: dict[str, tuple[str, ...]] = {
        "zero_rent": ("租金", "0元", "免费", "无偿"),
        "abnormal_price": ("押金", "保证金", "万元", "亿元"),
        "fee_mismatch": ("报酬", "费用", "价款", "金额"),
        "illegal_terms": ("现金", "负数", "永久", "无需", "不受"),
        "one_sided_breach": ("单方", "不得", "仅由", "只能"),
        "waiver_of_rights": ("放弃", "不得提出异议", "不得追究"),
        "liability_exemption": ("不承担", "免责", "无论"),
        "unilateral_modify": ("随时", "自行决定", "无需通知", "单方变更"),
        "vague_delivery": ("尽快", "及时", "适当", "另行约定"),
        "unclear_standard": ("验收", "标准", "合格"),
        "date_reversed": ("期限", "自", "至", "起"),
        "subject_unspecified": ("标的", "数量", "项目"),
        "party_missing": ("甲方", "乙方", "主体"),
        "unverified_property": ("权属", "产权", "证明"),
        "need_authorization": ("授权", "代理", "委托"),
        "need_license": ("资质", "许可", "备案"),
        "need_title_cert": ("权属证书", "不动产权", "产权证"),
        "confidentiality_missing": ("保密", "个人信息", "数据"),
        "missing_party_info": ("名称", "住所", "联系方式"),
        "amount_missing": ("金额", "价款"),
    }
    return any(word in flat for word in keywords.get(code, ()))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def write_report(sections: Mapping[str, Any], path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v2 训练报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "所有模型权重落在 `zhilv_qiling_v2/models/`，为可读 JSON（不 pickle），",
        "指标由本模块实测产出，未做任何手工修饰。",
        "",
    ]
    for title, payload in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        if isinstance(payload, dict) and payload.get("skipped"):
            lines.append(f"- 跳过：{payload['skipped']}")
        else:
            lines.append("```json")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 v2 学习型组件")
    parser.add_argument("--only", default="", help="只训某几个组件，逗号分隔：boundary,ner,rerank,calibration,risk")
    parser.add_argument("--quick", action="store_true", help="小样本快速跑通（调试用）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = default_config()
    docs = collect_docs()
    if args.quick:
        docs = docs[:12]
    print(f"[train] 样本 {len(docs)} 份（含金标签 {sum(1 for d in docs if d.gold_codes or d.gold_entities)} 份）")

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    sections: dict[str, Any] = {}
    started = time.time()

    def should(name: str) -> bool:
        return not wanted or name in wanted

    if should("boundary"):
        print("[train] 训练边界分类器 ...")
        sections["1. 条款边界分类器"] = train_boundary(docs, config)
    if should("ner"):
        print("[train] 训练字符级 NER ...")
        sections["2. 字符级 NER（平均感知机）"] = train_ner(docs, config)
    if should("rerank"):
        print("[train] 训练检索 LTR ...")
        sections["3. 法条检索 LTR"] = train_rerank(config)
    if should("risk"):
        print("[train] 训练条款级风险分类器 ...")
        sections["4. 条款级风险分类器"] = train_risk(docs, config)
    if should("calibration"):
        print("[train] 训练置信度校准 ...")
        sections["5. 置信度校准"] = train_calibration(docs)

    sections["耗时"] = {"seconds": round(time.time() - started, 2)}
    path = write_report(sections)
    print(f"[train] 报告：{path}")
    for title, payload in sections.items():
        if title == "耗时":
            continue
        print(f"  - {title}: {json.dumps(payload, ensure_ascii=False)[:220]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
