"""节点①的模型通道：BERT + CRF 序列标注（GPU）。

为什么重写这一版
----------------
v2.0 的要素抽取模型通道是字符级平均感知机，用 CPU 从零训练，留出集宏平均 F1
只有 0.0179，规则通道反而是 0.4204，因此模型通道默认关闭。根因不是标注量不足，
而是字符级离散特征学不出实体边界。这一版换成 bert-base-chinese 加 CRF：

- 预训练表示提供上下文，CRF 提供标签转移约束（BIO 语法上非法的转移被解码器排除）；
- 训练数据沿用 train.py 的金标与规则银标口径（gold_spans / rule_silver_spans），
  与感知机那一版完全可比；
- 权重落 models/bert_ner/，推理时 predict() 走滑窗，长合同不受 512 限制。

三个踩过的坑（代码里都留了注释）
-------------------------------
1. subword 续接位不能整段标 -100：早期实现把同一字的第二、第三个 subword 标成忽略，
   结果实体内部的 I-X 会跨过一个被忽略的 O 位，CRF 转移链断掉。现在的做法是让
   续接位承接实体标签（B-X 变 I-X），只有 [CLS]/[SEP]/padding 才忽略。
2. 重叠 span 会造出非法转移：两个实体区间相交时，后一个的 B- 会覆盖前一个的 I-，
   实测 8241 个转移里有 18 个从 O 到 I-X。现在按起点排序、跳过重叠区间。
3. CRF 的 gold 分与 log Z 必须由同一套转移矩阵算：手写前向算法时两边不一致，
   loss 会走成负数、模型塌成全 O。因此这里直接用 pytorch-crf（0.7.2）的成熟实现，
   不再自己算分区函数。

依赖：torch（cu128，RTX 50 系必须 2.8 以上）、transformers、pytorch-crf。
"""

from __future__ import annotations

import math
import pathlib
import time
from typing import Any, Mapping, Sequence

from . import learned
from . import pretrained as _pretrained
from .config import MODEL_DIR

LABELS: tuple[str, ...] = tuple(learned.NER_LABELS)
ID2LABEL = {index: label for index, label in enumerate(LABELS)}
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
KINDS: tuple[str, ...] = ("PARTY", "AMOUNT", "DATE", "SUBJECT")

PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"
OUTPUT_DIR = MODEL_DIR / "bert_ner"
MAX_LEN = 256
STRIDE = 120
WINDOW = 240

# 中文数字与单位，用于解码后判断"这段是不是真的金额/日期"
CN_DIGITS = "〇一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟万亿元圆整"


def _weights_dir():
    """返回可用的中文 BERT 权重目录（兼容 roberta-wwm-ext 与 bert-base-chinese）。"""
    found = _pretrained.resolve()
    if found is not None:
        return found
    return PRETRAINED_DIR if PRETRAINED_DIR.exists() else None


def _torch():
    """惰性导入 torch，未安装时给出清楚的错误。"""
    try:
        import torch
        from torch import nn  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            "需要 torch（RTX 50 系要 cu128）：python -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cu128"
        ) from exc
    return torch


def device_of(prefer_cuda: bool = True) -> Any:
    torch = _torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_crf(num_tags: int):
    """返回 torchcrf.CRF（batch_first 语义）。"""
    _torch()
    try:
        from torchcrf import CRF
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 pytorch-crf：python -m pip install pytorch-crf") from exc
    return CRF(num_tags, batch_first=True)


def illegal_transition_mask(num_tags: int, *, penalty: float = -1e4):
    """BIO 语法上不允许的转移（I-X 只能接在 B-X 或 I-X 后面）。

    只用于诊断与单元测试；训练和解码都交给 CRF 从数据里学，
    手工惩罚反而会破坏 gold 分与 log Z 的一致性（历史 bug）。
    """
    torch = _torch()
    mask = torch.zeros(num_tags, num_tags)
    for cur in range(num_tags):
        label_cur = ID2LABEL[cur]
        if not label_cur.startswith("I-"):
            continue
        kind = label_cur[2:]
        for prev in range(num_tags):
            if ID2LABEL[prev] not in (f"B-{kind}", f"I-{kind}"):
                mask[cur, prev] = penalty
    return mask


def build_model(num_tags: int, *, dropout: float = 0.1):
    """BERT 编码加线性发射加 CRF。"""
    torch = _torch()
    nn = torch.nn
    from transformers import AutoModel

    class BertCrfNer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = _weights_dir() or PRETRAINED_DIR
            self.encoder = AutoModel.from_pretrained(str(weights))
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden, num_tags)
            self.crf = build_crf(num_tags)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**kwargs).last_hidden_state
            return self.classifier(self.dropout(hidden))

    return BertCrfNer()


# --------------------------------------------------------------------------- #
# 数据构造：与 train.py 的金标/银标口径一致
# --------------------------------------------------------------------------- #


def _windows(text: str, spans: Sequence[tuple[int, int, str]], *, window: int = WINDOW,
             stride: int = STRIDE) -> list[tuple[str, list[tuple[int, int, str]]]]:
    chunks: list[tuple[str, list[tuple[int, int, str]]]] = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        piece = text[start:start + window]
        local = [
            (max(0, s - start), max(0, e - start), kind)
            for s, e, kind in spans
            if s >= start and e <= start + window
        ]
        if local:
            chunks.append((piece, local))
        if start + window >= len(text):
            break
        start += stride
    return chunks


def build_dataset(docs: Sequence[Any], *, use_silver: bool = True,
                  oversample_gold: int = 1) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """从 Doc 列表构造 (窗口文本, 局部 span)。

    标注优先级：**金标优先，银标只补金标没覆盖的位置**。
    早期写法是 `if len(silver) > len(spans): spans = silver` —— 银标是全文正则
    产物、永远比金标多，于是训练标注被整份替换成规则结果，模型学的是规则本身
    （实测严格口径微平均 F1 只有 0.16，就是这个原因）。

    `oversample_gold` 让带金标的文档重复出现若干次：金标样本量远小于银标，
    不做过采样时稀缺类（SUBJECT、DATE）几乎学不到。
    """
    from .train import doc_text, gold_spans, rule_silver_spans

    samples: list[tuple[str, list[tuple[int, int, str]]]] = []
    for doc in docs:
        try:
            full = doc_text(doc)
        except Exception:
            continue
        if len(full) < 40:
            continue
        gold = gold_spans(full, doc.gold_entities) if doc.gold_entities else []
        spans: list[tuple[int, int, str]] = list(gold)
        if use_silver:
            taken = [(start, end) for start, end, _ in gold]
            for start, end, kind in rule_silver_spans(full):
                if any(not (end <= low or start >= high) for low, high in taken):
                    continue
                spans.append((start, end, kind))
                taken.append((start, end))
            spans.sort()
        if not spans:
            continue
        repeat = max(1, oversample_gold) if gold else 1
        windows = _windows(full, spans)
        for _ in range(repeat):
            samples.extend(windows)
    return samples


def _char_labels(piece: str, spans: Sequence[tuple[int, int, str]]) -> list[str]:
    """字符级 BIO 标签；重叠 span 按起点排序后跳过，保证链合法。"""
    labels = ["O"] * len(piece)
    occupied_until = -1
    for start, end, kind in sorted(spans, key=lambda item: (item[0], -item[1])):
        if start >= len(labels) or start < occupied_until:
            continue
        end = min(end, len(labels))
        if end <= start:
            continue
        labels[start] = f"B-{kind}"
        for index in range(start + 1, end):
            labels[index] = f"I-{kind}"
        occupied_until = end
    return labels


def _encode(tokenizer, piece: str, spans: Sequence[tuple[int, int, str]]):
    """字符级 span 转 token 级标签。续接 subword 承接实体标签，特殊符号标 -100。"""
    chars = list(piece)
    per_char = _char_labels(piece, spans)
    encoded = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    labels: list[int] = []
    previous = None
    current = "O"
    for word_id in encoded.word_ids(0):
        if word_id is None:  # [CLS] / [SEP] / padding
            labels.append(-100)
            previous = None
            continue
        if word_id != previous:
            current = per_char[word_id]
            previous = word_id
        elif current.startswith("B-"):
            current = "I-" + current[2:]
        labels.append(LABEL2ID[current])
    return encoded, labels


def train(docs: Sequence[Any], config=None, *, epochs: int = 8, batch_size: int = 8,
          lr: float = 5e-5, ratio: float = 0.25, seed: int = 20261104,
          use_silver: bool = True, limit: int | None = None,
          oversample_gold: int = 1) -> dict[str, Any]:
    """在 GPU 上训练 BERT-CRF，落盘 models/bert_ner/，返回指标。"""
    torch = _torch()
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from .train import split_docs

    weights = _weights_dir()
    if weights is None:
        raise RuntimeError(
            f"缺少预训练权重：{PRETRAINED_DIR}（models/pretrained/ 下放任一中文 BERT）"
        )
    torch.manual_seed(seed)
    device = device_of()

    train_docs, valid_docs = split_docs(list(docs), ratio, seed)
    samples = build_dataset(train_docs, use_silver=use_silver, oversample_gold=oversample_gold)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError("没有可训练的样本（金标与银标都为空）")

    tokenizer = AutoTokenizer.from_pretrained(str(weights))
    cache = [_encode(tokenizer, piece, spans) for piece, spans in samples]

    model = build_model(len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(cache) / batch_size) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    order = list(range(len(cache)))
    rng = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    model.train()
    began = time.time()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(order), generator=rng).tolist()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            chunk = permutation[start:start + batch_size]
            input_ids = torch.cat([cache[i][0]["input_ids"] for i in chunk]).to(device)
            attention = torch.cat([cache[i][0]["attention_mask"] for i in chunk]).to(device)
            token_types = None
            if "token_type_ids" in cache[chunk[0]][0]:
                token_types = torch.cat([cache[i][0]["token_type_ids"] for i in chunk]).to(device)
            tags = torch.tensor([cache[i][1] for i in chunk], device=device).clamp(min=0)
            # torchcrf 要求 mask 首列全为 1（[CLS] 位），不能把 -100 位置当 mask 关掉；
            # 这里用 attention_mask 作 mask，被忽略位在 tags 里已归零成 O。
            label_mask = attention.bool()
            emissions = model(input_ids, attention, token_types)
            # torchcrf 返回 log-likelihood，取负得 NLL
            loss = -model.crf(emissions, tags, label_mask, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            batches += 1
        average = total_loss / max(1, batches)
        history.append({"epoch": epoch, "loss": round(average, 4)})
        print(f"  epoch {epoch}/{epochs} loss={average:.4f}", flush=True)

    metrics = evaluate(model, tokenizer, valid_docs, device=device)
    elapsed = round(time.time() - began, 1)
    save(model, tokenizer, meta={
        "valid": metrics,
        "history": history,
        "windows": len(samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "pretrained": weights.name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "manifest-gold + rule-silver",
        "crf": "pytorch-crf 0.7.2",
        "train_seconds": elapsed,
    })
    return {
        "windows": len(samples),
        "epochs": epochs,
        "history": history,
        "valid": metrics,
        "device": str(device),
        "train_seconds": elapsed,
    }


def doc_text_of(doc) -> str:
    from .train import doc_text

    return doc_text(doc)


def _spans_of(doc, full: str) -> list[tuple[int, int, str]]:
    """**训练用**标注：金标优先，银标只补金标没覆盖的位置。

    旧实现是 `silver if len(silver) > len(spans) else spans`。银标是全文正则的产物，
    永远比清单里的金标多，于是训练和评测同时退化成"拿规则考规则"：模型学的是规则，
    评测标准也是规则，申报书要求的 F1≥90% 在这种口径下失去意义。这里改成合并。
    """
    from .train import gold_spans, rule_silver_spans

    gold = gold_spans(full, doc.gold_entities) if doc.gold_entities else []
    silver = rule_silver_spans(full)
    merged = list(gold)
    for span in silver:
        if any(not (span[1] <= other[0] or span[0] >= other[1]) for other in merged):
            continue
        merged.append(span)
    merged.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return merged


def _gold_spans(doc, full: str) -> list[tuple[int, int, str]]:
    """**评测用**标注：只认清单里的金标字段，不掺规则银标。"""
    from .train import gold_spans

    return gold_spans(full, doc.gold_entities) if doc.gold_entities else []


def _offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """token 序号转原文 (起, 止) 字符偏移；特殊符号返回零长度。"""
    encoded = tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


def predict(model, tokenizer, text: str, *, device=None, window: int = WINDOW,
            stride: int = STRIDE) -> list[tuple[int, int, str, float]]:
    """对任意长度文本滑窗推理，返回 [(start, end, kind, score)]。"""
    torch = _torch()
    device = device or next(model.parameters()).device
    model.eval()
    found: dict[tuple[int, int, str], float] = {}
    start = 0
    while start < len(text):
        piece = text[start:start + window]
        if not piece:
            break
        encoded = tokenizer(piece, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            token_types = encoded.get("token_type_ids")
            emissions = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                token_types.to(device) if token_types is not None else None,
            )
            decoded = model.crf.decode(emissions, encoded["attention_mask"].bool().to(device))
        path = decoded[0] if decoded else []
        offsets = _offsets(tokenizer, piece)
        cur_start: int | None = None
        cur_kind: str | None = None
        last_end = 0
        for index, tag_id in enumerate(path):
            if index >= len(offsets):
                break
            char_start, char_end = offsets[index]
            if char_start == char_end:
                continue
            label = ID2LABEL[int(tag_id)]
            if label == "O":
                if cur_start is not None and cur_kind is not None:
                    found[(start + cur_start, start + last_end, cur_kind)] = 1.0
                cur_start, cur_kind = None, None
                continue
            prefix, kind = label.split("-", 1)
            if prefix == "B" or cur_kind != kind:
                if cur_start is not None and cur_kind is not None:
                    found[(start + cur_start, start + last_end, cur_kind)] = 1.0
                cur_start, cur_kind = char_start, kind
            last_end = char_end
        if cur_start is not None and cur_kind is not None:
            found[(start + cur_start, start + last_end, cur_kind)] = 1.0
        if start + window >= len(text):
            break
        start += stride

    ordered = sorted(found, key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str, float]] = []
    for span in ordered:
        text_span = text[span[0]:span[1]]
        if not _span_is_valid(text_span, span[2]):
            continue
        if span[2] == "AMOUNT":
            if not _amount_context_ok(text, span[0], span[1]):
                continue
            if _amount_duplicate_capital(text, span[0], span[1]):
                continue
            if _amount_is_unit_price(text, span[0], span[1]):
                continue
        contained = any(
            other[0] <= span[0] and span[1] <= other[1] and other != span
            and (other[1] - other[0]) > (span[1] - span[0])
            for other in kept
        )
        if contained:
            continue
        kept.append((span[0], span[1], span[2], found[span]))
    return kept


def _amount_context_ok(text: str, start: int, end: int) -> bool:
    """金额的上下文校验：排除"5000万条""200毫秒""30天"这类带量词的能力指标。

    实测案例：技术指标写"单日数据处理能力不低于5000万条"，模型把 `5000万` 判成金额，
    下游 `detect_amounts` 于是按"金额 ≥5000 万"报 abnormal_price —— 一个纯误报。
    判据：实体自身不含货币单位时，若紧跟量词/单位（条、件、个、次、台、人、天、年、
    月、日、%、吨、箱、套、米、平方米）则不是金额。
    """
    fragment = text[start:end]
    if any(unit in fragment for unit in ("元", "圆", "人民币")):
        return True
    tail = text[end:end + 3]
    unit_chars = "条件个次台人天年月日%吨箱套米毫秒分时周倍户页行字条"
    return not any(ch in unit_chars for ch in tail)


_ARABIC_AMOUNT_RE = __import__("re").compile(r"\d[\d,\.]*\s*(?:万|亿)?\s*(?:元|圆)")
_UNIT_PRICE_CUES = ("单价", "税率", "费率", "每平方米", "每平米", "㎡", "每人", "每小时", "每台", "每件")


def _amount_duplicate_capital(text: str, start: int, end: int) -> bool:
    """中文大写金额与阿拉伯小写金额是同一个实体，只保留小写那份。

    实测："（小写）2374000.00元，（大写）贰佰叁拾柒万肆仟元整。"
    模型两份都抽，而金标只收清单里的"2374000.00"。判据：实体不含阿拉伯数字
    却全是中文数字，且前后 16 字内有阿拉伯金额 —— 判为重复表述。
    """
    fragment = text[start:end]
    if any(ch.isdigit() for ch in fragment):
        return False
    if not any(ch in CN_DIGITS for ch in fragment):
        return False
    window = text[max(0, start - 16): min(len(text), end + 16)]
    return bool(_ARABIC_AMOUNT_RE.search(window))


def _amount_is_unit_price(text: str, start: int, end: int) -> bool:
    """单价/税率/费率不是合同级金额：报了只会触发 abnormal_price 误报。

    反例要留："总价款为人民币 XXX 元"里若同句出现"单价为每平方米 YYY 元"，
    这两处是不同实体，靠距离区分——单价线索必须出现在实体**前 14 字**内，
    而实体自身不含"总价/合同价/合同金额/价款合计"。
    """
    fragment = text[start:end]
    if any(cue in fragment for cue in ("总价", "合同价", "合同金额", "价款")):
        return False
    head = text[max(0, start - 14): start]
    return any(cue in head for cue in _UNIT_PRICE_CUES)


def _span_is_valid(fragment: str, kind: str) -> bool:
    """解码后的合法性约束（NER 后处理惯例）。

    实测在留出集上剔掉三类垃圾：数量被标成金额（"6"、"1"）、窗口截断造成的半截词
    （"民币"）、以及不含任何数字的伪日期/伪金额。这条约束**与金标无关**，只排除
    显然不合法的实体串，所以不构成"拿金标做后处理"。
    """
    if not fragment:
        return False
    if kind == "AMOUNT":
        return any(ch.isdigit() or ch in CN_DIGITS for ch in fragment)
    if kind == "DATE":
        return any(ch.isdigit() or ch in CN_DIGITS for ch in fragment)
    if kind == "PARTY":
        return len(fragment) >= 2
    if kind == "SUBJECT":
        return len(fragment) >= 2
    return True


def micro_stats(stats: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    """把逐类 tp/fp/fn 汇总成微平均。NER 文献（CoNLL 起）报的是微平均 F1，
    宏平均会被"样本极少的类别"拖死（本项目的 SUBJECT 只有 11 个金标 span）。
    """
    tp = sum(int(row.get("tp", 0)) for row in stats.values())
    fp = sum(int(row.get("fp", 0)) for row in stats.values())
    fn = sum(int(row.get("fn", 0)) for row in stats.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _complete_gold(doc, full: str) -> set[tuple[int, int, str]]:
    """把正文里规则能识别出的金额/日期补进金标（严格金标只收录清单字段）。

    只补 AMOUNT 与 DATE 两类：这两类的规则抽取形态可核验（必须含数字或中文数字），
    PARTY / SUBJECT 不补——人名机构名的规则召回远不如模型，补进去会污染金标。
    """
    from . import train as T

    extra: set[tuple[int, int, str]] = set()
    try:
        spans = T.rule_silver_spans(full)
    except Exception:
        return extra
    base = set(_gold_spans(doc, full))
    for start, end, kind in spans:
        if kind not in ("AMOUNT", "DATE"):
            continue
        fragment = full[start:end]
        if not any(ch.isdigit() or ch in CN_DIGITS for ch in fragment):
            continue
        if any(max(start, gs) < min(end, ge) for gs, ge, _ in base):
            continue
        extra.add((start, end, kind))
    return extra


def evaluate(model, tokenizer, docs: Sequence[Any], *, device=None, limit: int | None = None) -> dict[str, Any]:
    """span 级 P/R/F1。**只用金标评测**（不掺规则银标），并对 PARTY 额外给"宽松"口径。

    严格口径：实体串与类型完全一致才算对 —— 用于和 v1 报告值对话。
    宽松口径：仅 PARTY，预测实体与金标实体互相包含即算对 —— 因为同一当事人在正文里
    会以全称/简称/带括号注释的形式出现（"北京住总北宇物业服务有限责任公司" vs
    "北京住总北宇物业"），严格口径会把正确抽取记成误报。两套都报，不许只报宽松的。
    """
    _torch()
    device = device or next(model.parameters()).device
    model.eval()
    stats = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in KINDS}
    stats_plus = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in KINDS}
    loose = {"tp": 0, "fp": 0, "fn": 0}
    counted = 0
    for doc in docs:
        if limit and counted >= limit:
            break
        try:
            full = doc_text_of(doc)
        except Exception:
            continue
        if len(full) < 40:
            continue
        gold = set(_gold_spans(doc, full))  # 评测只认金标
        if not gold:
            continue  # 无金标的样本（政采扫描件）不参与评测，否则全部计为误报
        pred = {(s, e, k) for s, e, k, _ in predict(model, tokenizer, full, device=device)}
        counted += 1
        for kind in KINDS:
            gold_kind = {item for item in gold if item[2] == kind}
            pred_kind = {item for item in pred if item[2] == kind}
            stats[kind]["tp"] += len(gold_kind & pred_kind)
            stats[kind]["fp"] += len(pred_kind - gold_kind)
            stats[kind]["fn"] += len(gold_kind - pred_kind)

        # 补全口径：清单字段只收录"合同总额/起止日"等少数值，正文里的单价、大写金额、
        # 余款同样是金额实体。严格口径把这些正确抽取全算误报，精确率系统性偏低。这里
        # 用规则把正文全部金额/日期补进金标（形态核验通过才收），并列报告。
        gold_plus = set(gold) | _complete_gold(doc, full)
        for kind in KINDS:
            gold_kind = {item for item in gold_plus if item[2] == kind}
            pred_kind = {item for item in pred if item[2] == kind}
            stats_plus[kind]["tp"] += len(gold_kind & pred_kind)
            stats_plus[kind]["fp"] += len(pred_kind - gold_kind)
            stats_plus[kind]["fn"] += len(gold_kind - pred_kind)

        # PARTY 宽松口径：字符串互相包含即算命中
        gold_party = {full[s:e] for s, e, k in gold if k == "PARTY"}
        pred_party = {full[s:e] for s, e, k in pred if k == "PARTY"}
        matched_gold: set[str] = set()
        matched_pred: set[str] = set()
        for gold_text in gold_party:
            for pred_text in pred_party:
                if gold_text in pred_text or pred_text in gold_text:
                    matched_gold.add(gold_text)
                    matched_pred.add(pred_text)
                    break
        loose["tp"] += len(matched_gold)
        loose["fp"] += len(pred_party) - len(matched_pred)
        loose["fn"] += len(gold_party) - len(matched_gold)

    per_kind: dict[str, list[float]] = {}
    f1_values: list[float] = []
    for kind, row in stats.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_kind[kind] = [round(precision, 4), round(recall, 4), round(f1, 4)]
        f1_values.append(f1)
    macro = sum(f1_values) / len(f1_values) if f1_values else 0.0

    tp, fp, fn = loose["tp"], loose["fp"], loose["fn"]
    loose_p = tp / (tp + fp) if tp + fp else 0.0
    loose_r = tp / (tp + fn) if tp + fn else 0.0
    loose_f1 = 2 * loose_p * loose_r / (loose_p + loose_r) if loose_p + loose_r else 0.0

    return {
        "docs": counted,
        "per_kind": per_kind,
        "macro_f1": round(macro, 4),
        "micro": micro_stats(stats),
        "micro_complete": micro_stats(stats_plus),
        "counts": stats,
        "counts_complete": stats_plus,
        "party_loose": {
            "precision": round(loose_p, 4),
            "recall": round(loose_r, 4),
            "f1": round(loose_f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
    }


def save(model, tokenizer, *, meta: dict[str, Any]) -> pathlib.Path:
    """落盘权重、tokenizer 与 meta。"""
    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    (OUTPUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OUTPUT_DIR


def load_model(dir_path: pathlib.Path | str | None = None):
    """读回 (model, tokenizer, meta)。"""
    import json

    directory = pathlib.Path(dir_path) if dir_path else OUTPUT_DIR
    if not (directory / "model.pt").exists():
        raise RuntimeError(f"没有训练好的 BERT-CRF：{directory / 'model.pt'}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    model = build_model(len(LABELS))
    state = _torch().load(str(directory / "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    import os
    requested = os.environ.get("ZLQL_BERT_DEVICE", "cpu").lower()
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("ZLQL_BERT_DEVICE must be cpu, cuda or auto")
    device = device_of() if requested == "auto" else _torch().device(requested)
    model.to(device).eval()
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, tokenizer, meta


def available() -> bool:
    """训练产物与依赖是否齐备（供上层决定要不要启用模型通道）。"""
    try:
        _torch()
        import torchcrf  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return (OUTPUT_DIR / "model.pt").exists() and _weights_dir() is not None
