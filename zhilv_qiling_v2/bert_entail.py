"""条款—条文双塔判别（BERT + 余弦 + 多头注意力 + 分类头，GPU 版）。

申报书节点④写的是"BERT/Mamba 编码 → 余弦相似度 → 多头注意力判一致/冲突"。
`entailment.py` 里的弱监督判别器只用了 8 维手写特征，判得了"均不承担任何赔偿责任"
对第五百七十七条这种显式冲突，判不了含蓄的改述。本文件把那三步**真的实现**：

1. 双塔编码：条款与条文各过一遍 BERT（共享权重），取 `[CLS]` 作句向量；
2. 余弦相似度：句向量点积，作为一个标量特征，也作为可解释信号单独输出；
3. 多头注意力：以条款 token 为 query、条文 token 为 key/value 做 8 头 cross-attention，
   得到"条款每个字在看条文的哪一段"的对齐表示，再 masked-mean 池化。

监督信号来自哪里（这点必须诚实）
--------------------------------
PROVDATA 那类人工标注的矛盾/蕴含对没有，所以用 `entailment.weak_label` 给
"条款 × 检索到的条文"自动打标（弱监督蒸馏）：老师是 8 维规则，学生是 BERT。
好处是学生能借预训练语义把老师的判断泛化到没见过的表述；
坏处是学生的上限被老师框住 —— 想突破必须有人工标注，这一点写进 docs。

对外接口
--------
- `train(pairs, ...) -> dict`：训练并落盘 `models/bert_entail/`
- `load_model()`：读回 (model, tokenizer, meta)
- `Predictor`：`predict(clause, article) -> (label, score, cosine, attention_peaks)`
"""

from __future__ import annotations

import json
import hashlib
import math
import pathlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from . import entailment as E
from . import entailment_windows as W
from .config import MODEL_DIR

from . import pretrained as _pretrained

PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"


def _weights_dir():
    """返回可用的中文 BERT 权重目录。

    优先走 `pretrained.resolve()`（按 config.json + 权重体积探测，
    兼容 chinese-roberta-wwm-ext / bert-base-chinese 等任意中文 BERT）；
    一个都没探测到时回落到默认路径，交给调用方报错。
    """
    found = _pretrained.resolve()
    if found is not None:
        return found
    return PRETRAINED_DIR if PRETRAINED_DIR.exists() else None

OUTPUT_DIR = MODEL_DIR / "bert_entail"
LABELS = E.LABELS
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
MAX_CLAUSE = 192
MAX_ARTICLE = 256
HEADS = 8


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "需要 torch（GPU 版建议 cu128）：python -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cu128"
        ) from exc
    return torch


def device_of(prefer_cuda: bool = True):
    torch = _torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(*, dropout: float = 0.15, rule_dim: int = 8, enable_mamba: bool = False,
                enable_rule_head: bool = True, mamba_config: dict[str, Any] | None = None):
    """双塔 + cross-attention + 对比特征 → 三分类。"""
    torch = _torch()
    nn = torch.nn
    from transformers import AutoModel

    class ClauseArticleJudge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = _weights_dir() or PRETRAINED_DIR
            self.encoder = AutoModel.from_pretrained(str(weights), local_files_only=True)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.attn = nn.MultiheadAttention(hidden, HEADS, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            # Mamba is optional and may only participate when its weights were
            # actually trained and saved. Old BERT checkpoints keep their exact architecture.
            self.mamba = _build_mamba(hidden, nn) if enable_mamba else None
            if self.mamba is not None:
                self.mamba_gate = nn.Parameter(torch.zeros(1))
                scan_hidden = self.mamba.config.hidden_size
                self.mamba_input_projection = nn.Linear(hidden, scan_hidden) if scan_hidden != hidden else nn.Identity()
                self.mamba_output_projection = nn.Linear(scan_hidden, hidden) if scan_hidden != hidden else nn.Identity()
            # 特征：[pool_c, pool_a, |c-a|, c*a, cos, attn_pool]
            # 5 个 hidden 向量 + 1 个标量 = 5*hidden + 1（早期写成 6*hidden+1，实跑报
            # mat1 and mat2 shapes cannot be multiplied (8x3841 and 4609x768)，已按真实拼接宽度改）
            self.head = nn.Sequential(
                nn.Linear(hidden * 5 + 1, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(LABELS)),
            )
            # 规则残差旁路：把 entailment 的 8 维特征接一条线性层，与 BERT 头相加。
            # 权重用 DEFAULT_WEIGHTS 精确初始化，且 head 最后一层零初始化 ——
            # 训练起点恰好等于规则后端的判断，模型只在其上学残差。
            # 是否优于规则仍须独立评测；加载旧权重时不会凭空增加该旁路。
            self.rule_head = nn.Linear(rule_dim, len(LABELS)) if enable_rule_head else None
            with torch.no_grad():
                if self.rule_head is not None:
                    for index, label in enumerate(LABELS):
                        row = E.DEFAULT_WEIGHTS.get(label, {})
                        for slot, name in enumerate(E.FEATURES):
                            self.rule_head.weight[index, slot] = float(row.get(name, 0.0))
                        self.rule_head.bias[index] = float(row.get("_bias", 0.0))
                    for param in self.head[-1].parameters():
                        nn.init.zeros_(param)

        def forward(self, clause_ids, clause_mask, article_ids, article_mask, rule_feats=None):
            clause = self.encoder(input_ids=clause_ids, attention_mask=clause_mask).last_hidden_state
            article = self.encoder(input_ids=article_ids, attention_mask=article_mask).last_hidden_state
            pool_c = _masked_mean(clause, clause_mask)
            pool_a = _masked_mean(article, article_mask)
            if self.mamba is not None:
                embed = self.encoder.get_input_embeddings()
                gate = torch.sigmoid(self.mamba_gate)
                scan_c = self.mamba(inputs_embeds=self.mamba_input_projection(embed(clause_ids)), attention_mask=clause_mask).last_hidden_state
                scan_a = self.mamba(inputs_embeds=self.mamba_input_projection(embed(article_ids)), attention_mask=article_mask).last_hidden_state
                pool_c = pool_c + gate * self.mamba_output_projection(_masked_mean(scan_c, clause_mask))
                pool_a = pool_a + gate * self.mamba_output_projection(_masked_mean(scan_a, article_mask))
            attended, weights = self.attn(
                query=clause,
                key=article,
                value=article,
                key_padding_mask=(article_mask == 0),
                need_weights=True,
                average_attn_weights=True,
            )
            attended = self.norm(attended + clause)
            pool_attn = _masked_mean(self.dropout(attended), clause_mask)
            cosine = nn.functional.cosine_similarity(pool_c, pool_a, dim=1, eps=1e-8).unsqueeze(1)
            features = torch.cat(
                [pool_c, pool_a, (pool_c - pool_a).abs(), pool_c * pool_a, cosine, pool_attn], dim=1
            )
            logits = self.head(features)
            if rule_feats is not None and self.rule_head is not None:
                logits = logits + self.rule_head(rule_feats)
            return logits, cosine.squeeze(1), weights

    def _build_mamba(hidden: int, nn):
        """Mamba（选择性状态空间）序列编码器，纯 PyTorch 参考实现。

        没有安装 `mamba_ssm` 时 transformers 会回落到参考实现：结果正确但更慢，
        因此层数压到 2、state_size 压到 16，只在 GPU 上跑。
        权重是随机初始化、随双塔任务一起训练（本机取不到 Mamba 预训练权重），
        所以它提供的是"另一条序列建模通道"，而不是迁移学习。
        """
        try:
            from transformers.models.mamba.modeling_mamba import MambaConfig, MambaModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Mamba was explicitly requested but its implementation is unavailable") from exc
        options = {"hidden_size": hidden, "num_hidden_layers": 2, "state_size": 16,
                   "expand": 2, "conv_kernel": 4, "use_cache": False, "use_mamba_kernels": False}
        options.update(mamba_config or {})
        config = MambaConfig(**options)
        return MambaModel(config)

    def _masked_mean(hidden, mask):
        weights = mask.unsqueeze(2).float()
        total = (hidden * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp(min=1.0)
        return total / count

    return ClauseArticleJudge()


def build_pairs(docs: Sequence[Any], config=None, *, per_clause: int = 3, limit_clauses: int = 400,
                seed: int = 20261104) -> list[dict[str, Any]]:
    """从语料里构造 (条款, 条文, 弱标签) 三元组。

    条款来自真实/合成合同的切分结果；条文来自三路召回的 top-N；
    标签由 `entailment.weak_label` 给出（弱监督，见模块说明）。
    """
    import random

    from .config import V1_STATUTES
    from .pipeline import prepare
    from .retrieve import get_retriever, rewrite_query
    from .train import doc_text

    rng = random.Random(seed)
    retriever = get_retriever(V1_STATUTES, config)
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        if len(pairs) >= limit_clauses * per_clause:
            break
        try:
            document = prepare(doc.path.name, path=doc.path, config=config)
        except Exception:
            continue
        clauses = [clause for clause in document.clauses if len(clause.text.strip()) >= 20]
        rng.shuffle(clauses)
        for clause in clauses[: max(1, limit_clauses // max(1, len(docs)))]:
            key = clause.text.strip()[:60]
            if key in seen:
                continue
            seen.add(key)
            plan = rewrite_query(clause.text, contract_type=getattr(doc, "kind", "通用") or "通用")
            try:
                result = retriever.search(plan, top_k=per_clause)
            except Exception:
                continue
            for hit in result.hits[:per_clause]:
                features = E.clause_features(
                    clause.text, hit.content, risk_tags=hit.risk_tags or [], codes=()
                )
                label = E.weak_label(features)
                if label not in LABEL2ID:
                    continue
                pairs.append(
                    {
                        "clause": clause.text.strip(),
                        "article": f"{hit.law_name}{hit.article_no} {hit.content}".strip(),
                        "label": label,
                        "features": features,
                        "article_uid": hit.article_uid,
                        "contract": doc_text(doc)[:0],  # 占位，避免重复读盘
                    }
                )
    return pairs


def train(pairs: Sequence[dict[str, Any]], *, epochs: int = 6, batch_size: int = 8,
          lr: float = 2e-5, ratio: float = 0.2, seed: int = 20261104) -> dict[str, Any]:
    torch = _torch()
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    if _weights_dir() is None:
        raise RuntimeError(f"缺少预训练权重：{PRETRAINED_DIR}（可用 models/pretrained/ 下任一中文 BERT）")
    if not pairs:
        raise RuntimeError("没有可训练的三元组")
    torch.manual_seed(seed)
    device = device_of()

    rows = list(pairs)
    torch_generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=torch_generator).tolist()
    split = max(1, int(len(rows) * (1 - ratio)))
    train_rows = [rows[i] for i in order[:split]]
    valid_rows = [rows[i] for i in order[split:]] or train_rows[: min(64, len(train_rows))]

    tokenizer = AutoTokenizer.from_pretrained(str(_weights_dir()))

    def encode(batch: Sequence[dict[str, Any]]):
        clauses = tokenizer(
            [row["clause"] for row in batch],
            padding=True, truncation=True, max_length=MAX_CLAUSE, return_tensors="pt",
        )
        articles = tokenizer(
            [row["article"] for row in batch],
            padding=True, truncation=True, max_length=MAX_ARTICLE, return_tensors="pt",
        )
        labels = torch.tensor([LABEL2ID[row["label"]] for row in batch])
        feats = torch.tensor(
            [[float(row["features"].get(name, 0.0)) for name in E.FEATURES] for row in batch],
            dtype=torch.float32,
        )
        return clauses, articles, labels, feats

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, math.ceil(len(train_rows) / batch_size) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(steps * 0.1)), num_training_steps=steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_rows), generator=torch_generator).tolist()
        total, batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch = [train_rows[i] for i in perm[start:start + batch_size]]
            clauses, articles, labels, feats = encode(batch)
            logits, cosine, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
                rule_feats=feats.to(device),
            )
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            batches += 1
        average = total / max(1, batches)
        history.append({"epoch": epoch, "loss": round(average, 4)})
        print(f"  epoch {epoch}/{epochs} loss={average:.4f}")

    metrics = evaluate(model, tokenizer, valid_rows, device=device, batch_size=batch_size)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    meta = {
        "valid": metrics,
        "history": history,
        "pairs": len(rows),
        "label_counts": _label_counts(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "supervision": "weak_label（entailment 8 维规则的蒸馏，非人工标注）",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "pairs": len(rows),
        "history": history,
        "valid": metrics,
        "valid_rows": valid_rows,
        "device": str(device),
    }


def _label_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def evaluate(model, tokenizer, rows: Sequence[dict[str, Any]], *, device=None,
             batch_size: int = 16) -> dict[str, Any]:
    torch = _torch()
    device = device or device_of()
    model.eval()
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            clauses = tokenizer([row["clause"] for row in batch], padding=True, truncation=True,
                                max_length=MAX_CLAUSE, return_tensors="pt")
            articles = tokenizer([row["article"] for row in batch], padding=True, truncation=True,
                                 max_length=MAX_ARTICLE, return_tensors="pt")
            feats = torch.tensor(
                [[float(row["features"].get(name, 0.0)) for name in E.FEATURES] for row in batch],
                dtype=torch.float32,
            )
            logits, _, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
                rule_feats=feats.to(device),
            )
            predicted = logits.argmax(dim=1).tolist()
            for row, index in zip(batch, predicted):
                confusion[row["label"]][LABELS[index]] += 1
    per_label: dict[str, list[float]] = {}
    f1_values: list[float] = []
    accuracy_hits = sum(confusion[label][label] for label in LABELS)
    total = sum(sum(row.values()) for row in confusion.values())
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = [round(precision, 4), round(recall, 4), round(f1, 4), tp + fn]
        f1_values.append(f1)
    return {
        "rows": total,
        "accuracy": round(accuracy_hits / total, 4) if total else 0.0,
        "per_label": per_label,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "confusion": confusion,
    }


def evaluate_rules(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """同一批样本上跑 rules 后端（8 维特征判别），口径与 evaluate() 完全一致。

    为什么要这个：训练出来的双塔如果还不如规则，必须当场看得见。
    旧版就是这么埋掉的 —— 模型 macro-F1 0.5771 看着像"有效果"，但它把
    「均不承担赔偿责任」判成 neutral 0.9995，而规则判 contradict 0.9983。
    """
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    for row in rows:
        verdict = E.judge_pair(
            row["clause"],
            {"article_no": "", "content": row["article"], "law_name": "", "risk_tags": []},
            codes=(),
        )
        if verdict.label in confusion.get(row["label"], {}):
            confusion[row["label"]][verdict.label] += 1
    per_label: dict[str, list[float]] = {}
    f1_values: list[float] = []
    total = sum(sum(row.values()) for row in confusion.values())
    hits = sum(confusion[label][label] for label in LABELS)
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = [round(precision, 4), round(recall, 4), round(f1, 4), tp + fn]
        f1_values.append(f1)
    return {
        "rows": total,
        "accuracy": round(hits / total, 4) if total else 0.0,
        "per_label": per_label,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "confusion": confusion,
    }


def selected_model_dir(dir_path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Resolve an explicit checkpoint selection without substituting another model.

    The opt-in environment variable affects newly started processes. Omitting it
    preserves the legacy BERT checkpoint and its existing default behavior.
    """
    if dir_path is not None:
        return pathlib.Path(dir_path).expanduser().resolve()
    configured = os.environ.get("ZLQL_ENTAILMENT_MODEL_DIR", "").strip()
    return (pathlib.Path(configured).expanduser() if configured else OUTPUT_DIR).resolve()


def load_model(dir_path: pathlib.Path | str | None = None):
    torch = _torch()
    from transformers import AutoTokenizer

    path = selected_model_dir(dir_path)
    weights = path / "model.pt"
    if not weights.exists():
        return None, None, {}
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    has_mamba = any(key.startswith("mamba.") for key in state)
    has_rule_head = any(key.startswith("rule_head.") for key in state)
    meta = {}
    meta_file = path / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    architecture = meta.get("architecture") or {}
    if architecture.get("mamba") and not has_mamba:
        raise RuntimeError("Checkpoint metadata claims Mamba but its parameters are missing")
    model = build_model(enable_mamba=has_mamba, enable_rule_head=has_rule_head,
                        mamba_config=architecture.get("mamba_config") if has_mamba else None,
                        dropout=float(architecture.get("dropout", 0.15)))
    # Do not silently inject an untrained random Mamba branch into an older BERT checkpoint.
    model.load_state_dict(state, strict=True)
    model.to(inference_device())
    meta["loaded_architecture"] = {"mamba": has_mamba, "rule_residual": has_rule_head,
                                   "strict_checkpoint_load": True,
                                   "mamba_config": model.mamba.config.to_dict() if has_mamba else None}
    # Existing runtime diagnostics expose Predictor.meta; no extra model load is
    # needed when the health endpoint reports this checkpoint's provenance.
    supervision = str(meta.get("supervision") or "unspecified_in_checkpoint_metadata")
    meta["loaded_checkpoint"] = {
        "directory": str(path), "weights_file": str(weights),
        "selection": "explicit_argument" if dir_path is not None else
                     "environment" if os.environ.get("ZLQL_ENTAILMENT_MODEL_DIR", "").strip() else "legacy_default",
        "supervision": supervision,
        "weak_supervision": "weak" in supervision,
        "human_gold": False if "not_human_gold" in supervision else meta.get("human_gold"),
        "training_scope": str(meta.get("training_scope") or "unspecified_in_checkpoint_metadata"),
    }
    return model, tokenizer, meta


class WindowInferenceError(RuntimeError):
    """A failed real window call, retaining completed scopes without input text."""

    def __init__(self, audit: dict[str, Any], reason: str = "bert_window_prediction_failed"):
        super().__init__(reason)
        self.window_audit = audit
        self.reason_code = reason


@dataclass
class Predictor:
    """推理封装：给条款与条文，返回标签、置信、余弦与注意力峰值位置。"""

    model: Any
    tokenizer: Any
    device: Any = None
    meta: dict[str, Any] | None = None
    max_window_pairs: int | None = None

    def __post_init__(self) -> None:
        self.device = self.device or inference_device()
        self.model.to(self.device).eval()

    def _source_tokens(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        try:
            encoded = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                     return_offsets_mapping=True)
        except (NotImplementedError, TypeError):
            # Slow tokenizers need not implement offsets. Token IDs and hashes
            # remain authoritative; do not invent source character positions.
            encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            offsets = [(int(start), int(end)) for start, end in offsets]
            if len(offsets) != len(ids):
                raise ValueError("token_offset_count_mismatch")
        return ids, offsets

    def _window_text(self, source: str, window: W.TokenWindow) -> tuple[str, str]:
        if window.char_start is not None and window.char_end is not None:
            return source[window.char_start:window.char_end], "source_character_offsets"
        try:
            return self.tokenizer.decode(list(window.token_ids), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False), "decoded_original_token_ids"
        except TypeError:
            return self.tokenizer.decode(list(window.token_ids), skip_special_tokens=True), "decoded_original_token_ids"

    def _run_window(self, clause_ids: Sequence[int], article_ids: Sequence[int],
                    features: dict[str, float], *, full_clause: str | None = None,
                    full_article: str | None = None) -> dict[str, Any]:
        """Execute the unchanged trained model on one already-bounded pair."""
        torch = _torch()
        if full_clause is not None and full_article is not None:
            # Preserve the original one-window encoding behavior exactly.
            clause_encoded = self.tokenizer(full_clause, truncation=True, max_length=MAX_CLAUSE, return_tensors="pt")
            article_encoded = self.tokenizer(full_article, truncation=True, max_length=MAX_ARTICLE, return_tensors="pt")
        else:
            clause_encoded = self.tokenizer.prepare_for_model(
                list(clause_ids), add_special_tokens=True, truncation=False,
                return_attention_mask=True, return_tensors="pt", prepend_batch_axis=True)
            article_encoded = self.tokenizer.prepare_for_model(
                list(article_ids), add_special_tokens=True, truncation=False,
                return_attention_mask=True, return_tensors="pt", prepend_batch_axis=True)
        if clause_encoded["input_ids"].shape[-1] > MAX_CLAUSE or article_encoded["input_ids"].shape[-1] > MAX_ARTICLE:
            raise ValueError("prepared_window_exceeds_trained_budget")
        rule_feats = torch.tensor(
            [[float(features.get(name, 0.0)) for name in E.FEATURES]], dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            logits, cosine, weights = self.model(
                clause_encoded["input_ids"].to(self.device),
                clause_encoded["attention_mask"].to(self.device),
                article_encoded["input_ids"].to(self.device),
                article_encoded["attention_mask"].to(self.device),
                rule_feats=rule_feats,
            )
            probability = torch.softmax(logits, dim=1)[0]
        best = int(probability.argmax())
        article_tokens = self.tokenizer.convert_ids_to_tokens(article_encoded["input_ids"][0])[: weights.shape[2]]
        attention = weights[0].mean(dim=0).cpu().numpy()
        peaks = sorted(range(len(attention)), key=lambda i: -attention[i])[:5]
        return {
            "label": LABELS[best],
            "score": round(float(probability[best]), 4),
            "probabilities": {LABELS[i]: round(float(probability[i]), 4) for i in range(len(LABELS))},
            "cosine": round(float(cosine[0]), 4),
            "article_tokens_at_peak": [article_tokens[i] for i in peaks if i < len(article_tokens)],
        }

    def predict(self, clause: str, article: str) -> dict[str, Any]:
        """Run source-token windows; never equate local agreement with a full proof."""
        from . import model_audit
        clause_ids, clause_offsets = self._source_tokens(clause)
        article_ids, article_offsets = self._source_tokens(article)
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        if special_count < 0 or special_count >= min(MAX_CLAUSE, MAX_ARTICLE):
            raise ValueError("invalid_special_token_budget")
        max_pairs = int(self.max_window_pairs if self.max_window_pairs is not None else
                        os.environ.get("ZLQL_ENTAILMENT_MAX_WINDOW_PAIRS", "64"))
        plan = W.make_plan(clause_ids, article_ids,
                           clause_capacity=MAX_CLAUSE - special_count,
                           article_capacity=MAX_ARTICLE - special_count,
                           clause_offsets=clause_offsets, article_offsets=article_offsets,
                           max_pairs=max_pairs)
        call_id = uuid.uuid4().hex
        if not clause_ids or not article_ids:
            audit = W.summarize(plan, [])
            audit.update(call_id=call_id,
                         clause_source_sha256=hashlib.sha256(clause.encode("utf-8")).hexdigest(),
                         article_source_sha256=hashlib.sha256(article.encode("utf-8")).hexdigest(),
                         special_tokens_per_input=special_count,
                         max_clause_input_tokens=MAX_CLAUSE, max_article_input_tokens=MAX_ARTICLE)
            model_audit.record("entailment", "bert_entail", "validate_input", "skipped", 0,
                               "bert_empty_token_input", call_id=call_id, backend="bert", pairs=0,
                               token_coverage_complete=False, pair_coverage_complete=False,
                               semantic_pending=True)
            raise WindowInferenceError(audit, "bert_empty_token_input")
        predictions: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        failure: Exception | None = None
        single = plan.planned_pair_count == 1
        for pair in plan.scheduled_pairs:
            clause_scope, article_scope = pair.clause.audit(), pair.article.audit()
            payload = json.dumps([pair.clause.token_ids, pair.article.token_ids], separators=(",", ":")).encode("ascii")
            receipt = {"call_id": call_id, "window_id": pair.pair_id,
                       "input_sha256": hashlib.sha256(payload).hexdigest(),
                       "backend": "bert", "device": str(self.device), "pairs": 1,
                       "clause_token_start": pair.clause.token_start,
                       "clause_token_end": pair.clause.token_end,
                       "article_token_start": pair.article.token_start,
                       "article_token_end": pair.article.token_end}
            if single:
                clause_text, article_text = clause, article
                feature_scope = {"clause": "whole_input", "article": "whole_input"}
            else:
                clause_text, clause_feature_scope = self._window_text(clause, pair.clause)
                article_text, article_feature_scope = self._window_text(article, pair.article)
                feature_scope = {"clause": clause_feature_scope, "article": article_feature_scope}
            features = E.clause_features(clause_text, article_text, risk_tags=(), codes=())
            started = time.perf_counter()
            model_audit.record("entailment", "bert_entail", "predict_window", "started", **receipt)
            try:
                result = self._run_window(pair.clause.token_ids, pair.article.token_ids, features,
                                          full_clause=clause if single else None,
                                          full_article=article if single else None)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                model_audit.record("entailment", "bert_entail", "predict_window", "failed", elapsed,
                                   "bert_window_prediction_failed", error_type=type(exc).__name__, **receipt)
                failures.append({"pair_id": pair.pair_id, "error_type": type(exc).__name__,
                                 "reason": "bert_window_prediction_failed", "clause_scope": clause_scope,
                                 "article_scope": article_scope, "elapsed_seconds": round(elapsed, 6)})
                failure = exc
                # Retain successful results and explicitly leave the rest pending.
                # Repeating an OOM/device error across every planned pair is unsafe.
                break
            elapsed = time.perf_counter() - started
            model_audit.record("entailment", "bert_entail", "predict_window", "succeeded", elapsed, **receipt)
            predictions.append({**result, "pair_id": pair.pair_id,
                                "clause_scope": clause_scope, "article_scope": article_scope,
                                "feature_scope": feature_scope, "elapsed_seconds": round(elapsed, 6)})
        audit = W.summarize(plan, predictions, failures=failures)
        audit.update(call_id=call_id, clause_source_sha256=hashlib.sha256(clause.encode("utf-8")).hexdigest(),
                     article_source_sha256=hashlib.sha256(article.encode("utf-8")).hexdigest(),
                     special_tokens_per_input=special_count,
                     max_clause_input_tokens=MAX_CLAUSE, max_article_input_tokens=MAX_ARTICLE)
        selected = audit["selected_prediction"]
        if selected is None:
            raise WindowInferenceError(audit) from failure
        return {key: selected[key] for key in
                ("label", "score", "probabilities", "cosine", "article_tokens_at_peak")} | {
            "clause_tokens": len(clause_ids) + special_count,
            "article_tokens": len(article_ids) + special_count,
            "input_truncated": not audit["token_coverage_complete"],
            "coverage_complete": not audit["semantic_pending"],
            "token_coverage_complete": audit["token_coverage_complete"],
            "pair_coverage_complete": audit["pair_coverage_complete"],
            "semantic_pending": audit["semantic_pending"],
            "manual_required": audit["manual_required"],
            "window_disagreement": audit["window_disagreement"],
            "decision_scope": audit["decision_scope"], "window_audit": audit,
        }


def available() -> bool:
    return (selected_model_dir() / "model.pt").exists() and _weights_dir() is not None


_PREDICTOR: dict[str, object] = {}


def load_predictor(dir_path: pathlib.Path | str | None = None):
    """Cache successful predictors by resolved checkpoint path, never failures.

    An explicitly missing or incompatible checkpoint raises rather than falling
    back to the legacy directory. The caller may separately audit a rules fallback.
    """
    path = selected_model_dir(dir_path)
    cache_key = str(path).casefold() if os.name == "nt" else str(path)
    cached = _PREDICTOR.get(cache_key)
    if cached is not None:
        return cached
    model, tokenizer, meta = load_model(dir_path)
    if model is None or tokenizer is None:
        raise FileNotFoundError(f"entailment checkpoint is missing: {path}")
    predictor = Predictor(model=model, tokenizer=tokenizer, meta=meta)
    _PREDICTOR[cache_key] = predictor
    return predictor


def inference_device():
    torch = _torch()
    requested = os.environ.get("ZLQL_ENTAILMENT_DEVICE", os.environ.get("ZLQL_BERT_DEVICE", "cpu")).lower()
    if requested == "auto":
        return device_of()
    if requested not in ("cpu", "cuda"):
        raise ValueError("ZLQL_ENTAILMENT_DEVICE must be cpu, cuda or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested entailment CUDA device is unavailable")
    return torch.device(requested)
