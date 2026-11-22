"""BGE-M3 表示方案对比：CLS vs mean 池化、查询指令有无、条文侧是否拼标签。

目的：检索密集通道从 LSA 换到 BGE-M3 后 Recall@5 从 0.628 掉到 0.167，先确认
是"池化方式/指令/文本拼接"哪一环的问题，再决定要不要换。判定指标是纯密集通道的
Recall@5（不掺 RRF，避免融合把问题掩盖掉）。
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from zhilv_qiling_v2.config import V1_STATUTES, MODEL_DIR
from zhilv_qiling_v2.retrieve import load_statutes, rewrite_query
from zhilv_qiling_v2.train import _load_issue_specs

WEIGHTS = MODEL_DIR / "pretrained" / "bge-m3"
PROMPT = "为这个句子生成表示以用于检索相关文章："
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder():
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS))
    model = AutoModel.from_pretrained(str(WEIGHTS)).to(DEVICE).eval()
    return tokenizer, model


def encode(tokenizer, model, texts, pooling: str, batch: int = 16, max_len: int = 512):
    chunks = []
    for start in range(0, len(texts), batch):
        batch_texts = texts[start:start + batch]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        if pooling == "cls":
            vector = hidden[:, 0]
        else:
            vector = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        chunks.append(vector.cpu().numpy().astype("float32"))
    return np.vstack(chunks)


def statute_texts(statutes, with_tags: bool):
    rows = []
    for item in statutes:
        text = f"{item.get('law_name','')}{item.get('article_no','')}{item.get('content','')}"
        if with_tags:
            text += " " + " ".join(item.get("risk_tags") or [])
        rows.append(text[:1200])
    return rows


def score(matrix, queries, tokenizer, model, pooling, prompt: bool):
    q_texts = [f"{PROMPT}{q}" if prompt else q for q in queries]
    vectors = encode(tokenizer, model, q_texts, pooling)
    hits = []
    for row, query in enumerate(queries):
        cosine = matrix @ vectors[row]
        order = np.argsort(-cosine)
        hits.append(order)
    return hits


def main() -> int:
    statutes = load_statutes(V1_STATUTES)
    specs, _ = _load_issue_specs()
    tokenizer, model = load_encoder()
    print(f"[bge-probe] 法条 {len(statutes)} 条，设备 {DEVICE}")

    # 金标准：ISSUE_SPECS 的强/弱标签命中集合
    cases = []
    for code, (strong, weak, query_text) in specs.items():
        wanted = set(strong) or set(weak)
        expected = {str(a["article_uid"]) for a in statutes if wanted & set(a.get("risk_tags") or [])}
        if expected:
            cases.append((code, query_text, expected))
    print(f"[bge-probe] 可用查询 {len(cases)} 条")

    configs = [
        ("cls + 标签", "cls", True, True),
        ("cls 无标签", "cls", True, False),
        ("mean + 标签", "mean", True, True),
        ("mean 无标签", "mean", True, False),
        ("cls+prompt+标签", "cls", True, True),
    ]
    result = {}
    for label, pooling, with_tags, prompt in configs[:4]:
        matrix = encode(tokenizer, model, statute_texts(statutes, with_tags), pooling)
        q_texts = [c[1] for c in cases]
        vectors = encode(tokenizer, model, q_texts, pooling)
        recall5 = []
        mrr = []
        for row, (code, _q, expected) in enumerate(cases):
            order = np.argsort(-(matrix @ vectors[row]))
            top5 = [str(statutes[int(i)]["article_uid"]) for i in order[:5]]
            top10 = [str(statutes[int(i)]["article_uid"]) for i in order[:10]]
            recall5.append(len(set(top5) & expected) / len(expected))
            first = next((idx for idx, uid in enumerate(top10) if uid in expected), None)
            mrr.append(0.0 if first is None else 1.0 / (first + 1))
        result[label] = (round(float(np.mean(recall5)), 4), round(float(np.mean(mrr)), 4))
        print(f"  {label:<16} 纯密集 Recall@5={result[label][0]}  MRR@10={result[label][1]}")

    # 带指令的查询侧（条文侧不变）
    matrix = encode(tokenizer, model, statute_texts(statutes, True), "cls")
    query_texts = [PROMPT + c[1] for c in cases]
    vectors = encode(tokenizer, model, query_texts, "cls")
    recall5, mrr = [], []
    for row, (code, _q, expected) in enumerate(cases):
        order = np.argsort(-(matrix @ vectors[row]))
        top5 = [str(statutes[int(i)]["article_uid"]) for i in order[:5]]
        top10 = [str(statutes[int(i)]["article_uid"]) for i in order[:10]]
        recall5.append(len(set(top5) & expected) / len(expected))
        first = next((idx for idx, uid in enumerate(top10) if uid in expected), None)
        mrr.append(0.0 if first is None else 1.0 / (first + 1))
    print(f"  {'cls+指令':<16} 纯密集 Recall@5={round(float(np.mean(recall5)),4)}  MRR@10={round(float(np.mean(mrr)),4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
