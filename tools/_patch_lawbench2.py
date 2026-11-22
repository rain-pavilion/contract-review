"""一次性补丁：lawbench 的 entailment 任务要先取 docs 再构造三元组。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

target = ROOT / "zhilv_qiling_v2" / "lawbench.py"
text = target.read_text(encoding="utf-8")

old = """    model, tokenizer, meta = bert_entail.load_model()
    pairs = bert_entail.build_pairs(None, config, per_clause=2, limit_clauses=120)"""
new = """    model, tokenizer, meta = bert_entail.load_model()
    from . import train

    docs = train.collect_docs()
    pairs = bert_entail.build_pairs(docs, config, per_clause=2, limit_clauses=120)"""
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
target.write_text(text, encoding="utf-8")
print("ok entailment docs")

import zhilv_qiling_v2.lawbench as W  # noqa: E402

print("compile ok")
