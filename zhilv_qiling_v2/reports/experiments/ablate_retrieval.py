"""检索消融：对比 raw / rewritten、是否用 tag 通道、是否重排，对齐 v1 协议。

用法：python scripts\_tmp_v2_ret_ablate.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhilv_qiling_v2 import learned as L
from zhilv_qiling_v2 import retrieve as R
from zhilv_qiling_v2.config import V1_STATUTES, WORKSPACE, default_config


def load_specs():
    path = WORKSPACE / "scripts" / "eval_legal_retrieval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target in ("ISSUE_SPECS", "TEMPLATE_TO_TYPE"):
            found[target] = ast.literal_eval(node.value)
    return found["ISSUE_SPECS"], found["TEMPLATE_TO_TYPE"]


def main() -> None:
    specs, _ = load_specs()
    config = default_config()
    statutes = R.load_statutes(V1_STATUTES)
    index = R.StatuteIndex(statutes, config)

    expected = {}
    for code, (strong, weak, text) in specs.items():
        wanted = set(strong) if strong else set(weak)
        gold = {str(a["article_uid"]) for a in statutes if wanted & set(a.get("risk_tags") or [])}
        if gold:
            expected[code] = (gold, text, strong, weak)
    print(f"可用查询 {len(expected)} 条 / 法条 {len(statutes)} 条")

    def run(name, *, rewrite: bool, use_tag: bool, rerank: bool, k=10):
        recalls = {i: 0.0 for i in range(1, k + 1)}
        mrr5 = 0.0
        for code, (gold, text, strong, weak) in expected.items():
            plan = R.rewrite_query(text) if rewrite else R.rewrite_query(text)
            query = plan.rewritten if rewrite else text
            channels = {
                "sparse": index.bm25(query, limit=config.retrieve_candidate_limit),
                "dense": index.lsa(query, limit=config.retrieve_candidate_limit),
            }
            if use_tag:
                channels["tag"] = index.tag_channel(plan.tags, "通用", limit=config.retrieve_candidate_limit)
            retr = R.Retriever(index, config, ranker=L.default_ranker() if rerank else None)
            fused = retr._fuse(channels)[: config.retrieve_candidate_limit]
            scored = []
            for doc_id, feats, rrf in fused:
                doc = index.statutes[doc_id]
                feature_map = retr._features(doc, plan, feats, query_len=len(query))
                feature_map["q_rrf"] = rrf
                score = retr.ranker.score(feature_map) if rerank and retr.ranker else rrf
                scored.append((str(doc.get("article_uid")), score))
            scored.sort(key=lambda pair: -pair[1])
            got = [uid for uid, _ in scored[:k]]
            for i in range(1, k + 1):
                recalls[i] += len(set(got[:i]) & gold) / len(gold)
            first = next((i for i, uid in enumerate(got[:5], start=1) if uid in gold), 0)
            mrr5 += 1.0 / first if first else 0.0
        n = len(expected)
        line = " ".join(f"@{i}={recalls[i]/n:.3f}" for i in (1, 5, 10))
        print(f"{name:34s} {line}  mrr@5={mrr5/n:.3f}")

    run("raw-bm25+lsa", rewrite=False, use_tag=False, rerank=False)
    run("rewrite-bm25+lsa", rewrite=True, use_tag=False, rerank=False)
    run("rewrite+tag", rewrite=True, use_tag=True, rerank=False)
    run("rewrite+tag+hand-prior", rewrite=True, use_tag=True, rerank=True)


if __name__ == "__main__":
    main()
