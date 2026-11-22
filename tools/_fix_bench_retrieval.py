"""修 bench 的检索查询与 MRR 口径：
1) 用 ISSUE_SPECS 里自带的 query_text，而不是把风险码的下划线换成空格（后者会低估）。
2) MRR@5 只算前 5 位命中，MRR@10 算前 10 位（之前两者相同是错的）。
"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/bench.py")
text = path.read_text(encoding="utf-8")

old_occ = '''        if expected:
            tier = "strong" if strong else "weak"
            occurrences.append((code, tier, expected))'''
new_occ = '''        if expected:
            tier = "strong" if strong else "weak"
            occurrences.append((code, tier, expected, query))'''
assert old_occ in text
text = text.replace(old_occ, new_occ, 1)

old_occ_type = '''    occurrences: list[tuple[str, str, set[str]]] = []'''
new_occ_type = '''    occurrences: list[tuple[str, str, set[str], str]] = []'''
assert old_occ_type in text
text = text.replace(old_occ_type, new_occ_type, 1)

old_loop = '''            hits_at = {k: 0 for k in range(1, top_k + 1)}
            reciprocal: list[float] = []
            for code, _tier, expected in subset:
                plan = rewrite_query(code.replace("_", " "), contract_type="通用")
                result = retriever.search(
                    plan, top_k=top_k, use_rerank=(name == "rrf_ltr" and ranker is not None)
                )
                ranked = [hit.article_uid for hit in result.hits]
                first = 0
                for rank, uid in enumerate(ranked, start=1):
                    if uid in expected:
                        first = rank
                        break
                for k in range(1, top_k + 1):
                    if ranked[:k] and set(ranked[:k]) & expected:
                        hits_at[k] += 1
                reciprocal.append(1.0 / first if first else 0.0)
            n = len(subset)
            bench = RetrievalBench(
                tier=tier,
                n_queries=n,
                recall={f"@{k}": hits_at[k] / n for k in range(1, top_k + 1)},
                mrr5=sum(reciprocal) / n,
                mrr10=sum(reciprocal) / n,
                source=name,
            )'''
new_loop = '''            hits_at = {k: 0 for k in range(1, top_k + 1)}
            reciprocal5: list[float] = []
            reciprocal10: list[float] = []
            for _code, _tier, expected, query in subset:
                plan = rewrite_query(query or _code, contract_type="通用")
                result = retriever.search(
                    plan, top_k=top_k, use_rerank=(name == "rrf_ltr" and ranker is not None)
                )
                ranked = [hit.article_uid for hit in result.hits]
                first = 0
                for rank, uid in enumerate(ranked, start=1):
                    if uid in expected:
                        first = rank
                        break
                for k in range(1, top_k + 1):
                    if set(ranked[:k]) & expected:
                        hits_at[k] += 1
                reciprocal5.append(1.0 / first if 0 < first <= 5 else 0.0)
                reciprocal10.append(1.0 / first if 0 < first <= 10 else 0.0)
            n = len(subset)
            bench = RetrievalBench(
                tier=tier,
                n_queries=n,
                recall={f"@{k}": hits_at[k] / n for k in range(1, top_k + 1)},
                mrr5=sum(reciprocal5) / n,
                mrr10=sum(reciprocal10) / n,
                source=name,
            )'''
assert old_loop in text
text = text.replace(old_loop, new_loop, 1)

path.write_text(text, encoding="utf-8")
print("bench 检索口径已修")
