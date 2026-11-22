"""把 bench 的检索任务改为**复用 train.evaluate_retrieval** 的协议实现。

上一版自己在 bench 里重写了一遍指标：查询用了 ISSUE_SPECS 的 query_text 但合同类型固定
"通用"、occurrence 按风险码去重、MRR 口径另算 —— 于是得到 rrf 0.875 / ltr 0.125 这种
和论文里 0.6278 / 0.6944 完全不同的数。同一套指标出现两个口径是写论文的大忌。

正确做法：指标只有一处实现（`train.evaluate_retrieval`，与 v1 脚本逐行对齐），
bench 只负责"调度 + 落库 + 报表"。
"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/bench.py")
text = path.read_text(encoding="utf-8")

start = text.index("def bench_retrieval(")
end = text.index("def _adversarial_records(")
new_fn = '''def bench_retrieval(config: Config | None = None, *, top_k: int = 10) -> dict[str, Any]:
    """法条预测基准：直接复用 `train.evaluate_retrieval`（与 v1 评测脚本同协议）。

    指标口径只允许有一处实现 —— 之前 bench 自己重算过一次，得到与论文不同的数字，
    这种"两个口径"在答辩现场是硬伤。这里只做形状转换与分层整理。
    """
    config = config or config_from_env()
    from . import learned
    from .retrieve import Retriever, StatuteIndex, load_statutes
    from .train import _load_issue_specs, evaluate_retrieval

    statutes = load_statutes(V1_STATUTES)
    specs, _template_to_type = _load_issue_specs()
    index = StatuteIndex(statutes, config)
    ranker = None
    weights = MODEL_DIR / "rerank_ltr.json"
    if weights.exists():
        ranker = learned.RetrievalRanker.load(weights)
    retriever = Retriever(index, config, ranker=ranker)

    raw = evaluate_retrieval(retriever, index, statutes, specs, ranker, config, top_k=top_k)
    curves: dict[str, dict[str, Any]] = {}
    for curve_name, tiers in raw.items():
        if not isinstance(tiers, dict):
            continue
        bucket: dict[str, Any] = {}
        for tier, row in tiers.items():
            if not isinstance(row, dict) or "recall@1" not in row:
                continue
            recall = {
                f"@{k}": float(row.get(f"recall@{k}", 0.0)) for k in range(1, top_k + 1)
            }
            bench = RetrievalBench(
                tier=tier,
                n_queries=int(row.get("n", 0) or 0),
                recall=recall,
                mrr5=float(row.get("mrr@5", 0.0) or 0.0),
                mrr10=float(row.get("mrr@5", 0.0) or 0.0),
                source=curve_name,
            )
            bucket[tier] = bench.as_dict()
        if bucket:
            curves[curve_name] = bucket
    return {"curves": curves, "protocol": "train.evaluate_retrieval（v1 兼容）"}


'''
text = text[:start] + new_fn + text[end:]
path.write_text(text, encoding="utf-8")
print("bench_retrieval 已改为复用 train.evaluate_retrieval")
