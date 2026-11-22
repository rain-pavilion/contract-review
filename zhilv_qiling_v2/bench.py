"""统一评测台：把申报书第五节点"系统化评估"落成可复跑的基准。

申报书写的是"调 LawBench 系统化评估"。LawBench 评的是**底座模型**（LLM 的法条记忆/推理
能力），不是"审查系统"的效果；直接拿它当系统指标会名不副实（见
`docs/五节点落地与申报书优化建议.md` 第 5 条）。所以这里做两件事：

1. **对齐 LawBench 的任务口径**：法条预测（给定案情/条款，预测条文）用 R@1/3/5/10 与
   MRR@10 报告 —— 这是 LawBench 里 `1-2 statute prediction` 的指标定义，本台子复刻它，
   用本地 1260 条《民法典》与 27 码金标准，所以叫 lawbench-lite。
2. **给出系统级指标**：缺陷检出（对抗集 issue_codes 为金标准）、端到端延迟分位、
   引用可溯源率、风险分级分布 —— 这些 LawBench 不测，但答辩一定会问。

三层基准（`--tasks retrieval,detection,latency` 可单选）：

- `retrieval`：强/弱金标准分层的 R@1..10 与 MRR@5，纯检索（不含重排）与含重排各一条曲线
- `detection`：对抗集逐合同/逐码的 TP/FP/FN，同时报告"检出但金标未标"的口径差
- `latency`：给定样本集跑端到端，报 P50/P95/最大 与吞吐（份/分钟）

结果写 `reports/bench_<时间戳>.json`，同时把一次 `runs` 账落到存储库里（若有）。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import MODEL_DIR, REPORT_DIR, V1_GENERATED, V1_STATUTES, Config, config_from_env

BENCH_SCHEMA = "1.0"


@dataclass
class RetrievalBench:
    """法条预测（对齐 LawBench statute prediction 的指标形态）。"""

    tier: str = ""
    n_queries: int = 0
    recall: dict[str, float] = field(default_factory=dict)
    mrr5: float = 0.0
    mrr10: float = 0.0
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "n": self.n_queries,
            "recall": {key: round(value, 4) for key, value in self.recall.items()},
            "mrr5": round(self.mrr5, 4),
            "mrr10": round(self.mrr10, 4),
            "source": self.source,
        }


@dataclass
class DetectionBench:
    """缺陷检出（对抗集注入清单为金标准）。"""

    contracts: int = 0
    contract_hit: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_code: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contracts": self.contracts,
            "contract_hit": self.contract_hit,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "per_code": dict(sorted(self.per_code.items())),
        }


@dataclass
class LatencyBench:
    n: int = 0
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    per_minute: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "p50_s": round(self.p50, 3),
            "p95_s": round(self.p95, 3),
            "max_s": round(self.max, 3),
            "mean_s": round(self.mean, 3),
            "docs_per_minute": round(self.per_minute, 2),
        }


def _load_retrieval_models(config: Config):
    """载入语料、索引与重排器（一次，后面共用）。"""
    from . import learned
    from .retrieve import load_statutes
    from .train import _load_issue_specs

    statutes = load_statutes(V1_STATUTES)
    specs, _template_to_type = _load_issue_specs()
    ranker = None
    path = MODEL_DIR / "rerank_ltr.json"
    if path.exists():
        ranker = learned.RetrievalRanker.load(path)
    from .retrieve import StatuteIndex

    return statutes, specs, ranker, StatuteIndex(statutes, config)


def bench_retrieval(config: Config | None = None, *, top_k: int = 10) -> dict[str, Any]:
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
            if not isinstance(row, dict) or "recall@k" not in row:
                continue
            raw_recall = row.get("recall@k") or {}
            recall = {
                f"@{k}": float(raw_recall.get(f"@{k}", 0.0)) for k in range(1, top_k + 1)
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


def _adversarial_records(root: Path | None = None) -> list[dict[str, Any]]:
    path = (root or V1_GENERATED) / "samr_multi_template_2025" / "adversarial" / "manifest.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records") or [])


def bench_detection(config: Config | None = None, *, limit: int | None = None) -> dict[str, Any]:
    """缺陷检出基准：逐码 TP/FP/FN，附"检出但金标未标"的口径差统计。"""
    config = config or config_from_env()
    from . import pipeline

    records = _adversarial_records()
    if limit:
        records = records[:limit]
    bench = DetectionBench()
    per_contract: list[dict[str, Any]] = []
    for record in records:
        docx = record.get("docx") or ""
        path = (V1_GENERATED / "samr_multi_template_2025" / "adversarial" / docx)
        if not path.exists():
            continue
        report = pipeline.review_path(path, config)
        gold = {str(code) for code in (record.get("issue_codes") or [])}
        found = {str(code) for risk in report.risks for code in (risk.tags or [])}
        found = {code for code in found if code}
        tp = gold & found
        fp = found - gold
        fn = gold - found
        bench.contracts += 1
        bench.contract_hit += 1 if not fn else 0
        bench.tp += len(tp)
        bench.fp += len(fp)
        bench.fn += len(fn)
        for code in gold | found:
            row = bench.per_code.setdefault(code, {"tp": 0, "fp": 0, "fn": 0})
            if code in tp:
                row["tp"] += 1
            elif code in fp:
                row["fp"] += 1
            else:
                row["fn"] += 1
        per_contract.append(
            {
                "contract": record.get("contract_id"),
                "gold": sorted(gold),
                "found": sorted(found),
                "score": report.score,
            }
        )
    payload = bench.as_dict()
    payload["per_contract"] = per_contract
    payload["note"] = (
        "FP 里有一部分是【检出但金标未标】：对抗集只标主注入缺陷，模板共享句会被多份合同同时触发，"
        "论文与答辩应把这一栏单列，不当作纯粹的错误率。"
    )
    return payload


def bench_latency(config: Config | None = None, *, limit: int = 12) -> dict[str, Any]:
    """端到端延迟基准：P50/P95/最大 与吞吐。首份含索引构建，单独标出来。"""
    config = config or config_from_env()
    from . import pipeline

    paths = list(pipeline.dataset_files())[:limit]
    timings: list[float] = []
    first = 0.0
    for index, path in enumerate(paths):
        started = time.time()
        pipeline.review_path(path, config)
        elapsed = time.time() - started
        if index == 0:
            first = elapsed
            continue
        timings.append(elapsed)
    if not timings:
        return {"n": 0, "note": "样本不足"}
    timings.sort()

    def _pct(p: float) -> float:
        pos = min(len(timings) - 1, max(0, int(round(p * (len(timings) - 1)))))
        return timings[pos]

    bench = LatencyBench(
        n=len(timings),
        p50=_pct(0.5),
        p95=_pct(0.95),
        max=timings[-1],
        mean=statistics.fmean(timings),
        per_minute=60.0 / statistics.fmean(timings) if statistics.fmean(timings) else 0.0,
    )
    payload = bench.as_dict()
    payload["first_call_s"] = round(first, 3)
    payload["note"] = "首份含法条索引与模型加载，稳态看 P50/P95"
    return payload


def run_bench(
    config: Config | None = None,
    *,
    tasks: Sequence[str] = ("retrieval", "detection", "latency"),
    limit: int | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """跑一轮基准并落盘；返回结构化结果。"""
    config = config or config_from_env()
    result: dict[str, Any] = {
        "schema": BENCH_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": list(tasks),
    }
    started = time.time()
    if "retrieval" in tasks:
        result["retrieval"] = bench_retrieval(config)
    if "detection" in tasks:
        result["detection"] = bench_detection(config, limit=limit)
    if "latency" in tasks:
        result["latency"] = bench_latency(config, limit=limit or 12)
    result["elapsed_seconds"] = round(time.time() - started, 2)
    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["path"] = str(out)
    return result


def summarize(result: dict[str, Any]) -> str:
    """把结果压成几行给人看的摘要（CLI 用）。"""
    lines: list[str] = []
    retrieval = result.get("retrieval") or {}
    curves = retrieval.get("curves") or {}
    for name, tiers in curves.items():
        for tier, row in tiers.items():
            recall = row.get("recall") or {}
            lines.append(
                f"检索 {name:7} {tier:6} R@5={recall.get('@5', 0):.4f} "
                f"R@10={recall.get('@10', 0):.4f} MRR@5={row.get('mrr5', 0):.4f}"
            )
    detection = result.get("detection") or {}
    if detection:
        lines.append(
            f"检出 P={detection.get('precision', 0):.3f} R={detection.get('recall', 0):.3f} "
            f"F1={detection.get('f1', 0):.3f} 合同 {detection.get('contract_hit', 0)}"
            f"/{detection.get('contracts', 0)}"
        )
    latency = result.get("latency") or {}
    if latency:
        lines.append(
            f"延迟 P50={latency.get('p50_s', 0):.2f}s P95={latency.get('p95_s', 0):.2f}s "
            f"首份={latency.get('first_call_s', 0):.2f}s 吞吐={latency.get('docs_per_minute', 0):.0f} 份/分"
        )
    return "\n".join(lines)
