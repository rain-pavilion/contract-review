"""LawBench 式系统化评测（申报书节点⑤"调用 LawBench 对系统进行系统化评估"的落点）。

**为什么不是"直接用 LawBench"**：LawBench 的 20 个子任务是"给定一段法律文本，做单标签
分类 / 多标签分类 / 抽取 / 回归 / 生成"，训练与测试都依赖它自己的标注数据（公开可下载，
但需要按它的格式重训或至少按它的划分微调）。本项目用的是自建的 113 份合同语料，没法直接
报 LawBench 分数——硬报就是造假。

**这里做的是"对齐 LawBench 的评测口径"**：把本项目已有的五项能力，逐一映射到 LawBench
的任务类型与指标类型上，用同一套"任务ID → 任务类型 → 指标"的表来汇报，这样：

1. 评审能看懂"你的抽取任务相当于 LawBench 2-6、你的法条推荐相当于 3-1"；
2. 指标类型与 LawBench 一致（micro/macro-F1、Recall@k、MRR、Accuracy），读者可横向比；
3. 数据来源标注清楚（自建语料 / 对抗集 / 真实政采合同），不冒用 LawBench 划分。

映射表（LawBench 任务 ID 取自 EMNLP 2024 论文第 3 节）：

| LawBench 任务 | 类型 | 本项目落点 | 指标 |
|---|---|---|---|
| 2-6 Extraction | extraction | 节点① BERT-CRF 要素抽取 | micro / macro F1 |
| 2-3 MLC (20 类) | multi-label | 27 风险码检出（对抗集金标） | micro / macro P/R/F1 |
| 3-1 MLC (118 类) | multi-label 排序 | 节点④ 法条推荐（v1 协议） | Recall@1/3/5/10、MRR@5 |
| 2-4 SLC (20 类) | single-label | 节点④ 条款—条文一致性判别 | Accuracy、macro-F1 |
| 1-2 SLC (4 类) | single-label | 节点③ 主体资格审查（需/不需资质） | Accuracy（弱监督，标注为推断） |
| generation | generation | 节点⑤ 复核意见 | 无自动指标（只报生成率与长度） |
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config, MODEL_DIR, WORKSPACE
from . import config as C

REPORT = WORKSPACE / "zhilv_qiling_v2" / "reports" / "lawbench.json"

# LawBench 任务类型 → 本项目对应能力（供表格展示与横向比较）
TASK_TABLE: tuple[dict[str, str], ...] = (
    {"lawbench": "2-6 Extraction", "type": "extraction",
     "ours": "节点① 要素抽取（BERT-CRF）", "metric": "micro/macro-F1"},
    {"lawbench": "2-3 MLC (20 类)", "type": "multi-label",
     "ours": "27 风险码检出（对抗集）", "metric": "P/R/F1（micro）"},
    {"lawbench": "3-1 MLC (118 类)", "type": "multi-label ranking",
     "ours": "节点④ 法条推荐", "metric": "Recall@1/3/5/10、MRR@5"},
    {"lawbench": "2-4 SLC (20 类)", "type": "single-label",
     "ours": "节点④ 条款—条文一致性判别", "metric": "Accuracy、macro-F1"},
    {"lawbench": "1-2 SLC (4 类)", "type": "single-label",
     "ours": "节点③ 主体资格审查", "metric": "Accuracy（弱监督）"},
    {"lawbench": "generation", "type": "generation",
     "ours": "节点⑤ 复核意见", "metric": "无自动指标"},
)


@dataclass
class TaskResult:
    lawbench_id: str
    task_type: str
    ours: str
    dataset: str
    n: int
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lawbench_id": self.lawbench_id,
            "task_type": self.task_type,
            "ours": self.ours,
            "dataset": self.dataset,
            "n": self.n,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in sorted(self.metrics.items())},
            "notes": self.notes,
            "seconds": round(self.seconds, 2),
        }


def _safe(label: str, func: Callable[[], TaskResult]) -> TaskResult | None:
    """单个任务失败不影响整张表——评测台最忌讳"一处崩全表空"。"""
    try:
        return func()
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {label}: {type(exc).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# 各任务的取数与计算
# --------------------------------------------------------------------------- #
def task_extraction(config: Config) -> TaskResult:
    """2-6 Extraction：要素抽取，复用训练时的留出集划分与 gold 口径。"""
    from . import bert_ner, train

    docs = train.collect_docs()
    _, valid = train.split_docs(docs, ratio=0.25, seed=config.seed)
    model, tokenizer, meta = bert_ner.load_model()[:3]
    started = time.perf_counter()
    stats = bert_ner.evaluate(model, tokenizer, valid)
    micro = stats.get("micro") or {}
    result = TaskResult(
        lawbench_id="2-6 Extraction",
        task_type="extraction",
        ours="节点① 要素抽取（BERT-CRF）",
        dataset=f"自建语料留出集 {len(valid)} 份（金标=公告字段）",
        n=int(micro.get("tp", 0) + micro.get("fn", 0)),
        metrics={
            "micro_p": float(micro.get("precision", 0.0)),
            "micro_r": float(micro.get("recall", 0.0)),
            "micro_f1": float(micro.get("f1", 0.0)),
            "macro_f1": float(stats.get("macro_f1", 0.0)),
        },
        notes=f"训练时监督={meta.get('supervision', 'gold+silver')}；"
              f"补全金标口径见 reports/gpu_upgrade_bert_ner.json",
        seconds=time.perf_counter() - started,
    )
    complete = stats.get("micro_complete") or {}
    if complete:
        result.metrics["micro_f1_complete_gold"] = float(complete.get("f1", 0.0))
    return result


def task_detection(config: Config) -> TaskResult:
    """2-3 MLC：多标签风险检出，金标=对抗集 manifest 的 issue_codes。"""
    from . import bench

    detection = bench.bench_detection(config)
    return TaskResult(
        lawbench_id="2-3 MLC (20 类)",
        task_type="multi-label",
        ours="27 风险码检出",
        dataset="合成对抗集 10 份（模板注入，金标只标主缺陷）",
        n=int(detection.get("contracts", 0) or 0),
        metrics={
            "precision": float(detection.get("precision", 0.0)),
            "recall": float(detection.get("recall", 0.0)),
            "f1": float(detection.get("f1", 0.0)),
            "tp": float(detection.get("tp", 0)),
            "fp": float(detection.get("fp", 0)),
            "fn": float(detection.get("fn", 0)),
            "contract_hit": float(detection.get("contract_hit", 0)),
        },
        notes="FP 含【检出但金标未标】与【模板共享句】两类，不能直接当误报率",
    )


def task_retrieval(config: Config) -> TaskResult:
    """3-1 MLC（排序型）：法条推荐，v1 兼容协议。"""
    from . import bench

    curves = bench.bench_retrieval(config)
    row = (curves.get("curves", {}).get("rrf_plus_ltr", {}) or {}).get("strong", {}) or {}
    recall = row.get("recall") or {}
    return TaskResult(
        lawbench_id="3-1 MLC (118 类)",
        task_type="multi-label ranking",
        ours="节点④ 法条推荐（三路召回+加权RRF+LTR）",
        dataset=f"强金标准 {row.get('n', 0)} 条查询（按标签族定义）",
        n=int(row.get("n", 0) or 0),
        metrics={
            "recall@1": float(recall.get("@1", 0.0)),
            "recall@3": float(recall.get("@3", 0.0)),
            "recall@5": float(recall.get("@5", 0.0)),
            "recall@10": float(recall.get("@10", 0.0)),
            "mrr@5": float(row.get("mrr5", 0.0)),
        },
        notes="金标准规模 2~6 条，Recall@1 理论上限≈0.37，横向比只看 @5/@10",
    )


def task_entailment(config: Config) -> TaskResult:
    """2-4 SLC：条款—条文一致性三分类（需已训练 bert_entail）。"""
    from . import bert_entail

    model, tokenizer, meta = bert_entail.load_model()
    from . import train

    docs = train.collect_docs()
    pairs = bert_entail.build_pairs(docs, config, per_clause=2, limit_clauses=120)
    if not pairs:
        raise RuntimeError("没有构造出三元组（检索或条款抽取为空）")
    started = time.perf_counter()
    stats = bert_entail.evaluate(model, tokenizer, pairs)
    return TaskResult(
        lawbench_id="2-4 SLC (20 类)",
        task_type="single-label",
        ours="节点④ 条款—条文一致性判别（双塔+cross-attention）",
        dataset=f"弱监督三元组 {len(pairs)} 条（标签由 8 维规则蒸馏）",
        n=len(pairs),
        metrics={
            "accuracy": float(stats.get("accuracy", 0.0)),
            "macro_f1": float(stats.get("macro_f1", 0.0)),
        },
        notes="监督信号来自规则蒸馏，天花板受规则质量限制；默认后端仍用 rules",
        seconds=time.perf_counter() - started,
    )


def task_subject(config: Config) -> TaskResult:
    """1-2 SLC：主体资格审查（需资质 / 不需资质），在对抗集上按 need_license 计。"""
    from . import bench

    detection = bench.bench_detection(config)
    per_code = detection.get("per_code") or {}
    bucket = per_code.get("need_license") or {}
    tp = int(bucket.get("tp", 0))
    fp = int(bucket.get("fp", 0))
    fn = int(bucket.get("fn", 0))
    if tp + fn == 0:
        # 该批样本的金标里没有 need_license，单码代理指标无从计算，如实报空表而不是报 0
        return TaskResult(
            lawbench_id="1-2 SLC (4 类)",
            task_type="single-label",
            ours="节点③ 主体资格审查（资质知识图谱 + RAG）",
            dataset="合成对抗集 10 份",
            n=0,
            metrics={},
            notes=f"金标无 need_license 标注（检出 {fp} 条，均无金标）；该任务需人工标注子集",
        )
    total = max(tp + fp + fn, 1)
    return TaskResult(
        lawbench_id="1-2 SLC (4 类)",
        task_type="single-label",
        ours="节点③ 主体资格审查（资质知识图谱 + RAG）",
        dataset="合成对抗集 10 份（按 need_license 单码计）",
        n=total,
        metrics={
            "accuracy": (tp + max(total - tp - fp - fn, 0)) / total,
            "tp": float(tp), "fp": float(fp), "fn": float(fn),
        },
        notes="只有单码样本，属代理指标；四分类未做（缺标注）",
    )


def task_generation(config: Config) -> TaskResult:
    """generation：节点⑤ 复核意见，只看生成可用性（无参考标准，不编指标）。"""
    from . import local_llm, nodes

    llm = nodes.llm_settings(config)
    usable = bool(llm.endpoint and llm.model)
    return TaskResult(
        lawbench_id="generation",
        task_type="generation",
        ours="节点⑤ 复核意见（模型只读，不改分数）",
        dataset="单份技术开发合同（冒烟）",
        n=1 if usable else 0,
        metrics={"endpoint_configured": 1.0 if usable else 0.0,
                 "local_model_available": 1.0 if local_llm.find_model_dir() else 0.0},
        notes="生成任务没有参考答案，按 LawBench 惯例不报自动指标，只报可用性",
    )


TASKS: dict[str, Callable[[Config], TaskResult]] = {
    "extraction": task_extraction,
    "detection": task_detection,
    "retrieval": task_retrieval,
    "entailment": task_entailment,
    "subject": task_subject,
    "generation": task_generation,
}


def run(config: Config | None = None, *, tasks: tuple[str, ...] | None = None,
        save: bool = True) -> dict[str, Any]:
    config = config or C.config_from_env()
    names = tasks or tuple(TASKS)
    started = time.perf_counter()
    results: list[TaskResult] = []
    print(f"LawBench 式评测：{len(names)} 项任务")
    for name in names:
        func = TASKS.get(name)
        if func is None:
            print(f"  [skip] 未知任务 {name}")
            continue
        print(f"- {name}")
        result = _safe(name, lambda f=func: f(config))
        if result is not None:
            results.append(result)
            top = ", ".join(f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:4])
            print(f"    {result.lawbench_id:<20} n={result.n:<5} {top}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - started, 1),
        "task_table": list(TASK_TABLE),
        "results": [item.as_dict() for item in results],
        "caveat": (
            "本表是【按 LawBench 口径组织自家评测】，不是 LawBench 官方分数："
            "数据来自自建 113 份语料 / 对抗集 / 真实政采合同，未使用 LawBench 的划分与标注。"
            "生成类任务没有参考答案，按论文惯例不报自动指标。"
        ),
    }
    if save:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写 {REPORT}")
    return payload


def format_table(payload: dict[str, Any]) -> str:
    """把结果排成可读表（CLI / 文档直接用）。"""
    lines = ["LawBench 式评测结果", "-" * 96]
    for item in payload.get("results", []):
        metrics = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in item["metrics"].items())
        lines.append(f"[{item['lawbench_id']}] {item['ours']}")
        lines.append(f"    n={item['n']}  {metrics}")
        if item.get("dataset"):
            lines.append(f"    数据：{item['dataset']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LawBench 式系统化评测")
    parser.add_argument("--tasks", default="", help="逗号分隔，默认全部")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    names = tuple(n.strip() for n in args.tasks.split(",") if n.strip()) or None
    payload = run(tasks=names, save=not args.no_save)
    print()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_table(payload))
        print()
        print(payload["caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
