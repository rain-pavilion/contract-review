"""端到端管线：文件/字节 → 解析 → 版面 → 切分 → 要素 → 引擎 → 报告。

这一层是 v1 `contract_review/service.py` + `reviewer.py` 的对应物，但做了三处收紧：

- **每一步都能单独跑**：`prepare()` 只做解析+版面+切分，`extract()` 只做要素，
  `review()` 才跑智能体；故障定位不需要复现整条链。
- **降级显式化**：任何一步缺失（没有 OCR、没有 sklearn、没有法条索引）都写进
  `report.stats["degraded"]`，报告里能直接引用。
- **批量入口**：`batch_review()` 返回逐份报告与汇总，供论文第 6 章的批量实验直接用。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import text as T
from .config import Config, V1_GENERATED, config_from_env
from .engine import ReviewEngine
from .learned import registry
from .models import Contract, ContractElements, DocumentParse, ReviewReport
from .ner import extract_elements
from .parsers import ParseError, parse_document, parse_path
from .splitter import split_document
from .review_context import build_review_context

CONTRACT_TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("房屋租赁", ("房屋租赁", "租赁合同", "出租人", "承租人", "租金")),
    ("商品房预售", ("商品房预售", "预售合同", "出卖人", "买受人", "建筑面积")),
    ("房屋买卖", ("房屋买卖", "二手房", "买卖合同")),
    ("委托", ("委托合同", "委托事项", "受托人")),
    ("中介", ("中介合同", "居间", "中介人", "促成")),
    ("保管", ("保管合同", "寄存人", "保管人", "保管物")),
    ("数据提供", ("数据提供", "数据处理", "数据服务")),
    ("技术开发", ("技术开发", "研究开发", "技术成果")),
    ("政府采购", ("政府采购", "采购人", "供应商", "中标")),
    ("承揽", ("承揽", "加工", "定作")),
)


def infer_contract_type(document: DocumentParse) -> str:
    """按标题与开头 600 字的角色词判类型（比 v1 的"从文件名猜"稳）。"""
    head = (document.source_name + "\n" + document.full_text[:600])
    best, best_score = "通用", 0
    for name, keywords in CONTRACT_TYPE_HINTS:
        score = sum(1 for keyword in keywords if keyword in head)
        if score > best_score:
            best, best_score = name, score
    return best


def prepare(
    source_name: str,
    content: bytes | None = None,
    *,
    path: Path | str | None = None,
    config: Config | None = None,
) -> DocumentParse:
    """解析 + 版面 + 切分。"""
    config = config or config_from_env()
    started = time.time()
    if path is not None:
        parsed = parse_path(path)
    elif content is not None:
        parsed = parse_document(source_name, content)
    else:
        raise ValueError("prepare() 需要 content 或 path 之一")
    # 版面分析（可选：解析层已经给了 bbox 才有效）
    try:
        from .layout import analyze_layout

        parsed.blocks, layout_stats = analyze_layout(parsed.blocks, parsed.page_sizes)
        parsed.stats["layout"] = layout_stats
    except Exception as exc:  # 版面失败不影响切分
        parsed.stats["layout"] = {"error": str(exc)[:80]}
    document = split_document(
        parsed.blocks,
        config,
        classifier=registry().boundary(),
        page_sizes=parsed.page_sizes,
        source_name=source_name,
    )
    document.stats["needs_ocr_pages"] = list(parsed.needs_ocr_pages)
    document.stats["parse"] = dict(parsed.stats)
    document.stats["prepare_seconds"] = round(time.time() - started, 3)
    return document


def extract(
    document: DocumentParse,
    config: Config | None = None,
) -> tuple[ContractElements, list, dict[str, Any]]:
    config = config or config_from_env()
    elements, signals, stats = extract_elements(document, config, registry().ner())
    return elements, signals, stats


def build_contract(
    document: DocumentParse,
    config: Config | None = None,
) -> Contract:
    config = config or config_from_env()
    document = replace(document, stats=dict(document.stats))
    elements, signals, ner_stats = extract(document, config)
    detected_type = infer_contract_type(document)
    selected_type = config.review_contract_type
    contract_type = detected_type if selected_type == "auto" else selected_type
    review_context = build_review_context(contract_type, config, detected_type)
    document.stats["ner"] = ner_stats
    document.stats["contract_type"] = contract_type
    document.stats["detected_contract_type"] = detected_type
    document.stats["review_context"] = dict(review_context)
    # Version the identity so historical filename/length IDs cannot authorize a
    # revised contract. Exact parsed content is bound without normalization.
    identity = json.dumps([document.source_name, document.full_text], ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
    content_id = "ct_v2_" + hashlib.sha256(identity).hexdigest()
    document.stats["contract_identity"] = {
        "schema": "source-name-and-full-text-sha256-v2",
        "text_sha256": hashlib.sha256(document.full_text.encode("utf-8")).hexdigest(),
    }
    return Contract(
        contract_id=content_id,
        document=document,
        elements=elements,
        signals=signals,
        contract_type=contract_type,
        review_context=review_context,
    )


def review_document(
    document: DocumentParse,
    config: Config | None = None,
    *,
    engine: ReviewEngine | None = None,
) -> ReviewReport:
    config = config or config_from_env()
    contract = build_contract(document, config)
    engine = engine or ReviewEngine(config)
    return engine.run(contract)


def review_bytes(
    source_name: str,
    content: bytes,
    config: Config | None = None,
    *,
    engine: ReviewEngine | None = None,
) -> ReviewReport:
    document = prepare(source_name, content, config=config)
    return review_document(document, config, engine=engine)


def review_path(
    path: Path | str,
    config: Config | None = None,
    *,
    engine: ReviewEngine | None = None,
) -> ReviewReport:
    path = Path(path)
    document = prepare(path.name, path=path, config=config)
    return review_document(document, config, engine=engine)


@dataclass
class BatchSummary:
    reports: list[ReviewReport] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    elapsed: float = 0.0

    def aggregate(self) -> dict[str, Any]:
        total = len(self.reports)
        if not total:
            return {"contracts": 0, "failed": len(self.failed)}
        scores = [r.score for r in self.reports]
        risks = [len(r.risks) for r in self.reports]
        codes: dict[str, int] = {}
        for report in self.reports:
            for risk in report.risks:
                codes[risk.code] = codes.get(risk.code, 0) + 1
        return {
            "contracts": total,
            "failed": len(self.failed),
            "score_mean": round(sum(scores) / total, 2),
            "score_min": min(scores),
            "score_max": max(scores),
            "risks_mean": round(sum(risks) / total, 2),
            "risks_total": sum(risks),
            "codes": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
            "elapsed_seconds": round(self.elapsed, 2),
            "avg_seconds": round(self.elapsed / total, 3),
        }


def batch_review(
    paths: Iterable[Path | str],
    config: Config | None = None,
    *,
    on_report: Any = None,
) -> BatchSummary:
    config = config or config_from_env()
    engine = ReviewEngine(config)
    summary = BatchSummary()
    started = time.time()
    for path in paths:
        try:
            report = review_path(path, config, engine=engine)
        except (ParseError, OSError, ValueError) as exc:
            summary.failed.append({"path": str(path), "error": str(exc)[:120]})
            continue
        summary.reports.append(report)
        if on_report is not None:
            on_report(report)
    summary.elapsed = time.time() - started
    return summary


def dataset_files(root: Path | str | None = None, suffixes: Sequence[str] = (".docx", ".pdf", ".txt")) -> list[Path]:
    """枚举数据集里的合同文件（跳过 manifest/说明）。"""
    root = Path(root or V1_GENERATED)
    if not root.exists():
        return []
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if path.name.lower().startswith(("readme", "manifest")):
            continue
        out.append(path)
    return out
