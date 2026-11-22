"""真实政府采购合同上的要素抽取测试（字段级口径 + 端到端稳定性）。

为什么不用 span 级 F1：公告只给了 4 个字段（采购人/供应商/合同金额/签订日期），正文里
却有几十个金额（单价、税率、分项、大写）、十几个日期、多个机构名。用"清单字段"当金标算
精确率，会把正确抽取全记成误报，指标没有意义。所以这里报**字段级召回**——"这 4 个字段抽到
没有"，并配端到端耗时、风险条数、降级标记，回答"在真实数据上跑得动、抽得到吗"。

用法：
    python tools/_real_contract_eval.py                 # 有文字层的合同
    python tools/_real_contract_eval.py --ocr            # 含已 OCR 的扫描件
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import bert_ner, ner, pipeline, text as T  # noqa: E402

MANIFEST = ROOT / "data" / "gov_contracts" / "manifest.jsonl"


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def contract_text(folder: pathlib.Path) -> str:
    text_dir = folder / "text"
    if not text_dir.exists():
        return ""
    parts = []
    for path in sorted(text_dir.glob("page_*.txt")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def text_quality(full: str) -> float:
    if not full:
        return 0.0
    good = sum(
        1 for ch in full if "\u4e00" <= ch <= "\u9fff" or ch.isdigit() or ch in "，。；：、（）《》%％.,"
    )
    return good / len(full)


def party_variants(name: str) -> list[str]:
    base = str(name).strip()
    stripped = re.sub(r"[（(][^）)]*[）)]", "", base).strip()
    variants = [base, stripped]
    for suffix in ("本级事业", "本级", "事业"):
        if stripped.endswith(suffix):
            variants.append(stripped[: -len(suffix)].strip())
    return [item for item in dict.fromkeys(variants) if len(item) >= 4]


def target_amount(raw) -> float | None:
    match = re.search(r"([\d.]+)\s*(万元|亿元|元)", str(raw or ""))
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    return value * 10_000 if unit == "万元" else value * 100_000_000 if unit == "亿元" else value


def target_date(raw) -> str | None:
    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(raw or ""))
    if not match:
        return None
    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    return f"{year:04d}-{month:02d}-{day:02d}"


def elements_of(full: str) -> tuple[set[str], list[float], set[str]]:
    """跑生产链路的要素抽取，返回（当事人文本集合, 金额数值列表, 日期 ISO 集合）。"""
    from zhilv_qiling_v2.models import DocumentParse, SourceBlock

    block = SourceBlock(block_id="b", order=0, kind="paragraph", text=full)
    document = DocumentParse(source_name="<real>", file_type="txt", full_text=full, blocks=[block])
    parties: set[str] = set()
    amounts: list[float] = []
    dates: set[str] = set()
    for hit in ner.rule_elements(document):
        kind = hit.kind.lower()
        if kind == "party":
            parties.add(str(hit.value))
        elif kind == "amount" and isinstance(hit.extra.get("numeric"), (int, float)):
            amounts.append(float(hit.extra["numeric"]))
        elif kind == "date" and hit.normalized:
            dates.add(str(hit.normalized)[:10])
    for start, end, kind, _ in bert_ner.predict(*_bert(), full) if _bert()[0] else []:
        fragment = full[start:end]
        if kind == "PARTY":
            parties.add(fragment)
        elif kind == "AMOUNT":
            value = T.money_to_float(fragment)
            if value:
                amounts.append(float(value))
        elif kind == "DATE":
            iso = T.parse_date_text(fragment)
            if iso:
                dates.add(str(iso)[:10])
    return parties, amounts, dates


_BERT_CACHE: list = []


def _bert():
    if not _BERT_CACHE:
        try:
            model, tok = bert_ner.load_model()[:2]
        except Exception:
            model, tok = None, None
        _BERT_CACHE.extend([model, tok])
    return _BERT_CACHE[0], _BERT_CACHE[1]


def field_hits(record: dict, full: str, parties: set[str], amounts: list[float], dates: set[str]) -> dict[str, str]:
    """四个字段各自的结果：hit（抽到）/ miss（正文本有但没抽到）/ absent（正文本就没有）。

    真实政采合同的签订日期常只出现在公告字段里，正文签章页是图片，OCR 也拿不到，
    把这类算成漏报会冤枉模型。所以先判断"正文本里有没有这个值"，没有就记 absent。
    """
    hits: dict[str, str] = {}
    for label, name in (("purchaser", record.get("purchaser")), ("supplier", record.get("supplier"))):
        if not name:
            continue
        variants = party_variants(name)
        present = any(variant in full for variant in variants)
        if not present:
            hits[label] = "absent"
            continue
        found = any(
            variant in extracted or extracted in variant
            for variant in variants
            for extracted in parties
            if len(extracted) >= 4
        )
        hits[label] = "hit" if found else "miss"

    amount = target_amount(record.get("amount"))
    if amount:
        core = f"{int(amount)}" if amount == int(amount) else f"{amount:.2f}"
        present = bool(re.search(re.escape(core[:6]), full)) or f"{amount / 10_000:.3f}".rstrip("0") in full
        if not present:
            hits["amount"] = "absent"
        else:
            found = any(abs(value - amount) <= max(1.0, amount * 0.01) for value in amounts)
            hits["amount"] = "hit" if found else "miss"

    date = target_date(record.get("sign_date"))
    if date:
        year, month, day = date.split("-")
        pattern = rf"{int(year)}[^\d]{{0,3}}年[^\d]{{0,3}}0?{int(month)}[^\d]{{0,3}}月[^\d]{{0,3}}0?{int(day)}[^\d]{{0,3}}日"
        present = bool(re.search(pattern, full)) or date in full
        if not present:
            hits["sign_date"] = "absent"
        else:
            hits["sign_date"] = "hit" if date in dates else "miss"
    return hits


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    use_ocr = "--ocr" in argv
    records = load_manifest()
    root = ROOT / "data" / "gov_contracts"

    rows = []
    field_total = field_hit = 0
    perf: list[float] = []
    risk_hist: dict[str, int] = {}
    degraded_any: dict[str, int] = {}
    skipped_quality = 0

    for record in records:
        folder = root / record["contract_id"]
        full = contract_text(folder)
        if not use_ocr and record["scanned_pages"]:
            continue
        if len(full) < 400:
            continue
        quality = text_quality(full)
        if quality < 0.82:
            skipped_quality += 1
            continue

        parties, amounts, dates = elements_of(full)
        hits = field_hits(record, full, parties, amounts, dates)
        applicable = [v for v in hits.values() if v != "absent"]
        field_total += len(applicable)
        field_hit += sum(1 for v in applicable if v == "hit")

        started = time.time()
        try:
            report = pipeline.review_bytes(f"{record['contract_id']}.txt", full.encode("utf-8"))
            elapsed = time.time() - started
            perf.append(elapsed)
            for risk in report.risks:
                risk_hist[risk.code] = risk_hist.get(risk.code, 0) + 1
            for flag in report.stats.get("degraded", []) or []:
                degraded_any[flag] = degraded_any.get(flag, 0) + 1
            score = round(report.score)
            risks = len(report.risks)
        except Exception as exc:  # noqa: BLE001
            elapsed = -1.0
            score = risks = -1
            print("  端到端失败", record["contract_id"], exc)

        rows.append(
            {
                "contract_id": record["contract_id"],
                "chars": len(full),
                "quality": round(quality, 3),
                "fields": hits,
                "score": score,
                "risks": risks,
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"  {record['contract_id']} {len(full):6}字 q{quality:.2f} "
            f"字段 {sum(1 for v in hits.values() if v == 'hit')}/{len(applicable)}"
            f"（absent {sum(1 for v in hits.values() if v == 'absent')}） "
            f"评分 {score} 风险 {risks} 耗时 {elapsed:.2f}s"
        )

    print("\n=== 汇总 ===")
    if field_total:
        print(f"  字段级召回 {field_hit}/{field_total} = {field_hit / field_total:.4f}")
    if perf:
        ordered = sorted(perf)
        print(
            f"  端到端 {len(perf)} 份：P50 {ordered[len(ordered) // 2]:.2f}s / "
            f"最长 {ordered[-1]:.2f}s / 均 {sum(ordered) / len(ordered):.2f}s"
        )
    print("  文字层质量过低跳过", skipped_quality, "份")
    if risk_hist:
        top = sorted(risk_hist.items(), key=lambda kv: -kv[1])[:8]
        print("  高频风险码", top)
    if degraded_any:
        print("  降级标记", degraded_any)

    out = ROOT / "zhilv_qiling_v2" / "reports" / "real_contract_eval.json"
    out.write_text(
        json.dumps(
            {
                "protocol": "字段级召回（采购人/供应商/合同金额/签订日期是否抽到）+ 端到端耗时与风险分布",
                "contracts": rows,
                "field_recall": round(field_hit / field_total, 4) if field_total else None,
                "risk_hist": risk_hist,
                "degraded": degraded_any,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("已写", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
