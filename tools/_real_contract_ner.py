"""在真实政府采购合同上评测 NER（用 manifest 的采购人/供应商/金额/签订日做金标）。

与合成集评测的区别：金标来自公告字段而非模板生成，正文是真实排版（含"（本级）"、
大写金额、多段付款、附件附表），因此可信度更高，但只有 3 类字段（PARTY/AMOUNT/DATE）。

用法：
    python tools/_real_contract_ner.py                  # 只用有文字层的
    python tools/_real_contract_ner.py --ocr 3           # 额外对 3 份扫描件跑 OCR
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import bert_ner, ner, text as T  # noqa: E402

MANIFEST = ROOT / "data" / "gov_contracts" / "manifest.jsonl"


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def contract_text(record: dict) -> str:
    """拼 text/ 层的页文本（扫描件为空串，需要先 OCR）。"""
    folder = ROOT / "data" / "gov_contracts" / record["contract_id"] / "text"
    if not folder.exists():
        return ""
    parts = []
    for path in sorted(folder.glob("page_*.txt")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def amount_text(raw) -> str:
    """把 manifest 的 "146.5000000 万元" 变成正文里可能出现的写法。"""
    if raw is None:
        return ""
    match = re.search(r"([\d.]+)\s*(万元|亿元|元)", str(raw))
    if not match:
        return str(raw).strip()
    value, unit = float(match.group(1)), match.group(2)
    if unit == "万元":
        value_yuan = value * 10_000
    elif unit == "亿元":
        value_yuan = value * 100_000_000
    else:
        value_yuan = value
    candidates = []
    if unit != "元":
        candidates.append(f"{match.group(1)}{unit}")
        stripped = match.group(1).rstrip("0").rstrip(".")
        if stripped and stripped != match.group(1):
            candidates.append(f"{stripped}{unit}")
    whole = f"{value_yuan:.2f}".rstrip("0").rstrip(".")
    candidates.append(f"{whole}元")
    candidates.append(f"{value_yuan:,.2f}元")
    candidates.append(f"{int(value_yuan)}元" if value_yuan == int(value_yuan) else whole)
    return candidates


def party_variants(name: str) -> list[str]:
    """公告里的采购人常带"（本级）""本级事业"等行政后缀，正文只写正名。"""
    base = str(name).strip()
    variants = [base]
    stripped = re.sub(r"[（(][^）)]*[）)]", "", base).strip()
    variants.append(stripped)
    for suffix in ("本级事业", "本级", "（本级）", "事业"):
        if stripped.endswith(suffix):
            variants.append(stripped[: -len(suffix)].strip())
    return [item for item in dict.fromkeys(variants) if len(item) >= 4]


def text_quality(full: str) -> float:
    """文字层的中文占比。PDF 里脏文字层（如整页乱码）会把评测变成噪声，需要先筛掉。"""
    if not full:
        return 0.0
    good = sum(1 for ch in full if "\u4e00" <= ch <= "\u9fff" or ch.isdigit() or ch in "，。；：、（）《》%％.,")
    return good / len(full)


def build_gold(record: dict, full: str) -> set[tuple[int, int, str]]:
    """用公告字段构造金标。真实合同排版不规整（`￥1465000.00 元`、`【2026】年【10】月【29 】日`），
    所以一律用"数字核心 + 宽松间隔"的正则去定位，而不是精确子串。"""
    gold: set[tuple[int, int, str]] = set()

    def put(spans) -> None:
        for start, end, kind in spans:
            gold.add((start, end, kind))

    parties = [p for p in (record.get("purchaser"), record.get("supplier")) if p]
    for name in parties:
        for candidate in party_variants(name):
            start = full.find(candidate)
            if start >= 0:
                put([(start, start + len(candidate), "PARTY")])
                break

    raw_amount = record.get("amount")
    if raw_amount:
        match = re.search(r"([\d.]+)\s*(万元|亿元|元)", str(raw_amount))
        if match:
            value, unit = float(match.group(1)), match.group(2)
            yuan = value * 10_000 if unit == "万元" else value * 100_000_000 if unit == "亿元" else value
            cores = []
            if unit != "元":
                cores.append(match.group(1).rstrip("0").rstrip("."))
            cores.append(f"{yuan:.2f}")
            cores.append(str(int(yuan)) if yuan == int(yuan) else f"{yuan:.2f}")
            for core in dict.fromkeys(c for c in cores if c):
                hit = re.search(re.escape(core) + r"[\d,.\s]{0,10}(?:万|亿)?\s*元", full)
                if hit:
                    put([(hit.start(), hit.end(), "AMOUNT")])
                    break

    sign = record.get("sign_date")
    if sign:
        match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(sign))
        if match:
            year, month, day = (int(match.group(i)) for i in (1, 2, 3))
            pattern = (
                rf"{year}[^\d]{{0,3}}年[^\d]{{0,3}}0?{month}[^\d]{{0,3}}月[^\d]{{0,3}}0?{day}[^\d]{{0,3}}日"
            )
            hit = re.search(pattern, full)
            if hit:
                put([(hit.start(), hit.end(), "DATE")])

    return gold


def rule_predict(full: str) -> set[tuple[int, int, str]]:
    """规则通道在同一段文本上的预测（走生产链路）。"""
    from zhilv_qiling_v2.models import DocumentParse, SourceBlock

    block = SourceBlock(block_id="b", order=0, kind="paragraph", text=full)
    document = DocumentParse(source_name="<real>", file_type="txt", full_text=full, blocks=[block])
    kind_map = {"party": "PARTY", "amount": "AMOUNT", "date": "DATE", "subject": "SUBJECT"}
    out: set[tuple[int, int, str]] = set()
    for hit in ner.rule_elements(document):
        kind = kind_map.get(hit.kind.lower())
        if kind and hit.span is not None:
            out.add((hit.span.char_start, hit.span.char_end, kind))
    return out


def score(gold: set, pred: set) -> dict:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    want_ocr = 0
    if "--ocr" in argv:
        want_ocr = int(argv[argv.index("--ocr") + 1])

    records = load_manifest()
    usable = [r for r in records if r["scanned_pages"] == 0]
    print(f"清单 {len(records)} 份，其中有文字层可直接测的 {len(usable)} 份")

    if want_ocr:
        scanned = [r for r in records if r["scanned_pages"] == r["page_count"]][:want_ocr]
        print(f"另取 {len(scanned)} 份扫描件跑 OCR（前 6 页/份，控制时间）")
        from zhilv_qiling_v2 import ocr

        engine = ocr.get_engine()
        for record in scanned:
            folder = ROOT / "data" / "gov_contracts" / record["contract_id"]
            pages = sorted(folder.glob("pages/page_*.jpeg"))[:6]
            chunks = []
            for page in pages:
                try:
                    result = engine.recognize_image(page)
                    chunks.append(getattr(result, "text", "") or "")
                except Exception as exc:  # noqa: BLE001
                    print("   OCR 失败", page.name, exc)
            text = "\n".join(chunks)
            (folder / "_ocr_sample.txt").write_text(text, encoding="utf-8")
            print(f"  {record['contract_id']} OCR {len(pages)} 页 / {len(text)} 字")
        usable = usable + scanned

    model = tok = None
    try:
        model, tok = bert_ner.load_model()[:2]
    except Exception as exc:  # noqa: BLE001
        print("BERT 不可用：", exc)

    totals = {"bert": {"tp": 0, "fp": 0, "fn": 0}, "rule": {"tp": 0, "fp": 0, "fn": 0}}
    rows = []
    for record in usable:
        full = contract_text(record)
        ocr_file = ROOT / "data" / "gov_contracts" / record["contract_id"] / "_ocr_sample.txt"
        if not full.strip() and ocr_file.exists():
            full = ocr_file.read_text(encoding="utf-8")
        if len(full) < 200:
            continue
        quality = text_quality(full)
        if quality < 0.82:
            print(f"  {record['contract_id']} 文字层质量 {quality:.2f} 过低（脏文字层），跳过")
            continue
        gold = build_gold(record, full)
        if not gold:
            print(f"  {record['contract_id']} 金标 0 条，跳过")
            continue
        rule_pred = rule_predict(full)
        bert_pred = (
            {(s, e, k) for s, e, k, _ in bert_ner.predict(model, tok, full)} if model else set()
        )
        row = {
            "contract_id": record["contract_id"],
            "chars": len(full),
            "gold": len(gold),
            "bert": score(gold, bert_pred),
            "rule": score(gold, rule_pred),
            "gold_detail": sorted({kind for _, _, kind in gold}),
        }
        rows.append(row)
        for name, stat in (("bert", row["bert"]), ("rule", row["rule"])):
            for key in ("tp", "fp", "fn"):
                totals[name][key] += stat[key]
        print(
            f"  {record['contract_id']} {len(full):6}字 金标{len(gold):2} | "
            f"BERT P{row['bert']['precision']:.2f} R{row['bert']['recall']:.2f} F{row['bert']['f1']:.2f} | "
            f"规则 P{row['rule']['precision']:.2f} R{row['rule']['recall']:.2f} F{row['rule']['f1']:.2f}"
        )

    print("\n=== 汇总（真实政采合同）===")
    for name in ("bert", "rule"):
        stat = totals[name]
        tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        print(f"  {name:5} P={precision:.4f} R={recall:.4f} F1={f1:.4f}  tp{tp} fp{fp} fn{fn}")

    out = ROOT / "zhilv_qiling_v2" / "reports" / "real_contract_ner.json"
    out.write_text(json.dumps({"contracts": rows, "totals": totals}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
