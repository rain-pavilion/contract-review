import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOV = ROOT / "data" / "gov_contracts"

rows = []
for line in (GOV / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        rows.append(json.loads(line))
print("contracts in manifest:", len(rows))
print("sources:", Counter(r.get("source") for r in rows))
pages = sum(r.get("page_count", 0) for r in rows)
scanned = sum(r.get("scanned_pages", 0) for r in rows)
print("pages total:", pages, "scanned pages:", scanned, "=%.1f%%" % (100 * scanned / max(1, pages)))
full_scan = [r for r in rows if r.get("page_count") and r.get("scanned_pages") == r.get("page_count")]
print("fully scanned contracts:", len(full_scan))
print("avg pages: %.1f" % (pages / max(1, len(rows))))
print("with amount:", sum(1 for r in rows if r.get("amount")))
print("with sign_date:", sum(1 for r in rows if r.get("sign_date")))

d = sorted(GOV.glob("*/meta.json"))
print("dirs:", len(d))
imgs = list(GOV.glob("*/pages/*.jpeg")) + list(GOV.glob("*/pages/*.png"))
txts = list(GOV.glob("*/text/*.txt"))
print("page images:", len(imgs), "text pages:", len(txts))
print("image size sample:", [p.stat().st_size for p in imgs[:3]])
print("text len sample:", [len(p.read_text(encoding='utf-8', errors='replace')) for p in txts[:5]])

# 生成数据集
gen = ROOT / "data" / "generated_contracts"
for sub in sorted(gen.iterdir()):
    if sub.is_dir():
        print(sub.name, len(list(sub.rglob("*.docx"))), "docx,", len(list(sub.rglob("*.pdf"))), "pdf")
man = list(gen.rglob("manifest.json"))
print("manifests:", [str(p.relative_to(gen)) for p in man])
