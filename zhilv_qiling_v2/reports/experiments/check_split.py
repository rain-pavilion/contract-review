import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhilv_qiling_v2 import parsers, layout, splitter, config as C
from zhilv_qiling_v2.learned import registry

ROOT = Path(__file__).resolve().parent.parent
cfg = C.config_from_env()
reg = registry()

targets = [ROOT / "samples" / "tech-development-contract.txt"]
gen = ROOT / "data" / "generated_contracts" / "samr_multi_template_2025"
if gen.exists():
    docx = sorted(gen.glob("*.docx"))[:2]
    targets += docx
gov = ROOT / "data" / "gov_contracts"
if gov.exists():
    metas = sorted(gov.glob("*/meta.json"))[:1]
    for m in metas:
        d = m.parent
        pdfs = sorted(d.glob("*.pdf"))
        targets += pdfs[:1]

for path in targets:
    if not path.exists():
        print("MISS", path)
        continue
    t0 = time.time()
    out = parsers.parse_path(path)
    blocks, lstats = layout.analyze_layout(out.blocks, out.page_sizes)
    doc = splitter.split_document(blocks, cfg, reg.boundary(), out.page_sizes, path.name)
    dt = time.time() - t0
    print("=" * 70)
    print(path.name, "|", out.file_type, "| blocks", len(out.blocks), "->", len(blocks), "| clauses", len(doc.clauses), "| %.2fs" % dt)
    print("  layout:", {k: v for k, v in lstats.items() if k != "removed_samples"})
    print("  parse :", {k: v for k, v in out.stats.items() if k not in ("styles",)})
    print("  seg   :", doc.stats["segmentation"])
    print("  issues:", [(i.code, i.message) for i in doc.issues])
    for c in doc.clauses[:8]:
        print("   [L%d] %-12s %-24s | %s" % (c.level, c.label or "-", c.title[:22], c.excerpt(46)))
    print("   ...")
    for c in doc.clauses[-3:]:
        print("   [L%d] %-12s %-24s | %s" % (c.level, c.label or "-", c.title[:22], c.excerpt(46)))
