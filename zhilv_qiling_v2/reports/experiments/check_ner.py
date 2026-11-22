import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhilv_qiling_v2 import parsers, layout, splitter, ner, config as C
from zhilv_qiling_v2.learned import registry

ROOT = Path(__file__).resolve().parent.parent
cfg = C.config_from_env()
reg = registry()

targets = [ROOT / "samples" / "tech-development-contract.txt"]
gen = ROOT / "data" / "generated_contracts" / "samr_multi_template_2025"
targets += sorted(gen.glob("*.docx"))[:2]
adv = gen / "adversarial"
if adv.exists():
    targets += sorted(adv.glob("*.docx"))[:2]

for path in targets:
    out = parsers.parse_path(path)
    blocks, _ = layout.analyze_layout(out.blocks, out.page_sizes)
    doc = splitter.split_document(blocks, cfg, reg.boundary(), out.page_sizes, path.name)
    els, sigs, st = ner.extract_elements(doc, cfg, reg.ner())
    print("=" * 72)
    print(path.name, st)
    for kind, items in (("party", els.parties), ("amount", els.amounts), ("date", els.dates),
                        ("subject", els.subjects), ("identifier", els.identifiers),
                        ("duration", els.durations)):
        for e in items[:4]:
            print("  %-10s %-28s norm=%-14s role=%-8s conf=%.2f %s" % (
                kind, e.value[:26], e.normalized[:14], e.role, e.confidence, e.source))
    for s in sigs:
        print("  SIGNAL", s.kind, "|", s.note[:70])
