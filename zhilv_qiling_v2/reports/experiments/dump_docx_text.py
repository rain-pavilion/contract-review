import sys, os
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

def iter_block_items(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

path = sys.argv[1]
out = sys.argv[2]
doc = Document(path)
lines = []
ti = 0
for b in iter_block_items(doc):
    if isinstance(b, Paragraph):
        t = b.text.strip()
        if t:
            lines.append(f"[{b.style.name}] {t}")
    else:
        ti += 1
        lines.append(f"=== TABLE {ti} ({len(b.rows)}x{len(b.columns)}) ===")
        for r in b.rows:
            cells = []
            for c in r.cells:
                cells.append(c.text.strip().replace("\n", " / "))
            lines.append(" | ".join(cells))
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("paragraphs+tables:", len(lines), "chars:", sum(len(x) for x in lines))
