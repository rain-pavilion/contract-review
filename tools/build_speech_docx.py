"""Convert the speech markdown into a printable DOCX (no external conversion)."""
from __future__ import annotations
import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
text = SRC.read_text(encoding='utf-8')
lines = text.splitlines()

doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(1.9); sec.bottom_margin = Cm(1.7)
sec.left_margin = Cm(2.0); sec.right_margin = Cm(2.0)
normal = doc.styles['Normal']
normal.font.name = '宋体'
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.25

for level in (1, 2, 3):
    st = doc.styles['Heading %d' % level]
    st.font.name = '黑体'
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    st.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
for size, level in ((18, 1), (14, 2), (12, 3)):
    doc.styles['Heading %d' % level].font.size = Pt(size)

def add_runs(paragraph, value):
    for index, part in enumerate(re.split(r'\*\*(.+?)\*\*', value)):
        if not part:
            continue
        run = paragraph.add_run(part)
        if index % 2 == 1:
            run.bold = True

i = 0
table_buffer = []

def flush_table():
    global table_buffer
    rows = [row for row in table_buffer if row.strip()]
    table_buffer = []
    if not rows:
        return
    cells = []
    for row in rows:
        parts = [cell.strip() for cell in row.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', part) for part in parts if part) and parts:
            continue
        cells.append(parts)
    if not cells:
        return
    width = max(len(row) for row in cells)
    table = doc.add_table(rows=len(cells), cols=width)
    table.style = 'Table Grid'
    for r, row in enumerate(cells):
        for c in range(width):
            value = row[c] if c < len(row) else ''
            cell = table.rows[r].cells[c]
            cell.text = ''
            paragraph = cell.paragraphs[0]
            add_runs(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.font.name = '宋体'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            if r == 0:
                for run in paragraph.runs:
                    run.bold = True
    doc.add_paragraph()

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith('|'):
        table_buffer.append(stripped)
        i += 1
        continue
    if table_buffer:
        flush_table()
    if not stripped:
        i += 1
        continue
    if stripped == '---':
        i += 1
        continue
    match = re.match(r'^(#{1,3})\s+(.*)$', stripped)
    if match:
        doc.add_heading(match.group(2), level=len(match.group(1)))
        i += 1
        continue
    if stripped.startswith('> '):
        p = doc.add_paragraph()
        add_runs(p, stripped[2:])
        for run in p.runs:
            run.italic = True
            run.font.color.rgb = RGBColor(0x7A, 0x4A, 0x00)
        i += 1
        continue
    if re.match(r'^[-*]\s+', stripped):
        p = doc.add_paragraph(style='List Bullet')
        add_runs(p, re.sub(r'^[-*]\s+', '', stripped))
        i += 1
        continue
    if re.match(r'^\d+\.\s+', stripped):
        p = doc.add_paragraph(style='List Number')
        add_runs(p, re.sub(r'^\d+\.\s+', '', stripped))
        i += 1
        continue
    p = doc.add_paragraph()
    add_runs(p, stripped)
    i += 1
if table_buffer:
    flush_table()

doc.core_properties.title = '五分钟答辩稿与答辩问题准备'
doc.core_properties.author = '智律契灵团队'
doc.save(str(DST))
print('saved', DST, 'paragraphs', len(doc.paragraphs), 'tables', len(doc.tables))
