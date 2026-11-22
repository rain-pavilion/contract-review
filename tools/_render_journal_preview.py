import pathlib

import pymupdf

doc = pymupdf.open(r"C:\Users\ROG\AppData\Local\Temp\journal_page1.pdf")
out = pathlib.Path("docs") / "figures" / "preview"
out.mkdir(parents=True, exist_ok=True)
for i in range(len(doc)):
    target = out / f"期刊首页-p{i + 1}.png"
    doc[i].get_pixmap(dpi=110).save(str(target))
    text = doc[i].get_text()
    head = text[:60].replace("\n", " ")
    print(f"p{i + 1} {len(text)} 字符 | {head}")
