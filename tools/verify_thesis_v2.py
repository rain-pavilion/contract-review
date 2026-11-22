"""论文成稿的结构化验收：样式、题注、图表编号、域、字数。

不渲染（本机无 LibreOffice），只做 XML/对象级核对 —— 这些检查覆盖了"打开会
不会乱、目录能不能刷新、图表编号是否自洽"这三类最常见的问题。

    python tools/verify_thesis_v2.py [docx 路径]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

_DOCS = Path(__file__).resolve().parent.parent / "docs"
# 论文 docx 可能被用户改名（去掉"毕业论文-"前缀），按存在性自动发现，避免死路径
DEFAULT = next(
    (_DOCS / _n for _n in ("智律契灵合同智能审查系统v2.docx",
                           "毕业论文-智律契灵合同智能审查系统v2.docx")
     if (_DOCS / _n).exists()),
    _DOCS / "智律契灵合同智能审查系统v2.docx",
)

REQUIRED_STYLES = {
    "论文1级标题", "论文2级标题", "论文3级标题", "论文正文", "Caption",
    "toc 1", "toc 2", "toc 3", "目录标题格式",
}


def iter_paragraphs(document: Document):
    for paragraph in document.paragraphs:
        yield paragraph


def field_codes(document: Document) -> list[str]:
    codes: list[str] = []
    for part in [document.element.body] + [section.footer._element
                                           for section in document.sections]:
        for node in part.iter(qn("w:instrText")):
            if node.text:
                codes.append(node.text.strip())
    return codes


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT
    document = Document(str(path))

    styles = Counter(paragraph.style.name for paragraph in iter_paragraphs(document))
    chinese = sum(len(re.findall(r"[\u4e00-\u9fff]", paragraph.text))
                  for paragraph in iter_paragraphs(document))
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                chinese += len(re.findall(r"[\u4e00-\u9fff]", cell.text))

    missing = REQUIRED_STYLES - set(styles)
    codes = field_codes(document)
    toc_fields = [code for code in codes if code.startswith("TOC")]
    page_fields = [code for code in codes if "PAGE" in code]
    images = len(document.inline_shapes)

    texts = [paragraph.text.strip() for paragraph in iter_paragraphs(document)]
    # 题注只认 Caption 样式的段落 —— 正文里"表 4-1 中 span_start 是……"这类引用句
    # 也以"表 x-y"开头，按文本前缀抓会全部误判。
    caption_texts = [paragraph.text.strip() for paragraph in iter_paragraphs(document)
                     if paragraph.style.name == "Caption"]
    figure_captions = [text for text in caption_texts if re.match(r"^图\s*\d+-\d+", text)]
    table_captions = [text for text in caption_texts if re.match(r"^表\s*\d+-\d+", text)]

    # 题注编号必须"章内递增、章外不跳"
    def check_caption_sequence(captions: list[str], kind: str) -> list[str]:
        problems: list[str] = []
        last_chapter = 0
        last_index = 0
        for caption in captions:
            match = re.match(rf"^{kind}\s*(\d+)-(\d+)", caption)
            if not match:
                continue
            chapter, index = int(match.group(1)), int(match.group(2))
            if chapter != last_chapter:
                if index != 1:
                    problems.append(f"{caption} 章首编号不是 1")
                last_chapter, last_index = chapter, index
            elif index != last_index + 1:
                problems.append(f"{caption} 编号不连续（上一张为 {last_chapter}-{last_index}）")
                last_index = index
            else:
                last_index = index
        return problems

    refs_block = [text for text in texts if re.match(r"^\[\d+\]\s", text)]
    ref_numbers = [int(re.match(r"^\[(\d+)\]", text).group(1)) for text in refs_block]
    body_text = "\n".join(texts)
    # [0] 不是引用，是附录代码里的下标（如 pair[0]），按引用统计会把去重数顶到 29。
    cited = sorted({int(match) for match in re.findall(r"\[(\d+)\]", body_text)
                    if int(match) > 0})

    print(f"文件            : {path}  ({path.stat().st_size} B)")
    print(f"段落/表格/插图  : {len(document.paragraphs)} / {len(document.tables)} / {images}")
    print(f"汉字数          : {chinese}")
    print(f"样式使用        : {dict(sorted(styles.items(), key=lambda kv: -kv[1]))}")
    print(f"缺失必需样式    : {sorted(missing) or '无'}")
    print(f"TOC 域 / 页码域 : {len(toc_fields)} / {len(page_fields)}  {toc_fields[:1]}")
    print(f"图题/表题       : {len(figure_captions)} / {len(table_captions)}")
    print(f"参考文献条数    : {len(refs_block)}  连续={ref_numbers == list(range(1, len(refs_block) + 1))}")
    print(f"正文引用编号    : 最大={max(cited) if cited else 0}  去重={len(cited)}  "
          f"越界={[n for n in cited if n > len(refs_block)] or '无'}")

    problems = []
    problems += check_caption_sequence(figure_captions, "图")
    problems += check_caption_sequence(table_captions, "表")
    if missing:
        problems.append(f"缺少样式 {sorted(missing)}")
    if not toc_fields:
        problems.append("没有 TOC 域")
    if len(ref_numbers) != len(set(ref_numbers)):
        problems.append("参考文献编号重复")
    if ref_numbers and ref_numbers != list(range(1, len(ref_numbers) + 1)):
        problems.append("参考文献编号不连续")
    problems += [f"引用越界 [n={n}]" for n in cited if n > len(refs_block)]

    if problems:
        print("\n[FAIL]")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\n[PASS] 结构与题注自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
