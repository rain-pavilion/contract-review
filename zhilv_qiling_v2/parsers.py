"""文档解析：DOCX / PDF / TXT / HTML / 图片 → `SourceBlock` 列表。

相对 v1 `parsers.py` 的升级
--------------------------
1. **DOCX 保留字号与加粗**：v1 用自研 OOXML 只取文本与段落样式名。v2 额外取
   `w:rPr/w:sz` 与 `w:b`，标题识别不再只靠正则（"三、付款方式"这种没有样式的手写
   标题也能靠字号+加粗判出来）。
2. **PDF 出 bbox**：交给 `layout.py` 做阅读序与页眉页脚处理；v1 直接拿
   `page.get_text()` 的字符串，双栏合同必然错序。
3. **PDF 表格**：`page.find_tables()` 可用时抽成 `kind="table"` 的块，表格里的金额
   不再被拆成散行。
4. **扫描页不假装有文字**：文字层少于阈值的页面标记 `needs_ocr=True`，转入
   `ocr.py` 的队列；v1 是静默返回空字符串，报告里看不出"这页其实没读到"。
5. **编码探测**：TXT 用 BOM → 严格 utf-8 → gb18030 → big5 的**有序**试探，
   并用"替换字符占比"打分选最优；v1 用 `apparent_encoding`，在 GBK 合同上常判成
   cp1252，正文变乱码。
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

from . import text as T
from .config import capabilities
from .models import SourceBlock, stable_id

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED = 160 * 1024 * 1024
MAX_PDF_PAGES = 1200
MAX_TEXT_CHARS = 6_000_000
SCANNED_PAGE_TEXT_THRESHOLD = 60

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ParseError(ValueError):
    """解析失败。消息要能直接贴进报告，所以写清楚原因与建议动作。"""


@dataclass
class ParseOutput:
    """解析产物：blocks + 版面统计 + 需要 OCR 的页。"""

    blocks: list[SourceBlock]
    file_type: str
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    needs_ocr_pages: list[int] = field(default_factory=list)
    image_blocks: list[SourceBlock] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 编码探测
# ---------------------------------------------------------------------------

_CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16", "latin-1")


def decode_bytes(content: bytes) -> tuple[str, str]:
    """返回 (文本, 编码名)。用替换字符率 + 中文字符率联合打分。"""
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig", errors="replace"), "utf-8-sig"
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16", errors="replace"), "utf-16"

    best: tuple[float, str, str] | None = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            decoded = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if not decoded:
            continue
        bad = decoded.count("\ufffd")
        cjk = T.cjk_ratio(decoded)
        # 编码对了：几乎没有替换字符；中文文档 cjk 比例通常 > 0.2
        score = (1.0 - min(1.0, bad / max(1, len(decoded)) * 50)) + cjk
        if best is None or score > best[0]:
            best = (score, decoded, encoding)
    if best is None:
        return content.decode("utf-8", errors="replace"), "utf-8(replace)"
    return best[1], best[2]


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _docx_styles(zf: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    try:
        xml = zf.read("word/styles.xml")
    except KeyError:
        return {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    styles: dict[str, dict[str, Any]] = {}
    for style in root.iter(f"{W}style"):
        style_id = style.get(f"{W}styleId") or ""
        name_el = style.find(f"{W}name")
        name = name_el.get(f"{W}val") if name_el is not None else style_id
        info: dict[str, Any] = {"name": name}
        rpr = style.find(f"{W}rPr")
        if rpr is not None:
            sz = rpr.find(f"{W}sz")
            if sz is not None:
                try:
                    info["size"] = float(sz.get(f"{W}val", "0")) / 2.0
                except ValueError:
                    pass
            bold = rpr.find(f"{W}b")
            if bold is not None and bold.get(f"{W}val", "1") not in ("0", "false"):
                info["bold"] = True
        outline = style.find(f"{W}pPr/{W}outlineLvl")
        if outline is not None:
            info["outline"] = outline.get(f"{W}val")
        styles[style_id] = info
    return styles


def _run_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return "".join(parts)


def _paragraph_props(paragraph: ET.Element, styles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    info: dict[str, Any] = {"style": "", "font_size": None, "is_bold": False, "outline": None}
    ppr = paragraph.find(f"{W}pPr")
    if ppr is not None:
        style_el = ppr.find(f"{W}pStyle")
        if style_el is not None:
            style_id = style_el.get(f"{W}val", "")
            info["style"] = style_id
            style_info = styles.get(style_id, {})
            info["style_name"] = style_info.get("name", style_id)
            if "size" in style_info:
                info["font_size"] = style_info["size"]
            if style_info.get("bold"):
                info["is_bold"] = True
            if "outline" in style_info:
                info["outline"] = style_info["outline"]
        num = ppr.find(f"{W}numPr")
        if num is not None:
            info["numbered"] = True
    # 段落内直接格式优先
    for run in paragraph.findall(f"{W}r"):
        rpr = run.find(f"{W}rPr")
        if rpr is None:
            continue
        sz = rpr.find(f"{W}sz")
        if sz is not None:
            try:
                info["font_size"] = float(sz.get(f"{W}val", "0")) / 2.0
            except ValueError:
                pass
        bold = rpr.find(f"{W}b")
        if bold is not None and bold.get(f"{W}val", "1") not in ("0", "false"):
            info["is_bold"] = True
        break
    return info


def _table_to_blocks(table: ET.Element, order_start: int) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    row_index = 0
    for row in table.findall(f"{W}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{W}tc"):
            cell_text = " ".join(
                t for t in (_run_text(p).strip() for p in cell.findall(f"{W}p")) if t
            )
            cells.append(T.clean_text(cell_text, keep_newlines=False))
        row_index += 1
        if not any(cells):
            continue
        blocks.append(
            SourceBlock(
                block_id=stable_id("sb", "tbl", order_start, row_index),
                order=order_start + row_index,
                kind="table_row",
                text=" | ".join(cells),
                metadata={"cells": cells, "row_index": row_index},
            )
        )
    return blocks


def parse_docx(content: bytes, source_name: str = "") -> ParseOutput:
    """解析 DOCX（纯 lxml/ElementTree，不依赖 python-docx）。"""
    if len(content) > MAX_FILE_BYTES:
        raise ParseError(f"文件过大：{len(content) / 1048576:.1f}MB > 40MB 上限")
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ParseError(f"不是有效的 DOCX（zip 结构损坏）：{exc}") from exc

    total_uncompressed = sum(info.file_size for info in zf.infolist())
    if total_uncompressed > MAX_DOCX_UNCOMPRESSED:
        raise ParseError(
            f"DOCX 解压后 {total_uncompressed / 1048576:.1f}MB 超过 160MB 上限（疑似压缩炸弹）"
        )

    try:
        document_xml = zf.read("word/document.xml")
    except KeyError as exc:
        raise ParseError("DOCX 缺少 word/document.xml，可能不是 Word 文档") from exc

    styles = _docx_styles(zf)
    root = ET.fromstring(document_xml)
    body = root.find(f"{W}body")
    if body is None:
        raise ParseError("DOCX 缺少 body 节点")

    blocks: list[SourceBlock] = []
    warnings: list[str] = []
    for child in list(body):
        if child.tag == f"{W}p":
            raw = _run_text(child)
            text = T.clean_text(raw)
            if not text:
                continue
            info = _paragraph_props(child, styles)
            blocks.append(
                SourceBlock(
                    block_id=stable_id("sb", "docx", len(blocks), text[:40]),
                    order=len(blocks),
                    kind="paragraph",
                    text=text,
                    style=str(info.get("style_name") or info.get("style") or ""),
                    font_size=info.get("font_size"),
                    is_bold=bool(info.get("is_bold")),
                    metadata={k: v for k, v in info.items() if k != "style"},
                )
            )
        elif child.tag == f"{W}tbl":
            table_blocks = _table_to_blocks(child, len(blocks))
            blocks.extend(table_blocks)
            if not table_blocks:
                warnings.append("发现空表格，已跳过")

    if not blocks:
        raise ParseError("DOCX 中没有可提取的文本（可能是纯图片文档，需走 OCR）")

    stats = {
        "paragraphs": sum(1 for b in blocks if b.kind == "paragraph"),
        "table_rows": sum(1 for b in blocks if b.kind == "table_row"),
        "chars": sum(len(b.text) for b in blocks),
        "styles": sorted({b.style for b in blocks if b.style})[:12],
    }
    return ParseOutput(blocks=blocks, file_type="docx", stats=stats, warnings=warnings)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _pdf_with_pymupdf(content: bytes) -> ParseOutput:
    import pymupdf  # type: ignore

    doc = pymupdf.open(stream=content, filetype="pdf")
    if doc.page_count > MAX_PDF_PAGES:
        raise ParseError(f"PDF 页数 {doc.page_count} 超过 {MAX_PDF_PAGES} 页上限")

    blocks: list[SourceBlock] = []
    page_sizes: dict[int, tuple[float, float]] = {}
    needs_ocr: list[int] = []
    warnings: list[str] = []
    table_pages = 0

    for page_index, page in enumerate(doc, start=1):
        rect = page.rect
        page_sizes[page_index] = (float(rect.width), float(rect.height))
        page_text_len = len(page.get_text().strip())

        table_ids: set[int] = set()
        try:
            found = page.find_tables()
            for table in found.tables:
                table_pages += 1
                for row in table.extract():
                    cells = [T.clean_text(str(c or ""), keep_newlines=False) for c in row]
                    if not any(cells):
                        continue
                    bbox = None
                    try:
                        bbox = tuple(float(v) for v in table.bbox)  # type: ignore[assignment]
                    except Exception:
                        bbox = None
                    blocks.append(
                        SourceBlock(
                            block_id=stable_id("sb", "pdf-tbl", page_index, len(blocks)),
                            order=len(blocks),
                            kind="table_row",
                            text=" | ".join(cells),
                            page=page_index,
                            bbox=bbox,  # type: ignore[arg-type]
                            metadata={"cells": cells},
                        )
                    )
        except Exception:  # pragma: no cover - 老版本 pymupdf 没有 find_tables
            warnings.append("当前 PyMuPDF 版本不支持表格抽取，表格按文本行处理")

        page_dict = page.get_text("dict")
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue
            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans)
                text = T.clean_text(text, keep_newlines=False)
                if not text:
                    continue
                sizes = [span.get("size", 0.0) for span in spans if span.get("text")]
                flags = [span.get("flags", 0) for span in spans if span.get("text")]
                font_size = round(sum(sizes) / len(sizes), 1) if sizes else None
                is_bold = any(int(flag) & 16 for flag in flags)
                blocks.append(
                    SourceBlock(
                        block_id=stable_id("sb", "pdf", page_index, len(blocks), text[:32]),
                        order=len(blocks),
                        kind="line",
                        text=text,
                        page=page_index,
                        bbox=tuple(float(v) for v in line.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
                        font_size=font_size,
                        is_bold=is_bold,
                    )
                )

        if page_text_len < SCANNED_PAGE_TEXT_THRESHOLD:
            needs_ocr.append(page_index)

    if not blocks:
        raise ParseError(
            "PDF 没有文字层（疑似整份扫描件）。请在 crawl --ocr 或 ocr 子命令里处理，"
            "不要静默当成空合同。"
        )

    stats = {
        "pages": doc.page_count,
        "text_blocks": len(blocks),
        "table_rows": sum(1 for b in blocks if b.kind == "table_row"),
        "table_pages": table_pages,
        "scanned_pages": len(needs_ocr),
        "chars": sum(len(b.text) for b in blocks),
    }
    return ParseOutput(
        blocks=blocks,
        file_type="pdf",
        page_sizes=page_sizes,
        needs_ocr_pages=needs_ocr,
        stats=stats,
        warnings=warnings,
    )


def _pdf_minimal(content: bytes) -> ParseOutput:
    """无 PyMuPDF 时的兜底：抽未压缩/Flate 内容流里的 Tj/TJ 文本。

    只覆盖常见简单字体（WinAnsi/Identity-H 且带 ToUnicode 的情况不做映射），
    命中率有限 —— 这是**降级路径**，能力表里会如实标注。
    """
    import zlib

    blocks: list[SourceBlock] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", content, re.S):
        chunk = match.group(1)
        try:
            data = zlib.decompress(chunk)
        except zlib.error:
            data = chunk
        if b"Tj" not in data and b"TJ" not in data:
            continue
        for text_match in re.finditer(rb"\((.*?)\)\s*Tj|\[(.*?)\]\s*TJ", data, re.S):
            raw = text_match.group(1) or text_match.group(2) or b""
            raw = re.sub(rb"\)\s*-?\d+\s*\(", b"", raw)
            try:
                decoded = raw.decode("latin-1")
            except UnicodeDecodeError:
                continue
            decoded = re.sub(r"[^\x20-\x7e\u4e00-\u9fff]", "", decoded).strip()
            if len(decoded) < 2:
                continue
            blocks.append(
                SourceBlock(
                    block_id=stable_id("sb", "pdfraw", len(blocks)),
                    order=len(blocks),
                    kind="line",
                    text=decoded,
                )
            )
    if not blocks:
        raise ParseError(
            "PDF 兜底抽取失败。建议安装 pymupdf（pip install pymupdf）或走 OCR 通道。"
        )
    return ParseOutput(
        blocks=blocks,
        file_type="pdf",
        stats={"pages": 0, "text_blocks": len(blocks), "fallback": True},
        warnings=["使用纯 Python PDF 兜底抽取，版面信息缺失，条款顺序可能不准"],
    )


def parse_pdf(content: bytes) -> ParseOutput:
    if capabilities().pymupdf:
        return _pdf_with_pymupdf(content)
    return _pdf_minimal(content)


# ---------------------------------------------------------------------------
# TXT / HTML
# ---------------------------------------------------------------------------

def parse_txt(content: bytes) -> ParseOutput:
    text, encoding = decode_bytes(content)
    if len(text) > MAX_TEXT_CHARS:
        raise ParseError(f"文本长度 {len(text)} 超过 {MAX_TEXT_CHARS} 字符上限")
    blocks: list[SourceBlock] = []
    for line in text.split("\n"):
        cleaned = T.clean_text(line)
        if not cleaned:
            continue
        blocks.append(
            SourceBlock(
                block_id=stable_id("sb", "txt", len(blocks)),
                order=len(blocks),
                kind="paragraph",
                text=cleaned,
            )
        )
    if not blocks:
        raise ParseError("文本文件为空")
    return ParseOutput(
        blocks=blocks,
        file_type="txt",
        stats={"lines": len(blocks), "chars": len(text), "encoding": encoding},
    )


_TAG_RE = re.compile(r"<[^>]+>")


def parse_html(content: bytes) -> ParseOutput:
    text, encoding = decode_bytes(content)
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    blocks: list[SourceBlock] = []
    for chunk in re.split(r"(?i)</(?:p|div|tr|br|h\d|li)>", body):
        cleaned = T.clean_text(T.full_to_half(_TAG_RE.sub(" ", chunk)))
        if len(cleaned) < 2:
            continue
        blocks.append(
            SourceBlock(
                block_id=stable_id("sb", "html", len(blocks)),
                order=len(blocks),
                kind="paragraph",
                text=cleaned,
            )
        )
    if not blocks:
        raise ParseError("HTML 中没有可提取的文本")
    return ParseOutput(
        blocks=blocks,
        file_type="html",
        stats={"blocks": len(blocks), "encoding": encoding},
    )


def image_block(path: Path, page: int | None = None) -> SourceBlock:
    """图片 → 待 OCR 的 block（正文由 ocr.py 填回）。"""
    return SourceBlock(
        block_id=stable_id("sb", "img", str(path), page),
        order=0,
        kind="image",
        text="",
        page=page,
        metadata={"path": str(path), "needs_ocr": True},
    )


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------

_SUFFIX_MAP = {
    ".docx": "docx", ".doc": "doc", ".pdf": "pdf",
    ".txt": "txt", ".md": "txt", ".csv": "txt",
    ".htm": "html", ".html": "html",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".tif": "image", ".tiff": "image", ".bmp": "image", ".webp": "image",
}


def detect_file_type(filename: str, content: bytes = b"") -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".doc":
        return "doc" if content[:4] == b"\xd0\xcf\x11\xe0" else "docx"
    if suffix in _SUFFIX_MAP:
        return _SUFFIX_MAP[suffix]
    if content[:4] == b"%PDF":
        return "pdf"
    if content[:2] == b"PK":
        return "docx"
    return "txt"


def parse_document(filename: str, content: bytes) -> ParseOutput:
    """按类型分发解析。失败抛 `ParseError`（消息可直接进报告）。"""
    if len(content) > MAX_FILE_BYTES:
        raise ParseError(f"文件过大：{len(content) / 1048576:.1f}MB")
    file_type = detect_file_type(filename, content)
    if file_type == "docx":
        return parse_docx(content, filename)
    if file_type == "pdf":
        return parse_pdf(content)
    if file_type == "html":
        return parse_html(content)
    if file_type == "image":
        return ParseOutput(
            blocks=[],
            file_type="image",
            image_blocks=[image_block(Path(filename))],
            stats={"images": 1},
            warnings=["图片需要 OCR"],
        )
    if file_type == "doc":
        raise ParseError(
            "旧版 .doc（OLE）二进制格式：请先转成 .docx（Word 另存为或 "
            "`soffice --headless --convert-to docx`），v2 不做 OLE 二进制解析。"
        )
    return parse_txt(content)


def parse_path(path: Path) -> ParseOutput:
    return parse_document(path.name, path.read_bytes())
