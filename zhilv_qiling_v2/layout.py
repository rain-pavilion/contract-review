"""版面分析：阅读序、双栏、页眉页脚、表格行、断行合并。

v2 新增模块（v1 没有版面概念：PDF 一律按 `page.get_text()` 的原始顺序切段，
遇到双栏排版或"页眉里塞了合同编号"的扫描件，条款顺序会错乱，进而把整个切分链带偏）。

四个能力
--------
1. `detect_columns` / `reading_order`：按几何空白把一页切成若干纵栏，栏内自上而下、
   栏间自左而右 —— 即简化版 XY-cut，不用递归，O(n log n)。
2. `strip_running_heads`：跨页重复文本（页眉/页脚/水印/页码）自动剔除。判据是
   "归一化后在同一文档 ≥40% 页面上重复出现"，而不是简单的 y 阈值 —— 因为很多
   政府采购合同把"合同编号"放在正文顶部第一行，位置像页眉但只出现一次。
3. `mark_table_rows`：同一 y 带内出现 ≥3 个 x 区间且区间间隙大 → 判为表格行，
   供要素抽取使用（金额表、清单表大量存在）。
4. `merge_wrapped_lines`：把排版硬换行接回去（行尾无标点且下一行首字为中文续写）。

所有函数都只吃/吐 `SourceBlock`，不碰 IO，因此可以在 DOCX（无 bbox）场景下安全跳过。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from . import text as T
from .models import SourceBlock

# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

BBox = tuple[float, float, float, float]


def bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_center_y(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def has_geometry(blocks: Sequence[SourceBlock]) -> bool:
    return any(b.bbox for b in blocks)


def _page_groups(blocks: Iterable[SourceBlock]) -> dict[int, list[SourceBlock]]:
    groups: dict[int, list[SourceBlock]] = defaultdict(list)
    for block in blocks:
        groups[block.page if block.page is not None else 0].append(block)
    return groups


# ---------------------------------------------------------------------------
# 1. 分栏 + 阅读序
# ---------------------------------------------------------------------------

def detect_columns(blocks: Sequence[SourceBlock], page_width: float) -> list[float]:
    """返回竖切位置（页内 x 坐标，升序）。

    做法：把每个 block 的 x 区间投影到坐标轴，统计"被覆盖"的 x 采样点；连续未覆盖
    且宽度 > 页宽 3% 的空白带即为栏间空白。空白带中点作为切点。单栏页返回空列表。
    """
    if not page_width or len(blocks) < 2:
        return []
    if any(b.bbox is None for b in blocks):
        return []
    step = max(1.0, page_width / 400.0)
    covered = [False] * int(page_width / step) + [False]
    for block in blocks:
        bbox = block.bbox
        assert bbox is not None
        start = max(0, int(bbox[0] / step))
        end = min(len(covered) - 1, int(bbox[2] / step))
        for idx in range(start, end + 1):
            covered[idx] = True
    min_gap = max(6.0, page_width * 0.03)
    cuts: list[float] = []
    run_start: int | None = None
    for idx, hit in enumerate(covered):
        if not hit and run_start is None:
            run_start = idx
        elif hit and run_start is not None:
            width = (idx - run_start) * step
            # 只接受"内部空白"：两侧都要有内容，避免把页边距当栏间距
            if width >= min_gap and run_start > 0 and idx < len(covered) - 1:
                cuts.append((run_start + idx) / 2.0 * step)
            run_start = None
    # 过多的切点说明是表格/散排，不作为分栏依据
    return cuts if len(cuts) <= 3 else []


def reading_order(
    blocks: Sequence[SourceBlock],
    page_width: float | None = None,
    *,
    tol: float | None = None,
) -> list[SourceBlock]:
    """按几何位置重排一页内的 blocks。没有 bbox 时原顺序返回。"""
    if not has_geometry(blocks):
        return sorted(blocks, key=lambda b: b.order)
    width = page_width or max((b.bbox[2] for b in blocks if b.bbox), default=0.0)
    cuts = detect_columns(blocks, width) if width else []
    line_tol = tol if tol is not None else 3.0

    def column_of(block: SourceBlock) -> int:
        if not cuts or block.bbox is None:
            return 0
        center = (block.bbox[0] + block.bbox[2]) / 2.0
        for index, cut in enumerate(cuts):
            if center < cut:
                return index
        return len(cuts)

    def sort_key(block: SourceBlock) -> tuple[int, float, float]:
        bbox = block.bbox or (0.0, 0.0, 0.0, 0.0)
        # y 方向按行吸附，避免同一行内左右两块因零点几像素错位来回颠倒
        row = round(bbox_center_y(bbox) / max(line_tol, 0.1))
        return (column_of(block), row, bbox[0])

    ordered = sorted(blocks, key=sort_key)
    for index, block in enumerate(ordered):
        block.order = index
        block.column = column_of(block)
    return ordered


# ---------------------------------------------------------------------------
# 2. 页眉页脚 / 水印
# ---------------------------------------------------------------------------

_PAGE_NO_RE = re.compile(
    r"^(?:[-—–\s]*\d{1,4}\s*[-—–]*|第\s*\d+\s*页(?:\s*[/共]\s*\d+\s*页)?|"
    r"page\s*\d+(?:\s*of\s*\d+)?|- ?\d+ ?-)$",
    re.I,
)


def _head_key(text: str) -> str:
    return re.sub(r"\d+", "#", T.squash(text))


def strip_running_heads(blocks: Sequence[SourceBlock]) -> tuple[list[SourceBlock], list[dict]]:
    """剔除跨页重复的页眉/页脚/水印与纯页码行。

    返回 (保留的 blocks, 剔除记录)。剔除记录进报告，符合"可审计"的要求。
    """
    pages = _page_groups(blocks)
    total_pages = max(1, len(pages))
    counter: Counter[str] = Counter()
    for page_blocks in pages.values():
        seen: set[str] = set()
        for block in page_blocks:
            key = _head_key(block.text)
            if key and len(key) >= 2 and key not in seen:
                seen.add(key)
                counter[key] += 1

    threshold = max(2, int(total_pages * 0.4))
    repeated = {key for key, count in counter.items() if count >= threshold}

    kept: list[SourceBlock] = []
    removed: list[dict] = []
    for block in blocks:
        reason = ""
        stripped = block.text.strip()
        if not stripped:
            continue
        if _PAGE_NO_RE.match(stripped):
            reason = "page_number"
        elif len(repeated) and _head_key(stripped) in repeated and len(stripped) <= 60:
            reason = "running_head"
        elif block.bbox and block.page is not None:
            # 兜底：位置极端贴边且极短
            height = block.metadata.get("page_height")
            if height and len(stripped) <= 24:
                top_gap = block.bbox[1]
                bottom_gap = height - block.bbox[3]
                if top_gap < height * 0.045 or bottom_gap < height * 0.045:
                    reason = "margin_short_line"
        if reason:
            removed.append({"block_id": block.block_id, "reason": reason,
                            "text": T.truncate(stripped, 40)})
        else:
            kept.append(block)
    for index, block in enumerate(kept):
        block.order = index
    return kept, removed


# ---------------------------------------------------------------------------
# 3. 表格行
# ---------------------------------------------------------------------------

_TABLE_SPLIT_RE = re.compile(r"\s{2,}|\t+|\|")


def mark_table_rows(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """把"一行内多列"的 block 标成 `table_row`，并填 `cells` 元数据。"""
    for block in blocks:
        if block.kind in ("table", "cell"):
            continue
        text = block.text
        if not text or len(text) < 6:
            continue
        cells = [c.strip() for c in _TABLE_SPLIT_RE.split(text) if c.strip()]
        multi_column = len(cells) >= 3 and max((len(c) for c in cells), default=0) < 60
        numericish = sum(1 for c in cells if re.fullmatch(r"[\d,，.\-—%元万圆/年月日]+", c)) >= 2
        if multi_column and (numericish or len(set(cells)) == len(cells)):
            block.kind = "table_row"
            block.metadata.setdefault("cells", cells)
    return list(blocks)


def table_rows_to_text(blocks: Sequence[SourceBlock]) -> str:
    """把连续表格行拼成 `| a | b |` 形式，供检索与要素抽取。"""
    lines = []
    for block in blocks:
        cells = block.metadata.get("cells")
        if block.kind == "table_row" and cells:
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(block.text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 断行合并
# ---------------------------------------------------------------------------

_TERMINAL = "。；;！!？?：:.\"'）)》】」』"

#: 这些行首是"标签：值"式的元信息，**不能**被上一条标题/短行吞并
_META_HEADS = (
    "项目名称", "合同编号", "合同名称", "甲方", "乙方", "丙方", "委托方", "受托方",
    "采购人", "供应商", "签订地点", "签订日期", "签订时间", "地址", "电话", "联系人",
    "统一社会信用代码", "法定代表人", "开户银行", "账号", "项目编号", "备案号",
)


def merge_wrapped_lines(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """合并排版硬换行。

    只在**证据充分**时合并，否则宁可分开（切分链里合并错了比不合并更难修）：
    * 上一行不以终止标点结尾；
    * 上一行长度接近本栏行宽（≥18 字），且本身不是条款编号行；
    * 下一行不以编号/序号开头，也不是"甲方："这类元信息行。
    """
    if len(blocks) < 2:
        return list(blocks)
    merged: list[SourceBlock] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        prev = merged[-1]
        same_page = prev.page == block.page
        same_col = prev.column == block.column
        prev_text = prev.text.rstrip()
        cur_text = block.text.lstrip()
        joinable = (
            same_page and same_col
            and prev.kind in ("paragraph", "line")
            and block.kind in ("paragraph", "line")
            and prev_text
            and cur_text
            and prev_text[-1] not in _TERMINAL
            and not T.detect_label(cur_text)["label"]
            and not T.detect_label(prev_text)["label"]
            and len(prev_text) >= 18
            and cur_text.startswith(_META_HEADS) is False
            and not cur_text.startswith(("第", "附件", "附表", "签章", "签字", "盖章"))
        )
        if joinable:
            prev.text = prev_text + cur_text
            prev.metadata.setdefault("merged_from", []).append(block.block_id)
            if prev.bbox and block.bbox:
                prev.bbox = (
                    min(prev.bbox[0], block.bbox[0]),
                    min(prev.bbox[1], block.bbox[1]),
                    max(prev.bbox[2], block.bbox[2]),
                    max(prev.bbox[3], block.bbox[3]),
                )
                prev.metadata.setdefault("page_height", prev.metadata.get("page_height"))
        else:
            merged.append(block)
    for index, block in enumerate(merged):
        block.order = index
    return merged


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def analyze_layout(
    blocks: Sequence[SourceBlock],
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[SourceBlock], dict]:
    """对整份文档跑一遍版面流水线，返回 (blocks, stats)。

    步骤：分页重排 → 去页眉页脚 → 标表格行 → 合并断行 → 全局重编号。
    """
    page_sizes = page_sizes or {}
    ordered: list[SourceBlock] = []
    for page, page_blocks in sorted(_page_groups(blocks).items()):
        width, height = page_sizes.get(page, (0.0, 0.0))
        if height:
            for block in page_blocks:
                block.metadata.setdefault("page_height", height)
        ordered.extend(reading_order(page_blocks, width or None))

    kept, removed = strip_running_heads(ordered)
    kept = mark_table_rows(kept)
    kept = merge_wrapped_lines(kept)

    for index, block in enumerate(kept):
        block.order = index

    stats = {
        "pages": len(_page_groups(blocks)),
        "blocks_in": len(blocks),
        "blocks_out": len(kept),
        "removed": len(removed),
        "removed_samples": removed[:8],
        "columns_detected": max((b.column for b in kept), default=0) + 1,
        "table_rows": sum(1 for b in kept if b.kind == "table_row"),
        "merged_blocks": sum(len(b.metadata.get("merged_from", [])) for b in kept),
        "geometry": has_geometry(kept),
    }
    return kept, stats
