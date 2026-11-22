"""生成论文插图（纯 Pillow 绘制，不依赖 matplotlib / graphviz）。

三张图对应论文第 4 章的"系统架构"与"核心机制"：

- `fig_architecture.png`  五层总体架构
- `fig_agents.png`        五个智能体的执行图与有界回环
- `fig_retrieval.png`     三路召回 + 加权 RRF + 学习型重排

画法全部是矩形 + 折线 + 文本，参数集中在顶部常量里，改字号/配色只改一处。

    python tools/make_thesis_figures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FONT_REGULAR = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

BG = (255, 255, 255)
INK = (34, 40, 49)
LINE = (120, 132, 148)
LAYER = [(236, 244, 252), (232, 246, 238), (255, 246, 228), (243, 238, 252), (245, 245, 245)]
ACCENT = (46, 116, 181)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def box(draw: ImageDraw.ImageDraw, xy, text: str, *, size: int = 17, fill=(255, 255, 255),
        outline=LINE, bold: bool = False, radius: int = 8) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)
    font_obj = font(size, bold)
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=font_obj)[3] for line in lines]
    total = sum(heights) + (len(lines) - 1) * 6
    y = xy[1] + ((xy[3] - xy[1]) - total) / 2
    for line, height in zip(lines, heights):
        width = draw.textbbox((0, 0), line, font=font_obj)[2]
        draw.text((xy[0] + ((xy[2] - xy[0]) - width) / 2, y), line, font=font_obj, fill=INK)
        y += height + 6


def arrow(draw: ImageDraw.ImageDraw, start, end, *, color=LINE, width: int = 2,
          head: int = 9) -> None:
    draw.line([start, end], fill=color, width=width)
    (x1, y1), (x2, y2) = start, end
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        draw.polygon([(x2, y2), (x2 - sign * head, y2 - head // 2),
                      (x2 - sign * head, y2 + head // 2)], fill=color)
    else:
        sign = 1 if y2 >= y1 else -1
        draw.polygon([(x2, y2), (x2 - head // 2, y2 - sign * head),
                      (x2 + head // 2, y2 - sign * head)], fill=color)


def architecture() -> Path:
    width, height = 1180, 760
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    layers = [
        ("用户层", ["Web 单页界面", "命令行 CLI", "HTTP API", "批量审查"]),
        ("预处理层", ["格式识别\nDOCX/PDF/TXT/图片", "版面感知\n分栏·页眉·断行", "OCR 通道\nRapidOCR", "切分\n规则+DP 全局"]),
        ("核心层", ["understand\n理解", "clause_review\n条款审查", "cross_review\n交叉一致性", "legal\n法条检索", "judge\n综合裁量"]),
        ("知识与检索层", ["法条知识库\n1260 条", "BM25 稀疏", "LSA 稠密", "标签图 PPR", "LTR 重排 + 校准"]),
        ("数据调度层", ["数据集 113 份", "政采爬虫\n多源可扩展", "训练脚本\ntrain.py", "权重 JSON\n可 diff 可手改"]),
    ]
    left, right = 40, width - 40
    top, gap = 24, 14
    row_height = (height - 2 * top - gap * (len(layers) - 1)) / len(layers)
    for index, (name, items) in enumerate(layers):
        y0 = top + index * (row_height + gap)
        y1 = y0 + row_height
        draw.rounded_rectangle((left, y0, right, y1), radius=10,
                               fill=LAYER[index % len(LAYER)], outline=LINE, width=2)
        draw.text((left + 16, y0 + 12), name, font=font(19, True), fill=ACCENT)
        cell_left = left + 168
        cell_width = (right - 20 - cell_left) / len(items)
        for cell, text in enumerate(items):
            x0 = cell_left + cell * cell_width + 6
            x1 = x0 + cell_width - 12
            box(draw, (x0, y0 + 16, x1, y1 - 16), text, size=16,
                fill=(255, 255, 255) if index != 2 else (255, 251, 240),
                bold=(index == 2))
        if index < len(layers) - 1:
            arrow(draw, ((left + right) / 2, y1), ((left + right) / 2, y1 + gap),
                  color=ACCENT, width=3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fig_architecture.png"
    image.save(path, "PNG")
    return path


def agents() -> Path:
    width, height = 1180, 620
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    box(draw, (60, 40, 300, 120), "① understand\n角色标注·要素抽取", bold=True,
        fill=(236, 244, 252))
    box(draw, (380, 20, 620, 100), "② clause_review\n27 码条款规则", fill=(232, 246, 238))
    box(draw, (380, 120, 620, 200), "③ cross_review\n跨条款一致性·图谱信号",
        fill=(232, 246, 238))
    box(draw, (700, 20, 940, 220), "merge\n候选合并·去重\n（issue_code 进 tags）",
        fill=(255, 246, 228))
    box(draw, (700, 260, 940, 360), "④ legal\n四元组改写 → 三路召回\n→ 只接受可溯源条文",
        fill=(243, 238, 252))
    box(draw, (700, 400, 940, 500), "⑤ judge\n七分类 · 0~100 分", fill=(243, 238, 252))
    box(draw, (380, 420, 620, 520), "arbitrate\n证据强度仲裁\n（碰撞 → 冲突记录）",
        fill=(255, 246, 228))
    box(draw, (380, 540, 620, 604), "fuse → ReviewReport\n校准置信 · 风险队列 · 轨迹",
        bold=True, fill=(236, 244, 252))

    arrow(draw, (300, 80), (380, 60), color=ACCENT)
    arrow(draw, (300, 90), (380, 160), color=ACCENT)
    arrow(draw, (620, 60), (700, 100), color=LINE)
    arrow(draw, (620, 160), (700, 150), color=LINE)
    arrow(draw, (820, 220), (820, 260), color=ACCENT)
    arrow(draw, (820, 360), (820, 400), color=ACCENT)
    arrow(draw, (700, 450), (620, 470), color=ACCENT)
    arrow(draw, (500, 520), (500, 540), color=ACCENT)
    # 有界回环：证据不足 → 回 ②③；引用为空 → 回 ④；裁量不足 → 回 ⑤
    draw.line([(500, 420), (500, 240), (660, 240), (660, 120)], fill=(200, 90, 60), width=2)
    arrow(draw, (660, 120), (620, 110), color=(200, 90, 60))
    draw.text((512, 250), "回环 A：证据不足（上限 max_evidence_rounds）",
             font=font(15), fill=(200, 90, 60))
    draw.line([(960, 300), (1040, 300), (1040, 460), (960, 460)], fill=(200, 90, 60), width=2)
    arrow(draw, (960, 460), (940, 450), color=(200, 90, 60))
    draw.text((980, 470), "回环 B/C：", font=font(15), fill=(200, 90, 60))
    draw.multiline_text((980, 494), "引用为空\n裁量不足", font=font(14),
                        fill=(200, 90, 60), spacing=3)
    path = OUT_DIR / "fig_agents.png"
    image.save(path, "PNG")
    return path


def retrieval() -> Path:
    width, height = 1180, 520
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    box(draw, (30, 200, 230, 320), "条款文本\n+ 风险码 + 合同类型", bold=True,
        fill=(236, 244, 252))
    box(draw, (280, 60, 500, 140), "查询改写\n(行为·主体·标的·后果)", fill=(255, 246, 228))
    box(draw, (280, 190, 500, 260), "BM25 稀疏召回\n字 uni+bigram", fill=(232, 246, 238))
    box(draw, (280, 290, 500, 360), "LSA 稠密召回\nchar_wb TF-IDF+SVD", fill=(232, 246, 238))
    box(draw, (280, 390, 500, 460), "标签图召回\nPPR 扩散", fill=(232, 246, 238))
    box(draw, (550, 220, 760, 330), "加权 RRF 融合\nw=(1.0, 0.9, 1.0), k=15", bold=True,
        fill=(255, 246, 228))
    box(draw, (810, 130, 1010, 250), "15 维特征\n列标准化 LTR", fill=(243, 238, 252))
    box(draw, (810, 300, 1010, 420), "标签扩展\ntop-k 尾部补位", fill=(243, 238, 252))
    box(draw, (1050, 220, 1160, 330), "Top-k\n+ 引用", bold=True, fill=(236, 244, 252))

    arrow(draw, (230, 250), (280, 100), color=ACCENT)
    arrow(draw, (230, 260), (280, 225), color=ACCENT)
    arrow(draw, (230, 270), (280, 325), color=ACCENT)
    arrow(draw, (230, 285), (280, 425), color=ACCENT)
    arrow(draw, (500, 225), (550, 250), color=LINE)
    arrow(draw, (500, 325), (550, 275), color=LINE)
    arrow(draw, (500, 425), (550, 300), color=LINE)
    arrow(draw, (760, 260), (810, 190), color=ACCENT)
    arrow(draw, (1010, 190), (1030, 240), color=ACCENT)
    arrow(draw, (760, 300), (810, 360), color=LINE)
    arrow(draw, (1010, 360), (1035, 300), color=LINE)
    draw.text((300, 482), "硬护栏：只接受本地知识库中真实存在的条文", font=font(15), fill=INK)
    path = OUT_DIR / "fig_retrieval.png"
    image.save(path, "PNG")
    return path


def main() -> int:
    for maker in (architecture, agents, retrieval):
        path = maker()
        print(f"[figures] {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
