"""生成论文插图（中间产物，输出到 paper_figures/）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

C_LAYER = "#dbe7f3"
C_NODE = "#eaf1fa"
C_AGENT = "#fde9d9"
C_KB = "#e2f0e0"
C_EDGE = "#5b7fa6"
C_LOOP = "#c0392b"


def canvas(w: float, h: float, xmax: float = 10.0, ymax: float = 10.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=C_NODE, ec="#4a6f92", fs=9, bold=False, lw=1.0, r=0.12):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        weight="bold" if bold else "normal", zorder=3, linespacing=1.45,
    )


def arrow(ax, p1, p2, color=C_EDGE, style="-", rad=0.0, lw=1.2, ls="solid"):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
            linestyle=ls, zorder=1, shrinkA=2, shrinkB=2,
        )
    )


def polyline(ax, points, color=C_LOOP, ls="dashed", lw=1.0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=10,
            color=color, lw=lw, linestyle=ls, zorder=1,
        )
    )


# ---------------------------------------------------------------- 图4-1 总体架构
def fig_architecture():
    fig, ax = canvas(7.2, 6.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    layers = [
        ("表现层", "合同上传   审查报告预览   风险条目定位   人工复核队列", "#c9dcf0"),
        ("编排层", "M6 统一状态图   四路并行   有界回环   结果融合", "#f8dcc6"),
        ("智能体层", "合同理解   逐条审查   跨条款审查   法律适用   裁判仲裁", "#f8dcc6"),
        ("知识层", "文档解析   条款切分   要素抽取   结构图谱   法条检索", "#d7ead4"),
        ("基础设施层", "PyMuPDF   LangGraph   Qdrant   Ollama(bge-m3)   LLM API", "#e6e6e6"),
    ]
    y = 8.6
    for name, items, color in layers:
        box(ax, 0.15, y, 1.55, 1.15, name, fc=color, fs=9.5, bold=True)
        box(ax, 1.85, y, 8.0, 1.15, items, fc=color, fs=9)
        y -= 1.75

    for i in range(4):
        yy = 8.6 - i * 1.75
        ax.annotate(
            "", xy=(5.85, yy - 0.52), xytext=(5.85, yy),
            arrowprops=dict(arrowstyle="-|>", color="#7f7f7f", lw=1.1),
        )
    fig.savefig(OUT / "fig4_1_architecture.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- 图4-2 统一图与四条回环
def fig_graph():
    fig, ax = canvas(7.0, 6.4, xmax=11.5, ymax=10.5)
    cx, w = 4.2, 4.0

    box(ax, cx - 0.8, 9.75, 1.6, 0.5, "START", fc="#f0f0f0", fs=8.5)
    box(ax, cx - w / 2, 8.9, w, 0.75, "① 合同理解\n（语义层，仅执行一次）", fc=C_AGENT, fs=8.5, bold=True)

    pw = 2.3
    box(ax, 0.35, 7.35, pw, 0.8, "elements\n要素槽位 + 数学信号", fc=C_NODE, fs=7.6)
    box(ax, 2.85, 7.35, pw, 0.8, "② 逐条审查\n条款级风险候选", fc=C_AGENT, fs=7.6)
    box(ax, 5.35, 7.35, pw, 0.8, "③ 跨条款审查\n一致性风险候选", fc=C_AGENT, fs=7.6)

    box(ax, cx - w / 2, 6.55, w, 0.62, "merge  候选合并去重", fc=C_NODE, fs=8.4)
    box(ax, cx - w / 2, 5.6, w, 0.72, "④ 法律检索与适用（七分类）", fc=C_AGENT, fs=8.4, bold=True)
    box(ax, cx - w / 2, 4.6, w, 0.72, "⑤ LLM 裁判 + 整体合理性结论", fc=C_AGENT, fs=8.4, bold=True)
    box(ax, cx - w / 2, 3.6, w, 0.7, "arbitrate  跨模块冲突仲裁", fc=C_NODE, fs=8.4)
    box(ax, cx - w / 2, 2.6, w, 0.7, "fuse  确定性融合 + 复核队列", fc=C_NODE, fs=8.4)
    box(ax, cx - 1.15, 1.7, 2.3, 0.5, "END  统一审查报告", fc="#f0f0f0", fs=8.2)

    arrow(ax, (cx, 9.75), (cx, 9.65))
    arrow(ax, (cx, 8.9), (cx, 8.30))
    ax.plot([1.5, 6.5], [8.30, 8.30], color=C_EDGE, lw=1.1, zorder=1)
    for x0 in (1.5, 4.0, 6.5):
        arrow(ax, (x0, 8.30), (x0, 8.15))
    for x0 in (1.5, 4.0, 6.5):
        arrow(ax, (x0, 7.35), (cx, 7.17))
    arrow(ax, (cx, 6.55), (cx, 6.32))
    arrow(ax, (cx, 5.60), (cx, 5.32))
    arrow(ax, (cx, 4.60), (cx, 4.32))
    arrow(ax, (cx, 3.60), (cx, 3.32))
    arrow(ax, (cx, 2.60), (cx, 2.20))

    # ⑤ → ②③ 证据回环：沿右侧竖井上行，再折回并行分支
    polyline(
        ax,
        [(cx + w / 2, 4.96), (8.75, 4.96), (8.75, 8.55), (3.35, 8.55), (3.35, 8.17)],
    )
    ax.text(8.86, 6.6, "⑤→②③ 证据回环（≤1）", fontsize=7.2, color=C_LOOP,
            rotation=90, va="center", ha="left")

    # ④ 自环：检索重试
    ax.add_patch(
        FancyArrowPatch(
            (cx - w / 2, 5.96), (cx - w / 2, 5.66), arrowstyle="-|>",
            mutation_scale=9, connectionstyle="arc3,rad=1.7",
            color=C_LOOP, lw=1.0, linestyle="dashed", zorder=1,
        )
    )
    ax.text(cx - w / 2 - 0.12, 5.81, "检索重试 ≤1", fontsize=7.2, color=C_LOOP,
            ha="right", va="center")

    # ⑤ 自环：裁判重试
    ax.add_patch(
        FancyArrowPatch(
            (cx - w / 2, 4.96), (cx - w / 2, 4.66), arrowstyle="-|>",
            mutation_scale=9, connectionstyle="arc3,rad=1.7",
            color=C_LOOP, lw=1.0, linestyle="dashed", zorder=1,
        )
    )
    ax.text(cx - w / 2 - 0.12, 4.81, "裁判重试 ≤1", fontsize=7.2, color=C_LOOP,
            ha="right", va="center")

    # arbitrate → ⑤ 重裁
    polyline(ax, [(cx + w / 2, 3.95), (7.6, 3.95), (7.6, 5.10), (cx + w / 2, 5.10)])
    ax.text(7.72, 4.5, "仲裁重裁 ≤1", fontsize=7.2, color=C_LOOP,
            rotation=90, va="center", ha="left")

    ax.text(0.15, 0.35, "红色虚线为四条有界回环，计数上限均为 1；触发重试即消耗额度，超限后转人工复核",
            fontsize=7.6, color="#555555", ha="left")
    fig.savefig(OUT / "fig4_2_graph.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 图4-3 混合检索与图谱
def fig_retrieval():
    fig, ax = canvas(7.2, 2.9, xmax=11.5, ymax=4.4)

    box(ax, 0.15, 2.45, 1.95, 1.75, "风险候选\n风险码 + 条款原文\n+ 要素摘要", fc=C_NODE, fs=7.6)
    box(ax, 2.55, 3.35, 2.45, 0.95, "dense：bge-m3\n（1024 维余弦）", fc="#f3ecc7", fs=7.6)
    box(ax, 2.55, 1.95, 2.45, 0.95, "sparse：BM25\n（字级 uni/bi-gram）", fc="#f3ecc7", fs=7.2)
    box(ax, 5.35, 2.6, 2.05, 1.0, "RRF 融合\n→ top-k 法条池", fc=C_NODE, fs=7.8, bold=True)
    box(ax, 7.75, 2.6, 2.0, 1.0, "七分类评估\n+ 引用白名单", fc=C_AGENT, fs=7.8, bold=True)
    box(ax, 10.05, 2.6, 1.3, 1.0, "带法条的\n风险条目", fc=C_NODE, fs=7.4)

    arrow(ax, (2.10, 3.55), (2.55, 3.80))
    arrow(ax, (2.10, 3.10), (2.55, 2.45))
    arrow(ax, (5.00, 3.80), (5.35, 3.20))
    arrow(ax, (5.00, 2.45), (5.35, 2.95))
    arrow(ax, (7.40, 3.10), (7.75, 3.10))
    arrow(ax, (9.75, 3.10), (10.05, 3.10))

    box(ax, 3.4, 0.55, 4.6, 0.95,
        "法条知识图谱：1260 条法条 / risk_tags 二部图\n召回后按标签扩展相邻法条", fc=C_KB, fs=7.4)
    arrow(ax, (5.7, 1.50), (5.7, 2.60), color="#3f7d3f")

    ax.text(0.15, 0.18, "引用白名单：article_uid 不在召回池内即置空 citation，禁止模型自造法条",
            fontsize=7.4, color="#555555")
    fig.savefig(OUT / "fig4_3_retrieval.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 图4-4 切分与结构回流
def fig_split():
    fig, ax = canvas(7.2, 2.6, xmax=11.5, ymax=4.15)

    box(ax, 0.15, 2.55, 1.85, 1.0, "DOCX / PDF\n文档解析", fc=C_NODE, fs=7.6)
    box(ax, 2.30, 2.55, 1.85, 1.0, "原子区块\n页码 + 坐标", fc=C_NODE, fs=7.6)
    box(ax, 4.55, 3.05, 2.05, 0.95, "规则切分\n编号 / 样式 / 版面", fc="#f3ecc7", fs=7.2)
    box(ax, 4.55, 1.95, 2.05, 0.95, "LLM 语义分割\nlabel / block_ids", fc="#f3ecc7", fs=7.2)
    box(ax, 6.95, 2.55, 2.3, 1.0, "结构一致性校验\n遗漏·重复·层级·编号", fc=C_KB, fs=7.0)
    box(ax, 9.55, 2.55, 1.85, 1.0, "条款树\n可回溯原文与页码", fc=C_NODE, fs=7.0, bold=True)

    arrow(ax, (2.00, 3.05), (2.30, 3.05))
    arrow(ax, (4.15, 3.20), (4.55, 3.55))
    arrow(ax, (4.15, 2.90), (4.55, 2.45))
    arrow(ax, (6.60, 3.55), (6.95, 3.30))
    arrow(ax, (6.60, 2.45), (6.95, 2.80))
    arrow(ax, (9.25, 3.05), (9.55, 3.05))

    polyline(
        ax,
        [(6.95, 2.62), (6.72, 2.62), (6.72, 1.40), (5.20, 1.40), (5.20, 1.95)],
        color=C_LOOP,
    )
    ax.text(6.85, 1.16, "局部回流：仅当可修正问题数严格减少才采纳，上限 1 轮",
            fontsize=7.4, color=C_LOOP)
    fig.savefig(OUT / "fig4_4_split.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 图6-1 数据集统计
def fig_dataset():
    rows = []
    for meta_path in (ROOT / "data" / "gov_contracts").glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append(meta)
    pages = [r["page_count"] for r in rows]
    total = sum(pages)

    counts = {"scan_low": 0, "scan_high": 0, "render": 0}
    for r in rows:
        for p in r.get("pages", []):
            if p.get("source") != "embedded":
                counts["render"] += 1
            elif p.get("text_len", 0) >= 50:
                counts["scan_high"] += 1
            else:
                counts["scan_low"] += 1
    scanned = counts["scan_low"]
    img_text = counts["scan_high"]
    rendered = counts["render"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})

    bins = [0, 8, 16, 32, 64, 128]
    labels = ["2-7", "8-15", "16-31", "32-63", "64-128"]
    counts = [0] * len(labels)
    for p in pages:
        for i in range(len(bins) - 1):
            if bins[i] <= p < bins[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    ax1.bar(labels, counts, color="#6d9bc3", width=0.62)
    for i, c in enumerate(counts):
        ax1.text(i, c + 0.4, str(c), ha="center", fontsize=8)
    ax1.set_xlabel("单份合同页数区间", fontsize=9)
    ax1.set_ylabel("合同数量 / 份", fontsize=9)
    ax1.set_title(f"共 {len(rows)} 份合同、{total} 页", fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    sizes = [scanned, img_text, rendered]
    names = ["整页扫描图·无文字层", "整页扫描图·有文字层", "按 DPI 渲染页"]
    colors = ["#c0504d", "#f0b27a", "#9bbb59"]
    wedges, _, autotexts = ax2.pie(
        sizes, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 20 else "", startangle=90,
        textprops={"fontsize": 8},
    )
    ax2.legend(wedges, [f"{n} {s} 页" for n, s in zip(names, sizes)],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=7.2, frameon=False)
    ax2.set_title("页面构成", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_1_dataset.png", dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"数据集：{len(rows)} 份 / {total} 页 / 扫描无文字 {scanned} / 扫描有文字 {img_text} / 渲染页 {rendered}")


if __name__ == "__main__":
    fig_architecture()
    fig_graph()
    fig_retrieval()
    fig_split()
    fig_dataset()
    print("输出目录", OUT)
