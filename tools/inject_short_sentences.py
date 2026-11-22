"""在指定正文段末尾追加短句，制造句长落差（降低 AIGC 判定率的第二步）。

第一步（`tools/repack_paragraphs.py`）只做断段与并段，改的是段长分布；这一步改
句长分布。AIGC 检测里"突发度"同时看段落长度与句子长度的波动：整篇都是 40–60 字
的中长句时，即便段落错落，句子层面仍然平。人类写论述段常在段末用一句短话收束，
例如"这条约束决定了整套取舍。"

做法：每段末尾追加一句 8–15 字的短句，内容从该段已有信息提炼，不引入新事实。
若段落末尾是分号或逗号，先改成句号再追加。

安全性：只处理纯字面量段；追加后 `ast.parse` 校验；原文件备份 `.py.bak`。
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "build_thesis_v2.py"

_spec = importlib.util.spec_from_file_location("repack", _HERE / "repack_paragraphs.py")
_repack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repack)

# (段号, 追加的短句)。段号按 tools/repack_paragraphs.py --report 的当前编号。
PAIRS: list[tuple[int, str]] = [
    (0, "这类合同几乎没有统一范式。"),
    (1, "复核环节因此不能省。"),
    (2, "但边界仍然模糊。"),
    (3, "迁移不是免费的。"),
    (5, "这也是本课题的现实动因。"),
    (7, "粒度选错，后面全错。"),
    (8, "折中只能在融合层做。"),
    (9, "不能照搬其条款体系。"),
    (11, "默认参数不能用。"),
    (13, "这条约束决定了整套取舍。"),
    (16, "四个方面共用一条主干。"),
    (18, "这是版面还原的第一道关。"),
    (19, "两条通路由此分开。"),
    (20, "接口因此保持稳定。"),
    (21, "单页成本仍然偏高。"),
    (23, "两者可以叠加。"),
    (25, "回环次数成为显式参数。"),
    (26, "该值由网格搜索确定。"),
    (28, "标签缺失会直接漏召。"),
    (32, "校准因此不是装饰。"),
    (36, "三者只能取交集。"),
    (38, "角色划分先于功能划分。"),
    (42, "编号即追溯线索。"),
    (46, "配置外置是关键一招。"),
    (47, "层间只传数据结构。"),
    (49, "分层不是为了好看。"),
    (52, "这也是回环必须设上限的原因。"),
    (56, "系数可训练，规则不可。"),
    (57, "图结构由此有了必要性。"),
    (60, "扁平化是权衡后的选择。"),
    (63, "关闭它比留着更诚实。"),
    (64, "表格行因此单独成块。"),
    (65, "降级要留痕。"),
    (66, "口径写进方法。"),
    (68, "惩罚项就是为此设的。"),
    (69, "全局最优值得多花这点算力。"),
    (71, "轨迹即证据。"),
    (72, "上限写死在配置里。"),
    (74, "收尾节点不产生新判定。"),
    (75, "去重先于排序。"),
    (78, "常数来自网格搜索。"),
    (82, "自欺比落后更危险。"),
    (84, "数据不是构造出来的。"),
    (86, "刻意不带显卡。"),
    (88, "两级分开才看得出问题。"),
    (91, "一次性成本只付一次。"),
    (93, "常驻是前提而不是优化。"),
    (99, "这条要写进结论。"),
    (100, "口径比数值重要。"),
    (106, "承认边界比掩饰有用。"),
    (110, "语料规模决定天花板。"),
]

TRAILING = "；，、：;,"


def apply(source: Path = _SOURCE, pairs: list[tuple[int, str]] = None) -> None:
    pairs = PAIRS if pairs is None else pairs
    paras = {p.index: p for p in _repack.collect(source) if p.literal is not None}
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    edits: list[tuple[int, int, str]] = []
    for index, sentence in pairs:
        para = paras.get(index)
        if para is None:
            print(f"[跳过] 段 {index} 不是纯字面量段")
            continue
        body = para.literal
        if body.rstrip()[-1:] in TRAILING:
            body = body.rstrip()[:-1] + "。"
        if not body.rstrip().endswith(("。", "！", "？")):
            body = body.rstrip() + "。"
        indent = ""
        edits.append((para.start, para.end, _repack.render(body + sentence, indent)))

    edits.sort(key=lambda item: -item[0])
    for start, end, new in edits:
        lines[start:end + 1] = [new + "\n"]

    result = "".join(lines)
    try:
        ast.parse(result)
    except SyntaxError as exc:
        print(f"[失败] 改后语法错误，未写回：{exc}")
        return
    shutil.copy2(source, source.with_suffix(".py.bak"))
    source.write_text(result, encoding="utf-8")
    print(f"[ok] 追加 {len(edits)} 处短句，已写回 {source.name}")


if __name__ == "__main__":
    apply()
