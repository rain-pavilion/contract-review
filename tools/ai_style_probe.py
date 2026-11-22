"""AI 文风探针：从论文生成器里抽出所有正文段，量化"机器味"指标并支持前后对比。

为什么不用 docx 或 txt：那两个都混了标题、表题、代码、参考文献，会把指标带偏。
这里直接解析 `tools/build_thesis_v2.py` 的 AST，只取 `P(...)`（正文段）与两个摘要常量。

用法：
    python tools/ai_style_probe.py                 # 打印当前指标
    python tools/ai_style_probe.py --save base.json # 存基线
    python tools/ai_style_probe.py --compare base.json  # 与基线对比并判定

判据（改写目标，均按"每千汉字"或占比计，避免篇幅变化干扰）：
    句长变异系数 ≥ 0.55；句长标准差 ≥ 24；短句占比 ≥ 0.14；极短句占比 ≥ 0.06；
    长句占比 ≤ 0.16；段长标准差 ≥ 90；短段占比 ≥ 0.18；
    套话合计 ≤ 6 次；抽象开头段 ≤ 6 段；评价性三连 ≤ 8 次。

为什么看变异系数（std/mean）而不只看标准差：标准差会随均值漂移，换一段文字就可能
"虚高"；CV 除掉了均值，才真正反映"长短起伏"这一维——AIGC 检测里叫突发度。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "build_thesis_v2.py"

# 机器味最重的套话（学位论文里能不写就不写）
CLICHES = [
    "首先", "其次", "最后", "一方面", "另一方面", "不仅", "而且",
    "综上所述", "综上", "值得注意", "至关重要", "不可或缺", "有效地",
    "具有重要", "发挥重要作用", "起到关键作用", "在当今", "随着.*的发展",
    "为.*提供了新思路", "奠定了坚实的基础",
]
# 以这些词开头的小节，往往是"抽象主语 + 空转"的典型
WEAK_OPENERS = ["通过", "基于", "针对", "鉴于", "根据"]

# 评价性形容词：只有"评价词"连成三项才算机器味；"解析、切分、检索"这类技术列举属必要内容，不计
EVAL_WORDS = [
    "高效", "准确", "可靠", "稳定", "灵活", "可扩展", "易用", "安全", "智能", "全面",
    "深入", "丰富", "显著", "优异", "先进", "合理", "科学", "规范", "完善", "健全",
    "流畅", "便捷", "清晰", "直观", "低成本", "高可用", "鲁棒", "轻量", "简洁", "良好",
]


def _const_str(node: ast.AST) -> str | None:
    """把 AST 节点还原成字符串（支持隐式串联、f-string 之外的常量）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, (ast.List, ast.Tuple)):
        parts = [_const_str(e) for e in node.elts]
        if all(p is not None for p in parts):
            return "".join(parts)  # type: ignore[arg-type]
    return None


def collect_paragraphs(path: Path) -> list[str]:
    """取 P(...) 的正文文本 + ABSTRACT_CN / ABSTRACT_EN / ACK_TEXT。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "P" and node.args:
            text = _const_str(node.args[0])
            if text:
                out.append(text.strip())
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in ("ABSTRACT_CN", "ABSTRACT_EN", "ACK_TEXT"):
                text = _const_str(node.value)
                if text:
                    out.append(text.strip())
    # f-string 拼进去的正文段（含变量）单独再捞一遍，取其中的字面量部分
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "P" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.JoinedStr):
                literal = "".join(v.value for v in arg.values
                                  if isinstance(v, ast.Constant) and isinstance(v.value, str))
                if len(literal) > 60:
                    out.append(literal.strip())
    return out


def sentences(paragraphs: list[str]) -> list[str]:
    sents: list[str] = []
    for para in paragraphs:
        for chunk in re.split(r"[。！？；]", para):
            chunk = chunk.strip()
            if len(chunk) >= 4:
                sents.append(chunk)
    return sents


def count_eval_triads(text: str) -> int:
    """只数"评价性形容词"的三连（如"高效、准确、可靠"）。

    技术名词列举（"解析、切分、抽取"）是必要内容，不算机器味。
    """
    hits = 0
    pattern = re.compile(r"[\u4e00-\u9fa5]{2,6}、[\u4e00-\u9fa5]{2,6}、[\u4e00-\u9fa5]{2,6}")
    for m in pattern.finditer(text):
        items = [x for x in m.group(0).split("、") if x]
        if sum(1 for it in items if it in EVAL_WORDS) >= 2:
            hits += 1
    return hits


def measure(paragraphs: list[str]) -> dict:
    paras = [p for p in paragraphs if len(p) >= 40]
    sents = sentences(paras)
    plen = [len(p) for p in paras]
    slen = [len(s) for s in sents]
    cjk = sum(len(re.findall(r"[\u4e00-\u9fa5]", p)) for p in paras) or 1
    per_k = lambda n: round(n * 1000 / cjk, 2)

    cliche_hits = sum(len(re.findall(pat, "\n".join(paras))) for pat in CLICHES)
    weak_open = sum(1 for p in paras if any(p.startswith(w) for w in WEAK_OPENERS))
    triads = count_eval_triads("\n".join(paras))
    short_ratio = sum(1 for s in slen if s < 20) / max(len(slen), 1)
    tiny_ratio = sum(1 for s in slen if s < 15) / max(len(slen), 1)
    long_ratio = sum(1 for s in slen if s > 80) / max(len(slen), 1)
    short_para = sum(1 for n in plen if n < 140) / max(len(plen), 1)
    cv = round(statistics.pstdev(slen) / statistics.mean(slen), 3) if slen and statistics.mean(slen) else 0.0

    return {
        "段落数": len(paras),
        "汉字数": cjk,
        "句数": len(slen),
        "段长均值": round(statistics.mean(plen), 1) if plen else 0,
        "段长标准差": round(statistics.pstdev(plen), 1) if len(plen) > 1 else 0,
        "句长均值": round(statistics.mean(slen), 1) if slen else 0,
        "句长标准差": round(statistics.pstdev(slen), 1) if len(slen) > 1 else 0,
        "句长变异系数": cv,
        "短句占比": round(short_ratio, 3),
        "极短句占比": round(tiny_ratio, 3),
        "长句占比": round(long_ratio, 3),
        "短段占比": round(short_para, 3),
        "套话次数": cliche_hits,
        "抽象开头段数": weak_open,
        "四字并列次数": triads,
        "通过_每千字": per_k(len(re.findall("通过", "\n".join(paras)))),
        "本文_每千字": per_k(len(re.findall("本文", "\n".join(paras)))),
        "破折号与括号": len(re.findall(r"[—（）()]", "\n".join(paras))),
    }


TARGETS = {
    "句长变异系数": (">=", 0.55),
    "句长标准差": (">=", 24.0),
    "短句占比": (">=", 0.14),
    "极短句占比": (">=", 0.06),
    "长句占比": ("<=", 0.16),
    "段长标准差": (">=", 90.0),
    "短段占比": (">=", 0.18),
    "套话次数": ("<=", 6),
    "抽象开头段数": ("<=", 6),
    "四字并列次数": ("<=", 8),
}


def check(m: dict) -> list[tuple[str, str, bool]]:
    rows = []
    for key, (op, limit) in TARGETS.items():
        value = m.get(key)
        if value is None:
            continue
        ok = value >= limit if op == ">=" else value <= limit
        rows.append((key, f"{value} {op} {limit}", ok))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="论文 AI 文风探针")
    ap.add_argument("--save", help="把本次指标写到 JSON（作基线）")
    ap.add_argument("--compare", help="与基线 JSON 对比并判定")
    ap.add_argument("--generator", default=str(GENERATOR))
    args = ap.parse_args(argv)

    paras = collect_paragraphs(Path(args.generator))
    if not paras:
        print("[×] 没抽到正文段，检查生成器的 P(...) 调用是否被改写")
        return 1
    m = measure(paras)

    print(f"正文段 {m['段落数']} 段 / {m['汉字数']} 汉字 / {m['句数']} 句\n")
    for k, v in m.items():
        if k not in ("段落数", "汉字数", "句数"):
            print(f"  {k:12s} {v}")

    print("\n对照目标：")
    rows = check(m)
    for key, expr, ok in rows:
        print(f"  [{'✓' if ok else '×'}] {key:12s} {expr}")

    if args.save:
        Path(args.save).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已存 {args.save}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print(f"\n与基线 {args.compare} 对比：")
        for k in m:
            if k in base and k not in ("段落数", "汉字数", "句数"):
                delta = m[k] - base[k] if isinstance(m[k], (int, float)) else 0
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "＝")
                print(f"  {k:12s} {base[k]} → {m[k]}  {arrow}{abs(delta)}")
        failed = [k for k, _, ok in rows if not ok]
        if failed:
            print(f"\n[!] 仍未达标：{', '.join(failed)}")
        else:
            print("\n[✓] 全部目标达成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
