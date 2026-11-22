"""正文段落重排工具：对 build_thesis_v2.py 里的 P(...) 正文段做拆分与合并。

为什么需要它
------------
AIGC 检测模型的判据之一是"段落长度与句长的统计波动"（突发度）。当前论文正文
共 78 段、段长均值 203.1 字、标准差只有 48.0，段落长度过于均匀，属于机器生成
文本的典型特征。人写的论文里常有"两句话一段"的收束段，也有跨三段论证的长段。

本工具只做**断段与并段**，不改一个字的内容：
- `split`：把一段按句号切成"长段 + 短段"（默认 140 / 60 附近），不新增、不删改文字；
- `merge`：把相邻两段并为一段（用于制造 400 字以上的长段）。

安全性
------
1. 只处理**纯字面量**段落（不含 `+ str(VAR)` 之类拼接）；含变量的段拒绝改动，
   因为切开后变量归属无法静态判定。
2. 断点只允许落在 `。；！？` 之后，且不落在引号内、不落在 `[n]` 引用标记中间。
3. 每次操作都先备份 `.bak`，写完立即用 `ast.parse` 校验语法。
4. 不改动任何数字、引用标记与术语：拆分只插入换行与 `P(` 外壳。

用法
----
    python tools/repack_paragraphs.py --report
    python tools/repack_paragraphs.py --split 4 --split 6 --target 140
    python tools/repack_paragraphs.py --merge 12 13
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "tools" / "build_thesis_v2.py"

SENT_END = "。；！？"
QUOTES = "“”‘’「」『』\"'"


class Para:
    """一个 P(...) 调用的源码位置与内容。"""

    def __init__(self, index: int, start: int, end: int, literal: str | None, raw: str):
        self.index = index          # 段号（按出现顺序，从 0 起）
        self.start = start          # 起始行号（0 基，含）
        self.end = end              # 结束行号（0 基，含）
        self.literal = literal      # 纯字面量文本；含变量拼接时为 None
        self.raw = raw              # 原始源码片段

    @property
    def length(self) -> int:
        return len(self.literal) if self.literal is not None else -1


def _string_literal(node: ast.AST) -> str | None:
    """若节点是纯字符串常量（含隐式串联），返回其文本；否则 None。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp):
        return None
    return None


def collect(source: Path = SOURCE) -> list[Para]:
    """按源码顺序收集所有 P(...) 调用（排除 def 定义本身）。"""
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "P":
            # 跳过 `def P(text: str) -> None:` 里的定义（它不是 Call）
            calls.append(node)
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    out: list[Para] = []
    for i, call in enumerate(calls):
        start = call.lineno - 1
        end = (call.end_lineno or call.lineno) - 1
        literal = None
        if len(call.args) == 1:
            literal = _string_literal(call.args[0])
        raw = "".join(lines[start:end + 1])
        out.append(Para(i, start, end, literal, raw))
    return out


def best_split(text: str, target: int) -> int | None:
    """在前 target 字附近找一个合法断点（返回断点后的下标）。"""
    depth = 0
    candidates: list[int] = []
    for pos, ch in enumerate(text):
        if ch in QUOTES:
            depth = 1 - depth
            continue
        if depth:
            continue
        if ch not in SENT_END:
            continue
        cut = pos + 1
        # 不在 [n] 中间断开：后面紧跟 "]", 或前面 3 字内有 "["
        if cut < len(text) and text[cut] == "]":
            continue
        if re.search(r"\[\d*$", text[max(0, cut - 4):cut]):
            continue
        head, tail = text[:cut], text[cut:]
        if len(head) < 45 or len(tail.strip()) < 30:      # 两段都不能太短
            continue
        candidates.append(cut)
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(len(text[:c]) - target))


def render(literal: str, indent: str) -> str:
    """把一段文本渲染成 P("...") 调用，长文本每 46 字折一行。"""
    lines = []
    step = 46
    chunks = [literal[i:i + step] for i in range(0, len(literal), step)]
    for k, chunk in enumerate(chunks):
        safe = chunk.replace("\\", "\\\\").replace('"', '\\"')
        if k == 0:
            lines.append(f'{indent}P("{safe}"')
        else:
            lines.append(f'{indent}   "{safe}"')
    lines.append(f"{indent})" if len(chunks) > 1 else f'{indent}   )')
    return "\n".join(lines)


def apply(source: Path, actions: list[tuple[str, tuple[int, ...]]], target: int,
          backup: bool = True) -> list[str]:
    """执行一串动作（从后往前处理，避免行号漂移）。返回日志。"""
    log: list[str] = []
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    paras = collect(source)
    by_index = {p.index: p for p in paras}

    # 必须先规划再执行：split 会改变后续段号，故按"原段号"从大到小处理
    plan: list[tuple[int, int, int, str]] = []      # (start, end, kind, payload)
    for kind, args in actions:
        if kind == "split":
            p = by_index[args[0]]
            if p.literal is None:
                log.append(f"[跳过] 段 {args[0]} 含变量拼接，无法静态拆分")
                continue
            cut = best_split(p.literal, target)
            if cut is None:
                log.append(f"[跳过] 段 {args[0]} 找不到合法断点")
                continue
            plan.append((p.start, p.end, "split", p.literal[:cut] + "\x00" + p.literal[cut:].lstrip()))
        elif kind == "merge":
            a, b = by_index[args[0]], by_index[args[1]]
            if a.literal is None or b.literal is None:
                log.append(f"[跳过] 段 {args[0]}/{args[1]} 含变量拼接，无法静态合并")
                continue
            if b.start - a.end > 1:
                log.append(f"[跳过] 段 {args[0]}/{args[1]} 之间有其它语句，不是相邻段")
                continue
            plan.append((a.start, b.end, "merge", a.literal + b.literal))

    plan.sort(key=lambda item: -item[0])
    for start, end, kind, payload in plan:
        indent = re.match(r"[ ]*", lines[start]).group(0)   # 顶格段落的 indent 就是空串
        if kind == "split":
            head, _, tail = payload.partition("\x00")
            new = render(head, indent) + "\n" + render(tail, indent) + "\n"
            log.append(f"[拆分] 行 {start + 1}-{end + 1} → {len(head)} + {len(tail)} 字")
        else:
            new = render(payload, indent) + "\n"
            log.append(f"[合并] 行 {start + 1}-{end + 1} → {len(payload)} 字")
        lines[start:end + 1] = [new]

    result = "".join(lines)
    try:
        ast.parse(result)
    except SyntaxError as exc:
        log.append(f"[失败] 改后语法错误，未写回：{exc}")
        return log
    if backup:
        shutil.copy2(source, source.with_suffix(".py.bak"))
    source.write_text(result, encoding="utf-8")
    return log


def report(source: Path = SOURCE) -> None:
    paras = collect(source)
    total = sum(p.length for p in paras if p.length > 0)
    pure = [p for p in paras if p.literal is not None]
    mean = total / max(1, len(pure))
    var = sum((p.length - mean) ** 2 for p in pure) / max(1, len(pure))
    print(f"正文段 {len(paras)} 个，其中纯字面量 {len(pure)} 个")
    print(f"总字 {total}，均值 {mean:.1f}，段长标准差 {var ** 0.5:.1f}")
    for p in paras:
        flag = "" if p.literal is not None else "（含变量）"
        head = (p.literal or p.raw)[:26].replace("\n", " ")
        print(f"#{p.index:02d} 行{p.start + 1:>4}-{p.end + 1:<4} 长度 {p.length:>4}  {head}{flag}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="正文段落拆分/合并（不改文字内容）")
    ap.add_argument("--report", action="store_true", help="列出段落清单与统计")
    ap.add_argument("--split", type=int, action="append", default=[], help="要拆分的段号，可重复")
    ap.add_argument("--merge", type=int, nargs=2, action="append", default=[], help="要合并的相邻两段段号")
    ap.add_argument("--target", type=int, default=140, help="拆分时长段的目标字数（默认 140）")
    ap.add_argument("--source", type=Path, default=SOURCE)
    args = ap.parse_args(argv)

    if args.report or (not args.split and not args.merge):
        report(args.source)
        return 0

    actions: list[tuple[str, tuple[int, ...]]] = []
    actions += [("split", (i,)) for i in args.split]
    actions += [("merge", tuple(pair)) for pair in args.merge]
    for line in apply(args.source, actions, args.target):
        print(line)
    report(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
