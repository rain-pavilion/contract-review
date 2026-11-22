"""修两处：bench.py 里嵌套 ASCII 双引号的字符串、cli.py 里不存在的 _config_from。

中文项目里用 `python -c` 传多行代码很容易被 shell/heredoc 咬坏，所以固定用
本文件（write 工具落盘）再执行。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(".")
LQ, RQ = chr(0x3010), chr(0x3011)  # 【 】，用来替代内层双引号

# ---------- bench.py ----------
bench_path = ROOT / "zhilv_qiling_v2" / "bench.py"
text = bench_path.read_text(encoding="utf-8")
lines = text.splitlines()
start = None
for index, line in enumerate(lines):
    if "payload[" in line and "note" in line and "=" in line and "FP" not in line:
        start = index
        break
if start is None:
    raise SystemExit("找不到 note 赋值")

# 该块形如：
#     payload["note"] = (
#         "FP 里…"…",
#         "论文与答辩…",
#     )
end = start
while end < len(lines) and ")" not in lines[end]:
    end += 1
if end >= len(lines):
    raise SystemExit("note 块没有闭合")

new_block = [
    '    payload["note"] = (',
    f'        "FP 里有一部分是{LQ}检出但金标未标{RQ}：对抗集只标主注入缺陷，'
    '模板共享句会被多份合同同时触发，"',
    '        "论文与答辩应把这一栏单列，不当作纯粹的错误率。"',
    "    )",
]
lines[start : end + 1] = new_block
bench_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("bench.py note 块已重写")

# ---------- cli.py ----------
cli_path = ROOT / "zhilv_qiling_v2" / "cli.py"
cli = cli_path.read_text(encoding="utf-8")
count = cli.count("_config_from(args)")
cli = cli.replace("_config_from(args)", "config_from_env()")
if "config_from_env" in cli and "from .config import" in cli:
    # 确保 config_from_env 在导入列表里
    import re

    match = re.search(r"from \.config import ([^\n]+)", cli)
    if match and "config_from_env" not in match.group(1):
        names = [name.strip() for name in match.group(1).split(",")]
        names.append("config_from_env")
        cli = cli.replace(match.group(0), "from .config import " + ", ".join(sorted(set(names))))
cli_path.write_text(cli, encoding="utf-8")
print(f"cli.py 替换 {count} 处 _config_from")
