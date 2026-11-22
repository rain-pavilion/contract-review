"""修 _engine_config：RequestContext.query 的值是 list（parse_qs 的产物）。"""

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "api.py"
s = p.read_text(encoding="utf-8")

old = """    raw = ""
    try:
        raw = str((ctx.query or {}).get("engine", "") or "")
    except Exception:
        raw = \"\"\""""
new = """    raw = ""
    try:
        # ctx.query 来自 parse_qs：值是 list，直接 str() 会得到 \"['multi']\"
        values = (ctx.query or {}).get("engine") or []
        raw = str(values[0]) if values else ""
    except Exception:
        raw = \"\""""
assert old in s, "锚点没找到"
p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("_engine_config 已按 list 取值")
