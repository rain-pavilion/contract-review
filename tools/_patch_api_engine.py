"""给 /api/review 与 /api/batch 加 engine=multi 开关（多智能体编排）。

支持三种写法：query ?engine=multi、JSON body {"engine":"multi"}、multipart 字段 engine=multi。
"""

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "api.py"
s = p.read_text(encoding="utf-8")

helper = '''
_ENGINE_ALIASES = {"single": False, "pipeline": False, "multi": True, "goa": True, "agents": True}


def _engine_config(ctx: "RequestContext", config):
    """按请求里的 engine 参数决定走单进程链路还是多智能体编排。

    多智能体要一个新 config（`enable_multi_agent` 是构造期字段语义），
    直接改传入对象会污染服务端常驻配置——并发请求之间会互相串。
    """
    from dataclasses import replace

    raw = ""
    try:
        raw = str((ctx.query or {}).get("engine", "") or "")
    except Exception:
        raw = ""
    if not raw and isinstance(ctx.json_body, dict):
        raw = str(ctx.json_body.get("engine") or "")
    raw = raw.strip().lower()
    if raw not in _ENGINE_ALIASES:
        return config
    try:
        return replace(config, enable_multi_agent=_ENGINE_ALIASES[raw])
    except TypeError:
        config.enable_multi_agent = _ENGINE_ALIASES[raw]
        return config

'''

anchor = "def handle_review(ctx: RequestContext) -> dict[str, Any]:"
assert anchor in s, "handle_review 锚点没找到"
s = s.replace(anchor, helper.lstrip("\n") + "\n" + anchor, 1)

old = """    config = ctx.config
    fields, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
    if files:"""
new = """    config = _engine_config(ctx, ctx.config)
    fields, files = parse_multipart(ctx.body, ctx.handler.headers.get("Content-Type", ""))
    if fields.get("engine"):
        from dataclasses import replace as _replace

        try:
            config = _replace(config, enable_multi_agent=str(fields["engine"]).lower() in ("multi", "goa", "agents"))
        except TypeError:
            pass
    if files:"""
assert old in s, "handle_review body 锚点没找到"
s = s.replace(old, new, 1)

old_batch = """def handle_batch(ctx: RequestContext) -> dict[str, Any]:
    from .pipeline import batch_review, dataset_files

    config = ctx.config"""
new_batch = """def handle_batch(ctx: RequestContext) -> dict[str, Any]:
    from .pipeline import batch_review, dataset_files

    config = _engine_config(ctx, ctx.config)"""
assert old_batch in s, "handle_batch 锚点没找到"
s = s.replace(old_batch, new_batch, 1)

p.write_text(s, encoding="utf-8")
print("api.py: /api/review 与 /api/batch 已支持 engine=multi")
