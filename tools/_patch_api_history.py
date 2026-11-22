"""给 HTTP 接口补三个只读的历史/统计路由，供 Web 端渲染"审查记录"。

为什么走 store.py 而不是重算：Web 端要能翻历史（谁审的、多少分、几条风险），
重算一遍既慢又不一致。store 是 SQLite 单文件，读它最便宜。

中文文件必须用 Python 读写。
"""

from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "api.py"

HANDLERS = '''

def handle_history(ctx: RequestContext) -> dict[str, Any]:
    """审查历史（分页）。库不存在时返回空表而不是 500 —— 首次启动就是这个状态。"""
    from .store import default_store

    try:
        store = default_store(ctx.config)
    except Exception as exc:  # noqa: BLE001 - 库打不开也要给前端一个可用响应
        return {"items": [], "total": 0, "page": 1, "size": 0, "note": f"存储不可用：{exc}"}
    query = ctx.query
    page = int((query.get("page") or ["1"])[0] or 1)
    size = int((query.get("size") or ["20"])[0] or 20)
    code = (query.get("code") or [""])[0]
    severity = (query.get("severity") or [""])[0]
    try:
        rows = store.list_contracts(page=page, size=size, code=code, severity=severity)
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "total": 0, "page": page, "size": size, "note": f"查询失败：{exc}"}
    items = [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in rows]
    return {"items": items, "total": len(items), "page": page, "size": size, "trace_id": ctx.trace_id}


def handle_history_detail(ctx: RequestContext) -> dict[str, Any]:
    """单份记录的风险明细（含引用条文）。"""
    from .store import default_store

    contract_id = ctx.params.get("contract_id", "")
    try:
        store = default_store(ctx.config)
        risks = store.get_risks(contract_id)
    except Exception as exc:  # noqa: BLE001
        return {"contract_id": contract_id, "risks": [], "note": f"读取失败：{exc}"}
    return {
        "contract_id": contract_id,
        "risks": [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in risks],
        "trace_id": ctx.trace_id,
    }


def handle_store_stats(ctx: RequestContext) -> dict[str, Any]:
    """全库统计：合同数、风险数、严重度分布、高频风险码。"""
    from .store import default_store

    try:
        stats = default_store(ctx.config).stats()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"存储不可用：{exc}"}
    payload = stats.as_dict() if hasattr(stats, "as_dict") else dict(stats)
    payload["available"] = True
    payload["trace_id"] = ctx.trace_id
    return payload

'''

ROUTES_OLD = '''    Route("GET", r"^/api/jobs/(?P<job_id>[0-9a-f]+)$", handle_job, "任务状态"),
]'''
ROUTES_NEW = '''    Route("GET", r"^/api/jobs/(?P<job_id>[0-9a-f]+)$", handle_job, "任务状态"),
    Route("GET", r"^/api/history$", handle_history, "审查历史（分页）"),
    Route("GET", r"^/api/history/(?P<contract_id>[^/]+)$", handle_history_detail, "单份历史明细"),
    Route("GET", r"^/api/stats$", handle_store_stats, "全库统计"),
]'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    original = text

    anchor = "\n\nROUTES: list[Route] = ["
    assert anchor in text, "ROUTES 锚点未命中"
    text = text.replace(anchor, HANDLERS + "\nROUTES: list[Route] = [", 1)

    assert ROUTES_OLD in text, "路由表未命中"
    text = text.replace(ROUTES_OLD, ROUTES_NEW, 1)

    if text == original:
        print("没有改动")
        return 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"已改写 {TARGET}（{len(original.splitlines())} → {len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
