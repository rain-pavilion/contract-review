"""修 /api/history：把 store.list_contracts 的返回值当列表用。

`list_contracts()` 真实签名是 `-> tuple[list[StoredContract], int]`，
元组的第二个元素是总数。我当初直接 `for row in rows` 遍历，第二个元素是个 int，
`dict(row)` 就抛 `TypeError: cannot convert dictionary update sequence element #0 to a sequence`。
"""

from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "api.py"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    original = text

    old = """    try:
        rows = store.list_contracts(page=page, size=size, code=code, severity=severity)
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "total": 0, "page": page, "size": size, "note": f"查询失败：{exc}"}
    items = [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in rows]
    return {"items": items, "total": len(items), "page": page, "size": size, "trace_id": ctx.trace_id}"""
    new = """    try:
        # list_contracts() 返回的是 (rows, total) 元组，不是裸列表
        rows, total = store.list_contracts(page=page, size=size, code=code, severity=severity)
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "total": 0, "page": page, "size": size, "note": f"查询失败：{exc}"}
    items = [row.as_dict() if hasattr(row, "as_dict") else dict(row) for row in rows]
    return {"items": items, "total": int(total), "page": page, "size": size, "trace_id": ctx.trace_id}"""
    assert old in text, "handle_history 未命中"
    text = text.replace(old, new, 1)

    if text == original:
        print("没有改动")
        return 1
    TARGET.write_text(text, encoding="utf-8")
    print("已修 /api/history 的返回解包")
    return 0


if __name__ == "__main__":
    sys.exit(main())
