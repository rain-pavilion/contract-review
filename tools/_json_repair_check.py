"""验证 JSON 修补：小模型写坏的结构能不能救回来。"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from zhilv_qiling_v2.llm_client import _extract_json, normalize_payload  # noqa: E402

CASES = {
    "裸键名": '{ "code": "paymentUnclear", "severity": "medium", message": "付款条件不明确" }',
    "数组缺闭合": '{"defects": [{"code": "vague_delivery", "message": "交付时间模糊", "quote": "及时"}]',
    "裸键+缺闭合": (
        '```json\n{ "defects": [ { code: "vague_delivery", severity: "medium", '
        'message: "交付时间模糊" } ] }\n```'
    ),
    "单引号": "{'defects': [{'code': 'zero_rent', 'message': '租金为零'}]}",
    "尾逗号": '{"defects": [{"code": "zero_rent", "message": "租金为零"},]}',
    "对象包数组": '{"defects": {"code": "zero_rent", "message": "租金为零"}}',
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for name, text in CASES.items():
        try:
            content = normalize_payload(_extract_json(text))
            print(f"OK   {name}: {content}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
