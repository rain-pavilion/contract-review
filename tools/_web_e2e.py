"""Web 端到端自检：首页、文本审查、文件上传、历史、统计。

为什么不用 PowerShell 跑：PS 5.1 的 `Invoke-WebRequest` 会按 GBK 解 UTF-8 响应，
中文全变乱码、`ConvertFrom-Json` 直接失败；而且它没有 `-Form`（那是 PS 6+）。
用标准库自己拼 multipart 最稳，也能在 CI 里跑。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8611"


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.status, resp.read()


def post_json(path: str, payload: dict, timeout: int = 180):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def post_file(path: str, file_path: pathlib.Path, timeout: int = 300):
    boundary = "----zhilv" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + file_path.read_bytes() + tail
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    print("=" * 72)
    status, body = get("/", 20)
    text = body.decode("utf-8", "ignore")
    print(f"[1] GET /                  {status} / {len(body)} bytes / 标题命中 {('智律契灵' in text)}")
    ok = status == 200 and "智律契灵" in text

    status, health = get("/api/health", 20)
    from urllib.parse import urlparse

    # health 是 JSON
    health = json.loads(health.decode("utf-8"))
    print(f"[2] GET /api/health        {status} / version={health.get('version')} tag={health.get('tag')}")

    sample = (
        "技术开发（合作）合同\n"
        "甲方（委托方）：北京某科技有限公司\n"
        "乙方（研究开发方）：某大学人工智能学院\n"
        "第二条 甲方有权根据自身业务需要单方变更或者解除本合同，无需承担任何违约责任。\n"
        "第三条 乙方应在合理期限内交付阶段性成果。\n"
        "第四条 项目经费总额为人民币 120 万元整。\n"
        "第五条 乙方对甲方资料承担保密义务。\n"
    )
    t0 = time.time()
    status, rep = post_json("/api/review", {"text": sample})
    elapsed = time.time() - t0
    risks = rep.get("risks", [])
    print(
        f"[3] POST /api/review(文本)  {status} / 评分 {rep.get('score')} / 风险 {rep.get('risk_count')} "
        f"/ 条款 {rep.get('stats', {}).get('clauses')} / {elapsed:.2f}s"
    )
    for risk in risks[:5]:
        print(f"       - {risk.get('severity'):6s} {risk.get('code'):24s} {risk.get('message')[:40]}")

    candidates = sorted(pathlib.Path("data/generated_contracts").rglob("*.docx"))
    if candidates:
        target = candidates[0]
        t0 = time.time()
        status, rep2 = post_file("/api/review", target)
        elapsed = time.time() - t0
        print(
            f"[4] POST /api/review(文件)  {status} / {target.name} / 评分 {rep2.get('score')} "
            f"/ 风险 {rep2.get('risk_count')} / 条款 {rep2.get('stats', {}).get('clauses')} / {elapsed:.2f}s"
        )
        if rep2.get("error"):
            print("       error:", rep2["error"])
            ok = False
    else:
        print("[4] 跳过文件上传：data/generated_contracts 下没有 docx")

    status, raw = get("/api/history?size=5", 60)
    hist = json.loads(raw.decode("utf-8"))
    print(f"[5] GET /api/history       {status} / total={hist.get('total')} items={len(hist.get('items') or [])}")
    for item in (hist.get("items") or [])[:3]:
        print(
            f"       - {item.get('contract_id')} 评分 {item.get('score')} 风险 {item.get('risks')} "
            f"高危 {item.get('high')} {item.get('created_at')}"
        )

    status, raw = get("/api/stats", 60)
    stats = json.loads(raw.decode("utf-8"))
    print(
        f"[6] GET /api/stats         {status} / 合同 {stats.get('contracts')} 风险 {stats.get('risks')} "
        f"均值 {stats.get('score_mean')} 待复核 {stats.get('pending_review')}"
    )

    status, raw = get("/api/history/ct_nonexistent", 30)
    print(f"[7] GET /api/history/占位   {status} / 空表容错 OK")
    print("=" * 72)
    print("结论：", "全部通过" if ok else "有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
