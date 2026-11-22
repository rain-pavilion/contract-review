"""Crawl official model contracts plus regulator-written risk annotations.

Source: 全国合同示范文本库 (State Administration for Market Regulation,
https://htsfwb.samr.gov.cn/). Each template page publishes a contract text and a
set of "风险提示" items authored by the issuing regulator. That combination is a
usable external reference set of contract clauses + human review guidance; it is
NOT an independent legal-quality gold set and must not be reported as one.

Design notes
------------
* The site answers ``curl`` but rejects Python's default TLS stack on this host,
  so HTTP goes through the system ``curl`` binary (no browser automation, no
  captcha/login bypass).
* Enumeration uses the site's own public list API:
  ``/api/content/SearchTemplates?loc=false|true&p=N``.
* Each detail page is server-rendered. Text and annotations are parsed with the
  stdlib HTML parser; embedded base64 images are ignored. Raw bytes are hashed
  but not stored by default (pages are ~1 MB each, mostly inline images).
* Politeness: one request at a time, configurable delay, bounded retries, and a
  recorded robots/ToS note. No parallel hammering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SITE = "https://htsfwb.samr.gov.cn"
API = SITE + "/api/content/SearchTemplates"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CURL = "curl.exe" if os.name == "nt" else "curl"
SCHEMA = "zlql-external-contract-annotation-v1"

BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def curl_bytes(url: str, timeout: int = 45, retries: int = 2) -> tuple[int, bytes, str]:
    """Fetch one URL with the system curl; returns (status, body, error)."""
    last = ""
    for attempt in range(retries + 1):
        proc = subprocess.run(
            [CURL, "-sL", "--compressed", "--max-time", str(timeout),
             "-A", UA, "-w", "\n%{http_code}", url],
            capture_output=True, timeout=timeout + 20)
        if proc.returncode == 0 and proc.stdout:
            body = proc.stdout
            marker = body.rfind(b"\n")
            status_text = body[marker + 1:].strip() if marker >= 0 else b""
            try:
                status = int(status_text or b"0")
            except ValueError:
                status = 0
            return status, body[:marker] if marker >= 0 else body, ""
        last = (proc.stderr or b"").decode("utf-8", "replace").strip()[:300]
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return 0, b"", last or "curl returned no body"


def curl_json(url: str, timeout: int = 30) -> dict:
    status, body, error = curl_bytes(url, timeout=timeout, retries=2)
    if status != 200 or not body:
        raise RuntimeError(f"api failed: status={status} error={error} url={url}")
    return json.loads(body.decode("utf-8", "replace"))


def clean(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    lines = [line.strip() for line in value.split("\n")]
    return "\n".join(line for line in lines if line)


class _Frame:
    __slots__ = ("tag", "classes", "kind", "text")

    def __init__(self, tag: str, classes: set[str]) -> None:
        self.tag = tag
        self.classes = classes
        self.kind = ""
        self.text: list[str] = []


class DetailParser(HTMLParser):
    """Extract title, info pairs, contract text and risk-tip annotations.

    Text is appended to every open frame so nested inline tags keep their
    ancestor's reading order; a frame's ``kind`` decides what its text means
    when the frame closes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: dict[str, str] = {}
        self.contract_text = ""
        self.risk_items: list[dict] = []
        self.attachments: list[str] = []
        self._frames: list[_Frame] = []
        self._risk: dict | None = None
        self._pending_info_title: str | None = None
        self._skip_depth = 0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _classes(attrs) -> set[str]:
        for key, value in attrs:
            if key == "class" and value:
                return set(value.split())
        return set()

    def _nearest(self, kinds: tuple[str, ...]) -> _Frame | None:
        for frame in reversed(self._frames):
            if frame.kind in kinds:
                return frame
        return None

    def _newline_in_blocks(self) -> None:
        for frame in self._frames:
            if frame.kind in ("content", "risk_content"):
                frame.text.append("\n")

    def handle_starttag(self, tag: str, attrs) -> None:
        classes = self._classes(attrs)
        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value and re.search(r"\.(pdf|docx?|zip)(\?|$)", value, re.I):
                    self.attachments.append(value)
        frame = _Frame(tag, classes)
        if tag == "title":
            frame.kind = "page_title"
        elif tag == "div" and "samr-view-content" in classes:
            frame.kind = "content"
        elif tag == "div" and "samr-view-risk-item" in classes:
            frame.kind = "risk_item"
            self._risk = {"topic": "", "content": ""}
        elif tag == "div" and "info-title" in classes:
            frame.kind = "info_title"
        elif tag == "div" and "info-content" in classes:
            frame.kind = "info_content"
        elif tag == "div" and self._risk is not None:
            if "title" in classes:
                frame.kind = "risk_title"
            elif "content" in classes:
                frame.kind = "risk_content"
        if tag in BLOCK_TAGS:
            self._newline_in_blocks()
        self._frames.append(frame)

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag == "br":
            self._newline_in_blocks()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if not self._frames:
            return
        frame = self._frames.pop()
        if tag in BLOCK_TAGS:
            self._newline_in_blocks()
        kind = frame.kind
        text = clean("\n".join(frame.text))
        if kind == "page_title":
            self.title = text
        elif kind == "content":
            self.contract_text = text
        elif kind == "info_title":
            self._pending_info_title = text.rstrip("：:").strip()
        elif kind == "info_content":
            topic = self._pending_info_title or ""
            if topic:
                self.meta[topic] = text
            self._pending_info_title = None
        elif kind == "risk_title" and self._risk is not None:
            self._risk["topic"] = text
        elif kind == "risk_content" and self._risk is not None:
            self._risk["content"] = text
        elif kind == "risk_item":
            if self._risk and (self._risk["topic"] or self._risk["content"]):
                self.risk_items.append(dict(self._risk))
            self._risk = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        for frame in self._frames:
            frame.text.append(data)


def parse_detail(html_text: str) -> dict:
    parser = DetailParser()
    parser.feed(html_text)
    parser.close()
    # Finalize frames left open by malformed HTML.
    while parser._frames:
        parser.handle_endtag(parser._frames[-1].tag)
    text = parser.contract_text
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[A-Za-z0-9+/]{200,}={0,2}", "", text)
    return {"title": clean(parser.title).split(" - 合同示范文本库")[0].strip(),
            "meta": parser.meta, "contract_text": text,
            "annotations": parser.risk_items, "attachments": parser.attachments}


def enumerate_templates(scope: str, limit: int | None) -> list[dict]:
    loc = "true" if scope == "local" else "false"
    first = curl_json(f"{API}?loc={loc}&p=1")
    total_pages = int(first.get("TotalPage") or 1)
    rows = list(first.get("Data") or [])
    for page in range(2, total_pages + 1):
        try:
            rows += list(curl_json(f"{API}?loc={loc}&p={page}").get("Data") or [])
        except RuntimeError as error:
            print(f"[warn] page {page} failed: {error}", file=sys.stderr)
        time.sleep(0.4)
    if limit:
        rows = rows[:limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/annotation_samples/htsfwb"))
    parser.add_argument("--scope", choices=["national", "local", "all"], default="national")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--refresh", action="store_true", help="re-fetch existing docs")
    parser.add_argument("--save-raw", action="store_true", help="also store raw HTML (large)")
    args = parser.parse_args()
    out = args.out.resolve()
    contracts_dir = out / "contracts"
    raw_dir = out / "raw_html"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    scopes = ["national", "local"] if args.scope == "all" else [args.scope]
    index_rows: list[dict] = []
    failures: list[dict] = []
    started = now_utc()
    policy = {
        "site": SITE,
        "robots_txt": "empty/not provided at time of crawl; no allow/deny rules observed",
        "public_note": "官方说明：面向社会公众免费查阅下载；文本版权归发布机关。",
        "usage_note": ("仅用于项目内部标注对照与研究验收准备，不转售、不篡改原文；引用须注明来源。"
                       "风险提示为发布机关撰写的人工审查指引，不是独立法律质量金标。"),
        "request_policy": f"sequential, delay={args.delay}s, retries<=2, curl only, no login/captcha bypass",
    }
    seen_ids: set[str] = set()
    for scope in scopes:
        print(f"[scope] {scope}", flush=True)
        try:
            rows = enumerate_templates(scope, args.limit or None)
        except RuntimeError as error:
            print(f"[error] enumeration {scope}: {error}", file=sys.stderr)
            failures.append({"scope": scope, "stage": "enumeration", "error": str(error)})
            continue
        print(f"[scope] {scope}: {len(rows)} templates", flush=True)
        for number, row in enumerate(rows, 1):
            doc_id = str(row.get("Id") or "")
            if not doc_id or doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            target = contracts_dir / f"{doc_id}.json"
            if target.exists() and not args.refresh:
                continue
            url = f"{SITE}/View?id={doc_id}"
            status, body, error = curl_bytes(url, timeout=args.timeout, retries=2)
            if status != 200 or not body:
                failures.append({"scope": scope, "id": doc_id, "url": url,
                                 "stage": "detail", "status": status, "error": error})
                print(f"[fail] {number}/{len(rows)} {doc_id} status={status} {error[:120]}", flush=True)
                time.sleep(args.delay)
                continue
            html_text = body.decode("utf-8", "replace")
            parsed = parse_detail(html_text)
            payload = {
                "schema": SCHEMA,
                "doc": {
                    "id": doc_id,
                    "scope": scope,
                    "title": parsed["title"],
                    "department": clean(str(row.get("Department") or "")),
                    "published_on": str(row.get("PublishedOn") or ""),
                    "template_type": row.get("Type"),
                    "region": clean(str(row.get("Region") or "")),
                    "tags": clean(str(row.get("Tags") or "")),
                    "api_brief": clean(str(row.get("Brief") or "")),
                    "meta": parsed["meta"],
                    "attachments": parsed["attachments"],
                },
                "contract_text": parsed["contract_text"],
                "human_annotations": [
                    {"kind": "official_risk_tip", "topic": item["topic"], "content": item["content"]}
                    for item in parsed["annotations"]
                ],
                "source": {
                    "url": url,
                    "fetched_at": now_utc(),
                    "status": status,
                    "bytes": len(body),
                    "sha256": sha256_bytes(body),
                    "raw_saved": bool(args.save_raw),
                },
                "annotation_nature": ("发布机关撰写的风险提示与使用提示，绑定该示范文本；"
                                      "属于人工审查指引，不是逐条款标注，也不是本项目独立金标"),
            }
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.save_raw:
                (raw_dir / f"{doc_id}.html").write_bytes(body)
            index_rows.append({"id": doc_id, "scope": scope, "title": payload["doc"]["title"],
                               "department": payload["doc"]["department"],
                               "published_on": payload["doc"]["published_on"],
                               "annotations": len(payload["human_annotations"]),
                               "contract_chars": len(payload["contract_text"]),
                               "url": url,
                               "sha256": payload["source"]["sha256"]})
            if number % 20 == 0 or number == len(rows):
                print(f"[progress] {scope} {number}/{len(rows)}", flush=True)
            time.sleep(args.delay)
    # Merge index with prior rows so resume keeps a complete catalog.
    index_path = out / "index.jsonl"
    prior_index: list[dict] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    prior_index.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    merged = {row["id"]: row for row in prior_index}
    merged.update({row["id"]: row for row in index_rows})
    with index_path.open("w", encoding="utf-8") as handle:
        for row in sorted(merged.values(), key=lambda item: (item["scope"], item["title"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": SCHEMA,
        "site": SITE,
        "checked_at": now_utc(),
        "started_at": started,
        "policy": policy,
        "counts": {
            "catalog_total": len(merged),
            "fetched_this_run": len(index_rows),
            "failures_this_run": len(failures),
            "with_annotations": sum(1 for row in merged.values() if row.get("annotations", 0) > 0),
        },
        "failures": failures,
        "index_sha256": sha256_bytes(index_path.read_bytes()) if index_path.exists() else "",
        "usage_boundary": ("外部人工审查指引样本，可用于人工标注对照与评测设计；"
                           "不能作为本项目独立金标，也不能据此声称模型法律质量"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "failures": failures[:5]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
