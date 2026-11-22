"""Fetch official contract-review case write-ups as external reference samples.

What this is
------------
Curated pages published by the State Administration for Market Regulation and
courts that quote an actual or templated contract clause and then give the
regulator's / court's human reasoning: why the clause is unfair, invalid, or
how it should be read. This is case-level human legal analysis. It is NOT
clause-level annotation, NOT an independent gold set, and NOT evidence that the
project model is correct.

What this is not
----------------
* no login, captcha or paywall bypass;
* no full-text scraping of 人民法院案例库 / 裁判文书网 (both need login tokens);
* no claim that a case write-up is a ground-truth risk label for a contract.

Fetching goes through the system ``curl`` because this host's Python TLS stack
is rejected by these government sites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SCHEMA = "zlql-external-case-analysis-v1"

SOURCES: list[dict] = [
    {
        "slug": "samr_2023_ironfist_7",
        "url": "https://www.samr.gov.cn/zt/pgt/art/2023/art_17a60e87f5a44267916dbaa889dffd9e.html",
        "agency": "国家市场监督管理总局",
        "doc_type": "合同格式条款行政执法典型案例",
        "containers": ["bgt_content", "contentLeft0102box"],
        "note": "第七批铁拳行动典型案例；多起利用不公平格式条款侵权案，含条款原文与认定理由",
    },
    {
        "slug": "samr_2023_guangdong_ironfist_1",
        "url": "https://www.samr.gov.cn/zfjcj/dxal/art/2023/art_5665d3ec4920431fa9d79ded9862da55.html",
        "agency": "国家市场监督管理总局（转广东省市场监管局）",
        "doc_type": "合同格式条款行政执法典型案例",
        "containers": ["zt_xilan_07", "contentLeft0102box", "bgt_content"],
        "note": "广东省民生领域铁拳行动第一批；含健身、培训等合同格式条款案",
    },
    {
        "slug": "samr_2025_anhui_ironfist_1",
        "url": "https://www.samr.gov.cn/zt/pgt/art/2025/art_f047e492f6474917be6e2bc6fbb776bb.html",
        "agency": "国家市场监督管理总局（转安徽省市场监管局）",
        "doc_type": "合同格式条款行政执法典型案例",
        "containers": ["bgt_content", "contentLeft0102box"],
        "note": "安徽省守护消费铁拳行动第一批；含预付费培训合同投诉权排除条款",
    },
    {
        "slug": "spc_construction_contract_cases",
        "url": "https://www.court.gov.cn/zixun/xiangqing/504211.html",
        "agency": "最高人民法院",
        "doc_type": "合同纠纷典型案例",
        "containers": ["detail", "detail_mes", "content"],
        "note": "建设工程施工合同纠纷六个典型案例；含合同效力、价款结算、质量保证金等裁判理由",
    },
    {
        "slug": "supreme_contract_rules_103",
        "url": "http://cdqlfy.scssfw.gov.cn/article/detail/2019/07/id/4208967.shtml",
        "agency": "最高人民法院（地方人民法院转载汇编）",
        "doc_type": "合同类纠纷裁判规则汇编",
        "containers": ["text", "contentLeft0102box", "bgt_content", "content"],
        "note": "103条合同类纠纷裁判规则汇编，逐条含案例来源与裁判要旨",
    },
    {
        "slug": "spc_contract_book_cases_2023",
        "url": "https://hnsqzy.hncourt.gov.cn/public/detail.php?id=19469",
        "agency": "最高人民法院（河南省某中级人民法院转载）",
        "doc_type": "合同编通则司法解释典型案例",
        "containers": ["detail_content", "detail", "content"],
        "note": "民法典合同编通则司法解释相关典型案例；含裁判要点与判决理由",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def curl(url: str, timeout: int = 40, retries: int = 2) -> tuple[int, bytes, str]:
    last = ""
    for attempt in range(retries + 1):
        proc = subprocess.run(
            ["curl.exe" if sys.platform == "win32" else "curl", "-sL", "--compressed",
             "--max-time", str(timeout), "-A", UA, "-w", "\n%{http_code}", url],
            capture_output=True, timeout=timeout + 15)
        if proc.returncode == 0 and proc.stdout:
            marker = proc.stdout.rfind(b"\n")
            try:
                status = int(proc.stdout[marker + 1:].strip() or b"0")
            except ValueError:
                status = 0
            return status, proc.stdout[:marker], ""
        last = (proc.stderr or b"").decode("utf-8", "replace").strip()[:300]
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return 0, b"", last


class ArticleParser(HTMLParser):
    def __init__(self, containers: list[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.containers = containers
        self.frames: list[dict] = []
        self.captures: list[str] = []
        self._skip = 0
        self._active: dict | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip += 1
            return
        classes = set()
        for key, value in attrs:
            if key == "class" and value:
                classes = set(value.split())
        frame = {"tag": tag, "classes": classes, "text": []}
        self.frames.append(frame)
        if self._active is None and classes & set(self.containers):
            self._active = frame
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            frame["text"].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            if self._skip:
                self._skip -= 1
            return
        if not self.frames:
            return
        frame = self.frames.pop()
        if frame is self._active:
            text = "\n".join(frame["text"])
            self.captures.append(text)
            self._active = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        for frame in self.frames:
            frame["text"].append(data)


def clean_text(raw: str) -> str:
    raw = raw.replace("\u00a0", " ").replace("\u200b", "")
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    lines = [line.strip() for line in raw.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_page(html_text: str, containers: list[str]) -> dict:
    parser = ArticleParser(containers)
    parser.feed(html_text)
    parser.close()
    candidates = sorted((clean_text(item) for item in parser.captures), key=len, reverse=True)
    text = next((item for item in candidates if len(item) >= 300), "")
    title = ""
    match = re.search(r"<title>(.*?)</title>", html_text, re.S)
    if match:
        title = clean_text(match.group(1)).split(" - ")[0]
    published = ""
    for pattern in (r"发布时间[:：]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
                    r"时间[:：]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
                    r"([0-9]{4}-[0-9]{2}-[0-9]{2})\s+[0-9]{2}:[0-9]{2}"):
        found = re.search(pattern, text or html_text)
        if found:
            published = found.group(1)
            break
    case_sections = []
    for match in re.finditer(r"(?:^|\n)\s*([一二三四五六七八九十百]{1,3}、[^\n]{4,90})", text):
        start = match.start(1)
        case_sections.append({"heading": match.group(1).strip(), "start": start})
    for index, section in enumerate(case_sections):
        end = case_sections[index + 1]["start"] if index + 1 < len(case_sections) else len(text)
        section["text"] = text[section["start"]:end].strip()
    markers = {name: bool(re.search(pattern, text)) for name, pattern in {
        "案情": r"案情|基本案情", "处理": r"处理|处罚|判决|裁判结果",
        "评析/理由": r"评析|理由|典型意义|裁判要旨|裁判要点"}.items()}
    return {"title": title, "published": published, "text": text,
            "cases": case_sections, "markers": markers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/annotation_samples/official_cases"))
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    sources = SOURCES[:args.limit] if args.limit else SOURCES
    for source in sources:
        status, body, error = curl(source["url"])
        if status != 200 or not body:
            failures.append({**source, "status": status, "error": error})
            print(f"[fail] {source['slug']} status={status} {error[:120]}", flush=True)
            time.sleep(args.delay)
            continue
        html_text = body.decode("utf-8", "replace")
        parsed = parse_page(html_text, source["containers"])
        if len(parsed["text"]) < 300:
            failures.append({**source, "stage": "extract", "error": "main_text_too_short"})
            print(f"[fail] {source['slug']} extract too short", flush=True)
            continue
        payload = {
            "schema": SCHEMA,
            "case_id": source["slug"],
            "document": {
                "title": parsed["title"],
                "agency": source["agency"],
                "doc_type": source["doc_type"],
                "published": parsed["published"],
                "note": source["note"],
            },
            "human_analysis": {
                "text": parsed["text"],
                "cases": parsed["cases"],
                "markers": parsed["markers"],
            },
            "source": {
                "url": source["url"],
                "fetched_at": now_utc(),
                "status": status,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            },
            "annotation_nature": ("官方案例/裁判规则中的人工法律分析：引用争议合同条款并说明认定理由；"
                                  "属案例级人工分析，不是逐条款人工标注，也不是本项目独立金标"),
        }
        (out / f"{source['slug']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"case_id": source["slug"], "title": parsed["title"],
                     "agency": source["agency"], "doc_type": source["doc_type"],
                     "published": parsed["published"], "chars": len(parsed["text"]),
                     "cases": len(parsed["cases"]), "url": source["url"],
                     "sha256": payload["source"]["sha256"]})
        print(f"[ok] {source['slug']} chars={len(parsed['text'])} cases={len(parsed['cases'])}", flush=True)
        time.sleep(args.delay)
    index_path = out / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": SCHEMA,
        "checked_at": now_utc(),
        "policy": {
            "access": "only public pages; no login/captcha/paywall bypass",
            "excluded_sources": {
                "人民法院案例库 rmfyalk.court.gov.cn": "search API returns 401 未登录; full text needs login",
                "中国裁判文书网 wenshu.court.gov.cn": "not crawled: login/captcha protected",
                "国家基础学科公共数据中心 nbsdc.cn": "dataset page needs login; host TLS unreachable from script",
            },
            "usage": "仅用于人工标注对照与评测设计；不转售、不篡改；引用注明来源",
        },
        "counts": {"sources": len(sources), "ok": len(rows), "failures": len(failures),
                   "total_chars": sum(row["chars"] for row in rows),
                   "split_cases": sum(row["cases"] for row in rows)},
        "failures": failures,
        "index_sha256": sha256_bytes(index_path.read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
