"""抓取政府采购合同公告里的扫描件，整理成可做 OCR 的图片数据集。

目前只接了北京市政府采购网，因为它的合同附件是直链 PDF，不需要验证码，
而且很多附件本身是"整页扫描图 + 文字层"的形式，正好能拿文字层做 OCR 对照。

每个合同落到 data/gov_contracts/<合同ID>/ 下：
    pages/page_0001.jpg   原始扫描图（没有整页图时按 DPI 渲染）
    text/page_0001.txt    文字层，作为 OCR 的弱标注
    meta.json             公告字段与文件信息
汇总写到 data/gov_contracts/manifest.jsonl。

    python scripts/crawl_contracts.py --pages 3 --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib3
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pymupdf
import requests

urllib3.disable_warnings()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "gov_contracts"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}

SOURCES = {
    "beijing": {
        "base": "http://www.ccgp-beijing.gov.cn/",
        "list": "xxgg/sjxxgg/htgg/A002004001003index_{page}.htm",
        "detail_pattern": re.compile(
            r"//www\.ccgp-beijing\.gov\.cn/xxgg/sjxxgg/htgg/\d{4}/\d{1,2}/[0-9a-f]+\.htm"
        ),
    }
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def fetch(session: requests.Session, url: str, *, timeout: int = 30, binary: bool = False):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=timeout, verify=False, headers=HEADERS)
            resp.raise_for_status()
            if binary:
                return resp.content
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last_error}")


def field(html: str, label: str, max_len: int = 120) -> str:
    # 公告正文是 div/段落混排，用"标签 + 后面一小段"截取比解析 DOM 稳
    match = re.search(
        rf"{re.escape(label)}\s*[：:]\s*(?:</?[^>]+>\s*)*([^<]{{0,{max_len}}})",
        html,
    )
    return clean(match.group(1)) if match else ""


def attachment_urls(html: str, base: str) -> list[str]:
    result = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        path = urlparse(href).path.lower()
        if path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
            result.append(urljoin(base, href))
    return list(dict.fromkeys(result))


def collect_listing(session, cfg, pages: int, delay: float) -> list[str]:
    links: list[str] = []
    for page in range(1, pages + 1):
        url = urljoin(cfg["base"], cfg["list"].format(page=page))
        html = fetch(session, url)
        for href in cfg["detail_pattern"].findall(html):
            # 站点里的链接是 //host/path 形式，补上协议才能请求
            links.append("http:" + href if href.startswith("//") else href)
        time.sleep(delay)
    return list(dict.fromkeys(links))


def extract_pages(pdf_bytes: bytes, out_dir: Path, dpi: int) -> dict[str, Any]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_dir = out_dir / "pages"
    text_dir = out_dir / "text"
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    page_info = []
    for index, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        (text_dir / f"page_{index:04d}.txt").write_text(text, encoding="utf-8")

        saved = False
        # 整页扫描图直接抽原图，避免二次重采样；否则按 DPI 渲染
        for img in sorted(
            page.get_images(full=True),
            key=lambda item: item[2] * item[3],
            reverse=True,
        ):
            rects = page.get_image_rects(img[0])
            if not rects:
                continue
            covered = sum(abs(rect.width * rect.height) for rect in rects)
            if covered < page.rect.width * page.rect.height * 0.6:
                continue
            info = doc.extract_image(img[0])
            ext = info["ext"]
            (pages_dir / f"page_{index:04d}.{ext}").write_bytes(info["image"])
            saved = True
            page_info.append(
                {
                    "page": index,
                    "width": info["width"],
                    "height": info["height"],
                    "format": ext,
                    "source": "embedded",
                    "text_len": len(text),
                }
            )
            break

        if not saved:
            pix = page.get_pixmap(dpi=dpi)
            (pages_dir / f"page_{index:04d}.png").write_bytes(pix.tobytes("png"))
            page_info.append(
                {
                    "page": index,
                    "width": pix.width,
                    "height": pix.height,
                    "format": "png",
                    "source": "rendered",
                    "text_len": len(text),
                }
            )

    scanned_pages = sum(1 for p in page_info if p["text_len"] < 50)
    image_pages = sum(1 for p in page_info if p["source"] == "embedded")
    return {
        "page_count": doc.page_count,
        "scanned_pages": scanned_pages,
        "text_pages": doc.page_count - scanned_pages,
        "image_pages": image_pages,
        "image_with_text_pages": image_pages - scanned_pages,
        "pages": page_info,
    }


def crawl(source: str, pages: int, limit: int, out_dir: Path, dpi: int, delay: float, keep_pdf: bool):
    cfg = SOURCES[source]
    cfg["detail_pattern"] = re.compile(cfg["detail_pattern"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    session = requests.Session()
    fetch(session, cfg["base"])

    known: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                known.add(json.loads(line)["contract_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    links = collect_listing(session, cfg, pages, delay)
    print(f"列表抓到 {len(links)} 条合同公告，已有 {len(known)} 条，目标新增 {limit} 条")

    saved = 0
    for detail_url in links:
        if saved >= limit:
            break
        contract_id = hashlib.sha1(detail_url.encode()).hexdigest()[:12]
        if contract_id in known:
            continue

        html = fetch(session, detail_url)
        title = field(html, "合同名称", max_len=200) or field(html, "项目名称", max_len=200)
        attachments = attachment_urls(html, cfg["base"])
        if not attachments:
            time.sleep(delay)
            continue

        pdf_url = next((u for u in attachments if u.lower().endswith(".pdf")), None)
        if not pdf_url:
            # 图片附件直接存，不依赖 PDF
            target_dir = out_dir / contract_id
            target_dir.mkdir(parents=True, exist_ok=True)
            images_dir = target_dir / "pages"
            images_dir.mkdir(exist_ok=True)
            for i, url in enumerate(attachments, start=1):
                ext = Path(urlparse(url).path).suffix.lower()
                (images_dir / f"page_{i:04d}{ext}").write_bytes(
                    fetch(session, url, binary=True, timeout=90)
                )
            page_meta = {"page_count": len(attachments), "scanned_pages": len(attachments),
                         "text_pages": 0, "pages": []}
        else:
            pdf_bytes = fetch(session, pdf_url, binary=True, timeout=120)
            target_dir = out_dir / contract_id
            page_meta = extract_pages(pdf_bytes, target_dir, dpi)
            if keep_pdf:
                (target_dir / "contract.pdf").write_bytes(pdf_bytes)

        meta = {
            "contract_id": contract_id,
            "source": source,
            "detail_url": detail_url,
            "pdf_url": pdf_url,
            "title": title,
            "project_no": field(html, "项目编号"),
            "contract_no": field(html, "合同编号"),
            "purchaser": field(html, "采购人（甲方）") or field(html, "采购人"),
            "supplier": field(html, "供应商（乙方）") or field(html, "供应商"),
            "amount": field(html, "合同金额"),
            "sign_date": field(html, "合同签订日期"),
            "publish_date": field(html, "合同公告日期"),
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **page_meta,
        }
        (target_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        saved += 1
        print(
            f"[{saved}/{limit}] {meta['contract_no'] or contract_id} {title[:40]} "
            f"页数={page_meta['page_count']} 扫描页={page_meta['scanned_pages']}"
        )
        time.sleep(delay)

    print(f"完成：新增 {saved} 条，输出目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), default="beijing")
    parser.add_argument("--pages", type=int, default=3, help="列表页数，每页约 18 条")
    parser.add_argument("--limit", type=int, default=20, help="最多下载多少份合同")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--delay", type=float, default=1.2, help="请求间隔秒数")
    parser.add_argument("--keep-pdf", action="store_true", help="保留原始 PDF")
    args = parser.parse_args()
    crawl(args.source, args.pages, args.limit, args.out, args.dpi, args.delay, args.keep_pdf)


if __name__ == "__main__":
    main()
