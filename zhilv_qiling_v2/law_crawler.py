"""Bounded official-law acquisition. Every hop is allowlisted, resolved and IP pinned."""
from __future__ import annotations

import http.client
import io
import ipaddress
import json
import re
import socket
import ssl
import threading
import time
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .law_library import LawError, _admin, _now

MAX_BYTES = 20 * 1024 * 1024
OFFICIAL_SUFFIXES = ("gov.cn",)
_HOST_LAST = {}
_HOST_LOCK = threading.Lock()


def source_catalog():
    payload = json.loads(Path(__file__).with_name("law_sources.json").read_text(encoding="utf-8"))
    return [dict(source, url=source["urls"][0]) for source in payload["sources"]]


def _public_addresses(host, port):
    try:
        addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    except OSError as exc:
        raise LawError("官方域名DNS解析失败", 502) from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise LawError("禁止访问非公网地址", 400)
    return addresses


def _validate_url(url, *, resolve=True):
    if not isinstance(url, str) or len(url) > 4096 or re.search(r"[\x00-\x20\\]", url):
        raise LawError("来源URL无效")
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise LawError("来源URL无效") from exc
    if parsed.scheme not in {"http", "https"} or parsed.username is not None or parsed.password is not None or not host:
        raise LawError("仅允许官方HTTP(S)来源")
    if port != (443 if parsed.scheme == "https" else 80):
        raise LawError("不允许自定义端口")
    if not any(host == suffix or host.endswith("." + suffix) for suffix in OFFICIAL_SUFFIXES):
        raise LawError("来源不在官方域名白名单")
    addresses = _public_addresses(host, port) if resolve else []
    return {"host": host, "port": port, "scheme": parsed.scheme, "addresses": addresses, "target": parsed.path or "/", "query": parsed.query}


def _throttle(host):
    # Holding the lock while reserving/starting is intentional: parallel callers cannot burst.
    with _HOST_LOCK:
        delay = 1.0 - (time.monotonic() - _HOST_LAST.get(host, 0))
        if delay > 0:
            time.sleep(delay)
        _HOST_LAST[host] = time.monotonic()


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address, timeout):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self):
        self.sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _transport(url, timeout, max_bytes, addresses):
    info = _validate_url(url, resolve=False)
    address = addresses[0]
    connection = _PinnedHTTPS(info["host"], address, timeout) if info["scheme"] == "https" else http.client.HTTPConnection(address, port=80, timeout=timeout)
    target = info["target"] + (("?" + info["query"]) if info["query"] else "")
    try:
        connection.request("GET", target, headers={"Host": info["host"], "User-Agent": "ZhilvLegalLibrary/1.0 (bounded official legal archive)", "Accept": "text/html,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Accept-Encoding": "identity"})
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        if headers.get("content-length"):
            try:
                if int(headers["content-length"]) > max_bytes:
                    raise LawError("原件超过20MiB上限", 413)
            except ValueError as exc:
                raise LawError("来源Content-Length无效", 502) from exc
        chunks, size = [], 0
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() > deadline:
                raise LawError("原件下载总时限已到", 504)
            chunk = response.read(min(65536, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise LawError("原件超过20MiB上限", 413)
            chunks.append(chunk)
        return {"status": response.status, "headers": headers, "content": b"".join(chunks)}
    finally:
        connection.close()


def _fetch(url, *, timeout=15, max_bytes=MAX_BYTES, transport=None):
    timeout = max(1, min(30, float(timeout)))
    max_bytes = max(1, min(MAX_BYTES, int(max_bytes)))
    transport = transport or _transport
    current = url
    redirects = []
    for hop in range(6):
        info = _validate_url(current)
        _throttle(info["host"])
        response = transport(current, timeout, max_bytes, info["addresses"])
        response["headers"] = {str(k).lower(): str(v) for k, v in response.get("headers", {}).items()}
        raw = response.get("content", b"")
        if not isinstance(raw, bytes) or len(raw) > max_bytes:
            raise LawError("原件超过下载上限", 413)
        if response["status"] in {301, 302, 303, 307, 308}:
            location = response["headers"].get("location")
            if not location:
                raise LawError("官方来源跳转缺少Location", 502)
            next_url = urljoin(current, location)
            _validate_url(next_url, resolve=False)
            if urlsplit(current).scheme == "https" and urlsplit(next_url).scheme != "https":
                raise LawError("禁止HTTPS降级跳转")
            redirects.append({"from": current, "to": next_url, "status": response["status"]})
            current = next_url
            continue
        if response["status"] != 200:
            raise LawError(f"官方来源HTTP {response['status']}", 502)
        if response["headers"].get("content-encoding", "identity").lower() not in {"", "identity"}:
            raise LawError("来源未遵守未压缩传输要求", 502)
        return {**response, "url": current, "redirects": redirects}
    raise LawError("官方来源跳转次数超过5次", 502)


class _Node:
    def __init__(self, tag="", attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.children = []

    def text(self):
        if self.tag in {"script", "style", "nav", "header", "footer", "noscript", "svg"}:
            return ""
        text = "".join(child.text() if isinstance(child, _Node) else child for child in self.children)
        return ("\n" + text + "\n") if self.tag in {"p", "div", "br", "article", "section", "main", "li", "h1", "h2", "h3", "tr"} else text


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self.stack = [self.root]
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs)
        self.stack[-1].children.append(node)
        self.nodes.append(node)
        if tag not in {"br", "meta", "link", "img", "hr", "input", "wbr", "source", "area", "base", "embed", "param", "track", "col"}:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        for position in range(len(self.stack) - 1, 0, -1):
            if self.stack[position].tag == tag:
                del self.stack[position:]
                return

    def handle_data(self, text):
        self.stack[-1].children.append(text)


def _clean(text):
    return re.sub(r"\n{2,}", "\n", "\n".join(re.sub(r"[\t \u3000\xa0]+", " ", line).strip() for line in text.splitlines())).strip()


_ARTICLE = re.compile(r"(?m)^\s*(第[零〇一二三四五六七八九十百千万两\d]{1,16}条(?:之[零〇一二三四五六七八九十百\d]+)?)\s*")
_NUMBERED = re.compile(r"(?m)^\s*([一二三四五六七八九十百\d]{1,5}[、．.])\s*")
_CHAPTER = re.compile(r"(?m)^[ \t]*(?:第[零〇一二三四五六七八九十百千万两\d]+[编章节][^\n。；]*|附[ \t]*则)[ \t]*$")


def _decode_html(raw, content_type):
    if raw.startswith((b"%PDF", b"PK\x03\x04")) or b"\x00" in raw[:4096]:
        raise LawError("二进制伪装为HTML")
    probe = raw[:8192].decode("ascii", "ignore")
    match = re.search(r"charset\s*=\s*[\"']?\s*([\w-]+)", content_type + " " + probe, re.I)
    encoding = match.group(1).lower() if match else "utf-8-sig"
    if encoding in {"gb2312", "gbk", "gb_2312-80"}:
        encoding = "gb18030"
    try:
        return raw.decode(encoding)
    except (LookupError, UnicodeError):
        for fallback in ("utf-8-sig", "gb18030"):
            try:
                return raw.decode(fallback)
            except UnicodeError:
                pass
    raise LawError("无法可靠解码官方原件")


def _split_legal_text(text, title, source_url):
    text = _clean(text)
    footer = re.search(r"(?m)^\s*(?:相关稿件|相关阅读|相关新闻|相关链接|责任编辑[：:]|网站地图|扫一扫在手机打开当前页)\s*", text)
    if footer:
        text = text[:footer.start()].rstrip()
    if re.search(r"验证码|人机验证|访问被拒绝|安全验证|页面不存在|Access Denied|captcha", title + "\n" + text[:1500], re.I):
        raise LawError("官方来源返回错误或验证码")
    if not re.search(r"法|条例|办法|规定|规则|通知|解释|决定|细则|章程|守则", title) or re.search(r"首页|网站导航|404|Not Found", title, re.I):
        raise LawError("未识别强法规标题，拒绝首页或导航")
    matches = list(_ARTICLE.finditer(text))
    if not matches:
        matches = list(_NUMBERED.finditer(text))
    if not matches or sum(len(text[m.start():matches[i + 1].start() if i + 1 < len(matches) else len(text)]) for i, m in enumerate(matches)) < 50:
        raise LawError("未提取到足够的编号法条正文")
    articles = []
    chapters = list(_CHAPTER.finditer(text))
    for index, match in enumerate(matches):
        content = text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip()
        content = _CHAPTER.sub("", content).strip()
        if len(re.findall(r"[\u4e00-\u9fff]", content)) < 10:
            raise LawError("疑似目录，没有实质法条内容")
        location = {"unit_kind": "article", "article_index": index + 1, "line_start": text[:match.start()].count("\n") + 1}
        preceding = [heading for heading in chapters if heading.start() < match.start()]
        if preceding:
            location["section_heading"] = preceding[-1].group(0).strip()
        articles.append({"article_no": match.group(1), "content": content, "location": location, "source_url": source_url})
    document = {"title": title.strip(), "validity": "unknown", "official_status": "unknown", "effective_from": "", "effective_to": "", "published_at": "", "region": "全国", "authority_level": "unknown", "issuing_authority": "unknown", "document_number": "unknown", "extraction_quality": "official_text_unverified", "date_evidence": []}
    effective = list(re.finditer(r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*起?\s*(?:施行|实施)", text))
    if effective:
        match = effective[-1]
        document["effective_from"] = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        document["date_evidence"].append({"field": "effective_from", "quote": match.group(0), "source_url": source_url})
    # An extracted commencement date is not evidence that a law is still effective today.
    return {"document": document, "articles": articles}


def extract_document(raw, source_url, *, content_type="text/html"):
    _validate_url(source_url, resolve=False)
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise LawError("原件为空或超过20MiB")
    media = content_type.split(";", 1)[0].lower().strip()
    if media == "application/pdf":
        if not raw.startswith(b"%PDF"):
            raise LawError("PDF签名无效")
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
        except ImportError as exc:
            raise LawError("PDF提取依赖PyMuPDF未安装", 422) from exc
        try:
            with fitz.open(stream=raw, filetype="pdf") as pdf:
                if len(pdf) > 2000:
                    raise LawError("PDF页数超过上限")
                text = "\n".join(page.get_text() for page in pdf)
        except LawError:
            raise
        except Exception as exc:
            raise LawError("PDF原件解析失败", 422) from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = next((line for line in lines[:15] if re.search(r"法|条例|办法|规定", line)), "")
        return _split_legal_text(text, title, source_url)
    if media == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if len(archive.infolist()) > 2000 or sum(entry.file_size for entry in archive.infolist()) > 60 * 1024 * 1024:
                    raise LawError("DOCX解压体积超过上限", 413)
            from docx import Document
            paragraphs = [p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()]
        except ImportError as exc:
            raise LawError("DOCX提取依赖python-docx未安装", 422) from exc
        except LawError:
            raise
        except Exception as exc:
            raise LawError("DOCX原件解析失败", 422) from exc
        return _split_legal_text("\n".join(paragraphs), next((p for p in paragraphs[:15] if re.search(r"法|条例|办法|规定", p)), ""), source_url)
    if media not in {"text/html", "application/xhtml+xml", ""}:
        raise LawError("不支持的原件类型", 422)
    html = _decode_html(raw, content_type)
    if not re.search(r"<(?:html|article|main|body)\b", html, re.I):
        raise LawError("来源不是HTML文档")
    parser = _HTML()
    parser.feed(html)
    title_nodes = [node for node in parser.nodes if node.tag == "h1"] or [node for node in parser.nodes if node.tag == "title"]
    title = next((_clean(node.text()) for node in title_nodes if _clean(node.text())), "")
    title = re.split(r"[_|]\s*(?:中国政府网|国家|国务院|中华|全国|.*政府)", title)[0].strip()
    title = re.split(r"\s+-\s+(?:中华|中国|国家|最高|.*政府)", title)[0].strip()
    strong = [node for node in parser.nodes if node.tag in {"article", "main"} or (node.tag in {"div", "section", "td"} and re.search(r"(?:^|[ _-])(?:pages_content|TRS_Editor|article-content|article_content|content_body|law-content|art-con|content)(?:$|[ _-])", node.attrs.get("class", "") + " " + node.attrs.get("id", ""), re.I))]
    candidates = strong or [node for node in parser.nodes if node.tag == "body"] or [parser.root]
    scored = [(_clean(node.text()), node) for node in candidates]
    # Among containers holding all the articles, prefer the narrowest one so footer
    # recommendations and sibling navigation do not become the final article's text.
    text = max(scored, key=lambda pair: (
        len(_ARTICLE.findall(pair[0])) + len(_NUMBERED.findall(pair[0])), -len(pair[0])))[0]
    extracted = _split_legal_text(text, title, source_url)
    for node in parser.nodes:
        if node.tag == "meta" and node.attrs.get("name", "").lower() in {"pubdate", "firstpublishedtime", "publishdate"}:
            value = node.attrs.get("content", "")
            match = re.match(r"\d{4}-\d{2}-\d{2}", value)
            if match:
                extracted["document"]["published_at"] = match.group(0)
                extracted["document"]["date_evidence"].append({"field": "published_at", "quote": value, "source_url": source_url, "kind": "page_publication_not_promulgation"})
    return extracted


def crawl_sources(library, actor, *, source_ids=None, max_documents=10, timeout=15):
    _admin(actor)
    if isinstance(max_documents, bool) or not isinstance(max_documents, int) or not 1 <= max_documents <= 50:
        raise LawError("max_documents必须在1至50之间")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
        raise LawError("timeout必须大于0且不超过30秒")
    catalog = source_catalog()
    if source_ids is not None and (not isinstance(source_ids, list) or any(s not in {x["id"] for x in catalog} for s in source_ids)):
        raise LawError("未知官方法条来源")
    selected = [source for source in catalog if source_ids is None or source["id"] in source_ids]
    run = {"run_id": "lc_" + uuid.uuid4().hex, "status": "running", "attempted": 0, "succeeded": 0, "failed": 0, "results": [], "started_at": _now(), "max_documents": max_documents, "timeout": timeout, "source_ids": [source["id"] for source in selected]}
    library._record_crawl_run(run)
    for source in selected:
        for url in source.get("urls", [source.get("url")]):
            if run["attempted"] >= max_documents:
                break
            run["attempted"] += 1
            receipt = {"source_id": source["id"], "url": url, "status": "running", "started_at": _now()}
            run["results"].append(receipt)
            library._record_crawl_run(run)
            try:
                response = _fetch(url, timeout=timeout)
                content_type = response["headers"].get("content-type", "text/html")
                parsed = extract_document(response["content"], response["url"], content_type=content_type)
                metadata = {**parsed["document"], "domains": source["domains"], "source_id": source["id"], "seed_url": url, "redirects": response["redirects"]}
                stored = library.ingest_document(actor, metadata, parsed["articles"], response["content"], response["url"], content_type=content_type)
                receipt.update(stored)
                receipt.update({"status": "succeeded", "version_status": stored["status"], "final_url": response["url"], "bytes": len(response["content"]), "article_count": len(parsed["articles"])})
                run["succeeded"] += 1
            except Exception as exc:
                # Keep a bounded durable failure receipt; don't turn extraction/network errors into successes.
                receipt.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:700]})
                run["failed"] += 1
            receipt["finished_at"] = _now()
            library._record_crawl_run(run)
    run["status"] = "partial" if run["succeeded"] and run["failed"] else ("failed" if run["failed"] else "succeeded")
    run["finished_at"] = _now()
    library._record_crawl_run(run)
    return run
