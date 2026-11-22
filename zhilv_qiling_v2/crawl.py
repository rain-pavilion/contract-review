"""多源合同爬取：令牌可配 + 无令牌自动降级 + 状态化去重。

相对 v1 的升级点
----------------
1. **v1 是单站点硬编码爬虫**：只认北京政采一个栏目，URL 改一次就整条链断。v2 把
   "源"抽成 `Source` 数据类（`kind=api/html`、`needs_token`、列表模板、详情正则），
   `SOURCES` 里同时挂天津开放数据 API、中国政府采购网、北京市政府采购网。
2. **令牌显式化，不靠"能不能跑通"猜**：天津 `open.data.tj.gov.cn` 的合同数据集接口实测
   必须带账号令牌，裸调返回 `{"code":500,"msg":"令牌不能为空","data":null}`。v1 遇到这种
   响应会当成"没有数据"。v2 分三种情况处理：有令牌（`ZLQL_TJ_TOKEN` 或 `--token`）→
   正常拉；无令牌 → 记 `degraded` 并**自动降级到无需令牌的公开栏目**（北京/中央政采）；
   网络不通 → 同样是 `degraded`，绝不抛未捕获异常。
3. **31 个真实字段的 API 解析**：`parse_tj_payload` 按 `f_3070020000012_000007xxx` 的
   真实字段号映射成统一 meta（合同签订日期/采购人/项目名称/合同金额/合同名称/合同附件
   链接/合同编号/统一社会信用代码/附件下载），并兼容 `data` / `rows` / `result` 三种包裹。
4. **状态化去重**：`CrawlState` 落 `cache/crawl_state.json`，记 `done_ids` / `failed_urls`
   / `last_run`。v1 每次重爬，重复下载几百兆 PDF。
5. **落盘约定沿用 v1**（`out_dir/<sha1(url)[:12]>/{meta.json,pages/,text/}`），这样
   `data/gov_contracts` 里已有的 63 份合同和新抓的合同能被同一套下游代码读。
6. **可选 OCR**：`run(use_ocr=True)` 对扫描页调用 `ocr.merge_ocr_into_parse`，识别文本
   写回 `text/page_NNNN.txt`，并把 `ocr_pages` 计入统计 —— 这是 v1 缺的那一段。

工程约束
--------
* HTTP 用标准库 `urllib`（不硬依赖 requests），3 次指数退避重试 + UA 伪装 + 限速；
  编码从 BOM → HTTP header charset → meta charset → 有序试探（utf-8/gb18030/big5）。
* 任一层失败都记进 `degraded`，`run()` 永远返回统计字典。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import text as T
from .config import CACHE_DIR, ensure_dirs

__all__ = [
    "CrawlError",
    "Source",
    "SOURCES",
    "Crawler",
    "CrawlState",
    "parse_tj_payload",
    "attachments",
    "parse_detail",
    "detail_id",
    "main",
]

#: 天津开放数据合同数据集接口（实测必须带令牌）。
TJ_API = "https://open.data.tj.gov.cn/api/invoke/43651ffd8e454d328544e4a5bbd39c53"
#: 天津数据集的人工页面（无令牌也能打开，只做说明用）。
TJ_PAGE = ("https://open.data.tj.gov.cn//sjjk/"
           "43651ffd8e454d328544e4a5bbd39c53.htm")
#: 北京政采合同公告列表模板（实测有效）。
BJ_LIST = ("http://www.ccgp-beijing.gov.cn/xxgg/sjxxgg/htgg/"
           "A002004001003index_{page}.htm")
#: 北京政采详情链接规律（实测有效）。
BJ_DETAIL_RE = r"//www\.ccgp-beijing\.gov\.cn/xxgg/sjxxgg/htgg/\d{4}/\d{1,2}/[0-9a-f]+\.htm"
#: 中央政采列表模板（实测有效，./cjgg/... 相对链接）。
NAT_LIST = "http://www.ccgp.gov.cn/cggg/zygg/index_{page}.htm"
NAT_DETAIL_RE = r"\.?(?:/[a-z]{2,6})?/\d{6}/t\d{8}_\d+\.htm"

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ATTACH_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".doc", ".docx")
MAX_ATTACH_BYTES = 80 * 1024 * 1024


class CrawlError(RuntimeError):
    """抓取/解析失败。消息要能直接进报告，所以写清楚 URL 与原因。"""


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """一个数据源。`kind="api"` 走 JSON，`kind="html"` 走列表页 + 详情页正则。"""

    key: str
    name: str
    kind: str
    base: str
    needs_token: bool = False
    notes: str = ""
    list_url: str = ""
    """列表页模板，`{page}` 会被替换；api 源可留空（直连 `base`）。"""
    detail_re: str = ""
    token_env: str = "ZLQL_TJ_TOKEN"
    token_param: str = "token"


SOURCES: dict[str, Source] = {
    "tj_open_data": Source(
        key="tj_open_data",
        name="天津市公共数据开放平台 · 政府采购合同",
        kind="api",
        base=TJ_API,
        needs_token=True,
        notes=("接口实测必须账号令牌：裸调返回 "
               "{'code':500,'msg':'令牌不能为空','data':null}。"
               "令牌从环境变量 ZLQL_TJ_TOKEN 或 --token 传入；"
               "无令牌时自动降级到公开栏目（北京/中央政采）。"
               f"数据集页面：{TJ_PAGE}"),
        list_url=TJ_API,
        token_env="ZLQL_TJ_TOKEN",
        token_param="token",
    ),
    "ccgp_national": Source(
        key="ccgp_national",
        name="中国政府采购网 · 中央公告",
        kind="html",
        base="http://www.ccgp.gov.cn/",
        needs_token=False,
        notes="公开栏目，实测列表页 200；详情链接形如 ./cjgg/202611/t20261104_27615418.htm",
        list_url=NAT_LIST,
        detail_re=NAT_DETAIL_RE,
    ),
    "ccgp_beijing": Source(
        key="ccgp_beijing",
        name="北京市政府采购网 · 合同公告",
        kind="html",
        base="http://www.ccgp-beijing.gov.cn/",
        needs_token=False,
        notes="公开栏目，列表页 A002004001003index_{page}.htm，详情链接为 //host/xxgg/.../<32hex>.htm",
        list_url=BJ_LIST,
        detail_re=BJ_DETAIL_RE,
    ),
}


def get_source(key: str) -> Source:
    if key not in SOURCES:
        raise CrawlError(f"未知数据源 {key!r}，可选：{', '.join(SOURCES)}")
    return SOURCES[key]


# ---------------------------------------------------------------------------
# 文本/HTML 工具
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_META_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?([\w-]+)""", re.I)
_HREF_RE = re.compile(r"""href\s*=\s*(?:"([^"]+)"|'([^']+)')""", re.I)
#: "地 址：" 这类被排版空格拆开的两字标签，先合回去，否则它不会被当标签，
#: 上一个字段（采购人）的值会把地址整段吞进去。
_SPACED_LABEL_RE = re.compile(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])(?=[：:])")
#: 值的右边界：这些词一出现就说明进了下一个信息块。
_VALUE_END_RE = re.compile(
    r"\s(?:地\s*址|联系方式|联系电话|电\s*话|传\s*真|邮\s*编|开户行|账\s*号|邮政编码)")


def _flatten_html(html_text: str) -> str:
    """HTML → 单行纯文本。

    先把 script/style 整段切掉，再把所有标签换成空格（**不是**空串：`<td>采购人</td>`
    这种必须留词边界），最后压空白。标签与值被拆在相邻元素里时，压成单行后就能用
    "标签：值"正则直接命中。
    """
    body = _SCRIPT_RE.sub(" ", html_text or "")
    body = _TAG_RE.sub(" ", body)
    body = (body.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'"))
    body = re.sub(r"\s+", " ", body).strip()
    return _SPACED_LABEL_RE.sub(r"\1\2", body)


def _norm_label(label: str) -> str:
    """标签归一：去空白、去列表序号、去括注（采购人（甲方） → 采购人）。"""
    s = re.sub(r"\s+", "", label or "")
    s = re.sub(r"^[（(]?[一二三四五六七八九十]{1,3}[）)]?、", "", s)
    s = re.sub(r"^\d{1,2}[.、)]", "", s)
    s = re.sub(r"^第[一二三四五六七八九十]{1,3}[条章节]", "", s)
    s = re.sub(r"[（(][^）)]{0,12}[）)]", "", s)
    return s.strip("·、:： -")


def _abs_url(base_url: str, href: str) -> str:
    """把 `//host/x`、`./x`、`/x`、`x` 统一成绝对 URL。"""
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "http:" + href
    if href.startswith(("http://", "https://")):
        return href
    return urllib.parse.urljoin(base_url, href)


def sanitize_filename(url: str, *, length: int = 12) -> str:
    """v1 目录约定：`sha1(url)[:12]`。"""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:length]


def detail_id(url: str) -> str:
    """合同 ID：详情 URL（无则附件 URL）的 sha1 前 12 位，与 v1 目录名一致。"""
    return sanitize_filename(url or "")


# ---------------------------------------------------------------------------
# 天津 API 字段映射
# ---------------------------------------------------------------------------

_TJ_PREFIX = "f_3070020000012_000007"

#: 真实字段号 → 统一 meta 键。右侧是本模块输出的键名。
TJ_FIELDS: dict[str, str] = {
    "001": "sign_date",        # 合同签订日期
    "007": "purchaser",        # 采购人
    "008": "project_name",     # 项目名称
    "009": "amount",           # 合同金额
    "011": "title",            # 合同名称
    "012": "attachment_url",   # 合同附件链接
    "013": "contract_no",      # 合同编号
    "029": "credit_code",      # 统一社会信用代码
    "031": "download_url",     # 附件下载
}

#: 31 个字段里其余字段（不在上面映射表里的）保留进 `raw`，供后续扩展。
TJ_ALL_FIELDS: tuple[str, ...] = tuple(
    f"{_TJ_PREFIX}{i:03d}" for i in range(1, 32)
)


def _cell_value(cell: Any) -> str:
    """API 单元格可能是标量、`{"value": x}`、`{"text": x}` 或单元素列表。"""
    if cell is None:
        return ""
    if isinstance(cell, (str, int, float)):
        return str(cell).strip()
    if isinstance(cell, Mapping):
        for key in ("value", "text", "label", "name", "url"):
            if key in cell and cell[key] not in (None, "", []):
                return _cell_value(cell[key])
        return json.dumps(dict(cell), ensure_ascii=False)
    if isinstance(cell, (list, tuple)):
        if not cell:
            return ""
        return " ".join(v for v in (_cell_value(c) for c in cell) if v).strip()
    return str(cell).strip()


def _extract_rows(payload: Any) -> list[dict]:
    """从 `data` / `rows` / `result` / `records` / `list` 各种包裹里挖出行列表。"""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("data", "rows", "result", "records", "list", "items", "content"):
        if key in payload:
            rows = _extract_rows(payload[key])
            if rows:
                return rows
    # 有些接口把行放在 data 下再包一层 {total, list}
    for value in payload.values():
        if isinstance(value, (list, Mapping)):
            rows = _extract_rows(value)
            if rows:
                return rows
    # 单条记录：含任意已知字段号就当一行
    if any(field_key in payload for field_key in TJ_ALL_FIELDS):
        return [dict(payload)]
    return []


def parse_tj_payload(payload: Any) -> list[dict]:
    """天津开放数据 JSON → 统一 meta 列表。

    * `code != 200` 抛 `CrawlError(msg)`（无令牌就是这条路径：`令牌不能为空`）；
    * 兼容 `data` / `rows` / `result` 包裹与 31 个真实字段号；
    * 附件 URL 优先取 `..._000007012`（合同附件链接），其次 `..._000007031`（附件下载）。
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise CrawlError(f"接口返回的不是 JSON：{exc}") from exc
    elif isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CrawlError(f"接口返回的不是 JSON：{exc}") from exc

    if isinstance(payload, Mapping):
        code = payload.get("code")
        if code is not None:
            try:
                ok = int(code) == 200
            except (TypeError, ValueError):
                ok = str(code) in ("200", "0", "success", "ok")
            if not ok:
                msg = str(payload.get("msg") or payload.get("message") or "接口返回非 200")
                raise CrawlError(f"天津开放数据接口拒绝：{msg}（code={code}）")

    rows = _extract_rows(payload)
    out: list[dict] = []
    for row in rows:
        meta: dict[str, Any] = {}
        for suffix, name in TJ_FIELDS.items():
            meta[name] = _cell_value(row.get(f"{_TJ_PREFIX}{suffix}"))
        attachment = meta.get("attachment_url") or meta.get("download_url") or ""
        if attachment and not attachment.startswith(("http://", "https://", "//")):
            attachment = _abs_url(TJ_PAGE, attachment)
        if attachment:
            attachment = _abs_url(TJ_PAGE, attachment)
            if meta.get("attachment_url"):
                meta["attachment_url"] = attachment
            else:
                meta["download_url"] = attachment
        meta["title"] = meta.get("title") or meta.get("project_name") or ""
        meta["source"] = "tj_open_data"
        meta["contract_id"] = detail_id(attachment or meta.get("contract_no")
                                        or meta.get("title") or "")
        meta["detail_url"] = TJ_PAGE
        meta["raw"] = {k: _cell_value(v) for k, v in row.items()}
        out.append(meta)
    return out


# ---------------------------------------------------------------------------
# 详情页解析
# ---------------------------------------------------------------------------

#: 字段 → 候选标签（按优先级）。覆盖政采合同公告的常见写法。
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "contract_no": ("合同编号", "采购合同编号", "合同编码", "项目编号"),
    "contract_name": ("合同名称", "合同标题"),
    "project_name": ("项目名称", "合同项目名称", "采购项目名称"),
    "purchaser": ("采购人", "采购单位", "采购人名称", "甲方", "招标人", "采购单位名称"),
    "supplier": ("供应商", "中标供应商", "成交供应商", "供应商名称", "中标人", "乙方"),
    "amount": ("合同金额", "合同价款", "成交金额", "合同总金额", "合同价格", "总金额"),
    "sign_date": ("合同签订日期", "签订日期", "合同签署日期", "签署日期", "签约日期"),
    "publish_date": ("合同公告日期", "公告日期", "发布日期", "发布时间"),
    "project_no": ("项目编号",),
}

#: "标签：值" 的标签形态：CJK/字母开头，最多 19 字，允许括注，**不含空格**。
#: 负向后视要求标签左边不是 CJK/字母，避免从"六、合同主要信息 采购人（甲方）"里
#: 从"六"开始吞掉一整段（那会让值全部错位）。
_LABEL_RE = re.compile(
    r"(?<![\u4e00-\u9fa5A-Za-z])([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9（）()]{0,18})[：:]")

_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_H1_RE = re.compile(r"(?is)<h[12][^>]*>(.*?)</h[12]>")

#: 章节序号（"二、"、"五、"）也是值的硬边界：政采公告里它紧挨下一个标签，
#: 不切掉就会变成 "…-001 二" 这种脏值。
_ORDINAL_RE = re.compile(r"[一二三四五六七八九十]{1,3}、")
#: 页面 UI 噪音标签，不当字段。
_LABEL_BLOCKLIST = ("字号", "免责声明", "打印本页", "分享到", "字体", "扫一扫")
#: 金额只取"数字 + 单位"那一段，后面的"履约期限：…"一律丢掉。
_AMOUNT_RE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:万元|万|元)?")
#: 编号只取字母数字串（合同编号/项目编号里常有空格与后续中文）。
_CODE_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z\-_/.]{4,}")


def _split_labeled(flat: str) -> list[tuple[str, str, int]]:
    """把单行文本切成 `(归一标签, 值, 标签起始位置)` 序列。

    值 = 到**下一个边界**为止的片段。边界既包括下一个标签，也包括"二、"这类章节序号
    （否则序号会粘在上一个值的尾巴上）。政采页面里"合同金额："与"134.16 万元"常被拆
    在相邻 `<td>` / `<span>` 里，压成单行后天然接上；用"下一个边界"当右边界比固定
    长度更稳。
    """
    matches = list(_LABEL_RE.finditer(flat))
    boundaries = sorted({m.start() for m in matches}
                        | {m.start() for m in _ORDINAL_RE.finditer(flat)})
    out: list[tuple[str, str, int]] = []
    for match in matches:
        label = _norm_label(match.group(1))
        if not label or any(bad in label for bad in _LABEL_BLOCKLIST):
            continue
        start = match.end()
        later = [pos for pos in boundaries if pos > start]
        end = later[0] if later else len(flat)
        value = flat[start:end].strip(" \t·、;；")
        cut = _VALUE_END_RE.search(value)
        if cut:
            value = value[:cut.start()].strip(" \t·、;；")
        out.append((label, value, match.start()))
    return out

def _pick(label_value: list[tuple[str, str, int]], aliases: Sequence[str]) -> str:
    """按别名优先级取值：先精确匹配，再前缀/包含匹配。"""
    for alias in aliases:
        for label, value, _pos in label_value:
            if label == alias and value:
                return value
    for alias in aliases:
        for label, value, _pos in label_value:
            if alias in label and value:
                return value
    return ""


def parse_detail(html: str) -> dict:
    """详情页 → 统一 meta（合同名称/编号/项目/双方/金额/签订日期…）。

    纯正则 + 标签切分，不依赖 lxml/bs4：政采页面结构二十年没变，但 DOM 路径常改，
    按"标签：值"抽比按 XPath 抽更耐改版。
    """
    html_text = html or ""
    flat = _flatten_html(html_text)
    labeled = _split_labeled(flat)

    meta: dict[str, Any] = {}
    for name, aliases in FIELD_ALIASES.items():
        value = _pick(labeled, aliases)
        if value:
            meta[name] = value[:200]

    # 金额只留"数字 + 单位"：政采公告里金额后面常紧跟"履约期限：…"。
    if meta.get("amount"):
        found = _AMOUNT_RE.search(meta["amount"])
        if found:
            meta["amount"] = found.group(0).strip()
    # 编号只留字母数字串，去掉粘上来的中文与空格。
    for name in ("contract_no", "project_no"):
        if meta.get(name):
            found = _CODE_RE.search(meta[name])
            if found:
                meta[name] = found.group(0).strip("-/._")

    title = ""
    for regex in (_TITLE_RE, _H1_RE):
        found = regex.search(html_text)
        if found:
            candidate = _flatten_html(found.group(1))
            if candidate and len(candidate) > 1:
                title = candidate
                break
    meta["title"] = title or meta.get("contract_name") or meta.get("project_name") or ""

    amount = meta.get("amount", "")
    if amount:
        value = T.money_to_float(amount)
        if value is not None:
            meta["amount_value"] = value
    if meta.get("sign_date"):
        meta["sign_date"] = T.parse_date_text(meta["sign_date"])
    if meta.get("publish_date"):
        meta["publish_date"] = T.parse_date_text(meta["publish_date"])

    meta["labels"] = {label: value for label, value, _pos in labeled if value}
    return meta


def attachments(html: str, base_url: str = "") -> list[str]:
    """详情页里的附件链接（.pdf/.jpg/.png/.tif/.bmp/.doc/.docx），按出现顺序去重。"""
    out: list[str] = []
    for match in _HREF_RE.finditer(html or ""):
        href = (match.group(1) or match.group(2) or "").strip()
        if not href:
            continue
        clean = href.split("?")[0].split("#")[0].lower()
        if not clean.endswith(ATTACH_EXT):
            continue
        url = _abs_url(base_url, href) if base_url else _abs_url("http://localhost/", href)
        if url and url not in out:
            out.append(url)
    return out


# ---------------------------------------------------------------------------
# 状态化去重
# ---------------------------------------------------------------------------

@dataclass
class CrawlState:
    """抓取状态：`cache/crawl_state.json`，跨进程保留，避免重复下载。"""

    path: Path = field(default_factory=lambda: CACHE_DIR / "crawl_state.json")
    done_ids: list[str] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    last_run: str = ""
    version: int = 1

    @classmethod
    def load(cls, path: Path | str | None = None) -> "CrawlState":
        target = Path(path) if path else (CACHE_DIR / "crawl_state.json")
        state = cls(path=target)
        if target.exists():
            try:
                raw = json.loads(target.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return state
            state.done_ids = list(raw.get("done_ids") or [])
            state.failed_urls = list(raw.get("failed_urls") or [])
            state.failures = dict(raw.get("failures") or {})
            state.last_run = str(raw.get("last_run") or "")
        return state

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "done_ids": sorted(set(self.done_ids)),
            "failed_urls": sorted(set(self.failed_urls)),
            "failures": dict(self.failures),
            "last_run": self.last_run or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return self.path

    def is_done(self, contract_id: str) -> bool:
        return contract_id in set(self.done_ids)

    def mark_done(self, contract_id: str) -> None:
        if contract_id and contract_id not in self.done_ids:
            self.done_ids.append(contract_id)

    def mark_failed(self, url: str, reason: str) -> None:
        if url and url not in self.failed_urls:
            self.failed_urls.append(url)
        if url:
            self.failures[url] = reason[:300]

    def touch(self) -> None:
        self.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 爬虫
# ---------------------------------------------------------------------------

class Crawler:
    """多源爬虫：列表 → 详情 → 附件 → 落盘（`out_dir/<sha1(url)[:12]>/`）。"""

    def __init__(self, out_dir: Path | str, sources: Sequence[str | Source] | None = None,
                 rate: float = 1.0, timeout: float = 20.0, *,
                 token: str | None = None, retries: int = 3,
                 state: CrawlState | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.rate = max(0.0, float(rate))
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.token = (token or os.environ.get("ZLQL_TJ_TOKEN") or "").strip()
        self._last_request = 0.0
        self._last_encoding = "utf-8"
        self.degraded: list[str] = []
        self.state = state or CrawlState.load()

        if sources is None:
            wanted: list[Source] = [SOURCES["ccgp_beijing"], SOURCES["ccgp_national"],
                                    SOURCES["tj_open_data"]]
        else:
            resolved: list[Source] = []
            for item in sources:
                resolved.append(get_source(item) if isinstance(item, str) else item)
            wanted = resolved
        self.sources = wanted

    # -- HTTP -------------------------------------------------------------
    def _throttle(self) -> None:
        if self.rate <= 0:
            return
        interval = 1.0 / self.rate
        delta = time.time() - self._last_request
        if delta < interval:
            time.sleep(interval - delta)
        self._last_request = time.time()

    def fetch(self, url: str, *, method: str = "GET", data: bytes | None = None,
              headers: Mapping[str, str] | None = None) -> bytes:
        """取字节流。3 次指数退避重试（0.8s→1.6s→3.2s），全失败抛 `CrawlError`。"""
        base_headers = {
            "User-Agent": DEFAULT_UA,
            "Accept": "text/html,application/json,application/pdf,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Connection": "close",
        }
        if headers:
            base_headers.update({k: v for k, v in headers.items() if v})
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                request = urllib.request.Request(url, data=data, headers=base_headers,
                                                 method=method.upper())
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    self._last_encoding = self._sniff_encoding(
                        raw, dict(response.headers or {}))
                    return raw
            except Exception as exc:  # URLError / HTTPError / timeout / ssl
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(8.0, 0.8 * (2 ** attempt)))
        raise CrawlError(f"取不到 {url}：{type(last).__name__}: {last}")

    @staticmethod
    def _sniff_encoding(raw: bytes, headers: Mapping[str, str]) -> str:
        """编码判定：BOM → header charset → meta charset → 有序试探。"""
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"
        content_type = ""
        for key, value in headers.items():
            if str(key).lower() == "content-type":
                content_type = str(value)
                break
        match = re.search(r"charset=([\w-]+)", content_type, re.I)
        if match:
            return match.group(1).lower()
        match = _META_CHARSET_RE.search(raw[:4096])
        if match:
            return match.group(1).decode("ascii", "ignore").lower()
        try:
            raw.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "gb18030"

    def fetch_text(self, url: str, **kwargs: Any) -> str:
        """取文本：用 `_sniff_encoding` 的顺序 + `gb18030` 兜底解出中文。"""
        raw = self.fetch(url, **kwargs)
        encoding = self._last_encoding or "utf-8"
        try:
            return raw.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError):
            for candidate in ("utf-8", "gb18030", "big5"):
                try:
                    return raw.decode(candidate)
                except (UnicodeDecodeError, LookupError):
                    continue
            return raw.decode("utf-8", errors="replace")

    # -- 列表 -------------------------------------------------------------
    def list_page(self, source: Source, page: int) -> list[str]:
        """取一页的详情 URL 列表。

        * html 源：拉列表页 → `detail_re` 找详情链接（无正则时用通用规律兜底）；
        * api 源：带令牌拉 JSON → `parse_tj_payload` → 返回附件/详情 URL。
        """
        if source.kind == "api":
            return self._list_api(source, page)

        template = source.list_url or source.base
        url = template.format(page=page) if "{page}" in template else template
        html_text = self.fetch_text(url)
        return self._extract_detail_links(html_text, source, url)

    def _extract_detail_links(self, html_text: str, source: Source, page_url: str) -> list[str]:
        pattern = source.detail_re or r"""href\s*=\s*["']([^"']+\.htm)"""
        try:
            regex = re.compile(pattern)
        except re.error:
            self.degraded.append(f"[{source.key}] 详情正则编译失败，改用通用规律：{pattern}")
            regex = re.compile(r"""href\s*=\s*["']([^"']+\.htm)""")
        links: list[str] = []
        for match in regex.finditer(html_text or ""):
            href = match.group(1) if match.groups() else match.group(0)
            url = _abs_url(page_url, href)
            if url and url not in links:
                links.append(url)
        return links

    def _list_api(self, source: Source, page: int) -> list[str]:
        """JSON 接口列表。无令牌直接抛 `CrawlError`，由 `run()` 记 degraded 并降级。"""
        token = self.token or os.environ.get(source.token_env, "").strip()
        if not token:
            raise CrawlError(
                f"[{source.key}] 缺少令牌（环境变量 {source.token_env} 或 --token）；"
                f"该接口裸调返回 {{\"code\":500,\"msg\":\"令牌不能为空\",\"data\":null}}"
            )
        query = urllib.parse.urlencode({
            source.token_param: token,
            "page": page,
            "pageSize": 20,
        })
        url = f"{source.base}?{query}"
        raw = self.fetch(url, headers={"Accept": "application/json"})
        rows = parse_tj_payload(raw)
        urls: list[str] = []
        for row in rows:
            candidate = row.get("attachment_url") or row.get("download_url") or row.get("detail_url")
            if candidate and candidate not in urls:
                urls.append(candidate)
        return urls

    # -- 详情 -------------------------------------------------------------
    def fetch_detail(self, url: str) -> str:
        return self.fetch_text(url)

    # -- 落盘 -------------------------------------------------------------
    def save_contract(self, meta: Mapping[str, Any], pdf_bytes: bytes,
                      *, extension: str = ".pdf") -> Path:
        """按 v1 约定落盘：`out_dir/<sha1(url)[:12]>/{meta.json,pages/,text/}`。

        页面图优先抽 PDF 内嵌图（`source="embedded"`，保真且快）；抽不到就整页渲染
        `dpi=200`（`source="rendered"`）。文字层同步写成 `text/page_NNNN.txt`，
        扫描页写空文件 —— 这与 `data/gov_contracts` 里已有的 63 份完全一致。
        """
        key_source = str(meta.get("detail_url") or meta.get("pdf_url")
                         or meta.get("attachment_url") or meta.get("contract_id") or "")
        folder = self.out_dir / sanitize_filename(key_source)
        pages_dir = folder / "pages"
        text_dir = folder / "text"
        pages_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        for old in list(pages_dir.glob("*")):
            old.unlink(missing_ok=True)
        for old in list(text_dir.glob("*")):
            old.unlink(missing_ok=True)

        if pdf_bytes:
            (folder / f"source{extension}").write_bytes(pdf_bytes)

        page_records: list[dict] = []
        suffix = extension.lower()
        if suffix == ".pdf" or pdf_bytes[:4] == b"%PDF":
            page_records = self._extract_pdf_pages(pdf_bytes, pages_dir, text_dir)
        elif suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
            image_suffix = ".jpeg" if suffix in (".jpg", ".jpeg") else suffix
            target = pages_dir / f"page_0001{image_suffix}"
            target.write_bytes(pdf_bytes)
            (text_dir / "page_0001.txt").write_text("", encoding="utf-8")
            page_records = [{"page": 1, "width": 0, "height": 0,
                             "format": image_suffix.lstrip("."), "source": "raw",
                             "text_len": 0}]

        meta_out = dict(meta)
        meta_out["contract_id"] = meta_out.get("contract_id") or sanitize_filename(key_source)
        meta_out["page_count"] = len(page_records)
        meta_out["scanned_pages"] = sum(1 for p in page_records if p.get("text_len", 0) == 0)
        meta_out["text_pages"] = sum(1 for p in page_records if p.get("text_len", 0) > 0)
        meta_out["pages"] = page_records
        meta_out.setdefault("crawled_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        (folder / "meta.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
        return folder

    def _extract_pdf_pages(self, pdf_bytes: bytes, pages_dir: Path,
                           text_dir: Path) -> list[dict]:
        """PDF → 每页一张图 + 一份文字层。pymupdf 缺失时返回空列表并记 degraded。"""
        try:
            import pymupdf  # type: ignore
        except Exception:
            self.degraded.append("pymupdf 不可用，无法抽取 PDF 页面图（已保留 source.pdf）")
            return []
        records: list[dict] = []
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            self.degraded.append(f"PDF 打不开：{type(exc).__name__}: {exc}")
            return []
        with doc:
            for index, page in enumerate(doc, start=1):
                ext = ".jpeg"
                source_kind = "rendered"
                width = height = 0
                try:
                    pix = None
                    images = page.get_images(full=True)
                    if images:
                        biggest = max(images, key=lambda info: (info[2] or 0) * (info[3] or 0))
                        xref = biggest[0]
                        info = doc.extract_image(xref)
                        if info and info.get("image"):
                            ext = "." + (info.get("ext") or "png").lower()
                            if ext in (".jfif", ".pjpeg"):
                                ext = ".jpeg"
                            (pages_dir / f"page_{index:04d}{ext}").write_bytes(info["image"])
                            width, height = int(info.get("width") or 0), int(info.get("height") or 0)
                            source_kind = "embedded"
                    if source_kind != "embedded":
                        pix = page.get_pixmap(dpi=200)
                        ext = ".jpeg"
                        pix.save(str(pages_dir / f"page_{index:04d}{ext}"))
                        width, height = int(pix.width), int(pix.height)
                except Exception as exc:
                    self.degraded.append(f"第 {index} 页抽图失败：{type(exc).__name__}: {exc}")
                    continue
                try:
                    layer = page.get_text().strip()
                except Exception:
                    layer = ""
                (text_dir / f"page_{index:04d}.txt").write_text(layer, encoding="utf-8")
                records.append({
                    "page": index,
                    "width": width,
                    "height": height,
                    "format": ext.lstrip("."),
                    "source": source_kind,
                    "text_len": len(layer),
                })
        return records

    # -- 主管线 -----------------------------------------------------------
    def run(self, limit: int = 10, pages: int = 1, use_ocr: bool = False) -> dict:
        """遍历 sources → 列表 → 详情 → 附件 → 保存。

        任何网络/解析异常都记进 `degraded`，函数始终返回统计字典（不抛）。
        """
        started = time.time()
        stats: dict[str, Any] = {
            "contracts": 0, "pages": 0, "scanned": 0, "skipped": 0,
            "failed": 0, "ocr_pages": 0, "degraded": [], "sources": {},
            "out_dir": str(self.out_dir), "seconds": 0.0,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.degraded = []

        for source in self.sources:
            if stats["contracts"] >= limit:
                break
            per_source = {"page_urls": 0, "done": 0, "failed": 0}
            for page in range(1, max(1, pages) + 1):
                if stats["contracts"] >= limit:
                    break
                try:
                    urls = self.list_page(source, page)
                except CrawlError as exc:
                    text = str(exc)
                    prefix = "" if text.startswith(f"[{source.key}]") else f"[{source.key}] "
                    self.degraded.append(f"{prefix}列表第 {page} 页失败：{text}")
                    break
                except Exception as exc:  # 兜底，绝不让 run 抛
                    self.degraded.append(
                        f"[{source.key}] 列表第 {page} 页异常：{type(exc).__name__}: {exc}")
                    break
                per_source["page_urls"] += len(urls)

                for url in urls:
                    if stats["contracts"] >= limit:
                        break
                    cid = detail_id(url)
                    if self.state.is_done(cid):
                        stats["skipped"] += 1
                        continue
                    outcome = self._process_one(source, url, cid, stats, use_ocr=use_ocr)
                    if outcome == "done":
                        per_source["done"] += 1
                    elif outcome == "failed":
                        per_source["failed"] += 1
                    elif outcome == "skip":
                        stats["skipped"] += 1
            stats["sources"][source.key] = per_source

        missing_token = [s.key for s in self.sources if s.needs_token and not self.token]
        if missing_token:
            public_keys = [s.key for s in self.sources if not s.needs_token]
            tail = ("，已自动降级到公开栏目：" + ", ".join(public_keys)) if public_keys \
                else "，本次没有配置任何公开栏目可用"
            self.degraded.append("缺少令牌的源已跳过：" + ", ".join(missing_token) + tail)

        self.state.touch()
        try:
            self.state.save()
        except OSError as exc:
            self.degraded.append(f"状态文件写入失败：{exc}")

        stats["degraded"] = list(dict.fromkeys(self.degraded))
        stats["seconds"] = round(time.time() - started, 2)
        return stats

    def _process_one(self, source: Source, url: str, cid: str,
                     stats: dict[str, Any], *, use_ocr: bool) -> str:
        """处理一份详情页。返回 `done` / `failed` / `skip`。"""
        try:
            html_text = self.fetch_detail(url)
        except CrawlError as exc:
            stats["failed"] += 1
            self.state.mark_failed(url, str(exc))
            self.degraded.append(f"[{source.key}] 详情失败：{exc}")
            return "failed"

        meta = parse_detail(html_text)
        meta["source"] = source.key
        meta["source_name"] = source.name
        meta["detail_url"] = url
        meta["contract_id"] = cid

        files = attachments(html_text, url)
        payload = b""
        extension = ".pdf"
        chosen = ""
        for candidate in files:
            suffix = Path(urllib.parse.urlparse(candidate).path).suffix.lower()
            if suffix not in ATTACH_EXT:
                continue
            try:
                blob = self.fetch(candidate)
            except CrawlError as exc:
                self.degraded.append(f"[{source.key}] 附件下载失败：{exc}")
                continue
            if len(blob) > MAX_ATTACH_BYTES:
                self.degraded.append(
                    f"[{source.key}] 附件过大跳过（{len(blob) / 1048576:.1f}MB）：{candidate}")
                continue
            payload, extension, chosen = blob, suffix, candidate
            break

        if not payload:
            # 详情页本身就是合同正文（无附件）时，把 HTML 存成 txt 页，不让样本白跑。
            body = _flatten_html(html_text)
            pages_dir = self.out_dir / sanitize_filename(url) / "pages"
            text_dir = self.out_dir / sanitize_filename(url) / "text"
            pages_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / sanitize_filename(url) / "meta.json").write_text(
                json.dumps({**meta, "page_count": 0, "pages": []}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            (text_dir / "page_0001.txt").write_text(body, encoding="utf-8")
            self.state.mark_done(cid)
            stats["contracts"] += 1
            self.degraded.append(
                f"[{source.key}] {url} 未找到 PDF/图片附件，仅保留详情页文本")
            return "done"

        meta["attachment_url"] = chosen
        folder = self.save_contract(meta, payload, extension=extension)

        meta_file = folder / "meta.json"
        try:
            saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved_meta = dict(meta)
        page_count = int(saved_meta.get("page_count") or 0)
        scanned = int(saved_meta.get("scanned_pages") or 0)
        stats["contracts"] += 1
        stats["pages"] += page_count
        stats["scanned"] += scanned

        if use_ocr and scanned > 0:
            stats["ocr_pages"] += self._ocr_saved(folder, saved_meta, stats)

        self.state.mark_done(cid)
        return "done"

    def _ocr_saved(self, folder: Path, meta: Mapping[str, Any],
                   stats: dict[str, Any]) -> int:
        """对已落盘的扫描页做 OCR，识别文本写回 `text/page_NNNN.txt`。

        走 `ocr.merge_ocr_into_parse` 的同一套合并逻辑（这样爬虫侧和审查管线侧对
        "OCR 块怎么并进 ParseOutput"只有一份实现），再把 `stats["ocr"]` 落进 meta.json，
        报告里能直接引用 `engine / avg_conf / pages`。
        """
        try:
            from . import ocr as ocr_mod
            from . import parsers as parsers_mod
        except Exception as exc:  # pragma: no cover
            self.degraded.append(f"OCR 模块导入失败：{exc}")
            return 0

        source_pdf = folder / "source.pdf"
        if not source_pdf.exists():
            self.degraded.append(f"{folder.name} 没有 source.pdf，跳过 OCR")
            return 0
        needs = [int(p["page"]) for p in meta.get("pages", []) if not p.get("text_len")]
        if not needs:
            return 0

        budget = ocr_mod.OcrBudget(max_pages=len(needs), max_seconds=180.0, dpi=200)
        engine = ocr_mod.get_engine()
        result = engine.recognize_pdf(source_pdf, budget, pages=needs,
                                      prefer_text_layer=False)
        for note in result.degraded:
            self.degraded.append(f"[{folder.name}] OCR 降级：{note}")

        # 原始 ParseOutput：全扫描件时 parse_path 会抛（无文字层），退化成空壳继续。
        try:
            base_output = parsers_mod.parse_path(source_pdf)
        except Exception:
            base_output = parsers_mod.ParseOutput(
                blocks=[], file_type="pdf", page_sizes={},
                needs_ocr_pages=list(needs), image_blocks=[], stats={}, warnings=[])
        merged = ocr_mod.merge_ocr_into_parse(base_output, result)

        written = 0
        for page in result.pages:
            if page.source == "skip" or not page.text.strip():
                continue
            target = folder / "text" / f"page_{page.page:04d}.txt"
            target.write_text(page.text, encoding="utf-8")
            written += 1

        try:
            meta_file = folder / "meta.json"
            saved = json.loads(meta_file.read_text(encoding="utf-8"))
            saved["ocr"] = dict((merged.stats or {}).get("ocr") or {})
            saved["ocr_pages"] = written
            for record in saved.get("pages", []):
                if int(record.get("page", 0)) in {p.page for p in result.pages
                                                  if p.text.strip()}:
                    record["text_len"] = len(
                        next(p.text for p in result.pages
                             if p.page == record["page"]).strip())
                    record["text_source"] = "ocr"
            meta_file.write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except (OSError, json.JSONDecodeError, StopIteration) as exc:
            self.degraded.append(f"[{folder.name}] meta.json 回写 OCR 统计失败：{exc}")

        return written


# ---------------------------------------------------------------------------
# 便捷入口 + 命令行
# ---------------------------------------------------------------------------

def crawl(limit: int = 10, pages: int = 1, *, out_dir: Path | str | None = None,
          sources: Sequence[str | Source] | None = None, use_ocr: bool = False,
          token: str | None = None, rate: float = 1.0) -> dict:
    """一行调用：`crawl(limit=2, pages=1)`。"""
    ensure_dirs()
    target = Path(out_dir) if out_dir else (CACHE_DIR.parent / "crawl")
    crawler = Crawler(target, sources=sources, rate=rate, token=token)
    return crawler.run(limit=limit, pages=pages, use_ocr=use_ocr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zhilv_qiling_v2.crawl",
        description="智律契灵 v2 · 多源合同爬虫（离线可跑；无令牌自动降级）")
    parser.add_argument("--source", default="",
                        help="逗号分隔的源 key，默认全部："
                             + ",".join(SOURCES))
    parser.add_argument("--limit", type=int, default=10, help="最多抓多少份合同")
    parser.add_argument("--pages", type=int, default=1, help="每个源翻几页列表")
    parser.add_argument("--out", default="", help="输出目录（默认 data/crawl）")
    parser.add_argument("--ocr", action="store_true", help="对扫描页做 OCR 并写回文字")
    parser.add_argument("--token", default="", help="天津开放数据令牌（也可用 ZLQL_TJ_TOKEN）")
    parser.add_argument("--rate", type=float, default=1.0, help="每秒请求数上限")
    args = parser.parse_args(list(argv) if argv is not None else None)

    sources = [s.strip() for s in args.source.split(",") if s.strip()] or None
    if args.token:
        os.environ["ZLQL_TJ_TOKEN"] = args.token

    result = crawl(limit=args.limit, pages=args.pages, out_dir=args.out or None,
                   sources=sources, use_ocr=args.ocr, token=args.token or None,
                   rate=args.rate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
