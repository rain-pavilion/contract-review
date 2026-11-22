"""`crawl.py` 的单元测试（unittest，**全程不联网**）。

覆盖：数据源定义、天津 31 字段解析、`parse_detail` 标签抽取、附件识别、状态化去重、
落盘结构、`run()` 在无网络时"返回 degraded 而不是抛"。

跑法：`python -m unittest zhilv_qiling_v2.tests_v2.test_crawl -v`
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zhilv_qiling_v2 import crawl as C


# ---------------------------------------------------------------------------
# 夹具：两段真实结构的 HTML（字段名照政采公告抄，值换成可验证的假数据）
# ---------------------------------------------------------------------------

DETAIL_HTML_BJ = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>某某单位购置设备项目政府采购合同</title></head>
<body>
<div class="content">
  <p>一、合同编号：11000026210200176848-XM001-001</p>
  <p>二、合同名称：某某单位购置设备项目政府采购合同</p>
  <p>三、项目编号：11000026210200176848-XM001</p>
  <p>四、项目名称：某某单位购置设备项目</p>
  <p>五、合同主体</p>
  <p>采购人（甲方）：北京示例学院</p>
  <p>地 址：北京市东城区示例路 1 号</p>
  <p>联系方式：010-00000000</p>
  <p>供应商（乙方）：北京示例科技有限公司</p>
  <p>地 址：北京市海淀区示例大厦</p>
  <p>六、合同主要信息</p>
  <p>主要标的名称：示例设备</p>
  <p>合同金额：134.1600000 万元</p>
  <p>履约期限：合同签订后 30 日</p>
  <p>七、合同签订日期：2026-10-27</p>
  <p>八、合同公告日期：2026-11-03</p>
  <p><a href="/content_file/xxgg/2026/11/97226b0a.pdf">合同附件</a></p>
  <p><a href="//www.ccgp-beijing.gov.cn/content_file/xxgg/2026/11/abc123.jpg">扫描件</a></p>
  <p><a href="/xxgg/sjxxgg/htgg/2026/11/other.htm">相关公告</a></p>
</div>
</body></html>
"""

DETAIL_HTML_ALT = """<html><head><title>备用标签合同</title></head><body>
<table>
  <tr><td>合同价款：</td><td>98.5万元</td></tr>
  <tr><td>采购单位：</td><td>天津示例局</td></tr>
  <tr><td>中标供应商：</td><td>天津示例公司</td></tr>
  <tr><td>项目编号：</td><td>TJ-2026-0001</td></tr>
  <tr><td>合同签订日期：</td><td>2026年9月8日</td></tr>
  <tr><td>项目名称：</td><td>天津示例采购项目</td></tr>
</table></body></html>
"""


def tj_row(**overrides: str) -> dict:
    """造一行天津开放数据的真实字段号。"""
    row = {
        "f_3070020000012_000007001": "2026-05-06",
        "f_3070020000012_000007007": "天津市示例单位",
        "f_3070020000012_000007008": "示例采购项目",
        "f_3070020000012_000007009": "1280000",
        "f_3070020000012_000007011": "示例采购合同",
        "f_3070020000012_000007012": "/files/contract/demo.pdf",
        "f_3070020000012_000007013": "TJHT-2026-0007",
        "f_3070020000012_000007029": "91120000MA0000000X",
        "f_3070020000012_000007031": "",
        "f_3070020000012_000007099": "等外字段",
    }
    row.update(overrides)
    return row


def tiny_pdf(pages: int = 2) -> bytes:
    """用 pymupdf 现造一个带文字层的 PDF（离线、无需外部文件）。"""
    import pymupdf

    doc = pymupdf.open()
    for index in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {index} contract text layer", fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------

class TestSources(unittest.TestCase):
    """三个源必须在位，且实测有效的 URL 规律要固化在定义里。"""

    def test_required_sources(self) -> None:
        for key in ("tj_open_data", "ccgp_national", "ccgp_beijing"):
            self.assertIn(key, C.SOURCES)
            source = C.SOURCES[key]
            self.assertIsInstance(source, C.Source)
            self.assertIn(source.kind, ("api", "html"))

    def test_tj_requires_token(self) -> None:
        tj = C.SOURCES["tj_open_data"]
        self.assertTrue(tj.needs_token)
        self.assertIn("43651ffd8e454d328544e4a5bbd39c53", tj.base)
        self.assertEqual(tj.token_env, "ZLQL_TJ_TOKEN")
        self.assertEqual(C.get_source("tj_open_data"), tj)

    def test_public_sources_need_no_token(self) -> None:
        self.assertFalse(C.SOURCES["ccgp_national"].needs_token)
        self.assertFalse(C.SOURCES["ccgp_beijing"].needs_token)
        self.assertEqual(C.SOURCES["ccgp_national"].base, "http://www.ccgp.gov.cn/")
        self.assertIn("A002004001003index_{page}.htm",
                      C.SOURCES["ccgp_beijing"].list_url)

    def test_beijing_detail_regex_matches_real_url(self) -> None:
        import re

        pattern = re.compile(C.SOURCES["ccgp_beijing"].detail_re)
        url = ("//www.ccgp-beijing.gov.cn/xxgg/sjxxgg/htgg/2026/11/"
               "e95602a7de4d4f2f88569f15aa6270c7.htm")
        self.assertTrue(pattern.search(url))
        self.assertFalse(pattern.search(
            "//www.ccgp-beijing.gov.cn/xxgg/sjxxgg/htgg/2026/11/not-a-hex-slug.htm"))

    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(C.CrawlError):
            C.get_source("nope")


# ---------------------------------------------------------------------------
# 天津 API 解析
# ---------------------------------------------------------------------------

class TestParseTjPayload(unittest.TestCase):
    """31 个真实字段的映射 + 三种包裹形态 + 非 200 抛错。"""

    def test_data_wrapper(self) -> None:
        rows = C.parse_tj_payload({"code": 200, "msg": "ok", "data": [tj_row()]})
        self.assertEqual(len(rows), 1)
        meta = rows[0]
        self.assertEqual(meta["sign_date"], "2026-05-06")
        self.assertEqual(meta["purchaser"], "天津市示例单位")
        self.assertEqual(meta["project_name"], "示例采购项目")
        self.assertEqual(meta["amount"], "1280000")
        self.assertEqual(meta["title"], "示例采购合同")
        self.assertEqual(meta["contract_no"], "TJHT-2026-0007")
        self.assertEqual(meta["credit_code"], "91120000MA0000000X")
        self.assertEqual(meta["source"], "tj_open_data")
        self.assertTrue(meta["attachment_url"].endswith("/files/contract/demo.pdf"))
        self.assertTrue(meta["attachment_url"].startswith("https://open.data.tj.gov.cn"))

    def test_rows_and_result_wrappers(self) -> None:
        for payload in ({"code": 200, "rows": [tj_row()]},
                        {"code": 200, "result": {"list": [tj_row()]}},
                        {"code": 200, "data": {"total": 1, "rows": [tj_row()]}}):
            rows = C.parse_tj_payload(payload)
            self.assertEqual(len(rows), 1, payload)
            self.assertEqual(rows[0]["contract_no"], "TJHT-2026-0007")

    def test_bare_list_payload(self) -> None:
        rows = C.parse_tj_payload([tj_row()])
        self.assertEqual(len(rows), 1)

    def test_download_url_fallback(self) -> None:
        row = tj_row(**{"f_3070020000012_000007012": "",
                        "f_3070020000012_000007031": "http://files.example.com/a.pdf"})
        meta = C.parse_tj_payload({"code": 200, "data": [row]})[0]
        self.assertEqual(meta["download_url"], "http://files.example.com/a.pdf")
        self.assertEqual(meta["attachment_url"], "")

    def test_cell_objects(self) -> None:
        row = tj_row(**{"f_3070020000012_000007007": {"value": "对象采购人"},
                        "f_3070020000012_000007009": ["100", "万元"]})
        meta = C.parse_tj_payload({"code": 200, "data": [row]})[0]
        self.assertEqual(meta["purchaser"], "对象采购人")
        self.assertEqual(meta["amount"], "100 万元")

    def test_token_error_raises(self) -> None:
        """无令牌的真实响应：必须变成 `CrawlError`，不能被当成空数据。"""
        with self.assertRaises(C.CrawlError) as ctx:
            C.parse_tj_payload({"code": 500, "msg": "令牌不能为空", "data": None})
        self.assertIn("令牌不能为空", str(ctx.exception))

    def test_non_json_raises(self) -> None:
        with self.assertRaises(C.CrawlError):
            C.parse_tj_payload(b"<html>502 Bad Gateway</html>")

    def test_string_payload(self) -> None:
        rows = C.parse_tj_payload(json.dumps({"code": 200, "data": [tj_row()]}))
        self.assertEqual(rows[0]["purchaser"], "天津市示例单位")

    def test_contract_id_is_stable(self) -> None:
        first = C.parse_tj_payload({"code": 200, "data": [tj_row()]})[0]["contract_id"]
        second = C.parse_tj_payload({"code": 200, "data": [tj_row()]})[0]["contract_id"]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)


# ---------------------------------------------------------------------------
# 详情页解析
# ---------------------------------------------------------------------------

class TestParseDetail(unittest.TestCase):
    """`parse_detail` 要能吃下"标签与值被相邻元素拆开"的政采页面。"""

    def setUp(self) -> None:
        self.meta = C.parse_detail(DETAIL_HTML_BJ)

    def test_core_fields(self) -> None:
        self.assertEqual(self.meta["title"], "某某单位购置设备项目政府采购合同")
        self.assertEqual(self.meta["contract_no"], "11000026210200176848-XM001-001")
        self.assertEqual(self.meta["project_no"], "11000026210200176848-XM001")
        self.assertEqual(self.meta["project_name"], "某某单位购置设备项目")
        self.assertEqual(self.meta["contract_name"], "某某单位购置设备项目政府采购合同")

    def test_parties(self) -> None:
        self.assertEqual(self.meta["purchaser"], "北京示例学院")
        self.assertEqual(self.meta["supplier"], "北京示例科技有限公司")

    def test_amount(self) -> None:
        self.assertEqual(self.meta["amount"], "134.1600000 万元")
        self.assertAlmostEqual(self.meta["amount_value"], 1341600.0, places=2)

    def test_dates(self) -> None:
        self.assertEqual(self.meta["sign_date"], "2026-10-27")
        self.assertEqual(self.meta["publish_date"], "2026-11-03")

    def test_ordinal_not_glued_to_value(self) -> None:
        """`…-001 二` 这种"章节序号粘在值尾巴上"的脏值必须清掉。"""
        for value in self.meta["labels"].values():
            self.assertNotRegex(value, r"[\s][一二三四五六七八九十]+$")

    def test_alt_labels(self) -> None:
        meta = C.parse_detail(DETAIL_HTML_ALT)
        self.assertEqual(meta["purchaser"], "天津示例局")
        self.assertEqual(meta["supplier"], "天津示例公司")
        self.assertEqual(meta["amount"], "98.5万元")
        self.assertAlmostEqual(meta["amount_value"], 985000.0, places=2)
        self.assertEqual(meta["sign_date"], "2026-09-08")
        self.assertEqual(meta["contract_no"], "TJ-2026-0001")
        self.assertEqual(meta["project_name"], "天津示例采购项目")

    def test_empty_html_is_safe(self) -> None:
        meta = C.parse_detail("")
        self.assertEqual(meta["title"], "")
        self.assertNotIn("purchaser", meta)


# ---------------------------------------------------------------------------
# 附件
# ---------------------------------------------------------------------------

class TestAttachments(unittest.TestCase):
    def test_extracts_and_absolutizes(self) -> None:
        found = C.attachments(DETAIL_HTML_BJ,
                              "http://www.ccgp-beijing.gov.cn/xxgg/sjxxgg/htgg/2026/11/x.htm")
        self.assertEqual(found, [
            "http://www.ccgp-beijing.gov.cn/content_file/xxgg/2026/11/97226b0a.pdf",
            "http://www.ccgp-beijing.gov.cn/content_file/xxgg/2026/11/abc123.jpg",
        ])

    def test_all_supported_extensions(self) -> None:
        html = "".join(f'<a href="/f/doc{i}{ext}">x</a>' for i, ext in enumerate(
            (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".doc", ".docx")))
        found = C.attachments(html, "http://h.example.com/a/b.htm")
        self.assertEqual(len(found), 9)

    def test_dedupes_and_skips_pages(self) -> None:
        """同一链接只保留一次；非附件扩展名丢弃；query 串保留（cdn 版本号会影响内容）。"""
        html = ('<a href="/a.pdf">1</a><a href="/a.pdf">2</a>'
                '<a href="/b.htm">3</a><a href="/c.pdf?v=2">4</a>')
        found = C.attachments(html, "http://h.example.com/x/")
        self.assertEqual(found, ["http://h.example.com/a.pdf",
                                "http://h.example.com/c.pdf?v=2"])

    def test_empty(self) -> None:
        self.assertEqual(C.attachments("", "http://h/"), [])


# ---------------------------------------------------------------------------
# 状态化去重
# ---------------------------------------------------------------------------

class TestCrawlState(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crawl_state.json"
            state = C.CrawlState.load(path)
            self.assertEqual(state.done_ids, [])
            state.mark_done("abc123")
            state.mark_done("abc123")
            state.mark_failed("http://x/y.htm", "timeout")
            state.touch()
            state.save()

            again = C.CrawlState.load(path)
            self.assertEqual(again.done_ids, ["abc123"])
            self.assertTrue(again.is_done("abc123"))
            self.assertFalse(again.is_done("zzz"))
            self.assertEqual(again.failed_urls, ["http://x/y.htm"])
            self.assertIn("timeout", again.failures["http://x/y.htm"])
            self.assertTrue(again.last_run)

    def test_corrupt_file_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crawl_state.json"
            path.write_text("{not json", encoding="utf-8")
            state = C.CrawlState.load(path)
            self.assertEqual(state.done_ids, [])


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------

class TestSaveContract(unittest.TestCase):
    def test_pdf_pages_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[], rate=0)
            meta = {"title": "示例合同", "detail_url": "http://h/x/1.htm",
                    "attachment_url": "http://h/x/1.pdf", "purchaser": "甲方"}
            folder = crawler.save_contract(meta, tiny_pdf(2), extension=".pdf")

            self.assertTrue(folder.is_dir())
            self.assertEqual(len(folder.name), 12)
            self.assertTrue((folder / "source.pdf").exists())
            self.assertEqual(sorted(p.name for p in (folder / "pages").glob("page_*")),
                             ["page_0001.png" if (folder / "pages" / "page_0001.png").exists()
                              else "page_0001.jpeg",
                              "page_0002.png" if (folder / "pages" / "page_0002.png").exists()
                              else "page_0002.jpeg"])
            saved = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["page_count"], 2)
            self.assertEqual(saved["contract_id"], folder.name)
            self.assertEqual(saved["purchaser"], "甲方")
            self.assertTrue(saved["crawled_at"])
            for record in saved["pages"]:
                self.assertGreater(record["text_len"], 0, "带文字层的页不该记成扫描页")
            self.assertEqual(saved["scanned_pages"], 0)

    def test_image_attachment_becomes_single_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[], rate=0)
            folder = crawler.save_contract(
                {"detail_url": "http://h/img/2.htm"}, b"\xff\xd8\xff\xe0JFIF", extension=".jpg")
            self.assertTrue((folder / "pages" / "page_0001.jpeg").exists())
            self.assertTrue((folder / "text" / "page_0001.txt").exists())
            saved = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["page_count"], 1)

    def test_same_url_same_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[], rate=0)
            meta = {"detail_url": "http://h/same.htm"}
            first = crawler.save_contract(meta, tiny_pdf(1))
            second = crawler.save_contract(meta, tiny_pdf(1))
            self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# 无网络降级
# ---------------------------------------------------------------------------

class TestOfflineDegrades(unittest.TestCase):
    """网络不通时 `run()` 必须返回带 degraded 的统计，而不是抛。"""

    def offline_source(self) -> C.Source:
        return C.Source(key="offline", name="离线源", kind="html",
                        base="http://127.0.0.1:9/",
                        list_url="http://127.0.0.1:9/index_{page}.htm",
                        detail_re=r"/detail/(\d+)\.htm")

    def test_run_returns_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[self.offline_source()],
                                rate=0, timeout=0.5, retries=1,
                                state=C.CrawlState.load(Path(tmp) / "state.json"))
            result = crawler.run(limit=2, pages=1)
            self.assertEqual(result["contracts"], 0)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(result["degraded"])
            self.assertIn("offline", result["degraded"][0])
            self.assertIn("offline", result["sources"])

    def test_fetch_raises_crawl_error(self) -> None:
        """单请求失败抛 `CrawlError`（由调用方记账），不是裸的 URLError。"""
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[], rate=0, timeout=0.5, retries=1)
            with self.assertRaises(C.CrawlError):
                crawler.fetch("http://127.0.0.1:9/none")

    def test_missing_token_is_degraded_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=["tj_open_data"], rate=0,
                                timeout=0.5, retries=1, token="",
                                state=C.CrawlState.load(Path(tmp) / "state.json"))
            crawler.token = ""
            result = crawler.run(limit=1, pages=1)
            self.assertEqual(result["contracts"], 0)
            joined = " | ".join(result["degraded"])
            self.assertIn("令牌", joined)
            self.assertIn("tj_open_data", joined)

    def test_with_token_it_actually_calls_api(self) -> None:
        """有令牌时不该再走"缺令牌"分支（用假 token 打到不可达地址也一样）。"""
        with tempfile.TemporaryDirectory() as tmp:
            crawler = C.Crawler(Path(tmp), sources=[], rate=0, timeout=0.3,
                                retries=1, token="dummy-token")
            source = C.Source(key="api_probe", name="假API", kind="api",
                              base="http://127.0.0.1:9/api/invoke/x", needs_token=True)
            with self.assertRaises(C.CrawlError) as ctx:
                crawler.list_page(source, 1)
            self.assertNotIn("缺少令牌", str(ctx.exception))


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_abs_url_forms(self) -> None:
        base = "http://www.ccgp-beijing.gov.cn/xxgg/a/b.htm"
        self.assertEqual(C._abs_url(base, "//www.x.cn/1.pdf"),
                         "http://www.x.cn/1.pdf")
        self.assertEqual(C._abs_url(base, "/content/2.pdf"),
                         "http://www.ccgp-beijing.gov.cn/content/2.pdf")
        self.assertEqual(C._abs_url(base, "3.pdf"),
                         "http://www.ccgp-beijing.gov.cn/xxgg/a/3.pdf")
        self.assertEqual(C._abs_url(base, "https://y.cn/4.pdf"), "https://y.cn/4.pdf")
        self.assertEqual(C._abs_url(base, ""), "")

    def test_detail_id_and_folder_name(self) -> None:
        url = "http://h/x/1.htm"
        self.assertEqual(C.detail_id(url), C.sanitize_filename(url))
        self.assertEqual(len(C.detail_id(url)), 12)
        self.assertNotEqual(C.detail_id(url), C.detail_id("http://h/x/2.htm"))

    def test_flatten_html_keeps_word_boundaries(self) -> None:
        flat = C._flatten_html("<td>采购人</td><td>北京示例</td>")
        self.assertEqual(flat, "采购人 北京示例")
        self.assertEqual(C._flatten_html("<p>地 址：x</p>"), "地址：x")

    def test_norm_label(self) -> None:
        self.assertEqual(C._norm_label("采购人（甲方）"), "采购人")
        self.assertEqual(C._norm_label("一、合同编号"), "合同编号")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
