"""`ocr.py` 的单元测试（unittest，离线可跑）。

分两层：
* **纯逻辑**（版面重建 / CER / 预算 / 合并）——不碰模型、不碰磁盘，毫秒级；
* **真引擎**（`data/gov_contracts` 里的真实扫描页）——只在数据存在时跑，缺数据自动
  skip，不给 CI 埋雷。

跑法：`python -m unittest zhilv_qiling_v2.tests_v2.test_ocr -v`
"""

from __future__ import annotations

import unittest
from pathlib import Path

from zhilv_qiling_v2 import ocr
from zhilv_qiling_v2.config import DATA_DIR, WORKSPACE
from zhilv_qiling_v2.models import SourceBlock
from zhilv_qiling_v2.parsers import ParseOutput

#: 真实语料在**工作区** `data/gov_contracts`（`config.DATA_DIR` 是包内数据目录，两者不同）。
GOV = next((candidate for candidate in (WORKSPACE / "data" / "gov_contracts",
                                        DATA_DIR / "gov_contracts")
            if candidate.exists()),
           WORKSPACE / "data" / "gov_contracts")


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

def box(text: str, x0: float, y0: float, x1: float, y1: float,
        conf: float = 0.99) -> dict:
    """造一个 rapidocr 风格的文本框（已归一成 x0y0x1y1）。"""
    return {"text": text, "conf": conf, "bbox": [x0, y0, x1, y1]}


def scanned_pages(limit: int = 2) -> list[Path]:
    """从真实数据里挑扫描页（同名 txt 是 0 字节）——按文件名排序，结果稳定。"""
    out: list[Path] = []
    if not GOV.exists():
        return out
    for txt in sorted(GOV.glob("*/text/page_*.txt")):
        if txt.stat().st_size != 0:
            continue
        stem = txt.stem
        for suffix in (".jpeg", ".png", ".jpg", ".tif"):
            image = txt.parent.parent / "pages" / f"{stem}{suffix}"
            if image.exists():
                out.append(image)
                break
        if len(out) >= limit:
            break
    return out


def text_layer_pages(min_chars: int = 100, limit: int = 5) -> list[tuple[Path, str]]:
    """有文字层的页面 `(图片, 文字层)`，用于 CER 对照。"""
    out: list[tuple[Path, str]] = []
    if not GOV.exists():
        return out
    for txt in sorted(GOV.glob("*/text/page_*.txt")):
        try:
            layer = txt.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if len(layer) < min_chars:
            continue
        stem = txt.stem
        for suffix in (".jpeg", ".png", ".jpg", ".tif"):
            image = txt.parent.parent / "pages" / f"{stem}{suffix}"
            if image.exists():
                out.append((image, layer))
                break
        if len(out) >= limit:
            break
    return out


ENGINES = ocr.available_engines()
HAS_ENGINE = any(ENGINES.values())


# ---------------------------------------------------------------------------
# 引擎探测
# ---------------------------------------------------------------------------

class TestEngines(unittest.TestCase):
    """`available_engines` 只能返回布尔，绝不允许把探测失败变成异常。"""

    def test_shape(self) -> None:
        result = ocr.available_engines(refresh=True)
        self.assertIsInstance(result, dict)
        for name in ("rapidocr", "pytesseract", "paddleocr"):
            self.assertIn(name, result)
            self.assertIsInstance(result[name], bool)

    def test_cache_copy_is_defensive(self) -> None:
        first = ocr.available_engines()
        first["rapidocr"] = not first["rapidocr"]
        second = ocr.available_engines()
        self.assertNotEqual(first["rapidocr"], second["rapidocr"])

    def test_lazy_init(self) -> None:
        """构造引擎不许触发 import/模型加载。"""
        engine = ocr.OcrEngine()
        self.assertFalse(engine.initialized)

    def test_recognize_missing_file_degrades(self) -> None:
        """不存在的文件 → `source="skip"`，不抛（错误原因放在 text 里）。"""
        page = ocr.OcrEngine().recognize_image(Path("__no_such_page__.png"), page=7)
        self.assertEqual(page.page, 7)
        self.assertEqual(page.source, "skip")
        self.assertEqual(page.confidence, 0.0)
        self.assertIn("OCR 失败", page.text)

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_pdf_missing_degrades(self) -> None:
        result = ocr.OcrEngine().recognize_pdf(Path("__no_such__.pdf"), ocr.OcrBudget())
        self.assertEqual(result.pages, [])
        self.assertTrue(result.degraded)
        self.assertIn("PDF 打不开", result.degraded[0])


# ---------------------------------------------------------------------------
# 版面重建
# ---------------------------------------------------------------------------

class TestLayout(unittest.TestCase):
    """版面重建：同行空格、跨行换行、大间距插空格、行内按 x 排序。"""

    def test_same_line_gets_space(self) -> None:
        """行内两个框间距 > 2.5×中位字宽 → 插空格（中位字宽按 框宽/字数 估）。"""
        boxes = [box("合同金额", 0, 100, 80, 130), box("134.16万元", 200, 100, 320, 130)]
        text = ocr.assemble_text(boxes)
        self.assertEqual(text, "合同金额 134.16万元")

    def test_tight_gap_no_space(self) -> None:
        """间距小于 2.5×中位字宽时不插空格（中文词内被切框的情况）。"""
        boxes = [box("合同", 0, 100, 40, 130), box("金额", 42, 100, 82, 130)]
        self.assertEqual(ocr.assemble_text(boxes), "合同金额")

    def test_two_lines_joined_with_newline(self) -> None:
        boxes = [box("第一行", 0, 10, 90, 40), box("第二行", 0, 60, 90, 90)]
        self.assertEqual(ocr.assemble_text(boxes), "第一行\n第二行")

    def test_x_order_inside_line(self) -> None:
        """输入乱序也要按 x 排回来。"""
        boxes = [box("B", 200, 10, 240, 40), box("A", 0, 12, 40, 42)]
        self.assertEqual(ocr.assemble_text(boxes), "A B")

    def test_lines_carry_bbox_and_conf(self) -> None:
        boxes = [box("合计", 0, 0, 60, 30, 0.9), box("100元", 160, 0, 250, 30, 0.7)]
        lines = ocr.assemble_lines(boxes)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "合计 100元")
        self.assertAlmostEqual(lines[0]["conf"], 0.8, places=4)
        self.assertEqual(lines[0]["bbox"], [0.0, 0.0, 250.0, 30.0])

    def test_empty_input(self) -> None:
        self.assertEqual(ocr.assemble_text([]), "")
        self.assertEqual(ocr.assemble_lines([box("   ", 0, 0, 10, 10)]), [])

    def test_normalize_bbox_forms(self) -> None:
        quad = [[10, 20], [110, 20], [110, 60], [10, 60]]
        self.assertEqual(ocr._normalize_bbox(quad), [10.0, 20.0, 110.0, 60.0])
        self.assertEqual(ocr._normalize_bbox([110, 60, 10, 20]), [10.0, 20.0, 110.0, 60.0])
        self.assertEqual(ocr._normalize_bbox("坏数据"), [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

class TestCER(unittest.TestCase):
    """CER = Levenshtein / len(reference)，空白先抹平。"""

    def test_identical_is_zero(self) -> None:
        self.assertEqual(ocr.char_error_rate("合同金额：100万元", "合同金额：100万元"), 0.0)

    def test_whitespace_ignored(self) -> None:
        self.assertEqual(ocr.char_error_rate("合 同\n金额", "合同金额"), 0.0)

    def test_one_substitution(self) -> None:
        self.assertAlmostEqual(ocr.char_error_rate("abcdXfghij", "abcdefghij"), 0.1)

    def test_deletion_and_insertion(self) -> None:
        self.assertAlmostEqual(ocr.char_error_rate("abcdefghi", "abcdefghij"), 0.1)
        self.assertAlmostEqual(ocr.char_error_rate("abcdefghijk", "abcdefghij"), 0.1)

    def test_empty_cases(self) -> None:
        self.assertEqual(ocr.char_error_rate("", ""), 0.0)
        self.assertEqual(ocr.char_error_rate("多出来的", ""), 1.0)

    def test_fullwidth_normalized(self) -> None:
        """全角数字先转半角再比，避免"１２３ vs 123"被当成全错。"""
        self.assertEqual(ocr.char_error_rate("金额１２３元", "金额123元"), 0.0)

    def test_evaluate_pairs(self) -> None:
        same = "本合同标准文本适用于购买现成货物的采购项目"
        mutated = same.replace("货物", "货特").replace("现成", "现減")   # 2 处替换
        pairs = [
            (same, same),                                        # 0.0，exact
            (mutated, same),                                     # 2 个替换
            ("短", "短"),                                         # 参考太短，被跳过
        ]
        report = ocr.evaluate_against_text_layer(pairs)
        length = len(same)
        self.assertEqual(ocr._levenshtein(mutated, same), 2)
        self.assertEqual(report["pages"], 2)
        self.assertEqual(report["exact"], 1)
        self.assertEqual(report["skipped"], 1)
        self.assertAlmostEqual(report["cer"], round(2 / (2 * length), 4), places=4)
        # 逐页 CER 的算术平均：一页 0.0、一页 2/len，两页取平均
        self.assertAlmostEqual(report["cer_mean"], round(1 / length, 4), places=4)

    def test_evaluate_empty(self) -> None:
        report = ocr.evaluate_against_text_layer([])
        self.assertEqual(report["pages"], 0)
        self.assertEqual(report["cer"], 0.0)


# ---------------------------------------------------------------------------
# 预算与降级
# ---------------------------------------------------------------------------

class TestBudget(unittest.TestCase):
    """超页数 / 超时 / 超预算必须停，并把原因写进 degraded。"""

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_page_budget_truncates(self) -> None:
        pages = scanned_pages(3)
        if len(pages) < 3:
            self.skipTest("真实扫描页不足 3 页")
        budget = ocr.OcrBudget(max_pages=1, max_seconds=120)
        result = ocr.OcrEngine().recognize_pages(pages, budget)
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.stats["requested"], 1)
        self.assertTrue(any("页数预算截断" in note for note in result.degraded))

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_time_budget_stops(self) -> None:
        pages = scanned_pages(2)
        if len(pages) < 2:
            self.skipTest("真实扫描页不足 2 页")
        budget = ocr.OcrBudget(max_pages=10, max_seconds=0.001)
        result = ocr.OcrEngine().recognize_pages(pages, budget)
        self.assertEqual(result.pages, [])
        self.assertTrue(any("时间预算截断" in note for note in result.degraded))

    def test_result_helpers(self) -> None:
        result = ocr.OcrResult(
            pages=[ocr.OcrPage(page=1, text="合同", confidence=0.9, source="rapidocr"),
                   ocr.OcrPage(page=2, text="", confidence=0.0, source="skip")],
            engine="rapidocr",
            stats={"pages": 1},
            degraded=["x"],
        )
        self.assertEqual(result.by_page()[1].text, "合同")
        self.assertEqual(result.full_text(), "合同")
        self.assertAlmostEqual(result.mean_confidence(), 0.9, places=4)
        self.assertEqual(result.to_dict()["engine"], "rapidocr")


# ---------------------------------------------------------------------------
# 合并进 ParseOutput
# ---------------------------------------------------------------------------

class TestMerge(unittest.TestCase):
    """`merge_ocr_into_parse`：补块、重排序、写 stats["ocr"]、不改入参。"""

    def make_parse(self) -> ParseOutput:
        blocks = [SourceBlock(block_id="p1b0", order=0, kind="line", text="页眉文字",
                              page=1, bbox=[0, 0, 100, 20])]
        return ParseOutput(blocks=blocks, file_type="pdf",
                           page_sizes={1: [595, 842], 2: [595, 842]},
                           needs_ocr_pages=[2], image_blocks=[], stats={"pages": 2})

    def make_result(self) -> ocr.OcrResult:
        page = ocr.OcrPage(
            page=2, text="甲方：北京警察学院\n乙方：北京金源盛海", confidence=0.93,
            boxes=[box("甲方：北京警察学院", 10, 20, 300, 50)],
            engine="rapidocr", seconds=1.5, source="rapidocr",
            lines=[{"text": "甲方：北京警察学院", "bbox": [10, 20, 300, 50], "conf": 0.93},
                   {"text": "乙方：北京金源盛海", "bbox": [10, 60, 300, 90], "conf": 0.93}],
        )
        return ocr.OcrResult(pages=[page], engine="rapidocr",
                             stats={"pages": 1, "avg_conf": 0.93}, degraded=["示例降级"])

    def test_blocks_added_and_sorted(self) -> None:
        parse = self.make_parse()
        merged = ocr.merge_ocr_into_parse(parse, self.make_result())
        self.assertEqual(len(parse.blocks), 1, "入参不允许被改动")
        self.assertGreater(len(merged.blocks), 1)
        self.assertEqual(merged.needs_ocr_pages, [])
        pages = [b.page for b in merged.blocks]
        self.assertEqual(pages, sorted(pages))
        self.assertEqual([b.order for b in merged.blocks], list(range(len(merged.blocks))))
        ocr_blocks = [b for b in merged.blocks if b.metadata.get("ocr")]
        self.assertTrue(ocr_blocks)
        self.assertTrue(all(b.page == 2 for b in ocr_blocks))
        self.assertEqual(ocr_blocks[0].kind, "line")

    def test_stats_recorded(self) -> None:
        merged = ocr.merge_ocr_into_parse(self.make_parse(), self.make_result())
        info = merged.stats["ocr"]
        self.assertEqual(info["pages"], 1)
        self.assertEqual(info["engine"], "rapidocr")
        self.assertAlmostEqual(info["avg_conf"], 0.93, places=4)
        self.assertIn("示例降级", info["degraded"])
        self.assertTrue(any("OCR 降级" in w for w in merged.warnings))

    def test_empty_result_keeps_parse(self) -> None:
        parse = self.make_parse()
        merged = ocr.merge_ocr_into_parse(
            parse, ocr.OcrResult(pages=[], engine="none", degraded=["无引擎"]))
        self.assertEqual(len(merged.blocks), 1)
        self.assertEqual(merged.stats["ocr"]["pages"], 0)
        self.assertEqual(merged.needs_ocr_pages, [2], "没识别出来的页要留队")


# ---------------------------------------------------------------------------
# 真引擎（真实扫描页）
# ---------------------------------------------------------------------------

class TestRealEngine(unittest.TestCase):
    """跑真实扫描页：断言"有字、有置信度、有耗时"，不断言具体字串（模型会升级）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pages = scanned_pages(2)

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_scanned_page_produces_text(self) -> None:
        if not self.pages:
            self.skipTest("data/gov_contracts 没有扫描页")
        engine = ocr.OcrEngine(min_confidence=0.5)
        page = engine.recognize_image(self.pages[0], page=2)
        self.assertEqual(page.source, "rapidocr")
        self.assertGreater(len(page.text), 20)
        self.assertGreater(page.confidence, 0.5)
        self.assertGreater(page.seconds, 0.0)
        self.assertTrue(page.boxes)
        self.assertTrue(page.lines)
        self.assertIn("text", page.boxes[0])
        self.assertIn("conf", page.boxes[0])
        self.assertIn("bbox", page.boxes[0])

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_min_confidence_filters(self) -> None:
        """把阈值抬到 1.01 → 一个框都不留，文本为空但页仍是 rapidocr。"""
        if not self.pages:
            self.skipTest("data/gov_contracts 没有扫描页")
        engine = ocr.OcrEngine(min_confidence=1.01)
        page = engine.recognize_image(self.pages[0], page=2)
        self.assertEqual(page.text, "")
        self.assertEqual(page.boxes, [])


class TestRealCER(unittest.TestCase):
    """真实配对页（有文字层）上的 CER —— 论文里那张表的原料。"""

    @unittest.skipUnless(HAS_ENGINE, "本机无可用 OCR 引擎")
    def test_pair_cer_reasonable(self) -> None:
        pairs = text_layer_pages(min_chars=100, limit=1)
        if not pairs:
            self.skipTest("没有 text_len>100 的配对页")
        image, layer = pairs[0]
        engine = ocr.OcrEngine(min_confidence=0.5)
        page = engine.recognize_image(image)
        cer = ocr.char_error_rate(page.text, layer)
        report = ocr.evaluate_against_text_layer([(page.text, layer)])
        self.assertEqual(report["pages"], 1)
        self.assertAlmostEqual(report["cer"], round(cer, 4), places=3)
        self.assertGreaterEqual(cer, 0.0)
        self.assertLess(cer, 1.0, "配对页 CER 不该接近全错")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
