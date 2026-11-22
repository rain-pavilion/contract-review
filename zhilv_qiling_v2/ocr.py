"""OCR 通道：把扫描页变成**可审查文本**，并给出可量化的识别质量。

相对 v1 的升级点
----------------
1. **v1 完全没有 OCR 通道**：`parsers.py` 遇到扫描件只能返回空字符串，报告里看不出
   "这页其实一个字都没读到"，合同金额、签订日期这些关键要素直接凭空消失。v2 在这里
   补一条独立通道：扫描页 → 行为单位重建版面 → 回填 `ParseOutput`。
2. **版面重建而不是行拼接**：rapidocr 返回的是一堆文本框（左上/右下四点），v1 的做法
   若直接按返回顺序拼，双栏合同会左右串行、表格会横竖交错。这里按 `(y_center 分行,
   x 排序)` 重建：同一行的框用空格连（间距 > 2.5 倍中位字宽时补一个空格，避免
   "合同金额134.16" 糊在一起），跨行用 `\\n`。
3. **预算化执行**：扫描件动辄上百页，CPU 上单页 3–6 秒，必须能被页数/时长/置信度三个
   预算截断，且截断原因写进 `degraded`（v1 是死跑到底，演示现场必然超时）。
4. **识别质量可度量**：实现 `char_error_rate`（自写 Levenshtein DP，不依赖
   python-Levenshtein）与 `evaluate_against_text_layer`。这批北京政采合同里有
   ~112 页同时带文字层，天然构成 OCR 对照集，论文里的 CER 数字就出在这里，而不是
   "目测还行"。
5. **诚实的降级**：rapidocr / pytesseract / paddleocr 全都没有时，`recognize_*` 不抛
   异常，返回 `source="skip"` 的页并把原因记进 `degraded`；报告口径与
   `config.DEGRADE_NOTES["ocr"]` 一致。

工程约束
--------
* 惰性初始化：模块导入时不 import rapidocr（它连带 onnxruntime，冷启动 ~1s），
  第一次真正识别才加载；加载失败一律降级，不影响主链路。
* 只用标准库 + 可选 pymupdf / rapidocr；没有网络依赖，离线可跑。
"""

from __future__ import annotations

import io
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import text as T
from .config import capabilities
from .models import SourceBlock, stable_id
from .parsers import SCANNED_PAGE_TEXT_THRESHOLD, ParseOutput

__all__ = [
    "OcrBudget",
    "OcrPage",
    "OcrResult",
    "OcrEngine",
    "available_engines",
    "assemble_lines",
    "assemble_text",
    "char_error_rate",
    "evaluate_against_text_layer",
    "get_engine",
    "merge_ocr_into_parse",
]

#: 行聚类容差：以"中位框高"为尺度，0.55 倍以内视为同一行。
LINE_TOLERANCE = 0.55
#: 行内插入空格的阈值：x 间距 > 2.5 × 中位字宽。
WORD_GAP_FACTOR = 2.5
#: 单页识别里低于该字符数的页不参与 CER（避免拿 3 个字的页签拉偏指标）。
MIN_EVAL_CHARS = 20


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class OcrBudget:
    """一次 OCR 任务的预算。三个维度任一用尽即停，并把原因写进 `degraded`。"""

    max_pages: int = 200
    max_seconds: float = 120.0
    dpi: int = 200
    min_confidence: float = 0.5


@dataclass
class OcrPage:
    """单页识别结果。

    `source` 取值：
    * `"rapidocr"` —— 真的跑了 OCR；
    * `"pymupdf-text"` —— 该页本来有文字层，直接取用（省时且零误差）；
    * `"skip"` —— 没有可用引擎 / 超预算，`text` 为空，原因在 `OcrResult.degraded`。
    """

    page: int
    text: str = ""
    confidence: float = 0.0
    boxes: list[dict] = field(default_factory=list)
    engine: str = "none"
    seconds: float = 0.0
    source: str = "skip"
    lines: list[dict] = field(default_factory=list)
    """重建后的行：[{"text", "bbox", "conf"}]。回填 blocks 时用它的 bbox 做定位。"""

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class OcrResult:
    """一次 OCR 任务的产物。`stats` 里的数字直接进报告与论文表格。"""

    pages: list[OcrPage] = field(default_factory=list)
    engine: str = "none"
    stats: dict[str, Any] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def by_page(self) -> dict[int, OcrPage]:
        return {p.page: p for p in self.pages}

    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)

    def mean_confidence(self) -> float:
        scores = [p.confidence for p in self.pages if p.source != "skip" and p.confidence > 0]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "stats": self.stats,
            "degraded": list(self.degraded),
            "pages": [
                {
                    "page": p.page,
                    "chars": p.length,
                    "confidence": p.confidence,
                    "engine": p.engine,
                    "source": p.source,
                    "seconds": p.seconds,
                }
                for p in self.pages
            ],
        }


# ---------------------------------------------------------------------------
# 引擎探测
# ---------------------------------------------------------------------------

def _probe_rapidocr() -> bool:
    """能 import 且能实例化才算可用（缺模型文件时实例化会抛）。"""
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return False
    try:
        RapidOCR()
        return True
    except Exception:
        return False


def _probe_paddleocr() -> bool:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return False
    try:
        PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
        return True
    except Exception:
        return False


def _probe_tesseract() -> bool:
    """pytesseract 是壳，真正要有 tesseract 可执行文件。"""
    try:
        import pytesseract  # type: ignore  # noqa: F401
    except Exception:
        return False
    try:
        import shutil

        if not shutil.which("tesseract"):
            return False
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


_ENGINE_CACHE: dict[str, bool] | None = None


def available_engines(refresh: bool = False) -> dict[str, bool]:
    """探测三个 OCR 后端：`{"rapidocr": bool, "pytesseract": bool, "paddleocr": bool}`。

    探测含"实例化"一步，因为包能 import 不代表模型文件在位。任何异常都被吞掉 ——
    这个函数不允许把启动失败变成主链路的崩溃点。
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None and not refresh:
        return dict(_ENGINE_CACHE)
    result = {
        "rapidocr": _probe_rapidocr(),
        "pytesseract": _probe_tesseract(),
        "paddleocr": _probe_paddleocr(),
    }
    _ENGINE_CACHE = result
    return dict(result)


def _preferred_engine() -> str:
    probes = available_engines()
    for name in ("rapidocr", "paddleocr", "pytesseract"):
        if probes.get(name):
            return name
    return "none"


# ---------------------------------------------------------------------------
# 版面重建：文本框 → 文本
# ---------------------------------------------------------------------------

def _normalize_bbox(raw: Any) -> list[float]:
    """把 rapidocr 的四种点/八数/矩形统一成 `[x0, y0, x1, y1]`。"""
    try:
        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            xs = [float(p[0]) for p in raw]
            ys = [float(p[1]) for p in raw]
            return [min(xs), min(ys), max(xs), max(ys)]
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            vals = [float(v) for v in raw[:4]]
            return [min(vals[0], vals[2]), min(vals[1], vals[3]),
                    max(vals[0], vals[2]), max(vals[1], vals[3])]
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 0.0]


def _group_lines(boxes: Sequence[dict], *, tolerance: float = LINE_TOLERANCE) -> list[list[dict]]:
    """按 y_center 聚类成行。容差是中位框高的 `tolerance` 倍。"""
    items = [b for b in boxes if str(b.get("text", "")).strip()]
    if not items:
        return []
    heights = [max(1.0, b["bbox"][3] - b["bbox"][1]) for b in items]
    med_h = statistics.median(heights) if heights else 10.0
    tol = max(1.0, med_h * tolerance)

    ordered = sorted(items, key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2.0, b["bbox"][0]))
    lines: list[list[dict]] = []
    for box in ordered:
        center = (box["bbox"][1] + box["bbox"][3]) / 2.0
        if lines:
            last = lines[-1]
            last_center = statistics.median([(b["bbox"][1] + b["bbox"][3]) / 2.0 for b in last])
            if abs(center - last_center) <= tol:
                last.append(box)
                continue
        lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b["bbox"][0])
    return lines


def assemble_lines(boxes: Sequence[dict], *, gap_factor: float = WORD_GAP_FACTOR) -> list[dict]:
    """文本框 → 行列表 `[{"text", "bbox", "conf"}]`。

    中位字宽用"框宽 / 框内字符数"估计，比按字号估更稳（OCR 框宽≈文字实际占宽）。
    行内两框水平间距 > `gap_factor × 中位字宽` 时插入一个空格：
    rapidocr 会把"合同金额"和"134.16万元"切成两个框，不加空格会糊成一个词。
    """
    lines = _group_lines(boxes)
    if not lines:
        return []
    widths: list[float] = []
    for box in boxes:
        chars = len(str(box.get("text", "")).strip())
        if chars <= 0:
            continue
        width = box["bbox"][2] - box["bbox"][0]
        if width > 0:
            widths.append(width / chars)
    med_char = statistics.median(widths) if widths else 8.0
    gap_threshold = max(1.0, med_char * gap_factor)

    out: list[dict] = []
    for line in lines:
        parts: list[str] = []
        prev_right: float | None = None
        for box in line:
            piece = str(box.get("text", "")).strip()
            if not piece:
                continue
            if prev_right is not None and (box["bbox"][0] - prev_right) > gap_threshold:
                parts.append(" ")
            parts.append(piece)
            prev_right = box["bbox"][2]
        joined = "".join(parts).strip()
        if not joined:
            continue
        confs = [b["conf"] for b in line if b.get("conf") is not None]
        out.append({
            "text": joined,
            "bbox": [
                min(b["bbox"][0] for b in line),
                min(b["bbox"][1] for b in line),
                max(b["bbox"][2] for b in line),
                max(b["bbox"][3] for b in line),
            ],
            "conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
        })
    return out


def assemble_text(boxes: Sequence[dict], *, gap_factor: float = WORD_GAP_FACTOR) -> str:
    """文本框 → 文本：同行用空格、跨行用 `\\n`（`assemble_lines` 的字符串版）。"""
    return "\n".join(line["text"] for line in assemble_lines(boxes, gap_factor=gap_factor))


# ---------------------------------------------------------------------------
# CER（字符错误率）
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_for_cer(text: str) -> str:
    """CER 前归一：去掉所有空白。

    原因：OCR 版面重建会引入换行与空格，而文字层是连续文本，不抹平的话行切分差异
    会被算进字符错误 —— 那不是识别质量。
    """
    return _WS_RE.sub("", T.full_to_half(text or ""))


def _levenshtein(a: str, b: str) -> int:
    """两行滚动 DP 的编辑距离（插入/删除/替换各计 1）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,        # 删除
                current[j - 1] + 1,     # 插入
                previous[j - 1] + cost,  # 替换
            ))
        previous = current
    return previous[-1]


def char_error_rate(hypothesis: str, reference: str) -> float:
    """`CER = Levenshtein(hyp, ref) / len(ref)`，空白先抹平。

    约定：`reference` 为空时，`hypothesis` 也为空返回 0.0，否则返回 1.0
    （分母为 0 不算 inf，避免聚合时被一个空对照页毁掉整表均值）。
    """
    ref = _normalize_for_cer(reference)
    hyp = _normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(hyp, ref) / len(ref)


def evaluate_against_text_layer(pairs: Iterable[tuple[str, str]]) -> dict:
    """对若干 `(ocr_text, text_layer_text)` 算 CER 汇总。

    返回 `{"cer", "pages", "exact", "cer_mean", "distance", "ref_chars", "pairs"}`：
    * `cer`：按参考长度加权的整体 CER（= 总编辑距离 / 总参考字符，池化口径，适合报数）；
    * `cer_mean`：逐页 CER 的算术平均（论文里的 "mean CER"）；
    * `exact`：完全一致的页数；
    * `pages`：参与统计的页数（参考侧有效字符 < `MIN_EVAL_CHARS` 的页被跳过）。
    """
    total_distance = 0
    total_ref = 0
    exact = 0
    pages = 0
    per_page: list[float] = []
    skipped = 0
    for ocr_text, layer_text in pairs:
        ref = _normalize_for_cer(layer_text)
        hyp = _normalize_for_cer(ocr_text)
        if len(ref) < MIN_EVAL_CHARS:
            skipped += 1
            continue
        distance = _levenshtein(hyp, ref)
        pages += 1
        total_distance += distance
        total_ref += len(ref)
        per_page.append(distance / len(ref))
        if distance == 0:
            exact += 1
    pooled = (total_distance / total_ref) if total_ref else 0.0
    return {
        "cer": round(pooled, 4),
        "pages": pages,
        "exact": exact,
        "cer_mean": round(sum(per_page) / len(per_page), 4) if per_page else 0.0,
        "cer_max": round(max(per_page), 4) if per_page else 0.0,
        "distance": total_distance,
        "ref_chars": total_ref,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

class OcrEngine:
    """OCR 引擎封装：惰性加载 + 预算控制 + 版面重建 + 降级记账。"""

    def __init__(self, lang: str = "ch", min_confidence: float = 0.5) -> None:
        self.lang = lang
        self.min_confidence = float(min_confidence)
        self._impl: Any = None
        self._engine_name: str = ""
        self._init_error: str = ""
        self._init_seconds: float = 0.0

    # -- 生命周期 ---------------------------------------------------------
    @property
    def engine_name(self) -> str:
        """未初始化时为空串；`ensure()` 之后是 rapidocr/paddleocr/tesseract/none。"""
        return self._engine_name

    @property
    def initialized(self) -> bool:
        return bool(self._engine_name)

    def ensure(self) -> bool:
        """真正加载引擎。第一次调用才 import；任何失败都降级为 `none`。"""
        if self._engine_name:
            return self._engine_name != "none"
        started = time.time()
        probes = available_engines()
        name = "none"
        impl: Any = None
        error = ""
        if probes.get("rapidocr"):
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                impl = RapidOCR()
                name = "rapidocr"
            except Exception as exc:  # pragma: no cover - 环境相关
                error = f"rapidocr 实例化失败：{type(exc).__name__}: {exc}"
        if impl is None and probes.get("paddleocr"):
            try:
                from paddleocr import PaddleOCR  # type: ignore

                impl = PaddleOCR(use_angle_cls=False, lang=self.lang, show_log=False)
                name = "paddleocr"
            except Exception as exc:  # pragma: no cover
                error = f"paddleocr 实例化失败：{type(exc).__name__}: {exc}"
        if impl is None and probes.get("pytesseract"):
            try:
                import pytesseract  # type: ignore

                impl = pytesseract
                name = "tesseract"
            except Exception as exc:  # pragma: no cover
                error = f"pytesseract 不可用：{type(exc).__name__}: {exc}"
        self._impl = impl
        self._engine_name = name
        self._init_error = error
        self._init_seconds = round(time.time() - started, 3)
        return name != "none"

    # -- 单页 -------------------------------------------------------------
    def _raw_boxes(self, image: Any) -> list[dict]:
        """调用后端，统一成 `[{text, conf, bbox}]`。"""
        boxes: list[dict] = []
        if self._engine_name == "rapidocr":
            result, _elapse = self._impl(image)
            for item in result or []:
                try:
                    bbox, text, score = item[0], item[1], item[2]
                except Exception:
                    continue
                piece = str(text or "").strip()
                if not piece:
                    continue
                boxes.append({
                    "text": piece,
                    "conf": round(float(score or 0.0), 4),
                    "bbox": _normalize_bbox(bbox),
                })
        elif self._engine_name == "paddleocr":
            result = self._impl.ocr(image, cls=False)
            for page in result or []:
                for item in page or []:
                    try:
                        bbox, (text, score) = item[0], item[1]
                    except Exception:
                        continue
                    piece = str(text or "").strip()
                    if not piece:
                        continue
                    boxes.append({
                        "text": piece,
                        "conf": round(float(score or 0.0), 4),
                        "bbox": _normalize_bbox(bbox),
                    })
        elif self._engine_name == "tesseract":
            import pytesseract  # type: ignore

            data = pytesseract.image_to_data(
                _to_pil(image), lang=self.lang or "chi_sim",
                output_type=pytesseract.Output.DICT,
            )
            n = len(data.get("text", []))
            for i in range(n):
                piece = str(data["text"][i] or "").strip()
                if not piece:
                    continue
                try:
                    conf = float(data["conf"][i]) / 100.0
                except Exception:
                    conf = 0.0
                x, y = float(data["left"][i]), float(data["top"][i])
                w, h = float(data["width"][i]), float(data["height"][i])
                boxes.append({"text": piece, "conf": round(conf, 4),
                              "bbox": [x, y, x + w, y + h]})
        boxes.sort(key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2.0, b["bbox"][0]))
        return boxes

    def recognize_image(self, bytes_or_path: Any, *, page: int | None = None) -> OcrPage:
        """识别单张图片。`bytes_or_path` 接受 bytes / str / Path。

        引擎不可用时返回 `source="skip"` 的空页，不抛异常（调用方据 `source` 判断）。
        """
        return self._recognize(bytes_or_path, page=page, min_conf=self.min_confidence)

    def _recognize(self, image: Any, *, page: int | None, min_conf: float) -> OcrPage:
        page_no = int(page) if page is not None else 1
        started = time.time()
        # A missing optional engine must not mask an invalid input file.
        if isinstance(image, (str, Path)) and not Path(image).is_file():
            return OcrPage(page=page_no, engine="none", source="skip", seconds=0.0,
                           boxes=[], text="[OCR 失败] 文件不存在或不是普通文件", confidence=0.0)
        if not self.ensure():
            return OcrPage(page=page_no, engine="none", source="skip", seconds=0.0,
                           boxes=[], text="", confidence=0.0)
        try:
            boxes = self._raw_boxes(image)
        except Exception as exc:
            return OcrPage(page=page_no, engine=self._engine_name, source="skip",
                           seconds=round(time.time() - started, 3),
                           text=f"[OCR 失败] {type(exc).__name__}: {exc}", confidence=0.0)
        kept = [b for b in boxes if b["conf"] >= min_conf]
        lines = assemble_lines(kept)
        text = "\n".join(line["text"] for line in lines)
        confidence = round(sum(b["conf"] for b in kept) / len(kept), 4) if kept else 0.0
        return OcrPage(
            page=page_no,
            text=text,
            confidence=confidence,
            boxes=kept,
            engine=self._engine_name,
            seconds=round(time.time() - started, 3),
            source="rapidocr" if self._engine_name in ("rapidocr", "paddleocr", "tesseract")
            else "skip",
            lines=lines,
        )

    # -- 多页 -------------------------------------------------------------
    @staticmethod
    def _page_no_from_path(path: Path, fallback: int) -> int:
        match = re.search(r"(\d{1,4})(?=\.[A-Za-z0-9]+$)", path.name)
        if match:
            try:
                value = int(match.group(1))
                if value > 0:
                    return value
            except ValueError:
                pass
        return fallback

    def recognize_pages(self, paths: Sequence[Path], budget: OcrBudget | None = None,
                        *, min_confidence: float | None = None) -> OcrResult:
        """批量识别图片页。超页数 / 超时 / 超置信度条件时停下并记 `degraded`。"""
        budget = budget or OcrBudget()
        limit_conf = self.min_confidence if min_confidence is None else float(min_confidence)
        started = time.time()
        pages: list[OcrPage] = []
        degraded: list[str] = []
        if not self.ensure():
            degraded.append(
                self._init_error
                or "未探测到可用 OCR 引擎，扫描页保持 ocr_pending，未臆造文本"
            )
            return OcrResult(pages=[], engine="none",
                             stats={"pages": 0, "requested": len(list(paths)),
                                    "seconds": 0.0, "avg_conf": 0.0},
                             degraded=degraded)

        path_list = list(paths)
        if budget.max_pages > 0 and len(path_list) > budget.max_pages:
            degraded.append(
                f"页数预算截断：请求 {len(path_list)} 页，max_pages={budget.max_pages}，"
                f"只处理前 {budget.max_pages} 页"
            )
            path_list = path_list[: budget.max_pages]

        low_conf_pages = 0
        for index, path in enumerate(path_list, start=1):
            if budget.max_seconds > 0 and (time.time() - started) > budget.max_seconds:
                degraded.append(
                    f"时间预算截断：已用 {time.time() - started:.1f}s > "
                    f"max_seconds={budget.max_seconds}，剩余 {len(path_list) - index + 1} 页未识别"
                )
                break
            page_no = self._page_no_from_path(Path(path), index)
            page = self._recognize(str(path), page=page_no, min_conf=limit_conf)
            if page.source == "skip":
                degraded.append(f"第 {page_no} 页识别失败：{page.text[:60]}")
            elif page.confidence < limit_conf:
                low_conf_pages += 1
            pages.append(page)

        elapsed = round(time.time() - started, 3)
        done = [p for p in pages if p.source != "skip"]
        stats = {
            "pages": len(done),
            "requested": len(path_list),
            "skipped": len(pages) - len(done),
            "low_confidence": low_conf_pages,
            "chars": sum(p.length for p in done),
            "seconds": elapsed,
            "sec_per_page": round(elapsed / len(done), 2) if done else 0.0,
            "avg_conf": round(sum(p.confidence for p in done) / len(done), 4) if done else 0.0,
            "engine_init_seconds": self._init_seconds,
            "dpi": budget.dpi,
        }
        return OcrResult(pages=pages, engine=self._engine_name, stats=stats, degraded=degraded)

    def recognize_pdf(self, path: Path | str, budget: OcrBudget | None = None,
                      *, pages: Sequence[int] | None = None,
                      prefer_text_layer: bool = True) -> OcrResult:
        """渲染 PDF 指定页到 `budget.dpi` 后识别。

        `pages` 为空 = 全篇。`prefer_text_layer=True` 时，本来有文字层（>= 阈值）的页
        直接取文字层（`source="pymupdf-text"`），既不浪费 CPU 也没有识别误差。
        """
        budget = budget or OcrBudget()
        target = Path(path)
        try:
            import pymupdf  # type: ignore
        except Exception:
            return OcrResult(pages=[], engine=self._engine_name,
                             stats={"pages": 0, "error": "pymupdf 不可用"},
                             degraded=["pymupdf 不可用，无法渲染 PDF 页；请直接给页面图片"])

        started = time.time()
        out: list[OcrPage] = []
        degraded: list[str] = []
        try:
            doc = pymupdf.open(str(target))
        except Exception as exc:
            return OcrResult(pages=[], engine=self._engine_name,
                             stats={"pages": 0},
                             degraded=[f"PDF 打不开：{type(exc).__name__}: {exc}"])

        total = doc.page_count
        wanted = list(pages) if pages else list(range(1, total + 1))
        if budget.max_pages > 0 and len(wanted) > budget.max_pages:
            degraded.append(
                f"页数预算截断：请求 {len(wanted)} 页，max_pages={budget.max_pages}"
            )
            wanted = wanted[: budget.max_pages]

        with doc:
            for page_no in wanted:
                if not (1 <= page_no <= total):
                    continue
                if budget.max_seconds > 0 and (time.time() - started) > budget.max_seconds:
                    degraded.append(
                        f"时间预算截断：已用 {time.time() - started:.1f}s > "
                        f"max_seconds={budget.max_seconds}，第 {page_no} 页起未识别"
                    )
                    break
                try:
                    pdf_page = doc[page_no - 1]
                except Exception:
                    degraded.append(f"第 {page_no} 页取不到")
                    continue
                layer = ""
                try:
                    layer = pdf_page.get_text().strip()
                except Exception:
                    layer = ""
                if prefer_text_layer and len(layer) >= SCANNED_PAGE_TEXT_THRESHOLD:
                    out.append(OcrPage(page=page_no, text=layer, confidence=1.0,
                                       boxes=[], engine="pymupdf-text",
                                       seconds=round(time.time() - started, 3),
                                       source="pymupdf-text"))
                    continue
                try:
                    pix = pdf_page.get_pixmap(dpi=int(budget.dpi))
                    png = pix.tobytes("png")
                except Exception as exc:
                    degraded.append(f"第 {page_no} 页渲染失败：{type(exc).__name__}: {exc}")
                    continue
                out.append(self._recognize(png, page=page_no,
                                           min_conf=budget.min_confidence))

        elapsed = round(time.time() - started, 3)
        done = [p for p in out if p.source != "skip"]
        stats = {
            "pages": len(done),
            "requested": len(wanted),
            "total_pages": total,
            "text_layer_pages": sum(1 for p in done if p.source == "pymupdf-text"),
            "ocr_pages": sum(1 for p in done if p.source == "rapidocr"),
            "chars": sum(p.length for p in done),
            "seconds": elapsed,
            "sec_per_page": round(elapsed / len(done), 2) if done else 0.0,
            "avg_conf": round(
                sum(p.confidence for p in done if p.source == "rapidocr")
                / max(1, sum(1 for p in done if p.source == "rapidocr")), 4
            ) if any(p.source == "rapidocr" for p in done) else 0.0,
            "dpi": budget.dpi,
        }
        return OcrResult(pages=out, engine=self._engine_name, stats=stats, degraded=degraded)

    def recognize_parse_output(self, parse_output: ParseOutput, source_name: str,
                              budget: OcrBudget | None = None) -> OcrResult:
        """处理 `parse_output.needs_ocr_pages`：把待识别页转成 `OcrResult`。

        `source_name` 可以是 PDF 路径，也可以是单张图片路径；若是纯图片文档，则走
        `parse_output.image_blocks` 里登记的路径。
        """
        budget = budget or OcrBudget()
        targets = list(getattr(parse_output, "needs_ocr_pages", []) or [])
        source_path = Path(source_name) if source_name else None

        if source_path is not None and source_path.exists():
            suffix = source_path.suffix.lower()
            if suffix == ".pdf":
                return self.recognize_pdf(source_path, budget, pages=targets or None,
                                          prefer_text_layer=False)
            if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
                page = targets[0] if targets else 1
                one = self._recognize(str(source_path), page=page,
                                      min_conf=budget.min_confidence)
                return OcrResult(pages=[one], engine=self._engine_name,
                                 stats={"pages": 1 if one.source != "skip" else 0,
                                        "seconds": one.seconds,
                                        "avg_conf": one.confidence},
                                 degraded=[] if one.source != "skip"
                                 else [f"图片识别失败：{one.text[:80]}"])

        image_paths = [
            Path(b.metadata.get("path", ""))
            for b in getattr(parse_output, "image_blocks", []) or []
            if b.metadata.get("path")
        ]
        image_paths = [p for p in image_paths if p.exists()]
        if image_paths:
            return self.recognize_pages(image_paths, budget)

        return OcrResult(pages=[], engine=self._engine_name, stats={"pages": 0, "requested": 0},
                         degraded=[f"没有可识别的页：needs_ocr_pages={targets}，"
                                   f"source={source_name!r} 也不存在"])


# ---------------------------------------------------------------------------
# 合并回 ParseOutput
# ---------------------------------------------------------------------------

def _page_to_blocks(page: OcrPage, start_order: int) -> list[SourceBlock]:
    """一页 OCR 文本 → 以**行为单位**的 `SourceBlock`。

    用 `kind="line"` 与 PDF 文字层保持一致，`layout.py` 的段落合并规则才能直接复用；
    每行保留 bbox 与置信度，报告里的定位能落到"第几页第几行"。
    """
    blocks: list[SourceBlock] = []
    if page.text:
        source_lines = page.lines or [
            {"text": ln, "bbox": None, "conf": page.confidence}
            for ln in page.text.split("\n") if ln.strip()
        ]
        for offset, line in enumerate(source_lines):
            cleaned = T.clean_text(str(line.get("text", "")), keep_newlines=False)
            if not cleaned:
                continue
            raw = line.get("bbox")
            bbox = tuple(float(v) for v in raw) if raw and len(raw) == 4 else None
            blocks.append(
                SourceBlock(
                    block_id=stable_id("sb", "ocr", page.page, start_order + len(blocks), cleaned[:32]),
                    order=start_order + len(blocks),
                    kind="line",
                    text=cleaned,
                    page=page.page,
                    bbox=bbox,  # type: ignore[arg-type]
                    metadata={
                        "ocr": True,
                        "engine": page.engine,
                        "confidence": float(line.get("conf") or page.confidence),
                    },
                )
            )
    if not blocks and page.text:
        blocks.append(
            SourceBlock(
                block_id=stable_id("sb", "ocr", page.page, "all"),
                order=start_order,
                kind="line",
                text=T.clean_text(page.text, keep_newlines=False),
                page=page.page,
                metadata={"ocr": True, "engine": page.engine, "confidence": page.confidence},
            )
        )
    return blocks


def merge_ocr_into_parse(parse_output: ParseOutput, ocr_result: OcrResult) -> ParseOutput:
    """把 OCR 文本按页补进 blocks，返回**新的** `ParseOutput`（不改入参）。

    * 扫描页原本没有 block，所以是"补"；若某页已有文字层 block（页眉可读），OCR 行
      追加在其后，按 `(page, order)` 重排并重编号。
    * `stats["ocr"]` 记 `pages / engine / avg_conf / degraded`，报告与论文表格直接引用。
    """
    ocr_pages = {p.page: p for p in ocr_result.pages if p.text.strip()}
    if not ocr_pages:
        new_stats = dict(parse_output.stats or {})
        new_stats["ocr"] = {
            "pages": 0,
            "engine": ocr_result.engine,
            "avg_conf": 0.0,
            "degraded": list(ocr_result.degraded),
        }
        return ParseOutput(
            blocks=list(parse_output.blocks),
            file_type=parse_output.file_type,
            page_sizes=dict(parse_output.page_sizes),
            needs_ocr_pages=list(parse_output.needs_ocr_pages),
            image_blocks=list(parse_output.image_blocks),
            stats=new_stats,
            warnings=list(parse_output.warnings),
        )

    blocks: list[SourceBlock] = list(parse_output.blocks)
    next_order = max((b.order for b in blocks), default=-1) + 1
    for page_no in sorted(ocr_pages):
        page = ocr_pages[page_no]
        blocks.extend(_page_to_blocks(page, next_order + len(blocks)))

    blocks.sort(key=lambda b: ((b.page if b.page is not None else 0), b.order))
    for index, block in enumerate(blocks):
        block.order = index

    done = [p for p in ocr_pages.values()]
    avg_conf = round(sum(p.confidence for p in done) / len(done), 4) if done else 0.0
    stats = dict(parse_output.stats or {})
    stats["ocr"] = {
        "pages": len(done),
        "engine": ocr_result.engine,
        "avg_conf": avg_conf,
        "chars": sum(p.length for p in done),
        "degraded": list(ocr_result.degraded),
        "pages_detail": ocr_result.stats,
    }
    stats["ocr_blocks"] = sum(1 for b in blocks if b.metadata.get("ocr"))
    warnings = list(parse_output.warnings)
    for note in ocr_result.degraded:
        if note not in warnings:
            warnings.append(f"OCR 降级：{note}")
    still_missing = [p for p in parse_output.needs_ocr_pages if p not in ocr_pages]
    return ParseOutput(
        blocks=blocks,
        file_type=parse_output.file_type,
        page_sizes=dict(parse_output.page_sizes),
        needs_ocr_pages=still_missing,
        image_blocks=list(parse_output.image_blocks),
        stats=stats,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 小工具与单例
# ---------------------------------------------------------------------------

def _to_pil(image: Any) -> Any:
    """tesseract 分支用：bytes/Path → PIL.Image。"""
    from PIL import Image  # type: ignore

    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image)))
    if hasattr(image, "read"):
        return Image.open(image)
    return Image.open(str(image))


_ENGINE: OcrEngine | None = None


def get_engine(min_confidence: float = 0.5) -> OcrEngine:
    """模块级单例：同一进程共用一份引擎（避免每页重复加载 onnx 模型）。

    传入的 `min_confidence` 在首次创建时生效，后续调用沿用已有实例（但会更新阈值）。
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = OcrEngine(min_confidence=min_confidence)
    else:
        _ENGINE.min_confidence = float(min_confidence)
    return _ENGINE


def capability_summary() -> dict[str, Any]:
    """给 `cli doctor` / 报告用的一句话能力快照。"""
    caps = capabilities()
    probes = available_engines()
    return {
        "capabilities.ocr_rapid": caps.ocr_rapid,
        "engines": probes,
        "preferred": _preferred_engine(),
    }
