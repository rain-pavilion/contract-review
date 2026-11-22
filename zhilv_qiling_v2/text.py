"""中文文本规范化、数字/金额/日期解析、相似度。

v1 把这些散在 `extractor.py`（`_money_to_number`/`_date_normalized`）、`splitter.py`
（`_decimal_level`）、`kg.py`（`cn_to_int`）、`legal_kb.py`（`tokenize`）四处，
同一件事有 3 份不完全一致的实现（例如 `cn_to_int` 只支持到 99，`_money_to_number`
不认大写金额"壹拾万贰仟元整"）。v2 收敛到这里，并且**补上两块 v1 缺的**：

1. 中文大写金额（合同里"人民币壹拾万零贰仟元整"极常见，v1 直接漏抽）；
2. 中文数字日期（"二〇二五年三月一日"，v1 只认阿拉伯数字，导致 date_reversed
   这类风险在中文日期合同上完全失效）。
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date
from typing import Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# 基础规范化
# ---------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_SPACE_RE = re.compile(r"[ \t\u3000\xa0]+")
_MULTI_NL = re.compile(r"\n{3,}")
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"


def strip_invisible(text: str) -> str:
    return text.translate(_ZERO_WIDTH)


def full_to_half(text: str) -> str:
    """全角 ASCII → 半角（不动中文标点，避免破坏"。"等断句符）。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str, *, keep_newlines: bool = True) -> str:
    """去不可见字符、压空白、统一换行。**不删任何可见字符**（可溯源要求）。"""
    if not text:
        return ""
    text = strip_invisible(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RE.sub(" ", text)
    if keep_newlines:
        text = _MULTI_NL.sub("\n\n", text)
    else:
        text = text.replace("\n", " ")
    return text.strip()


def squash(text: str) -> str:
    """比较用的极端压缩：去所有空白与标点。"""
    return re.sub(rf"[\s\W_]+", "", strip_invisible(text))


def has_cjk(text: str) -> bool:
    return re.search(rf"[{_CJK}]", text) is not None


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(re.findall(rf"[{_CJK}]", text))
    return cjk / max(1, len(text))


# ---------------------------------------------------------------------------
# 中文数字
# ---------------------------------------------------------------------------

_CN_DIGITS: dict[str, int] = {
    "零": 0, "〇": 0, "○": 0, "０": 0, "0": 0,
    "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2, "貳": 2, "２": 2, "2": 2,
    "三": 3, "叁": 3, "參": 3, "３": 3, "3": 3,
    "四": 4, "肆": 4, "４": 4, "4": 4,
    "五": 5, "伍": 5, "５": 5, "5": 5,
    "六": 6, "陆": 6, "陸": 6, "６": 6, "6": 6,
    "七": 7, "柒": 7, "７": 7, "7": 7,
    "八": 8, "捌": 8, "８": 8, "8": 8,
    "九": 9, "玖": 9, "９": 9, "9": 9,
}

_CN_UNITS: dict[str, int] = {
    "十": 10, "拾": 10, "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
}


def cn_to_int(text: str) -> int | None:
    """中文/混排数字 → int。支持"十一""二十一""一百零五""壹拾贰""二〇二五"。

    解析失败返回 None（调用方必须能处理 None，v1 里 `cn_to_int` 返回 0 会把
    "第十一条"的引用判定误伤成"第一条"）。
    """
    if not text:
        return None
    text = full_to_half(strip_invisible(text)).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return int(text)
        except ValueError:
            return None

    # 纯位值串（年份/编号）：二〇二五 → 2025
    if len(text) > 1 and all(ch in _CN_DIGITS and _CN_DIGITS[ch] < 10 for ch in text):
        if not any(ch in _CN_UNITS or ch in "万亿萬億" for ch in text):
            digits = "".join(str(_CN_DIGITS[ch]) for ch in text)
            return int(digits) if digits else None

    total = 0       # 已结算的"万/亿"段
    section = 0     # 当前段
    number = 0      # 当前累计个位
    seen = False
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            seen = True
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            section += (number if number else 1) * unit
            number = 0
            seen = True
        elif ch == "万" or ch == "萬":
            section += number
            total += (section if section else 1) * 10_000
            section = 0
            number = 0
            seen = True
        elif ch == "亿" or ch == "億":
            section += number
            total = (total + section) * 100_000_000
            section = 0
            number = 0
            seen = True
        else:
            continue
    if not seen:
        return None
    return total + section + number


# ---------------------------------------------------------------------------
# 金额
# ---------------------------------------------------------------------------

_AMOUNT_UNIT = {
    "元": 1.0, "圆": 1.0, "块": 1.0,
    "角": 0.1, "毛": 0.1,
    "分": 0.01,
}
_MAGNITUDE = {"万": 1e4, "萬": 1e4, "亿": 1e8, "億": 1e8, "千": 1e3, "百": 1e2}
_MONEY_RE = re.compile(
    r"(?:人民币|RMB|CNY|￥|¥)?\s*"
    r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|万圆|万|亿元|亿|元|圆)?",
    re.I,
)


def money_cn_to_float(text: str) -> float | None:
    """解析中文大写金额："人民币壹拾万零贰仟叁佰元整" → 102300.0。

    这是 v2 相对 v1 的实质补强：政府采购合同里金额同时出现大小写，v1 只能抽到
    阿拉伯数字那一路，遇到"总价以大写为准"的条款会抽空。
    """
    if not text:
        return None
    raw = full_to_half(strip_invisible(text))
    raw = raw.replace("人民币", "").replace("RMB", "").replace("人民币", "")
    raw = raw.replace("整", "").replace("正", "").replace("圆", "元").strip(" ￥¥")
    if not raw:
        return None

    # 拆出元/角/分三段
    yuan_part, rest = raw, ""
    for sep in ("元",):
        if sep in raw:
            yuan_part, rest = raw.split(sep, 1)
            break
    value = 0.0
    if yuan_part:
        whole = cn_to_int(yuan_part)
        if whole is None:
            return None
        value += float(whole)
    if rest:
        m = re.search(r"([零〇一二三四五六七八九壹贰叁肆伍陆柒捌玖\d]+)角", rest)
        if m:
            jiao = cn_to_int(m.group(1))
            if jiao is not None:
                value += jiao * 0.1
        m = re.search(r"([零〇一二三四五六七八九壹贰叁肆伍陆柒捌玖\d]+)分", rest)
        if m:
            fen = cn_to_int(m.group(1))
            if fen is not None:
                value += fen * 0.01
    return round(value, 2) if value else None


def money_to_float(text: str) -> float | None:
    """解析金额文本，阿拉伯数字与大写都能吃。

    支持："￥120,000.00" "12万元" "1.2亿元" "壹拾万元整" "120000元"。
    """
    if not text:
        return None
    raw = clean_text(text, keep_newlines=False)
    if not raw:
        return None

    # 先试阿拉伯数字（带可选量级单位）
    best: float | None = None
    for match in _MONEY_RE.finditer(raw):
        number_text, unit = match.group(1), match.group(2) or ""
        try:
            number = float(number_text.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        if unit in ("万元", "万圆", "万"):
            number *= 1e4
        elif unit in ("亿元", "亿"):
            number *= 1e8
        if number < 0:
            continue
        # 零金额是真实存在的缺陷信号（零租金、零费用），必须保留
        if best is None or number > best:
            best = number
    if best is not None:
        return round(best, 2)

    # 再试大写
    return money_cn_to_float(raw)


# ---------------------------------------------------------------------------
# 日期
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(?P<y>\d{4})\s*[-/.年]\s*[_＿\s]*(?P<m>\d{1,2})\s*[-/.月]\s*[_＿\s]*(?P<d>\d{1,2})\s*[_＿\s]*[日号]?",
)
# Word 表单里常用下划线占位（"2026年_10月29日"），下划线要先当分隔符吃掉，
# 否则 `20\d{2}年...` 这类日期整体解析失败，真实政采合同的签订日会成片漏抽。
_UNDERSCORE_GAP_RE = re.compile(r"(?<=[年月日])\s*[_＿]+\s*|\s*[_＿]+\s*(?=[年月日])")
_CN_DATE_RE = re.compile(
    r"(?P<y>[〇零○一二三四五六七八九]{2,4})\s*年\s*"
    r"(?P<m>[一二三四五六七八九十]{1,3})\s*月"
    r"(?:\s*(?P<d>[一二三四五六七八九十〇零]{1,3})\s*日)?",
)


def _deunderline(raw: str) -> str:
    """把"2026年_10月_29_日"里的下划线占位去掉，变成正常日期写法。"""
    return _UNDERSCORE_GAP_RE.sub("", raw)


def parse_date(text: str) -> date | None:
    """从任意文本里抓第一个日期。中文数字日期也能抓（v1 抓不到）。"""
    if not text:
        return None
    raw = _deunderline(full_to_half(strip_invisible(text)))
    m = _DATE_RE.search(raw)
    if m:
        try:
            return date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except ValueError:
            return None
    m = _CN_DATE_RE.search(raw)
    if m:
        year = cn_to_int(m.group("y"))
        month = cn_to_int(m.group("m"))
        day = cn_to_int(m.group("d")) if m.group("d") else 1
        if year and month and day:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def parse_date_text(text: str) -> str:
    """规范化日期字符串，解析失败返回空串（**不返回原串**，避免脏值污染要素表）。"""
    found = parse_date(text)
    return found.isoformat() if found else ""


def all_dates(text: str) -> list[date]:
    """抓出文本中所有日期（用于"签订日 vs 生效日"这类跨句比较）。"""
    raw = _deunderline(full_to_half(strip_invisible(text)))
    found: list[date] = []
    for m in _DATE_RE.finditer(raw):
        try:
            found.append(date(int(m.group("y")), int(m.group("m")), int(m.group("d"))))
        except ValueError:
            continue
    for m in _CN_DATE_RE.finditer(raw):
        year = cn_to_int(m.group("y"))
        month = cn_to_int(m.group("m"))
        day = cn_to_int(m.group("d")) if m.group("d") else 1
        if year and month and day:
            try:
                found.append(date(year, month, day))
            except ValueError:
                continue
    return found


# ---------------------------------------------------------------------------
# 条款编号
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"^\s*(?:第\s*([〇零一二三四五六七八九十百千壹贰叁肆伍陆柒捌玖\d]+)\s*条)"
    r"|^\s*([0-9]{1,2}(?:\.[0-9]{1,2}){0,3})[\.、\s]"
    r"|^\s*([（(]\s*([〇零一二三四五六七八九十\d]{1,3})\s*[)）])"
    r"|^\s*([一二三四五六七八九十]{1,3})\s*[、.]"
)

_CHAPTER_RE = re.compile(r"^\s*(第\s*[〇零一二三四五六七八九十百\d]+\s*[编章节部分])")


def detect_label(text: str) -> dict[str, object]:
    """识别段落开头的条款编号，返回 label/level/number。

    与 v1 `detect_heading` 的差别：v1 把"章"和"条"混在一个 level 递增的逻辑里，
    遇到"（一）"这种罗马式再嵌套就错层。v2 显式给层级：
    第X编/章 = 1，第X条 = 2，X.Y = 2+小数位数，X.Y.Z = 3，（一）= 4。
    """
    if not text:
        return {"label": "", "level": 0, "number": None, "kind": ""}
    head = text.lstrip()

    m = _CHAPTER_RE.match(head)
    if m:
        label = re.sub(r"\s+", "", m.group(1))
        number = cn_to_int(re.sub(r"[第编章节部分]", "", label))
        return {"label": label, "level": 1, "number": number, "kind": "chapter"}

    m = _ARTICLE_RE.match(head)
    if m:
        if m.group(1):
            label = f"第{m.group(1)}条"
            return {"label": label, "level": 2,
                    "number": cn_to_int(m.group(1)), "kind": "article"}
        if m.group(2):
            label = m.group(2)
            return {"label": label, "level": 2 + label.count("."),
                    "number": None, "kind": "decimal"}
        if m.group(4):
            label = f"（{m.group(4)}）"
            return {"label": label, "level": 4,
                    "number": cn_to_int(m.group(4)), "kind": "item"}
        if m.group(5):
            label = m.group(5)
            return {"label": label, "level": 4,
                    "number": cn_to_int(m.group(5)), "kind": "item"}
    return {"label": "", "level": 0, "number": None, "kind": ""}


_ARTICLE_LABEL_RE = re.compile(
    r"^\s*(第\s*[〇零一二三四五六七八九十百千壹贰叁肆伍陆柒捌玖\d]+\s*条"
    r"(?:之[〇零一二三四五六七八九十]+)?)"
)
_CHAPTER_LABEL_RE = re.compile(r"^\s*(第\s*[〇零一二三四五六七八九十百\d]+\s*[编章节部分])")
_DECIMAL_LABEL_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,3})(?=[\s、．]|[一-鿿])")
_ITEM_LABEL_RE = re.compile(r"^\s*([（(]\s*[〇零一二三四五六七八九十\d]{1,3}\s*[)）])")
_CN_ORD_LABEL_RE = re.compile(r"^\s*([一二三四五六七八九十]{1,3}[、．.])")


def label_heading_split(text: str) -> tuple[str, str]:
    """把"第一条 合同标的"拆成 ("第一条", "合同标的")。

    v2 相对 v1 补了数字编号（`1.1`/`1.1.1`）与中文序号（`一、`）两种，
    否则 Word 里用多级列表排的合同，标题会整段吞进正文。
    """
    if not text:
        return "", ""
    for pattern in (_ARTICLE_LABEL_RE, _CHAPTER_LABEL_RE, _DECIMAL_LABEL_RE,
                    _ITEM_LABEL_RE, _CN_ORD_LABEL_RE):
        match = pattern.match(text)
        if not match:
            continue
        label = re.sub(r"\s+", "", match.group(1))
        rest = text[match.end():]
        # 数字编号后面紧跟"年月日"的绝不可能是条款号（避免把 "2025年" 当编号）
        if pattern is _DECIMAL_LABEL_RE and rest[:1] in ("年", "月", "日", "%"):
            continue
        return label, rest.lstrip(" ：:、.．　")
    return "", text


# ---------------------------------------------------------------------------
# 相似度与分句
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[。；;！!？?\n])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts


def shingles(text: str, n: int = 3) -> set[str]:
    core = squash(text)
    if len(core) <= n:
        return {core} if core else set()
    return {core[i: i + n] for i in range(len(core) - n + 1)}


def jaccard(left: str, right: str, n: int = 3) -> float:
    a, b = shingles(left, n), shingles(right, n)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def overlap_coefficient(left: str, right: str, n: int = 3) -> float:
    a, b = shingles(left, n), shingles(right, n)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ---------------------------------------------------------------------------
# 检索分词（BM25 用）
# ---------------------------------------------------------------------------

_LATIN_NUM = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_STOP_CHARS = set("的了和与及或在为是对其之而则即也都着把被将从以于")


def tokenize(text: str, *, with_bigram: bool = True) -> list[str]:
    """字 unigram + bigram + 拉丁/数字串。

    与 v1 `legal_kb.tokenize` 保持一致的口径（保证对照实验公平），但补了
    * 全角转半角（v1 没转，"（一）"和"(一)"会被当成不同词）；
    * 停用字单字丢弃、双字保留，减少高频虚字对 BM25 的干扰。
    """
    if not text:
        return []
    raw = full_to_half(strip_invisible(text)).lower()
    tokens: list[str] = []
    prev = ""
    for ch in raw:
        if re.match(rf"[{_CJK}]", ch):
            if ch not in _STOP_CHARS:
                tokens.append(ch)
            if with_bigram and prev:
                tokens.append(prev + ch)
            prev = ch
        else:
            prev = ""
    tokens.extend(m.group(0) for m in _LATIN_NUM.finditer(raw))
    return tokens


# ---------------------------------------------------------------------------
# 其它
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def round_sig(value: float, digits: int = 4) -> float:
    if value == 0 or not math.isfinite(value):
        return value
    return round(value, digits)


def truncate(text: str, limit: int, ellipsis: str = "…") -> str:
    text = text or ""
    return text if len(text) <= limit else text[: max(0, limit - len(ellipsis))] + ellipsis


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def first_index(text: str, needles: Sequence[str]) -> int:
    positions = [text.find(n) for n in needles if n and text.find(n) >= 0]
    return min(positions) if positions else -1


def normalize_person(name: str) -> str:
    """统一当事人名称：去括号别名、去空白、全角转半角。"""
    name = full_to_half(strip_invisible(name))
    name = re.sub(r"[（(][^）)]{0,20}[）)]", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip("：: ,，、。")
