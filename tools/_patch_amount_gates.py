"""给 AMOUNT 加两道后处理闸：中文大写重复、单价/费率这类非合同级金额。

依据（实测，见 tools/_ner_amount_fp2.py 与 _ner_gold_cue.py）：
- 留出集 AMOUNT 误报 37 个，**全部是真金额**：中文大写（贰佰伍拾捌万元整）、
  单价（每平方米人民币21,500.00元）、余款（2374000.00元）。金标只收"合同总额"，
  所以这些在严格口径下全是 fp。
- 金标 AMOUNT 34 个，**34 个都带线索**（金额 9 / 人民币 9 / 总价 4 / 小写 12），
  且没有任何一个是大写金额或单价 —— 说明这两类判据与金标不冲突，是净收益。

闸门只做"同一金额的大写重复"与"单价/费率"两件事，不按线索词白名单放行，
避免把"合同金额 500000 元"这类正确抽取也筛掉。
"""

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "bert_ner.py"
s = p.read_text(encoding="utf-8")

helpers = '''
_ARABIC_AMOUNT_RE = __import__("re").compile(r"\\d[\\d,\\.]*\\s*(?:万|亿)?\\s*(?:元|圆)")
_UNIT_PRICE_CUES = ("单价", "税率", "费率", "每平方米", "每平米", "㎡", "每人", "每小时", "每台", "每件")


def _amount_duplicate_capital(text: str, start: int, end: int) -> bool:
    """中文大写金额与阿拉伯小写金额是同一个实体，只保留小写那份。

    实测："（小写）2374000.00元，（大写）贰佰叁拾柒万肆仟元整。"
    模型两份都抽，而金标只收清单里的"2374000.00"。判据：实体不含阿拉伯数字
    却全是中文数字，且前后 16 字内有阿拉伯金额 —— 判为重复表述。
    """
    fragment = text[start:end]
    if any(ch.isdigit() for ch in fragment):
        return False
    if not any(ch in CN_DIGITS for ch in fragment):
        return False
    window = text[max(0, start - 16): min(len(text), end + 16)]
    return bool(_ARABIC_AMOUNT_RE.search(window))


def _amount_is_unit_price(text: str, start: int, end: int) -> bool:
    """单价/税率/费率不是合同级金额：报了只会触发 abnormal_price 误报。

    反例要留："总价款为人民币 XXX 元"里若同句出现"单价为每平方米 YYY 元"，
    这两处是不同实体，靠距离区分——单价线索必须出现在实体**前 14 字**内，
    而实体自身不含"总价/合同价/合同金额/价款合计"。
    """
    fragment = text[start:end]
    if any(cue in fragment for cue in ("总价", "合同价", "合同金额", "价款")):
        return False
    head = text[max(0, start - 14): start]
    return any(cue in head for cue in _UNIT_PRICE_CUES)

'''

anchor = "def _span_is_valid(fragment: str, kind: str) -> bool:"
assert anchor in s, "锚点没找到"
s = s.replace(anchor, helpers.lstrip("\n") + "\n" + anchor, 1)

old = """        if span[2] == "AMOUNT" and not _amount_context_ok(text, span[0], span[1]):
            continue"""
new = """        if span[2] == "AMOUNT":
            if not _amount_context_ok(text, span[0], span[1]):
                continue
            if _amount_duplicate_capital(text, span[0], span[1]):
                continue
            if _amount_is_unit_price(text, span[0], span[1]):
                continue"""
assert old in s, "predict 过滤锚点没找到"
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("bert_ner.py: AMOUNT 两道后处理闸已加")
