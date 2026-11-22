"""Narrow literal fact slots with exact source offsets; never verified facts.

No model, network, fuzzy matching or cross-clause subject inheritance occurs.
Unknown roles/subjects/conditions remain explicit.  Values in different payment
roles are not interchangeable and are never blindly added together.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any

VERSION = "literal-fact-slots-v1"
ROLES = {"合同总金额": "contract_total", "合同总价": "contract_total", "合同价款": "contract_total",
    "总金额": "contract_total", "总价": "contract_total", "预付款": "advance", "首付款": "first_payment",
    "首期款": "first_instalment", "第二期款": "second_instalment", "第三期款": "third_instalment",
    "尾款": "tail", "质保金": "retention", "保证金": "deposit", "租金": "rent", "价款": "price_unspecified"}
_ROLE = re.compile("|".join(sorted(ROLES, key=len, reverse=True)))
_ARABIC = re.compile(r"(?<![\d.])(?P<number>\d+(?:[,，]\d{3})*(?:\.\d{1,2})?)\s*(?P<unit>亿元|万元|元)")
_FINANCIAL = re.compile(r"[零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟]+元(?:[零〇一二三四五六七八九壹贰叁肆伍陆柒捌玖]+角)?(?:[零〇一二三四五六七八九壹贰叁肆伍陆柒捌玖]+分)?(?:整|正)?")
_ACTOR = re.compile(r"甲方|乙方|丙方|买方|卖方|出租方|承租方")
_PAY_TO = re.compile(r"(?P<payer>甲方|乙方|买方|卖方)(?:应当|应|须|需)?向(?P<payee>甲方|乙方|买方|卖方)[^，。；;\n]{0,12}?(?:支付|付清|付款)")
_OBJECT = re.compile(r"本合同|本项目|本设备|本服务|(?:项目|设备|标的)[A-Za-z0-9甲乙丙丁一二三四五六七八九十]+")
_EVENT = re.compile(r"验收|支付|付清|付款|交付|交货|退还|退款")
_EVENT_NAMES = {"验收": "acceptance", "支付": "payment", "付清": "payment", "付款": "payment",
                "交付": "delivery", "交货": "delivery", "退还": "refund", "退款": "refund"}
_STAGE = re.compile(r"最终|初步|首批|第[一二三四五六七八九十0-9]+批")
_CONDITION = re.compile(r"(?:在|待|自)[^，。；;\n]{1,60}?(?:之后|以前|之日|后|前)|"
    r"(?:若|如果|如)[^，。；;\n]{1,60}|以[^，。；;\n]{1,60}?为前提|"
    r"(?:甲方|乙方|买方|卖方)?(?:已|完成|收到)?(?:本合同|本项目)?(?:最终|初步)?(?:验收|支付|付清|交付|交货)[^，。；;\n]{0,20}?(?:后|前)")
_EXCEPTION = re.compile(r"除非[^，。；;\n]{1,60}|除[^，。；;\n]{1,60}?外|但[^，。；;\n]{1,60}|另有约定")


def _id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def financial_yuan(value: str) -> Decimal | None:
    """Parse the explicit Chinese numeric grammar, abstaining on unknown forms."""
    digits = {char: number for number, chars in enumerate(("零〇", "一壹", "二贰", "三叁", "四肆", "五伍", "六陆", "七柒", "八捌", "九玖")) for char in chars}
    units = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
    match = _FINANCIAL.fullmatch(value)
    if not match:
        return None
    integer, tail = value.split("元", 1)
    total = section = number = 0
    previous_digit = None
    previous_small_unit = 10000
    for char in integer:
        if char in digits:
            if previous_digit not in (None, 0):
                return None  # Adjacent non-positional digits are ambiguous.
            number = digits[char]
            previous_digit = number
        elif char in units:
            if units[char] >= previous_small_unit:
                return None
            section += (number or 1) * units[char]
            number = 0
            previous_digit = None
            previous_small_unit = units[char]
        elif char in ("万", "亿"):
            section += number
            number = 0
            if char == "万":
                section *= 10000
            else:
                total = (total + section) * 100000000
                section = 0
            previous_digit = None
            previous_small_unit = 10000
        else:
            return None
    result = Decimal(total + section + number)
    for unit, divisor in (("角", 10), ("分", 100)):
        fractional = re.search(r"([零〇一二三四五六七八九壹贰叁肆伍陆柒捌玖])" + unit, tail)
        if fractional:
            result += Decimal(digits[fractional.group(1)]) / divisor
    return result


def extract_fact_slots(clause_id: str, text: str, *, start_offset: int = 0) -> dict[str, Any]:
    """Extract literal amounts and operative/conditional event mentions.

    Offsets are Unicode characters relative to the original clause.  A source
    hash binds the exact supplied segment, not an asserted complete contract.
    """
    if not isinstance(clause_id, str) or not clause_id or not isinstance(text, str):
        raise ValueError("An explicit clause ID and exact source string are required")
    if type(start_offset) is not int or start_offset < 0:
        raise ValueError("start_offset must be a nonnegative integer")
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def span(start, end):
        return {"clause_id": clause_id, "start": start_offset + start, "end": start_offset + end,
                "quote": text[start:end], "source_sha256": source_hash}

    def actor(field, base):
        explicit = _PAY_TO.search(field)
        if explicit:
            return {"subject": explicit.group("payer"), "recipient": explicit.group("payee"),
                "subject_span": span(base + explicit.start("payer"), base + explicit.end("payer")),
                "recipient_span": span(base + explicit.start("payee"), base + explicit.end("payee")), "ambiguous": False}
        found = list(_ACTOR.finditer(field))
        if len({match.group() for match in found}) == 1:
            match = found[0]
            return {"subject": match.group(), "recipient": None,
                    "subject_span": span(base + match.start(), base + match.end()), "recipient_span": None, "ambiguous": False}
        return {"subject": None, "recipient": None, "subject_span": None, "recipient_span": None,
                "ambiguous": len(found) > 1}

    def object_scope(field, base):
        found = list(_OBJECT.finditer(field))
        if len({match.group() for match in found}) == 1:
            match = found[0]
            return {"name": match.group(), "span": span(base + match.start(), base + match.end())}
        return {"name": None, "span": None}

    amounts, obligations, sentences = [], [], []
    for sentence in re.finditer(r"[^。；;\n]+[。；;\n]?", text):
        field, base = sentence.group(), sentence.start()
        scope = object_scope(field, base)
        conditions = [{"span": span(base + match.start(), base + match.end()),
            "timing": "before" if match.group().endswith(("前", "以前")) else "after" if match.group().endswith(("后", "之后")) else "conditional",
            "negated": bool(re.search(r"未|不|无", match.group())),
            "events": list(dict.fromkeys(_EVENT_NAMES[item.group()] for item in _EVENT.finditer(match.group())))}
            for match in _CONDITION.finditer(field)]
        exceptions = [span(base + match.start(), base + match.end()) for match in _EXCEPTION.finditer(field)]
        masked = list(field)
        for condition in conditions:
            left = condition["span"]["start"] - start_offset - base
            right = condition["span"]["end"] - start_offset - base
            masked[left:right] = " " * (right-left)
        operative_field = "".join(masked)
        actors = actor(operative_field, base)
        sentences.append({"span": span(sentence.start(), sentence.end()), "conditions": conditions, "exceptions": exceptions,
            "unparsed_condition_markers": bool(re.search(r"后|前|若|如果|为前提", field) and not conditions)})
        matches = [(match, "arabic") for match in _ARABIC.finditer(field)] + [(match, "chinese") for match in _FINANCIAL.finditer(field)]
        for match, writing in sorted(matches, key=lambda row: row[0].start()):
            if writing == "arabic":
                try:
                    value = Decimal(re.sub(r"[,，]", "", match.group("number"))) * {"元": 1, "万元": 10000, "亿元": 100000000}[match.group("unit")]
                except InvalidOperation:
                    value = None
            else:
                value = financial_yuan(match.group())
            prefix = field[:match.start()]
            roles = list(_ROLE.finditer(prefix))
            role = roles[-1] if roles and match.start() - roles[-1].end() <= 48 else None
            qualifiers = re.findall(r"不少于|不低于|不超过|最高|最低|至少|预算|估算|约为|预计|参考", prefix[-20:])
            polarity = "negative" if re.search(r"(?:无需|不得|不再|不予|免于|不)(?:向[甲乙丙]方)?(?:支付|付款|付清)[^，。；;]{0,15}$", prefix) else "positive"
            amount_actors = actors
            for condition in conditions:
                if condition["span"]["start"] <= start_offset + base + match.start() < condition["span"]["end"]:
                    amount_actors = actor(condition["span"]["quote"], condition["span"]["start"] - start_offset)
                    break
            row = {"amount_role": ROLES[role.group()] if role else "unknown", "role_span": span(base + role.start(), base + role.end()) if role else None,
                "value_yuan": format(value, "f") if value is not None else None, "currency": "CNY", "writing": writing,
                "source_span": span(base + match.start(), base + match.end()), "sentence_span": span(sentence.start(), sentence.end()),
                **amount_actors, "object_scope": scope, "stage": ROLES[role.group()] if role else "unknown",
                "conditions": conditions, "exceptions": exceptions, "polarity": polarity,
                "value_kind": "qualified" if qualifiers else "literal", "qualifiers": qualifiers,
                "fact_verified": False, "unparsed": (["amount_role_unknown"] if role is None else [])
                    + (["subject_unknown"] if amount_actors["subject"] is None else []) + (["object_scope_unknown"] if scope["name"] is None else [])}
            row["slot_id"] = _id("amount_", row)
            amounts.append(row)
        for event in _EVENT.finditer(field):
            # Section titles such as 付款方式 or 验收标准 do not assert an obligation.
            following, preceding = field[event.end():event.end()+4], field[:event.start()]
            if re.match(r"方式|标准|流程|条件|时间|日期|节点", following) and not re.search(r"应|须|需|必须|不得|启动|开始|完成", preceding[-10:]):
                continue
            event_start = base + event.start()
            conditional = any(item["span"]["start"] <= start_offset + event_start < item["span"]["end"] for item in conditions)
            operative_prefix = operative_field[:event.start()]
            # A phase applies only immediately before its event; a phase in
            # an earlier prerequisite must not leak into the operative verb.
            stage_match = re.search(r"(最终|初步|首批|第[一二三四五六七八九十0-9]+批)(?:完成|启动|开始|进行)?$", operative_prefix)
            stage = stage_match.group(1) if stage_match else "unspecified"
            negative = bool(re.search(r"(?:不得|不应|不再|不予|无需|不)(?:启动|开始|进行|予以)?$", preceding[-10:]))
            operative_actor = actor(operative_prefix, base)
            row = {"event": _EVENT_NAMES[event.group()], "event_literal": event.group(),
                "source_span": span(event_start, base + event.end()), "sentence_span": span(sentence.start(), sentence.end()),
                **operative_actor, "object_scope": scope, "stage": stage,
                "mention_role": "condition_event" if conditional else "operative",
                "polarity": "negative" if negative else "positive",
                "modality": "prohibited" if negative else "required" if re.search(r"应|须|需|必须|仅|才", preceding[-25:]) else "stated",
                "conditions": conditions if not conditional else [], "exceptions": exceptions,
                "fact_verified": False}
            row["proposition_id"] = _id("proposition_", row)
            obligations.append(row)
    return {"version": VERSION, "clause_id": clause_id,
        "source": {"text": text, "start_offset": start_offset, "end_offset": start_offset + len(text), "sha256": source_hash},
        "amounts": amounts, "obligations": obligations, "sentences": sentences,
        "meaning": "Literal source slots only; unknowns are not inherited, facts and legal truth remain unverified"}
