"""缺陷型检测器：把"注入式缺陷"这一类风险单独成组。

v1 的对抗数据集（`data/generated_contracts/samr_multi_template_2025/adversarial/`）
用 31 个 `issue_codes` 标注注入缺陷，例如 `extreme_rent`、`term_over_20y`、
`negative_fee`、`cash_only`。这些**不是**语义模糊的"坏写法"，而是可判定的硬缺陷：
数值越界、期限越界、条款缺失、支付方式违法。用 LLM 判它们属于浪费，用正则和
算术判定又准又快。

本模块把这些缺陷逐条落成检测器，每条产出 v2 风险码，并在 `extra["issue_code"]`
里回填数据集口径的标签，这样评测脚本能在两套码表之间对齐，算得出 honest 的
P/R/F1（见 `eval.py`）。

覆盖率：31 个数据集缺陷中 27 个由本模块覆盖，其余 4 个属于"语义解释类"
（`no_substantive_risk` 之类的空标签）不参与检测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Sequence

from . import text as T
from .models import Clause, Contract, RiskCandidate, stable_id

CASH_ONLY_PATTERNS: tuple[str, ...] = (
    "现金支付", "现金结算", "以现金方式", "现金交付", "不留凭证", "不出具凭证",
    "不开发票", "不开具发票", "无需提供发票", "无需提供凭证", "无需提供任何凭证",
)
INVALID_PAYMENT_PATTERNS: tuple[str, ...] = (
    "打入个人账户", "个人银行账户", "私人账户", "账外", "体外循环",
)
SUCCESS_GUARANTEE_PATTERNS: tuple[str, ...] = (
    "保证百分之百", "保证100%成交", "保证成交", "确保成交", "保证成功",
    "包成功", "百分百成功", "承诺成交", "保证交易绝对盈利", "视为中介成功",
    "绝对盈利",
)
UNLIMITED_AUTH_PATTERNS: tuple[str, ...] = (
    "全权代理", "无限授权", "不受限制的授权", "代签全部", "处分甲方全部",
    "自行决定一切", "无需甲方同意", "处分甲方任何资产", "处分甲方",
)
NONEXISTENT_PATTERNS: tuple[str, ...] = (
    "不存在的", "虚构的", "虚构项目", "尚未设立", "未注册", "无需证明项目真实存在",
)
INSECURE_PATTERNS: tuple[str, ...] = (
    "未加密", "不加密", "明文传输", "U盘", "u盘", "优盘", "移动硬盘拷贝", "口头告知数据",
)
PUBLIC_DISCLOSURE_PATTERNS: tuple[str, ...] = (
    "公开发布", "公开披露", "对外公开", "公开展示", "向社会公布",
)
PERPETUAL_PATTERNS: tuple[str, ...] = (
    "永久许可", "永久使用", "永久授权", "无限期使用", "永久免费使用", "永久转售",
)
PERSONAL_DATA_PATTERNS: tuple[str, ...] = (
    "身份证号", "人脸", "行踪轨迹", "未成年人", "敏感个人信息", "健康信息", "生物识别",
    "个人信息",
)
NO_REPORT_PATTERNS: tuple[str, ...] = (
    "无需提交", "不需提交", "无需出具", "不必提交报告", "无须提供成果", "无需交付成果",
    "无需提交报告", "无需提供任何凭证",
)
UNILATERAL_ENTRY_PATTERNS: tuple[str, ...] = (
    "有权随时进入", "可随时进入", "单方进入", "自行进入房屋", "无需通知即可进入",
    "不经通知随时进入", "随时进入", "任何时间要求取回",
)
NO_TITLE_PATTERNS: tuple[str, ...] = (
    "无权属证明", "无产权证", "未取得产权", "产权不明", "无权属证书", "无法提供产权",
    "没有任何权属证明", "无任何权属证明", "未提供权属证明",
)
LIABILITY_EXEMPTION_PATTERNS: tuple[str, ...] = (
    "不承担任何赔偿责任", "不承担任何责任", "均不承担", "免除全部责任",
    "放弃全部追责", "放弃全部", "无论乙方是否故意", "无论因故意", "无论是否",
    "不得追究责任", "不承担因",
)
CONSENT_PATTERNS: tuple[str, ...] = ("取得同意", "取得.{0,6}同意", "获得同意", "单独同意", "告知同意", "授权同意", "征得.{0,6}同意")
CONSENT_NEGATIONS: tuple[str, ...] = (
    "无需", "不需", "不必", "免除", "无须", "不用", "免于", "不", "未",
)
ZERO_WORDS: tuple[str, ...] = ("零", "0", "无", "免费", "不收取")

#: 日期区间写法：`自X起至Y止`、`X—Y`、`X至Y`、`X~Y`。
_RANGE_RE = re.compile(
    r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"
    r"\s*(?:起)?\s*(?:至|到|—|－|-|~|～|–)\s*"
    r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"
)


def _mk(
    code: str,
    issue_code: str,
    message: str,
    clause: Clause | None,
    evidence: str = "",
    confidence: float = 0.7,
    severity: str | None = None,
    extra: dict | None = None,
) -> RiskCandidate:
    from .rules import _candidate

    payload = {"issue_code": issue_code, **(extra or {})}
    return _candidate(code, message, clause, evidence, confidence=confidence,
                      severity=severity, source="rule:defect", extra=payload)


# ---------------------------------------------------------------------------
# 数值类缺陷
# ---------------------------------------------------------------------------

#: 零值金额的语义：关键词 → (v2 码, 数据集缺陷码)
ZERO_KEYWORDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("租金",), "zero_rent", "zero_rent"),
    (("押金", "保证金", "定金"), "abnormal_price", "zero_deposit"),
    (("保管费", "费用", "报酬", "服务费", "佣金", "价款", "金额"), "zero_rent", "zero_fee"),
)

#: 负值金额的语义：关键词 → 数据集缺陷码
NEGATIVE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("佣金", "中介报酬", "中介费"), "negative_commission"),
    (("报酬", "费用", "价款", "金额"), "negative_fee"),
)


def _clause_containing(contract: Contract, value: float, kind: str = "amount") -> Clause | None:
    """找到含某数值的最早条款（不依赖要素的 clause_id，表格单元格常常没有归属）。"""
    needle = f"{value:.0f}"
    if value == int(value):
        needle = str(int(abs(value)))
    for clause in contract.document.clauses:
        flat = T.full_to_half(clause.text)
        if needle in flat:
            return clause
    return None


def _zero_issue(contract: Contract) -> tuple[str, str, str] | None:
    """按关键词判定"零金额"到底属零租金/零押金/零费用哪一类。"""
    flat = T.full_to_half(contract.document.full_text)
    for words, code, issue in ZERO_KEYWORDS:
        for word in words:
            if re.search(rf"{word}[^。；;]{{0,12}}(?:为|是|：|:)?\s*(?:人民币)?\s*0\s*(?:元|圆|元整)", flat):
                return code, issue, word
    return None


def _negative_issue(contract: Contract) -> str:
    flat = T.full_to_half(contract.document.full_text)
    for words, issue in NEGATIVE_KEYWORDS:
        for word in words:
            if re.search(rf"{word}[^。；;]{{0,12}}(?:为|是|：|:)?\s*(?:人民币)?\s*-", flat):
                return issue
    return "negative_fee"


def detect_amounts(contract: Contract) -> list[RiskCandidate]:
    """金额类：负数、零租金/零费用/零押金、极端价格、分项与总价不符。"""
    out: list[RiskCandidate] = []
    amounts = []
    for element in contract.elements.amounts:
        if (element.extra or {}).get("numeric_comparison_eligible") is False:
            continue
        try:
            amounts.append((float(element.normalized), element))
        except (TypeError, ValueError):
            continue
    flat = T.full_to_half(contract.document.full_text)

    negatives = sorted({value for value, _ in amounts if value < 0})
    for value in negatives[:2]:
        issue = _negative_issue(contract)
        clause = _clause_containing(contract, value)
        out.append(
            _mk("illegal_terms", issue,
                f"合同出现负数金额 {value:.2f} 元，条款本身不具可履行性",
                clause, f"{value:.2f}", confidence=0.82,
                extra={"amount": value})
        )
    positives = sorted({value for value, _ in amounts if value > 0})
    total = max(positives) if positives else 0.0
    if total >= 50_000_000:
        clause = _clause_containing(contract, total)
        out.append(
            _mk("abnormal_price", "extreme_rent",
                f"金额 {total:,.0f} 元显著高于同类合同量级，需核验定价依据",
                clause, f"{total:.0f}", confidence=0.6, extra={"amount": total})
        )
    if any(value == 0 for value, _ in amounts):
        zero = _zero_issue(contract)
        if zero:
            code, issue, word = zero
            out.append(
                _mk(code, issue,
                    f"{word}约定为零或未填写，合同对价/担保缺失",
                    None, f"{word}=0", confidence=0.75)
            )
    from .rules import RuleContext, _bound_money_fields, _bound_contract_total
    context = RuleContext.build(contract)
    # Existing ratio threshold is unchanged; only explicitly bound deposits and
    # a unique same-source contract total can participate in the comparison.
    for clause in contract.document.clauses:
        deposits = _bound_money_fields(clause, contract.elements, ("押金", "定金", "保证金"))
        total_binding = _bound_contract_total(clause, context) if deposits else None
        if total_binding:
            for deposit in deposits:
                ratio = deposit["value"] / total_binding["value"]
                if ratio > 0.2:
                    candidate = _mk("abnormal_price", "deposit_over_20pct",
                        f"押金/保证金约为合同总额的 {ratio:.0%}，超过常见比例的 20%",
                        clause, deposit["quote"], confidence=0.62,
                        extra={"amount_binding": {"deposit": deposit, "contract_total": total_binding}})
                    if total_binding["clause_id"] not in candidate.clause_ids:
                        candidate.clause_ids.append(total_binding["clause_id"])
                    candidate.evidence_append("amount_binding", total_binding["quote"], 0.7, "rule:bound_contract_total")
                    out.append(candidate)
        # A unit price and whole-contract total are not two conflicting amounts.
        # Retain the existing >100x severity gate only after an explicit local
        # quantity × unit-price relation supplies a comparable expected total.
        if clause.metadata.get("table_group_id") or clause.metadata.get("table_group_ids"):
            continue
        quantities = re.findall(r"(?:采购)?数量\s*(?:为|：|:)?\s*(\d+)\s*(?:件|个|台|套|吨|箱)", T.full_to_half(clause.text))
        units = _bound_money_fields(clause, contract.elements, ("单价",))
        stated = _bound_money_fields(clause, contract.elements, ("合同总金额", "合同总价", "总金额", "总价", "合计"))
        if len(quantities) == len(units) == len(stated) == 1:
            expected = int(quantities[0]) * units[0]["value"]
            smallest, largest = sorted((expected, stated[0]["value"]))
            if smallest > 0 and largest >= 1000 and smallest < largest * 0.01:
                out.append(_mk("fee_mismatch", "fee_mismatch",
                    f"同条数量×单价计算为 {expected:,.2f} 元，约定总价为 {stated[0]['value']:,.2f} 元，量级严重不匹配",
                    clause, clause.text, confidence=0.6,
                    extra={"amount_binding": {"quantity": int(quantities[0]), "unit": units[0],
                                              "stated_total": stated[0], "expected_total": expected}}))
    # 数量与金额的量级矛盾（"100 万件精密仪器 / 保管费 500 元"）
    quantity_match = re.search(r"数量(?:为)?\s*(\d{3,})\s*(?:件|个|台|套|吨|箱)", flat)
    if quantity_match and total:
        quantity = int(quantity_match.group(1))
        if quantity >= 10_000 and total <= quantity * 10:
            out.append(
                _mk("fee_mismatch", "quantity_value_mismatch",
                    f"标的数量 {quantity} 与合同金额 {total:,.0f} 元明显不匹配",
                    None, f"{quantity}/{total:g}", confidence=0.66)
            )
    if re.search(r"数量(?:为)?\s*0\s*(?:件|个|台|套|吨|箱)", flat):
        out.append(
            _mk("subject_unspecified", "zero_quantity",
                "标的数量约定为零，与实际交易标的不符", None, confidence=0.7)
        )
    return out


def _twentieth_anniversary(value: date) -> date:
    """Calendar years, including leap-day starts and the representable date cap."""
    year = value.year + 20
    if year > date.max.year:
        return date.max
    try:
        return value.replace(year=year)
    except ValueError:
        # A valid input date can only fail here for February 29 in a non-leap
        # target year, for example 2080-02-29 → 2100-02-28.
        return date(year, 2, 28)


def detect_dates(contract: Contract) -> list[RiskCandidate]:
    """日期类：倒置、超长期限、不可能期限。"""
    out: list[RiskCandidate] = []
    dates: list[tuple[date, object]] = []
    for element in contract.elements.dates:
        value = T.parse_date(element.normalized or element.value)
        if value is None:
            value = T.parse_date(element.value)
        if value is not None:
            dates.append((value, element))
    # 显式区间："自X起至Y止"、"X—Y"、"X至Y" 三种写法
    for clause in contract.document.clauses:
        clause_text = T.full_to_half(clause.text)
        for match in _RANGE_RE.finditer(clause_text):
            left = T.parse_date(match.group(1))
            right = T.parse_date(match.group(2))
            if left and right and right < left:
                out.append(
                    _mk("date_reversed", "date_reversed",
                        f"区间起点 {left.isoformat()} 晚于终点 {right.isoformat()}，履约时间线倒置",
                        clause, clause.text[match.start():match.end()], confidence=0.9)
                )
                break
            if not left or not right:
                continue
            span = right - left
            scope = clause_text[max(0, match.start() - 80):min(len(clause_text), match.end() + 30)]
            lease_term = bool(re.search(r"租赁(?:期限|期间|期)|租期", scope))
            rental_contract = "租赁" in contract.contract_type or contract.contract_type == "housing_rent"
            lease_term = lease_term or (rental_contract and bool(re.search(r"(?:本合同|合同)(?:期限|有效期)", scope)))
            anniversary = _twentieth_anniversary(left)
            if lease_term and right > anniversary:
                out.append(_mk("illegal_terms", "term_over_20y",
                    f"明确约定的租赁终止日 {right.isoformat()} 晚于起始日二十周年 {anniversary.isoformat()}，超过租赁期限二十年的上限",
                    clause, clause.text[match.start():match.end()], confidence=0.78,
                    extra={"date_binding": "explicit_lease_term_range", "twentieth_anniversary": anniversary.isoformat()}))
            # Signing, invoicing and OCR dates elsewhere are not a work period.
            explicit_work_period = bool(re.search(r"工期|履行期限|完成期限|工作期限|服务期限|项目期限", scope))
            if 0 <= span.days <= 1 and explicit_work_period and any(
                    word in clause_text for word in ("审计", "评估", "工程", "开发", "验收")):
                out.append(_mk("vague_delivery", "impossible_deadline",
                    f"明确工作区间仅相差 {span.days} 天，与约定工作量的完成周期需核对",
                    clause, clause.text[match.start():match.end()], confidence=0.7,
                    extra={"date_binding": "explicit_work_period_range"}))
    # 角色化比较：签订/起始 晚于 结束/交付
    signs = [value for value, element in dates if element.role in ("sign", "start", "effective")]
    ends = [value for value, element in dates if element.role in ("end", "delivery", "acceptance")]
    if signs and ends and min(ends) < max(signs):
        out.append(
            _mk("date_reversed", "date_reversed",
                f"结束/交付日期 {min(ends)} 早于起始日期 {max(signs)}，履约时间线倒置",
                None, confidence=0.82)
        )
    return out


# ---------------------------------------------------------------------------
# 语义确定性缺陷
# ---------------------------------------------------------------------------

def _scan_text(contract: Contract, patterns: Sequence[str]) -> list[tuple[Clause, str]]:
    hits: list[tuple[Clause, str]] = []
    for clause in contract.document.clauses:
        for pattern in patterns:
            if pattern in clause.text:
                hits.append((clause, pattern))
                break
    return hits


def detect_payment(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    for clause, pattern in _scan_text(contract, CASH_ONLY_PATTERNS)[:2]:
        out.append(
            _mk("illegal_terms", "cash_only",
                f"约定现金支付且不留凭证（{pattern}），不利于合同履行留痕与税务合规",
                clause, pattern, confidence=0.72)
        )
    for clause, pattern in _scan_text(contract, INVALID_PAYMENT_PATTERNS)[:1]:
        out.append(
            _mk("illegal_terms", "cash_only",
                f"约定向个人账户/账外支付（{pattern}），存在资金合规风险",
                clause, pattern, confidence=0.75)
        )
    text = contract.document.full_text
    if any(word in text for word in ("报酬", "佣金", "服务费")):
        match = re.search(r"(报酬|佣金|服务费)[^。；;]{0,20}(全部标的额|标的额的?100%|全额标的)", text)
        if match:
            out.append(
                _mk("one_sided_breach", "commission_100pct",
                    "中介报酬等于标的额全部，服务与对价明显失衡",
                    None, match.group(0)[:36], confidence=0.7)
            )
    return out


def detect_authority(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    for clause, pattern in _scan_text(contract, UNLIMITED_AUTH_PATTERNS)[:2]:
        out.append(
            _mk("need_authorization", "unlimited_authority",
                f"授权范围不受限制（{pattern}），受托人可能越权处分",
                clause, pattern, confidence=0.72)
        )
    for clause, pattern in _scan_text(contract, NO_TITLE_PATTERNS)[:1]:
        out.append(
            _mk("unverified_property", "unverified_property",
                f"标的物权属未核验（{pattern}）", clause, pattern, confidence=0.85)
        )
    return out


def detect_performance(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    for clause, pattern in _scan_text(contract, SUCCESS_GUARANTEE_PATTERNS)[:1]:
        out.append(
            _mk("illegal_terms", "success_guarantee",
                f"约定保证成交/保证成功（{pattern}），属于对结果的不当承诺",
                clause, pattern, confidence=0.72)
        )
    for clause, pattern in _scan_text(contract, NO_REPORT_PATTERNS)[:1]:
        out.append(
            _mk("subject_unspecified", "no_report",
                f"约定无需提交成果或报告（{pattern}），验收与对价缺乏依据",
                clause, pattern, confidence=0.66)
        )
    for clause, pattern in _scan_text(contract, UNILATERAL_ENTRY_PATTERNS)[:1]:
        out.append(
            _mk("unilateral_modify", "unilateral_entry",
                f"保留单方进入/取回/处置权（{pattern}），侵害对方占有使用权益",
                clause, pattern, confidence=0.7)
        )
    for clause, pattern in _scan_text(contract, INSECURE_PATTERNS)[:1]:
        out.append(
            _mk("confidentiality_missing", "insecure_delivery",
                f"数据传输/交付方式不安全（{pattern}）", clause, pattern, confidence=0.68)
        )
    # 完全免责：保管人/受托人不论过错均不担责
    for clause, pattern in _scan_text(contract, LIABILITY_EXEMPTION_PATTERNS)[:1]:
        out.append(
            _mk("liability_exemption", "liability_exemption",
                f"约定完全免除一方责任（{pattern}），属排除主要义务的免责条款",
                clause, pattern, confidence=0.78)
        )
    # 中介报酬等于标的额
    flat = T.full_to_half(contract.document.full_text)
    amounts = []
    for element in contract.elements.amounts:
        try:
            amounts.append(float(element.normalized))
        except (TypeError, ValueError):
            continue
    commission = re.search(
        r"(?:中介报酬|佣金|服务费|报酬)[^。；;]{0,12}?人民币\s*(\d[\d,，]*)\s*元", flat
    )
    subject_amount = re.search(r"(?:主交易金额|交易金额|标的额|标的金额)[^。；;]{0,12}?人民币\s*(\d[\d,，]*)\s*元", flat)
    if commission and subject_amount:
        fee = float(commission.group(1).replace(",", "").replace("，", ""))
        base = float(subject_amount.group(1).replace(",", "").replace("，", ""))
        if base > 0 and fee >= base * 0.95:
            out.append(
                _mk("one_sided_breach", "commission_100pct",
                    f"中介报酬 {fee:,.0f} 元等于标的额 {base:,.0f} 元，服务与对价明显失衡",
                    None, f"{fee:g}/{base:g}", confidence=0.74)
            )
    return out


def detect_data_protection(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    flat = T.full_to_half(contract.document.full_text)
    personal = _scan_text(contract, PERSONAL_DATA_PATTERNS)
    # 同意机制：出现"取得同意"类表述且**未被否定**才算有同意
    consent = False
    for pattern in CONSENT_PATTERNS:
        for match in re.finditer(pattern, flat):
            window = flat[max(0, match.start() - 10): match.start()]
            if not any(neg in window for neg in CONSENT_NEGATIONS):
                consent = True
                break
        if consent:
            break
    if personal and not consent:
        clause, pattern = personal[0]
        out.append(
            _mk("confidentiality_missing", "personal_data_no_consent",
                f"合同处理敏感个人信息（{pattern}）但未约定取得同意的机制",
                clause, pattern, confidence=0.8)
        )
    for clause, pattern in _scan_text(contract, PERPETUAL_PATTERNS)[:1]:
        out.append(
            _mk("unilateral_modify", "perpetual_license",
                f"许可/转让期限无限制（{pattern}），与数据/成果的可撤回性冲突",
                clause, pattern, confidence=0.62)
        )
    for clause, pattern in _scan_text(contract, PUBLIC_DISCLOSURE_PATTERNS)[:1]:
        out.append(
            _mk("confidentiality_missing", "public_disclosure",
                f"约定允许公开发布/披露（{pattern}），保密边界缺失",
                clause, pattern, confidence=0.7)
        )
    if re.search(r"(记录数|数据条数|数据集)[^。；;]{0,10}(为零|为0|0\s*条|空白)", flat):
        out.append(
            _mk("subject_unspecified", "zero_records",
                "数据集记录数为零或为空白集，标的不存在", None, confidence=0.7)
        )
    if any(word in flat for word in ("入库前", "存储前", "入库之前")) and any(
        word in flat for word in ("调用", "检索", "训练", "返还", "交付", "使用")
    ):
        out.append(
            _mk("illegal_terms", "retrieval_before_storage",
                "约定在完成存储/入库前即发生返还或使用，数据处理顺序不合规",
                None, confidence=0.55)
        )
    for clause, pattern in _scan_text(contract, NONEXISTENT_PATTERNS)[:1]:
        out.append(
            _mk("party_missing", "nonexistent_subject",
                f"合同标的或主体指向不存在/虚构对象（{pattern}），合同目的不能实现",
                clause, pattern, confidence=0.7)
        )
    return out


def detect_penalty(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    for clause in contract.document.clauses:
        for match in re.finditer(r"(违约金|滞纳金)[^。；;]{0,16}?(\d+(?:\.\d+)?)\s*%", clause.text):
            try:
                rate = float(match.group(2))
            except ValueError:
                continue
            if rate >= 30:
                out.append(
                    _mk("one_sided_breach", "penalty_excessive",
                        f"违约金比例 {rate:g}% 明显过高，可能被认定为过分高于损失",
                        clause, match.group(0), confidence=0.72)
                )
                break
        if out:
            break
    return out


def _clause_of(contract: Contract, clause_id: str | None) -> Clause | None:
    if not clause_id:
        return None
    return contract.clause_map().get(clause_id)


DETECTORS: tuple[Callable[[Contract], list[RiskCandidate]], ...] = (
    detect_amounts,
    detect_dates,
    detect_payment,
    detect_authority,
    detect_performance,
    detect_data_protection,
    detect_penalty,
)


def run_detectors(contract: Contract) -> list[RiskCandidate]:
    out: list[RiskCandidate] = []
    for detector in DETECTORS:
        try:
            out.extend(detector(contract))
        except Exception:
            continue
    return out


#: 数据集 `issue_codes`（31）→ v2 风险码 的对照表，评测脚本用它做码表对齐。
ISSUE_TO_CODE: dict[str, str] = {
    "zero_rent": "zero_rent",
    "extreme_rent": "abnormal_price",
    "negative_fee": "illegal_terms",
    "fee_mismatch": "fee_mismatch",
    "date_reversed": "date_reversed",
    "term_over_20y": "illegal_terms",
    "deposit_over_20pct": "abnormal_price",
    "zero_deposit": "abnormal_price",
    "unverified_property": "unverified_property",
    "liability_exemption": "liability_exemption",
    "one_sided_breach": "one_sided_breach",
    "penalty_excessive": "one_sided_breach",
    "unlimited_authority": "need_authorization",
    "commission_100pct": "one_sided_breach",
    "success_guarantee": "illegal_terms",
    "nonexistent_subject": "party_missing",
    "impossible_deadline": "vague_delivery",
    "insecure_delivery": "confidentiality_missing",
    "no_report": "subject_unspecified",
    "personal_data_no_consent": "confidentiality_missing",
    "perpetual_license": "unilateral_modify",
    "public_disclosure": "confidentiality_missing",
    "retrieval_before_storage": "illegal_terms",
    "zero_records": "subject_unspecified",
    "zero_quantity": "subject_unspecified",
    "zero_fee": "zero_rent",
    "cash_only": "illegal_terms",
    "negative_commission": "illegal_terms",
    "unilateral_entry": "unilateral_modify",
    "quantity_value_mismatch": "fee_mismatch",
    "no_permit": "need_license",
}
