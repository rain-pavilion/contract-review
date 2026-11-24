"""Source-bound screening receipts, independent of risk scoring or legal clearance.

Only direct payment deadline language is covered. A negative receipt means this
small textual check found no issue; it never certifies the whole payment clause.
"""
from __future__ import annotations

import re
from datetime import date

from .models import stable_id

_PAYMENT = re.compile(r"支付|付款|付清|付讫|结清|结算|尾款|首款|预付款|进度款|货款|租金|价款")
_ACTION = re.compile(r"支付|付款|付清|付讫|结清|结算")
_VAGUE = re.compile(r"另行(?:协商|约定|确定)|待定|适时|尽快|及时|资金(?:到位|允许)|视.{0,8}情况|酌情|条件成熟")
_REFERENCE = re.compile(r"附件|附表|付款计划|参照|另见|第[一二三四五六七八九十百\d]+条|按.{0,16}(?:约定|执行)")
_UNKNOWN = re.compile(r"_{2,}|[□☐]|[XxＸｘ]{2,}|\[.{0,8}(?:填写|日期|天数).{0,8}\]")
_HEADING = re.compile(r"^(?:第[一二三四五六七八九十百\d]+条|[（(]?[一二三四五六七八九十\d]+[）).、])?\s*"
    r"(?:价款及付款|合同价款|付款(?:方式|期限|安排|条件|计划)?|支付(?:方式|期限|安排|条件)?|费用及支付)\s*[:：]?$")
_NEGATIVE = re.compile(r"不予|不再|无需|无须|不得|拒绝|暂停|暂缓|不支付|不付款|不确定|并非|不是")
_DIGIT_CN = r"[一二三四五六七八九]"
_NUMBER = (r"(?:[1-9]\d{0,3}|[一二三四五六七八九两]|" + _DIGIT_CN + r"?十" +
           _DIGIT_CN + r"?|[一二三四五六七八九两]百(?:零" + _DIGIT_CN + r"|" +
           _DIGIT_CN + r"?十" + _DIGIT_CN + r"?)?)")
_RELATIVE = re.compile(
    r"(?P<trigger>签订|签署|生效|验收合格|验收通过|验收|交付|交货|收到发票|收到货物|收到货款|收到款项|开具发票|结算确认)"
    r"(?:之日)?(?:起|之?后|次日起|日起)\s*" + _NUMBER +
    r"\s*(?:个)?(?:工作日|日|天|个月|月)内\s*(?:向[甲乙丙]方)?(?:一次性|全额|足额|全部|予以)?\s*$")
_ABSOLUTE = re.compile(r"(?:于|在|不迟于|最迟(?:于)?)\s*(\d{4})年(\d{1,2})月(\d{1,2})日(?:前|之前|以内|内)?\s*$")


def _direct_deadline(sentence: str) -> bool:
    """Every payment action in an obligation needs its own nearby bound date."""
    actions = list(_ACTION.finditer(sentence))
    if not actions or _NEGATIVE.search(sentence) or _UNKNOWN.search(sentence):
        return False
    for action in actions:
        before = re.split(r"[，,；;]", sentence[:action.start()])[-1]
        # A comma directly before the payment action can separate a trigger from
        # its deadline, so leave such forms for manual reading rather than guess.
        if _RELATIVE.search(before):
            continue
        match = _ABSOLUTE.search(before)
        if match:
            try:
                date(*(int(x) for x in match.groups()))
            except ValueError:
                return False
            continue
        return False
    return True


def evaluate_review_points(contract, config=None) -> list[dict]:
    """Return one payment-point receipt without altering contract or score."""
    document = contract.document
    sources = [(c.clause_id, c.text or c.body) for c in document.clauses]
    if not sources:
        sources = [(None, document.full_text)]
    evidence = []
    sentences = []
    source_mismatch = False
    unverified_ocr = False
    clause_map = document.clause_map()
    # Check the reverse direction as well: segmentation must not silently drop
    # another payment obligation from the full supplied text.
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", document.full_text):
        quote = match.group().strip()
        if quote and _PAYMENT.search(quote) and not _HEADING.fullmatch(quote) and not any(quote in source for _, source in sources):
            source_mismatch = True
    for clause_id, text in sources:
        clause = clause_map.get(clause_id)
        quality = (clause.metadata.get("extraction_quality") or {}) if clause else {}
        if (isinstance(quality, dict) and quality.get("status") == "machine_ocr_unverified"
                and quality.get("human_verified") is not True and _PAYMENT.search(text)):
            unverified_ocr = True
        for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
            quote = match.group().strip()
            if quote and _PAYMENT.search(quote) and not _HEADING.fullmatch(quote):
                # Never trust stale clause text or a caller-supplied located flag.
                if quote not in document.full_text:
                    source_mismatch = True
                    continue
                evidence.append({"clause_id": clause_id, "quote": quote, "located": True})
                sentences.append(quote)
    receipt = {
        "check_id": stable_id("check", contract.contract_id, "payment-deadline-v1"),
        "point_id": "commercial.payment_terms", "status": "not_checked",
        "execution_status": "completed", "scope_complete": False,
        "scope": "provided_contract_text_payment_sentences_only",
        "clause_ids": list(dict.fromkeys(e["clause_id"] for e in evidence if e["clause_id"])),
        "evidence": evidence, "actual_issue": "", "impact": "", "recommendation": "",
        "severity": "medium", "manual_required": True, "pending_reasons": [],
        "rule_version": "payment-deadline-v1", "legal_truth_established": False,
    }
    if not sentences and not source_mismatch:
        receipt["pending_reasons"] = ["未定位到可执行付款期限检查的原文，不能据此判断付款安排是否完整。"]
        return [receipt]
    reasons = []
    if document.warnings or document.stats.get("needs_ocr_pages") or unverified_ocr:
        reasons.append("解析/OCR存在提示，须核对完整原文后确认付款安排。")
    if source_mismatch:
        reasons.append("付款条款与所给全文不一致，无法确认该范围原文完整性。")
    if any(_REFERENCE.search(s) for s in sentences):
        reasons.append("付款安排引用其他条款或附件，当前直接期限规则未完成跨文档核对。")
    if any(_UNKNOWN.search(s) for s in sentences):
        reasons.append("付款条文含待填写内容，须补充确定日期或期限。")
    vague = any(_VAGUE.search(s) for s in sentences)
    # A payment heading/mention alone cannot establish an obligation or a pass.
    unclear = any(not _direct_deadline(s) for s in sentences)
    if vague:
        reasons.append("付款时间采用另行协商或不确定表述，尚无明确、可确定的付款期限。")
    elif unclear:
        reasons.append("已提供的付款句中存在未能直接绑定确定期限的付款安排，需人工核对。")
    if reasons:
        receipt.update(status="issue_found" if vague else "needs_manual",
            actual_issue="付款期限存在不明确表述。" if vague else "付款期限的明确性尚待核实。",
            impact="可能导致付款到期时间及逾期责任难以确定，影响履约和收款安排。",
            recommendation="补充明确、可确定的付款期限，并核对每笔首款、进度款和尾款及其触发条件。",
            pending_reasons=reasons)
    else:
        receipt.update(status="no_issue_detected", scope_complete=True, manual_required=False,
            recommendation="已提供付款句可直接定位期限；仍需结合完整合同及附件复核触发条件和其他付款问题。")
    return [receipt]
