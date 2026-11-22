"""Text-grounded arbitration for rule/model entities; not a legal classifier."""
from __future__ import annotations

from datetime import date
import math
import re

VERSION = "ner-context-constraints-v2"
GENERIC_ROLES = {"", "other", "unknown", "model"}


def is_role_definition_fragment(value):
    """A definition sentence (even a truncated OCR span) is not a party name."""
    clean = re.sub(r"\s+", "", value or "").lstrip("_＿")
    roles = r"(?:甲方|乙方|丙方|丁方|中方|采购人|供应商|委托方|受托方|买方|卖方|出租人|承租人|中标人)"
    return bool(re.match(r"(?:本合同(?:中(?:的)?)?)?" + roles +
                         r"(?:系指|是指|指的是|定义为)", clean))


def constrain_hits(document, hits):
    blocks = {block.block_id: block for block in document.blocks}
    accepted, rejected = [], []
    for hit in hits:
        reasons = []
        adjustments = []
        span = hit.span
        block = blocks.get(span.block_id) if span else None
        if block is None:
            reasons.append("source_block_missing")
        else:
            text = block.text
            start, end = span.char_start, span.char_end
            if start < 0 or end < start:
                # Legacy adapters may only provide a surface. Bind only unique
                # occurrences; never silently select the first repeated value.
                positions = [m.start() for m in re.finditer(re.escape(hit.value), text)] if hit.value else []
                if len(positions) == 1:
                    start, end = positions[0], positions[0] + len(hit.value)
                    span.char_start, span.char_end, span.quote = start, end, hit.value
                else:
                    reasons.append("source_span_ambiguous_or_missing")
            if not reasons:
                raw = text[start:end]
                if not (0 <= start < end <= len(text)) or raw.strip() != hit.value.strip():
                    reasons.append("source_span_surface_mismatch")
                else:
                    # Trim by source offsets, not just by entity surface.
                    start += len(raw) - len(raw.lstrip())
                    end -= len(raw) - len(raw.rstrip())
                    span.char_start, span.char_end = start, end
                    span.quote = text[start:end]
                    hit.value = span.quote
                    context = text[max(0, start - 16):min(len(text), end + 8)]
                    if hit.kind == "amount":
                        numbers = re.findall(r"\d+(?:[,，]\d{3})*(?:\.\d+)?", hit.value)
                        if len(numbers) > 1:
                            # Quantity + unit price is a span boundary error,
                            # not a single amount; keep separately grounded hits.
                            reasons.append("multiple_numeric_values_in_amount_span")
                        unit = bool(re.search(r"元|圆|RMB|人民币|[￥¥]", hit.value, re.I))
                        currency_context = bool(re.search(r"金额|价款|总价|单价|尾款|进度款|预付款|定金|保证金|人民币|[￥¥]", context))
                        if text[end:end + 1] in {"%", "％", "‰"}:
                            reasons.append("percentage_not_amount")
                        if not unit and (re.search(r"账号|帐号|账户|信用代码|电话|证号", context) or not currency_context):
                            reasons.append("numeric_identifier_or_missing_currency_context")
                        try:
                            if not math.isfinite(float(hit.normalized)):
                                reasons.append("non_finite_amount")
                        except (ValueError, TypeError):
                            reasons.append("invalid_normalized_amount")
                        prefix = text[max(0, start - 24):start]
                        if not reasons and re.search(
                                r"(?:增加|减少|新增|追加|调增|调减)(?:金额|价款)?(?:为|了)?[\s:：]*(?:人民币|RMB|[￥¥])?\s*$", prefix, re.I):
                            if hit.role != "adjustment":
                                adjustments.append({"field": "role", "from": hit.role,
                                    "to": "adjustment", "basis": "explicit_local_amount_change", "quote": prefix})
                                hit.role = "adjustment"
                        quality = block.metadata.get("extraction_quality", {})
                        if quality.get("status") == "machine_ocr_unverified" and re.search(r"[壹贰叁肆伍陆柒捌玖拾佰仟]", hit.value):
                            hit.extra = dict(hit.extra or {}, numeric_comparison_eligible=False,
                                manual_required=True, numeric_comparison_reason="unverified_ocr_chinese_amount_may_be_fragment")
                    elif hit.kind == "party":
                        if is_role_definition_fragment(hit.value):
                            reasons.append("role_definition_fragment_not_entity")
                    elif hit.kind == "date":
                        try:
                            date.fromisoformat(hit.normalized)
                        except (ValueError, TypeError):
                            reasons.append("invalid_calendar_date")
        receipt = {"version": VERSION, "accepted": not reasons, "reasons": reasons,
                   "adjustments": adjustments,
                   "legal_truth_established": False}
        hit.extra = dict(hit.extra or {}, constraint_audit=receipt)
        if reasons:
            rejected.append({"kind": hit.kind, "value": hit.value, "source": getattr(hit, "source", "rule"),
                             "block_id": span.block_id if span else None,
                             "span": span.to_dict() if span else None, **receipt})
        else:
            accepted.append(hit)
    return accepted, rejected


def location_key(hit):
    span = hit.span
    return (span.block_id, span.char_start, span.char_end) if span else (None, None, None)


def compatible_roles(left, right):
    return left == right or left in GENERIC_ROLES or right in GENERIC_ROLES
