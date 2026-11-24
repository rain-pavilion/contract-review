"""Validated, request-scoped review perspective without identity assertions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace


CONTRACT_TYPES = ("通用", "房屋租赁", "商品房预售", "房屋买卖", "委托", "中介",
                  "保管", "数据提供", "技术开发", "政府采购", "承揽")
POSITION_LABELS = {"neutral": "中立审查", "party_a": "甲方立场", "party_b": "乙方立场"}
REVIEW_OPTION_KEYS = ("contract_type", "review_position")


def normalize_review_options(options: dict) -> dict:
    """Missing fields default; malformed or unknown selections fail closed."""
    if not isinstance(options, dict):
        raise ValueError("审查选项须为对象")
    contract_type = options.get("contract_type", "auto")
    position = options.get("review_position", "neutral")
    if not isinstance(contract_type, str) or contract_type not in ("auto", *CONTRACT_TYPES):
        raise ValueError("contract_type须为auto或支持的合同类型")
    if not isinstance(position, str) or position not in POSITION_LABELS:
        raise ValueError("review_position须为neutral、party_a或party_b")
    return {"contract_type": contract_type, "review_position": position}


def apply_review_options(config, options: dict):
    """Clone a config so concurrent reviews cannot change each other's stance."""
    selected = normalize_review_options(options)
    return replace(config, review_contract_type=selected["contract_type"],
                   review_position=selected["review_position"])


def build_review_context(contract_type: str, config, inferred_type: str | None = None) -> dict:
    """Bind the effective type and requested perspective to a stable digest."""
    selected = normalize_review_options({
        "contract_type": getattr(config, "review_contract_type", "auto"),
        "review_position": getattr(config, "review_position", "neutral")})
    if contract_type not in CONTRACT_TYPES or (inferred_type is not None and inferred_type not in CONTRACT_TYPES):
        raise ValueError("审查上下文的合同类型无效")
    requested_type = selected["contract_type"]
    context = {"contract_type": contract_type,
        "detected_contract_type": inferred_type if inferred_type is not None else contract_type,
        "contract_type_source": "auto" if requested_type == "auto" else "user",
        "review_position": selected["review_position"],
        "position_label": POSITION_LABELS[selected["review_position"]],
        "requires_confirmation": requested_type == "auto" or selected["review_position"] == "neutral",
        "scope": "user_selected_review_perspective_not_verified_identity"}
    context["context_fingerprint"] = hashlib.sha256(json.dumps(context, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return context
