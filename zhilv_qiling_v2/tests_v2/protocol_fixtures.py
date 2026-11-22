"""Explicit SYNTHETIC/non-gold protocol evidence for isolated unit tests only.

These helpers never invoke a transport, model, or audit database. Their receipts
test binding contracts; they are not evidence that a real model was executed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


def synthetic_pair(content, *, audit_context, index, model="fixture"):
    call_id = f"synthetic-non-gold-call-{index}"
    receipt = {**dict(audit_context), "call_id": call_id,
        "transport_attempt": 1, "status": "succeeded", "reason": "",
        "synthetic_non_gold_fixture": True}
    return SimpleNamespace(content=content, model=model, elapsed=0.01,
        raw=json.dumps(content, ensure_ascii=False), call_id=call_id,
        request_receipts=[receipt], synthetic_non_gold_fixture=True)


def synthetic_feedback_calls(items, *, index):
    calls = []
    for slot, item in enumerate(items):
        call_id = f"synthetic-non-gold-feedback-{index}-{slot}"
        window_id = f"synthetic-window-{index}-{slot}"
        job_id = window_id + ":batch:1"
        source = "a" * 64
        row = {key: item[key] for key in ("feedback_item_id", "candidate_id", "article_uid")}
        row.update(verdict="confirmed", reason="Synthetic protocol fixture, not legal gold.",
                   clause_quote="合成原文片段", article_quote=item.get("article_quote", ""))
        calls.append({"call_id": call_id, "window_id": window_id, "job_id": job_id,
            "source_sha256": source, "status": "ok", "feedback_item_ids": [item["feedback_item_id"]],
            "verifications": [row], "verification_errors": [],
            "verified_feedback_item_ids": [item["feedback_item_id"]],
            "request_receipts": [{"call_id": call_id, "window_id": window_id,
                "job_id": job_id, "source_sha256": source, "status": "succeeded",
                "synthetic_non_gold_fixture": True}]})
    return calls
