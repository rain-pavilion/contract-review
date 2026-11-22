"""Startup-only model initialization; health reads never load a neural model."""
from __future__ import annotations

import time
from typing import Any

from . import entailment, model_audit

_STATUS: dict[str, Any] = {"initialized": False, "requested_backend": "rules", "active_backend": "rules"}


def initialize_backend(config: Any, *, force: bool = False) -> dict[str, Any]:
    choice = str(getattr(config, "entailment_backend", "auto") or "auto")
    if force:
        # Explicit maintenance action only; health never invokes this path.
        with entailment._BERT_LOCK:
            entailment._BERT.update(loaded=False, predictor=None, error="", reason="", load_seconds=0.0)
            from . import bert_entail
            bert_entail._PREDICTOR.clear()
    entailment.set_backend(choice)
    started = time.perf_counter()
    predictor = entailment._bert_predictor() if choice in ("auto", "bert") else None
    state = entailment.runtime_status()
    _STATUS.update({"initialized": True, "requested_backend": choice,
                    "initialization_seconds": round(time.perf_counter() - started, 3), **state})
    return runtime_status()


def runtime_status() -> dict[str, Any]:
    return {**_STATUS, **entailment.runtime_status(), "audit": model_audit.status()}
