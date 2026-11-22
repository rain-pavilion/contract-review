"""Read-only historical case summaries, isolated from review and training data."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

INDEX_PATH = Path(__file__).resolve().parent / "data" / "public_case_references.json"
SCHEMA_VERSION = "public-case-reference-v1"
USAGE = {"historical_reference_only": True, "automatic_training_allowed": False,
         "independent_human_gold": False, "subject_current_facts_verified": False,
         "public_fulltext_republication_allowed": False}
PUBLIC_FIELDS = ("case_id", "case_name", "guiding_case_number", "original_published_date",
                 "web_publication_date", "judgment_dates", "trial_courts", "case_numbers",
                 "focus", "summary", "source_url", "source_publisher")


def case_references(query: str = "") -> dict[str, Any]:
    """Load packaged metadata only. No requests, writes, imports into facts or models."""
    needle = query.strip().casefold()[:160]
    base: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "query": needle,
                           "usage": dict(USAGE), "scope": "历史指导案例简要参考，非本合同结论或主体当前事实"}
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        rows = data["items"]
        if not isinstance(rows, list):
            raise ValueError("Invalid reference list")
        items = []
        for row in rows:
            url = urlparse(row["source_url"])
            if (url.scheme != "https" or url.hostname != "www.chinacourt.cn"
                    or url.username is not None or url.password is not None
                    or not isinstance(row.get("summary"), str) or len(row["summary"]) > 300
                    or type(row.get("guiding_case_number")) is not int):
                raise ValueError("Invalid published reference")
            item = {key: row[key] for key in PUBLIC_FIELDS}
            item["usage"] = dict(USAGE)
            items.append(item)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return {**base, "available": False, "items": [], "total": 0, "count": 0,
                "note": "公开案例参考索引暂不可用"}
    matched = [row for row in items if not needle or needle in " ".join([
        row["case_name"], f"指导案例{row['guiding_case_number']}号", row["summary"],
        *row["focus"], *row["case_numbers"], *row["trial_courts"],
    ]).casefold()]
    return {**base, "available": True, "items": matched, "total": len(items), "count": len(matched)}
