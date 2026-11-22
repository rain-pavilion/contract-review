"""Read-only evidence inspection for the scoped v2 synthetic diagnostic.

This observes serialized engineering evidence, never legal accuracy. Planner
coverage, rule findings, aggregate call counts and a model's answer alone do
not establish an observed outcome. No model, network or product import occurs.

An outcome is ``observed`` only when a final risk retains one model candidate's
two exact source quotations, preserved model relationship claim, non-partial
provenance and an exact successful LegalOne call/source/receipt binding. Older
reports can therefore remain ``partial`` rather than acquiring invented links.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MODEL = "legalone-8b"
FIXTURE = "scoped_value_tail_v2"
STATUS_LABELS = {"missing": "缺失", "partial": "部分", "observed": "已观察"}
CASES = (
    {"id": "implicit_payment_value_difference", "description": "同合同尾款70000元 / 80000元",
     "sides": (("第二条", ("尾款", "70000元")), ("第七条", ("尾款", "80000元")))},
    {"id": "implicit_final_delivery_date_difference", "description": "本项目最终交付日期2026年12月1日 / 2027年1月15日",
     "sides": (("第三条", ("本项目最终交付日期", "2026年12月1日")),
               ("第八条", ("本项目最终交付日期", "2027年1月15日")))},
    {"id": "explicit_reference_cycle", "description": "第二条付款等待验收 / 第九条验收等待全部付款",
     "sides": (("第二条", ("第九条", "最终验收后支付")),
               ("第九条", ("支付全部合同价款后", "启动最终验收")))},
)
PROVENANCE_KEYS = ("candidate_id", "window_id", "job_id", "call_id", "source_sha256", "model_message", "issue_code")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _date_numbers(value: str) -> str:
    return re.sub(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?",
                  lambda m: f"{int(m[1]):04d}{int(m[2]):02d}{int(m[3]):02d}", _compact(value))


def _claim(case_id: str, message: Any) -> dict[str, Any]:
    """Conservative lexical observation of the retained model claim, not truth."""
    if not isinstance(message, str) or not message.strip():
        return {"observed": False, "reason": "preserved_model_message_missing"}
    text = _date_numbers(message)
    difference = r"不一致|冲突|矛盾|差异|不同|不明确|不确定|重复约定"
    cycle = r"循环|互为前提|相互依赖|互相依赖|相互等待|互相等待|死锁|闭环"
    relation = cycle if case_id == "explicit_reference_cycle" else difference
    # A negative answer must not be counted by the presence of the word alone.
    negative = (re.search(r"(?:未发现|不存在|没有|并无|并非|不构成|不产生|未形成).{0,10}(?:" + relation + r")", text)
                or re.search(r"不矛盾|并不冲突|无(?:明显)?冲突|可以协调|可协调", text))
    if negative:
        return {"observed": False, "reason": "negated_or_coordinated_relationship_not_counted"}
    if case_id == "implicit_payment_value_difference":
        values = (bool(re.search(r"70000|7万", text)), bool(re.search(r"80000|8万", text)))
        okay = all(values) and "尾款" in text and bool(re.search(relation, text))
    elif case_id == "implicit_final_delivery_date_difference":
        okay = all(value in text for value in ("20261201", "20270115", "交付")) and bool(re.search(relation, text))
    else:
        okay = "验收" in text and ("付款" in text or "支付" in text) and bool(re.search(relation, text))
    return {"observed": bool(okay), "reason": "model_relationship_claim_observed" if okay
            else "required_relationship_claim_not_explicit_in_preserved_model_message",
            "method": "conservative_lexical_observation_not_semantic_or_legal_gold"}


def _source_segments(call: dict[str, Any], clauses: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    spans = call.get("source_spans")
    if not isinstance(spans, list) or not spans:
        return [], ["source_spans_missing_or_invalid"]
    result = []
    for span in spans:
        if not isinstance(span, dict):
            return [], ["source_spans_missing_or_invalid"]
        cid, start, end = (span.get(key) for key in ("clause_id", "start", "end"))
        text = clauses.get(cid, {}).get("text")
        if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)):
            return [], ["source_spans_missing_or_invalid"]
        result.append({"clause_id": cid, "start": start, "end": end, "text": text[start:end]})
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result, [] if call.get("source_sha256") == digest(encoded) else ["source_digest_mismatch"]


def _calls(report: dict[str, Any]) -> list[dict[str, Any]]:
    logic = report.get("stats", {}).get("logic_llm", {})
    if not isinstance(logic, dict):
        return []
    rows = list(logic.get("calls", [])) if isinstance(logic.get("calls"), list) else []
    # latest_runs repeats historical snapshots and is intentionally not counted.
    for run in logic.get("feedback_runs", []) if isinstance(logic.get("feedback_runs"), list) else []:
        if isinstance(run, dict) and isinstance(run.get("calls"), list):
            rows.extend(run["calls"])
    return [row for row in rows if isinstance(row, dict)]


def _binding(extra: dict[str, Any], quotations: list[dict[str, Any]], prepared: dict[str, Any],
             report: dict[str, Any], clauses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    issues = []
    call_id = extra.get("call_id")
    matching = [row for row in _calls(report) if row.get("call_id") == call_id] if call_id else []
    if len(matching) != 1:
        return {"verified": False, "issues": ["call_id_missing_or_not_unique_in_recorded_logic_calls"]}
    call = matching[0]
    for key in ("window_id", "job_id", "source_sha256"):
        if call.get(key) != extra.get(key):
            issues.append("candidate_call_" + key + "_mismatch")
    if call.get("status") != "ok" or call.get("model") != MODEL:
        issues.append("call_not_complete_expected_model")
    source, source_issues = _source_segments(call, clauses)
    issues.extend(source_issues)
    for quote in quotations:
        if not any(row["clause_id"] == quote["clause_id"] and quote["quote"] in row["text"] for row in source):
            issues.append("model_quote_not_in_exact_sent_source:" + str(quote["clause_id"]))
    events = report.get("stats", {}).get("model_calls")
    selected = [row for row in events if isinstance(row, dict) and row.get("component") == "llm"
                and row.get("operation") == "request_json" and isinstance(row.get("details"), dict)
                and row["details"].get("call_id") == call_id] if isinstance(events, list) else []
    starts = [row for row in selected if row.get("status") == "started"]
    terminal = [row for row in selected if row.get("status") != "started"]
    if len(starts) != 1 or len(terminal) != 1:
        issues.append("exact_one_start_and_terminal_receipt_missing")
    else:
        start, stop = starts[0], terminal[0]
        sd, td = start["details"], stop["details"]
        for event in (start, stop):
            details = event["details"]
            if (event.get("model") != MODEL or details.get("requested_model") != MODEL
                    or event.get("trace_id") != report.get("trace_id") or not report.get("trace_id")
                    or event.get("contract_id") != prepared.get("contract_id")):
                issues.append("receipt_model_trace_or_contract_binding_invalid")
            if any(details.get(key) != call.get(key) for key in ("window_id", "job_id", "source_sha256")):
                issues.append("receipt_call_source_binding_mismatch")
        if (stop.get("status") != "succeeded" or td.get("model_identity_verified") is not True
                or td.get("actual_model") != MODEL or not isinstance(td.get("finish_reason"), str)
                or not td["finish_reason"] or td.get("finish_reason") in ("length", "time_limit")
                or stop.get("reason")):
            issues.append("receipt_not_verified_nontruncated_success")
        if (not isinstance(start.get("event_id"), str) or not start["event_id"]
                or not isinstance(stop.get("event_id"), str) or not stop["event_id"]
                or start["event_id"] == stop["event_id"]):
            issues.append("receipt_event_ids_missing_or_duplicate")
        try:
            if datetime.fromisoformat(start["created_at"]) > datetime.fromisoformat(stop["created_at"]):
                issues.append("receipt_time_order_invalid")
        except (KeyError, TypeError, ValueError):
            issues.append("receipt_time_order_missing_or_invalid")
    return {"verified": not issues, "call_id": call_id, "window_id": call.get("window_id"),
            "job_id": call.get("job_id"), "source_sha256": call.get("source_sha256"),
            "issues": sorted(set(issues)), "scope": "serialized_receipt_binding_not_independent_runtime_replay"}


def _quotation(evidence: dict[str, Any], clauses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    extra = evidence.get("extra") if isinstance(evidence.get("extra"), dict) else {}
    span = evidence.get("span") if isinstance(evidence.get("span"), dict) else {}
    cid, quote = extra.get("clause_id"), extra.get("quote")
    full = isinstance(cid, str) and isinstance(quote, str) and bool(quote)
    if not full:
        cid, quote = span.get("clause_id"), span.get("quote")
        if evidence.get("kind") == "cross_clause_quote" and isinstance(evidence.get("text"), str):
            prefix, separator, suffix = evidence["text"].partition(": ")
            if separator and prefix in clauses:
                cid, quote = prefix, suffix
    source = clauses.get(cid, {}).get("text")
    exact = isinstance(source, str) and isinstance(quote, str) and bool(quote) and quote in source
    return {"clause_id": cid, "quote": quote if isinstance(quote, str) else "", "exact_source_match": bool(exact),
            "full_quote_provenance_available": full, "extra": extra}


def _bound_quotations(evidence: dict[str, Any], clauses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Read retained model-owned pair bundles, never supplement from rules."""
    own = _quotation(evidence, clauses)
    extra = own["extra"]
    package = extra.get("bound_model_quotes")
    if not isinstance(package, list):
        return [own]
    result = []
    package_valid = bool(package)
    for row in package:
        if (not isinstance(row, dict) or not isinstance(row.get("clause_id"), str)
                or not isinstance(row.get("quote"), str) or not row["quote"]
                or type(row.get("quote_exact_source_match")) is not bool):
            package_valid = False
            continue
        inherited = {**extra, **{key: row[key] for key in ("clause_id", "quote", "quote_exact_source_match")}}
        result.append(_quotation({"extra": inherited}, clauses))
    own_listed = any(row["clause_id"] == own["clause_id"] and row["quote"] == own["quote"]
                     and row["extra"].get("quote_exact_source_match") is extra.get("quote_exact_source_match")
                     for row in result)
    for row in result or [own]:
        row["package_valid"] = package_valid and own_listed
    return result or [own]


def _risk_span_matches(risk: dict[str, Any], quotation: dict[str, Any], clauses: dict[str, dict[str, Any]]) -> bool:
    cid, quote = quotation["clause_id"], quotation["quote"]
    for span in risk.get("spans", []) if isinstance(risk.get("spans"), list) else []:
        if not isinstance(span, dict) or span.get("clause_id") != cid:
            continue
        start, end = span.get("char_start"), span.get("char_end")
        text = clauses.get(cid, {}).get("text", "")
        if (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                and text[start:end] == quote and span.get("quote") == quote[:80]
                and span.get("block_id") in clauses[cid].get("block_ids", [])):
            return True
    return False


def inspect(prepared: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    clauses_list = prepared.get("clauses") if isinstance(prepared.get("clauses"), list) else []
    clauses = {row.get("clause_id"): row for row in clauses_list if isinstance(row, dict)
               and isinstance(row.get("clause_id"), str) and row["clause_id"]}
    global_issues = []
    if len(clauses) != len(clauses_list) or not clauses:
        global_issues.append("prepared_clause_ids_missing_or_duplicate")
    if prepared.get("source_manifest", {}).get("fixture_version") != FIXTURE:
        global_issues.append("not_declared_scoped_value_tail_v2_fixture")
    parsed = prepared.get("parsed_text")
    if not isinstance(parsed, str) or prepared.get("parsed_text_sha256") != digest(parsed):
        global_issues.append("prepared_parsed_text_hash_invalid")
    if report is None:
        global_issues.append("report_missing")
    elif (report.get("contract_id") != prepared.get("contract_id") or not prepared.get("contract_id")
          or report.get("document", {}).get("full_text") != parsed):
        global_issues.append("report_source_does_not_match_prepared")
    risks = report.get("risks", []) if isinstance(report, dict) and isinstance(report.get("risks"), list) else []
    result = {"inspection_status": "report_missing" if report is None else "inspected",
              "scope": "three_synthetic_engineering_relationship_observations_only",
              "legal_accuracy_evaluated": False, "independent_human_gold_used": False,
              "planner_or_rule_coverage_counted_as_model_detection": False,
              "source_issues": global_issues, "outcomes": [],
              "limitations": ["Observed is an evidence-chain observation, not a legal correctness verdict.",
                              "Missing/partial does not prove that the model failed internally; retained evidence may be insufficient.",
                              "Receipt checks use serialized report data; use the separate postrun verifier for SQLite reconciliation.",
                              "Span offsets are checked against clause text; this does not certify block/page layout coordinates."]}
    for case in CASES:
        expected, source_issues = {}, []
        for label, terms in case["sides"]:
            rows = [row for row in clauses.values() if row.get("label") == label]
            if len(rows) != 1 or not all(term in _compact(rows[0].get("text", "")) for term in terms):
                source_issues.append("expected_clause_scope_missing_or_ambiguous:" + label)
            else:
                expected[rows[0]["clause_id"]] = {"label": label, "required_quote_terms": list(terms)}
        assessments = []
        rule_only = 0
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            model_rows = [row for row in risk.get("evidence", []) if isinstance(row, dict)
                          and row.get("source") == "node2_llm"] if isinstance(risk.get("evidence"), list) else []
            if not model_rows and _claim(case["id"], risk.get("message"))["observed"]:
                rule_only += 1
            groups = {}
            for index, evidence in enumerate(model_rows):
                quote = _quotation(evidence, clauses)
                extra = quote["extra"]
                key = (extra.get("candidate_id"), extra.get("call_id"))
                if not all(isinstance(value, str) and value for value in key):
                    key = ("legacy_unbound", "legacy_unbound")
                groups.setdefault(key, []).extend(_bound_quotations(evidence, clauses))
            for key, quotations in groups.items():
                relevant = [quote for quote in quotations if quote["clause_id"] in expected
                            and any(term in _compact(quote["quote"]) for term in expected[quote["clause_id"]]["required_quote_terms"])]
                if not relevant:
                    continue
                issues = list(source_issues) + list(global_issues)
                extras = [quote["extra"] for quote in quotations]
                first = extras[0]
                if any(not isinstance(first.get(field), str) or not first[field] for field in PROVENANCE_KEYS):
                    issues.append("model_candidate_provenance_or_preserved_claim_missing")
                if any(any(extra.get(field) != first.get(field) for field in PROVENANCE_KEYS) for extra in extras):
                    issues.append("model_candidate_provenance_inconsistent")
                packages = [extra.get("bound_model_quotes") for extra in extras if "bound_model_quotes" in extra]
                if packages and any(package != packages[0] for package in packages):
                    issues.append("bound_model_quote_packages_inconsistent")
                if any(quote.get("package_valid") is False for quote in quotations):
                    issues.append("bound_model_quote_package_invalid_or_missing_own_quote")
                if any(extra.get("partial_model_output") is not False for extra in extras):
                    issues.append("model_output_partial_or_completion_unknown")
                sides = []
                display_issues = []
                for cid, expectation in expected.items():
                    rows = [quote for quote in quotations if quote["clause_id"] == cid]
                    good = [quote for quote in rows if quote["full_quote_provenance_available"]
                            and quote["extra"].get("quote_exact_source_match") is True
                            and quote["exact_source_match"]
                            and all(term in _compact(quote["quote"]) for term in expectation["required_quote_terms"])]
                    source_verified = bool(good)
                    displayed = bool(good) and any(_risk_span_matches(risk, quote, clauses) for quote in good)
                    if not source_verified:
                        issues.append("exact_scoped_final_model_quote_missing:" + expectation["label"])
                    if not displayed or cid not in risk.get("clause_ids", []):
                        display_issues.append("legacy_final_span_or_clause_id_not_exact:" + expectation["label"])
                    sides.append({"clause_id": cid, "label": expectation["label"], "verified": source_verified,
                                  "final_display_span_exact": displayed, "final_clause_id_listed": cid in risk.get("clause_ids", []),
                                  "quotations": [{name: quote[name] for name in ("quote", "exact_source_match", "full_quote_provenance_available")}
                                                 for quote in rows]})
                claim = _claim(case["id"], first.get("model_message"))
                if not claim["observed"]:
                    issues.append(claim["reason"])
                binding = _binding(first, quotations, prepared, report, clauses) if report is not None else {"verified": False, "issues": ["report_missing"]}
                issues.extend(binding["issues"])
                assessments.append({"risk_id": risk.get("risk_id"), "candidate_id": first.get("candidate_id"),
                                    "call_id": first.get("call_id"), "model_message": first.get("model_message"),
                                    "issue_code": first.get("issue_code"), "status": "observed" if not issues and len(sides) == 2 else "partial",
                                    "sides": sides, "model_claim": claim, "call_binding": binding,
                                    "final_display_location_issues": display_issues,
                                    "issues": sorted(set(issues))})
        status = "observed" if any(row["status"] == "observed" for row in assessments) else "partial" if assessments else "missing"
        result["outcomes"].append({"id": case["id"], "description": case["description"], "status": status,
                                   "status_zh": STATUS_LABELS[status], "source_issues": source_issues,
                                   "final_model_evidence_assessments": assessments,
                                   "non_model_final_claims_not_counted": rule_only,
                                   "absence_reason": None if assessments else "no_retained_final_model_evidence_for_target_relationship"})
    result["counts"] = {status: sum(row["status"] == status for row in result["outcomes"]) for status in STATUS_LABELS}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="May be absent while the actual run is in progress")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.is_file() else None
    if not isinstance(prepared, dict) or (report is not None and not isinstance(report, dict)):
        parser.error("prepared and report must be JSON objects")
    result = inspect(prepared, report)
    result["files"] = {"prepared": str(args.prepared.resolve()), "prepared_sha256": digest(args.prepared.read_bytes()),
                       "report": str(args.report.resolve()), "report_sha256": digest(args.report.read_bytes()) if report is not None else None,
                       "inspector_sha256": digest(Path(__file__).read_bytes())}
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)
    return 2 if report is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
