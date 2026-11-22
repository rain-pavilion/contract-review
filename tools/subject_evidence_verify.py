"""Capture fixed public sources and verify subject evidence without certifying facts.

Only the built-in, explicitly named HTTPS sources can be fetched. Local bundle
verification does no network access. A successful verification proves capture
integrity and text/identity binding, not an enterprise's present legal status.
Imported records stay ``unknown`` until a separate qualified review supplies a
proper fact attestation through the existing product import entry.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2 import subject_facts as F

MAX_BYTES = 4_000_000
SOURCES = {
    "csrc_cms_2018": {
        "url": "https://www.csrc.gov.cn/shenzhen/c104305/c1583762/content.shtml",
        "publisher": "中国证券监督管理委员会深圳监管局",
        "source_role": "regulator_historical_approval",
        "subject_name": "招商证券股份有限公司",
        "subject_id": "91440300192238549B",
        "fact_type": "license",
        "document_date": "2018-06-01",
        "date_tokens": ["2018年6月1日", "2018年06月01日", "2018-06-01"],
        "required_text": ["设立16家分支机构", "自批复下发之日起6个月内"],
        "claim": "该监管批复载明同名同代码主体获准在所列城市设立16家分支机构，并规定后续办理事项；不能据此推定2026年整体证券业务资格或信用状况。",
    },
    "szcredit_cmb_public_report": {
        "url": "https://qr.szcredit.org.cn/GJQYCredit/GSZJGSPTS/Print.aspx?rid=6B553DC2860F51DDBBCFD2C85B6EADDED3F7C21878B1183F71C0815934AAD415",
        "publisher": "深圳市公共信用中心·深圳信用网",
        "source_role": "municipal_public_credit_report_not_registry_live_query",
        "subject_name": "招商银行股份有限公司",
        "subject_id": "9144030010001686XA",
        "fact_type": "registration",
        "date_extractor": "report_produced_at_local_time",
        "required_text": ["登记状态", "存续", "仅供参考"],
        "claim": "该公开报告在其报告时点列示登记状态为存续；报告自身声明仅供参考，应以相应公示系统查询页为准，不推定持续有效或无不良信用。",
    },
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def beijing_today() -> date:
    return datetime.now(timezone(timedelta(hours=8))).date()


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        elif tag in ("p", "div", "tr", "td", "th", "br", "li") and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)
        elif tag in ("p", "div", "tr", "td", "th", "li") and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def decode_html(raw: bytes) -> tuple[str, dict]:
    # Some official legacy pages declare UTF-8 but serve GBK bytes. Every actual
    # decode is strict; metadata declaration errors are recorded, never ignored.
    charset = re.search(br"charset\s*=\s*[\"']?([A-Za-z0-9_-]+)", raw[:4096], re.I)
    declared = charset.group(1).decode("ascii").lower() if charset else ""
    aliases = {"utf-8": "utf-8-sig", "utf8": "utf-8-sig", "gb2312": "gb18030",
               "gbk": "gb18030", "gb18030": "gb18030"}
    preferred = aliases.get(declared)
    candidates = list(dict.fromkeys(([preferred] if preferred else []) + ["utf-8-sig", "gb18030"]))
    failures = []
    for encoding in candidates:
        try:
            decoded = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            failures.append({"encoding": encoding, "start": exc.start, "end": exc.end, "reason": exc.reason})
            continue
        return decoded, {"declared_encoding": declared or None, "actual_decoding": encoding,
                         "declared_encoding_conflict": bool(declared and preferred != encoding),
                         "strict_decode_failures": failures, "replacement_or_ignored_bytes": False}
    raise ValueError("No supported strict HTML decoding succeeded; preserve raw evidence for manual review")


def extract_text_with_metadata(raw: bytes) -> tuple[str, dict]:
    decoded, metadata = decode_html(raw)
    parser = VisibleText()
    parser.feed(decoded)
    text = "\n".join(re.sub(r"[ \t\u3000]+", " ", line).strip()
                     for line in "".join(parser.parts).splitlines() if line.strip())
    return text, metadata


def extract_text(raw: bytes) -> str:
    return extract_text_with_metadata(raw)[0]


def source_date_metadata(cfg: dict, text: str) -> dict:
    if cfg.get("date_extractor") == "report_produced_at_local_time":
        match = re.search(r"报告生产时间\s*(\d{4}/\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2}:\d{2})", text)
        if not match:
            return {"document_date": None, "supported": False, "reason": "report_timestamp_field_missing"}
        raw_time = match.group(1) + " " + match.group(2)
        try:
            stamp = datetime.strptime(raw_time, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            return {"document_date": None, "supported": False, "reason": "invalid_report_timestamp"}
        return {"document_date": stamp.date().isoformat(), "supported": True,
                "reported_time_text": raw_time, "source_field": "报告生产时间",
                "time_zone_assumption": "Asia/Shanghai; source timestamp supplies no UTC offset"}
    supported = any(compact(token) in compact(text) for token in cfg["date_tokens"])
    return {"document_date": cfg["document_date"], "supported": supported,
            "source_field": "fixed historical decision date in source text"}


def valid_uscc(identifier: str) -> bool:
    alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    weights = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)
    if len(identifier) != 18 or any(c not in alphabet for c in identifier):
        return False
    check = (31 - sum(alphabet.index(c) * weight for c, weight in zip(identifier[:17], weights)) % 31) % 31
    return identifier[-1] == alphabet[check]


def source_config(key: str) -> dict:
    if key not in SOURCES:
        raise ValueError("Only named, fixed public sources are accepted; arbitrary URLs are not fetched")
    return SOURCES[key]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture(key: str, output: Path) -> Path:
    cfg = source_config(key)
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh directory; existing evidence is never overwritten")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl with HTTPS support is required for capture; offline verification remains available")
    output.mkdir(parents=True)
    raw_path = output / "source.html"
    # No shell interpolation, arbitrary URL, proxy supplied by the bundle, or redirect.
    command = [curl, "--fail", "--silent", "--show-error", "--proto", "=https",
               "--max-redirs", "0", "--max-time", "45", "--max-filesize", str(MAX_BYTES),
               "--output", str(raw_path), "--write-out", "%{http_code}\n%{url_effective}", cfg["url"]]
    run = subprocess.run(command, capture_output=True, text=True, timeout=55)
    if run.returncode:
        raise RuntimeError(f"Official capture failed (curl {run.returncode}): {run.stderr[-500:]}")
    lines = run.stdout.strip().splitlines()
    if len(lines) != 2 or lines[0] != "200" or lines[1] != cfg["url"]:
        raise ValueError("Unexpected HTTP status/redirect; source was not accepted")
    raw = raw_path.read_bytes()
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Empty or oversized evidence")
    text, decoding = extract_text_with_metadata(raw)
    # Byte-exact UTF-8, including LF on Windows; write_text would translate LF.
    (output / "source.txt").write_bytes(text.encode("utf-8"))
    flat = compact(text)
    date_metadata = source_date_metadata(cfg, text)
    if not date_metadata["supported"]:
        raise ValueError("Expected source date field is missing or invalid; preserve raw capture")
    name_at, id_at = flat.find(cfg["subject_name"]), flat.find(cfg["subject_id"])
    if name_at < 0 or id_at < 0 or abs(name_at - id_at) > 2000:
        raise ValueError("Expected company name and identifier are not in the same bounded source context")
    begin, end = max(0, min(name_at, id_at) - 50), min(len(flat), max(name_at, id_at) + 200)
    manifest = {
        "schema": "fixed-official-subject-evidence-v2", "source_key": key,
        "source_url": cfg["url"], "publisher": cfg["publisher"], "source_role": cfg["source_role"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "transport": "HTTPS curl; HTTP200; redirects rejected",
        "raw_file": raw_path.name, "raw_sha256": sha256(raw), "raw_bytes": len(raw),
        "visible_text_file": "source.txt", "visible_text_sha256": sha256(text.encode()),
        "decoding": decoding,
        "subject_name": cfg["subject_name"], "subject_id": cfg["subject_id"],
        "document_date": date_metadata["document_date"], "source_date_metadata": date_metadata,
        "fact_type": cfg["fact_type"],
        "identity_quote_whitespace_normalized": flat[begin:end], "source_claim_scope": cfg["claim"],
        "factual_status": "unknown", "human_review_completed": False,
        "limitation": "Transport/source binding and text support do not establish current registration, license, authorization or credit truth.",
    }
    target = output / "capture.json"
    write_json(target, manifest)
    write_json(output / "transport_receipt.json", {
        "observed_by": "fixed-source capture command", "retrieved_at_utc": manifest["retrieved_at_utc"],
        "source_url": cfg["url"], "observed_effective_url": lines[1], "observed_http_status": 200,
        "raw_sha256": manifest["raw_sha256"], "manifest_sha256": sha256(target.read_bytes()),
        "tls": "curl default certificate validation; no insecure flag; redirects rejected",
        "limitation": "Capture-time transport observation only. This local unsigned receipt and manifest can be edited; offline verification does not reauthenticate the publisher.",
    })
    return target


def verify(bundle: Path, *, today: date | None = None) -> dict:
    bundle = bundle.resolve()
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    cfg = source_config(manifest.get("source_key"))
    errors = []
    if manifest.get("schema") != "fixed-official-subject-evidence-v2":
        errors.append("unsupported_schema")
    for field, expected in (("source_url", cfg["url"]), ("publisher", cfg["publisher"]),
                            ("subject_name", cfg["subject_name"]), ("subject_id", cfg["subject_id"]),
                            ("fact_type", cfg["fact_type"]),
                            ("source_claim_scope", cfg["claim"]), ("source_role", cfg["source_role"])):
        if manifest.get(field) != expected:
            errors.append("manifest_binding_mismatch:" + field)
    for field in ("raw_file", "visible_text_file"):
        path = (bundle.parent / str(manifest.get(field, ""))).resolve()
        if path.parent != bundle.parent or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("Evidence files must be bounded regular files directly inside the bundle")
    raw = (bundle.parent / manifest["raw_file"]).read_bytes()
    raw_ok = sha256(raw) == manifest.get("raw_sha256") and len(raw) == manifest.get("raw_bytes")
    if not raw_ok:
        errors.append("raw_file_integrity_mismatch")
    text, decoding = extract_text_with_metadata(raw)
    if manifest.get("decoding") != decoding:
        errors.append("decoding_metadata_mismatch")
    text_ok = (sha256(text.encode()) == manifest.get("visible_text_sha256")
               and (bundle.parent / manifest["visible_text_file"]).read_bytes() == text.encode())
    if not text_ok:
        errors.append("visible_text_integrity_mismatch")
    flat, quote = compact(text), str(manifest.get("identity_quote_whitespace_normalized", ""))
    quote_ok = bool(quote) and len(quote) <= 2500 and quote in flat
    identity_ok = (quote_ok and cfg["subject_name"] in quote and cfg["subject_id"] in quote
                   and valid_uscc(cfg["subject_id"]))
    if not identity_ok:
        errors.append("source_identity_quote_not_bound")
    date_metadata = source_date_metadata(cfg, text)
    date_supported = date_metadata["supported"]
    if (manifest.get("document_date") != date_metadata["document_date"]
            or manifest.get("source_date_metadata") != date_metadata):
        errors.append("manifest_binding_mismatch:source_date")
    if not date_supported:
        errors.append("document_date_not_supported")
    assertions_supported = all(compact(token) in flat for token in cfg["required_text"])
    if not assertions_supported:
        errors.append("expected_source_statement_missing")
    try:
        captured = datetime.fromisoformat(manifest["retrieved_at_utc"])
        if captured.tzinfo is None or captured.utcoffset() != timedelta(0):
            raise ValueError("capture time must be UTC")
        reference = today or beijing_today()
        if captured.date() > reference or date.fromisoformat(date_metadata["document_date"]) > reference:
            raise ValueError("future capture or document")
    except (KeyError, TypeError, ValueError):
        errors.append("capture_or_document_date_invalid")
    return {
        "status": "source_bound_current_facts_pending" if not errors else "rejected",
        "bundle_sha256": sha256(bundle.read_bytes()), "raw_integrity": raw_ok,
        "visible_text_integrity": text_ok, "source_identity_and_quote_bound": identity_ok,
        "document_date_supported": date_supported, "source_statement_supported": assertions_supported,
        "decoding": decoding,
        "source_key": manifest.get("source_key"), "source_url": cfg["url"],
        "subject_name": cfg["subject_name"], "subject_id": cfg["subject_id"],
        "document_date": date_metadata["document_date"], "source_date_metadata": date_metadata, "errors": errors,
        "current_fact_status": "unknown", "human_review_completed": False,
        "source_authenticity": "offline_manifest_claim_not_reauthenticated",
        "network_reauthentication_performed": False,
        "reference_day_asia_shanghai": (today or beijing_today()).isoformat(),
        "source_claim_scope": cfg["claim"],
    }


def exercise_import(bundle: Path, output: Path, *, today: date | None = None) -> dict:
    today = today or beijing_today()
    verification = verify(bundle, today=today)
    if verification["errors"]:
        raise ValueError("Rejected evidence cannot enter the subject registry")
    if output.exists():
        raise ValueError("Use a fresh result directory; old verification is preserved")
    output.mkdir(parents=True)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    record = {
        "subject_name": manifest["subject_name"], "subject_id": manifest["subject_id"],
        "fact_type": manifest["fact_type"], "status": "unknown",
        "scope": manifest["source_claim_scope"], "valid_from": manifest["document_date"],
        "evidence_ref": f"sha256:{manifest['raw_sha256']}; capture:{bundle.name}",
        "source": {"url": manifest["source_url"], "publisher": manifest["publisher"], "actor_kind": "adapter",
            "reviewer": "automated-source-binding-adapter (not a human reviewer)",
            "verification_method": "Automated exact source/text/identity binding only; no human attestation or present legal-status certification. Imported status remains unknown.",
            "verified_at": manifest["retrieved_at_utc"][:10]},
    }
    # Exercise the actual product import entry with a fresh isolated file, not a mock.
    registry = output / "subject_facts.isolated.json"
    imported = F.import_facts([record], registry)
    records, errors = F.load_facts(registry)
    if errors:
        raise ValueError("Actual imported evidence could not be read back")
    resolved = F.resolve_subject(record["subject_name"], records, subject_ids=[record["subject_id"]], today=today)
    wrong_id = record["subject_id"][:-1] + ("0" if record["subject_id"][-1] != "0" else "1")
    wrong = F.resolve_subject(record["subject_name"], records, subject_ids=[wrong_id], today=today)
    ambiguous = F.resolve_subject(record["subject_name"], records,
                                  subject_ids=[record["subject_id"], wrong_id], today=today)
    assert resolved["records"][0]["identity_match"] == "name_and_identifier"
    assert all(value == "unknown" for value in resolved["checks"].values())
    assert wrong["records"][0]["identity_review_reason"] == "subject_identifier_mismatch"
    assert ambiguous["records"][0]["identity_review_reason"] == "subject_identifier_ambiguous"
    assert resolved["independently_verified"] is False
    report = {"status": "actual_import_identity_binding_passed_current_facts_pending",
              "verification": verification, "registry": registry.name, "import_result": imported,
              "correct_identity_resolution": resolved, "wrong_identity_resolution": wrong,
              "same_name_ambiguous_resolution": ambiguous,
              "product_import_exercised": True, "current_fact_status": "unknown",
              "no_business_registry_or_database_modified": True,
              "limitation": "Explicit adapter provenance is not a human attestation or independent authentication. Unknown business facts cannot be cleared by the actor marker."}
    write_json(output / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--source", choices=sorted(SOURCES), required=True)
    cap.add_argument("--out-dir", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--out-dir", type=Path, required=True)
    check.add_argument("--today", type=date.fromisoformat, default=None,
                       help="Reference day in Asia/Shanghai; defaults to current Beijing date")
    args = parser.parse_args(argv)
    if args.action == "capture":
        target = capture(args.source, args.out_dir)
        result = verify(target)
        print(json.dumps({"capture": str(target), "verification": result}, ensure_ascii=True))
        return 2 if result["errors"] else 0
    else:
        value = exercise_import(args.bundle, args.out_dir, today=args.today)
        print(json.dumps({"status": value["status"], "out_dir": str(args.out_dir),
                          "current_fact_status": value["current_fact_status"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
