"""Immutable local evidence bytes bound to unknown subject facts.

Storage integrity is checked; issuer authenticity is never inferred from a URL,
filename, document number, or a user-supplied declaration.
"""
from __future__ import annotations

import base64
import io
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import zipfile

from .subject_facts import import_facts, validate_fact

LOCK = threading.RLock()
MAX_BYTES = 8 * 1024 * 1024
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt"}


def check_material_fields(filename, content, fields):
    """Compare declarations to extracted bytes; a match is not authenticity."""
    result = {"version": "material-literal-fields-v1", "authenticity_established": False,
              "field_checks": {}, "extraction_status": "unavailable"}
    try:
        if Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raise ValueError("image_requires_independent_transcription_or_ocr_review")
        if Path(filename).suffix.lower() == ".docx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 32*1024*1024:
                    raise ValueError("expanded_document_size_exceeded")
        from .parsers import parse_document
        parsed = parse_document(filename, content)
        text = "\n".join(block.text for block in parsed.blocks)
        if not text.strip():
            raise ValueError("no_extractable_text_requires_manual_review")
        result.update(extraction_status="text_extracted", extracted_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      extracted_character_count=len(text))
        mapping = [index for index,char in enumerate(text) if not char.isspace()]
        compact = "".join(text[index] for index in mapping)
        for key, value in fields.items():
            declared = str(value or "").strip()
            options = [declared]
            if key in {"valid_from", "valid_until"} and declared:
                day = date.fromisoformat(declared)
                options += [f"{day.year}年{day.month}月{day.day}日", f"{day.year}年{day.month:02}月{day.day:02}日",
                            f"{day.year}/{day.month}/{day.day}"]
            found = None
            for option in options:
                target = re.sub(r"\s+", "", option)
                index = compact.find(target) if target else -1
                if index >= 0:
                    start, end = mapping[index], mapping[index+len(target)-1]+1
                    found = {"status": "literal_match", "start": start, "end": end, "quote": text[start:end]}
                    break
            result["field_checks"][key] = found or {"status": "not_found_requires_review" if declared else "not_supplied_requires_review"}
    except Exception as exc:
        result["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        result["field_checks"] = {key:{"status":"extraction_unavailable_requires_review"} for key in fields}
    result["all_fields_literally_matched"] = bool(result["field_checks"]) and all(
        item["status"]=="literal_match" for item in result["field_checks"].values())
    return result


class SubjectMaterialStore:
    def __init__(self, root, facts_path):
        self.root, self.facts_path = Path(root), Path(facts_path)

    def import_material(self, filename, content, metadata):
        if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("filename仅可填写文件名，不能包含路径")
        if Path(filename).suffix.lower() not in ALLOWED:
            raise ValueError("材料支持PDF、PNG、JPEG、DOCX和TXT")
        if not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES:
            raise ValueError("每个原件需为1字节至8MB")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("fact"), dict):
            raise ValueError("metadata.fact须为带来源的主体事实对象")
        number = str(metadata.get("document_number") or "").strip()
        issuer = str(metadata.get("issuer") or "").strip()
        if not number or not issuer or len(number) > 200 or len(issuer) > 200:
            raise ValueError("需填写证照/文书编号和发证/出具机构（未独立认证的声明）")
        supplied = dict(metadata["fact"])
        if not supplied.get("subject_id") or not supplied.get("scope"):
            raise ValueError("材料须绑定主体唯一代码及许可/授权/登记范围")
        declared_status = supplied.get("status", "unknown")
        supplied["status"] = "unknown"
        supplied.pop("record_sha256", None)
        digest = hashlib.sha256(content).hexdigest()
        supplied["evidence_ref"] = "material:" + digest + ":sha256:" + digest
        source = dict(supplied.get("source") or {})
        source["actor_kind"] = "adapter"
        # The supplied reviewer and date stay supplied provenance. This process
        # only binds bytes and does not impersonate a human fact verifier.
        source["verification_method"] = "原件字节绑定，业务事实待核验；提供者声明方法：" + str(source.get("verification_method") or "")
        supplied["source"] = source
        fact = validate_fact(supplied)
        warnings = ["issuer_authenticity_not_independently_verified", "scope_requires_human_review"]
        today = date.today().isoformat()
        if fact["valid_until"] and fact["valid_until"] < today:
            warnings.append("material_expired")
        if fact["valid_from"] and fact["valid_from"] > today:
            warnings.append("material_not_yet_effective")
        if fact["fact_type"] in {"license", "authorization"} and not fact["valid_until"]:
            warnings.append("validity_end_not_supplied")
        if issuer != source.get("publisher"):
            warnings.append("issuer_publisher_declarations_differ")
        consistency = check_material_fields(filename, content, {"subject_name": fact["subject_name"],
            "subject_id": fact["subject_id"], "document_number": number, "issuer": issuer,
            "scope": fact["scope"], "valid_from": fact["valid_from"], "valid_until": fact["valid_until"]})
        if not consistency["all_fields_literally_matched"]:
            warnings.append("material_fields_not_fully_matched_requires_review")
        record = {"schema": "subject-material-v1", "material_id": digest, "sha256": digest,
                  "filename": filename, "size_bytes": len(content), "document_number": number,
                  "issuer_declared": issuer, "subject_name": fact["subject_name"], "subject_id": fact["subject_id"],
                  "fact_id": fact["fact_id"], "fact": fact, "declared_status": declared_status,
                  "scope": fact["scope"], "valid_from": fact["valid_from"], "valid_until": fact["valid_until"],
                  "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                  "verification_status": "unknown", "independently_verified": False,
                  "review_reasons": warnings}
        record["material_consistency"] = consistency
        # Metadata is also immutable: same bytes can legitimately support two
        # subjects/claims; each binding receives its own deterministic key.
        binding = hashlib.sha256(json.dumps({"sha256": digest, "fact_id": fact["fact_id"],
                                            "number": number, "issuer": issuer}, sort_keys=True).encode()).hexdigest()
        record["binding_id"] = binding
        record["metadata_sha256"] = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            blob = self.root / (digest + ".blob")
            if blob.exists():
                if hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
                    raise ValueError("已存原件完整性失败，未覆盖原文件")
            else:
                with blob.open("xb") as stream:
                    stream.write(content)
            target = self.root / (binding + ".json")
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    raise ValueError("已有材料绑定须为对象，未覆盖")
                canonical = {key:value for key,value in existing.items() if key!="metadata_sha256"}
                if existing.get("metadata_sha256") != hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
                    raise ValueError("已有材料元数据完整性失败，未覆盖")
                if existing.get("fact_id") != fact["fact_id"] or existing.get("sha256") != digest:
                    raise ValueError("已有材料绑定不一致，未覆盖")
                record = existing
            else:
                temporary = self.root / (binding + ".tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, target)
            # If this fails, the immutable material remains for a transparent
            # retry; no positive business fact has been created.
            imported = import_facts([fact], self.facts_path)
        return {"material": record, "facts": imported, "status": "stored_pending_verification"}

    def list_materials(self):
        records, errors = [], []
        if not self.root.exists():
            return {"records": records, "total": 0, "errors": errors}
        for path in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("material_record_must_be_object")
                metadata = {key:value for key,value in record.items() if key!="metadata_sha256"}
                if record.get("metadata_sha256") != hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
                    raise ValueError("material_metadata_integrity_failed")
                digest = record.get("sha256", "")
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    raise ValueError("invalid_digest")
                blob = self.root / (digest + ".blob")
                record["storage_integrity"] = "valid" if blob.exists() and hashlib.sha256(blob.read_bytes()).hexdigest() == digest else "failed"
                records.append(record)
            except (ValueError, OSError, TypeError) as exc:
                errors.append({"binding_file": path.name, "error": type(exc).__name__})
        return {"records": records, "total": len(records), "errors": errors,
                "scope": "原件哈希与声明元数据核验；所有业务事实保持unknown，机构真伪和授权范围需人工/获授权接口核验"}


def decode_content(value):
    if not isinstance(value, str) or len(value) > (MAX_BYTES * 4 // 3 + 8):
        raise ValueError("base64材料缺失或超过8MB")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("材料base64不合法") from exc
