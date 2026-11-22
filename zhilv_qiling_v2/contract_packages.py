"""Auditable, offline-readable import of the public procurement contract package.

Only original sources, page-level OCR and statute texts are ingested.  Collection
notes and earlier review conclusions are never promoted to contract evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import Config, WORKSPACE
from .models import DocumentParse, SourceBlock
from .splitter import split_document

DEFAULT_STORE = WORKSPACE / "data" / "contract_packages"
SCHEMA_VERSION = 1
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}\Z")
_CN = "零〇一二三四五六七八九十百千万两"
_ARTICLE = re.compile(rf"^\s*第([{_CN}\d]+)\s*条\s*(.*)$")
_SECTION = re.compile(rf"^\s*([{_CN}]+)、\s*(.*)$")
_ITEM = re.compile(rf"^\s*[（(]([{_CN}]+)[）)]\s*(.*)$")
_PAGE = re.compile(r"^(?:=+\s*)?(?:\[|【)?第\s*(\d+)\s*页(?:\s*/\s*共\s*\d+\s*页)?(?:\]|】)?(?:\s*=+)?$")
_FOOTER = re.compile(r"^\s*第\s*\d+\s*页\s*(?:/|，|,)?\s*共\s*\d+\s*页\s*$")


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks escaping the declared package root."""
    relative = str(relative).replace("\\", "/")
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or ".." in parsed.parts or ":" in relative:
        raise ValueError(f"Unsafe package path: {relative}")
    path = root.joinpath(*parsed.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes root: {relative}")
    return path


def _package_dir(package_id: str, store_root=None) -> Path:
    if not _ID.fullmatch(str(package_id)):
        raise ValueError("Invalid package_id")
    return _safe(Path(store_root or DEFAULT_STORE).resolve(), package_id)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _cn_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    total, current = 0, 0
    for char in value:
        if char in digits:
            current = digits[char]
        else:
            unit = {"十": 10, "百": 100, "千": 1000, "万": 10000}.get(char)
            if not unit:
                raise ValueError(f"Unknown Chinese numeral: {value}")
            if unit == 10000:
                total = (total + current) * unit
            else:
                total += (current or 1) * unit
            current = 0
    return total + current


def _privacy_documents() -> list[dict]:
    # The inability to obtain these nonpublic files is the user's statement,
    # not a finding that every document is legally classified as confidential.
    rows = [
        ("award_notice", "中标通知书", [], ["中标通知内容和送达情况"]),
        ("bid_documents", "完整投标文件及澄清文件", [], ["完整投标承诺、技术和服务方案"]),
        ("delivery_specification", "合同引用的采购货物明细表", [], ["未在合同及公开报价表中载明的货物规格和承诺"]),
        ("authorization_documents", "实际签约授权委托书及签字人身份证明", ["authorization"], ["签字人身份、代理权限和授权范围"]),
        ("performance_security", "履约保证金凭证或保函", [], ["保证金是否实际提交及相关生效条件"]),
        ("acceptance_payment", "验收单、交付记录、付款凭证及发票", [], ["实际交货、履约验收、开票与付款情况"]),
        ("internal_approval", "采购内部审批及备案回执", [], ["内部审批、变更审批和备案办理情况"]),
    ]
    return [{"document_id": key, "title": title,
             "availability": "unavailable_private", "reason_source": "user_statement",
             "reason": "用户本轮说明中标通知书等非公开文件涉及隐私，无法获取；本声明仅解释材料缺口。",
             "fact_types": types, "subject_names": [], "subject_roles": ["buyer", "seller"] if types else [],
             "blocked_fact_scope": blocked, "status": "unknown", "auto_retry": False,
             "resolution": "相关结论保留待核验；可接受经授权的脱敏证据或人工核对结果。"}
            for key, title, types, blocked in rows]


def _role(relative: str) -> str:
    if relative.startswith("03_供应商主体资料/"):
        return "subject_collection_record"
    if relative.startswith("04_公开法条/"):
        return "statute_source"
    if relative.startswith("01_合同文本/"):
        return "contract_public_source"
    return "procurement_background"


def _read_ocr(path: Path, source: dict) -> list[dict]:
    pages = _json(path)
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Empty or invalid OCR pages: {path.name}")
    seen, result = set(), []
    for raw in pages:
        page = int(raw["page"])
        if page < 1 or page in seen:
            raise ValueError("OCR page numbering is duplicate or invalid")
        seen.add(page)
        lines = []
        for i, line in enumerate(raw.get("lines", [])):
            text = str(line.get("text", "")).strip()
            if not text or _FOOTER.fullmatch(text):
                continue
            lines.append({"line": i + 1, "text": text,
                          "min_score": line.get("min_score"), "y": line.get("y")})
        result.append({"page": page, "text": "\n".join(x["text"] for x in lines), "lines": lines})
    result.sort(key=lambda item: item["page"])
    if [p["page"] for p in result] != list(range(1, len(result) + 1)):
        raise ValueError("OCR page sequence contains missing pages")
    return result


def _pdf_pages(path: Path) -> list[dict]:
    import pymupdf
    with pymupdf.open(path) as doc:
        return [{"page": page.number + 1, "text": page.get_text(), "lines": []} for page in doc]


def _normative_units(text: str, *, sections_only=False) -> list[dict]:
    """Keep original article/item labels; never invent article numbers for notices."""
    lines = text.splitlines()
    page, content, units, current = None, [], [], None
    def flush():
        nonlocal content, current
        if current and content:
            current["content"] = "\n".join(content).strip()
            units.append(current)
        content, current = [], None
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        marker = _PAGE.fullmatch(line)
        if marker:
            page = int(marker.group(1))
            continue
        if not line or line.isdigit() or _FOOTER.fullmatch(line):
            continue
        article = None if sections_only else _ARTICLE.match(line)
        section = _SECTION.match(line) if sections_only else None
        item = _ITEM.match(line) if sections_only else None
        match = article or section or item
        if match:
            flush()
            kind = "article" if article else ("item" if item else "section")
            label = (f"第{match.group(1)}条" if article else
                     f"（{match.group(1)}）" if item else f"{match.group(1)}、")
            current = {"article_no": label, "unit_kind": kind,
                       "number": _cn_number(match.group(1)), "line_start": line_no,
                       "page_start": page, "page_end": page}
            content = [line]
        elif current:
            # Attached blank declaration forms are not part of the final article.
            if re.match(r"^(附(?:[件表]|\s*\d|[:：])|中小企业声明函[（(]|信息公开选项[:：]|抄送[:：])", line) and current["unit_kind"] == "article":
                flush()
                continue
            if line == "财政部" and current["unit_kind"] in {"section", "item"}:
                flush()
                continue
            # A new chapter title belongs to the next article, not the prior one.
            if re.match(rf"^第[{_CN}\d]+[章节编]", line):
                continue
            content.append(line)
            current["page_end"] = page
    flush()
    return units


def _summary(manifest: dict, package_dir: Path) -> dict:
    return {"package_id": manifest["package_id"], "title": manifest["title"],
            "imported_at": manifest["imported_at"], "source_count": len(manifest["sources"]),
            "document_count": len(manifest["documents"]), "coverage": manifest["coverage"],
            "statute_count": manifest["statute_count"], "statute_document_count": manifest["statute_document_count"],
            "missing_documents": manifest["missing_documents"],
            "statutes_path": str(package_dir / "statutes.jsonl"),
            "package_path": str(package_dir), "manifest_sha256": _hash(package_dir / "manifest.json")}


def import_contract_package(source_root, store_root=None, package_id=None) -> dict:
    root = Path(source_root).resolve()
    provenance_path = _safe(root, "05_采集与校验/provenance.json")
    provenance = _json(provenance_path)
    entries = {str(e["path"]).replace("\\", "/"): e for e in provenance["entries"]}
    # Validate all enumerated files before any durable write, including derived input.
    for rel, entry in entries.items():
        file = _safe(root, rel)
        if not file.is_file() or _hash(file) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {rel}")
    fingerprint = hashlib.sha256(json.dumps(
        sorted((p, e["sha256"]) for p, e in entries.items()), ensure_ascii=False).encode()).hexdigest()
    package_id = package_id or f"beijing_procurement_2026_{fingerprint[:12]}"
    destination = _package_dir(package_id, store_root)
    if destination.exists():
        existing = load_contract_package(package_id, store_root=store_root)
        if existing["input_fingerprint"] != fingerprint:
            raise ValueError("Existing package ID belongs to different source content")
        return _summary(existing, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{package_id}_", dir=destination.parent))
    sources, by_relative = [], {}
    try:
        (staging / "sources").mkdir()
        def add_source(rel: str, role: str | None = None) -> dict:
            rel = str(rel).replace("\\", "/")
            if rel in by_relative:
                return by_relative[rel]
            entry = entries.get(rel)
            if not entry:
                raise ValueError(f"Required input is absent from provenance: {rel}")
            path = _safe(root, rel)
            digest = entry["sha256"]
            source_id = "src_" + hashlib.sha256((rel + digest).encode()).hexdigest()[:20]
            stored = f"sources/{digest}{path.suffix.lower()}"
            if not (staging / stored).exists():
                shutil.copyfile(path, staging / stored)
            row = {"source_id": source_id, "title": path.name, "original_path": rel,
                   "stored_path": stored, "sha256": digest, "bytes": path.stat().st_size,
                   "source_url": entry.get("source_url", ""), "fetched_at": entry.get("fetched_at", ""),
                   "origin": entry.get("origin", "unknown"), "role": role or _role(rel),
                   "evidence_status": "source_preserved_not_fact_verified"}
            by_relative[rel] = row
            sources.append(row)
            return row
        for rel, entry in entries.items():
            if entry.get("origin") == "public_source":
                add_source(rel)
        definitions = [
            ("original_contract", "原合同", "01_合同文本/01_原合同/source/2026年年度人口抽样调查合同.pdf",
             "01_合同文本/01_原合同/原合同_OCR框.json", "original_contract"),
            ("supplemental_agreement", "补充协议", "01_合同文本/02_补充协议/source/2026年年度人口抽样调查--补充协议10.29.pdf",
             "01_合同文本/02_补充协议/补充协议_OCR框.json", "supplemental_agreement"),
            ("tender_background", "招标文件（背景与模板）", "02_采购交易背景/03_招标文件/source/人口抽样调查物资采购招标文件.pdf",
             None, "tender_background"),
        ]
        documents = []
        for doc_id, title, rel, ocr_rel, role in definitions:
            source = add_source(rel)
            source["role"] = role
            if ocr_rel:
                extraction = add_source(ocr_rel, "ocr_derivative")
                pages = _read_ocr(_safe(root, ocr_rel), source)
                actual_pages = len(_pdf_pages(_safe(root, rel)))
                if len(pages) != actual_pages:
                    raise ValueError(f"OCR/PDF page count mismatch: {title}")
                quality = {"method": "saved_rapidocr", "human_verified": False,
                           "status": "machine_ocr_unverified", "source_id": extraction["source_id"],
                           "normalization": "仅移除空行和独立页脚；不修正合同词句、金额、日期或手写字段。"}
            else:
                pages = _pdf_pages(_safe(root, rel))
                quality = {"method": "pdf_text_layer", "human_verified": False, "status": "text_layer_extracted"}
            documents.append({"document_id": doc_id, "title": title, "role": role,
                              "source_id": source["source_id"], "sha256": source["sha256"],
                              "page_count": len(pages), "pages": pages, "extraction_quality": quality,
                              "default_review": role != "tender_background",
                              "is_signed_evidence": False if role == "tender_background" else None})
        statutes, statute_documents = [], []
        for folder in sorted((root / "04_公开法条").iterdir()):
            if not folder.is_dir() or not (folder / "meta.json").is_file():
                continue
            meta_rel = (folder / "meta.json").relative_to(root).as_posix()
            meta = _json(_safe(root, meta_rel))
            add_source(meta_rel, "statute_metadata")
            raw_rel = (meta.get("全文直接来源") or meta.get("原始文件") or meta.get("原始文件_PDF") or meta.get("source_file"))
            source_rel = folder.relative_to(root).as_posix() + "/" + str(raw_rel).replace("\\", "/")
            source = add_source(source_rel)
            text_rel = (folder / "全文.txt").relative_to(root).as_posix()
            extracted = add_source(text_rel, "statute_extracted_text")
            text = _safe(root, text_rel).read_text(encoding="utf-8-sig")
            name = meta.get("名称") or meta.get("name") or folder.name
            sections_only = not isinstance(meta.get("条文数"), int)
            units = _normative_units(text, sections_only=sections_only)
            if not units:
                raise ValueError(f"No normative units found: {name}")
            expected = meta.get("条文数")
            if isinstance(expected, int) and [u["number"] for u in units] != list(range(1, expected + 1)):
                raise ValueError(f"Article sequence differs from source metadata: {name}")
            quality = {"method": "source_package_extracted_text", "human_verified": False,
                       "status": "machine_ocr_unverified" if folder.name.startswith("15_") else "extracted_text",
                       "notes": meta.get("ocr_review_status") or meta.get("全文提取方式") or "未声明独立逐字人工校核"}
            url = meta.get("全文直接来源URL") or meta.get("来源URL") or meta.get("source_url") or meta.get("url") or source["source_url"]
            authority = "法律" if folder.name[:2] in {"01", "02", "04"} else "行政法规" if folder.name[:2] in {"03", "05", "06"} else "地方性法规" if folder.name.startswith("11_") else "部门规章" if folder.name[:2] in {"07", "08", "09", "10"} else "规范性文件"
            for unit in units:
                uid = f"pkg_{package_id}_{folder.name[:2]}_{unit['unit_kind']}_{unit['number']}"
                statutes.append({"article_uid": uid, "law_name": name, "article_no": unit["article_no"],
                                 "content": unit["content"], "authority_level": authority,
                                 "status": "有效" if meta.get("时效性") == "有效" else "待时效核验",
                                 "risk_tags": [], "contract_types": ["买卖合同", "政府采购合同"],
                                 "source_url": url, "package_id": package_id, "source_id": source["source_id"],
                                 "original_sha256": source["sha256"], "text_sha256": extracted["sha256"],
                                 "text_source_id": extracted["source_id"], "source_status_statement": meta.get("时效性", "未记载"),
                                 "applicability_status": "requires_case_assessment",
                                 "extraction_quality": quality,
                                 "location": {k: unit[k] for k in ("unit_kind", "line_start", "page_start", "page_end")}})
            statute_documents.append({"title": name, "source_id": source["source_id"], "unit_count": len(units),
                                      "unit_style": "sections_and_items" if sections_only else "articles",
                                      "source_url": url, "extraction_quality": quality})
        _write_json(staging / "documents.json", documents)
        (staging / "statutes.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in statutes), encoding="utf-8")
        _write_json(staging / "input_provenance.json", provenance)
        manifest = {"schema_version": SCHEMA_VERSION, "package_id": package_id,
                    "title": "北京市2026年年度人口抽样调查物资采购合同包",
                    "imported_at": datetime.now(timezone.utc).isoformat(), "input_fingerprint": fingerprint,
                    "input_provenance_sha256": _hash(provenance_path), "sources": sources,
                    "documents": [{k: v for k, v in d.items() if k != "pages"} for d in documents],
                    "statute_documents": statute_documents, "statute_count": len(statutes),
                    "statute_document_count": len(statute_documents), "missing_documents": _privacy_documents(),
                    "coverage": {"contract_pages": sum(d["page_count"] for d in documents if d["default_review"]),
                                 "background_pages": sum(d["page_count"] for d in documents if not d["default_review"]),
                                 "contract_body_available": True, "all_referenced_annexes_available": False,
                                 "subject_truth_verified": False, "human_gold_labels_available": False},
                    "usage_boundaries": ["公开合同扫描件不自动证明签约授权和主体真实性。", "招标文件中的空白模板不是已签署授权。", "隐私缺件不会将unknown、pending或needs_manual改判通过。", "法条收录不代表对个案适用已确认。"],
                    "generated_files": {name: _hash(staging / name) for name in ("documents.json", "statutes.jsonl", "input_provenance.json")}}
        _write_json(staging / "manifest.json", manifest)
        # Directory rename publishes only a complete, hash-checked package.
        staging.rename(destination)
        return _summary(load_contract_package(package_id, store_root=store_root), destination)
    except Exception:
        if staging.exists() and staging.resolve().parent == destination.parent.resolve():
            shutil.rmtree(staging)
        raise


def load_contract_package(package_id, store_root=None, verify_files=True) -> dict:
    directory = _package_dir(package_id, store_root)
    manifest = _json(_safe(directory, "manifest.json"))
    if manifest.get("package_id") != package_id or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Package manifest identity or schema mismatch")
    for row in manifest["sources"]:
        path = _safe(directory, row["stored_path"])
        if verify_files and (not path.is_file() or _hash(path) != row["sha256"]):
            raise ValueError(f"Stored source hash mismatch: {row['source_id']}")
    for filename, digest in manifest["generated_files"].items():
        path = _safe(directory, filename)
        if verify_files and (not path.is_file() or _hash(path) != digest):
            raise ValueError(f"Stored package data hash mismatch: {filename}")
    # Derived return fields bind queued runs to these exact bytes. They are not
    # written into manifest.json, so the hash never depends on itself.
    manifest["manifest_sha256"] = _hash(directory / "manifest.json")
    manifest["statutes_path"] = str(directory / "statutes.jsonl")
    manifest["statutes_path_kind"] = "absolute_local_archive"
    return manifest


def list_contract_packages(store_root=None) -> list[dict]:
    root = Path(store_root or DEFAULT_STORE)
    if not root.exists():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and _ID.fullmatch(folder.name) and (folder / "manifest.json").exists():
            manifest = load_contract_package(folder.name, store_root=root)
            result.append(_summary(manifest, folder))
    return result


def package_source_path(package_id, source_id, store_root=None) -> Path:
    manifest = load_contract_package(package_id, store_root=store_root, verify_files=False)
    row = next((r for r in manifest["sources"] if r["source_id"] == source_id), None)
    if row is None:
        raise KeyError("Package source_id not found")
    path = _safe(_package_dir(package_id, store_root), row["stored_path"])
    if not path.is_file() or _hash(path) != row["sha256"]:
        raise ValueError("Source hash mismatch")
    return path


def load_package_statutes(package_id, store_root=None) -> list[dict]:
    load_contract_package(package_id, store_root=store_root)
    path = _package_dir(package_id, store_root) / "statutes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def package_evidence_summary(package_id, store_root=None) -> dict:
    manifest = load_contract_package(package_id, store_root=store_root)
    return {"package_id": package_id, "source_manifest": manifest["sources"],
            "missing_documents": manifest["missing_documents"], "coverage": manifest["coverage"],
            "documents": manifest["documents"], "statute_documents": manifest["statute_documents"],
            "privacy_statement": "用户说明中标通知书等非公开文件涉及隐私无法获取；相关事实仍待核验。",
            "privacy_reason_source": "user_statement", "usage_boundaries": manifest["usage_boundaries"],
            "statutes_path": str(_package_dir(package_id, store_root) / "statutes.jsonl"),
            "statutes_path_kind": "absolute_local_archive",
            "statutes_sha256": manifest["generated_files"]["statutes.jsonl"],
            "manifest_sha256": _hash(_package_dir(package_id, store_root) / "manifest.json")}


def search_package_background(package_id, query, limit=8, store_root=None) -> list[dict]:
    """Search preserved tender pages without promoting templates to signed facts."""
    load_contract_package(package_id, store_root=store_root)
    query = str(query).strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 50))
    tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower()))
    for token in list(tokens):
        if len(token) > 3 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token[i:i + 2] for i in range(len(token) - 1))
    rows = []
    for doc in _json(_package_dir(package_id, store_root) / "documents.json"):
        if doc["role"] != "tender_background":
            continue
        for page in doc["pages"]:
            body = page["text"].lower()
            score = sum(min(body.count(t), 5) * len(t) for t in tokens)
            if score:
                rows.append({"source_id": doc["source_id"], "title": doc["title"],
                             "document_id": doc["document_id"], "page": page["page"],
                             "text": page["text"], "score": score,
                             "source_role": "tender_background", "is_signed_evidence": False,
                             "original_sha256": doc["sha256"]})
    return sorted(rows, key=lambda r: (-r["score"], r["page"]))[:limit]


def _article_heading(text: str) -> bool:
    """Recognize explicit source headings, not numeric rows or ordinary references."""
    if len(text) > 80 or re.search(r"[。；;]", text):
        return False
    return bool(re.match(rf"^(?:第[{_CN}\d]+条|[{_CN}]+、)\s*\S", text))


def _table_header(text: str) -> bool:
    # Require several real column labels together. A narrative mentioning only
    # a quantity and price is not sufficient to start a table evidence group.
    if len(text) > 120 or re.search(r"[。；;]", text):
        return False
    return (bool(re.search(r"\s{2,}|\t|\|", text))
            and "单价" in text and ("总价" in text or "金额" in text)
            and any(term in text for term in ("货物名称", "品名", "品目", "规格")))


def _annotate_source_context_groups(parsed: DocumentParse, doc: dict) -> list[dict]:
    """Add context scopes without rewriting text, IDs or the archived package.

    Existing segmentation is intentionally preserved. These source-local scopes
    supply complete article/table evidence even where numbered subclauses were
    assigned to the wrong hierarchy by the general boundary classifier.
    """
    blocks, groups = parsed.blocks, []
    if not blocks:
        return groups

    def add_group(kind: str, start: int, end: int, title: str, **extra):
        selected = blocks[start:end]
        if not selected:
            return
        group_id = kind + "_" + hashlib.sha256(
            f"{doc['source_id']}:{selected[0].block_id}".encode()).hexdigest()[:20]
        block_ids = [b.block_id for b in selected]
        block_set = set(block_ids)
        clause_ids = [c.clause_id for c in parsed.clauses if block_set.intersection(c.block_ids)]
        group = {"group_id": group_id, "kind": kind, "source_id": doc["source_id"],
                 "document_id": doc["document_id"], "source_role": doc["role"],
                 "title": title, "block_ids": block_ids, "clause_ids": clause_ids,
                 "source_page_start": selected[0].metadata["source_page"],
                 "source_page_end": selected[-1].metadata["source_page"],
                 "text": "\n".join(b.text for b in selected),
                 "requires_full_context": True,
                 "original_sha256": doc["sha256"],
                 "extraction_quality": doc["extraction_quality"], **extra}
        groups.append(group)
        key = "table_group_id" if kind == "ocr_table" else "article_group_id"
        for block in selected:
            block.metadata[key] = group_id
            block.metadata.setdefault("context_group_ids", []).append(group_id)
            block.metadata["requires_full_context"] = True
            if kind == "ocr_table":
                block.metadata["table_amount_verification_status"] = "needs_manual"
        for clause in parsed.clauses:
            if clause.clause_id not in clause_ids:
                continue
            clause.metadata.setdefault(key + "s", []).append(group_id)
            clause.metadata.setdefault(key, group_id)
            clause.metadata.setdefault("context_group_ids", []).append(group_id)
            clause.metadata["requires_full_context"] = True
            if kind == "ocr_table":
                clause.metadata["table_amount_verification_status"] = "needs_manual"
                clause.metadata["column_alignment_verified"] = False

    start = None
    section_break = re.compile(r"^(?:合同书|合同(?:一般|特殊|通用|专用)条款)$")
    for index, block in enumerate(blocks):
        if _article_heading(block.text):
            if start is not None:
                add_group("article", start, index, blocks[start].text,
                          scope_reason="显式条号或中文章节标题及其后续子款，直到下一同级标题。")
            start = index
        elif start is not None and section_break.fullmatch(block.text.strip()):
            add_group("article", start, index, blocks[start].text,
                      scope_reason="显式条号或中文章节标题及其后续子款，截止新的合同部分。")
            start = None
    if start is not None:
        add_group("article", start, len(blocks), blocks[start].text,
                  scope_reason="显式条号或中文章节标题及其后续子款，截止本来源末尾。")

    # Table grouping is a conservative OCR context repair, not reconstruction of
    # rows/columns. It spans consecutive pages and ends at a total or new heading.
    index = 0
    while index < len(blocks):
        if not _table_header(blocks[index].text):
            index += 1
            continue
        start = index
        for previous in range(index - 1, max(-1, index - 7), -1):
            text = blocks[previous].text.strip()
            if (re.match(r"^单位\s*[:：]", text) or text in {"数量", "序", "号", "序号", "套", "台/件/"}):
                start = previous
            elif re.match(rf"^[（(][{_CN}]+[）)]", text) and any(
                    term in text for term in ("货物", "报价", "清单", "明细", "价目")):
                start = previous
                break
            else:
                break
        end, boundary = index + 1, "source_end_unconfirmed"
        first_page = blocks[start].metadata["source_page"]
        for nxt in range(index + 1, len(blocks)):
            candidate = blocks[nxt]
            text = candidate.text.strip()
            page = candidate.metadata["source_page"]
            previous_page = blocks[nxt - 1].metadata["source_page"]
            if page - previous_page > 1:
                end, boundary = nxt, "page_gap_unconfirmed"
                break
            if nxt - start >= 100 or page - first_page >= 4:
                end, boundary = nxt, "scope_cap_unconfirmed"
                break
            if (_article_heading(text) or section_break.fullmatch(text)
                    or re.match(rf"^[（(][{_CN}]+[）)]\s*[^\d\s]", text)):
                end, boundary = nxt, "next_explicit_heading"
                break
            end = nxt + 1
            if re.match(r"^(?:合计|总计)", text) and re.search(r"\d", text):
                boundary = "printed_total_line"
                break
        add_group("ocr_table", start, end, "OCR货物数量与价格表（待逐行核对）",
                  scope_reason="多个价格/货物列标题识别的连续表格区；保留跨页行文，不推断列归属。",
                  boundary_method=boundary, boundary_confirmed=boundary in {"next_explicit_heading", "printed_total_line"},
                  verification_status="needs_manual", amount_verification_status="needs_manual",
                  column_alignment_verified=False, auto_pass_eligible=False,
                  review_reason="OCR行列可能交错；金额计算只能作为基于识别数字的条件核算，不能代替原件逐行核对。")
        index = max(index + 1, end)
    return groups


def prepare_package_document(package_id, config=None, include_tender=False, store_root=None) -> DocumentParse:
    manifest = load_contract_package(package_id, store_root=store_root)
    documents = _json(_package_dir(package_id, store_root) / "documents.json")
    config = config or Config()
    blocks, clauses, boundaries, issues, segmentations, context_groups = [], [], [], [], [], []
    page_offset = 0
    for doc in documents:
        if not doc["default_review"] and not include_tender:
            continue
        local_blocks = []
        for page in doc["pages"]:
            lines = page["lines"] or [{"text": s} for s in page["text"].splitlines() if s.strip()]
            for line_index, line in enumerate(lines):
                text = line["text"].strip()
                if not text:
                    continue
                local_blocks.append(SourceBlock(
                    block_id=f"{doc['document_id']}_p{page['page']}_l{line_index + 1}",
                    order=len(blocks) + len(local_blocks), kind="paragraph", text=text,
                    page=page_offset + page["page"], metadata={
                        "package_id": package_id, "document_id": doc["document_id"], "source_id": doc["source_id"],
                        "source_title": doc["title"], "source_page": page["page"], "source_role": doc["role"],
                        "original_sha256": doc["sha256"], "extraction_quality": doc["extraction_quality"],
                        "ocr_line": line.get("line", line_index + 1), "ocr_min_score": line.get("min_score"),
                        "ocr_y": line.get("y"), "is_signed_evidence": doc["is_signed_evidence"]}))
        parsed = split_document(local_blocks, config, source_name=doc["title"])
        id_map = {c.clause_id: f"{doc['document_id']}_{c.clause_id}" for c in parsed.clauses}
        for clause in parsed.clauses:
            old_id = clause.clause_id
            clause.clause_id = id_map[old_id]
            clause.parent_id = id_map.get(clause.parent_id) if clause.parent_id else None
            clause.metadata.update({"package_id": package_id, "document_id": doc["document_id"],
                                    "source_id": doc["source_id"], "source_title": doc["title"],
                                    "source_role": doc["role"], "original_sha256": doc["sha256"],
                                    "source_page_start": clause.page_start - page_offset if clause.page_start else None,
                                    "source_page_end": clause.page_end - page_offset if clause.page_end else None,
                                    "extraction_quality": doc["extraction_quality"]})
        for boundary in parsed.boundaries:
            boundary.boundary_id = f"{doc['document_id']}_{boundary.boundary_id}"
        for issue in parsed.issues:
            issue.clause_ids = [id_map.get(cid, cid) for cid in issue.clause_ids]
        context_groups.extend(_annotate_source_context_groups(parsed, doc))
        blocks.extend(parsed.blocks)
        clauses.extend(parsed.clauses)
        boundaries.extend(parsed.boundaries)
        issues.extend(parsed.issues)
        segmentations.append({"document_id": doc["document_id"], **parsed.stats})
        page_offset += doc["page_count"]
    context = package_evidence_summary(package_id, store_root=store_root)
    context["include_tender_in_review"] = bool(include_tender)
    context["context_group_version"] = 1
    context["context_groups"] = context_groups
    context["clause_groups"] = context_groups
    context["ocr_table_reviews"] = [
        {key: group[key] for key in ("group_id", "source_id", "document_id", "title", "clause_ids",
                                     "source_page_start", "source_page_end", "verification_status",
                                     "amount_verification_status", "column_alignment_verified",
                                     "boundary_confirmed", "review_reason")}
        for group in context_groups if group["kind"] == "ocr_table"]
    return DocumentParse(source_name=manifest["title"], file_type="contract_package",
                         full_text="\n".join(b.text for b in blocks), blocks=blocks, clauses=clauses,
                         boundaries=boundaries, issues=issues,
                         warnings=["合同原件为扫描件，OCR未逐字独立人工复核。", "隐私缺件仅解释无法取得原因，依赖其证明的事实仍为待核验。"]
                                  + (["OCR表格已保留完整上下文；行列映射和金额一致性仍待原件核对。"] if context["ocr_table_reviews"] else []),
                         stats={"blocks": len(blocks), "clauses": len(clauses), "pages": page_offset,
                                "segmentation_by_document": segmentations, "contract_package": context})
