"""Law repository contracts against real temporary SQLite/filesystem storage."""
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ADMIN = {"user_id": "law-reviewer", "role": "admin"}
URL = "https://www.gov.cn/xinwen/2020-06/01/content_5516647.htm"
ARTICLES = [{"article_no": "第一条", "content": "为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，制定本法。", "location": {"paragraph": 1}}]


class LawLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.assertIsNotNone(importlib.util.find_spec("zhilv_qiling_v2.law_library"), "law repository has not been implemented")
        self.mod = importlib.import_module("zhilv_qiling_v2.law_library")
        self.lib = self.mod.LawLibrary(Path(self.tmp.name) / "laws.sqlite")

    def ingest(self, **changes):
        doc = {"title": "中华人民共和国民法典", "domains": ["general", "real_estate"], **changes}
        return self.lib.ingest_document(ADMIN, doc, ARTICLES, b"official original", URL)

    def effective(self, version_id, **changes):
        self.lib.set_version_metadata(ADMIN, version_id, {
            "validity": "effective", "effective_from": "2021-01-01",
            "verification": {"verified": True, "evidence": "核对官方全文、施行条款和现行状态", "verified_by": "forged"}, **changes})
        return self.lib.publish_version(ADMIN, version_id)

    def test_admin_required_and_draft_invisible(self):
        for actor in ({"role": "admin"}, {"user_id": "reader", "role": "user"}):
            with self.assertRaises(self.mod.LawError) as err:
                self.lib.ingest_document(actor, {"title": "民法典"}, ARTICLES, b"x", URL)
            self.assertEqual(err.exception.status, 403)
        row = self.ingest()
        self.assertEqual(self.lib.list_documents()["total"], 0)
        with self.assertRaises(self.mod.LawError):
            self.lib.get_version(row["version_id"])
        self.assertEqual(self.lib.stats(include_unpublished=True)["draft_versions"], 1)

    def test_duplicate_is_idempotent_and_preserves_multiple_originals(self):
        row = self.ingest()
        same = self.ingest()
        other = self.lib.ingest_document(ADMIN, {"title": "中华人民共和国 民法典", "domains": ["general"]}, ARTICLES, b"second html wrapper", URL + "?mirror=1")
        self.assertEqual(row["version_id"], same["version_id"])
        self.assertEqual(row["version_id"], other["version_id"])
        self.assertTrue(other["duplicate"])
        detail = self.lib.get_version(row["version_id"], include_unpublished=True)
        self.assertEqual(len(detail["sources"]), 2)
        self.assertEqual(self.lib.raw_source(row["version_id"], include_unpublished=True)["content"], b"official original")

    def test_changed_text_and_date_create_distinct_versions(self):
        first = self.ingest(published_at="2020-05-28")
        second = self.ingest(published_at="2024-01-01")
        third = self.lib.ingest_document(ADMIN, {"title": "中华人民共和国民法典"}, [{"article_no": "第一条", "content": ARTICLES[0]["content"] + "新增规定。"}], b"new raw", URL)
        self.assertEqual(len({first["version_id"], second["version_id"], third["version_id"]}), 3)
        a = self.lib.get_version(first["version_id"], include_unpublished=True)["articles"][0]
        b = self.lib.get_version(second["version_id"], include_unpublished=True)["articles"][0]
        self.assertNotEqual(a["article_uid"], b["article_uid"])

    def test_unknown_can_publish_but_is_excluded_from_retrieval(self):
        row = self.ingest()
        result = self.lib.publish_version(ADMIN, row["version_id"])
        self.assertTrue(result["warning"])
        self.assertEqual(self.lib.search("民事")["total"], 0)
        self.assertEqual(self.lib.create_snapshot()["article_count"], 0)
        self.assertEqual(self.lib.get_version(row["version_id"])["validity"], "unknown")

    def test_effective_requires_review_and_server_attribution(self):
        row = self.ingest(validity="effective", effective_from="2021-01-01")
        with self.assertRaises(self.mod.LawError):
            self.lib.publish_version(ADMIN, row["version_id"])
        self.effective(row["version_id"])
        version = self.lib.get_version(row["version_id"])
        self.assertEqual(version["metadata"]["verification"]["verified_by"], "law-reviewer")
        self.assertTrue(version["metadata"]["verification"]["verified_at"])
        self.assertEqual(self.lib.search("民事")["total"], 1)

    def test_metadata_change_of_published_creates_draft_and_preserves_snapshot(self):
        row = self.ingest()
        self.effective(row["version_id"])
        snap = self.lib.create_snapshot(domain="general", as_of="2025-01-01")
        original = Path(snap["path"]).read_bytes()
        changed = self.lib.set_version_metadata(ADMIN, row["version_id"], {"domains": ["construction"], "effective_from": "2026-01-01"})
        self.assertNotEqual(changed["version_id"], row["version_id"])
        self.assertEqual(changed["status"], "draft")
        self.assertFalse(changed["metadata"].get("verification", {}).get("verified", False))
        self.assertEqual(self.lib.get_version(row["version_id"])["domains"], ["general", "real_estate"])
        self.assertEqual(Path(snap["path"]).read_bytes(), original)
        record = json.loads(original.decode().splitlines()[0])
        self.assertEqual(record["retrieval_profile"], "published_raw_bm25_v1")
        self.assertEqual(record["source_sha256"], hashlib.sha256(b"official original").hexdigest())
        self.assertEqual(record["version_id"], row["version_id"])
        self.assertEqual(record.get("snapshot_id"), snap["snapshot_id"])

    def test_original_integrity_checked_before_publication_and_download(self):
        row = self.ingest()
        version = self.lib.get_version(row["version_id"], include_unpublished=True)
        Path(version["sources"][0]["path"]).write_bytes(b"tampered")
        for action in (lambda: self.lib.publish_version(ADMIN, row["version_id"]), lambda: self.lib.raw_source(row["version_id"], include_unpublished=True)):
            with self.assertRaises(self.mod.LawError): action()

    def test_temporal_region_and_latest_version_scope(self):
        old = self.ingest()
        self.effective(old["version_id"])
        new = self.lib.ingest_document(ADMIN, {"title": "中华人民共和国民法典", "published_at": "2025-12-01", "domains": ["general"]}, [{"article_no": "第一条", "content": "民事关系应当遵守新规定，保护民事主体的合法权益。"}], b"new official", URL)
        self.effective(new["version_id"], effective_from="2026-01-01")
        local = self.lib.ingest_document(ADMIN, {"title": "上海市民事条例", "region": "上海市", "domains": ["general"]}, ARTICLES, b"local raw", URL)
        self.effective(local["version_id"])
        self.assertEqual(self.lib.search("民事", as_of="2025-01-01")["items"][0]["version_id"], old["version_id"])
        self.assertEqual(self.lib.search("民事", as_of="2026-11-23")["total"], 1)
        self.assertEqual(self.lib.search("民事", region="上海市", as_of="2026-11-23")["total"], 2)
        self.assertEqual(self.lib.search('"() OR * -', as_of="2026-11-23")["total"], 0)

    def test_invalid_dates_domains_and_garbage_rejected(self):
        for doc in ({"published_at": "2021-02-30"}, {"domains": ["invented"]}, {"validity": "verified"}, {"effective_from": "2025-01-01", "effective_to": "2024-01-01"}):
            with self.assertRaises(self.mod.LawError): self.ingest(**doc)
        with self.assertRaises(self.mod.LawError):
            self.lib.ingest_document(ADMIN, {"title": "网站首页"}, [{"content": "首页"}], b"garbage", URL)

    def test_verified_abolished_version_usable_only_in_its_historical_period(self):
        row = self.ingest()
        self.lib.set_version_metadata(ADMIN, row["version_id"], {
            "validity": "abolished", "effective_from": "2021-01-01", "effective_to": "2024-01-01",
            "verification": {"verified": True, "evidence": "官方施行及废止文件核对"}})
        self.lib.publish_version(ADMIN, row["version_id"])
        self.assertEqual(self.lib.search("民事", as_of="2023-12-31")["total"], 1)
        self.assertEqual(self.lib.search("民事", as_of="2024-01-01")["total"], 0)
        self.assertEqual(self.lib.search("民事")["total"], 0)

    def test_empty_snapshot_is_real_and_tampered_existing_snapshot_is_rejected(self):
        snap = self.lib.create_snapshot()
        self.assertEqual(Path(snap["path"]).read_bytes(), b"")
        self.assertEqual(snap["article_count"], 0)
        Path(snap["path"]).write_bytes(b"foreign data")
        with self.assertRaises(self.mod.LawError): self.lib.create_snapshot()

    def test_import_preserves_original_groups_and_is_conservative_idempotent(self):
        path = Path(self.tmp.name) / "legacy.jsonl"
        path.write_text(json.dumps({"law_name": "中华人民共和国民法典", "article_no": "第一条", "content": ARTICLES[0]["content"], "effective_from": "2021-01-01", "validity": "effective"}, ensure_ascii=False) + "\n", encoding="utf-8")
        first = self.lib.import_jsonl(ADMIN, path)
        second = self.lib.import_jsonl(ADMIN, path)
        self.assertEqual(first["imported"], 1)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(self.lib.stats()["published_versions"], 0)
        version = self.lib.list_documents(include_unpublished=True)["items"][0]["current_version"]
        self.assertEqual(self.lib.raw_source(version["version_id"], include_unpublished=True)["content"], path.read_bytes())

    def test_structured_extraction_quality_retained_for_provenance(self):
        quality = {"method": "source_package_extracted_text", "human_verified": False}
        row = self.ingest(extraction_quality=quality)
        version = self.lib.get_version(row["version_id"], include_unpublished=True)
        self.assertEqual(version["metadata"]["extraction_quality"], quality)

    def test_publish_conflicts_if_metadata_changes_after_original_verification(self):
        row = self.ingest()
        original = self.lib.raw_source
        def mutate_after_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            self.lib.set_version_metadata(ADMIN, row["version_id"], {"validity": "effective", "effective_from": "2021-01-01"})
            return result
        with patch.object(self.lib, "raw_source", side_effect=mutate_after_validation):
            with self.assertRaises(self.mod.LawError) as err:
                self.lib.publish_version(ADMIN, row["version_id"])
        self.assertEqual(err.exception.status, 409)
        self.assertEqual(self.lib.get_version(row["version_id"], include_unpublished=True)["status"], "draft")

    def test_retrieval_defensively_excludes_published_effective_without_verification(self):
        row = self.ingest(validity="effective", effective_from="2021-01-01")
        with self.lib._db() as db:
            db.execute("UPDATE versions SET status='published' WHERE id=?", (row["version_id"],))
        self.assertEqual(self.lib.search("民事")["total"], 0)

    def test_migration_download_never_leaks_another_unpublished_law(self):
        path = Path(self.tmp.name) / "mixed.jsonl"
        first = {"law_name": "公开民事法规", "article_no": "第一条", "content": ARTICLES[0]["content"]}
        second = {"law_name": "未发布草稿法规", "article_no": "第一条", "content": "这段仅供管理人员核对的法规草稿内容不得从其他已发布法规的原件下载。"}
        lines = [(json.dumps(record, ensure_ascii=False) + "\n").encode() for record in (first, second)]
        raw = b"".join(lines)
        path.write_bytes(raw)
        self.lib.import_jsonl(ADMIN, path)
        row = self.lib.list_documents(query="公开民事", include_unpublished=True)["items"][0]
        version_id = row["current_version"]["version_id"]
        self.lib.publish_version(ADMIN, version_id)
        download = self.lib.raw_source(version_id)
        self.assertEqual(download["content"], lines[0])
        self.assertNotIn(second["law_name"].encode(), download["content"])
        version = self.lib.get_version(version_id)
        self.assertEqual(version["metadata"]["original_archive_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(version["metadata"]["source_kind"], "derived_migration_excerpt")
        self.assertEqual((self.lib.raw_dir / (hashlib.sha256(raw).hexdigest() + ".bin")).read_bytes(), raw)

    def test_snapshot_write_failure_leaves_version_unpublished(self):
        row = self.ingest()
        blocker = Path(self.tmp.name) / "not-a-directory"
        blocker.write_bytes(b"block snapshot creation")
        self.lib.snapshot_dir = blocker
        with self.assertRaises(OSError): self.lib.publish_version(ADMIN, row["version_id"])
        self.assertEqual(self.lib.get_version(row["version_id"], include_unpublished=True)["status"], "draft")

    def test_corrected_effective_date_supersedes_old_metadata_at_all_times(self):
        row = self.ingest()
        self.effective(row["version_id"], effective_from="2024-01-01")
        before = self.lib.create_snapshot()
        old_bytes = Path(before["path"]).read_bytes()
        corrected = self.lib.set_version_metadata(ADMIN, row["version_id"], {
            "effective_from": "2022-01-01", "verification": {"verified": True, "evidence": "更正原版施行日期录入错误"}})
        self.lib.publish_version(ADMIN, corrected["version_id"])
        for as_of in (None, "2023-01-01", "2025-01-01"):
            self.assertEqual(self.lib.search("民事", as_of=as_of)["items"][0]["version_id"], corrected["version_id"])
        self.assertEqual(Path(before["path"]).read_bytes(), old_bytes)

    def test_identical_titles_from_different_regions_and_issuers_are_separate_laws(self):
        rows = []
        for region, issuer in (("甲省", "甲省人大"), ("乙省", "乙省人大")):
            row = self.lib.ingest_document(ADMIN, {"title": "公共资源交易管理办法", "region": region, "issuing_authority": issuer}, ARTICLES, region.encode(), URL)
            self.effective(row["version_id"])
            rows.append(row)
        self.assertNotEqual(rows[0]["document_id"], rows[1]["document_id"])
        self.assertEqual(self.lib.search("民事", region="甲省")["total"], 1)
        self.assertEqual(self.lib.search("民事", region="乙省")["total"], 1)
        self.assertEqual(self.lib.search("民事")["total"], 0)

    def test_migration_skips_heading_units_and_keeps_official_article_provenance(self):
        path = Path(self.tmp.name) / "sections.jsonl"
        source = "https://flk.npc.gov.cn/detail?id=example"
        rows = [
            {"law_name": "政府采购履约验收指导意见", "article_no": "四、", "content": "四、工作要求", "location": {"unit_kind": "section", "line_start": 15}},
            {"law_name": "政府采购履约验收指导意见", "article_no": "（十）", "content": "采购人应当切实做好需求编制和履约验收工作，完善内部机制、强化内部监督。", "location": {"unit_kind": "item", "line_start": 16}, "source_url": source, "original_sha256": "a" * 64}]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
        result = self.lib.import_jsonl(ADMIN, path)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["skipped_structural_units"], 1)
        version_id = self.lib.list_documents(include_unpublished=True)["items"][0]["current_version"]["version_id"]
        self.effective(version_id)
        record = self.lib.search("采购人")["items"][0]
        self.assertEqual(record["location"]["section_heading"], "四、工作要求")
        self.assertEqual(record["source_url"], source)
        self.assertEqual(record["official_original_sha256"], "a" * 64)
        self.assertEqual(record["original_sha256"], "a" * 64)
        self.assertEqual(record["migration_source_sha256"], self.lib.get_version(version_id)["source_sha256"])

    def test_required_nonempty_snapshot_rejects_without_writing_empty_artifact(self):
        with self.assertRaises(self.mod.LawError) as err:
            self.lib.create_snapshot(require_articles=True)
        self.assertEqual(err.exception.status, 409)
        self.assertEqual(list(self.lib.snapshot_dir.iterdir()), [])


if __name__ == "__main__": unittest.main()
