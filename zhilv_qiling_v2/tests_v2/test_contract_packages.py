"""Contract-package provenance, scope and offline archive regression checks."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from zhilv_qiling_v2.contract_packages import (
    _annotate_source_context_groups, _normative_units, _safe, import_contract_package, load_contract_package,
    load_package_statutes, package_source_path, prepare_package_document,
    search_package_background,
)


class ContractPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "input"
        self.store = Path(self.temp.name) / "store"
        self.root.mkdir()
        self.entries = []

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, value, origin="locally_derived"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
        self.entries.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "origin": origin, "source_url": "https://fixture.invalid/evidence"})
        return path

    def fixture(self):
        import pymupdf
        def pdf(relative, text=None):
            doc = pymupdf.open()
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
            payload = doc.tobytes()
            doc.close()
            self.write(relative, payload, "public_source")
        pdf("01_合同文本/01_原合同/source/2026年年度人口抽样调查合同.pdf")
        self.write("01_合同文本/01_原合同/原合同_OCR框.json", json.dumps([
            {"page": 1, "lines": [{"text": "第一条 原合同付款期限为20个工作日。", "min_score": .9},
                                  {"text": "第1页共1页", "min_score": 1}]}], ensure_ascii=False))
        pdf("01_合同文本/02_补充协议/source/2026年年度人口抽样调查--补充协议10.29.pdf")
        self.write("01_合同文本/02_补充协议/补充协议_OCR框.json", json.dumps([
            {"page": 1, "lines": [{"text": "第一条 补充约定验收后20_日内支付。", "min_score": .8}]}], ensure_ascii=False))
        pdf("02_采购交易背景/03_招标文件/source/人口抽样调查物资采购招标文件.pdf", "AUTHORIZATION TEMPLATE unsigned")
        self.write("01_合同文本/02_补充协议/补充协议_全文_校核转录.md", "审查结论：虚构结论不能进入正文")
        self.write("04_公开法条/01_测试法/source/source.html", "<html>fixture law</html>", "public_source")
        self.write("04_公开法条/01_测试法/全文.txt", "测试法\n第一条 本法适用于测试。\n第二条 应保留原始证据。")
        self.write("04_公开法条/01_测试法/meta.json", json.dumps({"名称": "测试法", "原始文件": "source/source.html", "条文数": 2}, ensure_ascii=False))
        (self.root / "05_采集与校验").mkdir()
        (self.root / "05_采集与校验/provenance.json").write_text(
            json.dumps({"entries": self.entries}, ensure_ascii=False), encoding="utf-8")

    def test_full_import_roundtrip_scope_and_independent_source_pages(self):
        self.fixture()
        summary = import_contract_package(self.root, self.store, "fixture")
        self.assertEqual(summary["coverage"]["contract_pages"], 2)
        self.assertEqual(summary["statute_count"], 2)
        parsed = prepare_package_document("fixture", store_root=self.store)
        self.assertNotIn("虚构结论", parsed.full_text)
        self.assertNotIn("TEMPLATE", parsed.full_text)
        self.assertIn("20_日", parsed.full_text)
        self.assertNotIn("第1页共1页", parsed.full_text)
        self.assertEqual([b.page for b in parsed.blocks], [1, 2])
        self.assertEqual([b.metadata["source_page"] for b in parsed.blocks], [1, 1])
        self.assertEqual(len({c.clause_id for c in parsed.clauses}), len(parsed.clauses))
        self.assertEqual(len({b.metadata["source_id"] for b in parsed.blocks}), 2)
        context = parsed.stats["contract_package"]
        self.assertEqual(context["manifest_sha256"], summary["manifest_sha256"])
        self.assertTrue(all(d["status"] == "unknown" and not d["auto_retry"] for d in context["missing_documents"]))
        auth = next(d for d in context["missing_documents"] if d["document_id"] == "authorization_documents")
        self.assertEqual(auth["fact_types"], ["authorization"])
        award = next(d for d in context["missing_documents"] if d["document_id"] == "award_notice")
        self.assertEqual(award["fact_types"], [])
        background = search_package_background("fixture", "AUTHORIZATION", store_root=self.store)
        self.assertEqual(background[0]["page"], 1)
        self.assertFalse(background[0]["is_signed_evidence"])
        full = prepare_package_document("fixture", include_tender=True, store_root=self.store)
        self.assertIn("TEMPLATE", full.full_text)
        self.assertEqual(full.stats["pages"], 3)
        # Archive stays usable after removal of the source material directory.
        shutil.rmtree(self.root)
        replay = prepare_package_document("fixture", store_root=self.store)
        self.assertEqual(replay.full_text, parsed.full_text)
        self.assertTrue(package_source_path("fixture", parsed.blocks[0].metadata["source_id"], self.store).exists())

    def test_source_and_archive_tampering_fail_closed(self):
        self.fixture()
        original = self.root / self.entries[0]["path"]
        original.write_bytes(original.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            import_contract_package(self.root, self.store, "fixture")
        self.assertFalse((self.store / "fixture").exists())

    def test_archive_data_hash_and_source_hash_are_verified(self):
        self.fixture()
        import_contract_package(self.root, self.store, "fixture")
        manifest = load_contract_package("fixture", self.store)
        source = self.store / "fixture" / manifest["sources"][0]["stored_path"]
        source.write_bytes(source.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_contract_package("fixture", self.store)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            package_source_path("fixture", manifest["sources"][0]["source_id"], self.store)

    def test_idempotency_and_generated_text_integrity(self):
        self.fixture()
        one = import_contract_package(self.root, self.store, "fixture")
        two = import_contract_package(self.root, self.store, "fixture")
        self.assertEqual(one["manifest_sha256"], two["manifest_sha256"])
        p = self.store / "fixture/documents.json"
        p.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data hash mismatch"):
            prepare_package_document("fixture", store_root=self.store)

    def test_path_traversal_rejected_before_import(self):
        self.fixture()
        for bad in ["../outside", "/tmp/outside", "C:/outside", "foo/../../outside"]:
            with self.subTest(path=bad), self.assertRaises(ValueError):
                _safe(self.root, bad)
        p = self.root / "05_采集与校验/provenance.json"
        p.write_text(json.dumps({"entries": [{"path": "../outside", "sha256": "x"}]}))
        with self.assertRaisesRegex(ValueError, "Unsafe package path"):
            import_contract_package(self.root, self.store, "fixture")

    def test_normative_unit_boundaries_and_original_labels(self):
        text = "===== 第 1 页 / 共 2 页 =====\n第一条 第一条正文。\n第二章 标题\n[第2页]\n第二条 第二条正文。\n附：1.中小企业声明函\n表格模板"
        units = _normative_units(text)
        self.assertEqual([u["number"] for u in units], [1, 2])
        self.assertEqual(units[0]["page_start"], 1)
        self.assertEqual(units[1]["page_start"], 2)
        self.assertNotIn("表格模板", units[-1]["content"])
        notice = _normative_units("一、总体要求\n正文\n（一）具体措施。\n财政部\n2026年1月1日", sections_only=True)
        self.assertEqual([u["article_no"] for u in notice], ["一、", "（一）"])
        self.assertNotIn("财政部", notice[-1]["content"])

    def test_statute_index_can_consume_preserved_provenance(self):
        from zhilv_qiling_v2.config import Config
        from zhilv_qiling_v2.retrieve import StatuteIndex
        self.fixture()
        import_contract_package(self.root, self.store, "fixture")
        statutes = load_package_statutes("fixture", self.store)
        index = StatuteIndex(statutes, Config())
        self.assertEqual(len(index.statutes), 2)
        self.assertEqual(index.statutes[0]["package_id"], "fixture")
        self.assertEqual(index.statutes[0]["applicability_status"], "requires_case_assessment")

    def source_group_fixture(self, source_id="source_a"):
        from zhilv_qiling_v2.models import SourceBlock, Clause, DocumentParse
        texts = ["第一条货物数量", "（一）货物情况", "单位：元", "数量", "序",
                 "货物名称  规格  单价  总价", "鞋套  某制造商", "3  9500  12.00元  114000.00元",
                 "滑屏手套  某制造商", "4  9500  6.30元  59850.00元", "合计：173850.00元",
                 "（二）知识产权要求", "第二条付款方式", "1.预付款100元。", "2.尾款在验收后20_日内支付。",
                 "3.乙方应开票。", "第三条其他约定", "1.本补充条款优先适用。"]
        blocks = [SourceBlock(f"b{n}", n, "paragraph", text, page=1 if n < 8 else 2,
                              metadata={"source_page": 1 if n < 8 else 2}) for n, text in enumerate(texts)]
        clauses = [Clause(f"cl{n}", "", "", 2, "", b.text, b.text, block_ids=[b.block_id])
                   for n, b in enumerate(blocks)]
        parsed = DocumentParse("fixture", "pdf", "\n".join(texts), blocks=blocks, clauses=clauses)
        doc = {"source_id": source_id, "document_id": source_id, "role": "original_contract",
               "sha256": "fixture", "extraction_quality": {"status": "machine_ocr_unverified"}}
        return parsed, doc

    def test_article_groups_preserve_ids_text_and_all_numbered_subclauses(self):
        parsed, doc = self.source_group_fixture()
        original = [(c.clause_id, c.text, c.parent_id) for c in parsed.clauses]
        groups = _annotate_source_context_groups(parsed, doc)
        self.assertEqual(original, [(c.clause_id, c.text, c.parent_id) for c in parsed.clauses])
        payment = next(g for g in groups if g["title"] == "第二条付款方式")
        self.assertEqual(payment["clause_ids"], ["cl12", "cl13", "cl14", "cl15"])
        self.assertIn("20_日", payment["text"])
        self.assertNotIn("第三条", payment["text"])
        self.assertTrue(payment["requires_full_context"])
        for clause in parsed.clauses[12:16]:
            self.assertEqual(clause.metadata["article_group_id"], payment["group_id"])
            self.assertEqual(clause.metadata["article_group_ids"], [payment["group_id"]])

    def test_table_groups_span_pages_stop_before_heading_and_remain_pending(self):
        parsed, doc = self.source_group_fixture()
        table = next(g for g in _annotate_source_context_groups(parsed, doc) if g["kind"] == "ocr_table")
        self.assertEqual((table["source_page_start"], table["source_page_end"]), (1, 2))
        self.assertIn("鞋套", table["text"])
        self.assertIn("12.00元", table["text"])
        self.assertIn("滑屏手套", table["text"])
        self.assertIn("6.30元", table["text"])
        self.assertNotIn("知识产权", table["text"])
        self.assertEqual(table["verification_status"], "needs_manual")
        self.assertEqual(table["amount_verification_status"], "needs_manual")
        self.assertFalse(table["column_alignment_verified"])
        self.assertFalse(table["auto_pass_eligible"])
        self.assertEqual(parsed.clauses[7].metadata["table_group_ids"], [table["group_id"]])

    def test_context_group_ids_do_not_cross_sources_and_narrative_is_not_table(self):
        a, a_doc = self.source_group_fixture("source_a")
        b, b_doc = self.source_group_fixture("source_b")
        a_ids = {g["group_id"] for g in _annotate_source_context_groups(a, a_doc)}
        b_ids = {g["group_id"] for g in _annotate_source_context_groups(b, b_doc)}
        self.assertFalse(a_ids.intersection(b_ids))
        b, b_doc = self.source_group_fixture()
        b.blocks[5].text = "货物名称、规格、单价和总价须相符"
        self.assertFalse(any(g["kind"] == "ocr_table" for g in _annotate_source_context_groups(b, b_doc)))


if __name__ == "__main__":
    unittest.main()
