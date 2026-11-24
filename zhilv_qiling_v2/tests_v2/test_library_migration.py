"""Current workspace payloads retain nested source text during migration."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from zhilv_qiling_v2.contract_library import ContractLibrary


class LibraryMigrationTests(unittest.TestCase):
    def test_nested_document_full_text_is_the_migrated_readonly_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy.sqlite"
            with closing(sqlite3.connect(legacy)) as db, db:
                db.execute("CREATE TABLE workspace_reports(report_id TEXT,payload TEXT)")
                db.execute("INSERT INTO workspace_reports VALUES(?,?)", ("old-report", json.dumps({
                    "source_name": "原合同.docx", "document": {"full_text": "第一条 原始合同正文。"},
                    "risks": [], "score": 90})))
            store = ContractLibrary(root / "new.sqlite")
            admin = store.bootstrap_admin("admin", "Only-test-password-27364!")
            self.assertEqual(store.import_legacy(admin, legacy)["imported"], 1)
            contract = store.list_contracts(admin)["items"][0]
            version = store.get_version(admin, contract["contract_id"])
            self.assertEqual(version["content"].decode(), "第一条 原始合同正文。")
            self.assertTrue(contract["legacy_original_unavailable"])


if __name__ == "__main__":
    unittest.main()
