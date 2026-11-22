"""Contract revisions cannot inherit an authorization from another document."""
from datetime import date, timedelta
import unittest
from unittest.mock import patch

from zhilv_qiling_v2 import pipeline, subject_facts
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import ContractElements, DocumentParse, stable_id


class ContractContentBindingTests(unittest.TestCase):
    def build(self, text):
        document = DocumentParse("same-contract.txt", "txt", text)
        with patch.object(pipeline, "extract", return_value=(ContractElements(), [], {})):
            return pipeline.build_contract(document, Config())

    def authorization(self, contract_id):
        return subject_facts.validate_fact({
            "subject_name": "合成测试公司", "subject_id": "91310101TEST000001",
            "fact_type": "authorization", "status": "valid", "contract_id": contract_id,
            "valid_until": (date.today() + timedelta(days=30)).isoformat(),
            "source": {"url": "https://fixture.invalid/authorization", "publisher": "测试来源",
                       "reviewer": "测试核对人", "verification_method": "离线协议测试",
                       "verified_at": date.today().isoformat()}})

    def resolve(self, contract, record):
        return subject_facts.resolve_subject("合成测试公司", [record],
            contract_id=contract.contract_id, subject_ids=["91310101TEST000001"])

    def test_same_name_same_length_revision_cannot_reuse_authorization(self):
        original = self.build("合同价款1000元。")
        revised = self.build("合同价款9000元。")
        self.assertNotEqual(original.contract_id, revised.contract_id)
        record = self.authorization(original.contract_id)
        self.assertEqual(self.resolve(original, record)["checks"]["authorization"], "valid")
        result = self.resolve(revised, record)
        self.assertEqual(result["checks"]["authorization"], "unknown")
        self.assertIn("different_contract", result["records"][0]["review_reasons"])

    def test_exact_content_replay_keeps_identity(self):
        first, replay = self.build("合同价款1000元。"), self.build("合同价款1000元。")
        self.assertEqual(first.contract_id, replay.contract_id)
        self.assertEqual(first.document.stats["contract_identity"], replay.document.stats["contract_identity"])

    def test_legacy_filename_length_authorization_needs_rebinding(self):
        contract = self.build("合同价款1000元。")
        legacy_id = stable_id("ct", contract.document.source_name, len(contract.document.full_text))
        record = self.authorization(legacy_id)
        result = self.resolve(contract, record)
        self.assertEqual(result["checks"]["authorization"], "unknown")
        self.assertIn("different_contract", result["records"][0]["review_reasons"])

    def test_end_of_long_document_is_also_bound(self):
        prefix = "本合同其他条款。" * 10000
        self.assertNotEqual(self.build(prefix + "1000").contract_id,
                            self.build(prefix + "9000").contract_id)


if __name__ == "__main__":
    unittest.main()
