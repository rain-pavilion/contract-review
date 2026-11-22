"""No candidate is not a passing verdict and not a fabricated risk item."""
from copy import deepcopy
import unittest

from zhilv_qiling_v2 import agents, api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.models import Contract, FinalRisk, ReviewReport
from zhilv_qiling_v2.tests_v2.test_selected_safety_guards import document


class EmptyReviewStatusTests(unittest.TestCase):
    def notice(self):
        return agents.fuse([], [], Contract("fixture", document("测试文本")), Config())[0]

    def test_internal_notice_does_not_assert_complete_semantic_review(self):
        notice = self.notice()
        self.assertEqual(notice.code, "no_substantive_risk")
        self.assertTrue(notice.review_required)
        self.assertIn("不代表审查通过", notice.message)
        self.assertNotIn("未发现实质风险", notice.message)

    def test_api_separates_notice_from_risks_and_preserves_pending(self):
        notice = self.notice()
        report = ReviewReport("fixture", "fixture", risks=[notice],
            review_queue=[{"id": notice.risk_id}, {"id": "unresolved-work", "reason": "unknown"}],
            stats={"cross_validation": {"converged": False, "pending_work_item_ids": ["pending"]}})
        before = deepcopy(report.to_dict())
        payload = api._report_payload(report)
        self.assertEqual(payload["risk_count"], 0)
        self.assertEqual(payload["risks"], [])
        self.assertEqual(payload["severity_counts"], {})
        self.assertEqual(len(payload["notices"]), 1)
        self.assertEqual(payload["review_queue"], [{"id": "unresolved-work", "reason": "unknown"}])
        self.assertFalse(payload["stats"]["cross_validation"]["converged"])
        self.assertEqual(report.to_dict(), before)

    def test_real_risk_is_neither_hidden_nor_reclassified(self):
        risk = FinalRisk(risk_id="risk", code="need_authorization", severity="high", message="授权待核验",
                         review_required=True, category="主体审查")
        payload = api._report_payload(ReviewReport("c", "c", risks=[risk, self.notice()]))
        self.assertEqual(payload["risk_count"], 1)
        self.assertEqual(payload["risks"][0]["risk_id"], "risk")
        self.assertEqual(payload["severity_counts"], {"high": 1})
        self.assertTrue(payload["risks"][0]["review_required"])

    def test_markdown_no_candidate_wording_is_not_a_pass(self):
        for risks in ([], [self.notice()]):
            markdown = ReviewReport("c", "c", risks=risks).to_markdown()
            self.assertIn("不代表审查通过", markdown)
            self.assertNotIn("未发现实质风险", markdown)

    def test_engine_string_queue_keeps_real_work_and_excludes_notice(self):
        notice = self.notice()
        report = ReviewReport("c", "c", risks=[notice], review_queue=[notice.risk_id, "pending-real-work"])
        self.assertEqual(api._report_payload(report)["review_queue"], ["pending-real-work"])
        self.assertIn("pending-real-work", report.to_markdown())


if __name__ == "__main__":
    unittest.main()
