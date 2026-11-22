"""Display grouping preserves all original obligations and evidence boundaries."""
import copy
import unittest

from zhilv_qiling_v2.review_task_summary import build_task_summary


def fixture():
    items, works = [], []
    for index in range(2):
        fid, cid = f"feedback-{index}", f"clause-{index}"
        items.append({"item_id": fid, "kind": "subject_fact_feedback", "title": "甲公司",
            "status": "manual_review", "manual_required": True, "evidence_status": "evidence_pending",
            "contract_evidence": [{"clause_id": cid, "quote": f"原文{index}", "located": True}]})
        works.append({"work_item_id": f"work-{index}", "legacy_feedback_item_id": fid,
            "source_status": "external_evidence_pending", "semantic_status": "pending_manual",
            "execution_status": "not_executed", "pending": True, "protocol_complete": False,
            "dispatchable": False, "actual_processed": False,
            "provided_feedback": {"subject_node_id": "subject-a", "subject_id": "id-a",
                "subject_name": "甲公司", "subject_role": "seller", "fact_id": "fact-a",
                "fact_type": "authorization", "effective_status": "unknown", "source_id": "source-a",
                "source": {"publisher": "机构A"}, "scope": "本合同销售签约", "source_complete": False,
                "candidate_ids": [], "request_id": "", "clause_id": cid, "clause_ids": [cid]}})
    return items, works


class TaskSummaryTests(unittest.TestCase):
    def test_repeated_fact_keeps_every_id_anchor_and_original_state(self):
        items, works = fixture(); before = copy.deepcopy((items, works))
        result = build_task_summary(items, works, contract_id="contract-a")
        self.assertEqual(result["display_group_count"], 1)
        self.assertEqual(result["raw_item_count"], 2)
        self.assertEqual(result["manual_group_count"], 1)
        self.assertEqual(result["raw_manual_item_count"], 2)
        group = result["groups"][0]
        self.assertEqual(group["item_ids"], ["feedback-0", "feedback-1"])
        self.assertEqual(group["work_item_ids"], ["work-0", "work-1"])
        self.assertEqual(group["clause_ids"], ["clause-0", "clause-1"])
        self.assertEqual(group["status"], "manual_review")
        self.assertFalse(group["legal_truth_established"])
        self.assertEqual((items, works), before)

    def test_sources_scopes_subjects_types_and_states_do_not_merge(self):
        changes = [{"source_id": "source-b"}, {"source": {"publisher": "机构B"}},
            {"scope": "另一合同"}, {"applicable_scope": ["clause-other"]},
            {"valid_until": "2026-12-01"}, {"fact_evidence_text": '{"scope":"另一许可范围"}'},
            {"subject_node_id": "subject-b"}, {"subject_id": "id-b"},
            {"fact_type": "credit"}, {"effective_status": "conflicting_evidence"},
            {"check_states": {"registration": "unknown"}}, {"auto_retry": False}]
        for change in changes:
            with self.subTest(change=change):
                items, works = fixture(); works[1]["provided_feedback"].update(change)
                self.assertEqual(build_task_summary(items, works)["display_group_count"], 2)

    def test_execution_or_semantic_state_stays_separate(self):
        for change in [{"execution_status": "executed"}, {"semantic_status": "pending_next_cycle"},
                       {"protocol_complete": True}, {"actual_processed": True}]:
            with self.subTest(change=change):
                items, works = fixture(); works[1].update(change)
                self.assertEqual(build_task_summary(items, works)["display_group_count"], 2)

    def test_scoped_private_gap_is_part_of_identity(self):
        items, works = fixture()
        for index, work in enumerate(works):
            work["provided_feedback"]["unavailable_evidence"] = [{"document_id": "authorization",
                "clause_ids": [f"clause-{index}"], "availability": "unavailable_private"}]
        self.assertEqual(build_task_summary(items, works)["display_group_count"], 2)

    def test_candidate_specific_and_upstream_requests_stay_individual(self):
        for change in [{"candidate_ids": ["candidate-a"]}, {"request_id": "request-a"}]:
            items, works = fixture()
            for work in works:
                work["provided_feedback"].update(change)
            result = build_task_summary(items, works)
            self.assertEqual(result["display_group_count"], 2)
            self.assertTrue(all(group["kind"] == "original_item" for group in result["groups"]))

    def test_unknown_identity_and_missing_work_never_coalesce(self):
        items, works = fixture()
        for work in works:
            work["provided_feedback"].pop("subject_node_id")
        self.assertEqual(build_task_summary(items, works)["display_group_count"], 2)
        self.assertEqual(build_task_summary(items, [])["display_group_count"], 2)

    def test_other_items_and_confirmations_are_preserved_as_singletons(self):
        items, works = fixture()
        items.append({"item_id": "risk-1", "kind": "risk", "manual_required": True, "status": "manual_review"})
        items.append({"item_id": "checked-1", "kind": "subject_fact_feedback", "manual_required": False,
                      "status": "provided_relation_checked"})
        result = build_task_summary(items, works)
        self.assertEqual(result["raw_item_count"], 4)
        self.assertEqual(result["display_group_count"], 3)
        self.assertEqual(result["raw_manual_item_count"], 3)
        self.assertEqual(result["manual_group_count"], 2)
        self.assertEqual([item for group in result["groups"] for item in group["item_ids"]],
                         [item["item_id"] for item in items])

    def test_empty_and_stable_output(self):
        self.assertEqual(build_task_summary([], [])["display_group_count"], 0)
        items, works = fixture()
        self.assertEqual(build_task_summary(items, works), build_task_summary(items, works))


if __name__ == "__main__":
    unittest.main()
