"""Whole-contract context screening preserves exact evidence and abstains on gaps."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.fact_slots import extract_fact_slots
from zhilv_qiling_v2.models import Clause, Contract, DocumentParse

try:
    from zhilv_qiling_v2.context_review import evaluate_contract_context
except ImportError:
    evaluate_contract_context = None


PAYMENT = "本设备最终验收合格后支付尾款。"
DEEMED = "本设备收货后3日内未提出书面异议视为最终验收合格。"
TEST = "本设备最终验收必须包含性能测试，该测试最早在收货后第5日完成。"


def contract_of(texts, *, metadata=None, full_text=None):
    clauses = []
    for index, text in enumerate(texts):
        source = ("main", "annex-a", "annex-b")[index] if index < 3 else "other-" + str(index)
        meta = {"source_id": source, "document_id": "doc-" + source,
                "source_title": ("主合同", "附件A", "附件B")[index] if index < 3 else "补充条款",
                "source_role": "main_contract" if index == 0 else "attachment",
                "source_page_start": index + 1, "original_sha256": str(index) * 64,
                "extraction_quality": {"status": "text_extracted", "human_verified": False}}
        meta.update((metadata or {}).get(index, {}))
        clauses.append(Clause("c" + str(index), "", "", 1, "", text, text,
                              page_start=index + 1, metadata=meta))
    document = DocumentParse("测试合同包", "contract_package",
        "\n".join(texts) if full_text is None else full_text, clauses=clauses)
    return Contract("context-fixture", document)


class ContractContextReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(evaluate_contract_context),
                        "the contract-context review entry point must exist")

    def temporal(self, result):
        return [row for row in result["findings"] if row["kind"] == "acceptance_before_required_test"]

    def test_three_sources_produce_one_traceable_temporal_finding_and_receipt(self):
        contract = contract_of([PAYMENT, DEEMED, TEST])
        result = evaluate_contract_context(contract)
        self.assertEqual(result["schema_version"], "contract-context-review-v1")
        self.assertEqual(result["execution_status"], "completed")
        finding, = self.temporal(result)
        self.assertEqual(finding["status"], "issue_found")
        self.assertEqual(len(finding["evidence"]), 3)
        self.assertEqual({e["quote"] for e in finding["evidence"]}, {PAYMENT, DEEMED, TEST})
        for evidence in finding["evidence"]:
            source = next(c for c in contract.document.clauses if c.clause_id == evidence["clause_id"])
            self.assertEqual(source.text[evidence["start"]:evidence["end"]], evidence["quote"])
            self.assertEqual(evidence["source_sha256"], hashlib.sha256(source.text.encode()).hexdigest())
            for field in ("source_id", "document_id", "source_title", "source_role", "source_page_start",
                          "original_sha256", "extraction_quality"):
                self.assertEqual(evidence[field], source.metadata[field])
        self.assertGreaterEqual(len(finding["reasoning_steps"]), 4)
        self.assertTrue(all({"step_id", "claim", "premise_ids"} <= step.keys()
                            for step in finding["reasoning_steps"]))
        self.assertTrue(finding["manual_required"])
        self.assertTrue(finding["assumptions"])
        self.assertIn("5", finding["actual_issue"])
        receipt = next(row for row in result["checks"] if row["status"] == "issue_found")
        self.assertEqual(receipt["point_id"], "performance.acceptance")
        self.assertTrue(receipt["scope_complete"])
        self.assertFalse(receipt["legal_truth_established"])
        self.assertEqual(result["completeness"]["status"], "not_evaluated")
        self.assertEqual(result["summary"]["clause_count"], 3)
        self.assertEqual(result["summary"]["source_count"], 3)
        json.dumps(result, ensure_ascii=False)

    def test_removing_any_premise_never_confirms_the_same_combination_or_passes(self):
        for index in range(3):
            with self.subTest(removed=index):
                texts = [PAYMENT, DEEMED, TEST]
                del texts[index]
                result = evaluate_contract_context(contract_of(texts))
                self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
                self.assertFalse(any(c["status"] == "no_issue_detected" for c in result["checks"]))

    def test_different_object_or_acceptance_phase_is_not_confirmed(self):
        alternatives = [TEST.replace("本设备", "设备B"), TEST.replace("最终验收", "初步验收"),
                        DEEMED.replace("最终验收", "初步验收")]
        for changed in alternatives:
            with self.subTest(changed=changed):
                texts = [PAYMENT, changed, TEST] if "视为" in changed else [PAYMENT, DEEMED, changed]
                result = evaluate_contract_context(contract_of(texts))
                self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
                self.assertTrue(any(c["status"] == "needs_manual" and c["pending_reasons"] for c in result["checks"]))

    def test_unbound_scope_keeps_evidenced_pending_lead(self):
        result = evaluate_contract_context(contract_of([
            "最终验收合格后支付尾款。", "收货后3日内未提出书面异议视为最终验收合格。",
            "最终验收必须包含性能测试，该测试最早在收货后第5日完成。",]))
        finding, = self.temporal(result)
        self.assertEqual(finding["status"], "needs_manual")
        self.assertEqual(len(finding["evidence"]), 3)
        self.assertTrue(any("标的" in reason or "范围" in reason for reason in finding["pending_reasons"]))

    def test_exception_priority_revision_and_negation_cannot_be_flattened(self):
        for suffix in ("但性能测试完成前不视为最终验收合格。", "本附件优先于主合同适用。",
                       "本条修订此前验收约定。", "另有约定的除外。"):
            with self.subTest(suffix=suffix):
                finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED + suffix, TEST])))
                self.assertEqual(finding["status"], "needs_manual")
                self.assertTrue(finding["pending_reasons"])
                self.assertTrue(any(suffix in e["quote"] for e in finding["evidence"]))
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED,
            TEST.replace("必须包含", "无需包含")]))
        self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))

    def test_uncertain_trigger_and_incompatible_day_units_remain_pending(self):
        for deemed in (DEEMED.replace("3日", "3个工作日"), DEEMED.replace("收货后", "通知后"),
                       DEEMED.replace("3日", "若干日")):
            with self.subTest(deemed=deemed):
                finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, deemed, TEST])))
                self.assertEqual(finding["status"], "needs_manual")
                self.assertTrue(finding["pending_reasons"])

    def test_scanned_source_requires_human_verification(self):
        metadata = {1: {"extraction_quality": {"status": "machine_ocr_unverified", "human_verified": False}}}
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST], metadata=metadata))
        finding, = self.temporal(result)
        self.assertEqual(finding["status"], "needs_manual")
        self.assertTrue(any("OCR" in reason for reason in finding["pending_reasons"]))
        metadata[1]["extraction_quality"]["human_verified"] = True
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST], metadata=metadata)))
        self.assertEqual(finding["status"], "issue_found")

    def test_duplicate_clause_ids_block_affected_findings(self):
        contract = contract_of([PAYMENT, DEEMED, TEST])
        contract.document.clauses[1].clause_id = "c0"
        result = evaluate_contract_context(contract)
        self.assertTrue(any(v["kind"] == "duplicate_clause_id" for v in result["constraints"]["violations"]))
        self.assertFalse(self.temporal(result))
        self.assertGreater(result["summary"]["constraint_violation_count"], 0)
        self.assertTrue(any(c["status"] == "needs_manual" for c in result["checks"]))

    def test_stale_clause_or_omitted_relevant_full_text_cannot_confirm(self):
        contract = contract_of([PAYMENT, DEEMED, TEST], full_text="\n".join([PAYMENT, DEEMED]))
        result = evaluate_contract_context(contract)
        self.assertFalse(self.temporal(result))
        self.assertTrue(result["constraints"]["violations"])
        contract = contract_of([PAYMENT, DEEMED, TEST], full_text="\n".join([PAYMENT, DEEMED, TEST, "本设备验收以补充协议为准。 "]))
        result = evaluate_contract_context(contract)
        self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
        self.assertTrue(result["constraints"]["violations"])

    def test_forged_extractor_quote_is_rejected_before_finding_generation(self):
        def forged(clause_id, text, **kwargs):
            slots = extract_fact_slots(clause_id, text, **kwargs)
            slots["sentences"][0]["span"]["quote"] = "伪造引文"
            return slots
        with patch("zhilv_qiling_v2.context_review.extract_fact_slots", side_effect=forged):
            result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST]))
        self.assertFalse(result["findings"])
        self.assertTrue(any(v["kind"] == "source_binding_invalid" for v in result["constraints"]["violations"]))

    def test_graph_edges_with_missing_endpoints_are_reported_and_not_consumed(self):
        from zhilv_qiling_v2.proposition_graph import build_proposition_graph
        def changed_graph(*args, **kwargs):
            graph = build_proposition_graph(*args, **kwargs)
            graph["edges"].append({"kind": "explicit_prerequisite", "from": "forged", "to": "missing", "resolved": True})
            return graph
        with patch("zhilv_qiling_v2.context_review.build_proposition_graph", side_effect=changed_graph):
            result = evaluate_contract_context(contract_of(["甲方交付本设备。 "]))
        self.assertTrue(any(v["kind"] == "edge_endpoint_invalid" for v in result["constraints"]["violations"]))
        self.assertFalse(result["findings"])

    def test_graph_partition_mismatch_and_cycle_produce_source_bound_findings(self):
        cases = [(["甲方应支付本合同总价100000元，由预付款30000元和尾款80000元组成。"], "explicit_partition_mismatch"),
                 (["甲方在乙方完成本合同最终验收后支付尾款70000元。",
                   "乙方在甲方支付本合同尾款70000元后完成最终验收。"], "literal_prerequisite_cycle")]
        for texts, kind in cases:
            with self.subTest(kind=kind):
                result = evaluate_contract_context(contract_of(texts))
                finding, = [f for f in result["findings"] if f["kind"] == kind]
                self.assertTrue(finding["evidence"])
                self.assertTrue(finding["reasoning_steps"])
                self.assertTrue(all(e["source_id"] for e in finding["evidence"]))
                self.assertFalse(finding["legal_truth_established"])

    def test_empty_irrelevant_and_arithmetic_consistency_never_claim_composite_pass(self):
        for texts in ([], ["双方诚实信用履行。"], ["甲方应支付本合同总价100000元，由预付款30000元和尾款70000元组成。"]):
            with self.subTest(texts=texts):
                result = evaluate_contract_context(contract_of(texts))
                self.assertFalse(result["findings"])
                self.assertFalse(any(c["status"] == "no_issue_detected" for c in result["checks"]))

    def test_completeness_callback_receives_explicit_context_and_is_isolated(self):
        contract = contract_of([PAYMENT, DEEMED, TEST])
        seen = []
        def callback(actual_contract, *, context):
            seen.append((actual_contract, context))
            return {"status": "needs_manual", "pending_reasons": ["缺少人工指定材料"]}
        result = evaluate_contract_context(contract, completeness_check=callback)
        self.assertIs(seen[0][0], contract)
        self.assertEqual(seen[0][1]["schema_version"], "contract-context-review-v1")
        self.assertEqual(result["completeness"]["status"], "needs_manual")
        self.assertEqual(result["completeness"]["execution_status"], "completed")
        self.assertEqual(len(self.temporal(result)), 1)

    def test_completeness_error_and_invalid_return_do_not_crash_or_claim_complete(self):
        def broken(contract, *, context):
            raise RuntimeError("offline fixture failure")
        for callback in (broken, lambda contract, **kwargs: {"unexpected": True},
                         lambda contract, **kwargs: {"status": "completed", "bad": object()}):
            with self.subTest(callback=callback):
                result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST]), completeness_check=callback)
                self.assertEqual(result["execution_status"], "completed")
                self.assertEqual(result["completeness"]["status"], "error")
                self.assertTrue(result["completeness"]["pending_reasons"])
                json.dumps(result)

    def test_evaluation_does_not_mutate_contract(self):
        contract = contract_of([PAYMENT, DEEMED, TEST])
        before = copy.deepcopy(contract.to_dict())
        evaluate_contract_context(contract)
        self.assertEqual(contract.to_dict(), before)

    def test_multiple_premises_inside_one_clause_keep_three_distinct_exact_spans(self):
        contract = contract_of([PAYMENT + DEEMED + TEST])
        finding, = self.temporal(evaluate_contract_context(contract))
        self.assertEqual(finding["status"], "issue_found")
        self.assertEqual([e["quote"] for e in finding["evidence"]], [PAYMENT, DEEMED, TEST])
        self.assertEqual([e["start"] for e in finding["evidence"]], [0, len(PAYMENT), len(PAYMENT + DEEMED)])

    def test_later_default_acceptance_sentence_in_same_clause_is_not_silently_skipped(self):
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED.replace("3日", "8日") + DEEMED, TEST]))
        self.assertTrue(any(f["status"] == "issue_found" for f in self.temporal(result)))

    def test_earliest_completion_is_not_replaced_with_an_earlier_test_start_time(self):
        changed_test = TEST.replace("该测试最早", "该测试在收货后第1日启动，最早")
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED, changed_test])))
        self.assertEqual(finding["status"], "issue_found")
        self.assertIn("第5日", finding["actual_issue"])

    def test_default_deadline_is_bound_to_objection_condition_not_first_unrelated_time(self):
        changed_deemed = DEEMED.replace("本设备收货后", "本设备收货后10日内安装，收货后")
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, changed_deemed, TEST])))
        self.assertEqual(finding["status"], "issue_found")
        self.assertIn("3日", finding["actual_issue"])

    def test_optional_testing_does_not_establish_a_necessary_test_conflict(self):
        changed_test = TEST.replace("必须包含", "可以选择包含")
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED, changed_test])))
        self.assertEqual(finding["status"], "needs_manual")
        self.assertTrue(any("必要" in reason or "必须" in reason for reason in finding["pending_reasons"]))

    def test_callback_cannot_promote_legal_truth_even_in_nested_output(self):
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST]),
            completeness_check=lambda contract, **kwargs: {"status": "completed", "details": {"legal_truth_established": True}})
        self.assertEqual(result["completeness"]["status"], "error")

    def test_explicit_later_default_acceptance_is_not_reported_as_a_conflict(self):
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED.replace("3日", "8日"), TEST]))
        self.assertFalse(self.temporal(result))

    def test_invalid_duplicate_sources_remain_in_provenance_inventory(self):
        contract = contract_of([PAYMENT, DEEMED, TEST])
        contract.document.clauses[1].clause_id = "c0"
        result = evaluate_contract_context(contract)
        self.assertEqual(result["summary"]["source_count"], 3)
        self.assertEqual({s["source_id"] for s in result["source_documents"]}, {"main", "annex-a", "annex-b"})

    def test_later_or_equal_default_is_not_falsely_called_early_even_when_ocr_is_pending(self):
        metadata = {1: {"extraction_quality": {"status": "machine_ocr_unverified", "human_verified": False}}}
        for deemed in (DEEMED.replace("3日", "5日"), DEEMED.replace("3日", "8日")):
            with self.subTest(deemed=deemed):
                result = evaluate_contract_context(contract_of([PAYMENT, deemed, TEST], metadata=metadata))
                self.assertFalse(self.temporal(result))
                self.assertTrue(any(c["status"] == "needs_manual" and any("OCR" in r for r in c["pending_reasons"])
                                    for c in result["checks"]))

    def test_summary_counts_manual_receipts_and_unperformed_completeness(self):
        result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST]))
        self.assertGreaterEqual(result["summary"]["pending_count"], 2)

    def test_final_inference_retains_all_three_premises_when_exception_evidence_is_added(self):
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED + "另有约定的除外。", TEST])))
        self.assertGreater(len(finding["evidence"]), 3)
        final = finding["reasoning_steps"][-1]
        self.assertTrue({"step_1", "step_2", "step_3"} <= set(final["premise_ids"]))

    def test_separate_same_object_exception_is_preserved_in_temporal_evidence_chain(self):
        exception = "本设备性能测试未完成前，不触发最终验收及尾款支付。"
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST, exception])))
        self.assertEqual(finding["status"], "needs_manual")
        self.assertTrue(any(e["quote"] == exception for e in finding["evidence"]))

    def test_negated_or_prohibited_payment_does_not_establish_acceptance_payment_trigger(self):
        for payment in (
            "本设备最终验收合格后不支付尾款。",
            "本设备最终验收合格后禁止支付尾款。",
            "本设备最终验收合格后不再支付尾款。",
            "本设备最终验收合格后不得支付尾款。",
            "本设备最终验收合格后无须付款，尾款另议。",
            "本设备最终验收合格后无需支付尾款。",
            "本设备最终验收合格后并非支付尾款的前提。",
            "本设备最终验收合格后非支付尾款前提。",
            "本设备最终验收合格后不作为支付尾款的前提。",
        ):
            with self.subTest(payment=payment):
                result = evaluate_contract_context(contract_of([payment, DEEMED, TEST]))
                self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
                self.assertTrue(any(c["status"] == "needs_manual" for c in result["checks"]))

    def test_absence_of_objection_is_not_treated_as_negative_payment(self):
        finding, = self.temporal(evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST])))
        self.assertEqual(finding["status"], "issue_found")

    def test_invalid_exception_source_metadata_cannot_strengthen_remaining_three_premises(self):
        exception = "本设备性能测试未完成前，不触发最终验收及尾款支付。"
        metadata = {
            3: {"source_id": "shared", "document_id": "shared", "source_title": "one", "original_sha256": "a" * 64},
            4: {"source_id": "shared", "document_id": "shared", "source_title": "two", "original_sha256": "b" * 64},
        }
        contract = contract_of([PAYMENT, DEEMED, TEST, "双方由项目经理负责日常联系。", exception], metadata=metadata)
        result = evaluate_contract_context(contract)
        self.assertTrue(any(v["kind"] == "source_metadata_conflict" for v in result["constraints"]["violations"]))
        self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
        self.assertTrue(any(v["kind"] == "full_text_coverage_gap" for v in result["constraints"]["violations"]))

    def test_invalid_exception_quote_cannot_strengthen_remaining_three_premises(self):
        exception = "本设备性能测试未完成前，不触发最终验收及尾款支付。"
        def forged(clause_id, text, **kwargs):
            slots = extract_fact_slots(clause_id, text, **kwargs)
            if clause_id == "c3":
                slots["sentences"][0]["span"]["quote"] = "伪造例外引文"
            return slots
        with patch("zhilv_qiling_v2.context_review.extract_fact_slots", side_effect=forged):
            result = evaluate_contract_context(contract_of([PAYMENT, DEEMED, TEST, exception]))
        self.assertTrue(any(v["kind"] == "source_binding_invalid" for v in result["constraints"]["violations"]))
        self.assertFalse(any(f["status"] == "issue_found" for f in self.temporal(result)))
        self.assertTrue(any(v["kind"] == "full_text_coverage_gap" for v in result["constraints"]["violations"]))

    def test_graph_cycle_retains_separate_priority_and_negative_prerequisite_evidence(self):
        cycle = ("甲方在乙方完成本合同最终验收后支付尾款70000元。"
                 "乙方在甲方支付本合同尾款70000元后完成最终验收。")
        qualification = "本合同付款顺序以本条为准：尾款先行支付，无需以最终验收为前提。"
        source = {"source_id": "main", "document_id": "doc-main", "source_title": "main",
                  "source_role": "main_contract", "original_sha256": "a" * 64}
        contract = contract_of([cycle, qualification], metadata={0: source, 1: source})
        result = evaluate_contract_context(contract)
        finding, = [f for f in result["findings"] if f["kind"] == "literal_prerequisite_cycle"]
        self.assertEqual(finding["status"], "needs_manual")
        self.assertTrue(finding["pending_reasons"])
        self.assertTrue(any(e["clause_id"] == "c1" and e["quote"] == qualification for e in finding["evidence"]))
        self.assertFalse(finding["legal_truth_established"])


if __name__ == "__main__":
    unittest.main()
