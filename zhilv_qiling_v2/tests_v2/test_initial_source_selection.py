"""Offline protocol tests; source selection proves neither law nor model quality."""
from __future__ import annotations

import copy
import unittest

from zhilv_qiling_v2.initial_review import InitialReviewContinuation, source_sha256
from zhilv_qiling_v2.initial_source_selection import InitialSourceCatalogue, replay_source_selection
from zhilv_qiling_v2 import llm_client as L


POLICY = {"max_findings": 8, "message_max_chars": 160,
          "quote_max_chars": 160, "max_related_quotes": 2}


def window(*texts, window_id="original-window"):
    return {"window_id": window_id, "kind": "cross_clause", "segments": [
        {"clause_id": f"clause-{index}", "start": 10 + index * 100,
         "end": 10 + index * 100 + len(text), "text": text}
        for index, text in enumerate(texts or ("甲方验收后支付尾款。", "乙方收款后开始验收。"))]}


def initial_schema():
    quote = {"type": "string", "minLength": 1, "maxLength": 160}
    fields = {"clause_id": {"type": "string", "enum": ["clause-0", "clause-1"]},
        "code": {"type": "string", "enum": ["cross_clause_inconsistency", "fee_mismatch"]},
        "severity": {"type": "string", "enum": ["high", "medium", "low", "info"]},
        "message": {"type": "string", "minLength": 1, "maxLength": 160},
        "quote": quote, "related_quotes": {"type": "array", "minItems": 0, "maxItems": 2,
            "items": {"type": "object", "required": ["clause_id", "quote"],
                "additionalProperties": False, "properties": {
                    "clause_id": {"type": "string", "enum": ["clause-0", "clause-1"]},
                    "quote": copy.deepcopy(quote)}}}}
    return {"type": "object", "required": ["defects", "more_findings_possible"],
        "additionalProperties": False, "properties": {
            "defects": {"type": "array", "minItems": 0, "maxItems": 8,
                "items": {"type": "object", "required": list(fields),
                    "additionalProperties": False, "properties": fields}},
            "more_findings_possible": {"type": "boolean"}}}


class InitialSourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.window = window()
        self.catalogue = InitialSourceCatalogue(self.window)

    def finding(self, **changes):
        refs = self.catalogue.entries
        row = {"source_ref": refs[0]["source_ref"], "code": "cross_clause_inconsistency",
            "severity": "high", "message": "付款与验收互为前提，需复核条件。",
            "related_source_refs": [refs[1]["source_ref"]]}
        row.update(changes)
        return row

    def response(self, **changes):
        value = {"defects": [self.finding()], "more_findings_possible": True}
        value.update(changes)
        return value

    def resolve(self, value, **kwargs):
        return self.catalogue.resolve(value, window_payload=self.window, **kwargs)

    def test_exact_unicode_offsets_cover_tails_whitespace_and_short_segments(self):
        source = window("甲😀乙\n" * 31 + "尾", " ", "短")
        catalogue = InitialSourceCatalogue(source)
        manifest = catalogue.as_request_data()
        for index, segment in enumerate(source["segments"]):
            entries = [x for x in manifest["extracts"] if x["segment_index"] == index]
            covered = set()
            for entry in entries:
                self.assertEqual(entry["quote"], segment["text"][entry["local_start"]:entry["local_end"]])
                self.assertEqual(entry["clause_start"], segment["start"] + entry["local_start"])
                self.assertEqual(entry["clause_end"], segment["start"] + entry["local_end"])
                self.assertLessEqual(len(entry["quote"]), 80)
                covered.update(range(entry["local_start"], entry["local_end"]))
            self.assertEqual(covered, set(range(len(segment["text"]))))
            self.assertTrue(manifest["coverage"][index]["complete_supplied_segment"])
        self.assertEqual(manifest["extracts"][2]["quote"][-1], "尾")

    def test_empty_segments_are_explicit_without_fake_excerpt(self):
        catalogue = InitialSourceCatalogue(window("", "实际条文"))
        self.assertEqual(catalogue.as_request_data()["coverage"][0]["source_chars"], 0)
        self.assertFalse(any(x["segment_index"] == 0 for x in catalogue.entries))
        with self.assertRaises(ValueError):
            InitialSourceCatalogue(window("")).selection_schema(initial_schema())

    def test_no_gap_configuration_or_inconsistent_offset_is_accepted(self):
        for options in ({"stride_chars": 81}, {"extract_chars": 0}, {"stride_chars": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                InitialSourceCatalogue(self.window, **options)
        for changes in ({"end": 999}, {"start": True}, {"start": -1}, {"clause_id": ""}):
            bad = copy.deepcopy(self.window)
            bad["segments"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                InitialSourceCatalogue(bad)

    def test_same_clause_repeated_occurrences_keep_distinct_references(self):
        source = window("重复", "重复", "重复")
        source["segments"][1] = copy.deepcopy(source["segments"][0])
        source["segments"][2]["clause_id"] = "clause-0"
        catalogue = InitialSourceCatalogue(source)
        self.assertEqual(len({x["source_ref"] for x in catalogue.entries}), 3)
        self.assertEqual({x["segment_index"] for x in catalogue.entries}, {0, 1, 2})
        self.assertEqual({x["quote"] for x in catalogue.entries}, {"重复"})

    def test_deterministic_hashes_bind_window_source_and_excerpt_policy(self):
        same = InitialSourceCatalogue(copy.deepcopy(self.window))
        self.assertEqual(self.catalogue.entries, same.entries)
        self.assertEqual(self.catalogue.catalogue_sha256, same.catalogue_sha256)
        self.assertEqual(self.catalogue.source_sha256, source_sha256(self.window["segments"]))
        other_window = copy.deepcopy(self.window)
        other_window["window_id"] = "other"
        self.assertNotEqual(self.catalogue.entries[0]["source_ref"],
                            InitialSourceCatalogue(other_window).entries[0]["source_ref"])
        self.assertNotEqual(self.catalogue.catalogue_sha256,
                            InitialSourceCatalogue(self.window, extract_chars=40).catalogue_sha256)
        with self.assertRaises(ValueError):
            InitialSourceCatalogue(self.window, expected_source_sha256="incorrect")

    def test_snapshot_and_returned_catalogue_are_defensive_copies(self):
        saved = copy.deepcopy(self.window)
        self.window["segments"][0]["text"] = "changed"
        entries = self.catalogue.entries
        entries[0]["quote"] = "changed"
        request = self.catalogue.as_request_data()
        request["coverage"].clear()
        request["extracts"].clear()
        self.assertEqual(self.catalogue.entries[0]["quote"], saved["segments"][0]["text"])
        self.assertEqual(len(self.catalogue.as_request_data()["coverage"]), 2)

    def test_schema_preserves_legal_options_capacity_and_boolean(self):
        original = initial_schema()
        saved = copy.deepcopy(original)
        schema = self.catalogue.selection_schema(original)
        self.assertEqual(original, saved)
        before = original["properties"]["defects"]
        after = schema["properties"]["defects"]
        self.assertEqual(before["maxItems"], after["maxItems"])
        fields = after["items"]["properties"]
        for key in ("code", "severity", "message"):
            self.assertEqual(fields[key], before["items"]["properties"][key])
        self.assertEqual(fields["related_source_refs"]["maxItems"], 2)
        self.assertNotIn("clause_id", fields)
        self.assertNotIn("quote", fields)
        self.assertEqual(schema["properties"]["more_findings_possible"], {"type": "boolean"})
        L._validate_json_schema(self.response(), schema)

    def test_schema_corrections_use_explicit_known_prior_ids(self):
        schema = self.catalogue.selection_schema(initial_schema(), pending_finding_ids=["repair-1"])
        value = self.response(corrections=[{"prior_finding_id": "repair-1", "defect": self.finding()}])
        L._validate_json_schema(value, schema)
        value["corrections"][0]["prior_finding_id"] = "unknown"
        with self.assertRaises(L.LLMClientError):
            L._validate_json_schema(value, schema)
        empty = self.catalogue.selection_schema(initial_schema(), pending_finding_ids=[])
        self.assertEqual(empty["properties"]["corrections"]["maxItems"], 0)

    def test_smaller_existing_quote_capacity_cannot_be_relaxed(self):
        catalogue = InitialSourceCatalogue(window("甲" * 81))
        schema = initial_schema()
        schema["properties"]["defects"]["items"]["properties"]["quote"]["maxLength"] = 79
        with self.assertRaises(ValueError):
            catalogue.selection_schema(schema)

    def test_exact_resolution_retains_raw_json_and_model_fields_unchanged(self):
        raw = self.response()
        saved = copy.deepcopy(raw)
        result = self.resolve(raw)
        self.assertTrue(result["all_source_refs_resolved"])
        self.assertEqual(result["pending_reasons"], [])
        self.assertEqual(result["raw_content"], saved)
        self.assertEqual(raw, saved)
        defect = result["resolved_content"]["defects"][0]
        self.assertEqual(defect["clause_id"], "clause-0")
        self.assertEqual(defect["quote"], self.window["segments"][0]["text"])
        self.assertEqual(defect["related_quotes"], [{"clause_id": "clause-1",
            "quote": self.window["segments"][1]["text"]}])
        for field in ("code", "severity", "message"):
            self.assertEqual(defect[field], saved["defects"][0][field])
        self.assertEqual([x["output_path"] for x in result["bindings"]],
            ["defects[0].source_ref", "defects[0].related_source_refs[0]"])
        result["raw_content"]["defects"].clear()
        self.assertEqual(raw, saved)

    def test_more_boolean_is_never_rewritten_and_no_completion_is_inferred(self):
        for more in (True, False):
            result = self.resolve(self.response(more_findings_possible=more))
            self.assertIs(result["resolved_content"]["more_findings_possible"], more)
            self.assertNotIn("complete", result)
        for value in ("false", None, 0):
            result = self.resolve(self.response(more_findings_possible=value))
            self.assertEqual(result["resolved_content"]["more_findings_possible"], value)
            self.assertIn("initial_source_selection_root_invalid", result["pending_reasons"])

    def test_unknown_reference_never_guesses_quote_or_discards_finding(self):
        raw = self.response(defects=[self.finding(source_ref="unknown")])
        result = self.resolve(raw)
        self.assertFalse(result["all_source_refs_resolved"])
        self.assertEqual(result["resolved_content"]["defects"], raw["defects"])
        self.assertEqual(result["bindings"], [])
        self.assertTrue(any("unknown_reference" in x for x in result["pending_reasons"]))

    def test_unknown_related_reference_prevents_partial_finding_conversion(self):
        raw = self.response(defects=[self.finding(related_source_refs=["unknown"])])
        result = self.resolve(raw)
        self.assertEqual(result["resolved_content"]["defects"], raw["defects"])
        self.assertEqual(result["bindings"], [])

    def test_changed_source_window_or_external_ref_stays_pending(self):
        changed = copy.deepcopy(self.window)
        changed["segments"][0]["text"] = changed["segments"][0]["text"].replace("甲", "丙")
        for candidate in (changed, {**self.window, "window_id": "other"}):
            result = self.catalogue.resolve(self.response(), window_payload=candidate)
            self.assertFalse(result["all_source_refs_resolved"])
            self.assertEqual(result["resolved_content"], result["raw_content"])
            self.assertEqual(result["bindings"], [])
        external = InitialSourceCatalogue(changed).entries[0]["source_ref"]
        self.assertFalse(self.resolve(self.response(defects=[self.finding(source_ref=external)]))["all_source_refs_resolved"])

    def test_malformed_objects_are_retained_instead_of_silently_dropped(self):
        for defect in (None, {}, self.finding(clause_id="free-field"), self.finding(related_source_refs="bad")):
            raw = self.response(defects=[defect])
            result = self.resolve(raw)
            self.assertFalse(result["all_source_refs_resolved"])
            self.assertEqual(result["resolved_content"]["defects"], [defect])
        for raw in (None, {}, {"defects": "bad", "more_findings_possible": False}):
            result = self.resolve(raw)
            self.assertFalse(result["all_source_refs_resolved"])
            self.assertEqual(result["resolved_content"], raw)

    def test_explicit_corrections_preserve_prior_id_and_bind_provenance(self):
        raw = self.response(defects=[], corrections=[{"prior_finding_id": "repair-1", "defect": self.finding()}])
        result = self.resolve(raw, pending_finding_ids=["repair-1"])
        self.assertTrue(result["all_source_refs_resolved"])
        self.assertEqual(result["resolved_content"]["corrections"][0]["prior_finding_id"], "repair-1")
        self.assertEqual(result["raw_content"], raw)
        self.assertEqual(result["bindings"][0]["output_path"], "corrections[0].defect.source_ref")

    def test_unknown_duplicate_or_unsolicited_corrections_cannot_repair(self):
        correction = {"prior_finding_id": "repair-1", "defect": self.finding()}
        for rows, ids in (([correction], ["other"]), ([correction, correction], ["repair-1"]), ([correction], None)):
            raw = self.response(defects=[], corrections=rows)
            result = self.resolve(raw, pending_finding_ids=ids)
            self.assertFalse(result["all_source_refs_resolved"])
            self.assertEqual(result["resolved_content"]["corrections"], rows)
            self.assertEqual(result["bindings"], [])

    def test_unresolved_reference_enters_existing_ledger_and_empty_page_cannot_clear_it(self):
        state = InitialReviewContinuation(self.window, policy=POLICY, max_attempts=3)
        first = self.resolve(self.response(defects=[self.finding(source_ref="unknown")]))
        state.record_attempt(first["resolved_content"], call_id="offline-first",
                             pending_reasons=first["pending_reasons"])
        repair_id = state.pending_finding_ids[0]
        empty = self.resolve({"defects": [], "corrections": [], "more_findings_possible": False},
                             pending_finding_ids=[repair_id])
        second = state.record_attempt(empty["resolved_content"], call_id="offline-second",
                                      pending_reasons=empty["pending_reasons"])
        self.assertFalse(second["complete"])
        self.assertEqual(state.pending_finding_ids, [repair_id])
        correction = self.resolve({"defects": [], "corrections": [
            {"prior_finding_id": repair_id, "defect": self.finding()}], "more_findings_possible": False},
            pending_finding_ids=[repair_id])
        final = state.record_attempt(correction["resolved_content"], call_id="offline-third",
                                     pending_reasons=correction["pending_reasons"])
        self.assertTrue(final["complete"])
        self.assertEqual(final["attempts"][0]["content"]["defects"][0]["source_ref"], "unknown")
        self.assertEqual(final["attempts"][2]["repaired_finding_ids"], [repair_id])

    def test_source_resolvability_does_not_override_more_or_legal_validation(self):
        for changes, reason in (({}, "initial_output_more_findings_possible"),
                                ({"code": "invented_legal_code"}, "initial_output_finding_schema_invalid")):
            raw = self.response(defects=[self.finding(**changes)])
            result = self.resolve(raw)
            self.assertTrue(result["all_source_refs_resolved"])
            state = InitialReviewContinuation(self.window, policy=POLICY,
                allowed_codes=["cross_clause_inconsistency", "fee_mismatch"])
            summary = state.record_attempt(result["resolved_content"], call_id="offline-call",
                                            pending_reasons=result["pending_reasons"])
            self.assertFalse(summary["complete"])
            self.assertIn(reason, summary["pending_reasons"])

    def test_independent_replay_rebuilds_source_bindings_without_live_catalogue(self):
        raw = self.response()
        receipt = self.resolve(raw)
        report = replay_source_selection(copy.deepcopy(self.window), copy.deepcopy(raw), copy.deepcopy(receipt))
        self.assertTrue(report["replay_verified"])
        self.assertTrue(report["all_source_refs_resolved"])
        self.assertNotIn("complete", report)
        self.assertTrue(receipt["resolved_content"]["more_findings_possible"])

    def test_replay_detects_derived_quote_offset_hash_boolean_and_raw_tampering(self):
        raw = self.response()
        original = self.resolve(raw)
        mutations = (
            lambda x: x["resolved_content"]["defects"][0].update(quote="编造引用"),
            lambda x: x["bindings"][0].update(clause_start=0),
            lambda x: x.update(catalogue_sha256="wrong"),
            lambda x: x["resolved_content"].update(more_findings_possible=False),
            lambda x: x["raw_content"]["defects"][0].update(message="篡改原始返回"),
        )
        for mutate in mutations:
            receipt = copy.deepcopy(original)
            mutate(receipt)
            report = replay_source_selection(self.window, raw, receipt)
            self.assertFalse(report["replay_verified"])
            self.assertFalse(report["all_source_refs_resolved"])
            self.assertTrue(report["differences"])

    def test_replay_binds_external_original_sources_model_payload_and_prior_ids(self):
        raw = self.response(defects=[], corrections=[{"prior_finding_id": "repair-1", "defect": self.finding()}])
        receipt = self.resolve(raw, pending_finding_ids=["repair-1"])
        self.assertTrue(replay_source_selection(self.window, raw, receipt,
            pending_finding_ids=["repair-1"])["replay_verified"])
        self.assertFalse(replay_source_selection(self.window, raw, receipt,
            pending_finding_ids=["another-id"])["replay_verified"])
        modified_source = copy.deepcopy(self.window)
        modified_source["segments"][0]["text"] = modified_source["segments"][0]["text"].replace("甲", "丙")
        self.assertFalse(replay_source_selection(modified_source, raw, receipt,
            pending_finding_ids=["repair-1"])["replay_verified"])
        modified_raw = copy.deepcopy(raw)
        modified_raw["more_findings_possible"] = False
        self.assertFalse(replay_source_selection(self.window, modified_raw, receipt,
            pending_finding_ids=["repair-1"])["replay_verified"])

    def test_pending_receipt_can_replay_without_becoming_resolved(self):
        raw = self.response(defects=[self.finding(source_ref="unknown")])
        receipt = self.resolve(raw)
        report = replay_source_selection(self.window, raw, receipt)
        self.assertTrue(report["replay_verified"])
        self.assertFalse(report["all_source_refs_resolved"])
        self.assertEqual(report["pending_reasons"], receipt["pending_reasons"])

    def test_replay_rejects_unknown_protocol_and_malformed_receipts(self):
        raw = self.response()
        for receipt in (None, {}, {"protocol_version": "unrecognised"},
                        {**self.resolve(raw), "extract_chars": True}):
            with self.subTest(receipt=receipt):
                self.assertFalse(replay_source_selection(self.window, raw, receipt)["replay_verified"])


if __name__ == "__main__":
    unittest.main()
