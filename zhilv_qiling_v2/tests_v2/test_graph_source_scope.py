"""Clause references resolve within sources without silently choosing a duplicate."""

import unittest

from zhilv_qiling_v2.graph import build_contract_graph, graph_signals
from zhilv_qiling_v2.models import Clause, ContractElements


def clause(cid, label, body, **metadata):
    return Clause(cid, label, "", 2, label, body, label + "\n" + body,
                  metadata=metadata)


class GraphSourceScopeTests(unittest.TestCase):
    def graph(self, *clauses):
        return build_contract_graph(clauses, ContractElements())

    def references(self, graph):
        return {(edge.source, edge.target) for edge in graph.edges
                if edge.kind == "references"}

    def test_single_document_references_and_string_dangling_remain_compatible(self):
        clauses = [clause("one", "第一条", "依第二条执行，另见第四条。"),
                   clause("two", "第二条", "本第二条另有约定。")]
        graph = self.graph(*clauses)
        self.assertEqual(self.references(graph), {("one", "two")})
        self.assertEqual(graph.dangling, ["第四条"])
        self.assertEqual(graph.stats["dangling"], 1)
        self.assertTrue(all(isinstance(token, str) for token in graph.dangling))
        self.assertIn("第四条", graph_signals(graph, clauses)[0].value)

    def test_same_number_in_other_source_never_overwrites_local_target(self):
        graph = self.graph(
            clause("main_one", "第一条", "依第二条执行。", source_id="main"),
            clause("main_two", "第二条", "主合同付款。", source_id="main"),
            clause("annex_one", "第一条", "依第二条执行。", source_id="annex"),
            clause("annex_two", "第二条", "附件付款。", source_id="annex"))
        self.assertEqual(self.references(graph),
                         {("main_one", "main_two"), ("annex_one", "annex_two")})
        self.assertEqual(graph.dangling, [])
        self.assertEqual(graph.stats.get("source_scoped_references"), 2)

    def test_each_supported_source_metadata_field_can_scope_references(self):
        for field in ("document_id", "source_id", "source_title"):
            with self.subTest(field=field):
                graph = self.graph(
                    clause("main", "第一条", "依第二条执行。", **{field: "main"}),
                    clause("main_two", "第二条", "主合同。", **{field: "main"}),
                    clause("other_two", "第二条", "附件。", **{field: "other"}))
                self.assertEqual(self.references(graph), {("main", "main_two")})

    def test_explicit_attachment_reference_can_target_same_number_as_origin(self):
        graph = self.graph(
            clause("main_one", "第一条", "依附件A第一条执行。",
                   source_id="main", source_title="主合同"),
            clause("annex_one", "第一条", "附件约定。",
                   source_id="annex", source_title="附件A：技术协议"))
        self.assertEqual(self.references(graph), {("main_one", "annex_one")})
        self.assertEqual(graph.dangling, [])

    def test_explicit_source_title_resolves_in_named_document(self):
        graph = self.graph(
            clause("main", "第三条", "按照《技术验收协议》的第一条办理。",
                   source_id="main", source_title="主合同"),
            clause("local", "第一条", "主合同。", source_id="main"),
            clause("technical", "第一条", "验收测试。",
                   source_id="technical", source_title="技术验收协议"),
            clause("other", "第一条", "其他附件。", source_id="other"))
        self.assertEqual(self.references(graph), {("main", "technical")})

    def test_plain_source_title_resolves_before_local_number(self):
        graph = self.graph(
            clause("main", "第三条", "根据技术验收协议第一条办理。",
                   document_id="main", source_title="主合同"),
            clause("local", "第一条", "主合同。", document_id="main"),
            clause("technical", "第一条", "验收测试。",
                   document_id="technical", source_title="技术验收协议"),
            clause("other", "第一条", "其他附件。", document_id="other"))
        self.assertEqual(self.references(graph), {("main", "technical")})

    def test_unknown_attachment_does_not_fall_back_to_local_clause(self):
        clauses = [clause("main", "第一条", "依附件Z第二条执行。", source_id="main"),
                   clause("local", "第二条", "主合同。", source_id="main")]
        graph = self.graph(*clauses)
        self.assertEqual(self.references(graph), set())
        self.assertTrue(any("附件Z" in token and "第二条" in token
                            for token in graph.dangling))
        self.assertTrue(graph_signals(graph, clauses))

    def test_named_source_missing_number_does_not_fall_back_to_local_clause(self):
        graph = self.graph(
            clause("main", "第一条", "依附件A第二条执行。", source_id="main"),
            clause("local", "第二条", "主合同。", source_id="main"),
            clause("annex", "第一条", "附件。", source_id="annex", source_title="附件A"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(any("附件A" in token and "第二条" in token
                            for token in graph.dangling))

    def test_unqualified_reference_with_multiple_external_targets_stays_ambiguous(self):
        graph = self.graph(
            clause("main", "第三条", "依第一条执行。", source_id="main"),
            clause("a", "第一条", "附件A。", source_id="a"),
            clause("b", "第一条", "附件B。", source_id="b"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)
        self.assertEqual(graph.stats.get("ambiguous_references"), 1)

    def test_duplicate_number_within_source_stays_ambiguous(self):
        graph = self.graph(
            clause("main", "第三条", "依第一条执行。", source_id="main"),
            clause("one_a", "第一条", "版本A。", source_id="main"),
            clause("one_b", "第一条", "版本B。", source_id="main"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)
        self.assertEqual(graph.stats.get("ambiguous_references"), 1)

    def test_duplicate_self_number_is_not_silently_filtered(self):
        graph = self.graph(
            clause("one_a", "第一条", "依第一条执行。", source_id="main"),
            clause("one_b", "第一条", "另一第一条。", source_id="main"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)
        self.assertGreaterEqual(graph.stats.get("ambiguous_references", 0), 1)

    def test_explicit_source_with_duplicate_number_stays_ambiguous(self):
        graph = self.graph(
            clause("main", "第三条", "依附件A第一条执行。", source_id="main"),
            clause("a_one", "第一条", "版本A。", source_id="a", source_title="附件A"),
            clause("a_two", "第一条", "版本B。", source_id="a", source_title="附件A"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)
        self.assertEqual(graph.stats.get("ambiguous_references"), 1)

    def test_same_title_on_different_sources_does_not_choose_one(self):
        graph = self.graph(
            clause("main", "第三条", "依《技术协议》第一条执行。", source_id="main"),
            clause("a", "第一条", "版本A。", source_id="a", source_title="技术协议"),
            clause("b", "第一条", "版本B。", source_id="b", source_title="技术协议"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)
        self.assertEqual(graph.stats.get("ambiguous_references"), 1)

    def test_duplicate_number_without_metadata_never_overwrites_target(self):
        graph = self.graph(clause("main", "第三条", "依第一条执行。"),
                           clause("a", "第一条", "版本A。"),
                           clause("b", "第一条", "版本B。"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(graph.dangling)

    def test_unique_external_target_is_structural_link_not_legal_approval(self):
        graph = self.graph(
            clause("main", "第三条", "依第一条执行。", source_id="main"),
            clause("a", "第一条", "附件。", source_id="a"))
        self.assertEqual(self.references(graph), {("main", "a")})
        self.assertIs(graph.stats.get("automatic_legal_confirmation_allowed"), False)

    def test_coordinated_articles_keep_explicit_attachment_scope(self):
        for conjunction in ("、", "及", "和", "与", "以及", "或"):
            with self.subTest(conjunction=conjunction):
                graph = self.graph(
                    clause("main", "第三条", f"依附件A第一条{conjunction}第二条执行。",
                           source_id="main"),
                    clause("local_two", "第二条", "主合同付款。", source_id="main"),
                    clause("annex_one", "第一条", "附件第一条。", source_id="annex",
                           source_title="附件A"),
                    clause("annex_two", "第二条", "附件第二条。", source_id="annex",
                           source_title="附件A"))
                self.assertEqual(self.references(graph),
                                 {("main", "annex_one"), ("main", "annex_two")})
                self.assertEqual(graph.dangling, [])

    def test_missing_attachment_scope_is_retained_in_coordinated_articles(self):
        graph = self.graph(
            clause("main", "第三条", "依附件Z第一条、第二条执行。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(any("附件Z" in token and "第二条" in token
                            for token in graph.dangling))

    def test_parenthesized_missing_attachment_never_falls_back_to_local(self):
        for attachment in ("附件（Z）", "附件(Z)"):
            with self.subTest(attachment=attachment):
                graph = self.graph(
                    clause("main", "第三条", f"依{attachment}第二条执行。", source_id="main"),
                    clause("local_two", "第二条", "主合同付款。", source_id="main"))
                self.assertEqual(self.references(graph), set())
                self.assertTrue(graph.dangling)
                self.assertEqual(graph.stats.get("reference_issues", [])[0]["reason"],
                                 "missing_reference")

    def test_parenthesized_attachment_can_resolve_normalized_name(self):
        graph = self.graph(
            clause("main", "第三条", "依附件（A）第二条执行。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"),
            clause("annex_two", "第二条", "附件付款。", source_id="annex", source_title="附件A"))
        self.assertEqual(self.references(graph), {("main", "annex_two")})

    def test_line_break_does_not_erase_explicit_attachment_scope(self):
        graph = self.graph(
            clause("main", "第三条", "依附件A\n第二条执行。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"),
            clause("annex_two", "第二条", "附件付款。", source_id="annex", source_title="附件A"))
        self.assertEqual(self.references(graph), {("main", "annex_two")})

    def test_attachment_and_quoted_title_resolve_as_one_source(self):
        graph = self.graph(
            clause("main", "第三条", "依附件A《技术协议》第二条执行。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"),
            clause("annex_two", "第二条", "附件付款。", source_id="annex",
                   source_title="附件A：技术协议"))
        self.assertEqual(self.references(graph), {("main", "annex_two")})
        self.assertEqual(graph.dangling, [])

    def test_unknown_attachment_is_not_discarded_for_a_known_quoted_title(self):
        graph = self.graph(
            clause("main", "第三条", "依附件Z《技术协议》第二条执行。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"),
            clause("technical_two", "第二条", "另一技术协议。", source_id="technical",
                   source_title="技术协议"))
        self.assertEqual(self.references(graph), set())
        self.assertTrue(any("附件Z" in token and "第二条" in token
                            for token in graph.dangling))

    def test_attachment_scope_does_not_leak_into_next_sentence(self):
        graph = self.graph(
            clause("main", "第三条", "依附件A第一条执行。另依第二条付款。", source_id="main"),
            clause("local_two", "第二条", "主合同付款。", source_id="main"),
            clause("annex_one", "第一条", "附件第一条。", source_id="annex", source_title="附件A"),
            clause("annex_two", "第二条", "附件第二条。", source_id="annex", source_title="附件A"))
        self.assertEqual(self.references(graph),
                         {("main", "annex_one"), ("main", "local_two")})


if __name__ == "__main__":
    unittest.main()
