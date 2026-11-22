from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contract_review.agents.legal import apply_law
from contract_review.agents.state import AgentReviewConfig, RiskCandidate
from contract_review.kg import (
    LegalTagGraph,
    build_contract_graph,
    cn_to_int,
    graph_signals,
)
from contract_review.legal_kb import RetrievedArticle
from contract_review.models import (
    Clause,
    Contract,
    ContractElements,
    DocumentParseResult,
    Element,
    SourceBlock,
)


def clause(
    clause_id: str,
    label: str,
    text: str,
    parent_id: str | None = None,
    level: int = 1,
) -> Clause:
    return Clause(
        clause_id=clause_id,
        label=label,
        title=label,
        level=level,
        heading=label,
        body=text,
        text=f"{label} {text}",
        parent_id=parent_id,
        block_ids=[f"b-{clause_id}"],
    )


def make_contract(clauses: list[Clause], elements: ContractElements | None = None):
    blocks = [
        SourceBlock(
            block_id=f"b-{c.clause_id}",
            order=i,
            kind="clause",
            text=c.text,
        )
        for i, c in enumerate(clauses)
    ]
    document = DocumentParseResult(
        filename="kg-sample.txt",
        file_type=".txt",
        full_text="\n".join(c.text for c in clauses),
        blocks=blocks,
        clauses=clauses,
    )
    return Contract(
        contract_id="c-kg-test",
        document=document,
        elements=elements or ContractElements(),
    )


class ChineseNumeralChecks(unittest.TestCase):
    def test_basic_numbers(self) -> None:
        self.assertEqual(cn_to_int("七"), 7)
        self.assertEqual(cn_to_int("十"), 10)
        self.assertEqual(cn_to_int("二十一"), 21)
        self.assertEqual(cn_to_int("两百零五"), 205)
        self.assertEqual(cn_to_int("705"), 705)

    def test_garbage_is_none(self) -> None:
        self.assertIsNone(cn_to_int(""))
        self.assertIsNone(cn_to_int("甲乙"))


class ContractGraphChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.clauses = [
            clause("c1", "第一条", "总则。本合同未尽事宜参照第三条执行。"),
            clause("c2", "第二条", "争议处理按第八条约定办理。"),
            clause(
                "c3",
                "第三条",
                "租金支付依据《中华人民共和国民法典》第七百二十一条确定。",
            ),
        ]
        self.contract = make_contract(self.clauses)

    def test_resolved_reference_edge(self) -> None:
        graph = build_contract_graph(self.contract)
        refs = [e for e in graph.edges if e.kind == "references"]
        self.assertEqual(len(refs), 1)
        self.assertEqual((refs[0].source, refs[0].target), ("c1", "c3"))
        self.assertEqual(graph.resolved_refs, 1)

    def test_dangling_reference_detected(self) -> None:
        graph = build_contract_graph(self.contract)
        self.assertEqual(len(graph.dangling), 1)
        self.assertEqual(graph.dangling[0].clause_id, "c2")
        self.assertEqual(graph.dangling[0].raw, "第八条")

    def test_statute_citation_not_treated_as_clause_ref(self) -> None:
        # 第七百二十一条 is a Civil Code citation; it must never look like a
        # dangling contract clause.
        dangling = [
            d for d in build_contract_graph(self.contract).dangling
            if "七百二十一" in d.raw
        ]
        self.assertEqual(dangling, [])

    def test_self_heading_is_not_an_edge(self) -> None:
        graph = build_contract_graph(self.contract)
        self.assertFalse(
            any(e.source == e.target for e in graph.edges)
        )

    def test_kuan_resolves_only_among_siblings(self) -> None:
        clauses = [
            clause("a1", "第一条", "甲方义务。", parent_id=None),
            clause("a1-1", "第一款", "付款方式见第二款。", parent_id="a1", level=2),
            clause("a1-2", "第二款", "银行转账。", parent_id="a1", level=2),
            clause("b1", "第二条", "乙方义务。", parent_id=None),
            clause("b1-1", "第一款", "通知方式见第二款。", parent_id="b1", level=2),
        ]
        graph = build_contract_graph(make_contract(clauses))
        refs = {(e.source, e.target) for e in graph.edges if e.kind == "references"}
        self.assertIn(("a1-1", "a1-2"), refs)
        # b1-1 has no local 第二款: the a-tree's subsection must not satisfy it.
        self.assertTrue(
            any(d.clause_id == "b1-1" and d.raw == "第二款" for d in graph.dangling)
        )

    def test_parent_and_element_edges(self) -> None:
        elements = ContractElements(
            parties=[
                Element(
                    element_id="E1",
                    kind="party",
                    value="甲方公司",
                    role="委托方",
                    clause_id="c1",
                )
            ],
            amounts=[
                Element(
                    element_id="E2",
                    kind="amount",
                    value="5000元",
                    role="合同总价",
                    clause_id="c1",
                )
            ],
        )
        graph = build_contract_graph(make_contract(self.clauses, elements))
        self.assertIn(
            ("E1", "c1", "located_in"),
            {(e.source, e.target, e.kind) for e in graph.edges},
        )
        node_kinds = {n.kind for n in graph.nodes}
        self.assertEqual(node_kinds, {"clause", "party", "amount"})


class GraphSignalsChecks(unittest.TestCase):
    def test_dangling_reference_becomes_signal(self) -> None:
        contract = make_contract(
            [
                clause("c1", "第一条", "适用第十二条的约定。"),
            ]
        )
        signals, stats = graph_signals(contract)
        self.assertEqual(stats["dangling_references"], 1)
        self.assertEqual(signals[0].kind, "reference")
        self.assertEqual(signals[0].clause_id, "c1")
        self.assertIn("第十二条", signals[0].note)

    def test_unbound_amount_flagged(self) -> None:
        elements = ContractElements(
            amounts=[
                Element(
                    element_id="E1",
                    kind="amount",
                    value="999元",
                    role="",
                    clause_id="c1",
                ),
                Element(
                    element_id="E2",
                    kind="amount",
                    value="1000元",
                    role="租金",
                    clause_id="c1",
                ),
            ]
        )
        contract = make_contract([clause("c1", "第一条", "租金1000元。")], elements)
        signals, _ = graph_signals(contract)
        notes = [s.note for s in signals if s.kind == "amount"]
        self.assertEqual(len(notes), 1)
        self.assertIn("999元", notes[0])

    def test_dedup_repeated_dangling_same_clause(self) -> None:
        contract = make_contract(
            [clause("c1", "第一条", "见第十二条，另见第十二条。")]
        )
        signals, stats = graph_signals(contract)
        self.assertEqual(stats["dangling_references"], 2)
        self.assertEqual(
            len([s for s in signals if s.kind == "reference"]), 1
        )


STATUTES = [
    {
        "article_uid": "a1",
        "law_name": "民法典",
        "article_no": "第七百零五条",
        "content": "租赁期限不得超过二十年。",
        "status": "effective",
        "risk_tags": ["term_over_20y"],
        "contract_types": ["房屋租赁"],
    },
    {
        "article_uid": "a2",
        "law_name": "民法典",
        "article_no": "第七百零三条",
        "content": "租赁合同是出租人将租赁物交付承租人使用的合同。",
        "status": "effective",
        "risk_tags": ["term_over_20y", "lease"],
        "contract_types": ["房屋租赁"],
    },
    {
        "article_uid": "a3",
        "law_name": "民法典",
        "article_no": "第七百二十一条",
        "content": "承租人应当按照约定的期限支付租金。",
        "status": "effective",
        "risk_tags": ["lease"],
        "contract_types": ["房屋租赁"],
    },
    {
        "article_uid": "a4",
        "law_name": "旧法",
        "article_no": "第十条",
        "content": "已失效条文。",
        "status": "revised",
        "risk_tags": ["term_over_20y"],
        "contract_types": [],
    },
]


class LegalTagGraphChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = LegalTagGraph(STATUTES)

    def test_non_effective_articles_excluded(self) -> None:
        self.assertEqual(self.graph.article_count, 3)
        uids = {
            a.article_uid
            for a in self.graph.articles_for_tags(["term_over_20y"], limit=10)
        }
        self.assertEqual(uids, {"a1", "a2"})

    def test_expand_follows_shared_tags(self) -> None:
        expanded = self.graph.expand(["a1"], limit=5)
        self.assertEqual([a.article_uid for a in expanded], ["a2"])

    def test_expand_seeds_excluded_and_ranked(self) -> None:
        expanded = self.graph.expand(["a1", "a2"], limit=5)
        self.assertEqual([a.article_uid for a in expanded], ["a3"])

    def test_expand_respects_limit_and_counts(self) -> None:
        self.assertEqual(len(self.graph.expand(["a1"], limit=0)), 0)
        self.assertEqual(self.graph.expansion_count, 0)
        self.graph.expand(["a1"], limit=1)
        self.assertEqual(self.graph.expansion_count, 1)

    def test_to_article_roundtrip(self) -> None:
        article = self.graph.to_article("a1")
        self.assertIsNotNone(article)
        self.assertEqual(article.citation(), "《民法典》第七百零五条")
        self.assertEqual(article.risk_tags, ["term_over_20y"])

    def test_file_cache_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "statutes.jsonl"
            path.write_text(
                "\n".join(json.dumps(s, ensure_ascii=False) for s in STATUTES),
                encoding="utf-8",
            )
            first = LegalTagGraph.load(path)
            first.expand(["a1"])
            second = LegalTagGraph.load(path)
            self.assertIs(first, second)
            # Counter resets per run so report stats stay run-scoped.
            self.assertEqual(second.expansion_count, 0)
            missing = Path(tmp) / "nope.jsonl"
            with self.assertRaises(FileNotFoundError):
                LegalTagGraph.load(missing)


class TagRetrievalIntegrationChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.tag_graph = LegalTagGraph(STATUTES)
        self.candidate = RiskCandidate(
            candidate_id="x1",
            code="term_over_20y",
            severity="high",
            message="租赁期限约定为二十五年。",
            clause_ids=["c1"],
        )
        self.contract = make_contract(
            [clause("c1", "第一条", "租期二十五年。")]
        )

    def test_tag_fallback_without_vector_index(self) -> None:
        config = AgentReviewConfig(enable_llm=False, top_k=3)
        applications, errors = apply_law(
            [self.candidate],
            self.contract,
            config,
            kb=None,
            tag_graph=self.tag_graph,
        )
        uids = [a["article_uid"] for a in applications[0]["articles"]]
        self.assertIn("a1", uids)
        self.assertTrue(any("降级为风险标签直检" in e for e in errors))
        self.assertFalse(any("跳过法条检索" in e for e in errors))

    def test_unknown_code_reports_skipping(self) -> None:
        config = AgentReviewConfig(enable_llm=False)
        candidate = RiskCandidate(
            candidate_id="x2",
            code="some_brand_new_code",
            severity="low",
            message="test",
        )
        _, errors = apply_law(
            [candidate],
            self.contract,
            config,
            kb=None,
            tag_graph=self.tag_graph,
        )
        self.assertTrue(any("跳过法条检索" in e for e in errors))

    def test_expansion_appends_after_kb_recall(self) -> None:
        class FakeKB:
            def search(self, query: str, top_k: int = 6):
                return [
                    RetrievedArticle(
                        article_uid="a1",
                        law_name="民法典",
                        article_no="第七百零五条",
                        content="租赁期限不得超过二十年。",
                        risk_tags=["term_over_20y"],
                    )
                ]

        config = AgentReviewConfig(
            enable_llm=False, top_k=3, tag_expansion_limit=2
        )
        applications, _ = apply_law(
            [self.candidate],
            self.contract,
            config,
            kb=FakeKB(),
            tag_graph=self.tag_graph,
        )
        uids = [a["article_uid"] for a in applications[0]["articles"]]
        self.assertEqual(uids[0], "a1")  # vector recall keeps first position
        self.assertIn("a2", uids)  # tag-expanded
        self.assertLessEqual(len(uids), 5)
        self.assertEqual(len(set(uids)), len(uids))  # no duplicates


if __name__ == "__main__":
    unittest.main()
