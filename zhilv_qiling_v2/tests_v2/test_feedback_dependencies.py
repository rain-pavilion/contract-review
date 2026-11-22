"""Long dependency graphs must preserve obligations without recursive DFS."""
import copy
import unittest

from zhilv_qiling_v2.feedback_dependencies import dependency_graph, strongly_connected_components


class LongDependencyTests(unittest.TestCase):
    def test_long_directed_chain_has_separate_components(self):
        nodes = [f"w{i:04}" for i in range(1600)]
        self.assertEqual(strongly_connected_components(nodes, list(zip(nodes, nodes[1:]))),
                         [[node] for node in nodes])

    def test_long_cycle_and_disconnected_components(self):
        nodes = [f"w{i:04}" for i in range(1600)]
        edges = list(zip(nodes, nodes[1:] + nodes[:1])) + [("a", "b"), ("b", "a")]
        self.assertEqual(strongly_connected_components(nodes + ["a", "b", "z"], edges),
                         [["a", "b"], nodes, ["z"]])

    def test_actual_long_contract_work_graph_retains_all_pending_items(self):
        items = [{"work_item_id": f"w{i:04}", "kind": "statute" if i % 2 else "subject_fact",
                  "clause_ids": [f"c{i}", f"c{i + 1}"], "pending": True}
                 for i in range(1100)]
        original = copy.deepcopy(items)
        graph = dependency_graph(items)
        self.assertEqual(graph["sccs"], [[item["work_item_id"] for item in items]])
        self.assertEqual(len(graph["edges"]), 2198)
        self.assertEqual(items, original)

    def test_empty_and_invalid_graphs_keep_existing_contract(self):
        self.assertEqual(strongly_connected_components([], []), [])
        with self.assertRaisesRegex(ValueError, "dependency_edge_has_unknown_work_item"):
            strongly_connected_components(["a"], [("a", "missing")])


if __name__ == "__main__":
    unittest.main()
