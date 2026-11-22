"""Deterministic SCC diagnostics and receipt-bound incremental dispatch.

Skipping unchanged processed work never implies its source or semantics became
valid. Legacy pending and initial-continuation policy remain authoritative.
"""
from __future__ import annotations

import copy
from typing import Any

from .review_work_items import fingerprint


def strongly_connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    graph = {node: [] for node in sorted(set(nodes))}
    for source, target in sorted(set(edges)):
        if source not in graph or target not in graph:
            raise ValueError("dependency_edge_has_unknown_work_item")
        graph[source].append(target)
    # Iterative Kosaraju avoids the Python recursion limit on long contracts.
    # Both passes use the same validated graph; presentation order stays stable.
    reverse: dict[str, list[str]] = {node: [] for node in graph}
    for source, targets in graph.items():
        for target in targets:
            reverse[target].append(source)
    seen: set[str] = set()
    finished: list[str] = []
    for root in graph:
        if root in seen:
            continue
        seen.add(root)
        frames = [(root, iter(graph[root]))]
        while frames:
            node, neighbors = frames[-1]
            neighbor = next(neighbors, None)
            if neighbor is None:
                finished.append(node)
                frames.pop()
            elif neighbor not in seen:
                seen.add(neighbor)
                frames.append((neighbor, iter(graph[neighbor])))

    assigned: set[str] = set()
    components: list[list[str]] = []
    for root in reversed(finished):
        if root in assigned:
            continue
        assigned.add(root)
        pending = [root]
        component = []
        while pending:
            node = pending.pop()
            component.append(node)
            for neighbor in reverse[node]:
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def dependency_graph(items: list[dict[str, Any]]) -> dict[str, Any]:
    edges: set[tuple[str, str]] = set()
    for left in items:
        for right in items:
            if left["work_item_id"] == right["work_item_id"]:
                continue
            overlap = set(left.get("clause_ids", [])) & set(right.get("clause_ids", []))
            if overlap and left["kind"] != right["kind"]:
                edges.add((left["work_item_id"], right["work_item_id"]))
    nodes = sorted(item["work_item_id"] for item in items)
    pairs = sorted(edges)
    return {"version": "feedback-dependencies-scc-v1", "nodes": nodes,
        "edges": [{"source": source, "target": target, "reason": "shared_explicit_clause_work_dependency"}
                  for source, target in pairs],
        "sccs": strongly_connected_components(nodes, pairs),
        "meaning": "Edges describe work dependencies, never subject identity or legal entailment."}


class IncrementalDispatch:
    def __init__(self) -> None:
        self.processed: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def plan(self, items: list[dict[str, Any]], graph: dict[str, Any], round_no: int) -> dict[str, Any]:
        by_id = {item["work_item_id"]: item for item in items}
        component_for = {item_id: component for component in graph["sccs"] for item_id in component}
        dispatch, reuse = [], []
        reasons = {}
        for item in items:
            if not item.get("dispatchable"):
                continue
            item_id = item["work_item_id"]
            component = component_for[item_id]
            dependency_input = {member: by_id[member]["input_fingerprint"] for member in component}
            input_key = fingerprint({"own": item["input_fingerprint"], "scc_inputs": dependency_input})
            previous = self.processed.get(item_id)
            reusable = bool(previous and previous["dependency_input_fingerprint"] == input_key
                            and previous["receipt_call_ids"])
            (reuse if reusable else dispatch).append(item_id)
            reasons[item_id] = {"decision": "reuse_processed_unchanged" if reusable else "dispatch_new_or_changed_or_unprocessed",
                "dependency_input_fingerprint": input_key,
                "receipt_call_ids": copy.deepcopy(previous["receipt_call_ids"]) if reusable else [],
                "pending_retained": bool(item.get("pending"))}
        event = {"seq": len(self.events) + 1, "round": round_no,
            "dispatch_work_item_ids": sorted(dispatch), "reused_work_item_ids": sorted(reuse),
            "decisions": reasons,
            "atomic_request_policy": "Any dirty item dispatches the complete legacy request; no synthetic merged acknowledgement."}
        self.events.append(copy.deepcopy(event))
        return event

    def record_processed(self, items: list[dict[str, Any]], graph: dict[str, Any]) -> None:
        by_id = {item["work_item_id"]: item for item in items}
        component_for = {item_id: component for component in graph["sccs"] for item_id in component}
        for item in items:
            if not item.get("dispatchable") or not item.get("actual_processed") or not item.get("receipt_call_ids"):
                continue
            members = component_for[item["work_item_id"]]
            self.processed[item["work_item_id"]] = {
                "dependency_input_fingerprint": fingerprint({"own": item["input_fingerprint"],
                    "scc_inputs": {member: by_id[member]["input_fingerprint"] for member in members}}),
                "receipt_call_ids": list(item["receipt_call_ids"])}

    def bind_prior_processing(self, items: list[dict[str, Any]], graph: dict[str, Any]) -> None:
        """Expose a prior real execution separately from this request's ack."""
        by_id = {item["work_item_id"]: item for item in items}
        component_for = {item_id: component for component in graph["sccs"] for item_id in component}
        for item in items:
            if item.get("actual_processed") or not item.get("dispatchable"):
                continue
            previous = self.processed.get(item["work_item_id"])
            members = component_for[item["work_item_id"]]
            key = fingerprint({"own": item["input_fingerprint"],
                "scc_inputs": {member: by_id[member]["input_fingerprint"] for member in members}})
            if previous and previous["dependency_input_fingerprint"] == key and previous["receipt_call_ids"]:
                item["execution_status"] = "reused_prior_processed_input"
                item["actual_processed"] = True
                item["receipt_call_ids"] = list(previous["receipt_call_ids"])
                item["reused_processing"] = True
                # Neither protocol_complete nor pending/source/semantic axes is
                # changed. This is not a synthetic current-fingerprint receipt.

    def snapshot(self) -> dict[str, Any]:
        return {"version": "incremental-feedback-dispatch-v1", "processed": copy.deepcopy(self.processed),
                "events": copy.deepcopy(self.events)}
