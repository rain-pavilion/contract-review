"""Opt-in immutable feedback scopes. Deferred work never acknowledges evidence.

A cycle owns one exact request and one dependency graph. Discoveries are versioned
pending work for an explicit later cycle; no receipt is copied across versions.
"""
from __future__ import annotations

import copy
from typing import Any

from .review_work_items import feedback_work_items, fingerprint


def _pack(request: tuple) -> dict[str, Any]:
    key, checks, subject, focus, statutes, facts = request
    return {"upstream_fingerprint": key, "checks": copy.deepcopy(checks),
        "subject": copy.deepcopy(subject), "focus": sorted(focus),
        "statutes": copy.deepcopy(statutes), "facts": copy.deepcopy(facts)}



def _resolved_subject_request(item: dict[str, Any], subject: dict[str, Any],
                              clause_sources: dict[str, str]) -> bool:
    """Release only one scope-bound request backed by reviewed source records.

    Supplied provenance remains a human claim; this does not independently
    authenticate the publisher or complete the separate model-feedback work.
    """
    from .subject_facts import TYPE_STATUSES, validate_fact

    request_id = item["inputs"]["request_id"]
    if request_id in subject.get("pending_verification_request_ids", []):
        return False
    rows = [row for row in subject.get("verification_results", [])
            if isinstance(row, dict) and row.get("request_id") == request_id]
    if len(rows) != 1:
        return False
    row = rows[0]
    status, fact_type = row.get("fact_status"), row.get("fact_type")
    binding = row.get("identity_binding")
    clause_ids = row.get("clause_ids")
    if (not isinstance(binding, dict) or not isinstance(fact_type, str) or not isinstance(status, str)
            or not isinstance(clause_ids, list) or not clause_ids
            or not all(isinstance(cid, str) for cid in clause_ids)):
        return False
    identifiers = binding.get("identifiers", [])
    if (row.get("status") != "completed" or row.get("reasons")
            or status not in TYPE_STATUSES.get(fact_type, set()) - {"unknown"}
            or not row.get("subject_name") or not row.get("subject_node_id")
            or not row.get("subject_id") or identifiers != [row["subject_id"]]
            or binding.get("status") != "bound"
            or any(cid not in clause_sources for cid in clause_ids)):
        return False
    # A completed result for another candidate/subject/scope cannot acknowledge
    # the saved obligation merely by copying its request_id.
    for old in item["inputs"].get("verification_results", []):
        for key in ("candidate_id", "article_uid", "risk_code", "subject_name",
                    "subject_node_id", "subject_id", "fact_type"):
            if old.get(key) and old[key] != row.get(key):
                return False
        if old.get("clause_ids") and sorted(old["clause_ids"]) != sorted(row["clause_ids"]):
            return False
    fact_ids = row.get("fact_ids", [])
    records = row.get("records", [])
    if (not isinstance(fact_ids, list) or not fact_ids
            or not all(isinstance(fact_id, str) and fact_id for fact_id in fact_ids)
            or len(set(fact_ids)) != len(fact_ids)
            or not isinstance(records, list) or not all(isinstance(record, dict) for record in records)):
        return False
    # Check all records for this exact identity and fact type, not just the
    # fact_ids selected by the completed row. A chosen subset cannot conceal
    # contradictory known evidence or an explicit conflicting status.
    for record in records:
        if (record.get("subject_name") == row["subject_name"]
                and record.get("subject_id") == row["subject_id"]
                and record.get("fact_type") == fact_type
                and record.get("effective_status") not in (status, "unknown")):
            return False
    selected = [record for record in records if record.get("fact_id") in fact_ids]
    if len(selected) != len(fact_ids) or {record.get("fact_id") for record in selected} != set(fact_ids):
        return False
    for record in selected:
        if (record.get("effective_status") != status or record.get("fact_type") != fact_type
                or record.get("subject_name") != row["subject_name"]
                or record.get("subject_id") != row["subject_id"]
                or record.get("identity_match") != "name_and_identifier"
                or record.get("review_reasons") or record.get("review_reason")
                or record.get("identity_review_reason")):
            return False
        try:
            validated = validate_fact(record)
        except (TypeError, ValueError):
            return False
        if validated["fact_id"] != record.get("fact_id") or validated["status"] != status:
            return False
    return True


class FeedbackCycle:
    schema_version = "frozen-feedback-cycle-v1"

    def __init__(self, contract_id: str, clause_sources: dict[str, str], request: tuple,
                 *, cycle_number: int = 1, parent_cycle_id: str | None = None):
        self.contract_id = str(contract_id)
        self.clause_sources = copy.deepcopy(clause_sources)
        self.source_fingerprint = fingerprint(self.clause_sources)
        self.cycle_number = int(cycle_number)
        if self.cycle_number < 1:
            raise ValueError("feedback_cycle_number_invalid")
        self.parent_cycle_id = parent_cycle_id
        self.active_request = _pack(request)
        self.cycle_id = "feedback_cycle_" + fingerprint({"contract_id": self.contract_id,
            "source": self.source_fingerprint, "number": self.cycle_number,
            "parent": parent_cycle_id, "request": self.active_request})[:24]
        self.key = fingerprint({"cycle_id": self.cycle_id, "request": self.active_request})
        self.active_versions = {item["work_item_id"]: item["input_fingerprint"]
                                for item in self._items(self.active_request)}
        self.deferred: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.active_graph: dict[str, Any] | None = None
        self.active_status: list[dict[str, Any]] = []
        self.prior_cycle_archive: dict[str, Any] | None = None

    def _items(self, packed: dict[str, Any]) -> list[dict[str, Any]]:
        return feedback_work_items(self.contract_id, packed["statutes"], packed["facts"],
            packed["subject"], {}, [], self.clause_sources)

    def request(self) -> tuple:
        value = copy.deepcopy(self.active_request)
        return (self.key, value["checks"], value["subject"], set(value["focus"]),
                value["statutes"], value["facts"])

    def observe(self, live_request: tuple) -> None:
        """Append discoveries, including changed versions of an active obligation."""
        packed = _pack(live_request)
        for original in self._items(packed):
            work_id, version = original["work_item_id"], original["input_fingerprint"]
            if self.active_versions.get(work_id) == version:
                continue
            version_id = fingerprint([work_id, version])
            if version_id in self.deferred:
                continue
            item = copy.deepcopy(original)
            item.update(work_item_id="work_" + fingerprint([self.cycle_id, "next", version_id])[:24],
                obligation_work_item_id=work_id, discovered_cycle_id=self.cycle_id,
                queue_status="deferred_next_cycle", dispatchable=False,
                actual_processed=False, execution_status="deferred_next_cycle",
                semantic_status="pending_next_cycle", pending=True, receipt_call_ids=[],
                protocol_complete=False,
                pending_reasons=sorted(set(original["pending_reasons"] + ["deferred_to_next_cycle"])))
            item["supersedes_active_version"] = self.active_versions.get(work_id)
            # Exact per-item evidence remains in provided_feedback/inputs. Do
            # not duplicate the entire request once for every queued item.
            item["discovery_request_fingerprint"] = packed["upstream_fingerprint"]
            self.deferred[version_id] = item
            self.events.append({"seq": len(self.events) + 1, "event": "deferred_next_cycle",
                "work_item_id": item["work_item_id"], "obligation_work_item_id": work_id,
                "input_fingerprint": version})

    def bind_graph(self, items: list[dict[str, Any]], graph: dict[str, Any]) -> None:
        """Do not weaken SCC invalidation; freeze what is admitted into this SCC."""
        shape = {"nodes": copy.deepcopy(graph["nodes"]), "edges": copy.deepcopy(graph["edges"]),
                 "sccs": copy.deepcopy(graph["sccs"])}
        if self.active_graph is None:
            self.active_graph = shape
        elif self.active_graph != shape:
            raise ValueError("feedback_cycle_active_dependency_graph_changed")
        active = {item["work_item_id"]: item["input_fingerprint"] for item in items
                  if item["kind"] != "initial_review"}
        if active != self.active_versions:
            raise ValueError("feedback_cycle_active_work_versions_changed")

    def deferred_items(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self.deferred.values()))

    def deferred_affects(self, provided: dict[str, Any]) -> bool:
        """A historical confirmation cannot release a target with newer input."""
        for item in self.deferred.values():
            other = item.get("provided_feedback", {})
            if provided.get("feedback_kind") == "subject_fact":
                keys = ("subject_node_id", "subject_name", "fact_type", "clause_id")
                if other.get("feedback_kind") == "subject_fact" and all(
                        other.get(key) == provided.get(key) for key in keys):
                    return True
            elif (other.get("candidate_id") == provided.get("candidate_id")
                    and other.get("article_uid") == provided.get("article_uid")):
                return True
        return False

    def update_status(self, active_items: list[dict[str, Any]]) -> None:
        self.active_status = copy.deepcopy(active_items)

    def snapshot(self) -> dict[str, Any]:
        feedback = [item for item in self.active_status
                    if item["kind"] in ("statute_feedback", "subject_fact")]
        attempted = bool(feedback) and all(item.get("actual_processed")
            and item.get("receipt_call_ids")
            and "current_fingerprint_not_attempted" not in item.get("pending_reasons", [])
            for item in feedback)
        pending = [item["work_item_id"] for item in self.active_status if item.get("pending")]
        complete = bool(self.active_status) and not pending
        return {"schema_version": self.schema_version, "contract_id": self.contract_id,
            "source_fingerprint": self.source_fingerprint, "cycle_id": self.cycle_id,
            "cycle_number": self.cycle_number, "parent_cycle_id": self.parent_cycle_id,
            "active_request_fingerprint": self.key, "active_request": copy.deepcopy(self.active_request),
            "active_work_versions": copy.deepcopy(self.active_versions),
            "active_dependency_graph": copy.deepcopy(self.active_graph),
            "active_work_items": copy.deepcopy(self.active_status),
            "active_attempt_complete": attempted, "active_cycle_complete": complete,
            "active_pending_work_item_ids": pending,
            "deferred_next_cycle": self.deferred_items(),
            "deferred_pending_work_item_ids": [item["work_item_id"] for item in self.deferred.values()],
            "overall_complete": complete and not self.deferred,
            "events": copy.deepcopy(self.events),
            "prior_cycle_archive": copy.deepcopy(self.prior_cycle_archive),
            "meaning": "A frozen snapshot may be processed; newer evidence remains pending and cannot inherit its receipt."}

    @classmethod
    def validate_scope(cls, snapshot: dict[str, Any], *, contract_id: str,
                       clause_sources: dict[str, str]) -> None:
        if (snapshot.get("schema_version") != cls.schema_version
                or snapshot.get("contract_id") != contract_id
                or snapshot.get("source_fingerprint") != fingerprint(clause_sources)):
            raise ValueError("feedback_cycle_continuation_scope_mismatch")
        # Reject altered saved active inputs rather than trusting a supplied key.
        packed = snapshot.get("active_request", {})
        original = cls(contract_id, clause_sources, (packed.get("upstream_fingerprint"),
            packed.get("checks", []), packed.get("subject", {}), set(packed.get("focus", [])),
            packed.get("statutes", []), packed.get("facts", [])),
            cycle_number=snapshot.get("cycle_number", 0), parent_cycle_id=snapshot.get("parent_cycle_id"))
        if (original.cycle_id != snapshot.get("cycle_id")
                or original.key != snapshot.get("active_request_fingerprint")
                or original.active_versions != snapshot.get("active_work_versions")):
            raise ValueError("feedback_cycle_continuation_snapshot_mismatch")

    @classmethod
    def advance(cls, previous_snapshot: dict[str, Any], *, contract_id: str,
                clause_sources: dict[str, str], request: tuple) -> "FeedbackCycle":
        """Explicitly admit queued versions. Historical receipts stay historical.

        Retain unresolved old obligations even if absent from a new live view;
        newer live versions supersede old versions of the same stable work ID.
        """
        cls.validate_scope(previous_snapshot, contract_id=contract_id, clause_sources=clause_sources)
        live = _pack(request)
        prior = previous_snapshot["active_request"]
        selected: dict[str, dict[str, Any]] = {}
        previous_status = {item["work_item_id"]: item for item in previous_snapshot.get("active_work_items", [])}
        helper = cls(contract_id, clause_sources, request)
        for item in helper._items(prior):
            if previous_status.get(item["work_item_id"], {}).get("pending", True):
                selected[item["work_item_id"]] = item
        for item in previous_snapshot.get("deferred_next_cycle", []):
            selected[item["obligation_work_item_id"]] = item
        for item in helper._items(live):
            # A stale live view of the prior active version must not erase a
            # newer queued version. Its exact evidence was already discovered.
            queued = selected.get(item["work_item_id"], {})
            if (queued.get("queue_status") == "deferred_next_cycle"
                    and item["input_fingerprint"] == previous_snapshot["active_work_versions"].get(item["work_item_id"])):
                continue
            selected[item["work_item_id"]] = item
        statutes, facts, pending_requests = [], [], []
        for item in selected.values():
            if item["kind"] == "subject_request":
                if not _resolved_subject_request(item, live["subject"], clause_sources):
                    pending_requests.append(item["inputs"]["request_id"])
            elif item["kind"] == "subject_fact":
                facts.append(copy.deepcopy(item["provided_feedback"]))
            elif item["kind"] == "statute_feedback":
                statutes.append(copy.deepcopy(item["provided_feedback"]))
        subject = copy.deepcopy(live["subject"])
        if pending_requests:
            subject["pending_verification_request_ids"] = sorted(set(pending_requests))
            subject["verification_complete"] = False
        focus = {cid for item in [*statutes, *facts] for cid in item.get("clause_ids", [])}
        cycle = cls(contract_id, clause_sources, (live["upstream_fingerprint"], statutes,
            subject, focus, statutes, facts), cycle_number=previous_snapshot["cycle_number"] + 1,
            parent_cycle_id=previous_snapshot["cycle_id"])
        cycle.prior_cycle_archive = copy.deepcopy(previous_snapshot)
        return cycle
