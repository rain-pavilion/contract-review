"""Durability, exclusive claims, restart recovery and idempotent archive checks."""
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path

from zhilv_qiling_v2.workspace_queue import QueueStore, QueueWorker, QueueError, PermanentTaskError


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = QueueStore(Path(self.temp.name) / "queue.sqlite")

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, n=2):
        key = self.store.create_batch({"engine": "multi", "llm": False})["batch_id"]
        tasks = self.store.add_many(key, [(f"contract{i}.txt", "txt", b"contract") for i in range(n)])
        return key, tasks

    def test_draft_waits_for_submit_and_survives_reopen(self):
        key, _ = self.batch()
        self.assertIsNone(self.store.claim_next("one"))
        restored = QueueStore(self.store.path)
        self.assertEqual(restored.get_batch(key)["total"], 2)
        restored.start_batch(key)
        task = restored.claim_next("one")
        self.assertEqual(task["content"], b"contract")
        self.assertEqual(task["options"], {"engine": "multi", "llm": False,
            "contract_type": "auto", "review_position": "neutral"})
        self.assertNotIn("content", restored.get_batch(key)["items"][0])

    def test_claims_are_exclusive_across_connections(self):
        key, _ = self.batch(8)
        self.store.start_batch(key)
        found = []
        lock = threading.Lock()
        def claim(i):
            task = QueueStore(self.store.path).claim_next(str(i))
            if task:
                with lock: found.append(task["task_id"])
        threads = [threading.Thread(target=claim, args=(i,)) for i in range(12)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(found), 8)
        self.assertEqual(len(set(found)), 8)

    def test_expired_claim_recovers_but_live_claim_is_not_stolen(self):
        key, _ = self.batch(1)
        self.store.start_batch(key)
        old = self.store.claim_next("old", lease_seconds=60)
        self.assertEqual(self.store.recover(), 0)
        self.assertIsNone(self.store.claim_next("new"))
        with closing(sqlite3.connect(self.store.path)) as db, db:
            db.execute("UPDATE workspace_tasks SET lease_until=0")
        fresh = QueueStore(self.store.path).claim_next("new")
        self.assertEqual(fresh["task_id"], old["task_id"])
        self.assertEqual(fresh["attempts"], 2)
        self.assertFalse(self.store.complete(old["task_id"], "bad", worker_id="old"))
        self.assertTrue(self.store.complete(fresh["task_id"], "report", {"score":80}, "new"))
        self.assertEqual(self.store.get_batch(key)["status"], "completed")

    def test_cancel_does_not_interrupt_current_task_and_retry_preserves_completed(self):
        key, _ = self.batch(3)
        self.store.start_batch(key)
        running = self.store.claim_next("worker")
        self.store.cancel_batch(key)
        batch = self.store.get_batch(key)
        self.assertEqual(batch["counts"], {"cancelled":2, "running":1})
        self.assertIsNone(self.store.claim_next("other"))
        self.store.complete(running["task_id"], "report", worker_id="worker")
        self.assertEqual(self.store.get_batch(key)["status"], "cancelled")
        self.store.retry_batch(key)
        self.assertEqual(self.store.get_batch(key)["counts"], {"completed":1,"pending":2})

    def test_transient_retry_and_permanent_failure(self):
        key, _ = self.batch(1)
        self.store.start_batch(key)
        task = self.store.claim_next("worker")
        self.store.fail(task["task_id"], "temporary", worker_id="worker")
        self.assertEqual(self.store.get_batch(key)["counts"], {"pending":1})
        with closing(sqlite3.connect(self.store.path)) as db, db:
            db.execute("UPDATE workspace_tasks SET available_at=0")
        def broken(_): raise PermanentTaskError("unreadable")
        QueueWorker(self.store, broken).run_once()
        self.assertEqual(self.store.get_batch(key)["status"], "failed")
        self.assertIn("unreadable", self.store.get_batch(key)["items"][0]["error"])

    def test_repeated_crashes_stop_at_retry_limit(self):
        key, _ = self.batch(1)
        self.store.start_batch(key)
        for attempt in range(1,4):
            task = self.store.claim_next(str(attempt))
            self.assertEqual(task["attempts"], attempt)
            with closing(sqlite3.connect(self.store.path)) as db, db:
                db.execute("UPDATE workspace_tasks SET lease_until=0")
            self.store.recover()
        self.assertIsNone(self.store.claim_next("fourth"))
        self.assertEqual(self.store.get_batch(key)["status"], "failed")
        self.store.retry_batch(key)
        self.assertEqual(self.store.claim_next("manual-retry")["attempts"],1)

    def test_worker_success_pagination_and_upload_validation(self):
        key, _ = self.batch(3)
        with self.assertRaises(QueueError):
            self.store.add_task(key, "bad.exe", "exe", b"no")
        self.store.start_batch(key)
        with self.assertRaises(QueueError):
            self.store.add_task(key, "late.txt", "txt", b"late")
        worker = QueueWorker(self.store, lambda t:(t["task_id"], {"score":99}))
        while worker.run_once(): pass
        batch = self.store.get_batch(key, page=2, size=2)
        self.assertEqual(batch["status"], "completed")
        self.assertEqual(len(batch["items"]), 1)
        self.assertEqual(batch["items"][0]["result_summary"]["score"], 99)
        self.assertEqual(self.store.list_batches()[0]["total"], 3)


if __name__ == "__main__":
    unittest.main()
