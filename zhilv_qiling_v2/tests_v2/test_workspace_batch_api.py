"""Real HTTP queue submission/restart plus report timing and fact import boundaries."""
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import date
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from zhilv_qiling_v2 import api
from zhilv_qiling_v2.config import Config
from zhilv_qiling_v2.tests_v2.test_web_workspace import SAMPLE
from zhilv_qiling_v2.workspace_queue import PermanentTaskError, QueueStore


class BatchApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = api._SERVER_CONFIG
        self.config = Config(store_path=str(Path(self.temp.name)/"workspace.sqlite"),
            subject_facts_path=str(Path(self.temp.name)/"facts.json"), ner_use_bert=False,
            ner_enable_model=False, use_lsa=False, entailment_backend="rules", enable_llm=False)
        api._SERVER_CONFIG = self.config
        self.start()

    def start(self):
        self.server = api.serve("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(5)

    def tearDown(self):
        self.stop()
        api._SERVER_CONFIG = self.old
        self.temp.cleanup()

    def request(self, path, body=None, content_type="application/json"):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        req = urllib.request.Request(self.base+path, data=data, headers={"Content-Type":content_type})
        try: response=urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as exc: response=exc
        with response: return response.status, json.loads(response.read())

    def upload(self, key, name="contract.txt"):
        boundary="DurableBatchTest"
        raw=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n\r\n'+SAMPLE+f'\r\n--{boundary}--\r\n').encode()
        return self.request(f"/api/workspace/batches/{key}/tasks", raw, f"multipart/form-data; boundary={boundary}")

    def test_explicit_model_request_without_endpoint_is_rejected_before_review(self):
        boundary = 'MissingMainModel'
        multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="llm"\r\n\r\n1\r\n'
                     f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{SAMPLE}\r\n'
                     f'--{boundary}--\r\n').encode()
        cases = [('/api/review', {'text': SAMPLE, 'llm': '1'}, 'application/json'),
                 ('/api/review?llm=1', {'text': SAMPLE}, 'application/json'),
                 ('/api/review', multipart, f'multipart/form-data; boundary={boundary}'),
                 ('/api/workspace/batches', {'llm': '1'}, 'application/json'),
                 ('/api/workspace/contract-packages/missing-package/review', {'llm': '1'}, 'application/json')]
        with patch('zhilv_qiling_v2.engine.ReviewEngine.run', side_effect=AssertionError('must not run rules silently')):
            for url, body, content_type in cases:
                with self.subTest(url=url, content_type=content_type):
                    status, result = self.request(url, body, content_type)
                    self.assertEqual(status, 503, result)
                    self.assertTrue(result['llm_requested'])
                    self.assertFalse(result['review_executed'])

    def test_persisted_model_task_without_endpoint_fails_explicitly(self):
        store = QueueStore(self.config.store_path)
        batch = store.create_batch({'engine': 'multi', 'llm': '1'})
        store.add_task(batch['batch_id'], 'model-required.txt', 'txt', SAMPLE.encode())
        store.start_batch(batch['batch_id'])
        task = store.claim_next('missing-endpoint-test')
        with patch('zhilv_qiling_v2.engine.ReviewEngine.run', side_effect=AssertionError('must not run rules silently')):
            with self.assertRaisesRegex(PermanentTaskError, '未改为规则审查'):
                api._run_workspace_task(task, self.config)

    def test_real_upload_draft_survives_server_restart_then_completes(self):
        status, batch=self.request('/api/workspace/batches', {"engine":"multi","llm":"0"})
        self.assertEqual(status,200)
        key=batch['batch_id']
        self.assertEqual(self.upload(key)[0],200)
        self.assertEqual(self.upload(key,'second.txt')[0],200)
        self.stop(); self.start()
        status, draft=self.request(f'/api/workspace/batches/{key}')
        self.assertEqual(draft['status'],'draft')
        self.assertEqual(draft['total'],2)
        self.request(f'/api/workspace/batches/{key}/start',{})
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            _, result=self.request(f'/api/workspace/batches/{key}')
            if result['status'] in ('completed','failed'): break
            time.sleep(.1)
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['counts'],{'completed':2})
        for task in result['items']:
            self.assertEqual(task['report_id'],task['task_id'])
            _, report=self.request('/api/workspace/reports/'+task['report_id'])
            self.assertTrue(report['saved'])
            self.assertIn('timings',report['stats'])

    def test_archive_recovery_is_idempotent(self):
        config=replace(self.config, store_path=str(Path(self.temp.name)/'recovery.sqlite'))
        store=QueueStore(config.store_path)
        batch=store.create_batch()['batch_id']
        task=store.add_task(batch,'contract.txt','txt',SAMPLE.encode())
        store.start_batch(batch)
        claimed=store.claim_next('simulated-crashed-worker')
        report_id, _=api._run_workspace_task(claimed,config)
        with patch('zhilv_qiling_v2.engine.ReviewEngine.run',side_effect=AssertionError('must not run twice')):
            recovered_id, summary=api._run_workspace_task(claimed,config)
        self.assertEqual(report_id,recovered_id)
        self.assertIn('risk_count',summary)

    def test_old_worker_cannot_rewrite_archive_after_lease_handoff(self):
        # Keep this deterministic lease transfer separate from the live API
        # worker; this test explicitly owns both simulated lease holders.
        config = replace(self.config, store_path=str(Path(self.temp.name)/'lease.sqlite'))
        store = QueueStore(config.store_path)
        batch = store.create_batch()['batch_id']
        task = store.add_task(batch, 'lease-handoff.txt', 'txt', SAMPLE.encode())
        store.start_batch(batch)
        claimed = store.claim_next('old-worker')
        self.assertEqual(claimed['task_id'], task['task_id'])
        original_save = api._save_workspace_report

        def save_then_transfer(payload, config, **kwargs):
            original_save(payload, config, **kwargs)
            with api.closing(api._workspace_db(config)) as connection, connection:
                connection.execute("UPDATE workspace_tasks SET worker_id='new-worker' WHERE task_id=?",
                                   (claimed['task_id'],))
                saved = json.loads(connection.execute(
                    'SELECT payload FROM workspace_reports WHERE report_id=?',
                    (claimed['task_id'],)).fetchone()['payload'])
                saved['new_owner_sentinel'] = True
                connection.execute('UPDATE workspace_reports SET payload=? WHERE report_id=?',
                                   (json.dumps(saved), claimed['task_id']))

        with patch.object(api, '_save_workspace_report', side_effect=save_then_transfer):
            with self.assertRaises(OSError):
                api._run_workspace_task(claimed, config)
        with api.closing(api._workspace_db(config)) as connection:
            saved = json.loads(connection.execute('SELECT payload FROM workspace_reports WHERE report_id=?',
                               (claimed['task_id'],)).fetchone()['payload'])
        self.assertTrue(saved.get('new_owner_sentinel'), saved)

    def test_server_timing_is_saved_and_separates_engine_scope(self):
        status,report=self.request('/api/review',{'text':SAMPLE,'store':True,'engine':'single','llm':'0'})
        self.assertEqual(status,200,report)
        timing=report['stats']['timings']
        self.assertGreaterEqual(timing['server_total_seconds'],timing['review_seconds'])
        self.assertGreater(timing['archive_seconds'],0)
        _,saved=self.request('/api/workspace/reports/'+report['report_id'])
        self.assertEqual(saved['stats']['timings'],timing)

    def test_fact_import_rejects_unproven_and_request_paths(self):
        status,result=self.request('/api/workspace/subject-facts',{'records':[{'subject_name':'Fictional test'}], 'path':'elsewhere.json'})
        self.assertEqual(status,400)
        self.assertFalse(Path(self.config.subject_facts_path).exists())
        status,result=self.request('/api/workspace/subject-facts')
        self.assertEqual(status,200)
        self.assertEqual(result['total'],0)

    def test_reviewed_evidence_import_is_persistent_and_deduplicated(self):
        record = {
            'subject_name': 'Synthetic evidence test company',
            'fact_type': 'registration', 'status': 'unknown',
            'scope': 'HTTP integration fixture; no real registration claim',
            'evidence_ref': 'synthetic-http-test',
            'source': {'url': 'https://example.invalid/synthetic-evidence',
                       'publisher': 'Synthetic test source',
                       'reviewer': 'test-fixture-not-a-person',
                       'verification_method': 'Synthetic integration fixture only',
                       'verified_at': date.today().isoformat()},
        }
        for _ in range(2):
            status, result = self.request('/api/workspace/subject-facts', {'records': [record]})
            self.assertEqual(status, 200, result)
            self.assertEqual(result['total'], 1)
        self.stop(); self.start()
        status, result = self.request('/api/workspace/subject-facts')
        self.assertEqual(status, 200)
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['records'][0]['status'], 'unknown')
        self.assertEqual(result['records'][0]['source']['publisher'], 'Synthetic test source')
        self.assertEqual(result['records'][0]['authenticity'],
                         'supplied_provenance_not_independently_verified_by_software')


if __name__=='__main__': unittest.main()
