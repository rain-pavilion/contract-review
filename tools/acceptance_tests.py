"""Guardrails for label and formal-benchmark acceptance tooling."""
import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from acceptance_annotations import SCHEMA,main as annotations_main,validate
from acceptance_common import digest
from acceptance_official_benchmarks import template,check
from acceptance_annotation_pack import create_pack,check_pack,records
from acceptance_lawbench import TASKS,prompt

class GuardrailTests(unittest.TestCase):
    def row(self):
        text='独立人工标注待办样本，不是金标。'
        return {'schema_version':SCHEMA,'id':'sample','text':text,'text_sha256':digest(text),
            'source':{'origin':'synthetic annotation workflow test','group_id':'g1','license_or_permission':'test fixture','collected_at':'2026-11-14'},
            'status':'unlabeled','split':'test','annotations':[],'reviews':[]}
    def test_unlabeled_is_not_gold(self):
        r=validate([self.row()],require_gold=True);self.assertFalse(r['gold_ready']);self.assertTrue(r['errors'])
    def test_duplicate_across_split_rejected(self):
        a=self.row();b=copy.deepcopy(a);b.update(id='second',split='train');b['source']['group_id']='g2'
        self.assertTrue(any('duplicate across splits' in e for e in validate([a,b])['errors']))
    def test_group_leakage_rejected(self):
        a=self.row();b=copy.deepcopy(a);b.update(id='second',split='train',text='另一个文本');b['text_sha256']=digest(b['text'])
        self.assertTrue(any('source group' in e for e in validate([a,b])['errors']))
    def test_changed_text_rejected(self):
        a=self.row();a['text']+='变更';self.assertTrue(any('hash changed' in e for e in validate([a])['errors']))
    def test_rules_cannot_be_human_gold(self):
        a=self.row();a['status']='adjudicated';a['reviews']=[{'method':'rule','reviewer_id':'automatic-rule'}]
        r=validate([a],require_gold=True);self.assertFalse(r['gold_ready']);self.assertTrue(any('manual review' in e for e in r['errors']))
    def test_official_template_cannot_run(self):
        for name in ('LawBench','ProvBench'):
            m=template(name);self.assertEqual(m['status'],'not_run');self.assertTrue(check(m))
    def test_training_overlap_rejected(self):
        a=self.row()
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'inventory.jsonl';f.write_text(json.dumps(a),encoding='utf-8')
            self.assertTrue(any('test leakage' in e for e in validate([a],f)['errors']))
    def test_empty_training_inventory_not_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'inventory.jsonl';f.write_text('',encoding='utf-8')
            self.assertTrue(any('inventory is empty' in e for e in validate([self.row()],f)['errors']))
    def test_candidate_pack_stays_unlabeled(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'pack';r=create_pack(folder)
            self.assertEqual(r['status'],'passed');self.assertEqual(r['records'],18)
            self.assertEqual(len(r['contract_types']),6);self.assertFalse(r['gold_ready'])
            self.assertEqual(check_pack(folder)['freeze_unlabeled'],'rejected_as_expected')
            for row in records():self.assertEqual(row['annotations'],[]);self.assertEqual(row['reviews'],[])
    def test_lawbench_prompt_never_appends_reference(self):
        row={'instruction':'official instruction','question':'official question','answer':'SECRET_REFERENCE'}
        self.assertEqual(prompt(row),'official instruction\nofficial question')
        self.assertNotIn(row['answer'],prompt(row));self.assertEqual(len(TASKS),20)
    def test_cli_freeze_refuses_unlabeled_pack(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'pack';create_pack(folder)
            inventory=Path(d)/'empty_inventory_fixture.jsonl';inventory.write_text('',encoding='utf-8')
            output=Path(d)/'freeze_rejection.json'
            args=['acceptance_annotations.py','freeze','--data',str(folder/'candidates.jsonl'),'--training-inventory',str(inventory),'--out',str(output)]
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):code=annotations_main()
            result=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(code,2);self.assertEqual(result['status'],'not_run');self.assertFalse(result['gold_ready'])
            self.assertTrue(any('not adjudicated' in e for e in result['errors']))
            self.assertFalse(output.with_suffix('.jsonl').exists())
if __name__=='__main__':unittest.main()
