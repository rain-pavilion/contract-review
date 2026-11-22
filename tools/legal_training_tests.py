"""Source, label and leakage guards; all data below are synthetic test fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from legal_training_data import prepare, sha256
from legal_training_lcr_cn import family, references, provision_families


class OriginalLegalSourceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'dataset_name':'synthetic_source_guard_fixture',
            'task':'fixture_binary_relation_not_real_legal_training',
            'official_source':'https://example.invalid/fixture',
            'license':{'training_permitted':True,'url':'https://example.invalid/fixture-license'},
            'label_provenance':{'kind':'original_released_labels','description':'synthetic fixture labels'},
            'fields':{'id':'id','left_text':'a','right_text':'b','left_family':'fa','right_family':'fb','label':'y'},
            'label_mapping':{'conflict':'contradict','no_conflict':'neutral'},
            'split_policy':'provided_disjoint_families','files':[]}
        self.records = {}
        for split in ('train','valid','test'):
            self.records[split] = [{'id':split+str(i),'a':split+' left '+str(i),'b':split+' right '+str(i),
                'fa':split+'-law','fb':split+'-law','y':label} for i,label in enumerate(('conflict','no_conflict'))]
        self.write()

    def write(self):
        self.manifest['files']=[]
        for split,rows in self.records.items():
            path=self.root/(split+'.jsonl')
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
            self.manifest['files'].append({'path':path.name,'sha256':sha256(path.read_bytes()),'split':split})
        path=self.root/'manifest.json'
        path.write_text(json.dumps(self.manifest),encoding='utf-8')
        return path

    def test_original_label_id_and_split_are_preserved(self):
        result=prepare(self.write())
        self.assertFalse(result['law_family_overlap'])
        self.assertFalse(result['project_human_gold'])
        for split,rows in result['splits'].items():
            self.assertEqual({r['original_split'] for r in rows},{split})
            self.assertEqual([r['original_id'] for r in rows],[split+'0',split+'1'])
            self.assertEqual([r['original_label'] for r in rows],['conflict','no_conflict'])

    def test_changed_original_bytes_fail_closed(self):
        path=self.write()
        with (self.root/'train.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):prepare(path)

    def test_unverified_training_permission_rejected(self):
        self.manifest['license']['training_permitted']=False
        with self.assertRaisesRegex(ValueError,'licence'):prepare(self.write())

    def test_project_generated_labels_rejected(self):
        self.manifest['label_provenance']['kind']='project_rules'
        with self.assertRaisesRegex(ValueError,'original released labels'):prepare(self.write())

    def test_unmapped_original_class_is_not_invented(self):
        self.records['train'][0]['y']='unspecified'
        with self.assertRaisesRegex(ValueError,'unmapped original label'):prepare(self.write())

    def test_shared_law_family_cannot_cross_holdout(self):
        for row in self.records['train']:row['fa']=row['fb']='test-law'
        with self.assertRaisesRegex(ValueError,'nonempty law-family-disjoint'):prepare(self.write())

    def test_duplicate_training_copy_is_removed_before_original_test(self):
        row=copy.deepcopy(self.records['test'][0])
        row['id']='duplicate-training-copy'
        self.records['train'].append(row)
        result=prepare(self.write())
        self.assertIn('test0',[r['original_id'] for r in result['splits']['test']])
        self.assertNotIn('duplicate-training-copy',[r['original_id'] for r in result['splits']['train']])

    def test_reversed_pair_with_conflicting_label_is_rejected(self):
        row=copy.deepcopy(self.records['train'][0])
        row['a'],row['b']=row['b'],row['a']
        row['y']='no_conflict'
        self.records['train'].append(row)
        with self.assertRaisesRegex(ValueError,'conflicting original labels'):prepare(self.write())

    def test_no_degenerate_single_class_training(self):
        self.manifest['label_mapping']={'conflict':'contradict','no_conflict':'contradict'}
        with self.assertRaisesRegex(ValueError,'every released target class'):prepare(self.write())

    def test_source_regulation_versions_share_query_family(self):
        self.assertEqual(family('某地管理办法（2020年修订）'),family('某地管理办法（2024年修订）'))

    def test_unknown_superior_law_schema_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError,'unknown original'):references([{'unknown_text':'source'}])

    def test_only_original_reference_prose_is_joined(self):
        text,titles=references([{'title':'来源法','clause':'第一条 原始条文。','golden_content':'不能作负例。'},
            {'title':'来源法','clause':'第二条 原始条文。','conflict_description':'不能泄漏解释。'}])
        self.assertEqual(text,'第一条 原始条文。\n第二条 原始条文。')
        self.assertEqual(titles,'来源法')

    def test_exact_revised_provision_merges_even_across_titles(self):
        rows=[{'query':'该款有错误甲。','expert_revised_content_for_family_only':'相同原始条文。','original_title':'甲法'},
            {'query':'该款有错误乙。','expert_revised_content_for_family_only':'相同原始条文。','original_title':'乙法'}]
        groups,audit=provision_families(rows)
        self.assertEqual(groups[0],groups[1])
        self.assertFalse(audit['revision_text_model_input'])


if __name__=='__main__':unittest.main()
