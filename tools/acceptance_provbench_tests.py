"""ProvBench paper-metric diagnostic guardrails, not benchmark/model results."""
import copy,unittest
from acceptance_provbench import metrics,validate_inputs

class ProvBenchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.corpus=[{'provision_id':x,'text':'SYNTHETIC '+x} for x in 'ABCDE']
        self.gold=[{'id':'synthetic','official_split':'test','gold_provisions':[
            {'provision_id':'A','label':'entailment'},{'provision_id':'B','label':'contradiction'},{'provision_id':'C','label':'entailment'}]}]
        self.pred=[{'id':'synthetic','ranked_provision_ids':list('DABCE'),
            'conflicts_on_gold':{'A':'entailment','B':'contradiction','C':'entailment'}}]
    def test_paper_hit_recall_not_fraction_of_three(self):
        r=metrics(self.gold,self.pred,self.corpus,fixture=True)['recommendation']
        self.assertEqual(r['R@1'],0);self.assertEqual(r['R@3'],1);self.assertEqual(r['R@5'],1);self.assertEqual(r['Top-3 EM'],0)
    def test_conflict_uses_gold_pairs_independent_of_ranking(self):
        before=metrics(self.gold,self.pred,self.corpus,fixture=True)['conflict_on_gold_top3']
        self.pred[0]['ranked_provision_ids']=list('EDCBA')
        after=metrics(self.gold,self.pred,self.corpus,fixture=True)['conflict_on_gold_top3']
        self.assertEqual(before,after);self.assertIsNone(after['official_overall_F1'])
    def test_duplicate_candidate_rejected(self):
        self.pred[0]['ranked_provision_ids']=list('AABCD')
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus,fixture=True)
    def test_predictions_must_cover_every_gold_pair(self):
        del self.pred[0]['conflicts_on_gold']['B']
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus,fixture=True)
    def test_project_risk_codes_are_not_binary_benchmark_labels(self):
        self.pred[0]['conflicts_on_gold']['A']='liability_exemption'
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus,fixture=True)
    def test_fixture_cannot_claim_full_official_dataset(self):
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus)
    def test_changed_split_rejected(self):
        self.gold[0]['official_split']='train'
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus,fixture=True)
    def test_missing_clause_prediction_rejected(self):
        self.pred[0]['id']='different'
        with self.assertRaises(ValueError):validate_inputs(self.gold,self.pred,self.corpus,fixture=True)
if __name__=='__main__':unittest.main()
