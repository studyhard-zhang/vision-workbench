import unittest
from app.preparation import make_class_assignments, PreparationRepository, CLASS_ALGORITHM
from app.storage import ProjectError
from test_preparation import PreparationFixture


class ClassSplitAlgorithmTests(unittest.TestCase):
    def rows(self, labels):
        return [dict(id=str(i), original_sha256=str(i), display_sha256=str(i),
                     state='annotated', boxes=[{'class_id':cid} for cid in classes])
                for i,classes in enumerate(labels)]

    def assign(self, labels, ratios=None):
        rows=self.rows(labels)
        classes=[{'id':cid,'name':cid} for cid in sorted({cid for row in labels for cid in row})]
        result=make_class_assignments(rows,classes,ratios or dict(train=80,val=20,test=0),42)
        for subset in ('train','val'):
            found={box['class_id'] for row in rows if result[row['id']]==subset for box in row['boxes']}
            self.assertEqual(found,{c['id'] for c in classes})
        return rows,classes,result

    def test_rare_classes_both_covered_and_assignment_repeatable(self):
        rows,classes,result=self.assign([['a']]*8+[['b'],['b'],['a','b'],['c'],['c']])
        self.assertEqual(result,make_class_assignments(rows,classes,dict(train=80,val=20,test=0),42))

    def test_overlapping_classes_require_search(self):
        self.assign([['a','b'],['b','c'],['a','c'],['a','b','c']])
        with self.assertRaises(ProjectError) as error:self.assign([['a','b'],['b','c'],['a','c']])
        self.assertEqual(error.exception.code,'SPLIT_CLASS_COVERAGE_UNAVAILABLE')

    def test_single_independent_sample_and_duplicates_are_insufficient(self):
        with self.assertRaises(ProjectError) as error:self.assign([['a'],['a'],['b']])
        self.assertEqual(error.exception.code,'SPLIT_CLASS_SAMPLES_INSUFFICIENT')
        rows=self.rows([['a'],['a'],['b'],['b']]);rows[3]['original_sha256']=rows[2]['original_sha256']
        with self.assertRaises(ProjectError):make_class_assignments(rows,[{'id':c,'name':c} for c in ('a','b')],dict(train=80,val=20,test=0),42)

    def test_test_split_nonempty_and_whole_images_retained(self):
        rows,classes,result=self.assign([['a','b']]*6,dict(train=60,val=20,test=20))
        self.assertEqual(set(result.values()),{'train','val','test'})
        self.assertEqual(len(result),len(rows))

    def test_instance_counts_influence_reservation(self):
        rows,classes,result=self.assign([['a']*10,['a']])
        self.assertEqual(result['0'],'train')
        self.assertEqual(result['1'],'val')


class ClassSplitRepositoryTests(PreparationFixture):
    def test_new_strategy_changes_algorithm_and_retry_is_idempotent(self):
        legacy=self.generate(train_percent=80,val_percent=20,test_percent=0)
        values={key:legacy['split'][key] for key in ('revision','fingerprint','train_percent','val_percent','test_percent','seed')}
        result=self.prep.generate(self.pid,self.a,{**values,'strategy':'classes'})
        self.assertEqual(result['split']['algorithm'],CLASS_ALGORITHM)
        self.assertEqual(result['split']['revision'],legacy['split']['revision']+1)
        self.assertEqual(result,self.prep.generate(self.pid,self.a,{**values,'strategy':'classes'}))
        self.assertTrue(all(result['split']['distribution']['val']['classes'].values()))

    def test_infeasible_new_strategy_preserves_legacy_plan(self):
        for iid in self.ids[1:]:
            value=self.repo.get(self.pid,self.a,iid)
            self.repo.save(self.pid,self.a,iid,{'state':'annotated','revision':value['revision'],'boxes':[value['boxes'][0]]})
        legacy=self.generate(train_percent=80,val_percent=20,test_percent=0)
        values={key:legacy['split'][key] for key in ('revision','fingerprint','train_percent','val_percent','test_percent','seed')}
        with self.assertRaises(ProjectError):self.prep.generate(self.pid,self.a,{**values,'strategy':'classes'})
        self.assertEqual(self.prep.read(self.pid,self.a),legacy)
