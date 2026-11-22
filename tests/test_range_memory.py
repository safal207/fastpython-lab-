"""Memory safety, quarantine, switching evidence and blind train separation."""
import math
import unittest
from alexpy.range_memory import RangeRouteMemory
from alexpy.range_routes import RangeWorkload, ROUTES
from alexpy.range_bench import run_session,switch_probe,measure_routes,_csv

class TestRangeMemory(unittest.TestCase):
    def setUp(self):
        self.a=RangeWorkload(128,1,1,2,'short')
        self.b=RangeWorkload(128,1,8,2,'short')

    def test_learning_changes_choice(self):
        for mode in ('ant','wave'):
            m=RangeRouteMemory(mode=mode,max_hops=5)
            self.assertEqual(m.choose(self.a),'direct')
            m.observe(self.a,'prefix',elapsed_ns=20,direct_ns=1000,verified=True)
            self.assertEqual(m.choose(self.a),'prefix')
            self.assertEqual(m.snapshot()['observations'],1)

    def test_quarantine_blocks_wrong_route(self):
        for mode in ('ant','wave'):
            m=RangeRouteMemory(mode=mode)
            with self.assertRaises(ValueError):
                m.observe(self.a,'fenwick',elapsed_ns=1,direct_ns=10,verified=False)
            with self.assertRaises(ValueError):
                m.observe(self.a,'fenwick',elapsed_ns=1,direct_ns=10,verified=True)
            self.assertNotEqual(m.choose(self.a),'fenwick')

    def test_quarantined_direct_not_proposed(self):
        m=RangeRouteMemory()
        with self.assertRaises(ValueError):
            m.observe(self.a,"direct",elapsed_ns=1,direct_ns=1,verified=False)
        with self.assertRaises(RuntimeError):
            m.choose(self.a)

    def test_nonfinite_and_negative_rejected(self):
        m=RangeRouteMemory()
        for ns in (math.nan,math.inf,0,-1,True):
            with self.assertRaises(ValueError):
                m.observe(self.a,'blocks',elapsed_ns=ns,direct_ns=100,verified=True)
        self.assertEqual(m.snapshot()['observations'],0)

    def test_switch_provenance_and_direction(self):
        m=RangeRouteMemory()
        evidence=switch_probe(self.a,self.b,'direct','prefix')
        self.assertGreater(evidence['setup_ns'],0)
        m.observe_switch(self.a,self.b,'direct','prefix',
                         setup_ns=evidence['setup_ns'],tick_ns=evidence['tick_ns'],verified=True)
        self.assertIsNone(m.switch_cost(self.b,self.a,'direct','prefix'))
        self.assertEqual(m.switch_cost(self.a,self.b,'direct','prefix')['samples'],1)
        with self.assertRaises(ValueError):
            m.observe_switch(self.a,self.b,'direct','prefix',setup_ns=0,tick_ns=0,verified=False)

    def test_train_holdout_cells_disjoint_and_immutable(self):
        report=run_session(train_sizes=(128,512),holdout_sizes=(256,),
                           changes=(1,),queries=(1,8),spans=('short','wide'),
                           ticks=2,pairs=2,probe=True)
        self.assertTrue(report['no_holdout_training'])
        self.assertEqual(len(report['holdout']),4)
        self.assertEqual(len(report['switch_probes']),16)
        self.assertEqual(report['memory']['ant']['real_switch_edges'],16)
        self.assertEqual(report['memory']['wave']['real_switch_edges'],16)
        for row in report['holdout']:
            self.assertEqual(row['gate'],'PASS')
            self.assertEqual(set(row['choices']),
                {'fixed_direct','fixed_train','simple_rule','ant','wave'})
            self.assertIn(row['oracle_route'],ROUTES)

    def test_coarse_cell_leak_rejected(self):
        with self.assertRaises(ValueError):
            run_session(train_sizes=(128,),holdout_sizes=(129,),changes=(1,),
                        queries=(1,),spans=('short',),ticks=1,pairs=1,probe=False)

    def test_exact_inputs_and_bad_args(self):
        self.assertEqual(_csv('1,8'),(1,8))
        for s in ('0,1','1,1',''):
            with self.assertRaises(ValueError):_csv(s)
        with self.assertRaises(ValueError):RangeRouteMemory(mode='invalid')
        with self.assertRaises(ValueError):RangeRouteMemory(max_hops=-1)
        with self.assertRaises(ValueError):RangeRouteMemory(wave_penalty=-1)
        with self.assertRaises(ValueError):RangeRouteMemory(min_gain=0.3)

if __name__=='__main__':unittest.main()
