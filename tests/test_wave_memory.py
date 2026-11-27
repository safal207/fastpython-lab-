"""Unit, metamorphic and safety tests for light/sound wave propagation."""
import json
import math
import unittest

from alexpy.ant_memory import Workload
from alexpy.wave_memory import WaveMemoryGraph


class TestWaveMemory(unittest.TestCase):
    def setUp(self):
        self.s = Workload(128, 64, 1, 2, 2)
        self.near = Workload(256, 64, 1, 2, 2)
        self.far = Workload(8192, 64, 1, 2, 2)

    def test_unknown_fails_closed_to_manual(self):
        for mode in ('light','sound'):
            m=WaveMemoryGraph(mode=mode)
            self.assertEqual(m.choose(self.far), 'manual')
            with self.assertRaises(RuntimeError): m.choose(self.s,allowed=('compiled',))

    def test_light_front_is_min_plus_latency(self):
        m=WaveMemoryGraph(mode='light',attenuation=.2,max_hops=2,min_gain=1)
        m.observe(self.s,'partition',elapsed_ns=10,reference_ns=100,verified=True)
        self.assertAlmostEqual(m.field(self.near)['partition'], -math.log(10)+.2,places=10)
        self.assertEqual(m.choose(self.near),'partition')
        self.assertEqual(m.choose(self.far),'manual')

    def test_sound_multiple_sources(self):
        m=WaveMemoryGraph(mode='sound',max_hops=3,min_gain=1)
        m.observe(self.s,'partition',elapsed_ns=20,reference_ns=100,verified=True)
        first=m.field(self.near)['partition']
        m.observe(self.near,'partition',elapsed_ns=10,reference_ns=100,verified=True)
        second=m.field(self.near)['partition']
        self.assertLess(second,first)
        self.assertEqual(m.choose(self.near),'partition')

    def test_invalid_quarantines_exact_cell_and_clears_cache(self):
        m=WaveMemoryGraph(min_gain=1)
        m.observe(self.s,'compiled',elapsed_ns=1,reference_ns=100,verified=True)
        self.assertEqual(m.choose(self.s),'compiled')
        with self.assertRaisesRegex(ValueError,'quarantined'):
            m.observe(self.s,'compiled',elapsed_ns=1,reference_ns=100,verified=False)
        self.assertEqual(m.choose(self.s),'manual')
        self.assertNotIn('compiled',m.field(self.s))
        with self.assertRaises(ValueError):
            m.observe(self.s,'compiled',elapsed_ns=1,reference_ns=100,verified=True)

    def test_decay_and_evidence_freeze(self):
        m=WaveMemoryGraph(age_penalty=.3,min_gain=1)
        m.observe(self.s,'partition',elapsed_ns=10,reference_ns=100,verified=True)
        start=m.field(self.s)['partition']
        frozen=m.freeze((self.s,self.near))
        self.assertEqual(len(frozen),2)
        self.assertEqual(m.snapshot()['compiled_decisions'],2)
        m.advance(3)
        self.assertEqual(m.snapshot()['compiled_decisions'],0)
        self.assertGreater(m.field(self.s)['partition'],start)
        self.assertEqual(m.snapshot()['state_action_edges'],1)

    def test_rejects_bad_parameters_and_invalid_observations(self):
        for kwargs in ({'mode':'heat'},{'attenuation':0},{'max_hops':-1},
                       {'min_gain':0.9},{'age_penalty':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                WaveMemoryGraph(**kwargs)
        m=WaveMemoryGraph()
        for n in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                m.observe(self.s,'manual',elapsed_ns=n,reference_ns=1,verified=True)
        self.assertEqual(m.snapshot()['state_action_edges'],0)

    def test_allowed_mask_and_sparse_memory(self):
        m=WaveMemoryGraph(mode='sound')
        m.observe(self.s,'partition',elapsed_ns=10,reference_ns=100,verified=True)
        self.assertEqual(m.choose(self.s,allowed=('manual',)),'manual')
        with self.assertRaises(ValueError):m.choose(self.s,allowed=())
        with self.assertRaises(ValueError):m.choose(self.s,allowed=('unknown',))
        self.assertEqual(m.snapshot()['state_action_edges'],1)
        json.dumps(m.snapshot())

    def test_compiled_min_plus_matches_dijkstra_on_many_cells(self):
        import random
        rng=random.Random(716)
        m=WaveMemoryGraph(mode='light',attenuation=.23,max_hops=4,min_gain=1)
        for _ in range(15):
            w=Workload(2**rng.randint(5,10),2**rng.randint(3,7),
                       2**rng.randint(0,2),2**rng.randint(0,3),2**rng.randint(0,3))
            m.observe(w,'partition',elapsed_ns=rng.randint(10,300),reference_ns=100,verified=True)
        for _ in range(12):
            w=Workload(2**rng.randint(5,10),2**rng.randint(3,7),
                       2**rng.randint(0,2),2**rng.randint(0,3),2**rng.randint(0,3))
            # Workload changes <= size guaranteed by chosen exponents.
            expected=m._light_arrival_reference(w.cell(),'partition')
            actual=m._light_arrival(w.cell(),'partition')
            if math.isinf(expected):self.assertTrue(math.isinf(actual))
            else:self.assertAlmostEqual(actual,expected,places=10)

    def test_light_monotonicity_and_equivalent_1d_distance(self):
        m=WaveMemoryGraph(mode='light',attenuation=.3,max_hops=4,min_gain=1)
        m.observe(self.s,'partition',elapsed_ns=10,reference_ns=100,verified=True)
        base=-math.log(10)
        for k in (0,1,2,3):
            w=Workload(128*(2**k),64,1,2,2)
            self.assertAlmostEqual(m.field(w)['partition'],base+k*.3,places=10)


if __name__=='__main__': unittest.main()
