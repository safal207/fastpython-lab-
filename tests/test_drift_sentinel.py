import unittest

from alexpy.drift_sentinel import ProbeCosts,DriftSentinel,BardoTrustControl
from alexpy.drift_sentinel_bench import hidden_states,evaluate,train_select,run,KINDS

class TestDriftSentinel(unittest.TestCase):
    def test_observations_are_paid_and_ordered(self):
        c=DriftSentinel()
        self.assertTrue(c.due(0))
        i=c.observe(0,True)
        self.assertFalse(c.due(1))
        self.assertTrue(c.due(i))
        with self.assertRaises(ValueError):c.observe(i-1,False)
        with self.assertRaises(ValueError):c.observe(0,False)
        with self.assertRaises(ValueError):c.observe(i,1)
        c.observe(i,False)
        self.assertEqual(c.observed_flips,1)
        self.assertGreater(c.rate,0)

    def test_hidden_stream_repeatable_and_no_policy_peeking(self):
        for kind in KINDS:
            a=hidden_states(41,kind)
            self.assertEqual(a,hidden_states(41,kind))
            self.assertEqual(len(a),1536)
            self.assertTrue(all(type(x) is bool for x in a))
            for policy in ('sentinel','bardo_trust','fixed8','fixed32'):
                result=evaluate(policy,a)
                self.assertGreater(result['probes'],0)
                self.assertGreaterEqual(result['cost'],0)
        self.assertNotEqual(hidden_states(41,'abrupt'),hidden_states(43,'abrupt'))

    def test_training_is_separate_from_holdout(self):
        p,results=train_select()
        self.assertIn(p,results)
        r=run(smoke=True)
        self.assertEqual(r['frozen_selected_fixed'],p)
        self.assertEqual(len(r['rows']),4*len(KINDS))
        self.assertTrue(all(row['seed'] not in r['train_seeds'] for row in r['rows']))

    def test_bardo_control_cannot_observe_unpaid_future(self):
        c=BardoTrustControl()
        d=c.observe(0,True)
        with self.assertRaises(ValueError):c.observe(d-1,False)
        with self.assertRaises(ValueError):c.observe(d,1)
        c.observe(d,False)
        self.assertEqual(c.flips,1)

    def test_zero_probe_cost_and_guards(self):
        c=DriftSentinel(ProbeCosts(probe=0))
        self.assertEqual(c.observe(0,True),8)
        for kwargs in ({'probe':float('nan')},{'stale':-1},{'min_interval':0},
                       {'max_interval':2,'min_interval':8}):
            with self.assertRaises(ValueError):ProbeCosts(**kwargs)
        with self.assertRaises(ValueError):DriftSentinel(alpha=0)
        with self.assertRaises(ValueError):DriftSentinel(floor=float('inf'))
