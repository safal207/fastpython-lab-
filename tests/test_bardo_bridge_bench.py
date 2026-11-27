"""Ensure new benchmark has honest controls and reproducible symbolic probes."""
import unittest
from alexpy.bardo_bridge_bench import verify, replay, simulate, SMOKE_CASES, ROUTES
from alexpy.qa_batch_bench import generate


class TestBardoBridgeBench(unittest.TestCase):
    def test_smoke_equivalence(self):
        case,batch=SMOKE_CASES[0]
        suites,events=generate(case)
        a=verify(case,batch,suites,events)
        self.assertEqual(a,replay(ROUTES[0],case,batch,suites,events))
        self.assertEqual(a,replay(ROUTES[1],case,batch,suites,events))

    def test_synthetic_no_drift_is_less_expensive_with_sparse_probe(self):
        every=simulate('every','stable')
        fixed=simulate('fixed64','stable')
        self.assertLess(fixed['net_cost'],every['net_cost'])

    def test_synthetic_drift_not_assumed_winner(self):
        # Freeze the known negative controls: do not silently tune heuristic.
        for regime in ('single','reversal'):
            self.assertGreater(simulate('adaptive',regime)['net_cost'],
                               simulate('fixed64',regime)['net_cost'])

    def test_replay_is_deterministic(self):
        for regime in ('stable','single','reversal'):
            self.assertEqual(simulate('adaptive',regime),simulate('adaptive',regime))
