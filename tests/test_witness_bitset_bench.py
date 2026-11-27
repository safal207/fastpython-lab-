"""Fair three-route benchmark contracts; timing is advisory only."""
import unittest
from alexpy.witness_bitset_bench import ROUTES, measure, run, verify, replay
from alexpy.native_witness_stress import SMOKE_CASES, generate


class TestBitsetBenchmark(unittest.TestCase):
    def test_smoke_matches_independent_oracle(self):
        case=SMOKE_CASES[0]
        suites,events=generate(case)
        expected=verify(case,suites,events)
        for route in ROUTES:
            self.assertEqual(replay(route,case,suites,events),expected)

    def test_smoke_timing_contract(self):
        row=measure(SMOKE_CASES[0],pairs=1)
        self.assertEqual(set(row['samples_ns']),set(ROUTES))
        self.assertGreater(row['bitset_vs_tree'],0)
        self.assertGreater(row['bitset_vs_set'],0)

    def test_reject_invalid_session_counts(self):
        for kwargs in ({'sessions':0},{'pairs':0}):
            with self.assertRaises(ValueError):run(**kwargs)
