"""Stress-case correctness and non-promotional timing contract."""
import unittest

from alexpy.native_witness_stress import SMOKE_CASES, CASES, generate, measure, replay, verify, suite, run
from alexpy.qa_lab import FAIL, PASS, SKIP, scan_reference, NOT_RUN
from alexpy.native_transition_tree import CTreeNativeQA
from alexpy.transition_tree import SetNativeQA
from alexpy.qa_lab import QAEvent


class TestNativeWitnessStress(unittest.TestCase):
    def test_fixed_seed_and_distribution(self):
        for case in CASES:
            a = generate(case)
            b = generate(case)
            self.assertEqual(a, b)
            failures = sum(e.status == FAIL for e in a[1])
            self.assertEqual(failures > 0, case.fail_percent > 0)
            self.assertEqual(len(a[1]), case.events)
            self.assertTrue(all(0 <= e.test_id < case.active for e in a[1]))

    def test_smoke_correctness_both_routes(self):
        case = SMOKE_CASES[0]
        suites, events = generate(case)
        expected = verify(case, suites, events)
        self.assertEqual(expected, replay("c_tree", case, suites, events))
        self.assertEqual(expected, replay("set", case, suites, events))

    def test_invalid_batch_retains_native_state(self):
        for klass in (CTreeNativeQA, SetNativeQA):
            e = klass([0, 1, 0, 1, 1], 2)
            e.apply_batch([QAEvent(1, FAIL, 7), QAEvent(2, FAIL, 8)])
            snap = (e.snapshot(), e.witnesses(), e.engine.version, e.engine.tick)
            for bad in ([QAEvent(0, FAIL, 99), QAEvent(2, SKIP, 10)],
                        [QAEvent(2, PASS, 12), QAEvent(999, FAIL, 3)],
                        [QAEvent(2, PASS, 12), object()]):
                with self.assertRaises((ValueError, TypeError, IndexError)):
                    e.apply_batch(bad)
                self.assertEqual(snap, (e.snapshot(), e.witnesses(), e.engine.version, e.engine.tick))
            self.assertTrue(e.verify())

    def test_empty_and_duplicate_stress(self):
        for klass in (CTreeNativeQA, SetNativeQA):
            e = klass([0, 0, 0], 1)
            e.apply_batch([])
            self.assertEqual(e.witnesses(), ())
            e.apply_batch([QAEvent(1, FAIL, 10), QAEvent(1, PASS, 20), QAEvent(1, FAIL, 30)])
            self.assertEqual(e.witnesses(), (1,))
            e.apply_batch([QAEvent(1, SKIP, 0)])
            self.assertEqual(e.witnesses(), ())
            self.assertTrue(e.verify())

    def test_smoke_measure_contract(self):
        result = measure(SMOKE_CASES[0], pairs=1)
        self.assertEqual(set(result['samples_ns']), {'c_tree','set'})
        self.assertGreater(result['c_tree_vs_set'], 0)
        self.assertEqual(len(result['samples_ns']['set']), 1)

    def test_bad_cli_parameters_do_not_tune(self):
        with self.assertRaises(ValueError):
            run(sessions=0)
        with self.assertRaises(ValueError):
            run(pairs=0)
        with self.assertRaises(ValueError):
            measure(SMOKE_CASES[0], pairs=0)


if __name__ == '__main__':
    unittest.main()
