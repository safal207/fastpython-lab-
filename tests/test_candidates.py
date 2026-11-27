"""Semantics gate for both opt-in interventions, including adversarial inputs."""
import random
import unittest

from alexpy import _native, sum_squares
from alexpy.candidates import CANDIDATES, bigint_native, small_dispatch


class AlteredInt(int):
    def __mul__(self, other):
        return 31


class IteratingList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.events = 0

    def __iter__(self):
        self.events += 1
        yield 7
        yield 8


class BrokenMultiply:
    def __mul__(self, other):
        raise RuntimeError('expected multiply failure')


class TestCandidates(unittest.TestCase):
    def test_native_present_and_no_promotion(self):
        if _native is None:
            self.skipTest('native extension unavailable')
        self.assertTrue(hasattr(_native, 'sum_squares_bigints'))
        self.assertIsNot(_native.sum_squares_bigints, _native.sum_squares_ints)
        self.assertIs(_native.sum_squares_bigints([1, AlteredInt(2)]), NotImplemented)
        self.assertIs(_native.sum_squares_bigints(IteratingList([1])), NotImplemented)
        self.assertEqual(sum_squares([10**40]), 10**80)

    def test_boundary_and_promotion(self):
        examples = [[], (), [0], (1,), [3037000499], [3037000500],
                    [-(10**100)], (10**100, 1, -4), [10**9]*20,
                    (3037000499, 3037000499), [2**63, 2**64]]
        for values in examples:
            expected = sum(v*v for v in values)
            for name, candidate in CANDIDATES.items():
                with self.subTest(name=name, values=repr(values)[:45]):
                    self.assertEqual(candidate(values), expected)

    def test_small_randomized(self):
        r = random.Random(814)
        for n in (0, 1, 2, 8, 16, 128):
            for _ in range(5):
                values = [r.randrange(-10**12, 10**12) for _ in range(n)]
                if values and r.randrange(2):
                    values[0] = 10**(r.randrange(19, 100))
                for source in (values, tuple(values)):
                    expected = sum(v*v for v in source)
                    self.assertEqual(bigint_native(source), expected)
                    self.assertEqual(small_dispatch(source), expected)

    def test_subclass_preserves_iteration_and_multiplication(self):
        for candidate in (bigint_native, small_dispatch):
            with self.subTest(candidate=candidate.__name__):
                source = IteratingList([50, 100])
                self.assertEqual(candidate(source), 7*7 + 8*8)
                self.assertEqual(source.events, 1)
                self.assertEqual(candidate([AlteredInt(9)]), 31)
                self.assertEqual(candidate([10**30, AlteredInt(9)]), 10**60 + 31)

    def test_consumed_iterator_exactly_once(self):
        for candidate in (bigint_native, small_dispatch):
            r = iter([1, 2, 3])
            self.assertEqual(candidate(r), 14)
            self.assertEqual(candidate(r), 0)

    def test_errors_preserved(self):
        for candidate in (bigint_native, small_dispatch):
            with self.subTest(candidate=candidate.__name__):
                with self.assertRaisesRegex(RuntimeError, 'expected multiply failure'):
                    candidate([10**40, BrokenMultiply()])
                with self.assertRaises(TypeError):
                    candidate([object()])

    def test_floats_booleans_and_non_exact_types(self):
        for values in ([True, False], (2.5, 3.0), [1, 2.5], [AlteredInt(9)], [2**80, True]):
            expected = sum(v*v for v in values)
            self.assertEqual(bigint_native(values), expected)
            self.assertEqual(small_dispatch(values), expected)


if __name__ == '__main__':
    unittest.main()
