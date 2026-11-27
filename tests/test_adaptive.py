import random
import unittest

from alexpy import sum_squares
from alexpy.adaptive import adaptive_sum_squares, route


class ListWithCustomIter(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        yield 7
        yield 8


class AlteredInt(int):
    def __mul__(self, other):
        return 31


class TestAdaptive(unittest.TestCase):
    def test_routes_are_explicit(self):
        self.assertEqual(route([]), "zero")
        self.assertEqual(route((4,)), "singleton-int")
        self.assertEqual(route(ListWithCustomIter([4])), "python-direct")
        self.assertEqual(route([4, 8]), "current")
        self.assertEqual(route([AlteredInt(3)]), "current")
        self.assertEqual(route((i for i in range(3))), "python-direct")
        self.assertIn(route([10**60]), ("singleton-int",))
        self.assertIn(route([10**60, 4]), ("bigint-native", "current"))

    def test_native_semantics_across_random_cases(self):
        rng = random.Random(12027)
        cases = [[], (), [0], [True], [1.5], [AlteredInt(3)],
                 [10**50, 2, 3], [10**9]*16, [3, 10**50],
                 (3037000500,), (3037000499, 3037000499)]
        for n in (1, 2, 8, 16, 1024):
            for _ in range(5):
                cases.append([rng.randrange(-10**12, 10**12) for _ in range(n)])
        for values in cases:
            for seq in (values, tuple(values)):
                with self.subTest(length=len(seq), prefix=repr(seq)[:30]):
                    self.assertEqual(adaptive_sum_squares(seq),
                                     sum(x*x for x in seq))

    def test_subclass_iterated_once(self):
        v = ListWithCustomIter([50, 60])
        self.assertEqual(adaptive_sum_squares(v), 7*7 + 8*8)
        self.assertEqual(v.iter_calls, 1)

    def test_iterator_consumed_once(self):
        data = iter([1, 2, 3])
        self.assertEqual(adaptive_sum_squares(data), 14)
        self.assertEqual(adaptive_sum_squares(data), 0)

    def test_custom_numeric_and_errors(self):
        self.assertEqual(adaptive_sum_squares([AlteredInt(3)]), 31)
        self.assertEqual(adaptive_sum_squares([10**50, AlteredInt(3)]), 10**100 + 31)
        with self.assertRaises(TypeError):
            adaptive_sum_squares([object()])
        self.assertEqual(sum_squares([10**40]), 10**80)  # public API unmodified


if __name__ == "__main__":
    unittest.main()
