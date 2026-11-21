import unittest
import random
from array import array

from alexpy import backend_name, sum_squares


class CustomInt(int):
    def __mul__(self, other):
        return 5


class CustomList(list):
    def __iter__(self):
        return iter([3])


class TestSumSquares(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(sum_squares([]), 0)
        self.assertEqual(sum_squares(()), 0)

    def test_positive_and_negative(self):
        for data in ([1, -2, 3], (1, -2, 3), [-1] * 1000):
            with self.subTest(data_type=type(data).__name__):
                self.assertEqual(sum_squares(data), sum(x * x for x in data))

    def test_64bit_boundary_and_unbounded_ints(self):
        examples = [
            [3037000499], [-3037000499], [3037000500],
            [10**50], [2**63], [2**64], [10**30, 2, -10**30],
            [10**9] * 10, [10**9, 10**9, 1],
        ]
        for values in examples:
            with self.subTest(values=values[:2]):
                self.assertEqual(sum_squares(values), sum(x * x for x in values))

    def test_other_numeric_types(self):
        for values in ([True, False], [2.5, 3.0], [CustomInt(2)], array('i', [1, 2])):
            with self.subTest(values=values):
                self.assertEqual(sum_squares(values), sum(x * x for x in values))

    def test_python_semantics_and_iterators(self):
        cases = [CustomList([100, 200]), (i for i in range(10)), iter([1, 2, 3])]
        for values in cases:
            with self.subTest(type=type(values).__name__):
                self.assertEqual(sum_squares(values), 9 if isinstance(values, CustomList)
                                 else (285 if isinstance(values, type((i for i in []))) else 14))
        self.assertEqual(sum_squares([CustomInt(2)]), 5)

    def test_deterministic_randomized_values(self):
        rng = random.Random(42017)
        for size in (0, 1, 2, 10, 100, 1000):
            values = [rng.randrange(-10_000_000, 10_000_000) for _ in range(size)]
            for data in (values, tuple(values)):
                with self.subTest(size=size, data_type=type(data).__name__):
                    self.assertEqual(sum_squares(data), sum(x * x for x in data))

    def test_native_fast_path_and_fallback(self):
        try:
            from alexpy import _native
        except ImportError:
            self.skipTest("native extension not built")
        if _native is None:
            self.skipTest("native extension not built")
        self.assertEqual(_native.sum_squares_ints([1, 2, 3]), 14)
        self.assertIs(_native.sum_squares_ints([10**30]), NotImplemented)
        self.assertIs(_native.sum_squares_ints([CustomInt(3)]), NotImplemented)
        self.assertIs(_native.sum_squares_ints(CustomList([2])), NotImplemented)

    def test_errors_propagate(self):
        with self.assertRaises(TypeError):
            sum_squares([object()])

    def test_backend_reported(self):
        self.assertIn(backend_name(), {"native-c", "python-fallback"})


if __name__ == "__main__":
    unittest.main()
