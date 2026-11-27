"""Memory-layout candidate semantics and guardset."""
from array import array
import random
import unittest

from alexpy import sum_squares
from alexpy import _buffer_native
from alexpy.memory_candidates import (
    CANDIDATES, eligible, linear_buffer, interleaved_buffer
)


class WeirdArray(array):
    def __iter__(self):
        yield 8
        yield -9


class TestBufferCandidates(unittest.TestCase):
    def test_admission_only_exact_signed64(self):
        self.assertTrue(eligible(array("q", [1])))
        self.assertTrue(eligible(memoryview(array("q", [1]))))
        self.assertFalse(eligible(array("d", [1.5])))
        self.assertFalse(eligible(WeirdArray("q", [1, 2])))
        self.assertFalse(eligible([1, 2]))
        self.assertFalse(eligible(memoryview(array("q", [1, 2, 3]))[::2]))

    def test_direct_entrypoints_return_not_implemented_for_non_q(self):
        for fn in (_buffer_native.sum_squares_linear,
                   _buffer_native.sum_squares_interleaved):
            for values in ([1, 2], array("d", [1.5]), memoryview(b"abcd"),
                           memoryview(array("q", [1, 2, 3]))[::2]):
                with self.subTest(fn=fn.__name__, type=type(values).__name__):
                    self.assertIs(fn(values), NotImplemented)

    def test_exhaustive_small_samples_and_overflow(self):
        samples = [-2**63, -3037000500, -3037000499, -1, 0, 1,
                   3037000499, 3037000500, 2**63 - 1]
        r = random.Random(726)
        for n in (0, 1, 2, 3, 8, 64, 128):
            for _ in range(8):
                values = array("q", (r.choice(samples) for _ in range(n)))
                for src in (values, memoryview(values)):
                    expected = sum(x*x for x in src)
                    for name, fn in CANDIDATES.items():
                        with self.subTest(n=n, source=type(src).__name__, fn=name):
                            self.assertEqual(fn(src), expected)

    def test_noncontiguous_fallback_and_identity(self):
        source = memoryview(array("q", [10, 3, -2, 20, 4]))[::2]
        for fn in CANDIDATES.values():
            self.assertEqual(fn(source), 10**2+(-2)**2+4**2)

    def test_class_override_iteration_not_bypassed(self):
        source = WeirdArray("q", [99, 100])
        for fn in CANDIDATES.values():
            self.assertEqual(fn(source), 8**2+(-9)**2)

    def test_different_numerical_formats_fall_back(self):
        for src in (array("d", [1.25, 2.75]), array("i", [3, 4]),
                    memoryview(b"abcd")):
            expected = sum(x*x for x in src)
            for fn in CANDIDATES.values():
                self.assertEqual(fn(src), expected)

    def test_generator_remains_one_shot(self):
        for fn in CANDIDATES.values():
            src = (x for x in (2, 3, 4))
            self.assertEqual(fn(src), 29)
            self.assertEqual(fn(src), 0)

    def test_public_api_unchanged_and_mutability(self):
        src = array("q", [2, 3, 5])
        self.assertEqual(sum_squares(src), 38)
        self.assertEqual(linear_buffer(src), 38)
        src[1] = 10
        self.assertEqual(interleaved_buffer(src), 129)
        self.assertEqual(sum_squares(src), 129)


if __name__ == "__main__":
    unittest.main()
