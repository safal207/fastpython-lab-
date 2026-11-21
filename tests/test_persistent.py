"""M9 graph-based state tests: sparse updates, ownership, and invariants."""
from __future__ import annotations

import random
import unittest
from array import array

from alexpy.persistent import OwnedQBuffer, Transition


class WeirdList(list):
    def __iter__(self):
        raise AssertionError("must not consume custom iterator")


class TestOwnedQBuffer(unittest.TestCase):
    def test_empty_and_no_op_tick(self):
        buf = OwnedQBuffer([], trace=True)
        self.assertEqual((len(buf), buf.aggregates(), buf.variance_numerator()), (0, (0, 0), 0))
        event = buf.apply_patch({})
        self.assertEqual(event, Transition(1, 0, (), 0, 0))
        self.assertEqual(buf.history, (event,))
        self.assertTrue(buf.verify())

    def test_snapshot_alias_isolation(self):
        original = [2, 3, 4]
        buf = OwnedQBuffer(original)
        original[0] = 120
        self.assertEqual(buf.at(0), 2)
        buf.apply_patch({1: -7})
        self.assertEqual(original[1], 3)
        copy = buf.to_list()
        copy[0] = 88
        self.assertEqual(buf.to_list(), [2, -7, 4])
        self.assertTrue(buf.verify())

    def test_boundary_exact_q_and_big_aggregate(self):
        numbers = [-(2**63), 2**63 - 1, -1, 0]
        buf = OwnedQBuffer(tuple(numbers))
        self.assertEqual(buf.aggregates(), (sum(numbers), sum(x*x for x in numbers)))
        buf.apply_patch({0: 2**63 - 1})
        numbers[0] = 2**63 - 1
        self.assertEqual(buf.aggregates(), (sum(numbers), sum(x*x for x in numbers)))
        self.assertTrue(buf.verify())

    def test_multi_update_sparse_dirty_and_version(self):
        buf = OwnedQBuffer([3, 3, 3, 3], trace=True)
        event = buf.apply_patch({3: 4, 1: -5, 2: 3})
        self.assertEqual((event.tick, event.version, event.changed_indices), (1, 1, (1, 3)))
        self.assertEqual((event.delta_sum, event.delta_sum_squares), (-7, 23))
        self.assertEqual(buf.aggregates(), (5, 59))
        self.assertEqual(buf.variance_numerator(), 4*59 - 5**2)
        self.assertTrue(buf.verify())
        e2 = buf.apply_patch({3: 4})
        self.assertEqual((e2.tick, e2.version, e2.changed_indices), (2, 1, ()))
        self.assertEqual(len(buf.history), 2)

    def test_rejected_patch_preserves_all_state(self):
        buf = OwnedQBuffer([1, 2, 3], trace=True)
        before = (buf.to_list(), buf.aggregates(), buf.tick, buf.version, buf.history)
        invalid = [({0: 5, 2: 2**64}, OverflowError),
                   ({0: 5, 99: 5}, IndexError),
                   ({True: 9}, IndexError),
                   ({0: False}, TypeError),
                   ({0: 2.5}, TypeError),
                   ({0: 4, -1: 7}, IndexError),
                   ({'0': 7}, IndexError)]
        for patch, error in invalid:
            with self.subTest(patch=patch):
                with self.assertRaises(error):
                    buf.apply_patch(patch)
                self.assertEqual((buf.to_list(), buf.aggregates(), buf.tick, buf.version, buf.history), before)
        self.assertTrue(buf.verify())

    def test_unsupported_source_is_not_misinterpreted(self):
        for source in (WeirdList([1]), range(2), array('q', [1]), [True], [1.0], [2**64]):
            with self.subTest(type=type(source).__name__):
                with self.assertRaises((TypeError, OverflowError)):
                    OwnedQBuffer(source)
        with self.assertRaises(TypeError):
            OwnedQBuffer([1], trace=1)
        with self.assertRaises(TypeError):
            OwnedQBuffer([1]).apply_patch([(0, 3)])

    def test_no_mutable_view_exposed_and_index_gate(self):
        buf = OwnedQBuffer([1, 2])
        self.assertFalse(hasattr(buf, 'view'))
        self.assertFalse(hasattr(buf, 'append'))
        self.assertEqual(buf.at(1), 2)
        for index in (-1, 2, True, 1.1):
            with self.assertRaises(IndexError):
                buf.at(index)
        self.assertEqual(buf.data_bytes, 16)

    def test_dependency_graph(self):
        deps = OwnedQBuffer.dependencies()
        self.assertEqual(deps['data[index]'], ('sum', 'sum_squares'))
        self.assertEqual(deps['sum_squares'], ('variance_numerator',))

    def test_randomized_transition_mirror(self):
        rng = random.Random(929)
        for size in (1, 8, 127, 4096):
            initial = [rng.randint(-1000, 1000) for _ in range(size)]
            control = initial.copy()
            buf = OwnedQBuffer(initial)
            for _ in range(20):
                patch = {rng.randrange(size): rng.randint(-1000, 1000) for _ in range(min(8, size))}
                event = buf.apply_patch(patch)
                control_indices = tuple(sorted(i for i, v in patch.items() if control[i] != v))
                for i, value in patch.items():
                    control[i] = value
                self.assertEqual(event.changed_indices, control_indices)
                self.assertEqual(buf.aggregates(), (sum(control), sum(x*x for x in control)))
                self.assertTrue(buf.verify())

    def test_mirror_detects_private_corruption(self):
        buf = OwnedQBuffer([2, 3, 4])
        self.assertTrue(buf.verify())
        buf._data[0] = 77  # illicit access to private storage
        self.assertFalse(buf.verify())

    def test_repeat_queries_are_pure(self):
        buf = OwnedQBuffer([1, 2, 3])
        expected = (6, 14)
        for _ in range(25):
            self.assertEqual(buf.aggregates(), expected)
        self.assertEqual((buf.tick, buf.version, buf.dirty), (0, 0, ()))


if __name__ == '__main__':
    unittest.main()
