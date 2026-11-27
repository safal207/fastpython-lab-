"""Differential, fail-closed and fallback coverage for the partition compiler."""
import random
import unittest

from alexpy.partition_graph import PartitionDependencyGraph
from alexpy.compiled_graph import CompiledDependencyGraph
from tests.test_compiled_graph import populate


def partition_graph(cls, data, width=4, trace=False):
    g = cls(data, bucket_size=width, trace=trace)
    sums = []
    for start in range(0, len(data), width):
        stop = min(len(data), start+width)
        s, q, v = f"s{start}", f"q{start}", f"v{start}"
        g.add_window(s, start, stop, "sum")
        g.add_window(q, start, stop, "squares")
        g.add_variance(v, s, q)
        sums.append(s)
    if sums:
        g.add_sum("global_sum", tuple(sums))
    g.compile()
    return g


class PartitionGraphTests(unittest.TestCase):
    def test_specialization_and_fallback(self):
        for n in (0, 1, 4, 5, 16, 41):
            fast = partition_graph(PartitionDependencyGraph, list(range(n)), width=4)
            self.assertEqual(fast.specialized, n != 0)
            plain = populate(PartitionDependencyGraph, list(range(n)), width=4)
            self.assertFalse(plain.specialized)

    def test_randomized_equivalence_specialized(self):
        randomizer = random.Random(74423)
        for n in (0, 1, 7, 32, 137):
            source = [randomizer.randrange(-20000,20000) for _ in range(n)]
            a = partition_graph(CompiledDependencyGraph, source, width=8, trace=True)
            b = partition_graph(PartitionDependencyGraph, source, width=8, trace=True)
            for _ in range(50):
                changes = {randomizer.randrange(n): randomizer.choice([
                    0, 1, -(1<<63), (1<<63)-1, randomizer.randrange(-5000,5000)])
                    for _ in range(randomizer.randrange(0, 5))} if n else {}
                ea, eb = a.apply_patch(changes), b.apply_patch(changes)
                self.assertEqual(a.snapshot(), b.snapshot())
                self.assertEqual(ea, eb)
                self.assertTrue(b.verify())

    def test_fallback_for_overlapping_and_extended_dag(self):
        a = populate(CompiledDependencyGraph, list(range(20)), width=4, trace=True,
                     overlapping=True)
        b = populate(PartitionDependencyGraph, list(range(20)), width=4, trace=True,
                     overlapping=True)
        self.assertFalse(b.specialized)
        for changes in ({1: 20}, {4:-10,15:15}, {3: -3}, {}):
            self.assertEqual(a.apply_patch(changes), b.apply_patch(changes))
            self.assertEqual(a.snapshot(), b.snapshot())
        self.assertTrue(b.verify())

    def test_bad_patch_no_mutation(self):
        b = partition_graph(PartitionDependencyGraph, [0,1,2,3,4,5], width=3, trace=True)
        before = (b.snapshot(),b.to_list(),b.tick,b.version,b.history)
        for changes, err in (({2:3,6:4}, IndexError), ({0:True}, TypeError),
                             ({0:2**63}, OverflowError), ([(0,1)], TypeError)):
            with self.assertRaises(err): b.apply_patch(changes)
            self.assertEqual(before, (b.snapshot(),b.to_list(),b.tick,b.version,b.history))

    def test_cancellation_and_sign_flip(self):
        a = partition_graph(CompiledDependencyGraph, [1,2,3,4,5,6], width=3)
        b = partition_graph(PartitionDependencyGraph, [1,2,3,4,5,6], width=3)
        for patch in ({0:-1}, {0:2,1:1}, {0:1, 3:6, 4:5}, {1:2}, {}):
            self.assertEqual(a.apply_patch(patch),b.apply_patch(patch))
            self.assertEqual(a.snapshot(),b.snapshot())


if __name__ == '__main__': unittest.main()
