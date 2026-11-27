"""Differential and adversarial checks for the opt-in compiled DAG."""
import random
import unittest

from alexpy.compiled_graph import CompiledDependencyGraph
from alexpy.dependency_graph import DependencyGraph


def populate(cls, values, *, width=4, trace=False, overlapping=False):
    graph = cls(values, bucket_size=width, trace=trace)
    segments = []
    for start in range(0, len(values), width):
        stop = min(len(values), start + width)
        s, q, v = f's{start}', f'q{start}', f'v{start}'
        graph.add_window(s, start, stop, 'sum')
        graph.add_window(q, start, stop, 'squares')
        graph.add_variance(v, s, q)
        segments.append((s, q, v))
    if overlapping and values:
        graph.add_window('all_s', 0, len(values), 'sum')
        graph.add_window('all_q', 0, len(values), 'squares')
        graph.add_variance('all_v', 'all_s', 'all_q')
    if segments:
        graph.add_sum('aggregate', tuple(x[0] for x in segments))
        graph.add_sum('variance_total', tuple(x[2] for x in segments))
        graph.add_sum('composite', ('aggregate', 'variance_total'))
    if isinstance(graph, CompiledDependencyGraph):
        graph.compile()
    return graph


class CompiledGraphTest(unittest.TestCase):
    def test_uncompiled_refuses_mutation_then_compiles_idempotently(self):
        g = CompiledDependencyGraph([1, 2, 3])
        g.add_window('s', 0, 3, 'sum')
        with self.assertRaisesRegex(RuntimeError, 'compile'):
            g.apply_patch({1: 4})
        self.assertEqual(g.to_list(), [1, 2, 3])
        g.compile()
        g.compile()
        self.assertEqual(g.apply_patch({1: 4}).changed_indices, (1,))
        self.assertTrue(g.verify())
        with self.assertRaisesRegex(RuntimeError, 'compiled'):
            g.add_window('another', 0, 3, 'sum')
        with self.assertRaisesRegex(RuntimeError, 'compiled'):
            g.add_sum('new', ('s',))
        self.assertTrue(g.verify())

    def test_metamorphic_random_patches_and_all_cached_nodes(self):
        rng = random.Random(919)
        for n in (0, 1, 7, 32, 137):
            source = [rng.randint(-2**28, 2**28) for _ in range(n)]
            for overlapping in (False, True):
                a = populate(DependencyGraph, source, width=8, trace=True,
                             overlapping=overlapping)
                b = populate(CompiledDependencyGraph, source, width=8, trace=True,
                             overlapping=overlapping)
                self.assertEqual(a.snapshot(), b.snapshot())
                for t in range(32):
                    patch = {rng.randrange(n): rng.choice([rng.randint(-1000, 1000),
                                                           -(2**63), 2**63-1])
                             for _ in range(rng.randint(0, min(n, 4)))} if n else {}
                    e1 = a.apply_patch(patch)
                    e2 = b.apply_patch(patch)
                    with self.subTest(n=n, overlap=overlapping, tick=t):
                        self.assertEqual(a.to_list(), b.to_list())
                        self.assertEqual(a.snapshot(), b.snapshot())
                        self.assertEqual(e1.changed_indices, e2.changed_indices)
                        self.assertEqual(set(e1.updated_nodes), set(e2.updated_nodes))
                        self.assertEqual(set(e1.touched_windows), set(e2.touched_windows))
                        self.assertEqual(e1.examined_windows, e2.examined_windows)
                        self.assertTrue(b.verify())
                self.assertEqual(len(b.history), 32)
                self.assertEqual(b.version, a.version)
                self.assertEqual(b.tick, a.tick)

    def test_non_overlapping_unchanged_branches_and_sign_flip(self):
        g = populate(CompiledDependencyGraph, [1,2,3,4,5,6,7,8], width=4)
        e = g.apply_patch({6: 100})
        self.assertTrue({'s4', 'q4', 'v4'} <= set(e.updated_nodes))
        self.assertTrue({'s0', 'q0', 'v0'}.isdisjoint(e.updated_nodes))
        previous = g.value('q4')
        e = g.apply_patch({6: -100})
        self.assertEqual(g.value('q4'), previous)
        self.assertNotIn('q4', e.updated_nodes)
        self.assertTrue(g.verify())

    def test_atomic_rejection(self):
        g = populate(CompiledDependencyGraph, [1,2,3,4], width=2, trace=True)
        before = (g.snapshot(), g.to_list(), g.version, g.tick, g.history)
        failures = [({0: 8, 4: 3}, IndexError),
                    ({0: 8, 1: True}, TypeError),
                    ({0: 8, 1: 2**64}, OverflowError),
                    ({0: 8, 1.0: 5}, IndexError),
                    ({True: 1}, IndexError),
                    ([(0, 8)], TypeError)]
        for patch, error in failures:
            with self.subTest(patch=patch):
                with self.assertRaises(error):
                    g.apply_patch(patch)
                self.assertEqual(before, (g.snapshot(), g.to_list(),
                                          g.version, g.tick, g.history))
        self.assertTrue(g.verify())

    def test_subclass_and_bounds_rejected(self):
        class TrickyInt(int):
            pass
        g = CompiledDependencyGraph([1, 2], bucket_size=2)
        for factory in (lambda: g.add_window('s', TrickyInt(0), 2, 'sum'),
                        lambda: g.add_window('s', 0, 3, 'sum'),
                        lambda: g.add_window('s', 0, 2, 'bogus')):
            with self.assertRaises(ValueError):
                factory()
        g.add_window('s', 0, 2, 'sum')
        g.compile()
        with self.assertRaises(IndexError):
            g.apply_patch({TrickyInt(1): 3})
        self.assertTrue(g.verify())

    def test_benchmark_workflow_matches_all_routes(self):
        from alexpy.compiled_bench import workflow, ROUTES
        from alexpy.dependency_bench import make_case
        for n in (8, 32, 128):
            initial, patches = make_case(n, 8, 2, 3)
            results = {route: workflow(route, initial, patches, 2, 8)
                       for route in ROUTES}
            self.assertEqual(len(set(results.values())), 1, results)


if __name__ == '__main__':
    unittest.main()
