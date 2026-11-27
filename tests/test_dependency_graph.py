"""Bounded graph correctness, atomic patch and sparse propagation tests."""
from __future__ import annotations

import random
import unittest

from alexpy.dependency_graph import DependencyGraph


class TestDependencyGraph(unittest.TestCase):
    def make_graph(self):
        graph = DependencyGraph([1, 2, 3, 4, 5, 6, 7, 8], bucket_size=2,
                                trace=True)
        for name, start, stop in (("left", 0, 4), ("right", 4, 8)):
            graph.add_window(name + "_sum", start, stop, "sum")
            graph.add_window(name + "_sq", start, stop, "squares")
            graph.add_variance(name + "_var", name + "_sum", name + "_sq")
        graph.add_sum("total", ("left_sum", "right_sum"))
        graph.add_sum("combined_var", ("left_var", "right_var"))
        return graph

    def test_registration_and_snapshot(self):
        graph = self.make_graph()
        self.assertEqual(graph.snapshot()["left_sum"], 10)
        self.assertEqual(graph.value("right_sq"), 5**2+6**2+7**2+8**2)
        self.assertEqual(graph.value("total"), 36)
        self.assertEqual(len(graph.nodes()), 8)
        self.assertEqual(len(graph.edges()), 8)
        self.assertTrue(graph.verify())

    def test_only_affected_side_and_ancestors_update(self):
        graph = self.make_graph()
        before = graph.snapshot()
        event = graph.apply_patch({6: 12})
        self.assertEqual(event.changed_indices, (6,))
        self.assertEqual(set(event.touched_windows), {"right_sum", "right_sq"})
        self.assertEqual(set(event.updated_nodes), {
            "right_sum", "right_sq", "right_var", "total", "combined_var"})
        self.assertTrue(all(graph.value(name) == before[name] for name in
                            ("left_sum", "left_sq", "left_var")))
        self.assertLessEqual(event.examined_windows, 2)
        self.assertTrue(graph.verify())

    def test_value_aware_propagation_sign_flip(self):
        graph = self.make_graph()
        before = graph.value("left_sq")
        event = graph.apply_patch({2: -3})
        self.assertEqual(graph.value("left_sq"), before)
        self.assertNotIn("left_sq", event.updated_nodes)
        self.assertIn("left_sum", event.updated_nodes)
        self.assertIn("left_var", event.updated_nodes)
        self.assertTrue(graph.verify())

    def test_equal_sum_equal_squares_means_no_dependency_updates(self):
        graph = self.make_graph()
        event = graph.apply_patch({0: 2, 1: 1})
        self.assertEqual(event.updated_nodes, ())
        self.assertEqual(set(event.touched_windows), {"left_sum", "left_sq"})
        self.assertEqual(graph.version, 1)
        self.assertEqual(graph.to_list()[:2], [2, 1])
        self.assertTrue(graph.verify())

    def test_noop_patch_advances_tick_but_not_version(self):
        graph = self.make_graph()
        event = graph.apply_patch({0: 1})
        self.assertEqual(event.updated_nodes, ())
        self.assertEqual(event.changed_indices, ())
        self.assertEqual(graph.tick, 1)
        self.assertEqual(graph.version, 0)
        self.assertTrue(graph.verify())
        self.assertEqual(graph.history, (event,))

    def test_invalid_patch_is_atomic(self):
        graph = self.make_graph()
        before = graph.snapshot()
        for patch, error in (({1: 3, 4: True}, TypeError),
                             ({0: 4, 30: 12}, IndexError),
                             ({0: 4, 1: 2**85}, OverflowError),
                             ({False: 1}, IndexError),
                             ([], TypeError)):
            with self.subTest(patch=patch):
                with self.assertRaises(error):
                    graph.apply_patch(patch)
                self.assertEqual(graph.snapshot(), before)
                self.assertEqual(graph.tick, 0)
                self.assertEqual(graph.history, ())
                self.assertTrue(graph.verify())

    def test_validation_and_cannot_make_cycle(self):
        graph = DependencyGraph([1, 2, 3])
        with self.assertRaises(ValueError):
            graph.add_window("x", 1, 4, "sum")
        with self.assertRaises(ValueError):
            graph.add_window("x", 0, 2, "min")
        graph.add_window("x", 0, 3, "sum")
        with self.assertRaises(ValueError):
            graph.add_window("x", 0, 3, "sum")
        with self.assertRaises(ValueError):
            graph.add_sum("foo", ("missing",))
        with self.assertRaises(ValueError):
            graph.add_sum("foo", ("x", "x"))
        with self.assertRaises(ValueError):
            graph.add_variance("var", "x", "x")
        with self.assertRaises(TypeError):
            DependencyGraph([True])
        with self.assertRaises(ValueError):
            DependencyGraph([1], bucket_size=0)

    def test_empty_window_and_copy_isolation(self):
        values = [4, 5]
        graph = DependencyGraph(values)
        graph.add_window("empty_sum", 1, 1, "sum")
        graph.add_window("empty_sq", 1, 1, "squares")
        graph.add_variance("empty_var", "empty_sum", "empty_sq")
        values[0] = 10000
        exported = graph.to_list()
        exported[0] = 10000
        self.assertEqual(graph.to_list(), [4, 5])
        graph.apply_patch({0: 6})
        self.assertEqual(graph.value("empty_var"), 0)
        self.assertEqual(graph.last_transition.examined_windows, 0)
        self.assertTrue(graph.verify())

    def test_graphviz_exposes_last_frontier(self):
        graph = self.make_graph()
        graph.apply_patch({6: 12})
        dot = graph.to_dot()
        self.assertIn('digraph AlexPyDependencies', dot)
        self.assertIn('right_sum', dot)
        self.assertIn('left_sum', dot)
        self.assertEqual(dot.count(' -> '), len(graph.edges()))
        self.assertTrue(dot.endswith('}\n'))

    def test_overlapping_intervals_and_multiscale_parent(self):
        values = list(range(1, 17))
        graph = DependencyGraph(values, bucket_size=4)
        graph.add_window("first", 0, 8, "sum")
        graph.add_window("second", 4, 12, "sum")
        graph.add_window("third", 12, 16, "sum")
        graph.add_sum("shared", ("first", "second"))
        before = graph.value("third")
        event = graph.apply_patch({5: 99})
        self.assertEqual(set(event.updated_nodes), {"first", "second", "shared"})
        self.assertEqual(graph.value("third"), before)
        self.assertTrue(graph.verify())
        self.assertEqual(graph.data_bytes, 16*8)

    def test_q_bounds_preserve_bigint_aggregates(self):
        maxval = (1 << 63) - 1
        graph = DependencyGraph([maxval, -maxval, 0])
        graph.add_window("s", 0, 3, "sum")
        graph.add_window("q", 0, 3, "squares")
        graph.add_variance("v", "s", "q")
        graph.apply_patch({2: -(1 << 63)})
        self.assertEqual(graph.value("s"), -(1 << 63))
        self.assertTrue(graph.verify())

    def test_multiple_parent_levels(self):
        graph = self.make_graph()
        graph.add_sum("doubled", ("total", "combined_var"))
        event = graph.apply_patch({5: -4})
        self.assertIn("doubled", event.updated_nodes)
        self.assertEqual(graph.value("doubled"),
                         graph.value("total")+graph.value("combined_var"))
        self.assertTrue(graph.verify())

    def test_randomized_delta_graph_vs_full_rebuild(self):
        rng = random.Random(20261122)
        for n in (1, 17, 64, 256):
            values = [rng.randrange(-1000, 1000) for _ in range(n)]
            graph = DependencyGraph(values, bucket_size=16)
            pairs = []
            for start in range(0, n, 16):
                stop = min(n, start + 16)
                s, q, var = f"s{start}", f"q{start}", f"v{start}"
                graph.add_window(s, start, stop, "sum")
                graph.add_window(q, start, stop, "squares")
                graph.add_variance(var, s, q)
                pairs.append((s, q, var))
            graph.add_sum("all", tuple(s for s, _, _ in pairs))
            self.assertTrue(graph.verify())
            for _ in range(40):
                indices = rng.sample(range(n), min(n, rng.randrange(0, 5)))
                patch = {i: rng.randrange(-10000, 10000) for i in indices}
                graph.apply_patch(patch)
                for i, v in patch.items():
                    values[i] = v
                self.assertEqual(graph.to_list(), values)
                self.assertEqual(graph.value("all"), sum(values))
                self.assertTrue(graph.verify())


if __name__ == "__main__":
    unittest.main()
