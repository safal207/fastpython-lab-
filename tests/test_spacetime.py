import math
import unittest
from dataclasses import replace

from alexpy.graph_search import State
from alexpy.spacetime import (Observation, check_path, derivatives,
                             measure_trace, multiscale_signature, planned_path,
                             targeted_path, shortest_graph_path, edge_deltas)


def point(tick, state, baseline=10.0, candidate=5.0):
    return Observation(tick, tick + 1, state, baseline, candidate, "current", True)


class TestSpacetime(unittest.TestCase):
    def test_walk_adjacent_and_monotonic(self):
        path = planned_path(hops=30)
        self.assertEqual(len(path), 31)
        self.assertEqual(len(set(path)) > 1, True)
        self.assertTrue(check_path([point(i, s) for i, s in enumerate(path)]))

    def test_reject_illegal_spatial_jump(self):
        a = State("list", 1, "small")
        b = State("tuple", 10000, "huge")
        with self.assertRaisesRegex(ValueError, "nonadjacent"):
            check_path([point(0, a), point(1, b)])

    def test_reject_backwards_time_and_ticks(self):
        s = State("list", 1, "small")
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            check_path([point(1, s), point(1, s)])
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            check_path([replace(point(0, s), timestamp_ns=5),
                        replace(point(1, s), timestamp_ns=4)])

    def test_exact_discrete_derivatives(self):
        s = State("list", 1, "small")
        ps = [point(i, s, baseline=math.exp(i*i), candidate=1.0) for i in range(5)]
        d = derivatives(ps)
        for actual, expected in zip(d["d1_per_tick"], [1, 3, 5, 7]):
            self.assertAlmostEqual(actual, expected, places=10)
        for actual in d["d2_per_tick2"]:
            self.assertAlmostEqual(actual, 2.0, places=10)

    def test_multiscale_incomplete_and_flat(self):
        s = State("list", 1, "small")
        ps = [point(i, s) for i in range(3)]
        a = multiscale_signature(ps)
        self.assertIsNone(a["roughness_by_hop_scale"]["4"])
        self.assertIsNone(a["log_log_scale_slopes"]["1->2"])
        with self.assertRaises(ValueError):
            multiscale_signature(ps, scales=(1, 1))
        with self.assertRaises(ValueError):
            multiscale_signature(ps, scales=(0, 1))

    def test_correctness_precedes_timing_and_small_experiment(self):
        out = measure_trace(planned_path(hops=6), repeats=2, loops=3)
        self.assertEqual(len(out["points"]), 7)
        self.assertTrue(all(p["correct"] for p in out["points"]))
        self.assertEqual(len(out["derivatives"]["d2_per_tick2"]), 5)
        self.assertIn("fractal dimension", out["multiscale"]["interpretation"])

    def test_targeted_path_covers_routing_regimes(self):
        from alexpy.adaptive import route
        from alexpy.graph_search import make_values
        path = targeted_path()
        self.assertTrue(check_path([point(i, s) for i, s in enumerate(path)]))
        routes = {route(make_values(s)) for s in path}
        self.assertTrue({"zero", "singleton-int", "bigint-native",
                         "python-direct", "current"} <= routes)
        self.assertEqual(shortest_graph_path(path[0], path[0]), [path[0]])
        self.assertTrue(any(row["axis"] == "length" for row in
                            edge_deltas([point(i, s) for i,s in enumerate(path)])))

    def test_real_time_derivative_uses_irregular_interval(self):
        s = State("list", 1, "small")
        ps = [replace(point(0, s, 1, 1), timestamp_ns=1_000_000_000),
              replace(point(1, s, math.e, 1), timestamp_ns=2_000_000_000),
              replace(point(2, s, math.e**3, 1), timestamp_ns=4_000_000_000)]
        d = derivatives(ps)
        self.assertAlmostEqual(d["d1_per_second"][0], 1.0)
        self.assertAlmostEqual(d["d1_per_second"][1], 1.0)
        self.assertAlmostEqual(d["d2_per_second2"][0], 0.0)

    def test_empty_and_invalid_paths(self):
        with self.assertRaises(ValueError):
            measure_trace([])
        with self.assertRaises(ValueError):
            measure_trace([State("list", 1, "small"), State("tuple", 10000, "huge")])
        with self.assertRaises(ValueError):
            measure_trace(repeats=0)


if __name__ == "__main__":
    unittest.main()
