"""Graph theory and safety invariants for heterogeneous wave propagation."""
import json
import math
import random
import unittest

from alexpy.ant_memory import Workload
from alexpy.hetero_wave import HeterogeneousWaveMemory
from alexpy.wave_memory import WaveMemoryGraph


class TestHeterogeneousWave(unittest.TestCase):
    def setUp(self):
        self.left = Workload(128, 64, 1, 2, 2)
        self.middle = Workload(256, 64, 1, 2, 2)
        self.right = Workload(512, 64, 1, 2, 2)

    def test_uniform_model_matches_homogeneous_shortcut(self):
        hetero = HeterogeneousWaveMemory(axis_weights=(.18,) * 5, size_slope=0,
                                          max_hops=3, min_gain=1)
        uniform = WaveMemoryGraph(attenuation=.18, max_hops=3, min_gain=1)
        rng = random.Random(918)
        for _ in range(12):
            state = Workload(2**rng.randint(5, 10), 2**rng.randint(3, 7),
                             2**rng.randint(0, 2), 2**rng.randint(0, 3),
                             2**rng.randint(0, 3))
            for model in (uniform, hetero):
                model.observe(state, "partition", elapsed_ns=rng.choice((25, 25)),
                              reference_ns=100, verified=True)
        for _ in range(15):
            state = Workload(2**rng.randint(5, 10), 64, 1, 2, 2)
            a = hetero._light_arrival(state.cell(), "partition")
            b = uniform._light_arrival(state.cell(), "partition")
            if math.isinf(a):
                self.assertTrue(math.isinf(b))
            else:
                self.assertAlmostEqual(a, b, places=9)

    def test_directed_transition_changes_arrival_and_path(self):
        model = HeterogeneousWaveMemory(axis_weights=(.1,) * 5,
                                        size_slope=0, max_hops=3, min_gain=1)
        model.observe(self.left, "partition", elapsed_ns=10,
                      reference_ns=100, verified=True)
        start = model._light_arrival(self.middle.cell(), "partition")
        self.assertEqual(model.arrival_path(self.middle, "partition")["cells"],
                         [list(self.left.cell()), list(self.middle.cell())])
        model.observe_transition(self.left, self.middle, "partition",
                                 elapsed_ns=2000, reference_ns=10, verified=True)
        after = model._light_arrival(self.middle.cell(), "partition")
        self.assertGreater(after, start)
        self.assertGreater(model.edge_cost(self.left.cell(), self.middle.cell(), "partition"),
                           model.edge_cost(self.middle.cell(), self.left.cell(), "partition"))

    def test_blocked_transition_uses_other_route_in_grid(self):
        model = HeterogeneousWaveMemory(axis_weights=(.1,) * 5,
                                        size_slope=0, max_hops=3)
        model.observe(self.left, "partition", elapsed_ns=10,
                      reference_ns=100, verified=True)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            model.observe_transition(self.left, self.middle, "partition",
                                     elapsed_ns=10, reference_ns=100, verified=False)
        with self.assertRaises(ValueError):
            model.observe_transition(self.left, self.middle, "partition",
                                     elapsed_ns=10, reference_ns=100, verified=True)
        self.assertEqual(model.edge_cost(self.left.cell(), self.middle.cell(), "partition"),
                         math.inf)
        # A detour within the radius can still connect the cells.
        self.assertTrue(math.isfinite(model._light_arrival(self.middle.cell(), "partition")))

    def test_bellman_ford_independent_witness(self):
        model = HeterogeneousWaveMemory(max_hops=2, size_slope=.4,
                                        axis_weights=(.2, .3, .1, .5, .4))
        target = self.middle.cell()
        coordinates = {target}
        for _ in range(2):
            coordinates |= {point[:axis] + (point[axis] + d,) + point[axis+1:]
                            for point in tuple(coordinates) for axis in range(5)
                            for d in (-1, 1) if point[axis] + d >= 0
                            and model._distance(point[:axis] + (point[axis] + d,) + point[axis+1:], target) <= 2}
        coords = sorted(coordinates)
        for _ in range(12):
            a = random.choice(coords)
            neighbours = [b for b in coords if model._distance(a, b) == 1]
            b = random.choice(neighbours)
            model.observe_transition(Workload(*(2**x for x in a)),
                                     Workload(*(2**x for x in b)), "partition",
                                     elapsed_ns=10, reference_ns=7, verified=True)
        # Independent Bellman-Ford relaxation, NOT the Dijkstra implementation.
        reference = {p: (0.0 if p == target else math.inf) for p in coords}
        for _ in range(len(coords) - 1):
            changed = False
            for a in coords:
                for b in coords:
                    if model._distance(a, b) != 1:
                        continue
                    candidate = reference[b] + model.edge_cost(a, b, "partition")
                    if candidate < reference[a]:
                        reference[a] = candidate
                        changed = True
            if not changed:
                break
        dijkstra, _ = model._reverse_distances(target, "partition")
        self.assertEqual(set(dijkstra), set(coords))
        for point in coords:
            self.assertAlmostEqual(dijkstra[point], reference[point], places=9)

    def test_invalid_parameters_and_edge_checks(self):
        for kw in ({"axis_weights": (.2,)*4}, {"axis_weights": (.2, .2, .2, .2, -1)},
                   {"size_slope": float("nan")}, {"size_threshold": -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                HeterogeneousWaveMemory(**kw)
        model = HeterogeneousWaveMemory()
        with self.assertRaises(ValueError):
            model.observe_transition(self.left, self.right, "partition",
                                     elapsed_ns=2, reference_ns=1, verified=True)
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.observe_transition(self.left, self.middle, "partition",
                                         elapsed_ns=bad, reference_ns=1, verified=True)
        self.assertEqual(model.snapshot()["observed_transition_edges"], 0)

    def test_fail_closed_memory_and_compiled_decisions(self):
        model = HeterogeneousWaveMemory(max_hops=2, min_gain=1)
        self.assertEqual(model.choose(self.middle), "manual")
        model.observe(self.left, "partition", elapsed_ns=1,
                      reference_ns=100, verified=True)
        self.assertEqual(model.choose(self.middle), "partition")
        self.assertEqual(model.freeze((self.left, self.middle)), ("partition", "partition"))
        snap = model.snapshot()
        self.assertEqual(snap["compiled_decisions"], 2)
        self.assertEqual(snap["observed_transition_edges"], 0)
        json.dumps(snap)
        model.mark_invalid(self.middle, "partition")
        self.assertEqual(model.choose(self.middle), "manual")

    def test_sound_damping_uses_weighted_cost(self):
        model = HeterogeneousWaveMemory(mode="sound", size_slope=0,
                                        axis_weights=(.2,) * 5, max_hops=2)
        model.observe(self.left, "partition", elapsed_ns=15,
                      reference_ns=100, verified=True)
        baseline = model._sound_arrival(self.middle.cell(), "partition")
        model.observe_transition(self.left, self.middle, "partition",
                                 elapsed_ns=200, reference_ns=10, verified=True)
        changed = model._sound_arrival(self.middle.cell(), "partition")
        # With one source, the weighted mean remains the same; confidence penalty rises.
        self.assertGreater(changed, baseline)


if __name__ == "__main__":
    unittest.main()


class TestDistanceCache(unittest.TestCase):
    def test_cache_is_shared_without_edge_measurements_and_invalidates(self):
        source = Workload(128, 64, 1, 2, 2)
        target = Workload(256, 64, 1, 2, 2)
        m = HeterogeneousWaveMemory(axis_weights=(.15,)*5,
                                    size_slope=0, max_hops=2)
        for route in ('partition', 'compiled'):
            m.observe(source, route, elapsed_ns=10, reference_ns=100, verified=True)
            self.assertTrue(math.isfinite(m._light_arrival(target.cell(), route)))
        self.assertEqual(m.snapshot()['compiled_distance_fields'], 1)
        old = m._light_arrival(target.cell(), 'partition')
        m.observe_transition(source, target, 'partition',
                             elapsed_ns=100, reference_ns=1, verified=True)
        self.assertEqual(m.snapshot()['compiled_distance_fields'], 0)
        new = m._light_arrival(target.cell(), 'partition')
        self.assertNotEqual(old, new)
        self.assertEqual(m.snapshot()['compiled_distance_fields'], 1)
