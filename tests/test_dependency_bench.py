"""Semantic parity and small smoke checks for graph-vs-control benchmark."""
import argparse
import unittest

from alexpy.dependency_bench import ROUTES, collect, make_case, measure_case, workflow


class TestDependencyBenchmark(unittest.TestCase):
    def test_parity_all_routes(self):
        for size in (1, 33, 256, 513):
            for k in (1, min(size, 7)):
                initial, patches = make_case(size, 32, k, 4)
                expected = workflow("dense", initial, patches, 2, 32)
                for route in ROUTES:
                    with self.subTest(size=size, k=k, route=route):
                        self.assertEqual(workflow(route, initial, patches, 2, 32),
                                         expected)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            make_case(8, 4, 9, 2)
        with self.assertRaises(ValueError):
            make_case(8, 0, 1, 2)
        with self.assertRaises(ValueError):
            workflow("bogus", [1, 2], [{0: 3}], 1, 2)

    def test_small_measurement_has_control_mirrors(self):
        result = measure_case(64, 16, 1, 2, 1, 1, 2, .05)
        self.assertEqual(result["correctness_gate"], "PASS")
        self.assertEqual(result["node_count"], 13)
        self.assertEqual(result["dirty_counters"]["touched_windows"], 4)
        self.assertTrue(result["speedup_vs_graph"]["dense"] > 0)
        self.assertTrue(result["speedup_vs_graph"]["manual"] > 0)

    def test_fresh_interpreter_smoke(self):
        args = argparse.Namespace(
            sizes="64", width=16, changes="1", ticks="2", reads="1",
            sessions=1, epochs=1, pairs=1, target_ms=.05, worker=False,
        )
        result = collect(args)
        self.assertEqual(result["protocol"], "ALEXPY-DEPENDENCY-M10/0.1")
        self.assertEqual(len(result["sessions"]), 1)
        self.assertEqual(len(result["summary"]), 1)
        self.assertEqual(result["summary"][0]["state"]["size"], 64)


if __name__ == "__main__":
    unittest.main()
