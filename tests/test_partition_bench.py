"""Research harness correctness and reproducibility smoke test."""
import unittest

from alexpy.partition_bench import ROUTES, workflow, measure_case
from alexpy.dependency_bench import make_case


class PartitionBenchmarkTests(unittest.TestCase):
    def test_all_routes_same_checksum(self):
        for n in (8, 32, 128):
            initial, patches = make_case(n, 8, 2, 3)
            values = {route: workflow(route, initial, patches, 2, 8)
                      for route in ROUTES}
            self.assertEqual(len(set(values.values())), 1)

    def test_gate(self):
        case = measure_case(64, 8, 2, 3, 2, 1, 2)
        self.assertEqual(case["gate"], "PASS")
        self.assertEqual(len(case["raw_ns"]["partition"]), 1)
        self.assertGreater(case["median_ns"]["partition"], 0)


if __name__ == "__main__": unittest.main()
