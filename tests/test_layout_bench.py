import unittest
from alexpy.layout_bench import (
    make_input, mirrors, paired_samples, summarize, adjacent_edges, report
)
from alexpy.memory_candidates import CANDIDATES


class TestLayoutBench(unittest.TestCase):
    def test_input_shapes(self):
        for name in ("array", "memoryview"):
            for profile in ("small", "large", "mixed"):
                for size in (0, 1, 8, 40):
                    x = make_input(size, profile, name)
                    self.assertEqual(len(x), size)
                    self.assertEqual(mirrors(x), CANDIDATES["linear"](x))

    def test_invalid_state(self):
        for args in ((-1, "small", "array"), (1, "other", "array"),
                     (1, "small", "other")):
            with self.assertRaises(ValueError):
                make_input(*args)

    def test_fail_closed_correctness_before_timing(self):
        x = make_input(8, "small", "array")
        def bad(_):
            return 1
        with self.assertRaises(AssertionError):
            paired_samples(CANDIDATES["current"], bad, x,
                           pairs=1, target_seconds=.00001, epoch=0)

    def test_order_and_record_schema(self):
        x = make_input(8, "small", "array")
        res = paired_samples(CANDIDATES["current"], CANDIDATES["linear"],
                             x, pairs=3, target_seconds=.00001, epoch=0)
        self.assertEqual([s["order"] for s in res["samples"]], ["AB", "BA", "AB"])
        self.assertTrue(all(s["candidate_ns_per_call"] > 0 for s in res["samples"]))

    def test_report_edges_and_gate(self):
        r = report(sizes=(1, 8), profiles=("small",),
                   representations=("array",), epochs=1,
                   pairs=1, target_seconds=.00001)
        self.assertEqual(len(r["cases"]), 2)
        self.assertEqual(len(r["summary"]), 6)
        self.assertEqual(len(r["spatial_edges"]), 3)
        self.assertIn("BLOCKED", r["promotion_gate"])
        self.assertEqual(r["hardware_counters"], "NOT_MEASURED")
        self.assertEqual(len(r["sha256_without_digest"]), 64)

    def test_summary_regression_classification(self):
        samples = [{"epoch": 0, "size": 8, "profile": "small", "representation": "array",
                    "observations": {"linear": {"speedup": .75}}},
                   {"epoch": 1, "size": 8, "profile": "small", "representation": "array",
                    "observations": {"linear": {"speedup": .8}}}]
        rows = summarize(samples)
        self.assertTrue(rows[0]["every_epoch_regression"])
        self.assertFalse(rows[0]["every_epoch_faster"])
        self.assertEqual(adjacent_edges(rows), [])


if __name__ == "__main__":
    unittest.main()
