import math
import unittest

from alexpy.graph_search import State
from alexpy.lattice_lab import (Cell, analyze, collect, coordinates, estimate_container_bytes,
                                lattice_edges, multiscale_lattice, separation, size_holdout,
                                validate_cells)


def cell(n, epoch=0, speedup=2., container="list", profile="small", stamp=None):
    return Cell(State(container, n, profile), epoch,
                stamp if stamp is not None else 100_000 * (epoch+1),
                100., 100./speedup, 48+n*8, "test")


class LatticeLabTests(unittest.TestCase):
    def test_coordinate_mapping_and_manhattan_grid(self):
        a = State("list", 1, "small")
        b = State("tuple", 1, "small")
        self.assertEqual(coordinates(a), (0, 1, 0, 1))
        self.assertEqual(separation(a, b), 1)
        self.assertEqual(separation(a, State("tuple", 8, "huge")), 4)

    def test_duplicate_cell_or_nonmonotonic_time_blocked(self):
        a = cell(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_cells([a, a])
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            validate_cells([a, cell(1, epoch=1, stamp=50)])
        with self.assertRaises(ValueError):
            Cell(State("list", 1, "small"), 0, 0, 0., 10., 8, "test")

    def test_spatial_edges_and_time_edges_distinct(self):
        cs = [cell(1), cell(8), cell(1, epoch=1)]
        e = lattice_edges(cs)
        self.assertEqual(sum(x["kind"] == "space" for x in e), 1)
        self.assertEqual(sum(x["kind"] == "time" for x in e), 1)
        self.assertEqual(next(x for x in e if x["kind"] == "time")["from"],
                         next(x for x in e if x["kind"] == "time")["to"])

    def test_multiscale_has_sparse_finite_scale_signature(self):
        sig = multiscale_lattice([cell(1), cell(8), cell(16)], scales=(1,2,4))
        self.assertEqual(sig["0"]["1"]["pair_count"], 2)
        self.assertEqual(sig["0"]["2"]["pair_count"], 1)
        self.assertIsNone(sig["0"]["4"]["median_abs_log_delta"])
        with self.assertRaises(ValueError):
            multiscale_lattice([cell(1)], (1,1))

    def test_memory_is_estimate_and_deduplicates_references(self):
        same = 10**120
        self.assertGreater(estimate_container_bytes([same,same]), 0)
        import sys
        self.assertEqual(estimate_container_bytes([same,same]),
                         sys.getsizeof([same,same]) + sys.getsizeof(same))

    def test_holdout_split_disallows_any_length_leakage(self):
        data = [cell(n, speedup=1+n/20) for n in (0,1,8,16,1024,10000)]
        test = size_holdout(data, holdout_lengths=(8,1024))
        self.assertEqual(test["holdout_count"], 2)
        self.assertEqual(test["train_count"], 4)
        self.assertTrue(all("|8|" in p["state"] or "|1024|" in p["state"]
                            for p in test["predictions"]))
        self.assertEqual(test["evidence"], "EXPLORATORY_ONLY_NO_PROMOTION")
        for k in ("prior", "one_hop", "multiscale"):
            self.assertTrue(math.isfinite(test["metrics"][k]["mae_log_speedup"]))
        with self.assertRaises(ValueError):
            size_holdout(data, holdout_lengths=(123,))

    def test_regression_detection_and_prediction_misses(self):
        data = [cell(n, speedup=.6 if n == 8 else 2.)
                for n in (0,1,8,16,1024,10000)]
        result = size_holdout(data, holdout_lengths=(8,))
        self.assertEqual(result["actual_regressions"], 1)
        self.assertEqual(result["missed_regressions"], 1)

    def test_measurement_smoke_correctness_and_report(self):
        cells = collect(epochs=2, repeats=2, loops=2,
                        lengths=(1,8), containers=("list",), profiles=("small",))
        report = analyze(cells, holdout_lengths=(8,))
        self.assertEqual(len(report["cells"]), 4)
        self.assertTrue(any(x["kind"] == "time" for x in report["edges"]))
        self.assertEqual(report["size_holdout"]["holdout_count"], 2)
        with self.assertRaises(ValueError):
            collect(epochs=0)
        with self.assertRaises(ValueError):
            collect(lengths=(1,1))


if __name__ == "__main__":
    unittest.main()
