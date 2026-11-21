"""Holdout protocol invariants for the heterogeneous selector."""
import unittest
from alexpy.hetero_bench import POLICIES, run_session


class TestHeterogeneousBench(unittest.TestCase):
    def test_training_holdout_freeze_and_correctness(self):
        out = run_session(train_sizes=(32,), holdout_sizes=(64,), changes=(1,),
                          reads=(1,), width=8, ticks=2, pairs=1)
        self.assertEqual(len(out["training"]), 1)
        self.assertEqual(len(out["holdout"]), 1)
        row = out["holdout"][0]
        self.assertEqual(set(row["frozen"]), set(POLICIES))
        self.assertEqual(row["state"]["size"], 64)
        for p in POLICIES:
            self.assertGreater(row["median_ns"][p], 0)
        self.assertEqual(out["memory"]["hetero_light"]["observed_transition_edges"], 0)

    def test_training_holdout_overlap_rejected(self):
        with self.assertRaises(ValueError):
            run_session(train_sizes=(64,), holdout_sizes=(65,), changes=(1,),
                        reads=(1,), width=8, ticks=2, pairs=1)


if __name__ == "__main__":
    unittest.main()
