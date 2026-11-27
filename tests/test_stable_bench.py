"""Small but genuine fresh-process end-to-end smoke verification."""
import unittest
from alexpy.stable_bench import parse,run_experiment


class TestStableBench(unittest.TestCase):
    def test_small_fresh_process_protocol(self):
        args=parse(["--train-sizes","128,512","--holdout-sizes","384",
                    "--changes","1","--queries","1,8","--spans","short",
                    "--ticks","1","--pairs","2","--sessions","3","--no-probe"])
        r=run_experiment(args)
        self.assertEqual(r["sessions_count"],3)
        self.assertEqual(r["snapshot"]["independent_samples"],12)
        self.assertEqual(len(r["eval_workers"]),3)
        self.assertEqual(len(r["summary"]),2)
        self.assertEqual({x["gate"] for s in r["eval_workers"] for x in s["rows"]},{"PASS"})
        self.assertTrue(all(x["chosen"]["stable"] in ("direct","blocks","fenwick","prefix")
                            for x in r["summary"]))

    def test_no_overlapping_workload(self):
        args=parse(["--train-sizes","128","--holdout-sizes","128"])
        with self.assertRaises(ValueError):run_experiment(args)

    def test_needs_independent_processes(self):
        args=parse(["--sessions","2"])
        with self.assertRaises(ValueError):run_experiment(args)


if __name__=="__main__":unittest.main()
