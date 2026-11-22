"""Integrity tests for held-out stream economics and hardware profile gate."""
import unittest
from argparse import Namespace
from unittest.mock import patch
from alexpy.stream_bench import (csv_positive,host_metadata,valid_splits,
                                  evaluate_worker,train_worker,experiment)
from alexpy.stable_memory import environment_key, workload_key
from alexpy.stream_economics import stream_schedule


def mini():
    return Namespace(train_sizes=(64,256),holdout_sizes=(128,),changes=(1,),
                     queries=(1,16),spans=("short",),ticks=1,pairs=2,
                     sessions=3,cycles=(1,3),reps=2,device_tag="unit-test-host")

class StreamBenchTests(unittest.TestCase):
    def test_csv_positive(self):
        self.assertEqual(csv_positive("1,3,7"),(1,3,7))
        for bad in ("","0","1,1","-1,3"):
            with self.subTest(value=bad),self.assertRaises(ValueError):csv_positive(bad)

    def test_train_holdout_coarse_disjoint(self):
        a,b=valid_splits(mini())
        self.assertTrue(a and b)
        args=mini();args.holdout_sizes=(256,)
        with self.assertRaises(ValueError):valid_splits(args)
        args=mini();args.holdout_sizes=(96,)
        with self.assertRaises(ValueError):valid_splits(args)

    def test_unsupported_changes(self):
        args=mini();args.changes=(65,)
        with self.assertRaises(ValueError):valid_splits(args)

    def test_environment_gate(self):
        args=mini()
        choices={workload_key(w):"direct" for w in
                 stream_schedule(128,changes=args.changes,queries=args.queries,
                                 spans=args.spans,ticks=args.ticks)}
        with self.assertRaises(AssertionError):
            evaluate_worker(args,{"choices":choices,"environment":"another-machine"})

    def test_host_metadata(self):
        meta=host_metadata()
        self.assertIn("processor",meta)
        self.assertIn("machine",meta)
        self.assertIn("python",meta)

    def test_training_worker_verified(self):
        result=train_worker(mini())
        self.assertTrue(all(row["gate"]=="PASS" for row in result["rows"]))
        self.assertEqual(len(result["rows"]),4)

    def test_eval_worker_blind(self):
        args=mini()
        choices={workload_key(w):"direct" for w in
                 stream_schedule(128,changes=args.changes,queries=args.queries,
                                 spans=args.spans,ticks=args.ticks)}
        data={"choices":choices,"environment":environment_key(device_tag=args.device_tag)}
        result=evaluate_worker(args,data)
        self.assertEqual(len(result["records"]),2)
        self.assertTrue(all(r["gate"]=="PASS" for r in result["records"]))
        self.assertTrue(all(r["details"]["stable"][0]["switches"]==0 for r in result["records"]))

    def test_disallow_two_sessions(self):
        args=mini();args.sessions=2
        with self.assertRaises(ValueError):experiment(args)

    def test_full_end_to_end_blind_protocol(self):
        args=mini()
        result=experiment(args)
        self.assertEqual(result["schema"],"ALEXPY-STREAM-ECONOMICS/1")
        self.assertEqual(len(result["training_workers"]),3)
        self.assertEqual(len(result["evaluation_workers"]),3)
        self.assertEqual(len(result["cases"]),2)
        self.assertEqual(len(result["pooled"]),2)
        if all(x["net_savings_ns"]<0 for x in result["pooled"]):
            self.assertIsNone(result["observed_break_even_cycles"])
        self.assertTrue(all(r["gate"]=="PASS" for x in result["evaluation_workers"]
                            for r in x["records"]))

if __name__=='__main__':unittest.main()
