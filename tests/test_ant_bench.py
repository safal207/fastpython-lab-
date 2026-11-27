"""Holdout-split correctness and workflow-level Ant-Memory smoke tests."""
import argparse
import unittest
from unittest.mock import patch

from alexpy.ant_bench import _csv, _strict_disjoint, prepare, online_pair, run_session, evaluate_routes
from alexpy.ant_memory import AntMemoryGraph, Workload, ROUTES


class AntBenchmarkTests(unittest.TestCase):
    def test_equivalence_against_dense(self):
        state=Workload(64,8,2,3,2)
        initial, patches, reference=prepare(state)
        self.assertIsInstance(reference,int)
        result=evaluate_routes(state,pairs=2)
        self.assertEqual(result["gate"],"PASS")
        self.assertEqual(set(result["raw_ns"]),set(ROUTES))

    def test_blind_split_validated(self):
        with self.assertRaises(ValueError): _strict_disjoint((256,), (256,))
        with self.assertRaises(ValueError): _csv("0,4")
        with self.assertRaises(ValueError): _csv("2,2")
        with self.assertRaises(ValueError):
            run_session(train_sizes=(64,),holdout_sizes=(65,),
                        changes=(1,),reads=(1,),width=8,ticks=2,pairs=1)

    def test_benchmark_runs_and_does_not_train_on_holdout(self):
        session=run_session(train_sizes=(32,),holdout_sizes=(64,),
                            changes=(1,),reads=(1,),width=8,ticks=2,pairs=1)
        self.assertEqual(len(session["train_samples"]),1)
        self.assertEqual(len(session["holdout"]),1)
        self.assertEqual(session["holdout"][0]["state"]["size"],64)
        # Exactly 4 trained actions, no holdout action created after evaluation.
        self.assertEqual(session["matrix_snapshot"]["nonzero_edges"],4)
        self.assertEqual(len(session["frozen_predictions"]),1)
        self.assertGreater(session["training_ns"],0)
        self.assertGreater(session["matrix_json_bytes"],0)

    def test_online_measures_selected_route_with_decision_cost(self):
        state=Workload(32,8,1,2,1)
        mem=AntMemoryGraph()
        report=online_pair(state,"manual",mem,pairs=2)
        self.assertEqual(set(report["median_ns"]),{"ant","manual","partition"})
        self.assertTrue(all(x>0 for x in report["median_ns"].values()))
        with self.assertRaisesRegex(AssertionError,"frozen prediction"):
            online_pair(state,"dense",mem,pairs=1)


if __name__=="__main__": unittest.main()
