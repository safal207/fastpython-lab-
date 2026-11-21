"""Bench protocol correctness, no implicit promotion, and baseline parity."""
from __future__ import annotations

import argparse
import unittest

from alexpy.persistent_bench import make_case, workflow, case_measurements, collect, graph_edges, dot_graph


class TestPersistentBench(unittest.TestCase):
    def test_workflows_equivalent(self):
        for size in (1, 4, 127, 1024):
            for changes in (1, 8, 123):
                for reads in (1, 5):
                    initial, patches = make_case(size, changes, reads, 3)
                    outputs = {route: workflow(route, initial, patches, reads)
                               for route in ("list_full", "array_full", "owned_incremental")}
                    self.assertEqual(len(set(outputs.values())), 1)
                    self.assertEqual(make_case(size, changes, reads, 3), (initial, patches))

    def test_workflow_rejects_route(self):
        with self.assertRaises(ValueError):
            workflow("unknown", [1], [{0: 3}], 1)

    def test_case_measurements(self):
        case = case_measurements(32, 2, 3, 2, 1, 2, 0.03)
        self.assertEqual(case['state']['size'], 32)
        self.assertEqual(case['correctness_mirrors'], 'PASS')
        self.assertEqual(case['logical_checks'], 6)
        self.assertEqual(len(case['samples_ns_per_workflow']['list_full']), 1)
        self.assertEqual(len(case['samples_ns_per_workflow']['list_full'][0]), 2)
        self.assertGreater(case['speedup_vs_owned']['array_full'], 0)

    def test_bad_case_configuration(self):
        for size, changes, reads, ticks in ((0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)):
            with self.assertRaises(ValueError):
                make_case(size, changes, reads, ticks)

    def test_one_axis_graph_edges(self):
        summary = [{"state": {"size": n, "changes_per_tick": k,
                              "reads_per_tick": r, "ticks": 3},
                    "list_full_ratio_median": 1.4}
                   for n in (8, 128) for k in (1, 4) for r in (1, 8, 16)]
        edges = graph_edges(summary)
        self.assertEqual(len(edges), 20)
        for edge in edges:
            differences = [axis for axis in ("size", "changes_per_tick", "reads_per_tick")
                           if edge["from"][axis] != edge["to"][axis]]
            self.assertEqual(differences, [edge["axis"]])
        text = dot_graph(summary, edges)
        self.assertIn("digraph", text)
        self.assertEqual(text.count('->'), 20)

    def test_fresh_process_probe(self):
        args = argparse.Namespace(worker=False, sizes='8', changes='1', reads='1',
                                  ticks=2, epochs=1, pairs=2, sessions=2,
                                  target_ms=0.01)
        report = collect(args)
        self.assertEqual(report['promotion'], 'BLOCKED_EXPERIMENTAL')
        self.assertEqual(len(report['sessions']), 2)
        self.assertNotEqual(report['sessions'][0]['pid'], report['sessions'][1]['pid'])
        self.assertTrue(report['sha256'])
        self.assertEqual(report['graph_edges'], [])


if __name__ == "__main__":
    unittest.main()
