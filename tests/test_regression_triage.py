import unittest
from unittest.mock import patch

from alexpy.regression_triage import (
    FOCUS_CASES, assess, compare_edges, gate_and_trace, reproduce,
    select_cases, trace_path,
)
from alexpy.graph_search import State, make_values, neighbors


class TriageTest(unittest.TestCase):
    def test_regression_cases_from_observed_report(self):
        self.assertEqual([x.id for x in FOCUS_CASES], [
            "tuple|1|huge|steady", "list_subclass|0|small|steady",
            "list_subclass|1|mixed|steady", "list_subclass|8|small|steady"])

    def test_code_path_witness_and_gate(self):
        reasons = {"tuple|1|huge|steady": "OUTSIDE_INT64_SQUARE",
                   "list_subclass|0|small|steady": "UNSUPPORTED_CONTAINER",
                   "list_subclass|1|mixed|steady": "UNSUPPORTED_CONTAINER",
                   "list_subclass|8|small|steady": "UNSUPPORTED_CONTAINER",
                   "tuple|8|acc_overflow|steady": "NATIVE_SUPPORTED",
                   "list|10000|acc_overflow|steady": "INT64_ACCUMULATOR_OVERFLOW"}
        for node, reason in reasons.items():
            tokens = node.split("|")
            state = State(tokens[0], int(tokens[1]), tokens[2], tokens[3])
            _, gate, trace = gate_and_trace(state)
            self.assertTrue(gate["correct"], (state, gate))
            self.assertEqual(trace["reason"], reason, node)

    def test_controls_are_graph_edges(self):
        cases, edges = select_cases()
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual(len(edges), 2 * len(FOCUS_CASES))
        for edge in edges:
            s = next(s for s in cases if s.id == edge["from"])
            control = next(v for v in cases if v.id == edge["to"])
            self.assertIn((edge["axis"], control), neighbors(s))
            if edge["axis"] != "container":
                self.assertNotEqual(list(make_values(control)), list(make_values(s)))

    def test_classification_requires_process_replication(self):
        self.assertEqual(assess([.2, .2]), "INSUFFICIENT_PROCESS_REPLICATION")
        self.assertEqual(assess([.7, .8, .7]), "REPRODUCED_REGRESSION_CANDIDATE")
        self.assertEqual(assess([1.2, 1.5, 1.4]), "REPRODUCED_GAIN_CANDIDATE")
        self.assertEqual(assess([.7, 1.2, .8]), "UNSTABLE_OR_NEUTRAL")
        self.assertEqual(assess([1., 1., 1.]), "UNSTABLE_OR_NEUTRAL")

    def test_no_native_has_distinct_path(self):
        with patch("alexpy.regression_triage._native", None):
            self.assertEqual(trace_path([1])["reason"], "NATIVE_UNAVAILABLE")

    def test_reproduce_two_processes_is_not_confirmed(self):
        state = State("list", 8, "small")
        report = reproduce(states=[state], sessions=2, repeats=1, target_seconds=.0001)
        self.assertEqual(len(report["raw_sessions"]), 2)
        self.assertEqual(report["rows"][0]["verdict"], "INSUFFICIENT_PROCESS_REPLICATION")
        self.assertEqual(len(report["sha256"]), 64)

    def test_fail_closed_before_child_run(self):
        state = State("list", 8, "small")
        with patch("alexpy.regression_triage.gate_and_trace",
                   return_value=([], {"correct": False}, {})):
            with patch("alexpy.regression_triage.subprocess.run") as child:
                with self.assertRaises(RuntimeError):
                    reproduce(states=[state])
                child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
