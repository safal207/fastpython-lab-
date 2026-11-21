import unittest
from unittest.mock import patch

from alexpy.graph_search import (
    AXES, CONTAINERS, LENGTHS, PHASES, PROFILES, State,
    all_states, bootstrap_interval, check_correctness, choose_cases,
    edges, explore, make_values, mirror_oracles, neighbors,
    paired_measure, propose_followups, to_dot, verdict,
)


class GraphTest(unittest.TestCase):
    def test_graph_all_states_unique(self):
        states = all_states()
        self.assertEqual(len(states), len(CONTAINERS)*len(LENGTHS)*len(PROFILES)*len(PHASES))
        self.assertEqual(len(states), len({s.id for s in states}))
        self.assertEqual(len(edges()), len(set(edges())))

    def test_edges_change_exactly_one_axis(self):
        for state in all_states():
            for axis, next_state in neighbors(state):
                diffs = [a for a in AXES if getattr(state, a) != getattr(next_state, a)]
                self.assertEqual(diffs, [axis])
                if axis == "length":
                    self.assertEqual(abs(LENGTHS.index(state.length) - LENGTHS.index(next_state.length)), 1)

    def test_case_selection_deterministic_and_diverse(self):
        self.assertEqual(choose_cases(20), choose_cases(20))
        picks = choose_cases(20)
        self.assertEqual(len(picks), len(set(picks)))
        self.assertEqual({s.container for s in picks}, set(CONTAINERS))
        self.assertEqual({s.length for s in picks}, set(LENGTHS))
        self.assertEqual({s.profile for s in picks}, set(PROFILES))
        with self.assertRaises(ValueError):
            choose_cases(0)

    def test_mirrors_and_native_or_fallback(self):
        for state in all_states():
            if state.phase != "steady":
                continue
            with self.subTest(state=state.id):
                values = make_values(state)
                a, b, c = mirror_oracles(values)
                self.assertEqual(a, b)
                self.assertEqual(b, c)
                gate = check_correctness(state, values)
                self.assertTrue(gate["correct"], gate)
                if state.container == "list_subclass":
                    self.assertFalse(gate["fastpath_used"])

    def test_incorrect_candidate_blocks_perf(self):
        state = State("list", 8, "small")
        with patch("alexpy.graph_search.sum_squares", return_value=-1):
            gate = check_correctness(state, make_values(state))
        self.assertEqual(gate["reason"], "MIRROR_DIVERGENCE")
        self.assertEqual(verdict(gate["correct"], [999]*20), "BLOCKED_INCORRECT")

    def test_verdict_statistics_are_conservative(self):
        self.assertEqual(verdict(True, [2.0]*5), "EXPLORATORY_FEW_SAMPLES")
        self.assertEqual(verdict(True, [2.0]*9), "CANDIDATE_GAIN")
        self.assertEqual(verdict(True, [.5]*9), "CANDIDATE_REGRESSION")
        self.assertEqual(verdict(True, [1.0]*9), "INCONCLUSIVE")
        self.assertEqual(bootstrap_interval([2.0]*9), [2.0, 2.0])

    def test_pairing_does_not_change_values(self):
        values = [0, 2, -3]
        report = paired_measure(values, repeats=3, target_seconds=.0001)
        self.assertEqual(len(report["steady"]["paired_speedup"]), 3)
        self.assertEqual(values, [0, 2, -3])

    def test_end_to_end_json_and_graph(self):
        report = explore(limit=4, repeats=2, target_seconds=.0001)
        self.assertEqual(len(report["results"]), 4)
        self.assertEqual(report["graph"]["total_states"], len(all_states()))
        self.assertEqual(len(report["report_sha256"]), 64)
        self.assertTrue(all(row["verdict"] == "EXPLORATORY_FEW_SAMPLES" for row in report["results"]))
        self.assertIn("graph AlexPy", to_dot(report))
        selected = {row["state"] for row in report["results"]}
        self.assertTrue(all(f["state"] not in selected for f in report["next_frontier"]))

    def test_followups_prioritize_verified_candidate_neighbors(self):
        state = State("list", 16, "small")
        rows = [{"axes": vars(state), "verdict": "CANDIDATE_GAIN"}]
        followups = propose_followups(rows, {state.id})
        self.assertTrue(followups)
        self.assertTrue(all(f["state"].endswith("|steady") for f in followups))

    def test_reject_invalid_axes(self):
        with self.assertRaises(ValueError):
            State("dict", 1, "small")
        with self.assertRaises(ValueError):
            State("list", 3, "small")


if __name__ == "__main__":
    unittest.main()
