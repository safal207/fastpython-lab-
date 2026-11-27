"""Fail-closed correctness and sparse-state controls for ant-inspired planner."""
import json
import math
import random
import unittest

from alexpy.ant_memory import AntMemoryGraph, Workload, ROUTES


class AntMemoryTests(unittest.TestCase):
    def setUp(self):
        self.s = Workload(1024, 256, 1, 4, 8)

    def test_cell_coarsening_and_validation(self):
        self.assertEqual(self.s.cell(), (10, 8, 0, 2, 3))
        self.assertEqual(self.s.cell(), Workload(1400, 256, 1, 4, 8).cell())
        for bad in ((0,256,1,4,8),(5,256,6,4,8),(4,256,1,True,8)):
            with self.assertRaises(ValueError): Workload(*bad)

    def test_empty_matrix_is_sparse_and_conservative(self):
        m = AntMemoryGraph()
        self.assertEqual(m.choose(self.s), "manual")
        self.assertEqual(m.snapshot()["nonzero_edges"], 0)
        self.assertEqual(m.snapshot()["state_transitions"], 0)

    def test_ant_reinforces_verified_fast_route(self):
        m = AntMemoryGraph()
        for _ in range(3):
            m.observe(self.s,"partition",elapsed_ns=10,reference_ns=40,verified=True)
            m.observe(self.s,"manual",elapsed_ns=40,reference_ns=40,verified=True)
            m.advance()
        self.assertEqual(m.choose(self.s),"partition")
        self.assertGreater(m.scores(self.s)["partition"],m.scores(self.s)["manual"])

    def test_mask_and_unknown_state_fail_conservatively(self):
        m = AntMemoryGraph(max_distance=1)
        m.observe(self.s,"partition",elapsed_ns=10,reference_ns=40,verified=True)
        self.assertEqual(m.choose(self.s,allowed=("manual",)),"manual")
        self.assertEqual(m.choose(Workload(65536,256,1,4,8)),"manual")
        with self.assertRaises(ValueError): m.choose(self.s,allowed=("fake",))
        with self.assertRaises(ValueError): m.choose(self.s,allowed=())

    def test_failed_gate_quarantines_route(self):
        m = AntMemoryGraph()
        m.observe(self.s,"partition",elapsed_ns=10,reference_ns=40,verified=True)
        with self.assertRaisesRegex(ValueError, "unverified"):
            m.observe(self.s,"partition",elapsed_ns=5,reference_ns=40,verified=False)
        self.assertNotIn("partition",m.scores(self.s))
        self.assertEqual(m.choose(self.s), "manual")
        with self.assertRaisesRegex(ValueError,"quarantined"):
            m.observe(self.s,"partition",elapsed_ns=10,reference_ns=40,verified=True)

    def test_invalid_measurements_dont_change_state(self):
        m=AntMemoryGraph()
        for invalid in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                m.observe(self.s,"manual",elapsed_ns=invalid,reference_ns=10,verified=True)
        self.assertEqual(m.snapshot()["nonzero_edges"],0)

    def test_temporal_decay_doesnt_mutate_on_read(self):
        m = AntMemoryGraph(rho=.5)
        m.observe(self.s,"partition",elapsed_ns=1,reference_ns=3,verified=True)
        before=m.scores(self.s)["partition"]
        m.advance(2)
        after=m.scores(self.s)["partition"]
        self.assertLess(after,before)
        self.assertEqual(after,m.scores(self.s)["partition"])

    def test_neighbors_and_seeded_exploration(self):
        m = AntMemoryGraph()
        m.observe(self.s,"partition",elapsed_ns=1,reference_ns=10,verified=True)
        holdout=Workload(2048,256,1,4,8)
        self.assertEqual(m.choose(holdout),"partition")
        a=random.Random(11);b=random.Random(11)
        self.assertEqual([m.choose(holdout,rng=a) for _ in range(10)],
                         [m.choose(holdout,rng=b) for _ in range(10)])

    def test_transition_memory_separate_from_performance(self):
        m=AntMemoryGraph()
        after=Workload(2048,256,1,4,8)
        m.trace(self.s,after,"manual")
        m.trace(self.s,after,"manual")
        snap=m.snapshot()
        self.assertEqual(snap["transitions"][0]["visits"],2)
        self.assertEqual(snap["nonzero_edges"],0)
        self.assertEqual(m.choose(after),"manual")

    def test_bounds_and_json_audit(self):
        m=AntMemoryGraph()
        m.observe(self.s,"manual",elapsed_ns=0.000001,reference_ns=1e30,verified=True)
        self.assertLessEqual(m.scores(self.s)["manual"],64)
        state=json.loads(json.dumps(m.snapshot()))
        self.assertEqual(state["schema"],"ALEXPY-ANT-MEMORY/1")
        self.assertEqual(state["cells"],1)
        self.assertEqual(state["nonzero_edges"],1)

    def test_compiled_policy_cache_invalidates_with_new_evidence(self):
        m=AntMemoryGraph()
        self.assertEqual(m.choose(self.s),"manual")
        self.assertEqual(m.snapshot()["compiled_policy_cells"],1)
        m.observe(self.s,"partition",elapsed_ns=1,reference_ns=20,verified=True)
        self.assertEqual(m.snapshot()["compiled_policy_cells"],0)
        self.assertEqual(m.choose(self.s),"partition")
        m.advance()
        self.assertEqual(m.snapshot()["compiled_policy_cells"],0)
        m.mark_invalid(self.s,"partition")
        self.assertEqual(m.choose(self.s),"manual")

    def test_parameters_rejected(self):
        for kw in ({"rho":0},{"rho":1},{"alpha":-1},{"neighbors":0},{"max_distance":-1}):
            with self.assertRaises(ValueError): AntMemoryGraph(**kw)
        m=AntMemoryGraph()
        with self.assertRaises(ValueError): m.advance(0)
        with self.assertRaises(ValueError): m.trace(self.s,self.s,"bad")


if __name__=="__main__": unittest.main()
