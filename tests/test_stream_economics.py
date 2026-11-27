"""Correctness and safe lifecycle for genuine multi-segment streams."""
import unittest
from alexpy.stream_economics import stream_schedule, run_stream, witness
from alexpy.stable_memory import workload_key, simple_rule


class StreamEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.schedule = stream_schedule(64, changes=(1,4), queries=(1,16),
                                        spans=("short","wide"), ticks=2)
        self.choices = {workload_key(w):simple_rule(w) for w in self.schedule}

    def test_continuous_one_cycle_matches_direct(self):
        expected = witness(self.schedule,cycles=1)
        self.assertEqual(run_stream(self.schedule,cycles=1,policy="simple").checksum,expected)
        self.assertEqual(run_stream(self.schedule,cycles=1,policy="stable",choices=self.choices).checksum,expected)

    def test_continuous_ten_cycles_retain_mutations(self):
        expected = witness(self.schedule,cycles=10)
        self.assertEqual(run_stream(self.schedule,cycles=10,policy="simple").checksum,expected)
        self.assertEqual(run_stream(self.schedule,cycles=10,policy="stable",choices=self.choices).checksum,expected)

    def test_switches_are_paid_and_counted(self):
        result=run_stream(self.schedule,cycles=3,policy="simple")
        self.assertGreater(result.switches,0)
        self.assertEqual(result.constructed,result.switches+1)
        self.assertGreater(result.switch_ns,0)
        self.assertEqual(result.segments,3*len(self.schedule))
        self.assertEqual(len(result.route_trace),result.segments)

    def test_frozen_all_direct_no_switch(self):
        frozen={workload_key(w):"direct" for w in self.schedule}
        result=run_stream(self.schedule,cycles=3,policy="stable",choices=frozen)
        self.assertEqual(result.switches,0)
        self.assertEqual(result.constructed,1)
        self.assertEqual(result.checksum,witness(self.schedule,cycles=3))

    def test_all_backends_as_single_choice(self):
        for name in ("direct","blocks","fenwick","prefix"):
            with self.subTest(name=name):
                choices={workload_key(w):name for w in self.schedule}
                result=run_stream(self.schedule,cycles=4,policy="stable",choices=choices)
                self.assertEqual(result.checksum,witness(self.schedule,cycles=4))
                self.assertEqual(result.switches,0)

    def test_invalid_routes_fail_closed(self):
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=1,policy="stable")
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=1,policy="stable",choices={})
        choices=dict(self.choices)
        choices[workload_key(self.schedule[0])]="unverified"
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=1,policy="stable",choices=choices)

    def test_no_hidden_input_resize(self):
        other=stream_schedule(128,changes=(1,),queries=(1,),spans=("short",))
        with self.assertRaises(ValueError):run_stream(self.schedule+other,cycles=1,policy="simple")

    def test_invalid_cycles(self):
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=0,policy="simple")
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=True,policy="simple")

    def test_precomputed_case_matches_generated(self):
        from alexpy.range_routes import make_case
        cases={workload_key(w):make_case(w) for w in self.schedule}
        a=run_stream(self.schedule,cycles=5,policy="simple")
        b=run_stream(self.schedule,cycles=5,policy="simple",case_cache=cases)
        self.assertEqual(a.checksum,b.checksum)

    def test_precomputed_case_missing_fails_closed(self):
        with self.assertRaises(ValueError):run_stream(self.schedule,cycles=1,policy="simple",case_cache={})

if __name__=='__main__':unittest.main()
