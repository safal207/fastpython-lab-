"""Opt-in route memory should fail closed under noise, drift and invalid data."""
import math
import unittest

from alexpy.range_routes import ROUTES, RangeWorkload, checked_routes
from alexpy.stable_memory import (StableRouteMemory, environment_key,
                                  simple_rule, workload_key)


TRAIN=RangeWorkload(128,1,8,4,"wide")
HELD=RangeWorkload(384,1,8,4,"wide")
SWITCH_AFTER=RangeWorkload(128,1,32,4,"wide")
PREFIX_TRAIN=RangeWorkload(512,1,32,4,"wide")
PREFIX_HELD=RangeWorkload(1536,1,32,4,"wide")


def samples(w, *, winner="blocks", poor=False):
    base=simple_rule(w)
    times={r:[10000,10100,9900] for r in ROUTES}
    times[base]=[10000,10100,9900]
    if not poor:
        times[winner]=[2000,2050,1950]
    return times


class TestStableMemory(unittest.TestCase):
    def new(self, **kwargs):
        return StableRouteMemory(environment="local-machine/a",**kwargs)

    def fit(self,m,w=TRAIN, *,winner="blocks",count=3):
        for i in range(count):
            m.observe_session(w,process_id=str(i),raw_ns=samples(w,winner=winner),
                              verified=True,environment="local-machine/a")

    def test_profile_has_explicit_device_tag(self):
        self.assertNotEqual(environment_key(device_tag="cpu-A"),environment_key(device_tag="cpu-B"))
        with self.assertRaises(ValueError):environment_key(device_tag="")
        with self.assertRaises(ValueError):environment_key(device_tag=False)

    def test_correctness_gate_mandatory(self):
        m=self.new()
        with self.assertRaisesRegex(ValueError,"unverified"):
            m.observe_session(TRAIN,process_id="1",raw_ns=samples(TRAIN),verified=False,
                              environment="local-machine/a")
        self.assertEqual(m.snapshot()["train_cells"],0)

    def test_training_foreign_device_rejected(self):
        with self.assertRaisesRegex(ValueError,"cross-environment"):
            self.new().observe_session(TRAIN,process_id="1",raw_ns=samples(TRAIN),
                                     verified=True,environment="different")

    def test_no_duplicate_process_pseudo_independence(self):
        m=self.new()
        self.fit(m,count=1)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            m.observe_session(TRAIN,process_id="0",raw_ns=samples(TRAIN),verified=True,
                              environment="local-machine/a")
        m.compile([HELD],environment="local-machine/a")
        self.assertEqual(m.choose(HELD,environment="local-machine/a").route,"direct")

    def test_accepts_large_stable_gain(self):
        m=self.new(); self.fit(m)
        decision=m.compile([HELD],environment="local-machine/a")[workload_key(HELD)]
        self.assertEqual(decision.route,"blocks")
        self.assertEqual(decision.independent_processes,3)
        self.assertLess(decision.predicted_upper_log_ratio,-math.log(1.1))

    def test_outlier_blocks_speculative_gain(self):
        m=self.new()
        for i in range(3):
            raw=samples(TRAIN)
            if i==2:raw["blocks"]=[14000,14200,13800]
            m.observe_session(TRAIN,process_id=str(i),raw_ns=raw,verified=True,
                              environment="local-machine/a")
        d=m.compile([HELD],environment="local-machine/a")[workload_key(HELD)]
        self.assertEqual(d.route,"direct")
        self.assertEqual(d.reason,"fallback_risk_or_gain")

    def test_no_training_profile_leak_on_compile(self):
        m=self.new(); self.fit(m)
        m.compile([HELD],environment="foreign")
        self.assertEqual(m.choose(HELD,environment="local-machine/a").route,"direct")
        self.assertEqual(m.choose(HELD,environment="foreign").reason,"environment_mismatch")

    def test_foreign_profile_fails_even_with_precompiled_good_choice(self):
        m=self.new(); self.fit(m); m.compile([HELD],environment="local-machine/a")
        self.assertEqual(m.choose(HELD,environment="local-machine/a").route,"blocks")
        self.assertEqual(m.choose(HELD,environment="other").route,"direct")

    def test_unknown_workload_fails_closed(self):
        m=self.new();self.fit(m);m.compile([HELD],environment="local-machine/a")
        other=RangeWorkload(2048,1,32,4,"wide")
        self.assertEqual(m.choose(other,environment="local-machine/a").route,simple_rule(other))

    def test_exact_holdout_training_leak_rejected(self):
        m=self.new();self.fit(m)
        with self.assertRaisesRegex(ValueError,"leaked"):
            m.compile([TRAIN],environment="local-machine/a")

    def test_freeze_does_not_accept_new_training(self):
        m=self.new();self.fit(m);m.compile([HELD],environment="local-machine/a")
        with self.assertRaisesRegex(RuntimeError,"frozen"):
            self.fit(m)
        with self.assertRaisesRegex(RuntimeError,"already frozen"):
            m.compile([PREFIX_HELD],environment="local-machine/a")

    def test_quarantine_route(self):
        m=self.new();self.fit(m);m.quarantine(TRAIN,"blocks")
        self.assertEqual(m.compile([HELD],environment="local-machine/a")[workload_key(HELD)].route,"direct")

    def test_quarantined_holdout_baseline_fails_closed(self):
        m=self.new();self.fit(m)
        m.quarantine(HELD,"direct")
        with self.assertRaisesRegex(RuntimeError,"no trusted fallback"):
            m.compile([HELD],environment="local-machine/a")

    def test_invalid_timings_fail_before_mutation(self):
        m=self.new()
        bad=samples(TRAIN)
        bad["blocks"]=[100,0,110]
        with self.assertRaisesRegex(ValueError,"positive"):
            m.observe_session(TRAIN,process_id="bad",raw_ns=bad,
                              verified=True,environment="local-machine/a")
        self.assertEqual(m.snapshot()["independent_samples"],0)

    def test_short_prepared_switch_not_amortized(self):
        m=self.new()
        for i in range(3):
            m.observe_switch(TRAIN,SWITCH_AFTER,"direct","direct",process_id=str(i),
                             setup_ns=0,tick_ns=1000,verified=True,
                             environment="local-machine/a")
            m.observe_switch(TRAIN,SWITCH_AFTER,"direct","prefix",process_id=str(i),
                             setup_ns=5000,tick_ns=100,verified=True,
                             environment="local-machine/a")
        self.assertEqual(m.advise_switch(TRAIN,SWITCH_AFTER,"direct","prefix",horizon=1,
                                        environment="local-machine/a")[0],"direct")
        choice=m.advise_switch(TRAIN,SWITCH_AFTER,"direct","prefix",horizon=10,
                               environment="local-machine/a")
        self.assertEqual(choice[0],"prefix")
        self.assertGreater(choice[2],0)

    def test_switch_must_share_multiple_processes(self):
        m=self.new()
        m.observe_switch(TRAIN,SWITCH_AFTER,"direct","prefix",process_id="0",setup_ns=1,
                         tick_ns=10,verified=True,environment="local-machine/a")
        self.assertEqual(m.advise_switch(TRAIN,SWITCH_AFTER,"direct","prefix",horizon=100,
                                        environment="local-machine/a")[1],
                         "insufficient_transition_evidence")

    def test_unverified_transition_rejected(self):
        with self.assertRaises(ValueError):
            self.new().observe_switch(TRAIN,SWITCH_AFTER,"direct","blocks",process_id="0",
                                      setup_ns=1,tick_ns=2,verified=False,
                                      environment="local-machine/a")

    def test_no_negative_horizon_or_nan(self):
        m=self.new()
        with self.assertRaises(ValueError):
            m.advise_switch(TRAIN,SWITCH_AFTER,"direct","blocks",horizon=0,
                            environment="local-machine/a")
        with self.assertRaises(ValueError):
            m.advise_switch(TRAIN,SWITCH_AFTER,"direct","blocks",horizon=1,
                            environment="local-machine/a",margin=float("nan"))

    def test_heldout_baseline_not_mistaken_for_training(self):
        m=self.new();self.fit(m,w=PREFIX_TRAIN,winner="blocks")
        self.assertEqual(simple_rule(PREFIX_HELD),"prefix")
        result=m.compile([PREFIX_HELD],environment="local-machine/a")
        self.assertEqual(result[workload_key(PREFIX_HELD)].route,"blocks")

    def test_correctness_of_four_real_workflows(self):
        self.assertIsInstance(checked_routes(RangeWorkload(33,2,3,2,"short")),int)


if __name__=="__main__": unittest.main()
