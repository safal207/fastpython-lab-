"""Correctness and paid-economics gates for frozen long-horizon followup."""
import hashlib
import json
import unittest

from alexpy.stream_horizon import frozen_evidence, summarize, worker
from alexpy.stream_bench import host_metadata
from alexpy.stream_economics import stream_schedule
from alexpy.stable_memory import workload_key, simple_rule


class FrozenHorizonTests(unittest.TestCase):
    def sample(self):
        choices = {workload_key(w): simple_rule(w) for w in stream_schedule(16)}
        return {"schema": "ALEXPY-STREAM-ECONOMICS-SUMMARY/1",
                "host": host_metadata(), "route_choices": choices,
                "frozen_sha256": hashlib.sha256(json.dumps(choices, sort_keys=True).encode()).hexdigest(),
                "calibration_wall_ns": 1000}

    def test_frozen_hash_and_cpu_gate(self):
        d = self.sample()
        self.assertEqual(frozen_evidence(d, host_metadata())[1], 1000)
        first = next(iter(d["route_choices"]))
        d["route_choices"][first] = "prefix" if d["route_choices"][first] != "prefix" else "direct"
        with self.assertRaises(ValueError):
            frozen_evidence(d, host_metadata())
        d = self.sample()
        host = {**host_metadata(), "processor": "different CPU"}
        with self.assertRaisesRegex(ValueError, "host mismatch"):
            frozen_evidence(d, host)

    def test_actual_stream_is_mirrored(self):
        d = self.sample()
        r = worker(d["route_choices"], cycles=(1, 2), sizes=(16,), repeats=2, session=0)
        self.assertEqual([x["correctness"] for x in r["rows"]], ["PASS", "PASS"])
        self.assertEqual(len(r["rows"][0]["raw_ns"]["stable"]), 2)

    def test_configurable_ci_schedule_and_missing_route_gate(self):
        schedule = stream_schedule(16, changes=(1,), queries=(1, 16), spans=("short", "wide"), ticks=2)
        routes = {workload_key(w): simple_rule(w) for w in schedule}
        report = worker(routes, cycles=(1,), sizes=(16,), repeats=2, session=2,
                        changes=(1,), queries=(1, 16), spans=("short", "wide"), ticks=2)
        self.assertEqual(report["rows"][0]["correctness"], "PASS")
        routes.pop(next(iter(routes)))
        with self.assertRaisesRegex(ValueError, "frozen routes do not cover"):
            worker(routes, cycles=(1,), sizes=(16,), repeats=2, session=2,
                   changes=(1,), queries=(1, 16), spans=("short", "wide"), ticks=2)

    def test_requires_real_processes_and_strong_net(self):
        fake = lambda pid,s: {"pid":pid,"rows":[{"size":16,"cycles":1,
                  "correctness":"PASS", "median_ns":{"simple":2000,"stable":2000-s}}]}
        with self.assertRaises(ValueError):
            summarize([fake(1,1500)]*3,1000,cycles=(1,),sizes=(16,))
        r = summarize([fake(1,1500),fake(2,1600),fake(3,1400)],1000,cycles=(1,),sizes=(16,))
        self.assertEqual(r["first_observed_paid_cycles"],1)
        r2 = summarize([fake(1,1500),fake(2,1600),fake(3,300)],1000,cycles=(1,),sizes=(16,))
        self.assertIsNone(r2["first_observed_paid_cycles"])


if __name__ == "__main__":
    unittest.main()
