"""AlexPy 2.9: recorded CI failures, independent oracle and advisory tail latency.

Source: Roby's PR 527, historical HEAD b73bcbb343f690a16950a6d16dab28de459fbfbf
and later HEAD 5d4c5c85d4b64f87fc2a8173816cf06e88556bee.
GitHub job conclusions, not individual browser-test logs. Connector omitted
job times, so all durations are zero (unknown, not measured). Within each
revision jobs are ordered by name, not claimed chronological.
Scaled replays are synthetic replicas, never additional observed failures.
No route is promoted by this experiment.
"""
import gc
import json
import math
import platform
import statistics
import time
import tracemalloc
import unittest
from heapq import nsmallest

from alexpy.qa_batch import StrongPythonBatchQA, _prepare
from alexpy.qa_lab import QAEvent, NOT_RUN, PASS, FAIL, SKIP, scan_reference
from alexpy.native_transition_tree import CTreeNativeQA
from alexpy.native_bitset import BitsetNativeQA
from alexpy.transition_tree import SetNativeQA

SOURCE = (
    ("lighthouse", 45062578135, 45091595153, (
        ("build-smoke / build-smoke", PASS, PASS),
        ("profiles / profile (mobile)", FAIL, PASS),
        ("profiles / profile (desktop)", FAIL, PASS),
        ("comment", PASS, PASS),
        ("fork-notice", SKIP, SKIP),
    )),
    ("dessert", 45062578160, 45091595210, (("contract", FAIL, PASS),)),
    ("webkit", 45062578180, 45091594974, (("ios-route-webkit", FAIL, PASS),)),
)


class PythonQA(StrongPythonBatchQA):
    """Pure-Python incremental counters plus set and exact top-K witnesses."""
    def __init__(self, suites, count):
        super().__init__(suites, count)
        self.failing = set()

    def apply_batch(self, events):
        latest = _prepare(events, len(self.suites))
        transition = super().apply_batch(events)
        for idx, (status, _) in latest.items():
            if status == FAIL:
                self.failing.add(idx)
            else:
                self.failing.discard(idx)
        return transition

    def witnesses(self):
        return tuple(nsmallest(4, self.failing))

    def verify(self):
        return (self.snapshot() == scan_reference(
            self.suites, len(self._counts), self._status, self._durations)
            and self.failing == {i for i, s in enumerate(self._status) if s == FAIL})


ROUTES = {"python": PythonQA, "c_tree": CTreeNativeQA,
          "bitset": BitsetNativeQA, "native_set": SetNativeQA}


def workload(copies, batch):
    layout, pairs = [], []
    for _ in range(copies):
        for suite, (_, _, _, rows) in enumerate(SOURCE):
            for _, first, last in sorted(rows):
                layout.append(suite)
                pairs.append((first, last))
    groups = []
    for phase in (0, 1):
        events = [QAEvent(i, pair[phase], 0) for i, pair in enumerate(pairs)]
        groups.extend(tuple(events[i:i+batch]) for i in range(0, len(events), batch))
    return tuple(layout), tuple(groups)


def validate(layout, groups):
    engines = {k: c(layout, 3) for k, c in ROUTES.items()}
    statuses = [NOT_RUN] * len(layout)
    durations = [0] * len(layout)
    if not all(engines[k].native for k in ("c_tree", "bitset", "native_set")):
        raise AssertionError("native route unavailable")
    for group in groups:
        for e in group:
            statuses[e.test_id] = e.status
        oracle = scan_reference(layout, 3, statuses, durations)
        witnesses = tuple(i for i, s in enumerate(statuses) if s == FAIL)[:4]
        transitions = []
        for key, eng in engines.items():
            transitions.append(eng.apply_batch(group))
            if eng.snapshot() != oracle or eng.witnesses() != witnesses or not eng.verify():
                raise AssertionError("independent oracle mismatch: " + key)
        if any(t != transitions[0] for t in transitions):
            raise AssertionError("batch transition mismatch")


def replay(cls, layout, groups):
    engine = cls(layout, 3)
    # Same observable report at every batch boundary; consume the reports.
    fingerprint = 0
    for group in groups:
        engine.apply_batch(group)
        snap = engine.snapshot()
        witnesses = engine.witnesses()
        fingerprint += snap.total[FAIL] + len(witnesses)
    return fingerprint


def stats(samples):
    ordered = sorted(samples)
    def q(p):
        return ordered[math.ceil(p * len(ordered)) - 1]
    return {"n": len(samples), "mean_ns": statistics.mean(samples),
            "median_ns": statistics.median(samples), "p95_ns": q(.95),
            "p99_ns": q(.99)}


def measure(copies, batch, rounds):
    layout, groups = workload(copies, batch)
    validate(layout, groups)
    routes = tuple(ROUTES)
    expected = replay(PythonQA, layout, groups)
    samples = {k: [] for k in routes}
    for cls in ROUTES.values():
        for _ in range(10):
            if replay(cls, layout, groups) != expected:
                raise AssertionError("warm-up mismatch")
    gc.collect()
    for i in range(rounds):
        order = routes[i % len(routes):] + routes[:i % len(routes)]
        for key in order:
            start = time.perf_counter_ns()
            result = replay(ROUTES[key], layout, groups)
            elapsed = time.perf_counter_ns() - start
            if result != expected:
                raise AssertionError("timed result mismatch")
            samples[key].append(elapsed)
    results = {}
    for key, cls in ROUTES.items():
        peaks = []
        for _ in range(5):
            gc.collect()
            tracemalloc.start()
            try:
                result = replay(cls, layout, groups)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            if result != expected:
                raise AssertionError("memory result mismatch")
        results[key] = {**stats(samples[key]), "peak_traced_bytes_median": statistics.median(peaks),
                        "peak_traced_bytes_samples": peaks, "samples_ns": samples[key]}
    tree, bits = results["c_tree"], results["bitset"]
    ratios = {k: bits[k]/tree[k] for k in ("median_ns", "p95_ns", "p99_ns")}
    return {"copies": copies, "synthetic_scale": copies != 1, "tests": len(layout),
            "events": len(layout)*2, "batch": batch, "rounds": rounds,
            "bitset_over_tree_latency": ratios,
            "bitset_regression_candidates_over_5pct": [k for k,v in ratios.items() if v>1.05],
            "results": results}


class QA29Benchmark(unittest.TestCase):
    def test_recorded_failures_and_scaled_replay(self):
        layout, groups = workload(1, 64)
        self.assertEqual(len(layout), 7)
        self.assertEqual(sum(e.status == FAIL for e in groups[0]), 4)
        self.assertEqual(sum(e.status == FAIL for e in groups[1]), 0)
        # Fixed protocol; quantiles describe replay latency (not browser or
        # individual event latency). Shared runners are noisy. No timing gate.
        rows = [measure(1, 1, 1000), measure(1, 8, 1000), measure(64, 64, 400)]
        report = {"schema": "alexpy-qa29/1", "python": platform.python_version(),
                  "system": platform.platform(), "machine": platform.machine(),
                  "source_runs": [[a,b] for _,a,b,_ in SOURCE],
                  "timing_scope": "fresh engine + replay + report each batch",
                  "memory_scope": "tracemalloc peak, not full RSS or untracked malloc",
                  "advisory": True, "default_promotion": False, "rows": rows}
        # Compact metrics in logs, full raw samples in a runner-local JSON file.
        from pathlib import Path
        Path("qa29-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for row in rows:
            for result in row["results"].values():
                del result["samples_ns"]
        print("ALEXPY_QA29 " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    unittest.main()
