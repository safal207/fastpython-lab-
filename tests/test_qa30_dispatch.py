"""Independent-process p99 rechecks and frozen QA dispatcher holdouts.

Training and holdout use separately seeded synthetic workloads, not new real
CI observations. Historical regression rechecks reuse the exact 2.9 replay.
All timings include engine construction, reports and (adaptive only) lookup.
No live state migration: route changes occur only at fresh session boundaries.
"""
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
import unittest
from pathlib import Path

from alexpy.qa_lab import QAEvent, PASS, FAIL, SKIP
from alexpy.qa_route_matrix import RouteMatrix, Cell, features
from test_qa29_benchmark import ROUTES, workload, validate, replay, stats

TRAIN = ((19, 1, 3101), (23, 8, 3102), (384, 64, 3103))
HOLDOUT = ((29, 1, 8101), (31, 13, 8102), (769, 96, 8103),
           (81, 1, 8104))


def synthetic(n, batch, seed):
    rng = random.Random(seed)
    layout = tuple(i % 3 for i in range(n))
    events = []
    for phase in range(3):
        for i in rng.sample(range(n), n):
            st = rng.choices((PASS, FAIL, SKIP), (7, 2+phase, 1))[0]
            events.append(QAEvent(i, st, 0 if st == SKIP else rng.randrange(1, 500)))
            if rng.randrange(9) == 0:
                events.append(QAEvent(i, PASS, rng.randrange(1, 500)))
    return layout, tuple(tuple(events[i:i+batch]) for i in range(0, len(events), batch))


def measure(layout, groups, rounds, matrix=None, batch=None, env=None):
    validate(layout, groups)
    expected = replay(ROUTES["python"], layout, groups)
    keys = list(ROUTES)
    def adaptive():
        decision = matrix.choose(len(layout), batch, environment=env)
        return replay(ROUTES[decision.route], layout, groups)
    funcs = {k: (lambda cls=cls: replay(cls, layout, groups)) for k, cls in ROUTES.items()}
    if matrix is not None:
        keys.append("adaptive")
        funcs["adaptive"] = adaptive
    for f in funcs.values():
        for _ in range(8):
            if f() != expected:
                raise AssertionError("warmup mismatch")
    samples = {k: [] for k in keys}
    gc.collect()
    for i in range(rounds):
        offset = i % len(keys)
        for k in keys[offset:] + keys[:offset]:
            start = time.perf_counter_ns()
            value = funcs[k]()
            samples[k].append(time.perf_counter_ns() - start)
            if value != expected:
                raise AssertionError("timed mismatch")
    result = {}
    for k in keys:
        peaks = []
        for _ in range(3):
            gc.collect()
            tracemalloc.start()
            try:
                value = funcs[k]()
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            if value != expected:
                raise AssertionError("memory mismatch")
        result[k] = dict(stats(samples[k]), peak_bytes=sorted(peaks)[1],
                         samples_ns=samples[k], peak_samples=peaks)
    return result


def worker():
    return {"pid": os.getpid(), "regression": [
        {"batch": b, "results": measure(*workload(1, b), 1000)}
        for b in (1, 8)], "train": [
        {"n": n, "batch": b, "seed": seed,
         "results": measure(*synthetic(n, b, seed), 200)}
        for n, b, seed in TRAIN]}


def cells(results):
    return {k: Cell(v["median_ns"], v["p99_ns"], v["peak_bytes"])
            for k, v in results.items()}


def overhead(env):
    # Estimate full feature extraction + lookup, not just a dict get.
    m = RouteMatrix(env)
    row = {k: Cell(100, 200, 100) for k in ROUTES}
    for i in range(3):
        m.observe(features(19, 1), str(i), row, environment=env, verified=True)
    m.freeze(overhead_ns=0)
    raw = []
    for _ in range(3000):
        t = time.perf_counter_ns()
        m.choose(19, 1, environment=env)
        raw.append(time.perf_counter_ns() - t)
    return stats(raw)["p99_ns"]


class QA30Tests(unittest.TestCase):
    def test_matrix_guards(self):
        env = "test-runner"
        m = RouteMatrix(env)
        base = Cell(1000, 1500, 1000)
        good = Cell(800, 1200, 800)
        bad = Cell(600, 1800, 500)
        row = dict(c_tree=base, bitset=good, python=bad, native_set=base)
        key = features(19, 8)
        with self.assertRaises(RuntimeError):
            m.choose(19, 8, environment=env)
        with self.assertRaises(ValueError):
            m.observe(key, "0", row, environment="other", verified=True)
        with self.assertRaises(ValueError):
            m.observe(key, "0", row, environment=env, verified=False)
        for i in range(3):
            m.observe(key, str(i), row, environment=env, verified=True)
        with self.assertRaises(ValueError):
            m.observe(key, "0", row, environment=env, verified=True)
        m.freeze(overhead_ns=50)
        self.assertEqual(m.choose(29, 13, environment=env).route, "bitset")
        self.assertEqual(m.choose(29, 13, environment="other").route, "c_tree")
        self.assertEqual(m.choose(90, 1, environment=env).route, "c_tree")
        with self.assertRaises(RuntimeError):
            m.observe(key, "4", row, environment=env, verified=True)
        slow = RouteMatrix(env)
        for i in range(3):
            slow.observe(key, str(i), row, environment=env, verified=True)
        slow.freeze(overhead_ns=300)
        self.assertEqual(slow.choose(29, 13, environment=env).route, "c_tree")
        sparse = RouteMatrix(env)
        sparse.observe(key, "1", row, environment=env, verified=True)
        sparse.freeze(overhead_ns=0)
        self.assertEqual(sparse.choose(29, 13, environment=env).reason,
                         "insufficient_processes")

    def test_independent_sessions_and_holdouts(self):
        env = "|".join((platform.platform(), platform.python_version(),
                        os.environ.get("GITHUB_RUN_ID", "local"),
                        os.environ.get("GITHUB_JOB", "local"), str(os.getpid())))
        runs = []
        for i in range(3):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                  capture_output=True, text=True, check=True, timeout=120)
            runs.append(json.loads(proc.stdout))
        matrix = RouteMatrix(env)
        cost = overhead(env)
        for i, run in enumerate(runs):
            for row in run["train"]:
                matrix.observe(features(row["n"], row["batch"]), str(i),
                               cells(row["results"]), environment=env, verified=True)
        decisions = matrix.freeze(overhead_ns=cost)
        holdouts = []
        for n, b, seed in HOLDOUT:
            d = matrix.choose(n, b, environment=env)
            result = measure(*synthetic(n, b, seed), 300, matrix, b, env)
            holdouts.append(dict(n=n, batch=b, seed=seed, route=d.route, reason=d.reason,
                                 results=result))
        regression = []
        for b in (1, 8):
            values = []
            for run in runs:
                row = next(x for x in run["regression"] if x["batch"] == b)["results"]
                values.append(row["bitset"]["p99_ns"]/row["c_tree"]["p99_ns"])
            regression.append(dict(batch=b, ratios=values, exceed_5pct=sum(v>1.05 for v in values),
                                   repeatable_candidate=all(v>1.05 for v in values)))
        report = dict(schema="alexpy-qa30/1", environment=env, overhead_p99_ns=cost,
                      sessions=runs, regression=regression, holdouts=holdouts,
                      decisions=[dict(key=list(k), route=v.route, reason=v.reason)
                                 for k,v in decisions.items()],
                      experimental=True, default_promotion=False,
                      switch_scope="fresh engine only; no live state migration")
        Path("qa30-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Keep full per-process timing/memory samples in artifact, concise
        # evidence in logs for inspection without artifact extraction.
        report.pop("sessions")
        for row in holdouts:
            for v in row["results"].values():
                v.pop("samples_ns")
                v.pop("peak_samples")
            a, c = row["results"]["adaptive"], row["results"]["c_tree"]
            row["adaptive_over_tree"] = {k: a[k]/c[k] for k in ("median_ns", "p99_ns")}
        print("ALEXPY_QA30 " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(), separators=(",", ":")))
    else:
        unittest.main()
