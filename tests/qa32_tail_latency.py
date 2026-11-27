"""QA32 preregistered advisory tail-latency study; no route promotion."""
import argparse
import gc
import hashlib
import itertools
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from alexpy.qa_route_matrix import RouteMatrix, Cell, features
from test_qa29_benchmark import ROUTES, validate, replay, stats
from test_qa30_dispatch import synthetic

CASES = (
    (29, 1, 8101, "historical"),
    (29, 1, 320031, "fresh"),
    (31, 1, 320033, "fresh"),
    (29, 8, 320039, "fresh"),
)
NAMES = ("direct", "dispatch", "bitset")
ORDERS = tuple(itertools.permutations(NAMES))
ROUNDS = 2000
SESSIONS = 5

def frozen_baseline(n, batch, env):
    # Equal synthetic cells FORCE the already-observed fallback decision.
    # This is not a trained or promoted performance model.
    matrix = RouteMatrix(env)
    row = {k: Cell(100, 200, 100) for k in ROUTES}
    for i in range(3):
        matrix.observe(features(n, batch), str(i), row,
                       environment=env, verified=True)
    matrix.freeze(overhead_ns=0)
    d = matrix.choose(n, batch, environment=env)
    if d.route != "c_tree" or d.reason != "risk_or_gain_gate":
        raise AssertionError("baseline selection changed")
    return matrix

def setup(case):
    n, b, seed, label = case
    layout, groups = synthetic(n, b, seed)
    env = "qa32-isolation"
    matrix = frozen_baseline(n, b, env)
    def direct():
        return replay(ROUTES["c_tree"], layout, groups)
    def dispatch():
        route = matrix.choose(n, b, environment=env).route
        return replay(ROUTES[route], layout, groups)
    def bitset():
        return replay(ROUTES["bitset"], layout, groups)
    return layout, groups, matrix, env, dict(direct=direct, dispatch=dispatch, bitset=bitset)

def rss_peak():
    # Process-lifetime maximum RSS, not heap-only or incremental allocation.
    if sys.platform not in ("darwin", "linux"):
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == "darwin" else 1024))

def worker(index, rounds, session):
    case = CASES[index]
    layout, groups, matrix, env, funcs = setup(case)
    validate(layout, groups)
    expected = replay(ROUTES["python"], layout, groups)
    for f in funcs.values():
        for _ in range(20):
            if f() != expected:
                raise AssertionError("warmup mismatch")
    raw = {k: [] for k in NAMES}
    order_sequence = []
    rng = random.Random(320100 + session * 17 + index)
    # All six orders once per block; shuffle using a frozen independent seed.
    for i in range(rounds):
        if i % len(ORDERS) == 0:
            block = list(ORDERS)
            rng.shuffle(block)
        order = block[i % len(ORDERS)]
        order_sequence.append(ORDERS.index(order))
        for k in order:
            start = time.perf_counter_ns()
            result = funcs[k]()
            raw[k].append(time.perf_counter_ns() - start)
            if result != expected:
                raise AssertionError("timed report mismatch")
    overhead = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        decision = matrix.choose(case[0], case[1], environment=env)
        overhead.append(time.perf_counter_ns() - start)
        if decision.route != "c_tree":
            raise AssertionError("unexpected route")
    metrics = {k: stats(v) for k, v in raw.items()}
    ratios = {k: {m: metrics[k][m] / metrics["direct"][m]
                  for m in ("median_ns", "p95_ns", "p99_ns")}
              for k in ("dispatch", "bitset")}
    paired = {k: stats([max(0, v - a) for v, a in zip(raw[k], raw["direct"])])
              for k in ("dispatch", "bitset")}
    return dict(case=list(case), pid=os.getpid(), session=session, rounds=rounds,
                metrics=metrics, ratios=ratios, raw_ns=raw,
                paired_positive_excess_ns=paired, order_codes=order_sequence,
                lookup_ns=overhead, lookup_metrics=stats(overhead),
                fingerprint=expected)

def memory_worker(index, route):
    # No other engine replay in this worker, to avoid contaminating peak RSS.
    # Parent timing workers independently validate the same deterministic stream.
    layout, groups, matrix, env, funcs = setup(CASES[index])
    gc.collect()
    before = rss_peak()
    tracemalloc.start()
    try:
        fingerprint = funcs[route]()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    after = rss_peak()
    return dict(pid=os.getpid(), route=route, fingerprint=fingerprint,
                peak_traced_bytes=peak, rss_lifetime_before_bytes=before,
                rss_lifetime_after_bytes=after)

def child(*args):
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *map(str,args)],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-12000:])
    return json.loads(proc.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--output", default="qa32-results.json")
    parser.add_argument("--worker", nargs=3, type=int)
    parser.add_argument("--memory", nargs=2)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return
    if args.memory:
        print(json.dumps(memory_worker(int(args.memory[0]), args.memory[1])))
        return
    if min(args.rounds, args.sessions) < 1:
        parser.error("positive rounds and sessions required")
    rows = []
    for i, case in enumerate(CASES):
        sessions = [child("--worker", i, args.rounds, s) for s in range(args.sessions)]
        if len({s["fingerprint"] for s in sessions}) != 1:
            raise AssertionError("session replay mismatch")
        memory = [child("--memory", i, k) for k in NAMES for _ in range(3)]
        if any(m["fingerprint"] != sessions[0]["fingerprint"] for m in memory):
            raise AssertionError("memory replay mismatch")
        tail = [s["ratios"]["dispatch"]["p99_ns"] for s in sessions]
        rows.append(dict(case=list(case), sessions=sessions, memory=memory,
                         median_dispatch_p99_ratio=statistics.median(tail),
                         repeatable_candidate=(args.sessions >= SESSIONS and
                         args.rounds >= ROUNDS and all(x > 1.05 for x in tail))))
    report = dict(schema="alexpy-qa32/1", python=sys.version, platform=platform.platform(),
                  machine=platform.machine(), github_run_id=os.getenv("GITHUB_RUN_ID"),
                  github_sha=os.getenv("GITHUB_SHA"), rounds=args.rounds, sessions=args.sessions,
                  order_names=[list(x) for x in ORDERS], default_promotion=False,
                  full_protocol=args.rounds >= ROUNDS and args.sessions >= SESSIONS,
                  timing_scope="fresh engine, replay, per-batch report; dispatch includes choose",
                  memory_scope="separate process lifetime peak RSS on Linux/macOS; traced Python allocations on all",
                  rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([dict(case=r["case"], p99=r["median_dispatch_p99_ratio"],
                           repeatable=r["repeatable_candidate"]) for r in rows]))

if __name__ == "__main__":
    main()
