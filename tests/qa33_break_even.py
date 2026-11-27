"""QA33: bounded, frozen break-even experiment. Not a production dispatcher."""
import argparse
import gc
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
from types import MappingProxyType

from alexpy.qa_lab import QAEvent, PASS, FAIL, SKIP
from test_qa29_benchmark import ROUTES, validate, stats

# Known, declared session plan, never the observed outcome of future events.
PLANS = tuple((n, 32, k, cadence)
              for n in (128, 4096) for k in (1, 8, 32)
              for cadence in (1, 8))
BASE = "c_tree"
ALT = "bitset"
ROUNDS = 200
SESSIONS = 3

def fixture(plan, seed):
    n, batch, k, cadence = plan
    rng = random.Random(seed)
    layout = tuple(i % 3 for i in range(n))
    groups = []
    for _ in range(k):
        group = []
        for _ in range(batch):
            i = rng.randrange(n)
            st = rng.choices((PASS, FAIL, SKIP), (7, 2, 1))[0]
            group.append(QAEvent(i, st, 0 if st == SKIP else rng.randrange(1, 500)))
        groups.append(tuple(group))
    return layout, tuple(groups)

def replay_reports(cls, layout, groups, cadence):
    eng = cls(layout, 3)
    result = []
    for i, group in enumerate(groups, 1):
        eng.apply_batch(group)
        if i % cadence == 0 or i == len(groups):
            result.append((eng.snapshot(), eng.witnesses()))
    return tuple(result)

def choose(table, plan):
    # Table scoped to this parent runner invocation, passed read-only in workers.
    return table.get(tuple(plan), BASE)

def select(rows, overhead):
    """Fixed conservative gate; train only. No interpolation or transfer."""
    if len(rows) < SESSIONS:
        return BASE
    for row in rows:
        a, b = row[ALT], row[BASE]
        if not (a["median_ns"] + overhead <= .95 * b["median_ns"]
                and a["p95_ns"] + overhead <= b["p95_ns"]
                and a["p99_ns"] + overhead <= b["p99_ns"]
                and a["peak_bytes"] <= b["peak_bytes"]):
            return BASE
    return ALT

def measure(plan, seed, rounds, session, phase, decision=BASE, frozen=None):
    layout, groups = fixture(plan, seed)
    validate(layout, groups)  # independent full scan and transitions at each batch
    expected = replay_reports(ROUTES["python"], layout, groups, plan[3])
    table = MappingProxyType(dict(frozen) if frozen is not None else
                             {p: decision if p == tuple(plan) else BASE for p in PLANS})
    if choose(table, plan) != decision:
        raise AssertionError("frozen plan mismatch")
    funcs = {k: (lambda cls=ROUTES[k]: replay_reports(cls, layout, groups, plan[3]))
             for k in (BASE, ALT)}
    if phase == "holdout":
        funcs["adaptive"] = lambda: replay_reports(
            ROUTES[choose(table, plan)], layout, groups, plan[3])
    keys = tuple(funcs)
    for f in funcs.values():
        for _ in range(8):
            if f() != expected:
                raise AssertionError("warmup mismatch")
    raw = {k: [] for k in keys}
    orders = tuple(itertools.permutations(keys))
    order_codes = []
    rng = random.Random(seed + 1009 * session)
    gc.collect()
    for i in range(rounds):
        if i % len(orders) == 0:
            block = list(range(len(orders)))
            rng.shuffle(block)
        code = block[i % len(orders)]
        order_codes.append(code)
        for k in orders[code]:
            t = time.perf_counter_ns()
            value = funcs[k]()
            raw[k].append(time.perf_counter_ns() - t)
            if value != expected:
                raise AssertionError("timed report mismatch")
    output = {}
    for k, f in funcs.items():
        peaks = []
        for _ in range(3):
            gc.collect()
            tracemalloc.start()
            try:
                value = f()
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            if value != expected:
                raise AssertionError("memory report mismatch")
        output[k] = dict(stats(raw[k]), samples_ns=raw[k],
                         peak_bytes=statistics.median(peaks), peak_samples=peaks)
    lookup = []
    for _ in range(2000):
        t = time.perf_counter_ns()
        value = choose(table, plan)
        lookup.append(time.perf_counter_ns() - t)
        if value != decision:
            raise AssertionError("wrong route")
    return dict(plan=list(plan), phase=phase, seed=seed, session=session, pid=os.getpid(),
                results=output, order_codes=order_codes,
                order_names=[list(o) for o in orders],
                lookup_ns=lookup, lookup_p99_ns=stats(lookup)["p99_ns"],
                chosen=decision, reports=len(expected), verified=True)

def worker(phase, session, rounds, decisions):
    rows = []
    frozen = dict(zip(PLANS, decisions)) if phase == "holdout" else {p: BASE for p in PLANS}
    if set(frozen) != set(PLANS):
        raise ValueError("incomplete frozen table")
    for i, plan in enumerate(PLANS):
        # Distinct fixed training and holdout seeds in each independent process.
        seed = (330100 if phase == "train" else 339100) + 101 * session + i
        rows.append(measure(plan, seed, rounds, session, phase,
                            decisions[i] if phase == "holdout" else BASE, frozen))
    return dict(pid=os.getpid(), rows=rows)

def child(phase, session, rounds, decisions):
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                          "--worker", phase, str(session), str(rounds),
                          json.dumps(decisions)], capture_output=True, text=True, timeout=600)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-12000:])
    return json.loads(proc.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--output", default="qa33-results.json")
    parser.add_argument("--worker", nargs=4)
    args = parser.parse_args()
    if args.worker:
        phase, session, rounds, decisions = args.worker
        print(json.dumps(worker(phase, int(session), int(rounds), json.loads(decisions))))
        return
    if args.rounds < 1 or args.sessions < 1 or args.sessions > 20:
        parser.error("positive rounds and 1..20 sessions required")
    start = time.perf_counter_ns()
    train = [child("train", i, args.rounds, []) for i in range(args.sessions)]
    overhead = max(r["lookup_p99_ns"] for s in train for r in s["rows"])
    full = args.rounds >= ROUNDS and args.sessions >= SESSIONS
    decisions = [select([s["rows"][i]["results"] for s in train], overhead)
                 if full else BASE for i in range(len(PLANS))]
    calibration_ns = time.perf_counter_ns() - start
    holdout = [child("holdout", i, args.rounds, decisions) for i in range(args.sessions)]
    summary = []
    for i, plan in enumerate(PLANS):
        rows = [s["rows"][i]["results"] for s in holdout]
        ratios = {k: [r["adaptive"][k]/r[BASE][k] for r in rows]
                  for k in ("median_ns", "p95_ns", "p99_ns", "peak_bytes")}
        alt_ratios = {k: [r[ALT][k]/r[BASE][k] for r in rows]
                      for k in ("median_ns", "p95_ns", "p99_ns", "peak_bytes")}
        savings = [r[BASE]["median_ns"]-r["adaptive"]["median_ns"] for r in rows]
        passed = full and decisions[i] == ALT and all(
            r["adaptive"]["median_ns"] <= .95*r[BASE]["median_ns"]
            and r["adaptive"]["p95_ns"] <= r[BASE]["p95_ns"]
            and r["adaptive"]["p99_ns"] <= r[BASE]["p99_ns"]
            and r["adaptive"]["peak_bytes"] <= r[BASE]["peak_bytes"] for r in rows)
        # Entire calibration wall cost, conservatively charged to this one
        # repeated plan. Descriptive only, assumes stationary future savings.
        payback = None
        if passed and min(savings) > 0:
            import math
            payback = math.ceil(calibration_ns/min(savings))
        summary.append(dict(plan=list(plan), route=decisions[i],
                            adaptive_over_tree=ratios, bitset_over_tree=alt_ratios,
                            holdout_gate_pass=passed, median_savings_ns=savings,
                            conservative_calibration_payback_sessions=payback))
    report = dict(schema="alexpy-qa33/1", github_run_id=os.getenv("GITHUB_RUN_ID"),
                  github_sha=os.getenv("GITHUB_SHA"), platform=platform.platform(),
                  machine=platform.machine(), python=sys.version, rounds=args.rounds,
                  sessions=args.sessions, full_protocol=full, default_promotion=False,
                  plans=[list(x) for x in PLANS], lookup_overhead_ns=overhead,
                  calibration_wall_ns=calibration_ns, train=train, holdout=holdout,
                  summary=summary, memory_scope="tracemalloc only; not full RSS",
                  timing_scope="fresh engine + declared batch horizon + reports + lookup for adaptive")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(selected=decisions, gate_pass=sum(r["holdout_gate_pass"] for r in summary),
                          calibration_ns=calibration_ns, summary=summary)))

if __name__ == "__main__":
    main()
