"""Frozen density/shift stress protocol for the opt-in Bardo native witness index.

Both routes share the SAME compiled QA accumulator and the SAME event batches.
Measurements include state setup, event validation, updates, snapshots and
witness reports; full independent scans are outside timing. Never auto-promote.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from .native_transition_tree import CTreeNativeQA
from .transition_tree import SetNativeQA
from .qa_lab import QAEvent, NOT_RUN, PASS, FAIL, SKIP, scan_reference
from .bardo_bridge_bench import digest_step, host

ROUTES = {"c_tree": CTreeNativeQA, "set": SetNativeQA}
SEED = 312_080_191


@dataclass(frozen=True)
class StressCase:
    name: str
    n: int
    suites: int
    events: int
    batch: int
    active: int
    fail_percent: int
    report_every: int
    reads_per_report: int
    pattern: str


CASES = (
    StressCase("dense-mostly-fail", 4096, 32, 8192, 512, 4096, 90, 4, 1, "random"),
    StressCase("dense-wide", 16384, 128, 12288, 2048, 16384, 85, 4, 1, "random"),
    StressCase("reversible-flips", 4096, 32, 8192, 512, 4096, 50, 2, 1, "flip"),
    StressCase("hot-retries", 4096, 32, 8192, 64, 64, 55, 4, 1, "random"),
    StressCase("frequent-diagnostics", 4096, 32, 6000, 64, 4096, 55, 1, 8, "random"),
    StressCase("zero-failures", 4096, 32, 6000, 64, 4096, 0, 4, 1, "random"),
    StressCase("all-failures", 4096, 32, 8192, 512, 4096, 100, 4, 1, "random"),
)
SMOKE_CASES = (StressCase("smoke-dense", 128, 4, 256, 32, 128, 90, 2, 2, "random"),)


def generate(case: StressCase, seed: int = SEED):
    if not (0 <= case.fail_percent <= 100 and 0 < case.active <= case.n and
            case.n > 0 and case.batch > 0 and case.suites > 0 and case.report_every > 0):
        raise ValueError("invalid stress scenario")
    rng = random.Random(seed + case.n * 11 + case.events * 7)
    suites = [(i * 17 + i // 13) % case.suites for i in range(case.n)]
    events = []
    for i in range(case.events):
        idx = i % case.active if case.pattern == "flip" else rng.randrange(case.active)
        if case.pattern == "flip":
            # A coherent state flip AFTER one full pass, not random noise.
            status = FAIL if (i // case.active) % 2 else PASS
        else:
            draw = rng.randrange(100)
            status = FAIL if draw < case.fail_percent else (SKIP if draw >= 98 else PASS)
        events.append(QAEvent(idx, status, 0 if status == SKIP else rng.randrange(0, 100001)))
    return suites, events


def replay(route: str, case: StressCase, suites, events):
    if route not in ROUTES:
        raise ValueError("unknown route")
    engine = ROUTES[route](suites, case.suites)
    if not engine.native:
        raise RuntimeError("native QA extension is required")
    digest = 0
    nb = (len(events) + case.batch - 1) // case.batch
    for bi, i in enumerate(range(0, len(events), case.batch), 1):
        engine.apply_batch(events[i:i + case.batch])
        if bi % case.report_every == 0 or bi == nb:
            for _ in range(case.reads_per_report):
                digest = digest_step(digest, (engine.snapshot(), engine.witnesses()))
    return digest


def verify(case: StressCase, suites, events):
    engines = [ROUTES[name](suites, case.suites) for name in ROUTES]
    statuses = [NOT_RUN] * case.n
    durations = [0] * case.n
    nb = (len(events) + case.batch - 1) // case.batch
    for bi, i in enumerate(range(0, len(events), case.batch), 1):
        batch = events[i:i + case.batch]
        for event in batch:
            statuses[event.test_id], durations[event.test_id] = event.status, event.duration_ms
        transitions = [engine.apply_batch(batch) for engine in engines]
        if transitions[0] != transitions[1]:
            raise AssertionError("transition mismatch")
        if bi % max(1, nb // 8) == 0 or bi == nb:
            expected = scan_reference(suites, case.suites, statuses, durations)
            witnesses = tuple(i for i, st in enumerate(statuses) if st == FAIL)[:4]
            for engine in engines:
                if engine.snapshot() != expected or engine.witnesses() != witnesses:
                    raise AssertionError(f"incorrect QA/witness state at batch {bi}")
    if not all(engine.verify() for engine in engines):
        raise AssertionError("internal C/Python witness invariant failed")
    digests = [replay(name, case, suites, events) for name in ROUTES]
    if len(set(digests)) != 1:
        raise AssertionError("end-to-end digest mismatch")
    return digests[0]


def measure(case: StressCase, pairs: int = 3, seed: int = SEED):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    suites, events = generate(case, seed)
    expected = verify(case, suites, events)
    samples = {name: [] for name in ROUTES}
    for i in range(pairs):
        order = tuple(ROUTES) if i % 2 == 0 else tuple(reversed(ROUTES))
        for name in order:
            t0 = time.perf_counter_ns()
            actual = replay(name, case, suites, events)
            samples[name].append(time.perf_counter_ns() - t0)
            if actual != expected:
                raise AssertionError("timed result mismatch")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    return {"case": asdict(case), "seed": seed, "digest": expected,
            "samples_ns": samples, "medians_ns": medians,
            "c_tree_vs_set": medians["set"] / medians["c_tree"]}


def memory_worker(case: StressCase, route: str):
    suites, events = generate(case)
    # Reference verification BEFORE starting tracemalloc.
    digest = verify(case, suites, events)
    tracemalloc.start()
    try:
        observed = replay(route, case, suites, events)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if observed != digest:
        raise AssertionError("memory-worker digest mismatch")
    return {"route": route, "case": case.name, "digest": digest,
            "current_bytes": current, "peak_traced_bytes": peak, "host": host()}


def suite(*, smoke=False, pairs=3):
    cases = SMOKE_CASES if smoke else CASES
    return {"schema": "alexpy-native-witness-stress/1", "host": host(),
            "scope": "same QA C kernel; C witness tree vs Python failure set; synthetic only",
            "rows": [measure(case, pairs) for case in cases],
            "default_promotion": False}


def run(*, smoke=False, pairs=3, sessions=3):
    if type(pairs) is not int or pairs < 1 or type(sessions) is not int or sessions < 1:
        raise ValueError("sessions and pairs must be positive exact integers")
    if sessions == 1:
        return suite(smoke=smoke, pairs=pairs)
    runs = []
    for _ in range(sessions):
        cmd = [sys.executable, "-m", "alexpy.native_witness_stress", "--worker", "--pairs", str(pairs)]
        if smoke:
            cmd.append("--smoke")
        p = subprocess.run(cmd, text=True, capture_output=True, check=True)
        runs.append(json.loads(p.stdout))
    return {"schema": "alexpy-native-witness-stress/1", "host": host(),
            "sessions": runs, "default_promotion": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--memory-route", choices=tuple(ROUTES))
    parser.add_argument("--memory-case", choices=tuple(c.name for c in CASES), default=CASES[0].name)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.memory_route:
        case = next(c for c in CASES if c.name == args.memory_case)
        result = memory_worker(case, args.memory_route)
    elif args.worker:
        result = suite(smoke=args.smoke, pairs=args.pairs)
    else:
        result = run(smoke=args.smoke, pairs=args.pairs, sessions=args.sessions)
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json:
        args.json.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)


if __name__ == "__main__":
    main()
