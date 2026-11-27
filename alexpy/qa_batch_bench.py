"""QA batch research: four fair routes, pairwise rotations, independent mirrors.

All workloads are synthetic. Every timed route includes initial engine
construction, slicing events into batches, validation, updates, snapshots and
hashing. The reported speedup against *StrongPythonBatchQA* is the primary
criterion. Batching hides intermediate events by design; this is an explicit
report-boundary contract, not a drop-in replacement for event notifications.
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
from dataclasses import asdict, dataclass
from pathlib import Path

from .qa_batch import AlexPyBatchQAGraph, StrongPythonBatchQA
from .qa_lab import (
    AlexPyQAGraph, StrongPythonQA, QAEvent, scan_reference,
    NOT_RUN, PASS, FAIL, SKIP,
)

ROUTES = ("strong_event", "alexpy_event", "strong_batch", "alexpy_batch")
BATCHES = (1, 8, 64, 256)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    tests: int
    suites: int
    events: int
    active_tests: int
    report_every_batches: int
    reads_per_report: int


FULL_CASES = (
    Case("small-frequent", 256, 8, 2400, 256, 1, 1),
    Case("medium-mixed", 4096, 32, 6000, 4096, 4, 2),
    Case("large-sparse", 16384, 128, 10000, 16384, 8, 2),
    Case("medium-hot-retries", 4096, 32, 8000, 96, 4, 2),
)
SMOKE_CASES = (
    Case("smoke-small", 128, 4, 160, 128, 2, 1),
    Case("smoke-retries", 256, 8, 240, 16, 4, 1),
)


def generate(case: Case, seed: int = 16641):
    rng = random.Random(seed + case.tests * 7 + case.events)
    suites = [(i // 3 + i * 17) % case.suites for i in range(case.tests)]
    events = []
    for _ in range(case.events):
        index = rng.randrange(case.active_tests)
        status = rng.choices((PASS, FAIL, SKIP), weights=(77, 18, 5))[0]
        duration = 0 if status == SKIP else rng.randrange(0, 100001)
        events.append(QAEvent(index, status, duration))
    return suites, events


def _digest_step(digest, snap):
    return (digest * 1000003 + hash(snap)) & ((1 << 64) - 1)


def replay(route, case, batch_size, suites, events):
    engines = {
        "strong_event": StrongPythonQA,
        "alexpy_event": AlexPyQAGraph,
        "strong_batch": StrongPythonBatchQA,
        "alexpy_batch": AlexPyBatchQAGraph,
    }
    if route not in engines:
        raise ValueError("unsupported QA batch route")
    engine = engines[route](suites, case.suites)
    digest = 0
    batch_count = (len(events) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(events), batch_size), 1):
        batch = events[start:start + batch_size]  # included in timed path
        if route.endswith("_batch"):
            engine.apply_batch(batch)
        else:
            for event in batch:
                engine.apply(event)
        if bi % case.report_every_batches == 0 or bi == batch_count:
            for _ in range(case.reads_per_report):
                digest = _digest_step(digest, engine.snapshot())
    return digest


def verify(case, batch_size, suites, events):
    """Fail-closed at every report boundary, plus independent scan witness."""
    engines = [StrongPythonQA(suites, case.suites),
               AlexPyQAGraph(suites, case.suites),
               StrongPythonBatchQA(suites, case.suites),
               AlexPyBatchQAGraph(suites, case.suites)]
    statuses = [NOT_RUN] * len(suites)
    durations = [0] * len(suites)
    batch_count = (len(events) + batch_size - 1) // batch_size
    # Independent O(N) scans at bounded, deterministic checkpoints, rather
    # than every report. Every report still mirrors all four incremental
    # engines; the independent scan is NOT removed, only limited to nine
    # evenly distributed checkpoints plus the final state.
    scan_every = max(1, batch_count // 8)
    witness = 0
    for bi, start in enumerate(range(0, len(events), batch_size), 1):
        batch = events[start:start + batch_size]
        for event in batch:
            engines[0].apply(event)
            engines[1].apply(event)
            statuses[event.test_id] = event.status
            durations[event.test_id] = event.duration_ms
        engines[2].apply_batch(batch)
        engines[3].apply_batch(batch)
        if bi % case.report_every_batches == 0 or bi == batch_count:
            expected = engines[0].snapshot()
            if any(e.snapshot() != expected for e in engines[1:]):
                raise AssertionError(f"incremental mirrors diverged at batch {bi} in {case.name}")
            for _ in range(case.reads_per_report):
                witness = _digest_step(witness, expected)
        if bi % scan_every == 0 or bi == batch_count:
            expected_full = scan_reference(suites, case.suites, statuses, durations)
            if any(e.snapshot() != expected_full for e in engines):
                raise AssertionError(f"independent scan diverged at batch {bi} in {case.name}")
    if not engines[1].verify() or not engines[3].verify():
        raise AssertionError("owned-buffer verification failed")
    for route in ROUTES:
        if replay(route, case, batch_size, suites, events) != witness:
            raise AssertionError(f"replay digest differs for {route}")
    return witness


def worker(pairs: int, smoke: bool, batch_sizes: tuple[int, ...]):
    cases = SMOKE_CASES if smoke else FULL_CASES
    output = []
    for case in cases:
        suites, events = generate(case)
        for size in batch_sizes:
            witness = verify(case, size, suites, events)
            measurements = []
            for pair in range(pairs):
                # One complete rotation of order every four pairs.
                shift = pair % len(ROUTES)
                order = ROUTES[shift:] + ROUTES[:shift]
                if pair % 2:
                    order = tuple(reversed(order))
                durations = {}
                for route in order:
                    start = time.perf_counter_ns()
                    result = replay(route, case, size, suites, events)
                    elapsed = time.perf_counter_ns() - start
                    if result != witness:
                        raise AssertionError(f"timed replay diverged for {route}")
                    durations[route] = elapsed
                measurements.append({
                    "order": list(order), "durations_ns": durations,
                    "speedup_vs_strong_batch": durations["strong_batch"] / durations["alexpy_batch"],
                    "speedup_vs_old_graph": durations["alexpy_event"] / durations["alexpy_batch"],
                    "speedup_vs_strong_event": durations["strong_event"] / durations["alexpy_batch"],
                })
            output.append({
                "case": asdict(case), "batch_size": size, "witness_digest": witness,
                "pairs": measurements,
                "median_speedup_vs_strong_batch": statistics.median(x["speedup_vs_strong_batch"] for x in measurements),
                "median_speedup_vs_old_graph": statistics.median(x["speedup_vs_old_graph"] for x in measurements),
                "median_speedup_vs_strong_event": statistics.median(x["speedup_vs_strong_event"] for x in measurements),
            })
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor(), "executable": sys.executable,
        "cases": output,
    }


def run(sessions=3, pairs=5, smoke=False, batch_sizes=BATCHES):
    if (type(sessions) is not int or sessions < 1 or
            type(pairs) is not int or pairs < 1 or
            not batch_sizes or any(type(n) is not int or n < 1 for n in batch_sizes)):
        raise ValueError("sessions, pairs, batch sizes must be positive exact integers")
    workers = []
    for _ in range(sessions):
        cmd = [sys.executable, "-m", "alexpy.qa_batch_bench", "--worker",
               "--pairs", str(pairs), "--batch-sizes", ",".join(map(str, batch_sizes))]
        if smoke:
            cmd.append("--smoke")
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        workers.append(json.loads(result.stdout))
    summary = []
    for ci, cell in enumerate(workers[0]["cases"]):
        speedups = [w["cases"][ci]["median_speedup_vs_strong_batch"] for w in workers]
        old_speedups = [w["cases"][ci]["median_speedup_vs_old_graph"] for w in workers]
        summary.append({
            "case": cell["case"]["name"], "batch_size": cell["batch_size"],
            "process_speedups_vs_strong_batch": speedups,
            "median_speedup_vs_strong_batch": statistics.median(speedups),
            "all_processes_over_1_1": all(x > 1.1 for x in speedups),
            "all_processes_under_0_9": all(x < 0.9 for x in speedups),
            "process_speedups_vs_old_graph": old_speedups,
            "median_speedup_vs_old_graph": statistics.median(old_speedups),
        })
    return {
        "schema": "ALEXPY-QA-BATCH/1", "source": "synthetic QA events, deterministic seed",
        "correctness": "all four implementations mirrored at report boundaries; nine O(N) scans spread across stream",
        "timed_scope": "initialization, in-loop batch slicing/validation, updates, snapshot and hashing",
        "semantics": "last event wins per test per atomic batch; no intermediate per-event reports",
        "primary_speedup": "strong_python_batch_ns / alexpy_batch_ns",
        "sessions": sessions, "pairs": pairs, "summary": summary, "workers": workers,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-sizes", default="1,8,64,256")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    batch_sizes = tuple(int(x) for x in args.batch_sizes.split(","))
    if args.worker:
        print(json.dumps(worker(args.pairs, args.smoke, batch_sizes)))
        return 0
    data = run(args.sessions, args.pairs, args.smoke, batch_sizes)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in data["summary"]:
        print(f'{item["case"]} batch={item["batch_size"]}: '
              f'{item["median_speedup_vs_strong_batch"]:.3f}x vs Python batch; '
              f'{item["median_speedup_vs_old_graph"]:.3f}x vs old AlexPy graph')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
