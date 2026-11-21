"""Opt-in end-to-end comparison for a guarded partition compiler.

Includes construction, registration, compilation, patches and reads.
Both generic and specialized modes must produce the same exact checksum
as manual incremental and dense recomputation controls before timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from .compiled_bench import workflow as old_workflow
from .dependency_bench import _bounds, make_case
from .partition_graph import PartitionDependencyGraph

ROUTES = ("dense", "manual", "compiled", "partition")


def _make_partition(initial, width):
    graph = PartitionDependencyGraph(initial, bucket_size=width)
    vars_, sums = [], []
    for start, stop in _bounds(len(initial), width):
        label = str(start)
        s, q, v = "s"+label, "q"+label, "v"+label
        graph.add_window(s, start, stop, "sum")
        graph.add_window(q, start, stop, "squares")
        graph.add_variance(v, s, q)
        vars_.append(v)
        sums.append(s)
    if not sums:
        raise ValueError("benchmark needs a nonempty input")
    graph.add_sum("global_sum", tuple(sums))
    graph.compile()
    if not graph.specialized:
        raise AssertionError("partition compiler did not recognize benchmark DAG")
    return graph, vars_


def workflow(route, initial, patches, reads, width):
    if route != "partition":
        return old_workflow(route, initial, patches, reads, width)
    graph, names = _make_partition(initial, width)
    checksum = 0
    for patch in patches:
        graph.apply_patch(patch)
        for _ in range(reads):
            checksum += graph.value("global_sum") + sum(graph.value(n) for n in names)
    return checksum


def _measure(route, initial, patches, reads, width):
    start = time.perf_counter_ns()
    result = workflow(route, initial, patches, reads, width)
    return time.perf_counter_ns() - start, result


def measure_case(size, width, changes, ticks, reads, epochs, pairs):
    initial, patches = make_case(size, width, changes, ticks)
    reference = workflow("dense", initial, patches, reads, width)
    for route in ROUTES[1:]:
        if workflow(route, initial, patches, reads, width) != reference:
            raise AssertionError(f"checksum mismatch: {route}")
    samples = {route: [] for route in ROUTES}
    for epoch in range(epochs):
        round_samples = {route: [] for route in ROUTES}
        for pair in range(pairs):
            order = list(ROUTES)
            k = (epoch+pair) % len(order)
            order = order[k:] + order[:k]
            if (epoch+pair) % 2:
                order.reverse()
            for route in order:
                ns, actual = _measure(route, initial, patches, reads, width)
                if actual != reference:
                    raise AssertionError("timed result mismatch")
                round_samples[route].append(ns)
        for route in ROUTES:
            samples[route].append(round_samples[route])
    medians = {route: statistics.median(x for batch in samples[route] for x in batch)
               for route in ROUTES}
    return {
        "state": {"size": size, "width": width, "changes": changes,
                  "ticks": ticks, "reads": reads},
        "checksum": str(reference), "gate": "PASS", "raw_ns": samples,
        "median_ns": medians,
        "ratios": {"compiled_over_partition": medians["compiled"] / medians["partition"],
                   "manual_over_partition": medians["manual"] / medians["partition"],
                   "dense_over_partition": medians["dense"] / medians["partition"]},
    }


def _positive_csv(s):
    values = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("expected positive comma-separated integers")
    return values


def worker(args):
    cases = [measure_case(n, args.width, k, t, r, args.epochs, args.pairs)
             for n in _positive_csv(args.sizes)
             for k in _positive_csv(args.changes)
             for t in _positive_csv(args.ticks)
             for r in _positive_csv(args.reads)]
    return {"pid": os.getpid(), "python": platform.python_version(),
            "platform": platform.platform(), "cases": cases}


def collect(args):
    if min(args.width, args.epochs, args.pairs, args.sessions) <= 0:
        raise ValueError("all control counts must be positive")
    if args.worker:
        return worker(args)
    sessions = []
    for _ in range(args.sessions):
        command = [sys.executable, "-m", "alexpy.partition_bench", "--worker",
                   "--sizes", args.sizes, "--width", str(args.width),
                   "--changes", args.changes, "--ticks", args.ticks,
                   "--reads", args.reads, "--epochs", str(args.epochs),
                   "--pairs", str(args.pairs)]
        process = subprocess.run(command, capture_output=True, text=True,
                                 check=True, timeout=180)
        sessions.append(json.loads(process.stdout))
    summary = []
    for i, first in enumerate(sessions[0]["cases"]):
        records = [s["cases"][i] for s in sessions]
        if any(c["state"] != first["state"] or c["checksum"] != first["checksum"]
               or c["gate"] != "PASS" for c in records):
            raise AssertionError("cross-process correctness or state mismatch")
        series = {k: [c["ratios"][k] for c in records]
                  for k in first["ratios"]}
        summary.append({"state": first["state"],
                        "ratios_median": {k: statistics.median(v) for k,v in series.items()},
                        "ratios_range": {k: [min(v),max(v)] for k,v in series.items()},
                        "partition_wins_all_10pct": {k: all(x>=1.1 for x in v)
                                                      for k,v in series.items()}})
    result = {"protocol": "ALEXPY-M12-PARTITION-COMPILER/0.1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "opt-in template-specialized DAG with provable partition alignment",
              "default_promotion": "BLOCKED_EXPERIMENTAL",
              "timing_boundary": "build+compile+patches+all_reads in fresh workflow",
              "settings": {"sessions": args.sessions, "epochs": args.epochs,
                           "pairs": args.pairs, "sizes": _positive_csv(args.sizes),
                           "changes": _positive_csv(args.changes),
                           "ticks": _positive_csv(args.ticks),
                           "reads": _positive_csv(args.reads), "width": args.width},
              "sessions": sessions, "summary": summary}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="4096,32768")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--changes", default="1,16")
    parser.add_argument("--ticks", default="2,32")
    parser.add_argument("--reads", default="1,8")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = collect(args)
    if args.worker:
        print(json.dumps(result, separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        for case in result["summary"]:
            s, r = case["state"], case["ratios_median"]
            print(f"n={s['size']:>6} k={s['changes']:>2} ticks={s['ticks']:>2} "
                  f"reads={s['reads']:>2}: previous/partition="
                  f"{r['compiled_over_partition']:.2f}x "
                  f"manual/partition={r['manual_over_partition']:.2f}x "
                  f"dense/partition={r['dense_over_partition']:.2f}x")


if __name__ == "__main__":
    main()
