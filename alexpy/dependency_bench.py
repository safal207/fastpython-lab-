"""Paired end-to-end controls for a sparse, declarative dependency DAG.

All routes build their own state, apply the same exact patches and answer the
same window-variance + global-sum queries. Construction is INCLUDED. The hand-
written incremental control separates an algorithmic gain from graph overhead.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from . import sum_squares
from .dependency_graph import DependencyGraph

ROUTES = ("dense", "manual", "graph")


def make_case(size: int, width: int, changes: int, ticks: int):
    if (any(type(x) is not int for x in (size, width, changes, ticks))
            or min(size, width, changes, ticks) <= 0 or changes > size):
        raise ValueError("size, width, changes, ticks must be positive and changes <= size")
    initial = [((i * 17) % 251) - 125 for i in range(size)]
    patches = []
    for tick in range(ticks):
        indices = []
        seen = set()
        attempt = 0
        stride = 997
        while gcd(stride, size) != 1:
            stride += 2
        while len(indices) < changes:
            idx = (tick * 37 + attempt * stride) % size
            attempt += 1
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        patches.append({idx: 10000 + tick * changes + j
                        for j, idx in enumerate(indices)})
    return initial, patches


def _bounds(n, width):
    return tuple((start, min(n, start + width)) for start in range(0, n, width))


def _make_graph(initial, width):
    graph = DependencyGraph(initial, bucket_size=width)
    var_names, sum_names = [], []
    for start, stop in _bounds(len(initial), width):
        name = str(start)
        s, q, v = "s" + name, "q" + name, "v" + name
        graph.add_window(s, start, stop, "sum")
        graph.add_window(q, start, stop, "squares")
        graph.add_variance(v, s, q)
        sum_names.append(s)
        var_names.append(v)
    graph.add_sum("global_sum", tuple(sum_names))
    return graph, tuple(var_names)


def workflow(route: str, initial: list[int], patches: list[dict[int, int]],
             reads: int, width: int) -> int:
    """Return one exact checksum, equal for all three routes."""
    if route not in ROUTES or type(reads) is not int or reads <= 0:
        raise ValueError("invalid route or read count")
    bounds = _bounds(len(initial), width)
    checksum = 0
    if route == "graph":
        graph, names = _make_graph(initial, width)
        for patch in patches:
            graph.apply_patch(patch)
            for _ in range(reads):
                checksum += (graph.value("global_sum")
                             + sum(graph.value(name) for name in names))
    elif route == "manual":
        data = initial.copy()
        totals = [sum(data[start:stop]) for start, stop in bounds]
        squares = [sum_squares(data[start:stop]) for start, stop in bounds]
        global_total = sum(totals)
        for patch in patches:
            for index, new in patch.items():
                old = data[index]
                data[index] = new
                segment = index // width
                difference = new-old
                totals[segment] += difference
                squares[segment] += new*new-old*old
                global_total += difference
            for _ in range(reads):
                checksum += global_total + sum(
                    (stop-start)*q-s*s for (start, stop), s, q in
                    zip(bounds, totals, squares))
    else:
        data = initial.copy()
        for patch in patches:
            for index, new in patch.items():
                data[index] = new
            for _ in range(reads):
                total = 0
                variants = 0
                for start, stop in bounds:
                    window = data[start:stop]
                    s = sum(window)
                    q = sum_squares(window)  # strong existing native C baseline
                    total += s
                    variants += (stop-start)*q - s*s
                checksum += total + variants
    return checksum


def _measure(route, initial, patches, reads, width, loops):
    t0 = time.perf_counter_ns()
    result = 0
    for _ in range(loops):
        result += workflow(route, initial, patches, reads, width)
    duration = time.perf_counter_ns() - t0
    if type(result) is not int:
        raise AssertionError("invalid checksum")
    return duration/loops


def measure_case(size, width, changes, ticks, reads, epochs, pairs, target_ms):
    initial, patches = make_case(size, width, changes, ticks)
    expected = workflow("dense", initial, patches, reads, width)
    for route in ROUTES[1:]:
        if workflow(route, initial, patches, reads, width) != expected:
            raise AssertionError("mirror mismatch: " + route)
    trial = _measure("dense", initial, patches, reads, width, 1)
    loops = max(1, min(12, int(target_ms * 1_000_000 / max(1., trial))))
    raw = {route: [] for route in ROUTES}
    for epoch in range(epochs):
        epoch_raw = {route: [] for route in ROUTES}
        for pair in range(pairs):
            order = list(ROUTES)
            order = order[(epoch+pair) % len(order):] + order[:(epoch+pair) % len(order)]
            if (epoch+pair) % 2:
                order.reverse()
            for route in order:
                epoch_raw[route].append(_measure(route, initial, patches, reads, width, loops))
        for route in ROUTES:
            raw[route].append(epoch_raw[route])
    median = {route: statistics.median([x for row in raw[route] for x in row])
              for route in ROUTES}
    graph, names = _make_graph(initial, width)
    counts = {"touched_windows": 0, "updated_nodes": 0,
              "evaluated_derived": 0, "examined_windows": 0}
    for patch in patches:
        event = graph.apply_patch(patch)
        for key in counts:
            counts[key] += (getattr(event, key) if key == "examined_windows"
                            else len(getattr(event, key)))
    return {
        "state": {"size": size, "width": width, "changes": changes,
                  "ticks": ticks, "reads": reads},
        "checksum": str(expected), "node_count": len(graph.nodes()),
        "edge_count": len(graph.edges()), "dirty_counters": counts,
        "loops_per_sample": loops, "raw_ns_per_workflow": raw,
        "median_ns_per_workflow": median,
        "speedup_vs_graph": {route: median[route]/median["graph"]
                             for route in ROUTES if route != "graph"},
        "correctness_gate": "PASS",
    }


def _positive_csv(v):
    result = [int(x.strip()) for x in v.split(",") if x.strip()]
    if not result or any(x <= 0 for x in result):
        raise ValueError("positive comma-separated integers required")
    return result


def _worker(args):
    sizes = _positive_csv(args.sizes)
    changes = _positive_csv(args.changes)
    ticks = _positive_csv(args.ticks)
    reads = _positive_csv(args.reads)
    cases = []
    for n in sizes:
        for change in changes:
            for t in ticks:
                for r in reads:
                    cases.append(measure_case(n, args.width, change, t, r,
                                              args.epochs, args.pairs, args.target_ms))
    return {"pid": os.getpid(), "platform": platform.platform(),
            "python": platform.python_version(), "cases": cases}


def collect(args):
    if (args.sessions <= 0 or args.epochs <= 0 or args.pairs <= 0
            or args.target_ms <= 0 or args.width <= 0):
        raise ValueError("sessions, epochs, pairs, width and target-ms must be positive")
    if args.worker:
        return _worker(args)
    sessions = []
    for _ in range(args.sessions):
        command = [sys.executable, "-m", "alexpy.dependency_bench", "--worker",
                   "--sizes", args.sizes, "--width", str(args.width),
                   "--changes", args.changes, "--ticks", args.ticks,
                   "--reads", args.reads, "--epochs", str(args.epochs),
                   "--pairs", str(args.pairs), "--target-ms", str(args.target_ms)]
        proc = subprocess.run(command, text=True, capture_output=True,
                              check=True, timeout=160)
        sessions.append(json.loads(proc.stdout))
    summary = []
    for i, record in enumerate(sessions[0]["cases"]):
        all_records = [s["cases"][i] for s in sessions]
        if any(x["state"] != record["state"] or x["checksum"] != record["checksum"]
               for x in all_records):
            raise AssertionError("inconsistent process results")
        ratios = {route: [x["speedup_vs_graph"][route] for x in all_records]
                  for route in ("dense", "manual")}
        summary.append({
            "state": record["state"], "node_count": record["node_count"],
            "edge_count": record["edge_count"],
            "dirty_counters": record["dirty_counters"],
            "ratio_median": {r: statistics.median(x) for r, x in ratios.items()},
            "ratio_range": {r: [min(x), max(x)] for r, x in ratios.items()},
            "graph_beats_all_sessions_10pct": {
                r: all(x >= 1.1 for x in xs) for r, xs in ratios.items()},
        })
    result = {
        "protocol": "ALEXPY-DEPENDENCY-M10/0.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "opt-in declared DAG; exact signed-q array and sum/square/variance windows only",
        "timing_boundary": "construct graph/control + apply patches + query all declared variances + global sum",
        "promotion": "BLOCKED_EXPERIMENTAL",
        "config": {"sizes": _positive_csv(args.sizes), "width": args.width,
                   "changes": _positive_csv(args.changes),
                   "ticks": _positive_csv(args.ticks),
                   "reads": _positive_csv(args.reads), "sessions": args.sessions,
                   "epochs": args.epochs, "pairs": args.pairs,
                   "target_ms": args.target_ms},
        "sessions": sessions,
        "summary": summary,
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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
    parser.add_argument("--target-ms", type=float, default=.5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = collect(args)
    if args.worker:
        print(json.dumps(report, separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        for row in report["summary"]:
            state = row["state"]
            ratios = row["ratio_median"]
            print(f"n={state['size']:>6} k={state['changes']:>2} "
                  f"ticks={state['ticks']:>2} reads={state['reads']:>2} "
                  f"dense/graph={ratios['dense']:.2f}x "
                  f"manual/graph={ratios['manual']:.2f}x")


if __name__ == "__main__":
    main()
