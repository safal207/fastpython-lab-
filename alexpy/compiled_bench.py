"""Paired end-to-end benchmark: compiled DAG vs legacy DAG and strong controls.

Timing includes typed data conversion, declaration, compilation, patches and
all requested reads. Independent processes, alternated orders and raw samples
are retained; this benchmark is advisory and never promotes a default path.
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

from .dependency_bench import _bounds, _make_graph, make_case, workflow as old_workflow
from .compiled_graph import CompiledDependencyGraph

ROUTES = ("dense", "manual", "graph", "compiled")


def _make_compiled(initial, width):
    graph = CompiledDependencyGraph(initial, bucket_size=width)
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
    graph.compile()
    return graph, tuple(var_names)


def workflow(route, initial, patches, reads, width):
    if route not in ROUTES or type(reads) is not int or reads <= 0:
        raise ValueError("invalid route or read count")
    if route != "compiled":
        return old_workflow(route, initial, patches, reads, width)
    graph, names = _make_compiled(initial, width)
    checksum = 0
    for patch in patches:
        graph.apply_patch(patch)
        for _ in range(reads):
            checksum += graph.value("global_sum") + sum(graph.value(n) for n in names)
    return checksum


def _measure(route, initial, patches, reads, width, loops):
    start = time.perf_counter_ns()
    checksum = 0
    for _ in range(loops):
        checksum += workflow(route, initial, patches, reads, width)
    elapsed = time.perf_counter_ns() - start
    if type(checksum) is not int:
        raise AssertionError("invalid checksum")
    return elapsed / loops


def measure_case(size, width, changes, ticks, reads, epochs, pairs):
    initial, patches = make_case(size, width, changes, ticks)
    expected = workflow("dense", initial, patches, reads, width)
    if any(workflow(route, initial, patches, reads, width) != expected
           for route in ROUTES[1:]):
        raise AssertionError("computed result differs among A/B controls")
    raw = {route: [] for route in ROUTES}
    for epoch in range(epochs):
        sample = {route: [] for route in ROUTES}
        for pair in range(pairs):
            order = list(ROUTES)
            step = (epoch + pair) % len(order)
            order = order[step:] + order[:step]
            if (epoch + pair) % 2:
                order.reverse()
            for route in order:
                sample[route].append(_measure(route, initial, patches, reads, width, 1))
        for route in ROUTES:
            raw[route].append(sample[route])
    medians = {route: statistics.median(value for group in raw[route] for value in group)
               for route in ROUTES}
    graph, _ = _make_compiled(initial, width)
    counts = {"examined_windows": 0, "touched_windows": 0,
              "evaluated_derived": 0, "updated_nodes": 0}
    for patch in patches:
        event = graph.apply_patch(patch)
        for name in counts:
            value = getattr(event, name)
            counts[name] += value if name == "examined_windows" else len(value)
    if not graph.verify():
        raise AssertionError("compiled graph failed full-state mirror")
    return {"state": {"size": size, "width": width, "changes": changes,
                       "ticks": ticks, "reads": reads},
            "checksum": str(expected), "nodes": len(graph.nodes()),
            "edges": len(graph.edges()), "dirty_counts": counts,
            "raw_ns": raw, "median_ns": medians,
            "legacy_over_compiled": medians["graph"] / medians["compiled"],
            "manual_over_compiled": medians["manual"] / medians["compiled"],
            "dense_over_compiled": medians["dense"] / medians["compiled"],
            "gate": "PASS"}


def _csv(raw):
    result = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not result or any(v <= 0 for v in result):
        raise ValueError("expected positive CSV numbers")
    return result


def _worker(args):
    cases = [measure_case(size, args.width, changes, ticks, reads,
                          args.epochs, args.pairs)
             for size in _csv(args.sizes) for changes in _csv(args.changes)
             for ticks in _csv(args.ticks) for reads in _csv(args.reads)]
    return {"pid": os.getpid(), "platform": platform.platform(),
            "python": platform.python_version(), "cases": cases}


def collect(args):
    if any(v <= 0 for v in (args.width, args.epochs, args.pairs, args.sessions)):
        raise ValueError("width, epochs, pairs, sessions must be positive")
    if args.worker:
        return _worker(args)
    sessions = []
    for _ in range(args.sessions):
        cmd = [sys.executable, "-m", "alexpy.compiled_bench", "--worker",
               "--sizes", args.sizes, "--width", str(args.width),
               "--changes", args.changes, "--ticks", args.ticks,
               "--reads", args.reads, "--epochs", str(args.epochs),
               "--pairs", str(args.pairs), "--sessions", "1"]
        child = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=180)
        sessions.append(json.loads(child.stdout))
    summary = []
    for i, reference in enumerate(sessions[0]["cases"]):
        records = [s["cases"][i] for s in sessions]
        if any(c["state"] != reference["state"] or c["checksum"] != reference["checksum"]
               or c["gate"] != "PASS" for c in records):
            raise AssertionError("cross-process correctness/control mismatch")
        values = {route: [c[route] for c in records]
                  for route in ("legacy_over_compiled", "manual_over_compiled",
                                "dense_over_compiled")}
        summary.append({"state": reference["state"],
                        "ratios_median": {k: statistics.median(v) for k, v in values.items()},
                        "ratios_range": {k: [min(v), max(v)] for k, v in values.items()},
                        "compiled_wins_all_sessions_10pct": {
                            k: all(x >= 1.1 for x in v) for k, v in values.items()},
                        "nodes": reference["nodes"], "edges": reference["edges"]})
    report = {"protocol": "ALEXPY-M11-COMPILED-DAG/0.1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "owned exact signed-q array; predefined sum/squares/variance DAG only",
              "timing_boundary": "full build + compile + updates + requested reads, no reuse across workflows",
              "default_promotion": "BLOCKED_OPT_IN_RESEARCH",
              "settings": {"sizes": _csv(args.sizes), "changes": _csv(args.changes),
                           "ticks": _csv(args.ticks), "reads": _csv(args.reads),
                           "width": args.width, "sessions": args.sessions,
                           "epochs": args.epochs, "pairs": args.pairs},
              "sessions": sessions, "summary": summary}
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
    return report


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
    report = collect(args)
    if args.worker:
        print(json.dumps(report, separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        for item in report["summary"]:
            state = item["state"]
            ratios = item["ratios_median"]
            print(f"n={state['size']:>6} k={state['changes']:>2} "
                  f"ticks={state['ticks']:>2} reads={state['reads']:>2} "
                  f"legacy/compiled={ratios['legacy_over_compiled']:.2f}x "
                  f"manual/compiled={ratios['manual_over_compiled']:.2f}x "
                  f"dense/compiled={ratios['dense_over_compiled']:.2f}x")


if __name__ == "__main__":
    main()
