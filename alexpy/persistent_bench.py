"""Full-workflow benchmark for owned incremental q buffer vs honest controls.

Benchmarks include data construction, all patches, and all reads. Three fresh
processes are supported for exploratory replication. Never auto-promotes.
"""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from . import sum_squares
from .memory_candidates import _buffer_native
from .persistent import OwnedQBuffer

ROUTES = ("list_full", "array_full", "owned_incremental")


def make_case(size: int, changes: int, reads: int, ticks: int, seed: int = 918):
    if size <= 0 or changes <= 0 or reads <= 0 or ticks <= 0:
        raise ValueError("size/changes/reads/ticks must all be positive")
    initial = [((i * 97 + 31) % 51) - 25 for i in range(size)]
    rng = random.Random(seed + size * 131 + changes * 17 + reads)
    patches = []
    for tick in range(ticks):
        indices = rng.sample(range(size), min(size, changes))
        patches.append({index: ((index * 19 + tick * 31) % 127) - 63 for index in indices})
    return initial, patches


def workflow(route: str, initial: list[int], patches: list[dict[int, int]], reads: int):
    """Identical logical outputs for all routes; no early exit / skipped query."""
    if route == "owned_incremental":
        values = OwnedQBuffer(initial)
    elif route == "array_full":
        values = array("q", initial)
    elif route == "list_full":
        values = list(initial)
    else:
        raise ValueError("unknown route")
    checksum = 0
    for patch in patches:
        if route == "owned_incremental":
            values.apply_patch(patch)
        else:
            for index, new in patch.items():
                values[index] = new
        for _ in range(reads):
            if route == "owned_incremental":
                total, squares = values.aggregates()
                variance_numerator = values.variance_numerator()
            elif route == "array_full":
                total = sum(values)
                if _buffer_native is not None:
                    squares = _buffer_native.sum_squares_linear(values)
                else:
                    squares = sum(v*v for v in values)
                variance_numerator = len(values) * squares - total * total
            else:
                total = sum(values)
                squares = sum_squares(values)
                variance_numerator = len(values) * squares - total * total
            checksum += total + squares + variance_numerator
    return checksum


def _measure(route: str, initial, patches, reads, loops: int) -> float:
    start = time.perf_counter_ns()
    checksum = 0
    for _ in range(loops):
        checksum += workflow(route, initial, patches, reads)
    elapsed = time.perf_counter_ns() - start
    # Do not include comparisons or witness checks in timing.
    if type(checksum) is not int:
        raise RuntimeError("unexpected checksum")
    return elapsed / loops


def case_measurements(size, changes, reads, ticks, epochs, pairs, target_ms):
    initial, patches = make_case(size, changes, reads, ticks)
    expected = workflow("list_full", initial, patches, reads)
    for route in ROUTES[1:]:
        if workflow(route, initial, patches, reads) != expected:
            raise AssertionError(f"mirror mismatch: {route}")
    # Cost of the three candidate initializations is included in measurements.
    for route in ROUTES:
        workflow(route, initial, patches, reads)
    trial = _measure("list_full", initial, patches, reads, 1)
    loops = max(1, min(100, int(target_ms * 1_000_000 / max(1.0, trial))))
    raw = {route: [] for route in ROUTES}
    for epoch in range(epochs):
        epoch_raw = {route: [] for route in ROUTES}
        for pair in range(pairs):
            order = list(ROUTES)
            if (epoch + pair) % 2:
                order.reverse()
            else:
                order = order[1:] + order[:1]
            for route in order:
                epoch_raw[route].append(_measure(route, initial, patches, reads, loops))
        for route in ROUTES:
            raw[route].append(epoch_raw[route])
    medians = {route: statistics.median([item for epoch in raw[route] for item in epoch])
               for route in ROUTES}
    ratios = {base: medians[base] / medians["owned_incremental"]
              for base in ROUTES[:2]}
    return {
        "state": {"size": size, "changes_per_tick": changes, "reads_per_tick": reads,
                  "ticks": ticks},
        "data_bytes": size * 8,
        "logical_checks": ticks * reads,
        "transition_edges": sum(len(p) for p in patches),
        "loops_per_sample": loops,
        "checksum": str(expected),
        "medians_ns_per_workflow": medians,
        "speedup_vs_owned": ratios,
        "samples_ns_per_workflow": raw,
        "correctness_mirrors": "PASS",
    }


def _parse_ints(text: str):
    out = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not out or any(v <= 0 for v in out):
        raise ValueError("expected positive comma-separated integers")
    return out


def _single_process(args):
    results = [case_measurements(size, change, reads, args.ticks, args.epochs,
                                 args.pairs, args.target_ms)
               for size in _parse_ints(args.sizes)
               for change in _parse_ints(args.changes)
               for reads in _parse_ints(args.reads)]
    return {"pid": os.getpid(), "interpreter": platform.python_version(),
            "platform": platform.platform(), "results": results}


def graph_edges(summary):
    """One-axis neighboring cells in the measured configuration grid."""
    axes = ("size", "changes_per_tick", "reads_per_tick")
    states = [row["state"] for row in summary]
    edges = []
    for axis in axes:
        frozen = [other for other in axes if other != axis]
        for left in states:
            neighbors = [right for right in states
                         if all(right[k] == left[k] for k in frozen)
                         and right[axis] > left[axis]]
            if neighbors:
                right = min(neighbors, key=lambda state: state[axis])
                edges.append({"from": left, "to": right, "axis": axis})
    edges.sort(key=lambda edge: (edge["axis"], str(edge["from"]), str(edge["to"])))
    return edges


def dot_graph(summary, edges):
    """Deterministic DOT summary; styles show measurements, not causal proof."""
    def node_id(state):
        return f"n{state['size']}_k{state['changes_per_tick']}_r{state['reads_per_tick']}"
    lines = ["digraph AlexPyPersistent {", "  rankdir=LR;",
             '  label="Measured workload state graph; ratios vs AlexPy list control";']
    for row in summary:
        state = row["state"]
        ratio = row["list_full_ratio_median"]
        color = "#d8ecdb" if ratio >= 1.1 else "#fce8e6" if ratio < 0.9 else "#f2f2f2"
        label = (f"n={state['size']} k={state['changes_per_tick']} "
                 f"r={state['reads_per_tick']}\\n{ratio:.2f}x")
        lines.append(f'  {node_id(state)} [label="{label}", style=filled, fillcolor="{color}"];')
    for edge in edges:
        lines.append(f'  {node_id(edge["from"])} -> {node_id(edge["to"])} '
                     f'[label="{edge["axis"]}"];')
    return "\n".join(lines + ["}", ""])


def collect(args):
    if args.sessions <= 0 or args.epochs <= 0 or args.pairs <= 0 or args.target_ms <= 0:
        raise ValueError("sessions, epochs, pairs, target_ms must be positive")
    if args.ticks <= 0:
        raise ValueError("ticks must be positive")
    config = {"sizes": _parse_ints(args.sizes), "changes": _parse_ints(args.changes),
              "reads": _parse_ints(args.reads), "ticks": args.ticks,
              "epochs": args.epochs, "pairs": args.pairs,
              "target_ms": args.target_ms, "sessions": args.sessions}
    if args.worker:
        return _single_process(args)
    sessions = []
    for _ in range(args.sessions):
        command = [sys.executable, "-m", "alexpy.persistent_bench", "--worker",
                   "--sizes", args.sizes, "--changes", args.changes,
                   "--reads", args.reads, "--ticks", str(args.ticks),
                   "--epochs", str(args.epochs), "--pairs", str(args.pairs),
                   "--target-ms", str(args.target_ms)]
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=True, timeout=180)
        sessions.append(json.loads(p.stdout))
    summary = []
    for i, first in enumerate(sessions[0]["results"]):
        state = first["state"]
        points = [session["results"][i] for session in sessions]
        if any(point["state"] != state for point in points):
            raise AssertionError("inconsistent session grids")
        record = {"state": state, "data_bytes": first["data_bytes"]}
        for base in ROUTES[:2]:
            values = [point["speedup_vs_owned"][base] for point in points]
            record[base + "_ratio_median"] = statistics.median(values)
            record[base + "_ratio_range"] = [min(values), max(values)]
            record[base + "_all_sessions_gain_10pct"] = all(v >= 1.1 for v in values)
        summary.append(record)
    results = {
        "protocol": "ALEXPY-PERSISTENT-M9/0.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single sum/sum_squares/variance aggregate; controlled exact signed-q arrays only",
        "baseline": "end-to-end build + patches + aggregate reads",
        "promotion": "BLOCKED_EXPERIMENTAL",
        "hardware_cache_causality": "NOT_ESTABLISHED",
        "config": config,
        "sessions": sessions,
        "summary": summary,
        "graph_edges": graph_edges(summary),
    }
    canonical = json.dumps(results, sort_keys=True, separators=(",", ":"))
    results["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="64,4096,32768")
    parser.add_argument("--changes", default="1,32")
    parser.add_argument("--reads", default="1,8,64")
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=1.5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    result = collect(args)
    if args.worker:
        print(json.dumps(result, separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.dot:
            args.dot.write_text(dot_graph(result["summary"], result["graph_edges"]), encoding="utf-8")
        for row in result["summary"]:
            print(f"n={row['state']['size']:>6} k={row['state']['changes_per_tick']:>3} "
                  f"reads={row['state']['reads_per_tick']:>3} "
                  f"list={row['list_full_ratio_median']:.3f}x "
                  f"array={row['array_full_ratio_median']:.3f}x "
                  f"all_sessions_list_gain={row['list_full_all_sessions_gain_10pct']}")
        print(f"sessions={args.sessions}; cases={len(result['summary'])}; no auto-promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
