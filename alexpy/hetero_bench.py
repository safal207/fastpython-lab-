"""Preregistered blind holdout: heterogeneous vs uniform / ant / manual.

All observations come from TRAIN workloads only. Heterogeneous axis weights
are heuristics, NOT estimates of cache misses; no held-out edge is fitted.
Frozen routes are checked for correctness on fresh input before timing.
Training, policy compilation, and runtime execution costs are separate.
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

from .ant_memory import AntMemoryGraph, ROUTES
from .ant_bench import _csv, _strict_disjoint, evaluate_routes, prepare, workloads
from .hetero_wave import HeterogeneousWaveMemory
from .partition_bench import workflow
from .wave_memory import WaveMemoryGraph

POLICIES = ("manual", "ant", "uniform", "hetero_light", "hetero_sound")


def run_session(*, train_sizes=(128, 1024), holdout_sizes=(256, 512),
                changes=(1, 4), reads=(1, 8), width=64, ticks=2, pairs=3):
    _strict_disjoint(train_sizes, holdout_sizes)
    if any(type(x) is not int or x <= 0 for x in (width, ticks, pairs)):
        raise ValueError("width, ticks and pairs must be positive integers")
    train = workloads(train_sizes, changes, reads, width=width, ticks=ticks)
    hold = workloads(holdout_sizes, changes, reads, width=width, ticks=ticks)
    if set(w.cell() for w in train) & set(w.cell() for w in hold):
        raise ValueError("holdout aliases to training coarse cell")
    models = {
        "ant": AntMemoryGraph(max_distance=4, neighbors=4),
        "uniform": WaveMemoryGraph(mode="light", max_hops=4),
        "hetero_light": HeterogeneousWaveMemory(mode="light", max_hops=4),
        "hetero_sound": HeterogeneousWaveMemory(mode="sound", max_hops=4),
    }
    t0 = time.perf_counter_ns()
    training = []
    for w in train:
        rec = evaluate_routes(w, pairs=pairs)
        if rec["gate"] != "PASS":
            raise AssertionError("training route failed the correctness gate")
        reference = rec["median_ns"]["manual"]
        for model in models.values():
            model.advance()
            for route in ROUTES:
                model.observe(w, route, elapsed_ns=rec["median_ns"][route],
                              reference_ns=reference, verified=True)
        training.append(rec)
    training_ns = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns()
    frozen = []
    for w in hold:
        frozen.append({"manual": "manual", **{k: model.choose(w) for k, model in models.items()}})
    freeze_ns = time.perf_counter_ns() - t0

    # Fail closed: before any timing, verify every selected route on holdout.
    # A route verification failure invalidates the experiment, not the baseline.
    out = []
    for w, plan in zip(hold, frozen):
        initial, patches, expected = prepare(w)
        for route in set(plan.values()):
            if workflow(route, initial, patches, w.reads, w.width) != expected:
                raise AssertionError("frozen route failed holdout correctness gate")
        elapsed = {p: [] for p in POLICIES}
        for rep in range(pairs):
            order = list(POLICIES)
            shift = rep % len(order)
            order = order[shift:] + order[:shift]
            if rep % 2:
                order.reverse()
            for policy in order:
                start = time.perf_counter_ns()
                route = "manual" if policy == "manual" else models[policy].choose(w)
                result = workflow(route, initial, patches, w.reads, w.width)
                duration = time.perf_counter_ns() - start
                if route != plan[policy] or result != expected:
                    raise AssertionError("holdout route or checksum drift")
                elapsed[policy].append(duration)
        medians = {p: statistics.median(v) for p, v in elapsed.items()}
        out.append({"state": w.__dict__, "frozen": plan,
                    "raw_ns": elapsed, "median_ns": medians,
                    "manual_over": {p: medians["manual"] / medians[p]
                                    for p in POLICIES[1:]}})
    return {
        "pid": os.getpid(), "python": platform.python_version(),
        "platform": platform.platform(), "training_ns": training_ns,
        "freeze_ns": freeze_ns, "training": training, "holdout": out,
        "memory": {name: model.snapshot() for name, model in models.items()},
        "edge_cost_provenance": "predeclared axis heuristics; no hardware-counter edge observations",
    }


def collect(args):
    train, held = _csv(args.train_sizes), _csv(args.holdout_sizes)
    changes, reads = _csv(args.changes), _csv(args.reads)
    _strict_disjoint(train, held)
    if min(args.width, args.ticks, args.pairs, args.sessions) <= 0:
        raise ValueError("all positive integer parameters required")
    if args.worker:
        return run_session(train_sizes=train, holdout_sizes=held, changes=changes,
                           reads=reads, width=args.width, ticks=args.ticks,
                           pairs=args.pairs)
    sessions = []
    for _ in range(args.sessions):
        command = [sys.executable, "-m", "alexpy.hetero_bench", "--worker",
                   "--train-sizes", args.train_sizes, "--holdout-sizes", args.holdout_sizes,
                   "--changes", args.changes, "--reads", args.reads,
                   "--width", str(args.width), "--ticks", str(args.ticks),
                   "--pairs", str(args.pairs)]
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        sessions.append(json.loads(process.stdout))
    summaries = []
    for n, original in enumerate(sessions[0]["holdout"]):
        rows = [s["holdout"][n] for s in sessions]
        if any(r["state"] != original["state"] for r in rows):
            raise AssertionError("holdout state sequence drift")
        ratios = {}
        for policy in POLICIES[1:]:
            values = [r["manual_over"][policy] for r in rows]
            ratios[policy] = {
                "median_manual_over": statistics.median(values),
                "all_processes_gain_10pct": all(x > 1.10 for x in values),
                "all_processes_regression_10pct": all(x < 1 / 1.1 for x in values),
                "selected": [r["frozen"][policy] for r in rows],
            }
        summaries.append({"state": original["state"], "results": ratios})
    result = {
        "protocol": "ALEXPY-M16-HETEROGENEOUS-WAVE/1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline verified-plan selector, no physical wave propagation",
        "promotion": "BLOCKED_EXPERIMENTAL",
        "controls": "fixed manual, homogeneous light, ant pheromones; frozen holdout",
        "cost_model": "predeclared heuristic heterogeneous axis weights; not measured cache behavior",
        "config": {"train_sizes": train, "holdout_sizes": held,
                   "changes": changes, "reads": reads, "width": args.width,
                   "ticks": args.ticks, "pairs": args.pairs, "sessions": args.sessions},
        "summary": summaries, "sessions": sessions,
    }
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sizes", default="128,1024")
    p.add_argument("--holdout-sizes", default="256,512")
    p.add_argument("--changes", default="1,4")
    p.add_argument("--reads", default="1,8")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    result = collect(args)
    if args.worker:
        print(json.dumps(result, separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"protocol": result["protocol"], "holdout": [
            {"size": row["state"]["size"], "changes": row["state"]["changes"],
             "reads": row["state"]["reads"],
             **{p: round(v["median_manual_over"], 3) for p, v in row["results"].items()}}
            for row in result["summary"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
