"""Predeclared blind-holdout, end-to-end trial for an ant-inspired route memory.

The *only* learnable evidence is measured on training sizes. Every candidate
must first pass dense checksum equivalence. Holdout selections are frozen
BEFORE any holdout route is timed. The strong manual incremental algorithm is
an explicit baseline, plus fixed partition and hindsight (nondeployable) oracle.
The timing boundary covers plan construction, patching, reads, and policy
selection for ACO. Training time and memory cost are reported separately.
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

from .ant_memory import AntMemoryGraph, ROUTES, Workload
from .dependency_bench import make_case
from .partition_bench import workflow


def _csv(value: str) -> tuple[int, ...]:
    items = tuple(int(s.strip()) for s in value.split(",") if s.strip())
    if not items or any(x <= 0 for x in items) or len(set(items)) != len(items):
        raise ValueError("expected unique, positive integers")
    return items


def workloads(sizes, changes, reads, *, width, ticks):
    return [Workload(n, width, k, ticks, r)
            for n in sizes for k in changes for r in reads]


def prepare(state):
    initial, patches = make_case(state.size, state.width, state.changes, state.ticks)
    reference = workflow("dense", initial, patches, state.reads, state.width)
    # Equivalence gate is prior to every timing, including the held-out data.
    for route in ROUTES:
        actual = workflow(route, initial, patches, state.reads, state.width)
        if actual != reference:
            raise AssertionError(f"route {route} not equivalent in {state}: {actual} != {reference}")
    return initial, patches, reference


def timed(route, state, initial, patches, reference):
    before = time.perf_counter_ns()
    result = workflow(route, initial, patches, state.reads, state.width)
    elapsed = time.perf_counter_ns() - before
    if result != reference:
        raise AssertionError("checksum drift while timing")
    return elapsed


def evaluate_routes(state, *, pairs):
    initial, patches, expected = prepare(state)
    durations = {r: [] for r in ROUTES}
    for pair in range(pairs):
        order = list(ROUTES)
        offset = pair % len(order)
        order = order[offset:] + order[:offset]
        if pair & 1:
            order.reverse()
        for route in order:
            durations[route].append(timed(route, state, initial, patches, expected))
    return {"state": state.__dict__, "checksum": str(expected),
            "gate": "PASS", "raw_ns": durations,
            "median_ns": {r: statistics.median(durations[r]) for r in ROUTES}}


def online_pair(state, prediction, memory, *, pairs):
    """Run the policy and the strong manual control, not just sum estimates."""
    initial, patches, reference = prepare(state)
    samples = {"ant": [], "manual": [], "partition": []}
    for sample in range(pairs):
        order = list(samples)
        order = order[sample % 3:] + order[:sample % 3]
        if sample & 1:
            order.reverse()
        for candidate in order:
            start = time.perf_counter_ns()
            if candidate == "ant":
                plan = memory.choose(state, allowed=ROUTES)
                if plan != prediction:
                    raise AssertionError("frozen prediction changed during evaluation")
            else:
                plan = candidate
            actual = workflow(plan, initial, patches, state.reads, state.width)
            duration = time.perf_counter_ns() - start
            if actual != reference:
                raise AssertionError("policy returned an incorrect result")
            samples[candidate].append(duration)
    return {"raw_ns": samples,
            "median_ns": {r: statistics.median(s) for r, s in samples.items()}}


def _strict_disjoint(training, holdouts):
    if not training or not holdouts or set(training).intersection(holdouts):
        raise ValueError("training and held-out sizes must be nonempty and disjoint")


def run_session(*, train_sizes=(4096, 32768), holdout_sizes=(8192, 16384),
                changes=(1, 8), reads=(1, 8), width=256, ticks=4, pairs=3):
    _strict_disjoint(train_sizes, holdout_sizes)
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("pairs must be positive")
    train = workloads(train_sizes, changes, reads, width=width, ticks=ticks)
    test = workloads(holdout_sizes, changes, reads, width=width, ticks=ticks)
    # A whole unseen size is held out, not just individual repeats of a state.
    if {x.cell() for x in train}.intersection({x.cell() for x in test}):
        raise ValueError("train and holdout collapse to same coarse grid cells")
    memory = AntMemoryGraph(rho=.1, alpha=1.0, max_distance=3, neighbors=4)
    train_start = time.perf_counter_ns()
    train_records = []
    previous = None
    for case in train:
        record = evaluate_routes(case, pairs=pairs)
        manual = record["median_ns"]["manual"]
        memory.advance()
        for route in ROUTES:
            memory.observe(case, route, elapsed_ns=record["median_ns"][route],
                           reference_ns=manual, verified=record["gate"] == "PASS")
        if previous is not None:
            memory.trace(previous, case, memory.choose(case))
        previous = case
        train_records.append(record)
    training_ns = time.perf_counter_ns() - train_start
    compile_start = time.perf_counter_ns()
    frozen_predictions = [memory.choose(case) for case in test]
    policy_compile_ns = time.perf_counter_ns() - compile_start
    # No training updates below this point. Holdout timings are unknown above.
    holdout_records = []
    for case, chosen in zip(test, frozen_predictions):
        independent_routes = evaluate_routes(case, pairs=pairs)
        online = online_pair(case, chosen, memory, pairs=pairs)
        oracle = min(independent_routes["median_ns"], key=independent_routes["median_ns"].get)
        holdout_records.append({
            "state": case.__dict__, "predicted_route": chosen,
            "oracle_route_hindsight_only": oracle,
            "independent_controls": independent_routes,
            "online": online,
            "online_manual_over_ant": online["median_ns"]["manual"] / online["median_ns"]["ant"],
            "ant_over_oracle": online["median_ns"]["ant"] /
                               independent_routes["median_ns"][oracle],
        })
    snapshot = memory.snapshot()
    return {
        "pid": os.getpid(), "python": platform.python_version(),
        "platform": platform.platform(),
        "train_sizes": list(train_sizes), "holdout_sizes": list(holdout_sizes),
        "train_samples": train_records,
        "frozen_predictions": frozen_predictions,
        "holdout": holdout_records,
        "training_ns": training_ns,
        "policy_compile_ns": policy_compile_ns,
        "matrix_snapshot": snapshot,
        "matrix_json_bytes": len(json.dumps(snapshot, separators=(",", ":")).encode()),
    }


def collect(args):
    train, test = _csv(args.train_sizes), _csv(args.holdout_sizes)
    changes, reads = _csv(args.changes), _csv(args.reads)
    _strict_disjoint(train, test)
    if min(args.width, args.ticks, args.pairs, args.sessions) <= 0:
        raise ValueError("positive benchmark parameters required")
    if args.worker:
        return run_session(train_sizes=train, holdout_sizes=test, changes=changes,
                           reads=reads, width=args.width, ticks=args.ticks, pairs=args.pairs)
    sessions=[]
    for _ in range(args.sessions):
        cmd=[sys.executable,"-m","alexpy.ant_bench","--worker",
             "--train-sizes",args.train_sizes, "--holdout-sizes",args.holdout_sizes,
             "--changes",args.changes,"--reads",args.reads,
             "--width",str(args.width),"--ticks",str(args.ticks),"--pairs",str(args.pairs)]
        proc=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=180)
        sessions.append(json.loads(proc.stdout))
    summary=[]
    for i, case in enumerate(sessions[0]["holdout"]):
        rows=[s["holdout"][i] for s in sessions]
        if any(x["state"] != case["state"] for x in rows):
            raise AssertionError("holdout state mismatch across sessions")
        outcomes=[x["online_manual_over_ant"] for x in rows]
        summary.append({
            "state": case["state"],
            "predictions": [x["predicted_route"] for x in rows],
            "hindsight_oracles": [x["oracle_route_hindsight_only"] for x in rows],
            "manual_over_ant_median": statistics.median(outcomes),
            "manual_over_ant_range": [min(outcomes),max(outcomes)],
            "consistent_gain_10pct": all(x>=1.1 for x in outcomes),
            "consistent_regression_10pct": all(x<=1/1.1 for x in outcomes),
        })
    payload={
        "protocol": "ALEXPY-M14-ANT-MEMORY/0.1", "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "claim_scope":"opt-in sparse ant-inspired routing for exact signed-q partition DAG checksums",
        "default_promotion":"BLOCKED_EXPERIMENTAL",
        "holdout_protocol":"unseen array sizes, predictions frozen before any holdout timing",
        "measurement_boundary":"full workflow builds+patches+reads; online cached ant selection included",
        "guard":"all actions validated against dense reference, then independently timed",
        "controls":"fixed manual (strong), fixed partition, nondeployable hindsight oracle",
        "config":{"train_sizes":train,"holdout_sizes":test,"changes":changes,
                  "reads":reads,"width":args.width,"ticks":args.ticks,
                  "pairs":args.pairs,"sessions":args.sessions},
        "sessions":sessions,"summary":summary,
        "counts":{"gain_10pct_all_sessions":sum(r["consistent_gain_10pct"] for r in summary),
                  "regression_10pct_all_sessions":sum(r["consistent_regression_10pct"] for r in summary)},
    }
    payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes",default="4096,32768")
    parser.add_argument("--holdout-sizes",default="8192,16384")
    parser.add_argument("--changes",default="1,8")
    parser.add_argument("--reads",default="1,8")
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--ticks",type=int,default=4)
    parser.add_argument("--pairs",type=int,default=3)
    parser.add_argument("--sessions",type=int,default=3)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--json",type=Path)
    args=parser.parse_args(argv)
    result=collect(args)
    if args.worker:
        print(json.dumps(result,separators=(",", ":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"protocol":result["protocol"],"counts":result["counts"],
                          "summary":[{"size":x["state"]["size"],
                                      "changes":x["state"]["changes"],
                                      "reads":x["state"]["reads"],
                                      "routes":x["predictions"],
                                      "manual_over_ant":round(x["manual_over_ant_median"],3)}
                                     for x in result["summary"]]},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
