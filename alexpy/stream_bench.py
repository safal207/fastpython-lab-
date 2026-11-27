"""Long-lived streams: measured calibration break-even and cross-host probe.

Only a profile-specific, frozen routing table is evaluated on unseen sizes.
Training uses three or more independent interpreter processes, not repetitions
of one process. Online timing includes persistent-state updates, all queries,
per-segment route lookup, true data-layout reconstruction and initial setup.
Offline calibration + compilation is added to net economics. Matrix reports
are advisory; no route promotion. Separate CI runners train their own models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from .range_bench import grid, measure_routes
from .range_routes import ROUTES, RangeWorkload, make_case
from .stable_memory import StableRouteMemory, environment_key, workload_key
from .stream_economics import stream_schedule, run_stream, witness


def csv_positive(value):
    result = tuple(int(i) for i in value.split(",") if i)
    if not result or len(result) != len(set(result)) or any(i < 1 for i in result):
        raise ValueError("comma-separated unique positive integers required")
    return result


def host_metadata():
    cpu = platform.processor() or platform.uname().processor or "not-reported"
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": cpu, "python": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "logical_cpus": os.cpu_count(), "ci": bool(os.getenv("CI")),
            "github_runner_os": os.getenv("RUNNER_OS"),
            "github_runner_arch": os.getenv("RUNNER_ARCH")}


def cmd(args, worker, data=None):
    command = [sys.executable, "-m", "alexpy.stream_bench", "--worker", worker,
               "--train-sizes", ",".join(map(str,args.train_sizes)),
               "--holdout-sizes", ",".join(map(str,args.holdout_sizes)),
               "--changes", ",".join(map(str,args.changes)),
               "--queries", ",".join(map(str,args.queries)),
               "--spans", "both" if args.spans == ("short", "wide") else args.spans[0], "--ticks", str(args.ticks),
               "--pairs", str(args.pairs), "--cycles", ",".join(map(str,args.cycles)),
               "--reps", str(args.reps), "--device-tag", args.device_tag]
    start = perf_counter_ns()
    result = subprocess.run(command, input=json.dumps(data) if data is not None else None,
                            text=True, capture_output=True, check=True, timeout=180)
    wall_ns = perf_counter_ns()-start
    return json.loads(result.stdout), wall_ns


def train_worker(args):
    rows = [measure_routes(w, pairs=args.pairs)
            for w in grid(args.train_sizes,args.changes,args.queries,args.spans,args.ticks)]
    return {"worker": "train", "pid": os.getpid(), "host": host_metadata(),
            "environment": environment_key(device_tag=args.device_tag), "rows": rows}


def evaluate_worker(args, data):
    env = environment_key(device_tag=args.device_tag)
    if data["environment"] != env:
        raise AssertionError("cannot reuse trained decisions in another profile")
    records = []
    # Plans already frozen; test process has no training-time values.
    for size in args.holdout_sizes:
        schedule = stream_schedule(size, changes=args.changes, queries=args.queries,
                                   spans=args.spans, ticks=args.ticks)
        # Input generation is common to all policies and not charged to any
        # runtime; it must not obscure the layout and query cost.
        case_cache = {workload_key(w):make_case(w) for w in schedule}
        for cycles in args.cycles:
            expected = witness(schedule, cycles=cycles, case_cache=case_cache)
            raw = {policy: [] for policy in ("simple", "stable", "direct")}
            details = {policy: [] for policy in raw}
            for rep in range(args.reps):
                order = ["simple", "stable", "direct"]
                order = order[rep%3:] + order[:rep%3]
                if rep%2:
                    order.reverse()
                for policy in order:
                    start = perf_counter_ns()
                    result = run_stream(schedule, cycles=cycles, policy=policy,
                                        choices=data["choices"], case_cache=case_cache)
                    elapsed = perf_counter_ns() - start
                    if result.checksum != expected:
                        raise AssertionError("full stream result mismatches independent witness")
                    raw[policy].append(elapsed)
                    details[policy].append({"setup_ns": result.setup_ns,
                                            "switch_ns": result.switch_ns,
                                            "switches": result.switches,
                                            "constructed": result.constructed,
                                            "segments": result.segments,
                                            "trace_sha256": hashlib.sha256(
                                                ",".join(result.route_trace).encode()).hexdigest()})
            records.append({"size":size,"cycles":cycles,"segments_per_cycle":len(schedule),
                            "gate":"PASS","raw_ns":raw,"details":details,
                            "median_ns":{p:median(v) for p,v in raw.items()},
                            "checksum_sha256":hashlib.sha256(str(expected).encode()).hexdigest()})
    return {"worker":"eval", "pid":os.getpid(), "host":host_metadata(),
            "environment":env,"records":records}


def valid_splits(args):
    training = grid(args.train_sizes,args.changes,args.queries,args.spans,args.ticks)
    heldout = grid(args.holdout_sizes,args.changes,args.queries,args.spans,args.ticks)
    if (set(args.train_sizes) & set(args.holdout_sizes) or
            {w.cell() for w in training} & {w.cell() for w in heldout}):
        raise ValueError("training/holdout sizes or coarse cells overlap")
    if any(ch > min(args.train_sizes + args.holdout_sizes) for ch in args.changes):
        raise ValueError("update count exceeds input size")
    return training, heldout


def experiment(args):
    training, heldout = valid_splits(args)
    if args.sessions < 3:
        raise ValueError("independent sessions must be >= 3")
    env = environment_key(device_tag=args.device_tag)
    wall_start = perf_counter_ns()
    workers_and_time = [cmd(args,"train") for _ in range(args.sessions)]
    memory = StableRouteMemory(environment=env)
    for process, (worker, elapsed) in enumerate(workers_and_time):
        if worker["environment"] != env:
            raise AssertionError("training host/profile mismatch")
        for row in worker["rows"]:
            w = RangeWorkload(**row["workload"])
            memory.observe_session(w,process_id=f"independent-{process}",
                                   raw_ns=row["raw_ns"],verified=row["gate"]=="PASS",
                                   environment=env)
    compile_started = perf_counter_ns()
    decisions = memory.compile(heldout, environment=env)
    choices = {workload_key(w): decisions[workload_key(w)].route for w in heldout}
    compilation_ns = perf_counter_ns() - compile_started
    calibration_ns = perf_counter_ns()-wall_start
    if any(workload_key(w) in {workload_key(x) for x in training} for w in heldout):
        raise AssertionError("training evidence leaked into holdout")
    freeze_hash = hashlib.sha256(json.dumps(choices,sort_keys=True).encode()).hexdigest()
    # Independent, blind evaluation workers receive ONLY frozen route choices.
    evaluated = [cmd(args,"eval",{"choices":choices,"environment":env}) for _ in range(args.sessions)]
    per_case = []
    for size in args.holdout_sizes:
        for cycles in args.cycles:
            rows = [next(x for x in worker["records"]
                         if x["size"]==size and x["cycles"]==cycles)
                    for worker, _ in evaluated]
            simple = median(x["median_ns"]["simple"] for x in rows)
            stable = median(x["median_ns"]["stable"] for x in rows)
            gains = [x["median_ns"]["simple"]-x["median_ns"]["stable"] for x in rows]
            per_case.append({"size": size, "cycles": cycles,
                "median_simple_ns":simple,"median_stable_ns":stable,
                "median_online_speedup":simple/stable,
                "gains_ns_by_process":gains,
                "all_processes_online_gain":all(x>0 for x in gains),
                "stable_switches": [median(d["switches"] for d in x["details"]["stable"])
                                    for x in rows],
                "simple_switches": [median(d["switches"] for d in x["details"]["simple"])
                                    for x in rows],
                "median_stable_switch_ns":median(median(d["switch_ns"] for d in x["details"]["stable"])
                                                  for x in rows),
                "reason_by_workload": sorted({decisions[workload_key(w)].reason
                    for w in heldout if w.size==size})})
    pooled=[]
    for cycles in args.cycles:
        subset=[x for x in per_case if x["cycles"]==cycles]
        online_savings=sum(x["median_simple_ns"]-x["median_stable_ns"] for x in subset)
        all_processes_gain=all(sum(
            next(z for z in worker["records"] if z["size"]==n and z["cycles"]==cycles)["median_ns"]["simple"]-
            next(z for z in worker["records"] if z["size"]==n and z["cycles"]==cycles)["median_ns"]["stable"]
            for n in args.holdout_sizes)>0 for worker,_ in evaluated)
        pooled.append({"cycles":cycles,"online_savings_ns":online_savings,
                       "calibration_ns":calibration_ns,
                       "net_savings_ns":online_savings-calibration_ns,
                       "observed_paid_break_even":online_savings>=calibration_ns and all_processes_gain,
                       "all_processes_online_gain":all_processes_gain})
    observed = next((x["cycles"] for x in pooled if x["observed_paid_break_even"]),None)
    # Strictly an extrapolation using the largest OBSERVED horizon. This is
    # not an empirically established amortization threshold: workloads,
    # frequency, clocks and layout allocations may change at larger horizons.
    last = pooled[-1]
    per_cycle = last["online_savings_ns"] / last["cycles"]
    projection = (math.ceil(calibration_ns / per_cycle) if per_cycle > 0
                  and last["all_processes_online_gain"] else None)
    return {"schema":"ALEXPY-STREAM-ECONOMICS/1", "created_utc":datetime.now(timezone.utc).isoformat(),
            "environment":env,"host":host_metadata(),"config":{
                "train_sizes":args.train_sizes,"holdout_sizes":args.holdout_sizes,
                "changes":args.changes,"queries":args.queries,"spans":args.spans,
                "ticks":args.ticks,"pairs":args.pairs,"sessions":args.sessions,
                "cycles":args.cycles,"reps":args.reps},
            "route_choices":choices,"frozen_sha256":freeze_hash,
            "calibration_wall_ns":calibration_ns,"compilation_ns":compilation_ns,
            "training_workers":[{**worker,"subprocess_wall_ns":elapsed}
                                for worker,elapsed in workers_and_time],
            "evaluation_workers":[{**worker,"subprocess_wall_ns":elapsed}
                                  for worker,elapsed in evaluated],
            "cases":per_case,"pooled":pooled,"observed_break_even_cycles":observed,
            "exploratory_linear_projection_cycles":projection,
            "notes": ["Offline calibration is paid once in each net comparison; evaluation process setup excluded from both policies.",
              "Break-even requires observed net savings and positive online savings in every independent process.",
              "No claim is made for another physical device without that host's own calibration and held-out timings.",
              "Training and holdout sizes and coarse state cells do not overlap.",
              "Every timed stream compares its checksum to an independent Direct witness."]}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sizes",type=csv_positive,default=(128,512,2048))
    p.add_argument("--holdout-sizes",type=csv_positive,default=(384,1536))
    p.add_argument("--changes",type=csv_positive,default=(1,8))
    p.add_argument("--queries",type=csv_positive,default=(1,8,32))
    p.add_argument("--spans",choices=("short","wide","both"),default="both")
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--pairs",type=int,default=3)
    p.add_argument("--sessions",type=int,default=3)
    p.add_argument("--cycles",type=csv_positive,default=(1,8,32,128))
    p.add_argument("--reps",type=int,default=3)
    p.add_argument("--device-tag",default="local-stream-lab")
    p.add_argument("--json",type=Path)
    p.add_argument("--worker",choices=("train","eval"))
    a=p.parse_args(argv)
    if a.ticks<1 or a.pairs<2 or a.reps<2 or a.sessions<1:
        p.error("ticks >= 1, paired repetitions >=2, sessions >=1 required")
    a.spans=("short","wide") if a.spans=="both" else (a.spans,)
    return a


def main(argv=None):
    args=parse(argv)
    if args.worker=="train":
        result=train_worker(args)
    elif args.worker=="eval":
        result=evaluate_worker(args,json.load(sys.stdin))
    else:
        result=experiment(args)
    if args.worker:
        print(json.dumps(result,separators=(",",":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"schema":result["schema"],"host":result["host"],
            "calibration_ms":round(result["calibration_wall_ns"]/1e6,2),
            "choices":{route:sum(x==route for x in result["route_choices"].values())
                       for route in ROUTES},
            "cases":[{"size":row["size"],"cycles":row["cycles"],
                       "online_speedup":round(row["median_online_speedup"],3)}
                       for row in result["cases"]],
            "observed_paid_break_even_cycles":result["observed_break_even_cycles"],
            "exploratory_linear_projection_cycles":result["exploratory_linear_projection_cycles"]},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
