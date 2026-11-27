"""Blind, fresh-process benchmark for the noise-aware AlexPy route planner.

Each training and each holdout worker is a separate interpreter. All holdout
choices are frozen BEFORE any worker receives a holdout case. Per-process
paired samples and setup cost are included; the offline training budget is
reported separately rather than pretending it is free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from .range_bench import grid, measure_routes, switch_probe
from .range_memory import RangeRouteMemory
from .range_routes import ROUTES, RangeWorkload, checked_routes, make_case, run_workflow
from .stable_memory import (StableRouteMemory, environment_key,
                             simple_rule, workload_key)

POLICIES = ("fixed_train", "simple_rule", "ant", "wave", "stable")


def _csv(s):
    result = tuple(int(x) for x in s.split(",") if x)
    if not result or len(set(result)) != len(result) or any(x < 1 for x in result):
        raise ValueError("expected nonempty unique positive integer list")
    return result


def _training_worker(args):
    rows = []
    start = perf_counter_ns()
    for w in grid(args.train_sizes, args.changes, args.queries, args.spans, args.ticks):
        rows.append(measure_routes(w, pairs=args.pairs))
    training_ns = perf_counter_ns() - start
    switches = []
    if args.probe:
        for size in args.train_sizes:
            for span in args.spans:
                for a, b in zip(args.queries[:-1], args.queries[1:]):
                    before = RangeWorkload(size,args.changes[0],a,args.ticks,span)
                    after = RangeWorkload(size,args.changes[0],b,args.ticks,span)
                    for old,new in (("direct","direct"),("direct","prefix"),
                                    ("blocks","blocks"),("blocks","fenwick")):
                        switches.append(switch_probe(before,after,old,new))
    total_ns = perf_counter_ns() - start
    return {"worker":"train", "pid":os.getpid(),"rows":rows,
            "switches":switches,"training_ns":training_ns,
            "transition_probe_ns":total_ns-training_ns,
            "total_calibration_ns":total_ns}


def _holdout_worker(args, data):
    choices=data["choices"]
    if data["environment"] != environment_key(device_tag=args.device_tag):
        raise AssertionError("cross-environment evaluation denied")
    rows=[]
    for w in grid(args.holdout_sizes,args.changes,args.queries,args.spans,args.ticks):
        case = make_case(w)
        expected = checked_routes(w,case=case)
        key = workload_key(w)
        locked = choices[key]
        raw_routes={r:[] for r in ROUTES}
        raw_policies={p:[] for p in POLICIES}
        for rep in range(args.pairs):
            order=list(ROUTES)
            order=order[rep%len(order):]+order[:rep%len(order)]
            if rep%2:order.reverse()
            for route in order:
                start=perf_counter_ns()
                got=run_workflow(route,w,case=case)
                elapsed=perf_counter_ns()-start
                if got != expected: raise AssertionError("route checksum drift")
                raw_routes[route].append(elapsed)
            order=list(POLICIES)
            order=order[rep%len(order):]+order[:rep%len(order)]
            if rep%2:order.reverse()
            for policy in order:
                start=perf_counter_ns()
                route=choices[key][policy]  # online lookup cost INCLUDED
                got=run_workflow(route,w,case=case)
                elapsed=perf_counter_ns()-start
                if route != locked[policy] or got != expected:
                    raise AssertionError("frozen plan or checksum changed")
                raw_policies[policy].append(elapsed)
        med_routes={r:median(v) for r,v in raw_routes.items()}
        med_policies={p:median(v) for p,v in raw_policies.items()}
        oracle=min(ROUTES,key=lambda r:(med_routes[r],ROUTES.index(r)))
        rows.append({"workload":w.__dict__,"gate":"PASS", "choices":locked,
                     "oracle_route":oracle,"raw_route_ns":raw_routes,
                     "raw_policy_ns":raw_policies,"median_route_ns":med_routes,
                     "median_policy_ns":med_policies,
                     "speedup_vs_simple":{p:med_policies["simple_rule"]/med_policies[p]
                                          for p in POLICIES}})
    return {"worker":"evaluate","pid":os.getpid(),"rows":rows}


def _worker_cmd(args, operation, data=None):
    cmd=[sys.executable,"-m","alexpy.stable_bench", "--worker",operation,
         "--train-sizes", ",".join(map(str,args.train_sizes)),
         "--holdout-sizes", ",".join(map(str,args.holdout_sizes)),
         "--changes", ",".join(map(str,args.changes)),
         "--queries", ",".join(map(str,args.queries)),
         "--ticks",str(args.ticks),"--pairs",str(args.pairs),
         "--device-tag",args.device_tag]
    if not args.probe:cmd.append("--no-probe")
    if args.spans==("short",):cmd.extend(("--spans","short"))
    if args.spans==("wide",):cmd.extend(("--spans","wide"))
    run=subprocess.run(cmd,input=json.dumps(data) if data is not None else None,
                       capture_output=True,text=True,check=True)
    return json.loads(run.stdout)


def _from_dict(d):
    return RangeWorkload(**d)


def run_experiment(args):
    train=grid(args.train_sizes,args.changes,args.queries,args.spans,args.ticks)
    holdout=grid(args.holdout_sizes,args.changes,args.queries,args.spans,args.ticks)
    if set(args.train_sizes) & set(args.holdout_sizes):
        raise ValueError("train and held-out sizes must be disjoint")
    # Coarse coordinate collisions are also forbidden as in M16.
    if {w.cell() for w in train} & {w.cell() for w in holdout}:
        raise ValueError("train and holdout state coordinates overlap")
    if args.sessions < 3:raise ValueError("at least three independent processes required")
    env=environment_key(device_tag=args.device_tag)
    train_workers=[_worker_cmd(args,"train") for _ in range(args.sessions)]
    # Sessions are launched as distinct subprocesses. PIDs may legally be
    # reused after termination, so unique session indices identify evidence.
    memory=StableRouteMemory(environment=env)
    ant,wave=RangeRouteMemory(mode="ant"), RangeRouteMemory(mode="wave")
    all_rows=[]
    for worker_id,worker in enumerate(train_workers):
        for row in worker["rows"]:
            w=_from_dict(row["workload"])
            memory.observe_session(w, process_id=f"train-{worker_id}",
                                   raw_ns=row["raw_ns"], verified=row["gate"]=="PASS",
                                   environment=env)
            direct=row["median_ns"]["direct"]
            for m in (ant,wave):
                m.advance()
                for route in ROUTES:
                    m.observe(w,route,elapsed_ns=row["median_ns"][route],
                              direct_ns=direct,verified=True)
            all_rows.append(row)
        for edge in worker["switches"]:
            memory.observe_switch(_from_dict(edge["from"]),_from_dict(edge["to"]),
                edge["old_route"],edge["new_route"],process_id=f"train-{worker_id}",
                setup_ns=int(edge["setup_ns"]),tick_ns=int(edge["tick_ns"]),
                verified=edge["verified"],environment=env)
    fixed = min(ROUTES,key=lambda r:(sum(row["median_ns"][r] for row in all_rows),ROUTES.index(r)))
    compile_start=perf_counter_ns()
    decisions=memory.compile(holdout,environment=env)
    choices={}
    for w in holdout:
        key=workload_key(w)
        choices[key]={"fixed_train":fixed,"simple_rule":simple_rule(w),
                      "ant":ant.choose(w),"wave":wave.choose(w),
                      "stable":decisions[key].route}
    compilation_ns=perf_counter_ns()-compile_start
    frozen_digest=hashlib.sha256(json.dumps(choices,sort_keys=True).encode()).hexdigest()
    # Holdout workers only receive the fixed choices, never training timings.
    eval_workers=[_worker_cmd(args,"eval",{"choices":choices,"environment":env})
                  for _ in range(args.sessions)]
    summary=[]
    for w in holdout:
        key=workload_key(w)
        rows=[next(row for row in worker["rows"] if workload_key(_from_dict(row["workload"]))==key)
              for worker in eval_workers]
        summary.append({"workload":w.__dict__,"chosen":choices[key],
                        "reason":decisions[key].reason,
                        "oracle_routes":[row["oracle_route"] for row in rows],
                        "median_speedup_vs_simple":{
                            p:median(row["speedup_vs_simple"][p] for row in rows)
                            for p in POLICIES},
                        "reliable_processes":decisions[key].independent_processes})
    return {"protocol":"ALEXPY-STABLE-ROUTER/1",
            "environment":env,"train_sizes":list(args.train_sizes),
            "holdout_sizes":list(args.holdout_sizes),"sessions_count":args.sessions,
            "pairs":args.pairs,"train_workers":train_workers,
            "training_ns_sum":sum(row["training_ns"] for row in train_workers),
            "transition_probe_ns_sum":sum(row["transition_probe_ns"] for row in train_workers),
            "total_calibration_ns_sum":sum(row["total_calibration_ns"] for row in train_workers),
            "compilation_ns":compilation_ns,
            "training_frozen_sha256":frozen_digest,
            "snapshot":memory.snapshot(),"fixed_train":fixed,
            "choices":choices,"eval_workers":eval_workers,"summary":summary}


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes",type=_csv,default=(128,512,2048))
    parser.add_argument("--holdout-sizes",type=_csv,default=(384,1536))
    parser.add_argument("--changes",type=_csv,default=(1,8))
    parser.add_argument("--queries",type=_csv,default=(1,8,32))
    parser.add_argument("--spans",choices=("both","short","wide"),default="both")
    parser.add_argument("--ticks",type=int,default=4)
    parser.add_argument("--pairs",type=int,default=5)
    parser.add_argument("--sessions",type=int,default=3)
    parser.add_argument("--device-tag",default="local-benchmark-device")
    parser.add_argument("--no-probe",action="store_true")
    parser.add_argument("--worker",choices=("train","eval"))
    parser.add_argument("--json",type=Path)
    args=parser.parse_args(argv)
    args.probe=not args.no_probe
    args.spans=("short","wide") if args.spans=="both" else (args.spans,)
    if args.ticks<1 or args.pairs<2 or args.sessions<1:
        parser.error("positive ticks, at least 2 paired observations and one session required")
    return args


def main(argv=None):
    args=parse(argv)
    if args.worker == "train":
        result=_training_worker(args)
    elif args.worker == "eval":
        result=_holdout_worker(args,json.load(sys.stdin))
    else:
        result=run_experiment(args)
    if args.worker:
        print(json.dumps(result,separators=(",",":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"protocol":result["protocol"],
                          "train_cells":result["snapshot"]["train_cells"],
                          "holdout_cases":len(result["summary"]),
                          "independent_processes":args.sessions,
                          "fixed_train":result["fixed_train"],
                          "selected":{p:sorted({row["chosen"][p] for row in result["summary"]})
                                      for p in POLICIES},
                          "policy_median_ratio_to_simple":{
                              p:round(median(row["median_speedup_vs_simple"][p]
                                             for row in result["summary"]),3)
                              for p in POLICIES},
                          "training_ns_sum":result["training_ns_sum"],
                          "transition_probe_ns_sum":result["transition_probe_ns_sum"],
                          "compilation_ns":result["compilation_ns"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
