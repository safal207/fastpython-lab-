"""AlexPy M16: measured competing range-query plans and blind routing.

Training and held-out size dimensions are disjoint. No holdout latencies are
shown to the planner until AFTER its complete route table has been frozen.
Four full algorithms include initial construction, updates and all queries.
Real layout-switch setup cost is measured on training states only. No speedup
of the Python interpreter or actual light/sound propagation is claimed.
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

from .range_memory import RangeRouteMemory
from .range_routes import (ROUTES, Direct, RangeWorkload, checked_routes,
                            make_backend, make_case, run_workflow)

POLICIES = ("fixed_direct", "fixed_train", "simple_rule", "ant", "wave")


def _csv(value):
    nums = tuple(int(i.strip()) for i in value.split(",") if i.strip())
    if not nums or len(set(nums)) != len(nums) or any(x <= 0 for x in nums):
        raise ValueError("expected unique positive integer values")
    return nums


def grid(sizes, changes, queries, spans, ticks):
    return [RangeWorkload(n,k,q,ticks,span)
            for n in sizes for k in changes for q in queries for span in spans]


def measure_routes(w, pairs):
    case = make_case(w)
    expected = checked_routes(w, case=case)  # fail before any benchmark
    samples = {r: [] for r in ROUTES}
    for rep in range(pairs):
        order = list(ROUTES)
        shift = rep % len(order)
        order = order[shift:] + order[:shift]
        if rep % 2:
            order.reverse()
        for route in order:
            start = perf_counter_ns()
            got = run_workflow(route, w, case=case)
            elapsed = perf_counter_ns()-start
            if got != expected:
                raise AssertionError("result diverged during timing")
            samples[route].append(elapsed)
    return {"workload":w.__dict__,"checksum":str(expected),"gate":"PASS",
            "raw_ns":samples,"median_ns":{r:median(v) for r,v in samples.items()}}


def switch_probe(before, after, old_route, new_route):
    """Measure real cost of switching layout across a workload transition.

    Advance old runtime by one tick, retain its current Python data, rebuild
    a different runtime, apply next tick + read the new workload. All steps
    are verified against a fresh direct runtime from the SAME snapshot.
    """
    if before.size != after.size or before.changes != after.changes or before.span != after.span:
        raise ValueError("layout-switch probe requires same input domain")
    source, stages = make_case(before)
    backend = make_backend(old_route, source)
    patch, queries = stages[0]
    for index,value in patch:
        backend.update(index,value)
    for start,stop in queries:
        backend.query(start,stop)
    current = backend.data[:]  # input snapshot is outside the switch timer
    _,target_stages = make_case(after)
    target_patch,target_queries = target_stages[0]
    control = Direct(current)
    for index,value in target_patch:
        control.update(index,value)
    expected = sum(sum(control.query(a,b)) for a,b in target_queries)

    t = perf_counter_ns()
    new_backend = make_backend(new_route,current)
    setup_ns = perf_counter_ns()-t
    t = perf_counter_ns()
    for index,value in target_patch:
        new_backend.update(index,value)
    got = sum(sum(new_backend.query(a,b)) for a,b in target_queries)
    tick_ns = perf_counter_ns()-t
    if got != expected:
        raise AssertionError("layout transition failed direct witness")
    return {"from":before.__dict__,"to":after.__dict__,
            "old_route":old_route,"new_route":new_route,
            "setup_ns":setup_ns,"tick_ns":tick_ns,"verified":True}


def simple_rule(w):
    # Strong preregistered, zero-training heuristic, deliberately not tuned to holdout.
    if w.queries >= 16 and w.span == "wide":
        return "prefix"
    return "direct"


def run_session(*, train_sizes=(128,512,2048),holdout_sizes=(256,1024),
                changes=(1,8),queries=(1,8,32),spans=("short","wide"),
                ticks=4,pairs=5,probe=True):
    if (set(train_sizes) & set(holdout_sizes)
            or set(n.bit_length()-1 for n in train_sizes)
            & set(n.bit_length()-1 for n in holdout_sizes)):
        raise ValueError("train and holdout sizes or coarse cells overlap")
    if type(ticks) is not int or type(pairs) is not int or min(ticks,pairs) < 1:
        raise ValueError("ticks and pairs must be positive exact integers")
    if not spans or len(set(spans)) != len(spans) or any(s not in ("short","wide") for s in spans):
        raise ValueError("invalid spans")
    train = grid(train_sizes,changes,queries,spans,ticks)
    held = grid(holdout_sizes,changes,queries,spans,ticks)
    train_cells = {w.cell() for w in train}
    if train_cells & {w.cell() for w in held}:
        raise ValueError("holdout cell leaked into training")
    ant, wave = RangeRouteMemory(mode="ant"), RangeRouteMemory(mode="wave")
    time_start = perf_counter_ns()
    training = []
    for workload in train:
        record = measure_routes(workload,pairs=pairs)
        direct = record["median_ns"]["direct"]
        for planner in (ant,wave):
            planner.advance()
            for route in ROUTES:
                planner.observe(workload,route,
                                elapsed_ns=record["median_ns"][route],
                                direct_ns=direct,verified=record["gate"]=="PASS")
        training.append(record)
    training_ns = perf_counter_ns()-time_start

    switch_probes = []
    if probe and len(queries)>1:
        for n in train_sizes:
            for span in spans:
                for q0,q1 in zip(queries[:-1],queries[1:]):
                    before = RangeWorkload(n,changes[0],q0,ticks,span)
                    after = RangeWorkload(n,changes[0],q1,ticks,span)
                    for old_route,new_route in (("direct","prefix"),("prefix","direct"),
                                                ("blocks","fenwick"),("fenwick","blocks")):
                        edge = switch_probe(before,after,old_route,new_route)
                        for planner in (ant,wave):
                            planner.observe_switch(before,after,old_route,new_route,
                                setup_ns=edge["setup_ns"],tick_ns=edge["tick_ns"],
                                verified=edge["verified"])
                        switch_probes.append(edge)

    # The strongest fixed route is chosen exclusively from TRAIN timings.
    sums = {r:sum(row["median_ns"][r] for row in training) for r in ROUTES}
    fixed = min(ROUTES,key=lambda r:(sums[r],ROUTES.index(r)))
    compile_start = perf_counter_ns()
    policy = {}
    for w in held:
        policy[w.cell()] = {"fixed_direct":"direct","fixed_train":fixed,
                            "simple_rule":simple_rule(w),
                            "ant":ant.choose(w),"wave":wave.choose(w)}
    compile_ns = perf_counter_ns()-compile_start
    frozen = tuple(
        (cell,tuple(sorted(choices.items()))) for cell,choices in sorted(policy.items()))
    evidence_count = (len(ant.matrix),len(ant.transitions))
    assert not train_cells & set(policy)

    holdout = []
    for w in held:
        case = make_case(w)
        expected = checked_routes(w,case=case)  # correctness before timing
        choices = policy[w.cell()]
        raw = {p:[] for p in POLICIES}
        route_raw = {r:[] for r in ROUTES}
        for rep in range(pairs):
            route_order = list(ROUTES)
            route_order = route_order[rep%len(ROUTES):]+route_order[:rep%len(ROUTES)]
            if rep%2: route_order.reverse()
            for r in route_order:
                t=perf_counter_ns()
                got=run_workflow(r,w,case=case)
                latency=perf_counter_ns()-t
                if got!=expected: raise AssertionError("holdout execution mismatch")
                route_raw[r].append(latency)
            order = list(POLICIES)
            order = order[rep%len(POLICIES):]+order[:rep%len(POLICIES)]
            if rep%2:order.reverse()
            for p in order:
                t=perf_counter_ns()
                # This exact frozen-map lookup is INCLUDED in runtime timing.
                route = policy[w.cell()][p]
                got = run_workflow(route,w,case=case)
                latency=perf_counter_ns()-t
                if got!=expected or route != choices[p]:
                    raise AssertionError("plan selection/checksum drift")
                raw[p].append(latency)
        route_medians={r:median(v) for r,v in route_raw.items()}
        winner = min(ROUTES,key=lambda r:(route_medians[r],ROUTES.index(r)))
        medians = {p:median(raw[p]) for p in POLICIES}
        holdout.append({"workload":w.__dict__,"gate":"PASS", "checksum":str(expected),
                        "choices":choices,"oracle_route":winner,
                        "route_median_ns":route_medians,"route_raw_ns":route_raw,
                        "policy_median_ns":medians,"policy_raw_ns":raw,
                        "speedup_over_fixed_train":{
                            p:medians["fixed_train"]/medians[p] for p in POLICIES},
                        "policy_over_oracle":{p:medians[p]/route_medians[winner] for p in POLICIES}})
    if ((len(ant.matrix),len(ant.transitions)) != evidence_count or
            frozen != tuple((cell,tuple(sorted(choices.items())))
                     for cell,choices in sorted(policy.items()))):
        raise AssertionError("holdout affected training or decisions")
    return {"pid":os.getpid(),"python":platform.python_version(),
            "platform":platform.platform(),"train_sizes":list(train_sizes),
            "holdout_sizes":list(holdout_sizes),"training_ns":training_ns,
            "compile_ns":compile_ns,"training":training,"switch_probes":switch_probes,
            "fixed_train_route":fixed,"fixed_train_ns":sums,
            "memory":{"ant":ant.snapshot(),"wave":wave.snapshot()},
            "holdout":holdout,"no_holdout_training":True}


def collect(args):
    train,held=_csv(args.train_sizes),_csv(args.holdout_sizes)
    changes,queries=_csv(args.changes),_csv(args.queries)
    spans=tuple(v.strip() for v in args.spans.split(",") if v.strip())
    if min(args.ticks,args.pairs,args.sessions)<1:
        raise ValueError("positive counts required")
    kwargs=dict(train_sizes=train,holdout_sizes=held,changes=changes,queries=queries,
                spans=spans,ticks=args.ticks,pairs=args.pairs,probe=not args.no_probe)
    if args.worker:
        return run_session(**kwargs)
    sessions=[]
    for _ in range(args.sessions):
        command=[sys.executable,"-m","alexpy.range_bench","--worker",
                 "--train-sizes",args.train_sizes,"--holdout-sizes",args.holdout_sizes,
                 "--changes",args.changes,"--queries",args.queries,
                 "--spans",args.spans,"--ticks",str(args.ticks),
                 "--pairs",str(args.pairs)]
        if args.no_probe:command.append("--no-probe")
        sub=subprocess.run(command,capture_output=True,text=True,check=True,timeout=180)
        sessions.append(json.loads(sub.stdout))
    summary=[]
    for i,first in enumerate(sessions[0]["holdout"]):
        rows=[s["holdout"][i] for s in sessions]
        if any(r["workload"]!=first["workload"] or r["checksum"]!=first["checksum"]
               or r["gate"]!="PASS" for r in rows):
            raise AssertionError("cross-process drift or invalid checksum")
        summary.append({"workload":first["workload"],
                        "oracle_routes":[r["oracle_route"] for r in rows],
                        "selected":{p:[r["choices"][p] for r in rows] for p in POLICIES},
                        "speedup_over_fixed_train":{
                            p:{"median":median(r["speedup_over_fixed_train"][p] for r in rows),
                               "gained_gt_10pct_all":all(r["speedup_over_fixed_train"][p]>1.10 for r in rows),
                               "lost_gt_10pct_all":all(r["speedup_over_fixed_train"][p]<1/1.1 for r in rows)}
                            for p in POLICIES}})
    result={"protocol":"ALEXPY-M16-REAL-TRANSITION-COMPETITION/1",
            "utc":datetime.now(timezone.utc).isoformat(),"promotion":"BLOCKED_RESEARCH",
            "scope":"range sums and squared sums over owned exact-int data; separate from public sum_squares",
            "timing_boundary":"full construction+updates+queries+frozen policy lookup",
            "transition_provenance":"timed runtime reconstruction after train-only state change; not hardware cache counters",
            "config":{"train":list(train),"holdout":list(held),"changes":list(changes),
                      "queries":list(queries),"spans":list(spans),
                      "ticks":args.ticks,"pairs":args.pairs,"sessions":args.sessions},
            "summary":summary,"sessions":sessions}
    result["sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=True).encode()).hexdigest()
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sizes",default="128,512,2048")
    p.add_argument("--holdout-sizes",default="256,1024")
    p.add_argument("--changes",default="1,8")
    p.add_argument("--queries",default="1,8,32")
    p.add_argument("--spans",default="short,wide")
    p.add_argument("--ticks",type=int,default=4)
    p.add_argument("--pairs",type=int,default=5)
    p.add_argument("--sessions",type=int,default=3)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--no-probe",action="store_true",help="skip real transition probe")
    p.add_argument("--json",type=Path)
    args=p.parse_args(argv)
    result=collect(args)
    if args.worker:
        print(json.dumps(result,separators=(",",":")))
    else:
        if args.json:
            args.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"protocol":result["protocol"],"sessions":args.sessions,
            "heldout":len(result["summary"]),"fixed_train_route":result["sessions"][0]["fixed_train_route"],
            "winners":sorted(set(x for row in result["summary"] for x in row["oracle_routes"])),
            "selected_wave":sorted(set(x for row in result["summary"] for x in row["selected"]["wave"])),
            "transition_edges":result["sessions"][0]["memory"]["wave"]["real_switch_edges"]},indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
