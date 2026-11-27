"""Blind-length holdout: light/sound wave selector vs ant memory and manual.

All candidate plans are independently correctness-gated against dense before
training or timing. Test-state route decisions are frozen before any test
measurement; CPU time includes runtime cached decision. Offline training and
freezing costs are reported separately. No default promotion.
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
from .ant_bench import _csv, _strict_disjoint, evaluate_routes, prepare, workloads
from .partition_bench import workflow
from .wave_memory import WaveMemoryGraph


POLICIES=("manual", "ant", "light", "sound")


def _online(state: Workload, frozen: dict, systems: dict, *, pairs: int) -> dict:
    initial, patches, ref = prepare(state)
    duration={r:[] for r in POLICIES}
    for k in range(pairs):
        order=list(POLICIES)
        shift=k%len(order)
        order=order[shift:]+order[:shift]
        if k%2: order.reverse()
        for policy in order:
            start=time.perf_counter_ns()
            route=("manual" if policy=="manual" else systems[policy].choose(state))
            if route!=frozen[policy]:
                raise AssertionError("holdout plan was not frozen")
            result=workflow(route,initial,patches,state.reads,state.width)
            elapsed=time.perf_counter_ns()-start
            if result != ref:
                raise AssertionError("checksum mismatch in online wave comparison")
            duration[policy].append(elapsed)
    return {"raw_ns":duration,"median_ns":{k:statistics.median(v) for k,v in duration.items()}}


def run_session(*, train_sizes=(128,1024), holdout_sizes=(256,512),
                changes=(1,4), reads=(1,8), width=64,ticks=2,pairs=3):
    _strict_disjoint(train_sizes,holdout_sizes)
    if type(pairs) is not int or pairs<=0:
        raise ValueError("pairs must be positive exact int")
    train=workloads(train_sizes,changes,reads,width=width,ticks=ticks)
    held=workloads(holdout_sizes,changes,reads,width=width,ticks=ticks)
    if set(w.cell() for w in train).intersection(w.cell() for w in held):
        raise ValueError("holdout and training alias to same coarse cell")
    models={"ant":AntMemoryGraph(max_distance=4,neighbors=4),
            "light":WaveMemoryGraph(mode="light"),
            "sound":WaveMemoryGraph(mode="sound")}
    t0=time.perf_counter_ns()
    train_records=[]
    for w in train:
        rec=evaluate_routes(w,pairs=pairs)
        if rec["gate"]!="PASS":raise AssertionError("train equivalence failed")
        manual=rec["median_ns"]["manual"]
        for model in models.values():
            model.advance()
            for r in ROUTES:
                model.observe(w,r,elapsed_ns=rec["median_ns"][r],
                              reference_ns=manual,verified=True)
        train_records.append(rec)
    training_ns=time.perf_counter_ns()-t0
    t0=time.perf_counter_ns()
    frozen=[]
    for w in held:
        frozen.append({"manual":"manual",**{name:model.choose(w) for name,model in models.items()}})
    freeze_ns=time.perf_counter_ns()-t0
    # No observes are performed after this line. Completely blind holdout.
    hold=[]
    for w,plans in zip(held,frozen):
        control=evaluate_routes(w,pairs=pairs)
        online=_online(w,plans,models,pairs=pairs)
        oracle=min(control["median_ns"],key=control["median_ns"].get)
        hold.append({"state":w.__dict__,"frozen":plans,
                     "hindsight_oracle_only":oracle,
                     "independent_route_controls":control,"online":online,
                     "manual_over":{p:online["median_ns"]["manual"] / online["median_ns"][p]
                                    for p in POLICIES[1:]}})
    return {"pid":os.getpid(),"python":platform.python_version(),
            "platform":platform.platform(),"training_ns":training_ns,
            "freeze_ns":freeze_ns,"training":train_records,"holdout":hold,
            "memory":{name:model.snapshot() for name,model in models.items()}}


def collect(args):
    train,holdout=_csv(args.train_sizes),_csv(args.holdout_sizes)
    change,reads=_csv(args.changes),_csv(args.reads)
    _strict_disjoint(train,holdout)
    if min(args.width,args.ticks,args.pairs,args.sessions)<=0:
        raise ValueError("parameters must be positive")
    if args.worker:
        return run_session(train_sizes=train,holdout_sizes=holdout,changes=change,
                           reads=reads,width=args.width,ticks=args.ticks,pairs=args.pairs)
    sessions=[]
    for _ in range(args.sessions):
        command=[sys.executable,"-m","alexpy.wave_bench","--worker",
                 "--train-sizes",args.train_sizes,"--holdout-sizes",args.holdout_sizes,
                 "--changes",args.changes,"--reads",args.reads,
                 "--width",str(args.width),"--ticks",str(args.ticks),
                 "--pairs",str(args.pairs)]
        proc=subprocess.run(command,text=True,capture_output=True,
                            check=True,timeout=120)
        sessions.append(json.loads(proc.stdout))
    summary=[]
    for i,original in enumerate(sessions[0]["holdout"]):
        rows=[s["holdout"][i] for s in sessions]
        if any(row["state"]!=original["state"] for row in rows):
            raise AssertionError("session state order drift")
        outcome={}
        for policy in POLICIES[1:]:
            ratios=[r["manual_over"][policy] for r in rows]
            outcome[policy]={"median_manual_over":statistics.median(ratios),
                             "all_processes_gain_10pct":all(x>1.10 for x in ratios),
                             "all_processes_regression_10pct":all(x<1/1.1 for x in ratios),
                             "selected":[r["frozen"][policy] for r in rows]}
        summary.append({"state":original["state"],"results":outcome})
    report={"protocol":"ALEXPY-M15-WAVE-MEMORY/1",
            "utc":datetime.now(timezone.utc).isoformat(),
            "scope":"opt-in offline verified-plan router; not physical wave simulation",
            "promotion":"BLOCKED_EXPERIMENTAL",
            "holdout":"unseen lengths; all decisions frozen before test route timing",
            "controls":"fixed manual, ant, hindsight oracle; checksum dense reference",
            "config":{"train_sizes":train,"holdout_sizes":holdout,
                      "changes":change,"reads":reads,"width":args.width,
                      "ticks":args.ticks,"pairs":args.pairs,"sessions":args.sessions},
            "summary":summary,"sessions":sessions}
    report["sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,default=list).encode()).hexdigest()
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sizes",default="128,1024")
    p.add_argument("--holdout-sizes",default="256,512")
    p.add_argument("--changes",default="1,4")
    p.add_argument("--reads",default="1,8")
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--ticks",type=int,default=2)
    p.add_argument("--pairs",type=int,default=3)
    p.add_argument("--sessions",type=int,default=3)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--json",type=Path)
    args=p.parse_args(argv)
    result=collect(args)
    if args.worker:
        print(json.dumps(result,separators=(",",":")))
    else:
        if args.json:args.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"protocol":result["protocol"],
                          "summary":[{"size":r["state"]["size"],
                                      "changes":r["state"]["changes"],
                                      "reads":r["state"]["reads"],
                                      **{k:round(v["median_manual_over"],3)
                                         for k,v in r["results"].items()}}
                                     for r in result["summary"]]},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
