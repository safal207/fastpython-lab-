"""Equal-native witness-tree comparison; no automatic route promotion.

Separate interpreter sessions, alternating presentation order, correctness
verification before timing, including report materialization and setup.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from .bardo_bridge_bench import CASES, SMOKE_CASES, digest_step, host
from .qa_batch_bench import generate
from .qa_lab import NOT_RUN, FAIL, scan_reference
from .qa_native import NativeBatchQA
from .transition_tree import SetNativeQA, TreeNativeQA
from .native_transition_tree import CTreeNativeQA

ROUTES={"c_tree":CTreeNativeQA,"set":SetNativeQA,"python_tree":TreeNativeQA}


def replay(route, case, batch, suites, events):
    engine=ROUTES[route](suites,case.suites)
    if not engine.native:raise RuntimeError("compiled _qa_native must be available")
    h=0
    total=(len(events)+batch-1)//batch
    for j,start in enumerate(range(0,len(events),batch),1):
        engine.apply_batch(events[start:start+batch])
        if j%case.report_every_batches==0 or j==total:
            for _ in range(case.reads_per_report):
                h=digest_step(h,(engine.snapshot(),engine.witnesses()))
    return h


def check_case(case,batch,suites,events):
    engines=[ROUTES[r](suites,case.suites) for r in ROUTES]
    n=len(suites);st=[NOT_RUN]*n;d=[0]*n
    nb=(len(events)+batch-1)//batch
    for j,start in enumerate(range(0,len(events),batch),1):
        part=events[start:start+batch]
        for event in part:
            st[event.test_id]=event.status;d[event.test_id]=event.duration_ms
        transitions=[e.apply_batch(part) for e in engines]
        if not all(x==transitions[0] for x in transitions):raise AssertionError("transition mismatch")
        if j%max(1,nb//6)==0 or j==nb:
            expected=scan_reference(suites,case.suites,st,d)
            witnesses=tuple(i for i,s in enumerate(st) if s==FAIL)[:4]
            if any(e.snapshot()!=expected or e.witnesses()!=witnesses for e in engines):
                raise AssertionError("independent QA snapshot/witness mismatch")
    if not all(e.verify() for e in engines):raise AssertionError("mirror structural check failed")
    digests=[replay(r,case,batch,suites,events) for r in ROUTES]
    if len(set(digests))!=1:raise AssertionError("digest mismatch")
    return digests[0]


def measure_case(case,batch,pairs,include_python_tree=True):
    suites,events=generate(case)
    digest=check_case(case,batch,suites,events)
    names=list(ROUTES if include_python_tree else ("c_tree","set"))
    samples={name:[] for name in names}
    for j in range(pairs):
        order=names[j%len(names):]+names[:j%len(names)]
        for name in order:
            t=time.perf_counter_ns()
            actual=replay(name,case,batch,suites,events)
            samples[name].append(time.perf_counter_ns()-t)
            if actual!=digest:raise AssertionError("timing digest mismatch")
    med={name:statistics.median(ns) for name,ns in samples.items()}
    return {"case":case.name,"batch":batch,"tests":case.tests,"events":case.events,
            "pairs":pairs,"samples_ns":samples,"median_ns":med,"digest":digest,
            "c_tree_vs_set":med["set"]/med["c_tree"],
            "python_tree_vs_c_tree":med.get("python_tree",float("nan"))/med["c_tree"]}


def memory(route,case_name):
    case,batch=next((c,b) for c,b in CASES if c.name==case_name)
    suites,events=generate(case)
    # Correctness check occurs before tracing and separately from the timed path.
    check_case(case,batch,suites,events)
    tracemalloc.start()
    try:
        digest=replay(route,case,batch,suites,events)
        current,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    return {"schema":"alexpy-bardo-native-memory/1","route":route,"case":case.name,
            "current_bytes":current,"peak_traced_bytes":peak,"digest":digest,"host":host()}


def suite(smoke=False,pairs=3):
    rows=[measure_case(case,batch,pairs) for case,batch in (SMOKE_CASES if smoke else CASES)]
    return {"schema":"alexpy-bardo-native-witness/1","host":host(),"rows":rows,
            "comparison":"SAME native QA accumulator; Python set vs Python tree vs C tree",
            "notes":"No tree promotion; synthetic QA stream; all processing and reports timed"}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--pairs",type=int,default=3)
    parser.add_argument("--sessions",type=int,default=1)
    parser.add_argument("--memory-worker",choices=list(ROUTES))
    parser.add_argument("--memory-case",default="medium-sparse")
    parser.add_argument("--json",type=Path)
    args=parser.parse_args(argv)
    if args.pairs<=0 or args.sessions<=0:parser.error("positive pairs/sessions required")
    if args.memory_worker:result=memory(args.memory_worker,args.memory_case)
    elif args.sessions==1:result=suite(args.smoke,args.pairs)
    else:
        command=[sys.executable,"-m","alexpy.native_tree_bench","--pairs",str(args.pairs)]
        if args.smoke:command.append("--smoke")
        runs=[]
        for _ in range(args.sessions):
            p=subprocess.run(command,capture_output=True,text=True,check=True)
            runs.append(json.loads(p.stdout))
        result={"schema":"alexpy-bardo-native-witness/1","sessions":runs,"host":host()}
    output=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.json:args.json.write_text(output+"\n",encoding="utf-8")
    print(output)
if __name__=="__main__":main()
