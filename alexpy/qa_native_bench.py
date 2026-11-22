"""End-to-end C QA batch comparison against strong optimized Python controls.

All payloads are synthetic. Preflight verifies final state and every report
boundary against independent full scans. Training/fixture generation is outside
measurement, engine construction, validation, slicing, snapshots inside.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .qa_native import NativeBatchQA
from .qa_batch import AlexPyBatchQAGraph, StrongPythonBatchQA
from .qa_lab import scan_reference, NOT_RUN
from .qa_batch_bench import Case, FULL_CASES, SMOKE_CASES, generate, _digest_step

ROUTES=("strong_batch", "alexpy_batch", "native_batch")
ENGINES={"strong_batch":StrongPythonBatchQA,
         "alexpy_batch":AlexPyBatchQAGraph,
         "native_batch":NativeBatchQA}
BATCHES=(1,8,64,256)


def replay(route,case,bs,suites,events):
    if route not in ENGINES:raise ValueError("unsupported route")
    eng=ENGINES[route](suites,case.suites)
    digest=0
    total_batches=(len(events)+bs-1)//bs
    for bi,start in enumerate(range(0,len(events),bs),1):
        eng.apply_batch(events[start:start+bs])
        if bi%case.report_every_batches==0 or bi==total_batches:
            for _ in range(case.reads_per_report):
                digest=_digest_step(digest,eng.snapshot())
    return digest


def verify(case,bs,suites,events):
    engines=[ENGINES[k](suites,case.suites) for k in ROUTES]
    statuses=[NOT_RUN]*len(suites);durations=[0]*len(suites)
    digest=0; nb=(len(events)+bs-1)//bs
    scan_every=max(1,nb//8)
    for bi,start in enumerate(range(0,len(events),bs),1):
        batch=events[start:start+bs]
        for event in batch:
            statuses[event.test_id]=event.status
            durations[event.test_id]=event.duration_ms
        transitions=[eng.apply_batch(batch) for eng in engines]
        if len(set(transitions))!=1:
            raise AssertionError(f"transition mismatch at batch={bi}")
        if bi%case.report_every_batches==0 or bi==nb:
            snaps=[eng.snapshot() for eng in engines]
            if snaps.count(snaps[0])!=len(snaps):
                raise AssertionError(f"report mismatch at batch={bi}")
            for _ in range(case.reads_per_report):digest=_digest_step(digest,snaps[0])
        if bi%scan_every==0 or bi==nb:
            reference=scan_reference(suites,case.suites,statuses,durations)
            if any(eng.snapshot()!=reference for eng in engines):
                raise AssertionError(f"full scan mismatch at batch={bi}")
    if not engines[1].verify() or not engines[2].verify():
        raise AssertionError("native/owned buffer check failed")
    for route in ROUTES:
        if replay(route,case,bs,suites,events)!=digest:
            raise AssertionError(f"timed replay witness diverged: {route}")
    return digest


def worker(pairs,smoke,batches):
    if NativeBatchQA([0],1).native is False:
        raise RuntimeError("compiled _qa_native missing: cannot benchmark fallback as C")
    cells=[]
    for case in SMOKE_CASES if smoke else FULL_CASES:
        suites,events=generate(case)
        for bs in batches:
            witness=verify(case,bs,suites,events)
            samples=[]
            for pair in range(pairs):
                shift=pair%len(ROUTES)
                order=ROUTES[shift:]+ROUTES[:shift]
                if pair%2:order=tuple(reversed(order))
                times={}
                for name in order:
                    t0=time.perf_counter_ns()
                    observed=replay(name,case,bs,suites,events)
                    elapsed=time.perf_counter_ns()-t0
                    if observed!=witness:raise AssertionError("digest divergence during timed replay")
                    times[name]=elapsed
                samples.append({"order":list(order),"ns":times,
                    "vs_python":times["strong_batch"]/times["native_batch"],
                    "vs_old_alexpy":times["alexpy_batch"]/times["native_batch"]})
            cells.append({"case":case.name,"tests":case.tests,"events":case.events,
                "batch_size":bs,"digest":witness,"pairs":samples,
                "median_vs_python":statistics.median(x["vs_python"] for x in samples),
                "median_vs_old_alexpy":statistics.median(x["vs_old_alexpy"] for x in samples)})
    return {"python":sys.version.split()[0],"platform":platform.platform(),
            "machine":platform.machine(),"native":True,"cells":cells}


def run(*,sessions=3,pairs=3,smoke=False,batches=BATCHES):
    if (type(sessions)is not int or sessions<1 or type(pairs)is not int or pairs<1
            or not batches or any(type(x)is not int or x<1 for x in batches)):
        raise ValueError("sessions, pairs, batches must be positive exact integers")
    sessions_data=[]
    for _ in range(sessions):
        args=[sys.executable,"-m","alexpy.qa_native_bench","--worker","--pairs",str(pairs),
              "--batch-sizes",",".join(map(str,batches))]
        if smoke:args.append("--smoke")
        cp=subprocess.run(args,capture_output=True,text=True,check=True)
        sessions_data.append(json.loads(cp.stdout))
    summary=[]
    for i,cell in enumerate(sessions_data[0]["cells"]):
        rp=[s["cells"][i]["median_vs_python"] for s in sessions_data]
        ra=[s["cells"][i]["median_vs_old_alexpy"] for s in sessions_data]
        summary.append({"case":cell["case"],"batch_size":cell["batch_size"],
                        "per_process_vs_python":rp,"median_vs_python":statistics.median(rp),
                        "all_processes_over_1_1":all(r>1.1 for r in rp),
                        "all_processes_below_0_9":all(r<0.9 for r in rp),
                        "per_process_vs_old_alexpy":ra,"median_vs_old_alexpy":statistics.median(ra)})
    return {"schema":"ALEXPY-QA-NATIVE/1", "type":"opt-in C extension",
            "source":"synthetic CI events; not production logs",
            "timing":"initialization + in-loop slicing, full validation, batches, reports and digest",
            "semantics":"last write per ID per batch; atomic invalid event rejection",
            "not_included":"fixture generation and correctness mirror (preflight)",
            "summary":summary,"sessions":sessions_data,"pairs":pairs,
            "session_count":sessions,"smoke":smoke}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions",type=int,default=3)
    ap.add_argument("--pairs",type=int,default=3)
    ap.add_argument("--batch-sizes",default="1,8,64,256")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    ap.add_argument("--json",type=Path)
    a=ap.parse_args(argv)
    batches=tuple(int(x) for x in a.batch_sizes.split(","))
    if a.worker:
        data=worker(a.pairs,a.smoke,batches)
    else:
        data=run(sessions=a.sessions,pairs=a.pairs,smoke=a.smoke,batches=batches)
    if a.json:
        a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if a.worker:
        print(json.dumps(data))
    else:
        for s in data["summary"]:
            print(f'{s["case"]} batch={s["batch_size"]}: {s["median_vs_python"]:.3f}x vs strong Python batch; {s["median_vs_old_alexpy"]:.3f}x vs AlexPy batch')
    return 0


if __name__=="__main__":raise SystemExit(main())
