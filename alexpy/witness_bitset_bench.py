"""AlexPy 2.8: three-way same-native QA witness comparison, no auto-promotion.

Frozen 2.7 synthetic density family is REUSED for regression analysis, not a
fresh statistical holdout. One real JUnit XML test report can be replayed
separately. Correctness scan is independent and outside timed regions.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from dataclasses import asdict

from .native_bitset import BitsetNativeQA
from .native_transition_tree import CTreeNativeQA
from .transition_tree import SetNativeQA
from .native_witness_stress import CASES, SMOKE_CASES, generate
from .qa_lab import FAIL, NOT_RUN, scan_reference
from .bardo_bridge_bench import digest_step, host

ROUTES = {"bitset": BitsetNativeQA, "c_tree": CTreeNativeQA, "set": SetNativeQA}


def process_rss_linux():
    """Current and high-water RSS in KiB from procfs, where available."""
    try:
        rows = Path("/proc/self/status").read_text().splitlines()
        data = {}
        for row in rows:
            if row.startswith(("VmRSS:", "VmHWM:")):
                key, value = row.split(":", 1)
                data[key] = int(value.strip().split()[0])
        return {"rss_kib": data.get("VmRSS"), "rss_hwm_kib": data.get("VmHWM")}
    except (FileNotFoundError, ValueError, OSError):
        return {"rss_kib": None, "rss_hwm_kib": None}


def replay(route, case, suites, events):
    engine = ROUTES[route](suites, case.suites)
    if not engine.native:
        raise RuntimeError("native QA required")
    digest = 0
    total_batches = (len(events) + case.batch - 1) // case.batch
    for bi, pos in enumerate(range(0, len(events), case.batch), 1):
        engine.apply_batch(events[pos:pos+case.batch])
        if bi % case.report_every == 0 or bi == total_batches:
            for _ in range(case.reads_per_report):
                digest = digest_step(digest, (engine.snapshot(), engine.witnesses()))
    return digest


def verify(case, suites, events):
    engines = {name: klass(suites, case.suites) for name, klass in ROUTES.items()}
    statuses = [NOT_RUN] * case.n
    durations = [0] * case.n
    batches = (len(events) + case.batch - 1) // case.batch
    for bi, pos in enumerate(range(0, len(events), case.batch), 1):
        batch = events[pos:pos+case.batch]
        for e in batch:
            statuses[e.test_id] = e.status
            durations[e.test_id] = e.duration_ms
        transitions = {name: engine.apply_batch(batch) for name, engine in engines.items()}
        if len(set(transitions.values())) != 1:
            raise AssertionError("transition mismatch")
        if bi % max(1,batches//5)==0 or bi==batches:
            expected = scan_reference(suites, case.suites, statuses, durations)
            witnesses = tuple(i for i, st in enumerate(statuses) if st==FAIL)[:4]
            for name, engine in engines.items():
                if engine.snapshot()!=expected or engine.witnesses()!=witnesses or not engine.verify():
                    raise AssertionError("failed independent oracle: " + name)
    digests = {name: replay(name,case,suites,events) for name in ROUTES}
    if len(set(digests.values()))!=1:
        raise AssertionError("report digest mismatch")
    return next(iter(digests.values()))


def measure(case, pairs=3):
    suites,events=generate(case)
    expected=verify(case,suites,events)
    samples={name: [] for name in ROUTES}
    names=list(ROUTES)
    for i in range(pairs):
        for name in names[i%3:]+names[:i%3]:
            t0=time.perf_counter_ns()
            digest=replay(name,case,suites,events)
            elapsed=time.perf_counter_ns()-t0
            if digest!=expected:raise AssertionError("timed digest mismatch")
            samples[name].append(elapsed)
    median={name:statistics.median(v) for name,v in samples.items()}
    return {"case":asdict(case),
            "digest":expected,"samples_ns":samples,"medians_ns":median,
            "bitset_vs_tree":median["c_tree"]/median["bitset"],
            "bitset_vs_set":median["set"]/median["bitset"]}


def suite(smoke=False,pairs=3):
    return {"schema":"alexpy-bardo-bitset-bench/1","host":host(),
            "experimental":True,"family":"2.7 reused, not fresh held-out data",
            "rows":[measure(case,pairs) for case in (SMOKE_CASES if smoke else CASES)]}


def memory_worker(case, route):
    suites,events=generate(case)
    baseline=process_rss_linux()
    tracemalloc.start()
    try:
        digest=replay(route,case,suites,events)
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    after=process_rss_linux()
    return {"case":case.name,"route":route,"digest":digest,
            "traced_peak_bytes":peak,"traced_current_bytes":current,
            "baseline":baseline,"after":after,
            "rss_delta_kib":None if after["rss_kib"] is None or baseline["rss_kib"] is None
                else after["rss_kib"]-baseline["rss_kib"],
            "hwm_delta_kib":None if after["rss_hwm_kib"] is None or baseline["rss_hwm_kib"] is None
                else after["rss_hwm_kib"]-baseline["rss_hwm_kib"],"host":host()}


def memory_suite(sessions=3,smoke=False):
    # Fully independent correctness checks are done in the parent *before*
    # each isolated memory worker; workers do not contaminate RSS with scans.
    cases=SMOKE_CASES if smoke else (CASES[1],CASES[4],CASES[5])
    rows=[]
    for case in cases:
        suites,events=generate(case)
        expected=verify(case,suites,events)
        for route in ROUTES:
            attempts=[]
            for _ in range(sessions):
                cmd=[sys.executable,"-m","alexpy.witness_bitset_bench","--memory-worker",route,
                     "--memory-case",case.name]
                p=subprocess.run(cmd,text=True,capture_output=True,check=True)
                row=json.loads(p.stdout)
                if row["digest"]!=expected:raise AssertionError("memory worker mismatch")
                attempts.append(row)
            rows.append({"case":case.name,"route":route,"sessions":attempts,
                         "median_peak_traced_bytes":statistics.median(r["traced_peak_bytes"] for r in attempts),
                         "median_rss_delta_kib":statistics.median(r["rss_delta_kib"] for r in attempts)
                         if all(r["rss_delta_kib"] is not None for r in attempts) else None})
    return {"schema":"alexpy-bardo-bitset-memory/1","host":host(),"rows":rows,
            "rss_note":"procfs snapshots, not continuous RSS profile; VmHWM high-water may predate workload"}


def run(smoke=False,pairs=3,sessions=3):
    if sessions<1 or pairs<1:raise ValueError("positive sessions and pairs required")
    if sessions==1:return suite(smoke,pairs)
    executions=[]
    for _ in range(sessions):
        cmd=[sys.executable,"-m","alexpy.witness_bitset_bench","--worker","--pairs",str(pairs)]
        if smoke:cmd.append("--smoke")
        p=subprocess.run(cmd,text=True,capture_output=True,check=True)
        executions.append(json.loads(p.stdout))
    return {"schema":"alexpy-bardo-bitset-bench/1","host":host(),"sessions":executions,
            "experimental":True,"default_promotion":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--pairs",type=int,default=3)
    parser.add_argument("--sessions",type=int,default=3)
    parser.add_argument("--memory",action="store_true")
    parser.add_argument("--memory-worker",choices=tuple(ROUTES))
    parser.add_argument("--memory-case",choices=tuple(c.name for c in CASES+SMOKE_CASES),default=CASES[1].name)
    parser.add_argument("--json",type=Path)
    args=parser.parse_args(argv)
    if args.sessions<1 or args.pairs<1:parser.error("sessions and pairs must be positive")
    if args.memory_worker:
        case=next(c for c in CASES+SMOKE_CASES if c.name==args.memory_case)
        result=memory_worker(case,args.memory_worker)
    elif args.memory:result=memory_suite(args.sessions,args.smoke)
    elif args.worker:result=suite(args.smoke,args.pairs)
    else:result=run(args.smoke,args.pairs,args.sessions)
    s=json.dumps(result,indent=2,ensure_ascii=False)
    if args.json:args.json.write_text(s+"\n",encoding="utf-8")
    print(s)


if __name__=="__main__":main()
