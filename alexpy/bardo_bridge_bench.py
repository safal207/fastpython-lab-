"""Predeclared equal-native QA witness experiments plus calibration probes.

Never label the symbolic trust simulation a CPU performance measurement.
Each route uses the same native QA engine. End-to-end includes construction,
event slicing, validation, snapshots, and witness extraction.
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

from .qa_batch_bench import Case, generate
from .qa_lab import NOT_RUN, FAIL, scan_reference
from .transition_tree import TreeNativeQA, SetNativeQA
from .calibration_gate import TrustEvidence, probe_decision

ROUTES = ("native_set", "native_tree")
ENGINES = {"native_set": SetNativeQA, "native_tree": TreeNativeQA}

CASES = (
    (Case("small-hot", 256, 8, 1800, 48, 1, 1), 8),
    (Case("medium-sparse", 4096, 32, 3200, 4096, 4, 1), 64),
    (Case("large-sparse", 16384, 128, 4500, 16384, 8, 1), 64),
    (Case("query-heavy", 4096, 32, 3200, 4096, 1, 8), 64),
)
SMOKE_CASES = CASES[:2]


def digest_step(digest: int, state) -> int:
    # Stable within process; both candidates produce identical state tuples.
    return (digest * 1000003 + hash(state)) & ((1 << 64) - 1)


def replay(route, case, batch, suites, events):
    engine=ENGINES[route](suites,case.suites)
    if not engine.native:
        raise RuntimeError("Benchmark must use the compiled native QA extension")
    digest=0
    batches=(len(events)+batch-1)//batch
    for j,start in enumerate(range(0,len(events),batch),1):
        engine.apply_batch(events[start:start+batch])
        if j%case.report_every_batches==0 or j==batches:
            for _ in range(case.reads_per_report):
                digest=digest_step(digest,(engine.snapshot(),engine.witnesses()))
    return digest


def verify(case,batch,suites,events):
    engines=(SetNativeQA(suites,case.suites), TreeNativeQA(suites,case.suites))
    status=[NOT_RUN]*len(suites);duration=[0]*len(suites)
    nb=(len(events)+batch-1)//batch
    for j,start in enumerate(range(0,len(events),batch),1):
        payload=events[start:start+batch]
        for ev in payload:
            status[ev.test_id]=ev.status
            duration[ev.test_id]=ev.duration_ms
        a,b=(e.apply_batch(payload) for e in engines)
        if a!=b or engines[0].witnesses()!=engines[1].witnesses():
            raise AssertionError(f"transition/witness mismatch at batch {j}")
        if j%max(1,nb//6)==0 or j==nb:
            expected=scan_reference(suites,case.suites,status,duration)
            witnesses=tuple(i for i,s in enumerate(status) if s==FAIL)[:4]
            if any(e.snapshot()!=expected or e.witnesses()!=witnesses for e in engines):
                raise AssertionError(f"independent reference mismatch at batch {j}")
    if not all(e.verify() for e in engines):
        raise AssertionError("mirror verification failed")
    digests=[replay(key,case,batch,suites,events) for key in ROUTES]
    if digests[0]!=digests[1]:
        raise AssertionError("end-to-end digest mismatch")
    return digests[0]


def measure(case,batch,pairs):
    suites,events=generate(case)
    verify(case,batch,suites,events)
    samples={r:[] for r in ROUTES}
    for i in range(pairs):
        for name in (ROUTES if i%2==0 else tuple(reversed(ROUTES))):
            begin=time.perf_counter_ns()
            digest=replay(name,case,batch,suites,events)
            samples[name].append(time.perf_counter_ns()-begin)
    medians={k:statistics.median(v) for k,v in samples.items()}
    return {"case":case.name,"tests":case.tests,"events":case.events,
            "batch":batch,"reports":(len(events)+batch-1)//batch//case.report_every_batches,
            "pairs":pairs,"samples_ns":samples,"medians_ns":medians,
            "tree_vs_set":medians["native_set"]/medians["native_tree"],
            "digest":digest}


def host():
    cpu=platform.processor()
    if platform.system()=="Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name") and ":" in line:
                    cpu=line.split(":",1)[1].strip()
                    break
        except OSError:
            pass
    return {"system":platform.system(),"python":platform.python_version(),
            "machine":platform.machine(),"processor":cpu}


def simulate(policy, pattern, ticks=1024, probe_cost=2., benefit=3., stale_loss=7.):
    """Synthetic monetary/time-unit *decision* costs, not measured CPU time.

    A hidden change affects only evaluation. The policy can observe current
    truth solely by paying an explicit probe, never by reading future labels.
    """
    observed_route=True
    last_probe=0
    nprobes=0
    loss=0.
    probe_next=0
    observed_count=256
    flips_seen=0
    for tick in range(ticks):
        # Hidden environment, inaccessible to the policy except via probe.
        actual = pattern=="stable" or (pattern=="single" and tick<512) or (
            pattern=="reversal" and (tick<512 or tick>=768))
        if policy=="fixed8": due=tick%8==0
        elif policy=="fixed64":due=tick%64==0
        elif policy=="every":due=True
        elif policy=="adaptive":due=tick>=probe_next
        else:raise ValueError("policy")
        if due:
            nprobes+=1
            loss+=probe_cost
            if observed_route!=actual:
                flips_seen+=1
            observed_route=actual
            last_probe=tick
            observed_count+=1
            if policy=="adaptive":
                # Only observed flips are known. On clean recent history, age
                # makes trust expire. The independent semantic gate remains.
                ev=TrustEvidence(observed_count,0,min(1.,flips_seen/8),0.)
                probe_next=tick+probe_decision(ev,probe_cost=probe_cost,miss_loss=stale_loss,
                                               min_interval=8,max_interval=128).interval
        if policy=="adaptive":
            ev=TrustEvidence(observed_count,tick-last_probe,min(1.,flips_seen/8),0.)
            use_fast=observed_route and not probe_decision(ev,probe_cost=probe_cost,
                                                           miss_loss=stale_loss,
                                                           min_interval=8,max_interval=128).fallback
        else:
            use_fast=observed_route
        if use_fast:
            loss+=(-benefit if actual else stale_loss)
        # safe reference route incurs zero marginal loss
    return {"policy":policy,"pattern":pattern,"ticks":ticks,
            "probes":nprobes,"net_cost":round(loss,6)}


def measure_memory(case_name, route):
    """Isolated traced allocations; not an RSS or runtime-speed figure."""
    import tracemalloc
    chosen = next(((case, batch) for case, batch in CASES if case.name == case_name), None)
    if chosen is None or route not in ROUTES:
        raise ValueError("unknown case or route")
    case, batch = chosen
    suites, events = generate(case)
    tracemalloc.start()
    try:
        digest = replay(route, case, batch, suites, events)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"case":case.name, "route":route,"current_bytes":current,
            "peak_traced_bytes":peak,"digest":digest,"host":host()}


def run_suite(smoke=False,pairs=3):
    rows=[]
    for case,batch in SMOKE_CASES if smoke else CASES:
        rows.append(measure(case,batch,pairs))
    simulations=[simulate(p,regime) for regime in ("stable","single","reversal")
                 for p in ("fixed8","fixed64","every","adaptive")]
    return {"schema":"alexpy-bardo-bridge/1","host":host(),
            "comparison":"SAME native QA kernel + exact witness tree vs Python failure set",
            "rows":rows,"synthetic_calibration_simulation":simulations,
            "claims":{"default_promotion":False,"tree_faster_not_assumed":True,
                      "synthetic_time_not_cpu_benchmark":True,
                      "Bardo_reference_sha":"f50490e8194f0e5c7aeeba251f2575eb8155ed54"}}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--pairs",type=int,default=3)
    parser.add_argument("--sessions",type=int,default=1)
    parser.add_argument("--json",type=Path)
    parser.add_argument("--memory-worker",choices=ROUTES)
    parser.add_argument("--memory-case",default="medium-sparse")
    args=parser.parse_args(argv)
    if args.pairs<=0 or args.sessions<=0:
        parser.error("positive --pairs and --sessions required")
    if args.memory_worker:
        print(json.dumps(measure_memory(args.memory_case,args.memory_worker),indent=2))
        return
    if args.sessions==1:
        result=run_suite(args.smoke,args.pairs)
    else:
        command=[sys.executable,"-m","alexpy.bardo_bridge_bench","--pairs",str(args.pairs)]
        if args.smoke:command.append("--smoke")
        runs=[]
        for _ in range(args.sessions):
            x=subprocess.run(command,capture_output=True,text=True,check=True)
            runs.append(json.loads(x.stdout))
        result={"schema":"alexpy-bardo-bridge/1","sessions":runs,
                "host":host(),"session_count":len(runs)}
    data=json.dumps(result,ensure_ascii=False,indent=2)
    if args.json:args.json.write_text(data+"\n",encoding="utf-8")
    print(data)

if __name__=="__main__":main()
