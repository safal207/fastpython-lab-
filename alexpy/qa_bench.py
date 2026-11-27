"""Auditable QA dashboard benchmark with independent witness and AB/BA pairs.

Timed paths: full initialization + all updates + dashboard materialization.
Includes strong O(1) Python incremental baseline; never represents a simple
full scan as the only alternative. All data is explicitly synthetic.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from .qa_lab import (
    AlexPyQAGraph, StrongPythonQA, QAEvent, scan_reference,
    NOT_RUN, PASS, FAIL, SKIP,
)


@dataclass(frozen=True)
class Case:
    name: str
    tests: int
    suites: int
    events: int
    report_every: int
    reads_per_report: int


FULL_CASES = (
    Case("small-frequent-reports", 256, 8, 1200, 1, 3),
    Case("medium-mixed-retries", 4096, 32, 6000, 16, 4),
    Case("large-infrequent-reports", 16384, 128, 12000, 128, 8),
    Case("medium-every-event-report", 4096, 32, 4000, 1, 1),
)
SMOKE_CASES = (
    Case("smoke-small", 256, 8, 160, 16, 1),
    Case("smoke-medium", 1024, 16, 320, 64, 2),
)


def generate(case: Case, seed: int = 98123):
    rng = random.Random(seed + case.tests + case.events)
    # Fixed suite mapping, skewed and repeatable. Test IDs may be retried.
    suites = [(i // 3 + i * 17) % case.suites for i in range(case.tests)]
    events = []
    for step in range(case.events):
        index = rng.randrange(case.tests)
        status = rng.choices((PASS, FAIL, SKIP), weights=(76, 18, 6))[0]
        duration = 0 if status == SKIP else rng.randrange(1, 100001)
        events.append(QAEvent(index, status, duration))
    return suites, events


def _report_hash(snapshot) -> int:
    # Hashing/materializing the same immutable snapshot in both timed routes.
    return hash(snapshot)


def replay(route: str, case: Case, suite_map, events) -> int:
    if route == "strong_python":
        engine = StrongPythonQA(suite_map, case.suites)
    elif route == "alexpy_graph":
        engine = AlexPyQAGraph(suite_map, case.suites)
    elif route == "scan_witness":
        statuses = [NOT_RUN] * len(suite_map)
        durations = [0] * len(suite_map)
    else:
        raise ValueError("unknown route")
    digest = 0
    for i,event in enumerate(events, 1):
        if route == "scan_witness":
            statuses[event.test_id] = event.status
            durations[event.test_id] = event.duration_ms
        else:
            engine.apply(event)
        if i % case.report_every == 0:
            for _ in range(case.reads_per_report):
                if route == "scan_witness":
                    result = scan_reference(suite_map, case.suites, statuses, durations)
                else:
                    result = engine.snapshot()
                digest = (digest * 1000003 + _report_hash(result)) & ((1 << 64) - 1)
    return digest


def verify(case: Case, suite_map, events):
    """Fail closed before any timings; exhaustive independent snapshot mirror."""
    a = StrongPythonQA(suite_map, case.suites)
    b = AlexPyQAGraph(suite_map, case.suites)
    statuses, durations = [NOT_RUN] * len(suite_map), [0] * len(suite_map)
    for i,event in enumerate(events, 1):
        a.apply(event); b.apply(event)
        statuses[event.test_id] = event.status
        durations[event.test_id] = event.duration_ms
        if i % case.report_every == 0 or i == len(events):
            expected = scan_reference(suite_map, case.suites, statuses, durations)
            if a.snapshot() != expected or b.snapshot() != expected:
                raise AssertionError("QA graph/strong baseline did not match independent scan")
    if not b.verify():
        raise AssertionError("owned-buffer mirror failed")
    digests = {name:replay(name,case,suite_map,events)
               for name in ("strong_python","alexpy_graph","scan_witness")}
    if len(set(digests.values())) != 1:
        raise AssertionError("timed routes produced distinct report digests")
    return digests["strong_python"]


def measure_once(route,case,suite_map,events):
    t0=time.perf_counter_ns()
    digest=replay(route,case,suite_map,events)
    return time.perf_counter_ns()-t0,digest


def measure_memory(route,case,suite_map,events):
    # Diagnostic only; do not mix tracemalloc-instrumented time with timings.
    tracemalloc.start()
    try:
        replay(route,case,suite_map,events)
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def worker_run(pairs,smoke):
    cases=SMOKE_CASES if smoke else FULL_CASES
    result=[]
    for case in cases:
        suite_map,events=generate(case)
        witness=verify(case,suite_map,events)
        samples=[]
        for pair in range(pairs):
            order=("strong_python","alexpy_graph") if pair%2==0 else ("alexpy_graph","strong_python")
            elapsed={}
            for route in order:
                t,digest=measure_once(route,case,suite_map,events)
                if digest != witness:
                    raise AssertionError("post-preflight digest drift")
                elapsed[route]=t
            samples.append({"order":list(order),"strong_python_ns":elapsed["strong_python"],
                            "alexpy_graph_ns":elapsed["alexpy_graph"],
                            "speedup":elapsed["strong_python"] / elapsed["alexpy_graph"]})
        result.append({"case":asdict(case),"witness_digest":witness,
                       "pairs":samples,
                       "python_peak_tracemalloc_bytes":measure_memory("strong_python",case,suite_map,events),
                       "alexpy_peak_tracemalloc_bytes":measure_memory("alexpy_graph",case,suite_map,events),
                       "median_speedup":statistics.median(s["speedup"] for s in samples)})
    return {"process": {"python":sys.version.split()[0], "platform":platform.platform(),
                        "processor":platform.processor(),"executable":sys.executable},
            "cases":result}


def run(sessions=3,pairs=5,smoke=False):
    if type(sessions) is not int or sessions < 1 or type(pairs) is not int or pairs < 1:
        raise ValueError("sessions/pairs must be positive integers")
    workers=[]
    for _ in range(sessions):
        cmd=[sys.executable,"-m","alexpy.qa_bench","--worker","--pairs",str(pairs)]
        if smoke:cmd.append("--smoke")
        proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        workers.append(json.loads(proc.stdout))
    summary=[]
    for case_index,case in enumerate(SMOKE_CASES if smoke else FULL_CASES):
        measures=[w["cases"][case_index]["median_speedup"] for w in workers]
        summary.append({"case":case.name,
                        "process_median_speedups":measures,
                        "median_process_speedup":statistics.median(measures),
                        "all_processes_over_1_1":all(v>1.1 for v in measures),
                        "all_processes_under_0_9":all(v<0.9 for v in measures)})
    return {"schema":"ALEXPY-QA-LAB/1","source":"deterministic synthetic QA events",
            "speedup_definition":"strong_python_total_ns / alexpy_graph_total_ns",
            "timed_scope":"initialization, all event updates, report snapshots and digest",
            "correctness":"independent O(N) scan and full-stream digest preflight",
            "caveat":"no production data; cross-process medians, not hardware-wide guarantees",
            "sessions":sessions,"pairs":pairs,"summary":summary,"workers":workers}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions",type=int,default=3)
    parser.add_argument("--pairs",type=int,default=5)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--json",type=Path)
    a=parser.parse_args(argv)
    if a.worker:
        print(json.dumps(worker_run(a.pairs,a.smoke)))
        return 0
    report=run(a.sessions,a.pairs,a.smoke)
    if a.json:
        a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for item in report["summary"]:
        print(f'{item["case"]}: {item["median_process_speedup"]:.3f}x vs strong Python '
              f'(per-process: {[round(v,3) for v in item["process_median_speedups"]]})')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
