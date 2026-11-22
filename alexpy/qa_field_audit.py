"""Opt-in cross-host native QA audit: correctness, allocation and resident memory.

Measure routes in separate interpreters with the SAME pre-generated workload.
Fixture generation and independent correctness mirrors are deliberately outside
measurement. tracemalloc measures Python-managed allocations (not necessarily
all extension/allocator pages); max RSS is a process high-water mark, not an
exact per-engine allocation. Wall time while tracing MUST NOT be presented as
an uninstrumented speed benchmark. No automatic promotion.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from .qa_actions import extract_jobs
from .qa_batch_bench import Case, FULL_CASES, SMOKE_CASES, generate
from .qa_native_bench import ENGINES, replay, verify
from .qa_native import NativeBatchQA

ROUTES = ("native_batch", "strong_batch", "alexpy_batch")
CASES = {c.name: c for c in FULL_CASES + SMOKE_CASES}
RECORDED = "recorded-actions"
DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "evidence" / "qa-actions-anonymized-replays.json"


def resident_peak():
    """Cross-platform process working-set high-water in bytes, or None.

    Windows returns PeakWorkingSetSize via documented PROCESS_MEMORY_COUNTERS;
    Linux ru_maxrss is KiB, macOS ru_maxrss is bytes. This is NOT per-object RSS.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                        ("peak_working_set", ctypes.c_size_t),
                        ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged", ctypes.c_size_t),
                        ("quota_paged", ctypes.c_size_t),
                        ("quota_peak_nonpaged", ctypes.c_size_t),
                        ("quota_nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t),
                        ("peak_pagefile", ctypes.c_size_t)]
        obj = Counters(); obj.cb = ctypes.sizeof(obj)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(obj), obj.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(obj.peak_working_set)
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def recorded_case(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    suites, test_names, layout, events = extract_jobs(payload)
    if not suites or not test_names or not events:
        raise ValueError("recorded fixture must contain completed jobs")
    case = Case(RECORDED, len(test_names), len(suites), len(events),
                len(test_names), 3, 1)
    return case, list(layout), list(events)


def workload(case_name: str, fixture: Path):
    if case_name == RECORDED:
        return recorded_case(fixture)
    case = CASES[case_name]
    suites, events = generate(case)
    return case, suites, events


def resident_current():
    """Current resident memory; unavailable where no dependency-free API is used."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as fh:
                pages = int(fh.readline().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                        ("peak_working_set", ctypes.c_size_t),
                        ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged", ctypes.c_size_t),
                        ("quota_paged", ctypes.c_size_t),
                        ("quota_peak_nonpaged", ctypes.c_size_t),
                        ("quota_nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t),
                        ("peak_pagefile", ctypes.c_size_t)]
        obj = Counters(); obj.cb = ctypes.sizeof(obj)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(obj), obj.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(obj.working_set)
    return None


def measure_worker(case_name: str, route: str, batch_size: int, fixture: Path,
                   *, expected_digest=None):
    if route not in ROUTES:
        raise ValueError("unsupported route")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be positive")
    if not NativeBatchQA([0], 1).native:
        raise RuntimeError("C extension is not installed; cannot audit native route")
    case, suites, events = workload(case_name, fixture)
    # Parent preflights correctness before starting separate workers.
    # Direct calls without a certified expected digest still verify locally.
    expected = (verify(case, batch_size, suites, events) if expected_digest is None
                else expected_digest)
    gc.collect()
    baseline_rss = resident_peak()
    baseline_current = resident_current()
    tracemalloc.start()
    start = time.perf_counter_ns()
    observed = replay(route, case, batch_size, suites, events)
    elapsed = time.perf_counter_ns() - start
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = resident_peak()
    current_rss = resident_current()
    if observed != expected:
        raise AssertionError("audited replay disagrees with independent reference")
    return {
        "case": case_name, "route": route, "batch_size": batch_size,
        "events": len(events), "tests": len(suites), "digest": expected,
        "trace_peak_bytes": peak_traced,
        "rss_baseline_highwater_bytes": baseline_rss,
        "rss_baseline_current_bytes": baseline_current,
        "rss_current_after_bytes": current_rss,
        "rss_current_delta_bytes": (current_rss - baseline_current
                                     if current_rss is not None and baseline_current is not None else None),
        "rss_peak_bytes": peak_rss,
        "rss_highwater_growth_bytes": (max(0, peak_rss - baseline_rss)
                                      if peak_rss is not None and baseline_rss is not None else None),
        "instrumented_ns": elapsed,
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def audit(*, cases=("smoke-retries", RECORDED), sizes=(64,),
          sessions=2, fixture=DEFAULT_FIXTURE):
    if (type(sessions) is not int or sessions < 1 or not cases or not sizes
            or any(c not in CASES and c != RECORDED for c in cases)
            or any(type(s) is not int or s < 1 for s in sizes)):
        raise ValueError("invalid audit configuration")
    fixture = Path(fixture)
    # Verify all routes at every report boundary *before* memory collection.
    # Each child independently verifies again, so a bad route fails closed.
    raw = []
    for name in cases:
        for batch_size in sizes:
            case, suites, events = workload(name, fixture)
            expected = verify(case, batch_size, suites, events)
            for session in range(sessions):
                # Rotate ordering to lessen warm-cache / host drift.
                for route in ROUTES[session % 3:] + ROUTES[:session % 3]:
                    argv = [sys.executable, "-m", "alexpy.qa_field_audit",
                            "--worker", "--case", name, "--route", route,
                            "--batch-size", str(batch_size), "--fixture", str(fixture),
                            "--expected-digest", str(expected)]
                    completed = subprocess.run(argv, capture_output=True, text=True,
                                               check=True, timeout=180)
                    row = json.loads(completed.stdout)
                    if row["digest"] != expected:
                        raise AssertionError("cross-process certified digest mismatch")
                    row["session"] = session
                    raw.append(row)
    summary = []
    for name in cases:
        for batch_size in sizes:
            subset = [x for x in raw if x["case"] == name and x["batch_size"] == batch_size]
            if len({x["digest"] for x in subset}) != 1:
                raise AssertionError("cross-session witness mismatch")
            summary.append({
                "case": name, "batch_size": batch_size,
                "routes": {route: {
                    "peak_traced_bytes": statistics.median(
                        x["trace_peak_bytes"] for x in subset if x["route"] == route),
                    "rss_peak_bytes": statistics.median(
                        x["rss_peak_bytes"] for x in subset if x["route"] == route)
                        if all(x["rss_peak_bytes"] is not None for x in subset if x["route"] == route)
                        else None,
                    "rss_current_delta_bytes": statistics.median(
                        x["rss_current_delta_bytes"] for x in subset if x["route"] == route)
                        if all(x["rss_current_delta_bytes"] is not None for x in subset if x["route"] == route)
                        else None,
                    "rss_growth_bytes": statistics.median(
                        x["rss_highwater_growth_bytes"] for x in subset if x["route"] == route)
                        if all(x["rss_highwater_growth_bytes"] is not None for x in subset if x["route"] == route)
                        else None,
                } for route in ROUTES},
            })
    return {
        "schema": "ALEXPY-QA-FIELD-AUDIT/1", "source": "synthetic cases + sanitized public CI job metadata",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0], "extension_required": True,
        "measurement_contract": {
            "trace": "peak tracemalloc bytes since engine construction; excludes pre-generated fixtures",
            "rss": "process high-water mark in bytes; includes Python runtime and imported modules",
            "rss_growth": "increase in high-water mark since fixture generation; may be zero",
            "rss_current_delta": "current working-set change over measured replay, can be negative or unavailable",
            "instrumented_time": "not comparable with previous uninstrumented speed benchmarks",
            "correctness": "independent full scan at checkpoints and complete replay digest",
        },
        "summary": summary, "sessions": sessions, "raw": raw,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="smoke-retries,recorded-actions")
    ap.add_argument("--batch-sizes", default="64")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--case", default="smoke-retries", help=argparse.SUPPRESS)
    ap.add_argument("--route", default="native_batch", help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=64, help=argparse.SUPPRESS)
    ap.add_argument("--expected-digest", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--json", type=Path)
    opts = ap.parse_args(argv)
    if opts.worker:
        print(json.dumps(measure_worker(opts.case, opts.route, opts.batch_size, opts.fixture,
                                        expected_digest=opts.expected_digest)))
        return 0
    result = audit(cases=tuple(opts.cases.split(",")),
                   sizes=tuple(int(x) for x in opts.batch_sizes.split(",")),
                   sessions=opts.sessions, fixture=opts.fixture)
    if opts.json:
        opts.json.parent.mkdir(parents=True, exist_ok=True)
        opts.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in result["summary"]:
        print(row["case"], "batch", row["batch_size"],
              ", ".join(f"{k} traced={v['peak_traced_bytes']/1024:.1f} KiB"
                        for k,v in row["routes"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
