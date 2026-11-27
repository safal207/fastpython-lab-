"""Reproducible local comparison, without arbitrary performance assertions."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, _native, backend_name, sum_squares


def baseline(values):
    return sum(x * x for x in values)


def measure(func, data, warmup: int, repeat: int, target_seconds: float):
    for _ in range(warmup):
        func(data)
    # Calibrate each competitor independently, keeping timeit-call overhead low.
    started = time.perf_counter()
    func(data)
    elapsed = max(time.perf_counter() - started, 1e-9)
    loops = max(1, min(100_000, int(target_seconds / elapsed)))
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        for _ in range(loops):
            func(data)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / loops / 1_000_000)
    return {"median_ms": statistics.median(samples), "loops_per_sample": loops,
            "samples_ms": samples, "min_ms": min(samples), "max_ms": max(samples)}


def run(sizes=(1000, 100_000), repeat=9, warmup=3, target_seconds=0.08, require_fastpath=False):
    experiments = []
    for n in sizes:
        if n < 0:
            raise ValueError("sizes must be non-negative")
        values = list(range(n))
        fast_path_used = (_native is not None and
                          _native.sum_squares_ints(values) is not NotImplemented)
        if require_fastpath and not fast_path_used:
            raise RuntimeError(f"Native fast path unavailable for n={n}; choose a supported input size")
        expected = baseline(values)
        actual = sum_squares(values)
        if expected != actual:
            raise AssertionError(f"Incorrect result for n={n}")
        a = measure(baseline, values, warmup, repeat, target_seconds)
        b = measure(sum_squares, values, warmup, repeat, target_seconds)
        experiments.append({"case": "sum_squares_list_int", "size": n,
                            "baseline": a, "alexpy": b,
                            "speedup_vs_generator": a["median_ms"] / b["median_ms"],
                            "correct": True, "native_fast_path_used": fast_path_used})
    return {
        "version": __version__, "backend": backend_name(),
        "baseline": "sum(x * x for x in values)",
        "scope": "specialized list[int] sum-of-squares operation, not all Python code",
        "python": sys.version, "implementation": platform.python_implementation(),
        "machine": platform.machine(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repeat": repeat, "warmup": warmup,
        "results": experiments,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 100000])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-seconds", type=float, default=0.08)
    parser.add_argument("--json", type=Path, help="Write results to JSON file")
    parser.add_argument("--require-native", action="store_true", help="Fail if C extension is missing")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 0 or args.target_seconds <= 0:
        parser.error("repeat >= 1, warmup >= 0, target-seconds > 0 required")
    if args.require_native and backend_name() != "native-c":
        parser.error("C extension missing: rebuild with 'python -m pip install -e .' before benchmarking")
    report = run(args.sizes, args.repeat, args.warmup, args.target_seconds, args.require_native)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
