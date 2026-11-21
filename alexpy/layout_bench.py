"""Preregistered layout / traversal comparisons on identical typed buffers.

The native sequence traversal does not implement a Python interpreter. All
comparisons include the candidate's dispatch cost and any list conversion.
This script does not install or promote performance optimizations.
"""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from .memory_candidates import CANDIDATES

SIZES = (0, 1, 8, 128, 4096, 32768, 262144)
PROFILES = ("small", "large", "mixed")
REPRESENTATIONS = ("array", "memoryview")


def make_input(size: int, profile: str, representation: str):
    if size < 0 or profile not in PROFILES or representation not in REPRESENTATIONS:
        raise ValueError("invalid experimental state")
    if profile == "small":
        data = array("q", ((i % 17) - 8 for i in range(size)))
    elif profile == "large":
        data = array("q", (2**40 + (i % 7) for i in range(size)))
    else:
        data = array("q", ((2**40 if i % 37 == 0 else (i % 97) - 48)
                            for i in range(size)))
    return data if representation == "array" else memoryview(data)


def mirrors(values):
    a = sum(x * x for x in values)
    b = sum(map(lambda x: pow(x, 2), values))
    if type(a) is not int or a != b:
        raise AssertionError("independent Python expressions disagree")
    return a


def measure(fn, values, repeats: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn(values)
    return time.perf_counter_ns() - start


def paired_samples(control, candidate, values, *, pairs: int, target_seconds: float,
                   epoch: int):
    if pairs < 1 or target_seconds <= 0:
        raise ValueError("pairs and target_seconds must be positive")
    expected = mirrors(values)
    for name, fn in (("control", control), ("candidate", candidate)):
        if fn(values) != expected:
            raise AssertionError(f"{name} violates integer semantics")
    for fn in (control, candidate):
        for _ in range(2):
            fn(values)
    calibration = max(1, measure(control, values, 1))
    loops = max(1, min(10000, int(target_seconds * 1e9 / calibration)))
    samples = []
    for i in range(pairs):
        if (i + epoch) % 2:
            second = measure(candidate, values, loops)
            first = measure(control, values, loops)
            order = "BA"
        else:
            first = measure(control, values, loops)
            second = measure(candidate, values, loops)
            order = "AB"
        samples.append({"order": order, "baseline_ns_per_call": first / loops,
                        "candidate_ns_per_call": second / loops})
    baseline = statistics.median(x["baseline_ns_per_call"] for x in samples)
    trial = statistics.median(x["candidate_ns_per_call"] for x in samples)
    return {"loops": loops, "samples": samples,
            "baseline_median_ns": baseline, "candidate_median_ns": trial,
            "speedup": baseline / trial if trial else None}


def measurements(*, sizes=SIZES, profiles=("small", "large"),
                 representations=REPRESENTATIONS, epochs=2, pairs=7,
                 target_seconds=0.002):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    cases = []
    for epoch in range(epochs):
        states = [(n, p, r) for n in sizes for p in profiles for r in representations]
        if epoch % 2:
            states.reverse()
        for size, profile, representation in states:
            values = make_input(size, profile, representation)
            observations = {}
            for name, fn in CANDIDATES.items():
                if name == "current":
                    continue
                observations[name] = paired_samples(
                    CANDIDATES["current"], fn, values, pairs=pairs,
                    target_seconds=target_seconds, epoch=epoch)
            cases.append({"epoch": epoch, "size": size, "profile": profile,
                          "representation": representation,
                          "payload_bytes": size * 8,
                          "root_object_bytes": sys.getsizeof(values),
                          "observations": observations})
    return cases


def summarize(cases):
    groups = {}
    for case in cases:
        key = (case["size"], case["profile"], case["representation"])
        for name, obs in case["observations"].items():
            groups.setdefault((key, name), []).append(obs["speedup"])
    summary = []
    for (key, name), speeds in sorted(groups.items()):
        summary.append({"size": key[0], "profile": key[1],
                        "representation": key[2], "candidate": name,
                        "epochs": len(speeds), "epoch_speedups": speeds,
                        "median_speedup": statistics.median(speeds),
                        "every_epoch_faster": all(v > 1.1 for v in speeds),
                        "every_epoch_regression": all(v < 0.9 for v in speeds)})
    return summary


def adjacent_edges(summary):
    groups = {}
    for row in summary:
        key = (row["candidate"], row["profile"], row["representation"])
        groups.setdefault(key, []).append(row)
    edges = []
    for (candidate, profile, rep), rows in groups.items():
        rows.sort(key=lambda x: x["size"])
        for a, b in zip(rows, rows[1:]):
            if a["size"] == b["size"]:
                continue
            x = a["median_speedup"]
            y = b["median_speedup"]
            if x <= 0 or y <= 0:
                continue
            # Discrete finite difference across the log size axis. Not a
            # time derivative or evidence of a fractal law/cache transition.
            dx = math.log2(b["size"] + 1) - math.log2(a["size"] + 1)
            edges.append({"candidate": candidate, "profile": profile,
                          "representation": rep, "from_size": a["size"],
                          "to_size": b["size"],
                          "d_log_speedup_d_log2_size":
                              (math.log(y) - math.log(x)) / dx})
    return edges


def report(**kwargs):
    cases = measurements(**kwargs)
    rows = summarize(cases)
    outcome = {
        "schema_version": "alexpy.layout.v1",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "machine": platform.machine()},
        "scope": "same typed input; public sum_squares unchanged",
        "hardware_counters": "NOT_MEASURED",
        "promotion_gate": "BLOCKED_UNTIL_MULTIHOST_GUARDSET_REVIEW",
        "cases": cases, "summary": rows,
        "spatial_edges": adjacent_edges(rows),
    }
    payload = json.dumps(outcome, sort_keys=True, separators=(",", ":"))
    outcome["sha256_without_digest"] = hashlib.sha256(payload.encode()).hexdigest()
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="0,1,8,128,4096,32768")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--target-seconds", type=float, default=0.002)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    sizes = tuple(int(x) for x in args.sizes.split(","))
    if not sizes or any(n < 0 for n in sizes):
        parser.error("sizes must be nonnegative integers")
    result = report(sizes=sizes, epochs=args.epochs, pairs=args.pairs,
                    target_seconds=args.target_seconds)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["summary"]:
        print(f"{row['representation']:10} {row['profile']:5} n={row['size']:7} "
              f"{row['candidate']:13} {row['median_speedup']:6.2f}x")
    print("promotion_gate:", result["promotion_gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
