"""Follow-up measurement: actual long-horizon online payback of frozen AlexPy 1.8 routes.

Does not retrain on held-out sizes, change decisions, or silently subtract the
historical three-process calibration charge. Runs on the same CPU/Python/OS
profile as the archived preregistered calibration; otherwise fails closed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import median
from time import perf_counter_ns

from .range_routes import ROUTES, make_case
from .stable_memory import workload_key
from .stream_bench import csv_positive, host_metadata
from .stream_economics import stream_schedule, run_stream, witness


def frozen_evidence(document: dict, host: dict) -> tuple[dict[str, str], int]:
    """Validate frozen decisions and prevent cross-profile performance reuse."""
    if document.get("schema") not in ("ALEXPY-STREAM-ECONOMICS-SUMMARY/1", "ALEXPY-STREAM-ECONOMICS/1"):
        raise ValueError("expected verified v1.8 research summary or fresh calibration report")
    saved = document.get("host", {})
    for key in ("platform", "machine", "processor", "python", "implementation"):
        if host.get(key) != saved.get(key):
            raise ValueError(f"host mismatch: {key}")
    choices = document.get("route_choices")
    if not isinstance(choices, dict) or not choices or any(
        not isinstance(k, str) or v not in ROUTES for k, v in choices.items()
    ):
        raise ValueError("invalid route choices")
    digest = hashlib.sha256(json.dumps(choices, sort_keys=True).encode()).hexdigest()
    if digest != document.get("frozen_sha256"):
        raise ValueError("route choices altered since training")
    cost = document.get("calibration_wall_ns")
    if type(cost) is not int or cost <= 0:
        raise ValueError("invalid calibration wall time")
    return choices, cost


def worker(choices: dict[str, str], *, cycles: tuple[int, ...],
           sizes: tuple[int, ...], repeats: int, session: int,
           changes: tuple[int, ...] = (1, 8),
           queries: tuple[int, ...] = (1, 8, 32),
           spans: tuple[str, ...] = ("short", "wide"), ticks: int = 2) -> dict:
    rows = []
    for n in sizes:
        schedule = stream_schedule(n, changes=changes, queries=queries, spans=spans, ticks=ticks)
        case_cache = {workload_key(w): make_case(w) for w in schedule}
        if any(workload_key(w) not in choices for w in schedule):
            raise ValueError("frozen routes do not cover requested size")
        for k in cycles:
            expected = witness(schedule, cycles=k, case_cache=case_cache)
            times = {p: [] for p in ("simple", "stable")}
            details = {p: [] for p in times}
            for rep in range(repeats):
                order = ("simple", "stable") if (rep + session) % 2 == 0 else ("stable", "simple")
                for policy in order:
                    start = perf_counter_ns()
                    result = run_stream(schedule, cycles=k, policy=policy,
                                        choices=choices, case_cache=case_cache)
                    elapsed = perf_counter_ns() - start
                    if result.checksum != expected:
                        raise AssertionError("continuous stream differs from independent direct witness")
                    times[policy].append(elapsed)
                    details[policy].append({"switches": result.switches,
                                            "switch_ns": result.switch_ns,
                                            "constructed": result.constructed,
                                            "segments": result.segments})
            rows.append({"size": n, "cycles": k,
                         "raw_ns": times, "details": details,
                         "median_ns": {p: median(v) for p, v in times.items()},
                         "correctness": "PASS",
                         "checksum_sha256": hashlib.sha256(str(expected).encode()).hexdigest()})
    return {"session": session, "pid": os.getpid(), "host": host_metadata(), "rows": rows}


def summarize(workers: list[dict], calibration_ns: int, *, cycles: tuple[int, ...],
              sizes: tuple[int, ...]) -> dict:
    if len(workers) < 3 or len({x["pid"] for x in workers}) != len(workers):
        raise ValueError("three independent interpreter processes required")
    cases = []
    for k in cycles:
        savings = []
        individual = []
        for session in workers:
            values = []
            for n in sizes:
                row = next((r for r in session["rows"] if r["size"] == n and r["cycles"] == k), None)
                if row is None or row["correctness"] != "PASS":
                    raise ValueError("missing verified case")
                values.append(row["median_ns"]["simple"]-row["median_ns"]["stable"])
            savings.append(sum(values))
            individual.append(values)
        observed = median(savings) >= calibration_ns and all(x >= calibration_ns for x in savings)
        cases.append({"cycles": k, "online_savings_ns_by_process": savings,
                      "by_size_ns_by_process": individual,
                      "median_online_savings_ns": median(savings),
                      "median_paid_net_ns": median(savings)-calibration_ns,
                      "all_processes_paid": all(x >= calibration_ns for x in savings),
                      "observed_paid_break_even": observed})
    return {"horizons": cases,
            "first_observed_paid_cycles": next((x["cycles"] for x in cases if x["observed_paid_break_even"]), None)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--sizes", type=csv_positive, default=(384, 1536))
    p.add_argument("--cycles", type=csv_positive, default=(128, 512, 6000))
    p.add_argument("--changes", type=csv_positive, default=(1, 8))
    p.add_argument("--queries", type=csv_positive, default=(1, 8, 32))
    p.add_argument("--spans", choices=("short", "wide", "both"), default="both")
    p.add_argument("--ticks", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=int, default=None)
    args = p.parse_args(argv)
    if args.ticks < 1:
        p.error("ticks must be positive")
    args.spans = ("short", "wide") if args.spans == "both" else (args.spans,)
    if args.repeats < 2 or args.sessions < 3:
        p.error("at least two paired repetitions and three processes are required")
    return args


def main(argv=None):
    args = parse_args(argv)
    raw = args.baseline.read_bytes()
    original = json.loads(raw)
    choices, calibration_ns = frozen_evidence(original, host_metadata())
    if args.worker is not None:
        result = worker(choices, cycles=args.cycles, sizes=args.sizes,
                        repeats=args.repeats, session=args.worker,
                        changes=args.changes, queries=args.queries,
                        spans=args.spans, ticks=args.ticks)
    else:
        sessions = []
        for session in range(args.sessions):
            command = [sys.executable, "-m", "alexpy.stream_horizon",
                       "--baseline", str(args.baseline),
                       "--sizes", ",".join(map(str, args.sizes)),
                       "--cycles", ",".join(map(str, args.cycles)),
                       "--repeats", str(args.repeats),
                       "--sessions", str(args.sessions),
                       "--changes", ",".join(map(str, args.changes)),
                       "--queries", ",".join(map(str, args.queries)),
                       "--spans", "both" if args.spans == ("short", "wide") else args.spans[0],
                       "--ticks", str(args.ticks),
                       "--worker", str(session)]
            subprocess_result = subprocess.run(command, check=True, text=True,
                                               capture_output=True, timeout=300)
            sessions.append(json.loads(subprocess_result.stdout))
        result = {"schema": "ALEXPY-STREAM-LONG-HORIZON/1",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "host": host_metadata(), "baseline_sha256": hashlib.sha256(raw).hexdigest(),
                  "frozen_sha256": original["frozen_sha256"],
                  "calibration_wall_ns_from_prior_session": calibration_ns,
                  "config": {"sizes": args.sizes, "cycles": args.cycles,
                             "repeats": args.repeats, "sessions": args.sessions,
                             "changes": args.changes, "queries": args.queries,
                             "spans": args.spans, "ticks": args.ticks},
                  "workers": sessions, **summarize(sessions, calibration_ns,
                                                   cycles=args.cycles, sizes=args.sizes),
                  "limitations": [
                      "Follow-up horizons were chosen after seeing the 128-cycle projection.",
                      "The original calibration was measured on this host earlier, not rerun now.",
                      "Held-out SIZES were seen in v1.8; long HORIZONS are new, not a fresh holdout.",
                      "No claim of transfer to other CPUs, tasks or real-world applications.",
                      "Correctness rechecked for each horizon with an independent direct witness."]}
    if args.worker is not None:
        print(json.dumps(result, separators=(",", ":")))
    else:
        if args.output:
            args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"first_observed_paid_cycles": result["first_observed_paid_cycles"],
                          "cases": [{"cycles": x["cycles"], "paid_net_ms": round(x["median_paid_net_ns"]/1e6, 2),
                                     "paid_in_all_processes": x["all_processes_paid"]}
                                    for x in result["horizons"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
