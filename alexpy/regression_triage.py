"""Reproduce observed regressions across fresh processes; explain code paths.

Inspired by COSMIC's dirty-frontier controls and ContractGraph-QA's counterexamples.
These are hypotheses and process-local observations, never whole-Python claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys

from . import _native
from .graph_search import State, check_correctness, make_values, neighbors, paired_measure

_MAX_SQUARE_ROOT = 3037000499  # matches bounded native source
_MAX_TOTAL = (1 << 63) - 1
# Frozen from the 24-case exploratory report. They are candidates, not verdicts.
FOCUS_CASES = (
    State("tuple", 1, "huge"),
    State("list_subclass", 0, "small"),
    State("list_subclass", 1, "mixed"),
    State("list_subclass", 8, "small"),
)


def trace_path(values):
    """Independent Python model of native eligibility; validate against C result.

    This describes why a code path is taken; it does NOT prove that this
    particular reason caused the measured performance difference.
    """
    if _native is None:
        return {"path": "PYTHON_ONLY", "reason": "NATIVE_UNAVAILABLE", "visited": 0}
    if type(values) not in (list, tuple):
        return {"path": "PYTHON_FALLBACK", "reason": "UNSUPPORTED_CONTAINER", "visited": 0}
    total = 0
    for i, value in enumerate(values):
        if type(value) is not int:
            return {"path": "C_REJECT_PYTHON_FALLBACK", "reason": "NON_EXACT_INT", "visited": i + 1}
        if abs(value) > _MAX_SQUARE_ROOT:
            return {"path": "C_REJECT_PYTHON_FALLBACK", "reason": "OUTSIDE_INT64_SQUARE", "visited": i + 1}
        sq = value * value
        if total > _MAX_TOTAL - sq:
            return {"path": "C_REJECT_PYTHON_FALLBACK", "reason": "INT64_ACCUMULATOR_OVERFLOW", "visited": i + 1}
        total += sq
    return {"path": "C_FAST", "reason": "NATIVE_SUPPORTED", "visited": len(values)}


def gate_and_trace(state):
    values = make_values(state)
    gate = check_correctness(state, values)
    trace = trace_path(values)
    if not gate["correct"]:
        return values, gate, trace
    observed_fastpath = bool(gate["fastpath_used"])
    expected_fastpath = trace["path"] == "C_FAST"
    if observed_fastpath != expected_fastpath:
        gate = {"correct": False, "reason": "TRACE_NATIVE_DISAGREEMENT"}
    return values, gate, trace


def select_cases(focus=FOCUS_CASES):
    """Select one-axis controls which change actual input or execution path.

    At length=0 all arithmetic profiles are equivalent; at length=1 the
    "small" and "mixed" profiles both contain just zero. Do not treat
    such no-op transitions as evidence for a workload-dependent effect.
    """
    cases = set(focus)
    edges = []
    for state in focus:
        nearby = neighbors(state)
        # One container control; plus a meaningful arithmetic or length control.
        container = [other for axis, other in nearby
                     if axis == "container" and other.phase == "steady"]
        if container:
            control = container[0]
            cases.add(control)
            edges.append({"from": state.id, "to": control.id, "axis": "container"})
        for axis in ("profile", "length"):
            options = [other for kind, other in nearby
                       if kind == axis and other.phase == "steady"
                       and list(make_values(other)) != list(make_values(state))]
            if options:
                # Choose the largest changed-workload length when profiles
                # produce identical payloads at tiny sizes.
                control = options[-1] if axis == "length" else options[0]
                cases.add(control)
                edges.append({"from": state.id, "to": control.id, "axis": axis})
                break
    return sorted(cases), edges


def _worker(states, repeats, target_seconds, session):
    """Exactly one independent interpreter process, randomized state order."""
    rng = random.Random(7123 + session)
    order = list(states)
    rng.shuffle(order)
    records = []
    for state in order:
        values, gate, trace = gate_and_trace(state)
        record = {"state": state.id, "gate": gate, "path": trace}
        if gate["correct"]:
            m = paired_measure(values, repeats=repeats, target_seconds=target_seconds)
            record["measurement"] = m
            record["median_speedup"] = m["steady"]["median_speedup"]
        records.append(record)
    return {"session": session, "records": records}


def assess(session_medians, threshold=0.10):
    """Conservative cross-process classification; no significance claims."""
    if len(session_medians) < 3:
        return "INSUFFICIENT_PROCESS_REPLICATION"
    if any(not isinstance(v, (int, float)) or v <= 0 for v in session_medians):
        return "INVALID_TIMING"
    if all(r < 1.0 / (1.0 + threshold) for r in session_medians):
        return "REPRODUCED_REGRESSION_CANDIDATE"
    if all(r > 1.0 + threshold for r in session_medians):
        return "REPRODUCED_GAIN_CANDIDATE"
    return "UNSTABLE_OR_NEUTRAL"


def compare_edges(rows, edges):
    """Match single-axis controls; a timing contrast is not causal proof."""
    by_id = {r["state"]: r for r in rows}
    return [{**edge,
             "from_median_speedup": by_id[edge["from"]]["median_speedup"],
             "to_median_speedup": by_id[edge["to"]]["median_speedup"],
             "from_path": by_id[edge["from"]]["path"]["reason"],
             "to_path": by_id[edge["to"]]["path"]["reason"]}
            for edge in edges]


def reproduce(states=None, sessions=3, repeats=9, target_seconds=.004):
    if sessions < 1 or repeats < 1 or target_seconds <= 0:
        raise ValueError("sessions/repeats must be positive and target_seconds > 0")
    chosen, edges = select_cases() if states is None else (list(states), [])
    if len(set(chosen)) != len(chosen) or any(x.phase != "steady" for x in chosen):
        raise ValueError("unique steady state nodes required")
    # All correctness gates must pass before child timing starts.
    checked = {}
    for state in chosen:
        _, gate, trace = gate_and_trace(state)
        checked[state.id] = {"gate": gate, "path": trace}
    if any(not info["gate"]["correct"] for info in checked.values()):
        raise RuntimeError("correctness or path witness failed; no timing allowed")
    runs = []
    for i in range(sessions):
        cmd = [sys.executable, "-m", "alexpy.regression_triage", "--worker",
               "--states", ",".join(x.id for x in chosen),
               "--session", str(i), "--repeat", str(repeats),
               "--target-seconds", str(target_seconds)]
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        response = json.loads(completed.stdout)
        if {r["state"] for r in response["records"]} != {s.id for s in chosen}:
            raise RuntimeError("child process omitted or added states")
        if any(not r["gate"]["correct"] or r["path"] != checked[r["state"]]["path"]
               for r in response["records"]):
            raise RuntimeError("child correctness/path witness mismatch")
        runs.append(response)
    rows = []
    for state in chosen:
        ratios = [next(r["median_speedup"] for r in run["records"]
                       if r["state"] == state.id) for run in runs]
        rows.append({"state": state.id, "axes": {"container": state.container,
                     "length": state.length, "profile": state.profile, "phase": state.phase},
                     "path": checked[state.id]["path"],
                     "gate": checked[state.id]["gate"],
                     "session_median_speedups": ratios,
                     "median_speedup": statistics.median(ratios),
                     "verdict": assess(ratios)})
    report = {"schema": "alexpy-regression-triage/v0.1",
              "scope": "sum_squares; local independent processes, not independent hardware",
              "python": sys.version, "platform": platform.platform(),
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "sessions": sessions, "repeats_per_session": repeats,
              "target_seconds": target_seconds,
              "note": "Path is a deterministic code-path explanation; timing contrast is an association, not proof of causality.",
              "focus_cases": [s.id for s in FOCUS_CASES],
              "rows": rows, "one_axis_controls": compare_edges(rows, edges), "raw_sessions": runs}
    blob = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    report["sha256"] = hashlib.sha256(blob).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce graph regressions in independent Python processes")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--target-seconds", type=float, default=.004)
    parser.add_argument("--json", type=Path, default=Path("regression-triage.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--states", help=argparse.SUPPRESS)
    parser.add_argument("--session", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.states:
            parser.error("--worker requires --states")
        try:
            states = []
            for token in args.states.split(","):
                parts = token.split("|")
                if len(parts) != 4:
                    raise ValueError("state needs four axes")
                states.append(State(parts[0], int(parts[1]), parts[2], parts[3]))
            if len(states) != len(set(states)):
                parser.error("duplicate states")
            payload = _worker(states, args.repeat, args.target_seconds, args.session)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(payload, ensure_ascii=True))
        return 0
    try:
        report = reproduce(sessions=args.sessions, repeats=args.repeat,
                           target_seconds=args.target_seconds)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    args.json.parent.mkdir(exist_ok=True, parents=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.json), "measured_states": len(report["rows"]),
                      "focus": [{"state": r["state"], "path": r["path"]["reason"],
                                  "speedup": round(r["median_speedup"], 3),
                                  "verdict": r["verdict"]}
                                 for r in report["rows"] if r["state"] in report["focus_cases"]]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
