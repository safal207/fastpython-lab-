"""Bounded, correctness-first performance state graph for AlexPy.

Research-inspired by COSMIC transition frontiers and independent witnesses;
no COSMIC/CaPU code is imported and no CPU-level speedup is implied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from . import _native, backend_name, sum_squares

CONTAINERS = ("list", "tuple", "list_subclass")
LENGTHS = (0, 1, 8, 16, 1024, 10000)
PROFILES = ("small", "mixed", "huge", "acc_overflow")
PHASES = ("prewarm", "steady")
AXES = ("container", "length", "profile", "phase")


class PlainListSubclass(list):
    """A list subtype that must trigger the safe fallback."""


@dataclass(frozen=True, order=True)
class State:
    container: str
    length: int
    profile: str
    phase: str = "steady"

    def __post_init__(self):
        if (self.container not in CONTAINERS or self.length not in LENGTHS or
                self.profile not in PROFILES or self.phase not in PHASES):
            raise ValueError("unknown graph state axis value")

    @property
    def id(self):
        return f"{self.container}|{self.length}|{self.profile}|{self.phase}"

    def change(self, **kwargs):
        return State(**(asdict(self) | kwargs))


def all_states():
    return [State(c, n, p, t)
            for c in CONTAINERS for n in LENGTHS
            for p in PROFILES for t in PHASES]


def neighbors(state):
    """Exactly one controlled axis changes per edge.

    Length/layout transitions move one notch; temporal and profile are discrete.
    """
    result = []
    for axis, values in (("container", CONTAINERS), ("length", LENGTHS),
                         ("profile", PROFILES), ("phase", PHASES)):
        pos = values.index(getattr(state, axis))
        candidates = ((values[pos - 1:pos] + values[pos + 1:pos + 2])
                      if axis in ("container", "length")
                      else tuple(v for v in values if v != getattr(state, axis)))
        for value in candidates:
            result.append((axis, state.change(**{axis: value})))
    return sorted(result, key=lambda item: (item[0], item[1].id))


def edges():
    found = set()
    for state in all_states():
        for axis, other in neighbors(state):
            lo, hi = sorted((state.id, other.id))
            found.add((lo, hi, axis))
    return sorted(found)


def _pair_coverage(state):
    values = (state.container, state.length, state.profile)
    return frozenset((i, j, values[i], values[j])
                     for i in range(3) for j in range(i + 1, 3))


def choose_cases(limit=12):
    """Deterministic pairwise-coverage greedy; not claimed to be globally optimal."""
    if not 1 <= limit <= len(CONTAINERS) * len(LENGTHS) * len(PROFILES):
        raise ValueError("limit must be between 1 and the number of unique cases")
    pool = [State(c, n, p) for c in CONTAINERS for n in LENGTHS for p in PROFILES]
    selected, covered = [], set()
    while len(selected) < limit:
        candidate = max(pool, key=lambda s: (
            len(_pair_coverage(s) - covered),
            int(s.length in (0, 1, 16, 10000)),
            int(s.profile in ("acc_overflow", "huge")),
            -CONTAINERS.index(s.container),
            -LENGTHS.index(s.length),
            -PROFILES.index(s.profile),
        ))
        selected.append(candidate)
        covered.update(_pair_coverage(candidate))
        pool.remove(candidate)
    return selected


def make_values(state):
    n = state.length
    if state.profile == "small":
        values = [i % 17 for i in range(n)]
    elif state.profile == "mixed":
        values = [(-1 if i % 2 else 1) * (i % 31) for i in range(n)]
    elif state.profile == "huge":
        values = [10**30 + i for i in range(n)]
    else:
        values = [10**9 for _ in range(n)]
    if state.container == "tuple":
        return tuple(values)
    if state.container == "list_subclass":
        return PlainListSubclass(values)
    return values


def mirror_oracles(values):
    """Two algebraically distinct exact-int witnesses, outside timed region.

    Both share CPython arithmetic internals; NOT a fully independent machine.
    """
    if not all(type(x) is int for x in values):
        raise TypeError("graph mirror requires exact Python int elements")
    mirror_a = sum(x * x for x in values)
    mirror_b = sum(pow(x, 2) for x in values)
    candidate = sum_squares(values)
    return mirror_a, mirror_b, candidate


def expected_fastpath(state, values):
    if _native is None:
        return False
    return _native.sum_squares_ints(values) is not NotImplemented


def check_correctness(state, values):
    """Gate: no timing / performance promotion if independent checks diverge."""
    try:
        mirror_a, mirror_b, actual = mirror_oracles(values)
        if mirror_a != mirror_b or mirror_a != actual:
            return {"correct": False, "reason": "MIRROR_DIVERGENCE"}
        fastpath = expected_fastpath(state, values)
        if state.container == "list_subclass" and fastpath:
            return {"correct": False, "reason": "SUBCLASS_UNSAFE_FASTPATH"}
        payload = repr((state.id, mirror_a)).encode("utf-8")
        return {"correct": True, "reason": "MIRRORS_AGREE",
                "fastpath_used": fastpath,
                "oracle_sha256": hashlib.sha256(payload).hexdigest()}
    except Exception as error:
        return {"correct": False, "reason": "ORACLE_ERROR",
                "error": f"{type(error).__name__}: {error}"}


def _time_call(func, values, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        func(values)
    return (time.perf_counter_ns() - start) / loops / 1_000_000


def _baseline(values):
    return sum(x * x for x in values)


def paired_measure(values, repeats=9, target_seconds=0.01):
    """AB/BA balanced pairs. Not a CPU cold-start or independent-lab proof."""
    if repeats < 1 or target_seconds <= 0:
        raise ValueError("repeats > 0 and target_seconds > 0 required")
    # This prewarm observation occurs AFTER correctness checking, but without
    # benchmark warmups. It is not a literal cold-process or cache-cold timing.
    pre_a = _time_call(_baseline, values, 1)
    pre_b = _time_call(sum_squares, values, 1)
    for _ in range(3):
        _baseline(values)
        sum_squares(values)
    slowest = max(_time_call(_baseline, values, 1),
                  _time_call(sum_squares, values, 1), 1e-9)
    loops = max(1, min(5000, int(target_seconds * 1000 / slowest)))
    a, b = [], []
    for i in range(repeats):
        if i % 2 == 0:
            a.append(_time_call(_baseline, values, loops))
            b.append(_time_call(sum_squares, values, loops))
        else:
            b.append(_time_call(sum_squares, values, loops))
            a.append(_time_call(_baseline, values, loops))
    ratios = [left / right for left, right in zip(a, b)]
    return {"prewarm": {"baseline_ms": pre_a, "alexpy_ms": pre_b},
            "steady": {"baseline_ms": a, "alexpy_ms": b,
                       "paired_speedup": ratios,
                       "median_speedup": statistics.median(ratios),
                       "loops_per_sample": loops}}


def bootstrap_interval(ratios, seed=2107, draws=1000):
    """Exploratory paired bootstrap; dependent samples can invalidate the CI."""
    if not ratios or any(x <= 0 for x in ratios):
        raise ValueError("positive nonempty ratios required")
    rng = random.Random(seed)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                     for _ in range(draws))
    return [samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]]


def verdict(correct, ratios, min_repeats=7, gain=1.10):
    """A *candidate* judgement, not a verified universal speedup."""
    if not correct:
        return "BLOCKED_INCORRECT"
    if len(ratios) < min_repeats:
        return "EXPLORATORY_FEW_SAMPLES"
    lo, hi = bootstrap_interval(ratios)
    if lo > gain:
        return "CANDIDATE_GAIN"
    if hi < 1 / gain:
        return "CANDIDATE_REGRESSION"
    return "INCONCLUSIVE"


def explore(limit=12, repeats=9, target_seconds=.01):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    results = []
    for state in choose_cases(limit):
        values = make_values(state)
        gate = check_correctness(state, values)
        item = {"state": state.id, "axes": asdict(state), "gate": gate,
                "prewarm_node": state.change(phase="prewarm").id,
                "steady_node": state.id}
        if gate["correct"]:
            measurements = paired_measure(values, repeats, target_seconds)
            ratios = measurements["steady"]["paired_speedup"]
            item["measurement"] = measurements
            item["bootstrap_95pct"] = (bootstrap_interval(ratios)
                                         if repeats >= 7 else None)
            item["verdict"] = verdict(True, ratios)
        else:
            item["verdict"] = "BLOCKED_INCORRECT"
        results.append(item)
    measured = {item["state"] for item in results}
    next_frontier = propose_followups(results, measured, limit=8)
    report = {"schema": "alexpy-graph-search/v0.1", "purpose": "bounded QA search for specialized kernels",
              "scope": "sum_squares only; no whole-Python speed claim",
              "python": sys.version, "implementation": platform.python_implementation(),
              "backend": backend_name(), "platform": platform.platform(),
              "machine": platform.machine(), "processor": platform.processor(),
              "cpu_count": os.cpu_count(), "source_revision": os.getenv("GITHUB_SHA"),
              "utc": datetime.now(timezone.utc).isoformat(),
              "axes": {"container": CONTAINERS, "length": LENGTHS,
                       "profile": PROFILES, "phase": PHASES},
              "graph": {"total_states": len(all_states()),
                        "total_edges": len(edges()),
                        "transition_rule": "change one axis; lengths/layouts adjacent"},
              "selection": "deterministic greedy pairwise coverage (not exhaustive)",
              "repeats": repeats, "target_seconds": target_seconds,
              "results": results, "next_frontier": next_frontier}
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def propose_followups(results, measured, limit=8):
    """Prioritize one-axis neighbours of observed performance candidates."""
    scored = {}
    for item in results:
        state = State(**item["axes"])
        importance = 3 if item["verdict"] in ("CANDIDATE_GAIN", "CANDIDATE_REGRESSION") else 1
        for axis, nxt in neighbors(state):
            # 'prewarm' states are sampled during the steady-state case run.
            if nxt.phase != "steady" or nxt.id in measured:
                continue
            priority = importance + int(axis == "length")
            scored[nxt.id] = max(scored.get(nxt.id, 0), priority)
    return [{"state": key, "priority": priority, "reason": "one-axis neighbor"}
            for key, priority in sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:limit]]


def to_dot(report):
    """Export the complete state graph, highlighting actually measured states."""
    statuses = {item["steady_node"]: item["verdict"] for item in report["results"]}
    for item in report["results"]:
        statuses[item["prewarm_node"]] = "PREWARM_OBSERVED" if item["gate"]["correct"] else "BLOCKED_INCORRECT"
    lines = ["graph AlexPy {", '  graph [label="AlexPy state transitions (not CPU topology)"];',
             '  node [shape=box, fontsize=8];']
    for node in all_states():
        status = statuses.get(node.id, "UNMEASURED")
        color = ("palegreen" if status == "CANDIDATE_GAIN" else
                 "lightcoral" if status in ("BLOCKED_INCORRECT", "CANDIDATE_REGRESSION") else
                 "lightblue" if status != "UNMEASURED" else "white")
        lines.append(f'  "{node.id}" [style=filled, fillcolor="{color}", tooltip="{status}"];')
    for a, b, axis in edges():
        lines.append(f'  "{a}" -- "{b}" [label="{axis}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explore time/shape/size states; correctness first")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--target-seconds", type=float, default=.01)
    parser.add_argument("--json", type=Path, default=Path("graph-results.json"))
    parser.add_argument("--dot", type=Path, default=Path("graph-states.dot"))
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.cases <= len(CONTAINERS) * len(LENGTHS) * len(PROFILES):
        parser.error("--cases must be between 1 and 72")
    if args.repeat < 1 or args.target_seconds <= 0:
        parser.error("--repeat and --target-seconds must be positive")
    if args.require_native and _native is None:
        parser.error("native module unavailable; install with python -m pip install -e .")
    report = explore(args.cases, args.repeat, args.target_seconds)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.dot.parent.mkdir(parents=True, exist_ok=True)
    args.dot.write_text(to_dot(report), encoding="utf-8")
    summary = {"results": str(args.json), "graph": str(args.dot),
               "cases": args.cases, "verdicts": {s: sum(r["verdict"] == s for r in report["results"])
                                                   for s in sorted({r["verdict"] for r in report["results"]})},
               "next_frontier": len(report["next_frontier"])}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(not item["gate"]["correct"] for item in report["results"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
