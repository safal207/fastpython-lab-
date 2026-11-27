"""Space-time transition traces, multiscale roughness and finite differences.

The 'fractal axis' is a finite-sample *scale signature*: it is NOT a measured
fractal dimension or a physical exponent. No candidate is auto-promoted.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import statistics
import time

from .adaptive import adaptive_sum_squares, route
from .graph_search import State, make_values, neighbors


@dataclass(frozen=True)
class Observation:
    tick: int
    timestamp_ns: int
    state: State
    baseline_ns: float
    candidate_ns: float
    decision: str
    correct: bool

    def __post_init__(self):
        if self.tick < 0 or self.timestamp_ns < 0:
            raise ValueError("ticks and timestamps must be nonnegative")
        if not (math.isfinite(self.baseline_ns) and math.isfinite(self.candidate_ns)
                and self.baseline_ns > 0 and self.candidate_ns > 0):
            raise ValueError("latencies must be finite and positive")

    @property
    def log_speedup(self):
        return math.log(self.baseline_ns / self.candidate_ns)


def check_path(points):
    """Require tick/time order and at most a single graph-axis hop per tick."""
    for a, b in zip(points, points[1:]):
        if b.tick != a.tick + 1 or b.timestamp_ns <= a.timestamp_ns:
            raise ValueError("nonmonotonic timeline")
        if b.state != a.state and b.state not in (neighbor for _, neighbor in neighbors(a.state)):
            raise ValueError("nonadjacent spatial graph transition")
    return True


def derivatives(points):
    """Two successive discrete derivatives of log speedup per graph tick.

    d1 is change in log-speedup per step; d2 is change in d1 per step.
    These are not CPU frequency or continuous physical-time derivatives.
    """
    check_path(points)
    values = [p.log_speedup for p in points]
    first = [values[i] - values[i - 1] for i in range(1, len(values))]
    second = [first[i] - first[i - 1] for i in range(1, len(first))]
    times_seconds = [point.timestamp_ns / 1e9 for point in points]
    per_second = [(values[i] - values[i-1]) / (times_seconds[i] - times_seconds[i-1])
                  for i in range(1, len(values))]
    second_per_second2 = [(per_second[i] - per_second[i-1]) /
                          ((times_seconds[i+1] - times_seconds[i-1]) / 2)
                          for i in range(1, len(per_second))]
    return {"log_speedup": values, "d1_per_tick": first, "d2_per_tick2": second,
            "d1_per_second": per_second,
            "d2_per_second2": second_per_second2}


def multiscale_signature(points, scales=(1, 2, 4)):
    """Median absolute differences in log-speedup at integer hop scales.

    This is a coarse multiresolution roughness proxy, NOT a fractal dimension.
    Scale ratios are emitted only when both roughness estimates are nonzero.
    """
    check_path(points)
    ss = tuple(scales)
    if not ss or len(ss) != len(set(ss)) or any(type(s) is not int or s <= 0 for s in ss):
        raise ValueError("scales must be unique positive integers")
    values = [p.log_speedup for p in points]
    roughness = {}
    for s in sorted(ss):
        steps = [abs(values[i] - values[i - s])
                 for i in range(s, len(values))]
        roughness[str(s)] = statistics.median(steps) if steps else None
    ratios = {}
    ordered = sorted(ss)
    for left, right in zip(ordered, ordered[1:]):
        a, b = roughness[str(left)], roughness[str(right)]
        key = f"{left}->{right}"
        ratios[key] = (math.log(b/a) / math.log(right/left)
                       if a is not None and b is not None and a > 0 and b > 0
                       else None)
    return {"roughness_by_hop_scale": roughness,
            "log_log_scale_slopes": ratios,
            "interpretation": "finite-sample exploratory scale signature, not fractal dimension"}


def planned_path(start=None, hops=15):
    """Deterministic one-axis walk through space, phase and arithmetic regime."""
    if hops < 0:
        raise ValueError("hops must be >= 0")
    state = start or State("list", 1, "small", "steady")
    sequence = [state]
    axes = ("length", "profile", "container", "phase")
    visited_edges = set()
    for i in range(hops):
        options = neighbors(state)
        wanted = axes[i % len(axes)]
        def ranking(item):
            axis, next_state = item
            edge = tuple(sorted((state.id, next_state.id)))
            return (axis != wanted, edge in visited_edges, next_state.id)
        axis, next_state = min(options, key=ranking)
        visited_edges.add(tuple(sorted((state.id, next_state.id))))
        state = next_state
        sequence.append(state)
    return sequence


def edge_deltas(points):
    """Record spatial direction (which workload axis changed) and delta per edge."""
    check_path(points)
    rows = []
    for left, right in zip(points, points[1:]):
        changes = [name for name in ("container", "length", "profile", "phase")
                   if getattr(left.state, name) != getattr(right.state, name)]
        rows.append({"from": left.state.id, "to": right.state.id,
                     "axis": changes[0] if changes else "repeat",
                     "delta_log_speedup": right.log_speedup - left.log_speedup,
                     "delta_time_ns": right.timestamp_ns - left.timestamp_ns})
    return rows


def shortest_graph_path(start, goal):
    """BFS over one-axis transitions (not fastest path through actual CPU)."""
    queue = deque([start]); previous = {start: None}
    while queue:
        state = queue.popleft()
        if state == goal:
            path = []
            while state is not None:
                path.append(state); state = previous[state]
            return list(reversed(path))
        for _, neighbor in neighbors(state):
            if neighbor not in previous:
                previous[neighbor] = state
                queue.append(neighbor)
    raise ValueError("unreachable state")


def targeted_path():
    """Cover the adaptive decision branches with graph-adjacent states."""
    stops = [State("list", 0, "small"),
             State("list", 1, "small"),
             State("list", 16, "acc_overflow"),
             State("tuple", 8, "huge"),
             State("list_subclass", 8, "mixed"),
             State("list", 10000, "mixed")]
    path = [stops[0]]
    for target in stops[1:]:
        path.extend(shortest_graph_path(path[-1], target)[1:])
    return path


def _bench_call(fn, values, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(values)
    return (time.perf_counter_ns() - start) / loops


def measure_trace(path=None, *, repeats=7, loops=200):
    """Verify full correctness before timing, then measure AB/BA in one process.

    Timing order rotates by tick and repetition; repeated experiments on
    separate machines are still necessary. Outputs are diagnostics, not proof.
    """
    if repeats < 1 or loops < 1:
        raise ValueError("repeats and loops must be positive")
    states = list(path if path is not None else planned_path())
    if not states:
        raise ValueError("path cannot be empty")
    # Ensure malformed paths never enter a timed benchmark.
    for a, b in zip(states, states[1:]):
        if a != b and b not in (other for _, other in neighbors(a)):
            raise ValueError("path must follow a single controlled axis")
    values_by_state = []
    for state in states:
        values = make_values(state)
        expected = sum(x*x for x in values)
        independent_expression = sum(pow(x, 2) for x in values)
        if expected != independent_expression or adaptive_sum_squares(values) != expected:
            raise ValueError("correctness mirror divergence: refuse benchmark")
        values_by_state.append(values)
    points = []
    stamp = time.perf_counter_ns()
    baseline = lambda values: sum(x*x for x in values)
    for tick, (state, values) in enumerate(zip(states, values_by_state)):
        a, b = [], []
        for rep in range(repeats):
            if (rep + tick) % 2:
                b.append(_bench_call(adaptive_sum_squares, values, loops))
                a.append(_bench_call(baseline, values, loops))
            else:
                a.append(_bench_call(baseline, values, loops))
                b.append(_bench_call(adaptive_sum_squares, values, loops))
        points.append(Observation(tick, time.perf_counter_ns()-stamp,
                                  state, statistics.median(a), statistics.median(b),
                                  route(values), True))
    check_path(points)
    result = {"schema": "alexpy-spacetime/v0.1", "candidate": "adaptive-opt-in",
              "scope": "single-process exploratory sum_squares only",
              "points": [{**asdict(point), "log_speedup": point.log_speedup}
                         for point in points],
              "derivatives": derivatives(points),
              "edges": edge_deltas(points),
              "multiscale": multiscale_signature(points),
              "phase_label_only": True}
    result["policy"] = "Correctness gate required; scale estimates are finite-sample; no automatic promotion"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlexPy space-time multi-scale QA")
    parser.add_argument("--hops", type=int, default=15)
    parser.add_argument("--path", choices=("targeted", "walk"), default="targeted")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loops", type=int, default=200)
    parser.add_argument("--json", type=Path, default=Path("spacetime.json"))
    args = parser.parse_args(argv)
    path = targeted_path() if args.path == "targeted" else planned_path(hops=args.hops)
    report = measure_trace(path, repeats=args.repeats, loops=args.loops)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"states": len(report["points"]),
                      "scales": report["multiscale"]["roughness_by_hop_scale"],
                      "output": str(args.json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
