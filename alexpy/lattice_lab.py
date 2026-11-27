"""Discrete workload-space/time lattice and held-out performance hypotheses.

Inspired by discretization in numerical science, not an implementation of
lattice quantum chromodynamics, a CPU simulator, or a causal identification
method. No candidate automatically enters the public dispatcher.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time

from .adaptive import adaptive_sum_squares, route
from .graph_search import CONTAINERS, LENGTHS, PROFILES, PHASES, State, make_values, neighbors

AXES = ("container", "length", "profile", "phase")
AXIS_VALUES = (CONTAINERS, LENGTHS, PROFILES, PHASES)


def coordinates(state: State) -> tuple[int, int, int, int]:
    """Discrete 4D configuration coordinates (time is a fifth axis)."""
    return tuple(values.index(getattr(state, axis))
                 for axis, values in zip(AXES, AXIS_VALUES))


def separation(a: State, b: State) -> int:
    return sum(abs(x - y) for x, y in zip(coordinates(a), coordinates(b)))


def estimate_container_bytes(values) -> int:
    """Approximate container + element footprint; not RSS, cache usage, or allocation.

    Shared element identities counted once to avoid counting repeated references
    as separate live objects. Excludes allocator overhead and other processes.
    """
    ids = set()
    total = sys.getsizeof(values)
    for value in values:
        if id(value) not in ids:
            ids.add(id(value))
            total += sys.getsizeof(value)
    return total


@dataclass(frozen=True)
class Cell:
    state: State
    epoch: int
    timestamp_ns: int
    baseline_ns: float
    candidate_ns: float
    footprint_bytes_est: int
    decision: str

    def __post_init__(self):
        if self.epoch < 0 or self.timestamp_ns < 0 or self.footprint_bytes_est < 0:
            raise ValueError("negative epoch/timestamp/footprint")
        if not all(math.isfinite(x) and x > 0
                   for x in (self.baseline_ns, self.candidate_ns)):
            raise ValueError("invalid latency")

    @property
    def log_speedup(self) -> float:
        return math.log(self.baseline_ns / self.candidate_ns)

    @property
    def key(self):
        return (self.epoch, self.state)


def validate_cells(cells):
    """Reject duplicate measurements and reversed timestamps within a state."""
    keys, by_state = set(), {}
    for c in cells:
        if c.key in keys:
            raise ValueError("duplicate lattice cell in epoch")
        keys.add(c.key)
        by_state.setdefault(c.state, []).append(c)
    for rows in by_state.values():
        rows = sorted(rows, key=lambda c: c.epoch)
        for left, right in zip(rows, rows[1:]):
            if right.timestamp_ns <= left.timestamp_ns:
                raise ValueError("nonmonotonic time for same state")
    return True


def lattice_edges(cells):
    """Spatial one-axis edges within epoch; temporal edges for same state.

    Do not connect sequentially measured *different* states as temporal edges.
    That would turn benchmarking order into fictitious physical dynamics.
    """
    validate_cells(cells)
    lookup = {c.key: c for c in cells}
    result = []
    for c in cells:
        for axis, other in neighbors(c.state):
            if c.state.id >= other.id:
                continue
            # An empty container has no numeric profile; changing only its
            # profile does not create a different input workload.
            if axis == "profile" and c.state.length == 0:
                continue
            next_cell = lookup.get((c.epoch, other))
            if next_cell is None:
                continue
            result.append({"kind": "space", "axis": axis, "epoch": c.epoch,
                           "from": c.state.id, "to": other.id,
                           "delta_log_speedup": next_cell.log_speedup-c.log_speedup,
                           "delta_estimated_bytes": next_cell.footprint_bytes_est-c.footprint_bytes_est})
    by_state = {}
    for c in cells:
        by_state.setdefault(c.state, []).append(c)
    for state, rows in by_state.items():
        ordered = sorted(rows, key=lambda c: c.epoch)
        for left, right in zip(ordered, ordered[1:]):
            if right.epoch != left.epoch + 1:
                continue
            elapsed = (right.timestamp_ns-left.timestamp_ns)/1e9
            result.append({"kind": "time", "axis": "wall_elapsed",
                           "from": state.id, "to": state.id, "epoch": right.epoch,
                           "delta_log_speedup": right.log_speedup-left.log_speedup,
                           "elapsed_seconds": elapsed,
                           "d_log_speedup_per_second": (right.log_speedup-left.log_speedup)/elapsed})
    return result


def multiscale_lattice(cells, scales=(1, 2, 4)):
    """Median spatial differences at Manhattan distances, separately by epoch.

    Sparse measurements may have no pairs at some scales. Result is not a
    fractal dimension and cannot establish self-similarity from 2 epochs.
    """
    validate_cells(cells)
    if not scales or len(set(scales)) != len(scales) or any(type(s) is not int or s < 1 for s in scales):
        raise ValueError("invalid scales")
    out = {}
    for epoch in sorted({c.epoch for c in cells}):
        subset = [c for c in cells if c.epoch == epoch]
        result = {}
        for scale in sorted(scales):
            deltas = [abs(x.log_speedup - y.log_speedup)
                      for i, x in enumerate(subset) for y in subset[i+1:]
                      if separation(x.state, y.state) == scale
                      and not (x.state.length == y.state.length == 0
                               and x.state.container == y.state.container
                               and x.state.phase == y.state.phase
                               and x.state.profile != y.state.profile)]
            result[str(scale)] = {"pair_count": len(deltas),
                                  "median_abs_log_delta": statistics.median(deltas) if deltas else None}
        out[str(epoch)] = result
    return out


def size_holdout(cells, *, holdout_lengths=(8, 1024)):
    """Compare multiscale-distance predictor to a global-median baseline.

    Split by *entire length groups* before fitting; never train on another
    container/profile of a held-out length. No train/test performance tuning.
    """
    validate_cells(cells)
    if not holdout_lengths or any(s not in LENGTHS for s in holdout_lengths):
        raise ValueError("unknown holdout sizes")
    holdouts = set(holdout_lengths)
    train = [c for c in cells if c.state.length not in holdouts]
    test = [c for c in cells if c.state.length in holdouts]
    if not train or not test:
        raise ValueError("both train and held-out samples required")
    prior = statistics.median(c.log_speedup for c in train)

    def predict(cell, radius):
        # Same epoch + container + arithmetic profile + phase; size differs.
        # Time is never used as an indirect leakage channel.
        source = [(separation(c.state, cell.state), c.log_speedup)
                  for c in train if c.epoch == cell.epoch
                  and c.state.container == cell.state.container
                  and c.state.profile == cell.state.profile
                  and c.state.phase == cell.state.phase]
        nearby = [(d, v) for d, v in source if 0 < d <= radius]
        if not nearby:
            return prior, False
        weights = [1.0/(d*d) for d, _ in nearby]
        return sum(w*v for w, (_,v) in zip(weights, nearby))/sum(weights), True

    out = []
    for item in test:
        p1, coverage1 = predict(item, 1)
        pm, coveragem = predict(item, 4)
        out.append({"state": item.state.id, "epoch": item.epoch,
                    "actual_log_speedup": item.log_speedup,
                    "prior_log_speedup": prior,
                    "one_hop_prediction": p1,
                    "multiscale_prediction": pm,
                    "one_hop_neighbor_covered": coverage1,
                    "multiscale_neighbor_covered": coveragem,
                    "regression_10pct_observed": item.log_speedup < -math.log(1.1),
                    "regression_10pct_predicted": pm < -math.log(1.1)})

    metrics = {}
    for name, key in (("prior", "prior_log_speedup"),
                      ("one_hop", "one_hop_prediction"),
                      ("multiscale", "multiscale_prediction")):
        metrics[name] = {"mae_log_speedup": statistics.mean(abs(r[key]-r["actual_log_speedup"]) for r in out)}
    actual_regressions = sum(r["regression_10pct_observed"] for r in out)
    false_negatives = sum(r["regression_10pct_observed"] and not r["regression_10pct_predicted"] for r in out)
    return {"holdout_lengths": sorted(holdouts),
            "train_count": len(train), "holdout_count": len(test),
            "actual_regressions": actual_regressions,
            "missed_regressions": false_negatives,
            "coverage_multiscale": sum(r["multiscale_neighbor_covered"] for r in out),
            "metrics": metrics, "predictions": out,
            "evidence": "EXPLORATORY_ONLY_NO_PROMOTION",
            "interpretation": "observational held-out MAE, not causal effect or CPU-cache measurement"}


def _time(fn, values, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(values)
    return (time.perf_counter_ns()-start)/loops


def collect(*, epochs=2, repeats=5, loops=25, lengths=(0,1,8,16,1024,10000),
            containers=("list", "tuple"), profiles=("small", "huge")):
    """Paired alternating-order timing across grid cells; correctness first."""
    if epochs < 1 or repeats < 1 or loops < 1:
        raise ValueError("epochs/repeats/loops must be positive")
    states = [State(c, n, p) for c in containers for n in lengths for p in profiles]
    if len(set(states)) != len(states) or not states:
        raise ValueError("duplicate or empty state selection")
    prepared = []
    for state in states:
        values = make_values(state)
        # All types in current grid are exact ints; fail closed if assumptions change.
        mirror1 = sum(x*x for x in values)
        mirror2 = sum(pow(x, 2) for x in values)
        candidate = adaptive_sum_squares(values)
        if not (type(mirror1) is int and mirror1 == mirror2 == candidate):
            raise ValueError("mirror disagreement: refuse timing")
        prepared.append((state, values, estimate_container_bytes(values)))
    baseline = lambda values: sum(x*x for x in values)
    cells = []
    for epoch in range(epochs):
        # Alternating traversal order reduces systematic chronological order bias.
        schedule = prepared if epoch % 2 == 0 else list(reversed(prepared))
        for state, values, footprint in schedule:
            samples_a, samples_b = [], []
            for rep in range(repeats):
                if (epoch+rep)%2:
                    samples_b.append(_time(adaptive_sum_squares, values, loops))
                    samples_a.append(_time(baseline, values, loops))
                else:
                    samples_a.append(_time(baseline, values, loops))
                    samples_b.append(_time(adaptive_sum_squares, values, loops))
            cells.append(Cell(state, epoch, time.perf_counter_ns(),
                              statistics.median(samples_a), statistics.median(samples_b),
                              footprint, route(values)))
    validate_cells(cells)
    return cells


def analyze(cells, holdout_lengths=(8,1024)):
    return {"schema": "alexpy-lattice-lab/v0.1",
            "grid_shape": [len(v) for v in AXIS_VALUES],
            "axes": [*AXES, "epoch"],
            "cells": [{**asdict(c), "coordinates": [*coordinates(c.state), c.epoch],
                       "log_speedup": c.log_speedup} for c in cells],
            "edges": lattice_edges(cells),
            "multiscale": multiscale_lattice(cells),
            "size_holdout": size_holdout(cells, holdout_lengths=holdout_lengths),
            "guardrail": "observational: no CPU-cache inference, no fractal dimension, no automatic optimization"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlexPy grid/lattice holdout QA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=25)
    parser.add_argument("--json", type=Path, default=Path("lattice.json"))
    args = parser.parse_args(argv)
    report = analyze(collect(epochs=args.epochs, repeats=args.repeats, loops=args.loops))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    p = report["size_holdout"]
    print(json.dumps({"cells": len(report["cells"]), "edges": len(report["edges"]),
                      "holdout": p["holdout_count"], "mae": p["metrics"],
                      "regression_misses": p["missed_regressions"],
                      "file": str(args.json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
