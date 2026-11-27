"""Experimental heterogeneous wavefront over *verified alternative plans*.

The state-space graph is a five-axis nonnegative, directed cost lattice.
Evidence from correctness-gated plan timing supplies source potentials; an
optional independently observed *transition* ratio changes an adjacent edge.
The router chooses whole plans; it NEVER skips mandatory dependency DAG nodes.

This is a computational analogy, not a simulation of optical/acoustic waves,
and heuristic edge weights are NOT measured cache-miss rates. Unknown cells
fall back to the strong manual plan, unless the caller explicitly removes it.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

from .ant_memory import ROUTES, Workload
from .wave_memory import WaveMemoryGraph


@dataclass
class _EdgeEvidence:
    sum_ratio: float = 0.0
    count: int = 0


class HeterogeneousWaveMemory(WaveMemoryGraph):
    """Bounded min-plus arrival and distance-attenuated evidence pooling.

    `max_hops` is the L1 radius around the requested state. Dijkstra solves
    the nonuniform *directed* graph induced by that radius, not an unbounded
    graph. Scoring is offline; freeze() caches selected plans for fast lookup.
    """

    def __init__(self, *, mode: str = "light", axis_weights=(.24, .20, .28, .22, .14),
                 size_slope: float = .15, size_threshold: int = 9,
                 **kwargs):
        super().__init__(mode=mode, **kwargs)
        weights = tuple(axis_weights)
        if (len(weights) != 5 or any(type(w) not in (int, float)
                    or not math.isfinite(w) or w <= 0 for w in weights)
                or type(size_slope) not in (int, float)
                or not math.isfinite(size_slope) or size_slope < 0
                or type(size_threshold) is not int or size_threshold < 0):
            raise ValueError("invalid heterogeneous axis weights")
        self.axis_weights = tuple(float(w) for w in weights)
        self.size_slope = float(size_slope)
        self.size_threshold = size_threshold
        self._edges: dict[tuple[tuple[int, ...], tuple[int, ...], str], _EdgeEvidence] = {}
        self._blocked_edges: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
        # Cache bounded reverse fields; if there are no route-specific edges,
        # different plans share exactly the same physical search geometry.
        self._distance_cache: dict[tuple, tuple[dict, dict]] = {}

    @staticmethod
    def _axis(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        if len(a) != 5 or len(b) != 5:
            raise ValueError("states must have five coordinates")
        changed = [i for i in range(5) if a[i] != b[i]]
        if (len(changed) != 1 or abs(a[changed[0]] - b[changed[0]]) != 1
                or min(b) < 0 or min(a) < 0):
            raise ValueError("edge must join adjacent nonnegative grid cells")
        return changed[0]

    def observe_transition(self, before: Workload, after: Workload, route: str,
                           *, elapsed_ns: float, reference_ns: float,
                           verified: bool) -> None:
        """Accept ONLY independently timed and correctness-verified transitions.

        Ratios are bounded to guard against one noisy timestamp overwhelming
        route selection. Ratio=1 means the preregistered axis model is kept.
        This metadata is independent of measured route-performance evidence.
        """
        self._check_routes((route,))
        a, b = before.cell(), after.cell()
        self._axis(a, b)
        key = (a, b, route)
        if verified is not True:
            self._blocked_edges.add(key)
            self._edges.pop(key, None)
            self._distance_cache.clear()
            self._decisions.clear()
            raise ValueError("unverified transition quarantined")
        if (type(elapsed_ns) not in (float, int)
                or type(reference_ns) not in (float, int)
                or not math.isfinite(elapsed_ns) or not math.isfinite(reference_ns)
                or elapsed_ns <= 0 or reference_ns <= 0):
            raise ValueError("transition timings must be positive and finite")
        if key in self._blocked_edges:
            raise ValueError("quarantined transition cannot be rewarded")
        ratio = min(20.0, max(.05, float(elapsed_ns) / float(reference_ns)))
        sample = self._edges.setdefault(key, _EdgeEvidence())
        sample.count += 1
        sample.sum_ratio += ratio
        self._distance_cache.clear()
        self._decisions.clear()

    def edge_cost(self, a: tuple[int, ...], b: tuple[int, ...], route: str) -> float:
        self._check_routes((route,))
        axis = self._axis(a, b)
        key = (a, b, route)
        if key in self._blocked_edges:
            return math.inf
        size_pressure = max(0, max(a[0], b[0]) - self.size_threshold)
        cost = self.axis_weights[axis] * (1 + self.size_slope * size_pressure)
        evidence = self._edges.get(key)
        if evidence is not None:
            cost *= evidence.sum_ratio / evidence.count
        return cost

    def _within(self, position: tuple[int, ...], target: tuple[int, ...]) -> bool:
        return self._distance(position, target) <= self.max_hops

    def _reverse_distances(self, target: tuple[int, ...], route: str,
                           stops: Iterable[tuple[int, ...]] = ()):
        """Dijkstra on reversed directed edges, giving cost(source -> target).

        Bound the search to a finite L1 ball; no fake direct Manhattan shortcut
        is valid for heterogeneous or directed edge costs.
        """
        requested = tuple(sorted(set(stops)))
        shared = not self._edges and not self._blocked_edges
        cache_key = (target, "*" if shared else route, requested)
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            return cached
        remaining = set(requested)
        best = {target: 0.0}
        successor: dict[tuple[int, ...], tuple[int, ...]] = {}
        queue = [(0.0, target)]
        while queue:
            dist, current = heapq.heappop(queue)
            if dist != best[current]:
                continue
            remaining.discard(current)
            if requested and not remaining:
                break
            for axis in range(5):
                for step in (-1, 1):
                    x = current[axis] + step
                    if x < 0:
                        continue
                    neighbour = current[:axis] + (x,) + current[axis + 1:]
                    if not self._within(neighbour, target):
                        continue
                    edge = self.edge_cost(neighbour, current, route)
                    candidate = dist + edge
                    if candidate < best.get(neighbour, math.inf):
                        best[neighbour] = candidate
                        successor[neighbour] = current
                        heapq.heappush(queue, (candidate, neighbour))
        if len(self._distance_cache) >= 96:
            self._distance_cache.pop(next(iter(self._distance_cache)))
        self._distance_cache[cache_key] = (best, successor)
        return best, successor

    def _light_arrival(self, target: tuple[int, ...], route: str) -> float:
        sources = tuple(self._sources(target, route))
        if not sources:
            return math.inf
        best, _ = self._reverse_distances(target, route, (pos for pos, _, _ in sources))
        return min((seed + best[pos] for pos, seed, _ in sources
                    if pos in best), default=math.inf)

    def _sound_arrival(self, target: tuple[int, ...], route: str) -> float:
        sources = tuple(self._sources(target, route))
        if not sources:
            return math.inf
        best, _ = self._reverse_distances(target, route, (pos for pos, _, _ in sources))
        total, weighted = 0.0, 0.0
        for pos, loss, count in sources:
            if pos not in best:
                continue
            strength = math.exp(-best[pos]) * count / (count + 2)
            total += strength
            weighted += loss * strength
        if not total:
            return math.inf
        return weighted / total + self.attenuation / (1 + total)

    def arrival_path(self, workload: Workload, route: str):
        """Explain the winning measured source and the lowest-cost state path.

        Returns None if there is no eligible source. This is not a Python DAG
        execution path; it is a route through the *configuration search grid*.
        """
        self._check_routes((route,))
        target = workload.cell()
        sources = tuple(self._sources(target, route))
        if not sources:
            return None
        distances, successor = self._reverse_distances(target, route, (pos for pos, _, _ in sources))
        options = [(seed + distances[pos], pos)
                   for pos, seed, _ in sources
                   if pos in distances]
        if not options:
            return None
        score, source = min(options)
        path = [source]
        while path[-1] != target:
            path.append(successor[path[-1]])
        return {"loss": score, "cells": [list(cell) for cell in path],
                "edge_cost": distances[source]}

    def snapshot(self) -> dict:
        basic = super().snapshot()
        basic.update({
            "schema": "ALEXPY-HETEROGENEOUS-WAVE/1",
            "axis_weights": list(self.axis_weights),
            "size_slope": self.size_slope,
            "size_threshold": self.size_threshold,
            "observed_transition_edges": len(self._edges),
            "compiled_distance_fields": len(self._distance_cache),
            "blocked_transition_edges": len(self._blocked_edges),
            "transitions": [
                {"from": list(a), "to": list(b), "route": route,
                 "samples": sample.count,
                 "mean_ratio": sample.sum_ratio / sample.count}
                for (a, b, route), sample in sorted(self._edges.items())
            ],
        })
        return basic


__all__ = ["HeterogeneousWaveMemory"]
