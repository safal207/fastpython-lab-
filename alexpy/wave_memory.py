"""Opt-in light/sound-inspired wavefront selector for verified execution plans.

This is a graph-algorithm analogy, NOT an electromagnetic/acoustic solver.
The safety-critical dependency DAG is never reordered; only complete,
previously correctness-gated *alternative* plans may be proposed.

'light': a multi-source earliest-arrival wavefront (Dijkstra) on the integer
        workload grid, with nonnegative edge penalties and per-source
        measured logarithmic latency ratios.
'sound': bounded distance-decayed pooling of independent measured evidence.
        This is an acoustic-inspired attenuation heuristic, not wave physics.

Unknown regions fail closed to 'manual'. Training, freezing and benchmarking
are offline operations. Public alexpy.sum_squares() is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

from .ant_memory import ROUTES, Workload


@dataclass
class _Evidence:
    log_gain_sum: float = 0.0
    samples: int = 0
    last_epoch: int = 0


class WaveMemoryGraph:
    """Sparse measured state/action memory plus a compiled wavefront policy."""

    def __init__(self, *, mode: str = "light", attenuation: float = .18,
                 age_penalty: float = .02, max_hops: int = 4,
                 min_gain: float = 1.05):
        if mode not in ("light", "sound"):
            raise ValueError("mode must be light or sound")
        if (not math.isfinite(attenuation) or attenuation <= 0
                or not math.isfinite(age_penalty) or age_penalty < 0
                or type(max_hops) is not int or not 0 <= max_hops <= 8
                or not math.isfinite(min_gain) or min_gain < 1):
            raise ValueError("invalid wave parameters")
        self.mode, self.attenuation = mode, attenuation
        self.age_penalty, self.max_hops, self.min_gain = age_penalty, max_hops, min_gain
        self.epoch = 0
        self._measurements: dict[tuple[tuple[int, ...], str], _Evidence] = {}
        self._invalid: set[tuple[tuple[int, ...], str]] = set()
        self._decisions: dict[tuple[tuple[int, ...], tuple[str, ...]], str] = {}

    @staticmethod
    def _check_routes(allowed: Iterable[str]) -> tuple[str, ...]:
        choices = tuple(dict.fromkeys(allowed))
        if not choices or any(r not in ROUTES for r in choices):
            raise ValueError("only nonempty sets of registered plans are allowed")
        return choices

    def advance(self, steps: int = 1) -> None:
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be positive exact int")
        self.epoch += steps
        self._decisions.clear()

    def mark_invalid(self, workload: Workload, route: str) -> None:
        self._check_routes((route,))
        key = (workload.cell(), route)
        self._invalid.add(key)
        self._measurements.pop(key, None)
        self._decisions.clear()

    def observe(self, workload: Workload, route: str, *, elapsed_ns: float,
                reference_ns: float, verified: bool) -> None:
        self._check_routes((route,))
        if verified is not True:
            self.mark_invalid(workload, route)
            raise ValueError("unverified action quarantined")
        if (not math.isfinite(elapsed_ns) or not math.isfinite(reference_ns)
                or elapsed_ns <= 0 or reference_ns <= 0):
            raise ValueError("measurements must be positive and finite")
        key = (workload.cell(), route)
        if key in self._invalid:
            raise ValueError("action quarantined")
        gain = reference_ns / elapsed_ns
        record = self._measurements.setdefault(key, _Evidence())
        record.log_gain_sum += math.log(gain)
        record.samples += 1
        record.last_epoch = self.epoch
        self._decisions.clear()

    @staticmethod
    def _distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return sum(abs(x - y) for x, y in zip(a, b))

    def _sources(self, target: tuple[int, ...], route: str):
        for (position, action), evidence in self._measurements.items():
            if (action != route or (position, action) in self._invalid
                    or self._distance(position, target) > self.max_hops):
                continue
            loss = -evidence.log_gain_sum / evidence.samples
            age = self.epoch - evidence.last_epoch
            yield position, loss + self.age_penalty * age, evidence.samples

    def _light_arrival_reference(self, target: tuple[int, ...], route: str) -> float:
        """Independent Dijkstra witness for min-plus wave propagation."""
        best: dict[tuple[int, ...], float] = {}
        heap = []
        for position, seed, _ in self._sources(target, route):
            if seed < best.get(position, math.inf):
                best[position] = seed
                heapq.heappush(heap, (seed, position))
        while heap:
            arrival, point = heapq.heappop(heap)
            if arrival != best[point]:
                continue
            if point == target:
                return arrival
            for axis in range(len(point)):
                for delta in (-1, 1):
                    new_value = point[axis] + delta
                    if new_value < 0:
                        continue
                    nxt = point[:axis] + (new_value,) + point[axis+1:]
                    if self._distance(nxt, target) > self.max_hops:
                        continue
                    cost = arrival + self.attenuation
                    if cost < best.get(nxt, math.inf):
                        best[nxt] = cost
                        heapq.heappush(heap, (cost, nxt))
        return math.inf

    def _light_arrival(self, target: tuple[int, ...], route: str) -> float:
        """Compile uniform-grid Dijkstra to min-plus over the sparse sources.

        On a unit axis-neighbor grid, the shortest path length between two
        cells is their Manhattan distance. With a constant nonnegative edge
        cost this is EXACTLY the same min-plus wave arrival as Dijkstra, but
        requires no heap or ephemeral grid allocation in the fast path.
        _light_arrival_reference remains available to test the equivalence.
        """
        return min((seed + self.attenuation * self._distance(point, target)
                    for point, seed, _ in self._sources(target, route)),
                   default=math.inf)

    def _sound_arrival(self, target: tuple[int, ...], route: str) -> float:
        """Damped multi-source evidence; return effective log-latency loss."""
        signal = 0.0
        weighted_loss = 0.0
        for position, loss, samples in self._sources(target, route):
            distance = self._distance(position, target)
            strength = math.exp(-self.attenuation * distance) * samples / (samples + 2)
            signal += strength
            weighted_loss += strength * loss
        if not signal:
            return math.inf
        # Penalize unsupported fields so one remote high-gain sample does not
        # win automatically. Averages, rather than summing gains, resist bias
        # from repeated training on a single route.
        return weighted_loss / signal + self.attenuation / (1 + signal)

    def field(self, workload: Workload, *, allowed: Iterable[str] = ROUTES) -> dict[str, float]:
        routes = self._check_routes(allowed)
        target = workload.cell()
        losses = {}
        for route in routes:
            if (target, route) in self._invalid:
                continue
            if route == "manual":
                losses[route] = 0.0  # strong measured-control prior
            else:
                value = (self._light_arrival(target, route) if self.mode == "light"
                         else self._sound_arrival(target, route))
                if math.isfinite(value):
                    losses[route] = value
        if not losses:
            raise RuntimeError("no verified plan available; manual control absent")
        return losses

    def choose(self, workload: Workload, *, allowed: Iterable[str] = ROUTES) -> str:
        routes = self._check_routes(allowed)
        key = (workload.cell(), routes)
        if key in self._decisions:
            return self._decisions[key]
        losses = self.field(workload, allowed=routes)
        route = min(losses, key=lambda r: (losses[r], ROUTES.index(r)))
        if "manual" in losses and route != "manual":
            if losses[route] > -math.log(self.min_gain):
                route = "manual"
        self._decisions[key] = route
        return route

    def freeze(self, workloads: Iterable[Workload], *, allowed: Iterable[str] = ROUTES) -> tuple[str, ...]:
        """Precompile lookup before holdout execution; never train here."""
        routes = self._check_routes(allowed)
        return tuple(self.choose(w, allowed=routes) for w in workloads)

    def snapshot(self) -> dict:
        return {"schema": "ALEXPY-WAVE-MEMORY/1", "mode": self.mode,
                "epoch": self.epoch, "state_action_edges": len(self._measurements),
                "quarantined": len(self._invalid), "compiled_decisions": len(self._decisions),
                "observations": [{"cell": list(cell), "route": route,
                                  "samples": e.samples,
                                  "mean_log_gain": e.log_gain_sum / e.samples,
                                  "last_epoch": e.last_epoch}
                                 for (cell, route), e in sorted(self._measurements.items())]}


__all__ = ["WaveMemoryGraph"]
