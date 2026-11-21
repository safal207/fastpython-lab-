"""Experimental sparse, ant-inspired memory over *alternative verified plans*.

This is deliberately NOT a replacement for the dependency DAG: required DAG
edges must always be evaluated. The matrix stores only performance evidence for
preapproved, semantically equivalent end-to-end execution plans.

Cells are coarse log2 workload coordinates, not raw Python object identities.
Missing cells are not allocated. Old evidence decays in experimental epochs.
A failed correctness gate permanently quarantines that cell/route until a new
memory model is explicitly created. No runtime auto-promotion is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Iterable

ROUTES = ("manual", "partition", "compiled", "dense")


@dataclass(frozen=True)
class Workload:
    size: int
    width: int
    changes: int
    ticks: int
    reads: int

    def __post_init__(self):
        values = (self.size, self.width, self.changes, self.ticks, self.reads)
        if any(type(n) is not int or n <= 0 for n in values):
            raise ValueError("workload dimensions must be positive exact ints")
        if self.changes > self.size:
            raise ValueError("changes must not exceed size")

    def cell(self) -> tuple[int, ...]:
        return tuple((n.bit_length() - 1) for n in
                     (self.size, self.width, self.changes, self.ticks, self.reads))


@dataclass
class Trail:
    tau: float = 1.0
    count: int = 0
    last_epoch: int = 0
    log_gain_sum: float = 0.0


class AntMemoryGraph:
    """A bounded, sparse pheromone matrix and adjacent-state transition log.

    choose() is a *proposal* and not a correctness proof. The caller is solely
    responsible for checking that all allowed routes are valid for the concrete
    workload before it executes the selected route. Runtime learning is opt-in.
    """

    def __init__(self, *, rho: float = .12, alpha: float = 1.0,
                 max_distance: int = 3, neighbors: int = 3):
        if not (0 < rho < 1 and 0 <= alpha <= 4):
            raise ValueError("invalid pheromone parameters")
        if type(max_distance) is not int or max_distance < 0:
            raise ValueError("max_distance must be nonnegative")
        if type(neighbors) is not int or neighbors <= 0:
            raise ValueError("neighbors must be positive")
        self.rho, self.alpha = rho, alpha
        self.max_distance, self.neighbors = max_distance, neighbors
        self.epoch = 0
        self._matrix: dict[tuple[tuple[int, ...], str], Trail] = {}
        self._blocked: set[tuple[tuple[int, ...], str]] = set()
        self._transitions: dict[tuple[tuple[int, ...], tuple[int, ...], str], int] = {}
        # Compile deterministic policies once; invalidate on any evidence change.
        self._policy_cache: dict[tuple[tuple[int, ...], tuple[str, ...]], str] = {}

    @staticmethod
    def _check_route(route: str) -> None:
        if route not in ROUTES:
            raise ValueError("route is not a registered safe candidate")

    def advance(self, steps: int = 1) -> None:
        if type(steps) is not int or steps <= 0:
            raise ValueError("epoch increment must be positive")
        self.epoch += steps
        self._policy_cache.clear()

    def _strength(self, trail: Trail) -> float:
        # Lazy evaporation toward the zero-evidence floor. No mutations on read.
        return max(.02, trail.tau * (1 - self.rho) ** (self.epoch - trail.last_epoch))

    def mark_invalid(self, state: Workload, route: str) -> None:
        """Quarantine the action on a failed equivalence check, never reward it."""
        self._check_route(route)
        key = (state.cell(), route)
        self._blocked.add(key)
        self._matrix.pop(key, None)
        self._policy_cache.clear()

    def observe(self, state: Workload, route: str, *, elapsed_ns: float,
                reference_ns: float, verified: bool) -> None:
        """Reinforce only after an independent correctness gate passed.

        `reference_ns` is the same state's measured time for a fixed manual
        algorithm. We clip ratios to prevent unstable microbenchmarks from
        permanently dominating the pheromone matrix.
        """
        self._check_route(route)
        if verified is not True:
            self.mark_invalid(state, route)
            raise ValueError("unverified/incorrect route cannot train")
        if (not math.isfinite(elapsed_ns) or not math.isfinite(reference_ns)
                or elapsed_ns <= 0 or reference_ns <= 0):
            raise ValueError("times must be positive finite numbers")
        key = (state.cell(), route)
        if key in self._blocked:
            raise ValueError("quarantined route cannot train")
        gain = reference_ns / elapsed_ns
        reward = max(.05, min(8.0, gain))
        trail = self._matrix.setdefault(key, Trail(last_epoch=self.epoch))
        trail.tau = min(64.0, max(.02, (1 - self.rho) * self._strength(trail) + reward))
        trail.log_gain_sum += math.log(gain)
        trail.count += 1
        trail.last_epoch = self.epoch
        self._policy_cache.clear()

    @staticmethod
    def _distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return sum(abs(x - y) for x, y in zip(a, b))

    def scores(self, state: Workload, *, allowed: Iterable[str] = ROUTES) -> dict[str, float]:
        candidates = tuple(dict.fromkeys(allowed))
        if not candidates:
            raise ValueError("no candidate plans")
        for route in candidates:
            self._check_route(route)
        cell = state.cell()
        candidates = tuple(r for r in candidates if (cell, r) not in self._blocked)
        if not candidates:
            raise RuntimeError("all candidate plans quarantined")
        learned = {coordinate for (coordinate, _), t in self._matrix.items() if t.count}
        selected = sorted(
            ((self._distance(cell, candidate), candidate) for candidate in learned),
            key=lambda x: (x[0], x[1]))
        selected = [(d, c) for d, c in selected if d <= self.max_distance][:self.neighbors]
        # Return a single conservative score when this space is unsupported.
        if not selected:
            return {r: 1.0 if r == "manual" else 0.0 for r in candidates}
        result = {}
        for route in candidates:
            weighted, weight = 0.0, 0.0
            for distance, candidate in selected:
                trail = self._matrix.get((candidate, route))
                if trail is None or not trail.count or (candidate, route) in self._blocked:
                    continue
                factor = 1 / (1 + distance)**2
                weighted += factor * self._strength(trail)
                weight += factor
            result[route] = (weighted / weight) ** self.alpha if weight else 0.0
        if not any(result.values()):
            return {r: 1.0 if r == "manual" else 0.0 for r in candidates}
        return result

    def choose(self, state: Workload, *, allowed: Iterable[str] = ROUTES,
               rng: random.Random | None = None) -> str:
        """Exploit highest score, or sample ant-style when seeded RNG is supplied."""
        routes = tuple(dict.fromkeys(allowed))
        if rng is None:
            key = (state.cell(), routes)
            cached = self._policy_cache.get(key)
            if cached is not None:
                return cached
        weights = self.scores(state, allowed=routes)
        if rng is None:
            # The strong manual control wins ties, preventing arbitrary novelty.
            chosen = max(weights, key=lambda r: (weights[r], -ROUTES.index(r)))
            self._policy_cache[key] = chosen
            return chosen
        total = sum(weights.values())
        if total <= 0:
            return next(iter(weights))
        draw = rng.random() * total
        for route, weight in weights.items():
            draw -= weight
            if draw <= 0:
                return route
        return next(reversed(weights))

    def trace(self, before: Workload, after: Workload, route: str) -> None:
        """Remember visited state transitions without assigning fake speedups."""
        self._check_route(route)
        key = (before.cell(), after.cell(), route)
        self._transitions[key] = self._transitions.get(key, 0) + 1

    def snapshot(self) -> dict:
        """JSON-safe auditable memory; no code or arbitrary object loading."""
        return {
            "schema": "ALEXPY-ANT-MEMORY/1", "epoch": self.epoch,
            "rho": self.rho, "alpha": self.alpha,
            "cells": len({key[0] for key in self._matrix}),
            "nonzero_edges": len(self._matrix),
            "blocked_edges": len(self._blocked),
            "state_transitions": len(self._transitions),
            "compiled_policy_cells": len(self._policy_cache),
            "trails": [
                {"cell": list(cell), "route": route, "tau": round(self._strength(t), 9),
                 "samples": t.count, "mean_log_gain": t.log_gain_sum / t.count}
                for (cell, route), t in sorted(self._matrix.items()) if t.count
            ],
            "transitions": [
                {"from": list(a), "to": list(b), "route": r, "visits": n}
                for (a,b,r), n in sorted(self._transitions.items())
            ],
            "quarantined": [
                {"cell": list(cell), "route": route}
                for cell, route in sorted(self._blocked)
            ],
        }


__all__ = ["ROUTES", "Workload", "AntMemoryGraph"]
