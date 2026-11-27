"""Sparse, opt-in route memory for *real* verified range-query workloads.

The five-axis grid indexes measurements; it does not change arithmetic order
or waive mandatory dependency edges. All calibration is out of the hot path.
This is an ant/min-plus *analogy*, not a simulation of light or sound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median

from .range_routes import ROUTES, RangeWorkload


@dataclass
class Evidence:
    log_ratio_sum: float = 0.0
    samples: int = 0
    pheromone: float = 1.0
    last_epoch: int = 0


@dataclass
class Transition:
    setup_ns: list[float]
    tick_ns: list[float]


class RangeRouteMemory:
    """Two comparable selectors using the same correctness-gated memory.

    mode='ant': locally weighted, evaporating pheromone affinity.
    mode='wave': min-plus arrival on a *uniform* L1 feature grid.
    Transition costs from actual layout switching are stored separately: they
    apply only to plan-switch decisions, not to geometric adjacency weights.
    """

    def __init__(self, *, mode="wave", rho=.08, max_hops=5,
                 wave_penalty=.09, min_gain=1.04):
        if mode not in ("wave", "ant"):
            raise ValueError("unknown route memory mode")
        if not (0 < rho < 1) or type(max_hops) is not int or max_hops < 0:
            raise ValueError("invalid memory decay/radius")
        if (not math.isfinite(wave_penalty) or wave_penalty < 0
                or not math.isfinite(min_gain) or min_gain < 1):
            raise ValueError("invalid wave penalty/gain threshold")
        self.mode = mode
        self.rho = rho
        self.max_hops = max_hops
        self.wave_penalty = wave_penalty
        self.min_gain = min_gain
        self.epoch = 0
        self.matrix: dict[tuple[tuple[int,...], str], Evidence] = {}
        self.invalid: set[tuple[tuple[int,...], str]] = set()
        self.transitions: dict[tuple, Transition] = {}
        self.decisions: dict[tuple[int,...], str] = {}

    @staticmethod
    def distance(a, b):
        if len(a) != 5 or len(b) != 5:
            raise ValueError("states must have five coordinates")
        return sum(abs(i-j) for i,j in zip(a,b))

    def advance(self):
        self.epoch += 1
        self.decisions.clear()

    def observe(self, workload: RangeWorkload, route: str, *,
                elapsed_ns: float, direct_ns: float, verified: bool):
        if route not in ROUTES:
            raise ValueError("unsupported route")
        key = workload.cell(), route
        if verified is not True:
            self.invalid.add(key)
            self.matrix.pop(key, None)
            self.decisions.clear()
            raise ValueError("incorrect route quarantined")
        if key in self.invalid:
            raise ValueError("quarantined route")
        if (type(elapsed_ns) not in (float,int)
                or type(direct_ns) not in (float,int)
                or not math.isfinite(elapsed_ns)
                or not math.isfinite(direct_ns)
                or min(elapsed_ns,direct_ns) <= 0):
            raise ValueError("timing evidence must be positive finite")
        ratio = float(elapsed_ns) / float(direct_ns)
        entry = self.matrix.setdefault(key, Evidence(last_epoch=self.epoch))
        gain = max(.05, min(20., 1./ratio))
        entry.pheromone = min(64., max(.01, entry.pheromone*(1-self.rho)**
                                   (self.epoch-entry.last_epoch) * (1-self.rho) + gain))
        entry.last_epoch = self.epoch
        entry.log_ratio_sum += math.log(ratio)
        entry.samples += 1
        self.decisions.clear()

    def observe_switch(self, before: RangeWorkload, after: RangeWorkload,
                       old_route: str, new_route: str, *,
                       setup_ns: float, tick_ns: float, verified: bool):
        if old_route not in ROUTES or new_route not in ROUTES:
            raise ValueError("unknown route")
        if verified is not True:
            raise ValueError("unverified switch cannot enter memory")
        if (not all(type(x) in (float,int) and math.isfinite(x) and x >= 0
                    for x in (setup_ns,tick_ns))):
            raise ValueError("invalid switch timings")
        key = before.cell(),after.cell(),old_route,new_route
        entry = self.transitions.setdefault(key,Transition([],[]))
        entry.setup_ns.append(float(setup_ns))
        entry.tick_ns.append(float(tick_ns))

    def _scores(self, cell):
        nearby = [(self.distance(cell,source), source, route, e)
                  for (source,route),e in self.matrix.items()
                  if e.samples and (source,route) not in self.invalid]
        nearby = [(d,s,r,e) for d,s,r,e in nearby if d <= self.max_hops]
        scores = {} if (cell,"direct") in self.invalid else {"direct": 0.0}
        for route in ROUTES[1:]:
            if (cell,route) in self.invalid:
                continue
            candidates = [(d, e) for d,s,r,e in nearby if r == route]
            if not candidates:
                continue
            if self.mode == "wave":
                score = min(e.log_ratio_sum/e.samples + self.wave_penalty*d
                            for d,e in candidates)
            else:
                num = den = 0.0
                for d,e in candidates:
                    weight = (1/(1+d)**2) * max(.01, e.pheromone *
                                           (1-self.rho)**(self.epoch-e.last_epoch))
                    num += (e.log_ratio_sum/e.samples)*weight
                    den += weight
                score = num/den
            scores[route] = score
        return scores

    def choose(self, workload: RangeWorkload):
        cell = workload.cell()
        if cell in self.decisions:
            return self.decisions[cell]
        scores = self._scores(cell)
        if not scores:
            raise RuntimeError("all routes quarantined for this workload")
        winning = min(scores, key=lambda name: (scores[name], ROUTES.index(name)))
        # No speculative optimization without an expected >=4% end-to-end gain.
        if winning != "direct" and "direct" in scores and scores[winning] > -math.log(self.min_gain):
            winning = "direct"
        self.decisions[cell] = winning
        return winning

    def freeze(self, workload_list):
        return {w.cell():self.choose(w) for w in workload_list}

    def switch_cost(self,before,after,old_route,new_route):
        entry = self.transitions.get((before.cell(),after.cell(),old_route,new_route))
        if entry is None:
            return None
        return {"setup_ns":median(entry.setup_ns),"tick_ns":median(entry.tick_ns),
                "samples":len(entry.setup_ns)}

    def snapshot(self):
        return {"schema":"ALEXPY-RANGE-ROUTE-MEMORY/1", "mode":self.mode,
                "epoch":self.epoch,"cells":len({k[0] for k in self.matrix}),
                "observations":len(self.matrix),"quarantined":len(self.invalid),
                "real_switch_edges":len(self.transitions),
                "frozen_decisions":len(self.decisions)}
