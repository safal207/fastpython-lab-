"""Research-only, payback-aware performance-calibration probe controller.

It sees only paid observations of the currently-best route, never the hidden
state trajectory. Rates are estimates, not physical hazards. The policy is
frozen before the seeded heldout experiment and is NOT active in AlexPy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite, sqrt

from .calibration_gate import TrustEvidence, probe_decision


@dataclass(frozen=True, slots=True)
class ProbeCosts:
    probe: float = 2.0
    stale: float = 7.0
    missed: float = 3.0
    min_interval: int = 8
    max_interval: int = 128

    def __post_init__(self):
        if not all(isfinite(x) and x >= 0 for x in (self.probe,self.stale,self.missed)):
            raise ValueError("cost values must be finite and nonnegative")
        if (type(self.min_interval) is not int or type(self.max_interval) is not int
            or self.min_interval<1 or self.max_interval<self.min_interval):
            raise ValueError("invalid interval bounds")


class DriftSentinel:
    """EWMA of surprises per observed time; cost-aware cadence.

    No future labels, regime IDs or per-tick hidden signals are accepted by
    this API. An unobserved even number of flips can remain undetected.
    """
    __slots__=("costs","rate","last_observed","last_probe","next_probe","probe_count",
               "observed_flips","alpha","floor")

    def __init__(self, costs=ProbeCosts(), *, alpha=.35, floor=.00005):
        if not isfinite(alpha) or not 0<alpha<=1:
            raise ValueError("alpha must be in (0,1]")
        if not isfinite(floor) or not 0<=floor<=1:
            raise ValueError("invalid floor")
        self.costs=costs;self.rate=floor;self.floor=floor;self.alpha=alpha
        self.last_observed=None;self.last_probe=None;self.next_probe=0
        self.probe_count=0;self.observed_flips=0

    def due(self, tick: int) -> bool:
        return tick>=self.next_probe

    def observe(self, tick: int, actual_best: bool):
        if type(tick) is not int or tick<0 or type(actual_best) is not bool:
            raise ValueError("invalid probe observation")
        if self.last_probe is not None and tick<=self.last_probe:
            raise ValueError("observations must move forward in time")
        if tick<self.next_probe:
            raise ValueError("cannot see unpaid observations")
        if self.last_probe is not None:
            delta=tick-self.last_probe
            flip=int(actual_best!=self.last_observed)
            self.observed_flips+=flip
            self.rate=max(self.floor,(1-self.alpha)*self.rate+self.alpha*flip/delta)
        self.last_observed=actual_best
        self.last_probe=tick
        self.probe_count+=1
        avg_consequence=(self.costs.stale+self.costs.missed)/2
        estimated_regret=max(1e-12,self.rate*avg_consequence)
        raw=sqrt(2*self.costs.probe/estimated_regret) if self.costs.probe else 0
        interval=max(self.costs.min_interval,min(self.costs.max_interval,round(raw)))
        self.next_probe=tick+interval
        return interval


class BardoTrustControl:
    """Frozen v2.5 trust/cadence control as a comparison, not a new oracle."""
    __slots__=("costs","count","flips","next_probe","last_observed","last_probe","fallback")

    def __init__(self,costs=ProbeCosts()):
        self.costs=costs;self.count=256;self.flips=0;self.next_probe=0
        self.last_observed=None;self.last_probe=None;self.fallback=False

    def due(self,tick:int)->bool:return tick>=self.next_probe

    def observe(self,tick:int,actual_best:bool):
        if type(tick) is not int or tick<0 or type(actual_best) is not bool:
            raise ValueError("invalid probe observation")
        if tick < self.next_probe or (self.last_probe is not None and tick<=self.last_probe):
            raise ValueError("cannot see unpaid observations")
        if self.last_probe is not None and actual_best!=self.last_observed:
            self.flips+=1
        self.count+=1
        self.last_observed=actual_best
        self.last_probe=tick
        ev=TrustEvidence(self.count,0,min(1.,self.flips/8),0.)
        decision=probe_decision(ev,probe_cost=self.costs.probe,
            miss_loss=self.costs.stale,min_interval=self.costs.min_interval,
            max_interval=self.costs.max_interval)
        self.fallback=decision.fallback
        self.next_probe=tick+decision.interval
        return decision.interval

    def use_fast(self,tick:int)->bool:
        if self.last_probe is None:return False
        ev=TrustEvidence(self.count,tick-self.last_probe,min(1.,self.flips/8),0.)
        return bool(self.last_observed) and not probe_decision(
            ev,probe_cost=self.costs.probe,miss_loss=self.costs.stale,
            min_interval=self.costs.min_interval,max_interval=self.costs.max_interval
        ).fallback
