"""Experimental Bardo-inspired trust gate for AlexPy calibration evidence.

Independent adaptation of a published engineering heuristic, not a verified
Bardo interoperability layer. No hidden regime is provided to the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt


@dataclass(frozen=True, slots=True)
class TrustEvidence:
    count: int
    age: int
    drift: float
    error: float
    matching_host: bool = True

    def __post_init__(self):
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be nonnegative")
        if type(self.age) is not int or self.age < 0:
            raise ValueError("age must be nonnegative")
        if any(not isfinite(x) or not 0 <= x <= 1 for x in (self.drift, self.error)):
            raise ValueError("drift and error must be finite probabilities")
        if type(self.matching_host) is not bool:
            raise TypeError("matching_host must be boolean")


def trust_score(evidence: TrustEvidence, *, prior=32., half_life=512.) -> float:
    """Sample × age × drift × error trust, hard-zero on host mismatch."""
    if not isfinite(prior) or not isfinite(half_life) or prior <= 0 or half_life <= 0:
        raise ValueError("prior and half_life must be positive finite values")
    if not evidence.matching_host:
        return 0.
    return (evidence.count / (evidence.count + prior)
            * exp(-log(2) * evidence.age / half_life)
            * (1 - evidence.drift) * (1 - evidence.error))


@dataclass(frozen=True, slots=True)
class ProbeDecision:
    interval: int
    fallback: bool
    trust: float


def probe_decision(evidence: TrustEvidence, *, probe_cost=4., miss_loss=20.,
                   min_interval=8, max_interval=256, minimum_trust=.50) -> ProbeDecision:
    """Estimate probing cadence with bounded costs; fall back when uncertain.

    This chooses *when to check performance again*, not whether output is
    semantically correct: the correctness guard must remain independent.
    """
    if not all(isfinite(x) and x >= 0 for x in (probe_cost, miss_loss)):
        raise ValueError("costs must be finite and nonnegative")
    if not (type(min_interval) is int and type(max_interval) is int
            and 0 < min_interval <= max_interval):
        raise ValueError("invalid probe interval range")
    if not isfinite(minimum_trust) or not 0 <= minimum_trust <= 1:
        raise ValueError("invalid trust threshold")
    trust = trust_score(evidence)
    risk = max(1e-6, miss_loss * (1 - trust) * max(evidence.drift, .001))
    raw = sqrt(2 * probe_cost / risk) if probe_cost else min_interval
    interval = max(min_interval, min(max_interval, round(raw)))
    return ProbeDecision(interval, trust < minimum_trust, trust)
