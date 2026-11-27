"""Conservative, environment-scoped route evidence for opt-in AlexPy experiments.

Training is explicitly offline; holdout route tables are frozen before timing.
One *independent process* contributes one observation per workload, rather than
counting repeated measurements in one process as independent evidence.
The bound is a risk heuristic (median + 2*MAD), NOT a confidence interval.
No arbitrary CPython calls are intercepted and correctness is never inferred
from fast timings.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import sys
from statistics import median

from .range_routes import ROUTES, RangeWorkload


def environment_key(*, device_tag: str) -> str:
    """The operator must identify the device; architecture alone is insufficient.

    Sharing a key across CPUs requires the caller to explicitly vouch for the
    same device/profile. A different tag always fails closed to the baseline.
    """
    if type(device_tag) is not str or not device_tag.strip() or len(device_tag) > 128:
        raise ValueError("explicit nonempty device_tag required")
    return (f"{sys.implementation.name}/{sys.version_info.major}."
            f"{sys.version_info.minor}/{platform.system()}/{platform.machine()}"
            f"/device={device_tag}")


def simple_rule(w: RangeWorkload) -> str:
    return "prefix" if w.queries >= 16 and w.span == "wide" else "direct"


def workload_key(w: RangeWorkload) -> str:
    return f"{w.size}:{w.changes}:{w.queries}:{w.ticks}:{w.span}"


def _med_mad(numbers):
    m = median(numbers)
    return m, median(abs(x - m) for x in numbers)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    baseline: str
    reason: str
    predicted_upper_log_ratio: float | None = None
    source_size: int | None = None
    independent_processes: int = 0


class StableRouteMemory:
    """Cost-sensitive conservative route selector with exact profile isolation."""

    def __init__(self, *, environment: str, min_processes: int = 3,
                 min_gain: float = 1.10, uncertainty: float = 2.0,
                 size_penalty: float = .07):
        if type(environment) is not str or not environment:
            raise ValueError("explicit environment required")
        if type(min_processes) is not int or min_processes < 2:
            raise ValueError("at least two independent processes required")
        if (any(type(x) not in (float, int) or not math.isfinite(x) for x in
                (min_gain, uncertainty, size_penalty))
                or min_gain < 1 or uncertainty < 0 or size_penalty < 0):
            raise ValueError("invalid risk bounds")
        self.environment = environment
        self.min_processes = min_processes
        self.min_gain = float(min_gain)
        self.uncertainty = float(uncertainty)
        self.size_penalty = float(size_penalty)
        self._samples: dict[str, dict[str, dict[str, float]]] = {}
        self._workloads: dict[str, RangeWorkload] = {}
        self._switch: dict[tuple[str, str, str, str], dict[str, tuple[float, float]]] = {}
        self._quarantined: set[tuple[str, str]] = set()
        self._frozen: dict[str, RouteDecision] | None = None

    def observe_session(self, w: RangeWorkload, *, process_id: str,
                        raw_ns: dict[str, list[int]], verified: bool,
                        environment: str):
        if self._frozen is not None:
            raise RuntimeError("frozen evidence cannot be modified")
        if environment != self.environment:
            raise ValueError("cross-environment training refused")
        if verified is not True:
            raise ValueError("unverified measurements cannot train selector")
        if type(process_id) is not str or not process_id:
            raise ValueError("process_id required")
        if type(raw_ns) is not dict or set(raw_ns) != set(ROUTES):
            raise ValueError("all routes need paired timing evidence")
        if any(type(v) is not list or len(v) < 2 or any(
                type(x) is not int or x <= 0 for x in v)
                for v in raw_ns.values()):
            raise ValueError("positive repeat measurements required")
        if len({len(v) for v in raw_ns.values()}) != 1:
            raise ValueError("unequal paired measurements")
        key = workload_key(w)
        by_process = self._samples.setdefault(key, {})
        if process_id in by_process:
            raise ValueError("duplicate process observation")
        baseline = simple_rule(w)
        baseline_ns = median(raw_ns[baseline])
        if baseline_ns <= 0:
            raise ValueError("invalid baseline")
        # Normalize by the independent process's own baseline measurement;
        # timings in different processes cannot share a raw absolute scale.
        by_process[process_id] = {route: (math.log(median(raw_ns[route])) - math.log(baseline_ns))
                                  for route in ROUTES}
        self._workloads[key] = w

    def quarantine(self, w: RangeWorkload, route: str):
        if self._frozen is not None:
            raise RuntimeError("cannot modify frozen evidence")
        if route not in ROUTES:
            raise ValueError("unknown route")
        self._quarantined.add((workload_key(w), route))

    def _predict(self, w: RangeWorkload) -> RouteDecision:
        baseline = simple_rule(w)
        if (workload_key(w), baseline) in self._quarantined:
            if baseline == "direct" or (workload_key(w), "direct") in self._quarantined:
                raise RuntimeError("no trusted fallback available")
            baseline = "direct"
        nearby = []
        # Keep all workload dimensions other than SIZE identical. This avoids
        # a misleading nearest-neighbor transfer between different patterns.
        for key, source in self._workloads.items():
            if (source.changes, source.queries, source.ticks, source.span) != (
                    w.changes, w.queries, w.ticks, w.span):
                continue
            if len(self._samples[key]) < self.min_processes:
                continue
            distance = abs(math.log2(w.size / source.size))
            nearby.append((distance, source.size, key))
        if not nearby:
            return RouteDecision(baseline, baseline, "insufficient_independent_evidence")
        distance, source_size, source_key = min(nearby)
        records = self._samples[source_key]
        acceptable = []
        for route in ROUTES:
            if route == baseline or (source_key, route) in self._quarantined:
                continue
            values = [row[route] for row in records.values()]
            middle, mad = _med_mad(values)
            risk_upper = max(middle + self.uncertainty * mad, max(values)) + self.size_penalty * distance
            # Safe comparator: only switch when *conservative estimate* is
            # at least min_gain times faster than the strong feature rule.
            if risk_upper < -math.log(self.min_gain):
                acceptable.append((risk_upper, route, source_size, len(records)))
        if not acceptable:
            return RouteDecision(baseline, baseline, "fallback_risk_or_gain")
        score, route, source_size, count = min(acceptable, key=lambda x:(x[0],ROUTES.index(x[1])))
        return RouteDecision(route, baseline, "measured_gain", score, source_size, count)

    def compile(self, holdout: list[RangeWorkload], *, environment: str):
        """Prepare a fixed read-only lookup; never inspect holdout timings."""
        if self._frozen is not None:
            raise RuntimeError("route table already frozen")
        if type(holdout) is not list:
            raise TypeError("holdout must be an exact list")
        keys = [workload_key(w) for w in holdout]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate holdout workload")
        if any(k in self._samples for k in keys):
            raise ValueError("holdout workload leaked into training")
        self._frozen = {k:(self._predict(w) if environment == self.environment else
                          RouteDecision(simple_rule(w),simple_rule(w),"environment_mismatch"))
                        for k,w in zip(keys, holdout)}
        return dict(self._frozen)

    def choose(self, w: RangeWorkload, *, environment: str) -> RouteDecision:
        if self._frozen is None:
            raise RuntimeError("compile before choosing")
        key = workload_key(w)
        if key not in self._frozen:
            return RouteDecision(simple_rule(w),simple_rule(w),"uncompiled_workload")
        if environment != self.environment:
            return RouteDecision(simple_rule(w),simple_rule(w),"environment_mismatch")
        return self._frozen[key]

    def observe_switch(self, before: RangeWorkload, after: RangeWorkload,
                       old_route: str, new_route: str, *, process_id: str,
                       setup_ns: int, tick_ns: int, verified: bool,
                       environment: str):
        if self._frozen is not None:
            raise RuntimeError("frozen evidence cannot be modified")
        if environment != self.environment or verified is not True:
            raise ValueError("cross-profile or unverified transition rejected")
        if old_route not in ROUTES or new_route not in ROUTES:
            raise ValueError("invalid route")
        if before.size != after.size:
            raise ValueError("cross-size layout transition needs new snapshot")
        if (type(process_id) is not str or not process_id or
                type(setup_ns) is not int or type(tick_ns) is not int or
                setup_ns < 0 or tick_ns <= 0):
            raise ValueError("invalid switch samples")
        key = workload_key(before),workload_key(after),old_route,new_route
        group = self._switch.setdefault(key,{})
        if process_id in group:
            raise ValueError("duplicate transition process")
        group[process_id] = (float(setup_ns),float(tick_ns))

    def advise_switch(self, before: RangeWorkload, after: RangeWorkload,
                      old_route: str, candidate: str, *, horizon: int,
                      environment: str, margin: float = .10):
        """Use measured *layout setup* and *per-tick* costs, not DAG weights.

        Incumbent already has a live layout. New route must pay its creation
        cost; workload geometry is assumed stationary over horizon ticks.
        This is only a prediction, not an automatic runtime mutation.
        """
        if old_route not in ROUTES or candidate not in ROUTES:
            raise ValueError("invalid routes")
        if (type(horizon) is not int or horizon < 1 or
                type(margin) not in (int,float) or not math.isfinite(margin)
                or not 0 <= margin < 1):
            raise ValueError("invalid horizon/margin")
        if environment != self.environment:
            return old_route,"environment_mismatch",None
        if old_route == candidate:
            return old_route,"already_current",None
        key_new = workload_key(before),workload_key(after),old_route,candidate
        key_old = workload_key(before),workload_key(after),old_route,old_route
        new = self._switch.get(key_new,{})
        old = self._switch.get(key_old,{})
        common = set(new) & set(old)
        if len(common) < self.min_processes:
            return old_route,"insufficient_transition_evidence",None
        # Paired processes remove much of system drift. Penalize dispersion;
        # treat the margin as a minimum required savings.
        diffs = [horizon*(old[p][1]-new[p][1])-new[p][0] for p in common]
        middle, spread = _med_mad(diffs)
        conservative_savings = middle - self.uncertainty*spread
        old_total = median(horizon*old[p][1] for p in common)
        if conservative_savings > margin*old_total:
            return candidate,"switch_amortized",conservative_savings
        return old_route,"switch_not_amortized",conservative_savings

    def snapshot(self):
        return {"schema":"ALEXPY-STABLE-MEMORY/1", "environment":self.environment,
                "train_cells":len(self._samples),
                "independent_samples":sum(len(v) for v in self._samples.values()),
                "transitions":len(self._switch),
                "quarantined":len(self._quarantined),
                "compiled":len(self._frozen or {})}


__all__ = ["StableRouteMemory", "RouteDecision", "environment_key",
           "simple_rule", "workload_key"]
