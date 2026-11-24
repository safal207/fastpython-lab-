"""Experimental QA route matrix. Explicit opt-in; no live state migration.

Rows are workload features known before execution, columns are routes.
Train on independently launched processes, freeze once, never learn from
holdout timings. Risk rule is a heuristic, not a confidence interval.
The baseline is C-tree; missing/out-of-profile evidence stays there.
"""
from dataclasses import dataclass
from math import isfinite
from statistics import median
from types import MappingProxyType

ROUTES = ("c_tree", "bitset", "python", "native_set")
BASELINE = "c_tree"


@dataclass(frozen=True)
class Cell:
    median_ns: float
    p99_ns: float
    peak_bytes: float

    def __post_init__(self):
        if any(type(x) not in (int, float) or not isfinite(x) or x <= 0
               for x in (self.median_ns, self.p99_ns, self.peak_bytes)):
            raise ValueError("invalid measured cell")
        if self.p99_ns < self.median_ns:
            raise ValueError("p99 below median")


@dataclass(frozen=True)
class Decision:
    route: str
    reason: str


def features(n, batch):
    if type(n) is not int or type(batch) is not int or min(n, batch) < 1:
        raise ValueError("positive exact dimensions required")
    return ("small" if n <= 32 else "medium" if n <= 1024 else "large",
            "single" if batch == 1 else "small" if batch <= 16 else "wide")


class RouteMatrix:
    """Evidence scoped to one explicit runner identity.

    Selection is restricted to fresh-engine workloads with one report per
    batch. No online switching, no state transfer, no automatic activation.
    """
    def __init__(self, environment):
        if type(environment) is not str or not environment:
            raise ValueError("explicit runner identity required")
        self.environment = environment
        self._rows = {}
        self._frozen = None

    def observe(self, key, process_id, cells, *, environment, verified):
        if self._frozen is not None:
            raise RuntimeError("matrix frozen")
        if environment != self.environment or verified is not True:
            raise ValueError("untrusted evidence")
        if type(process_id) is not str or not process_id:
            raise ValueError("process ID required")
        if (type(key) is not tuple or len(key) != 2
                or key[0] not in ("small", "medium", "large")
                or key[1] not in ("single", "small", "wide")):
            raise ValueError("invalid feature key")
        if set(cells) != set(ROUTES) or any(type(v) is not Cell for v in cells.values()):
            raise ValueError("complete measured row required")
        row = self._rows.setdefault(key, {})
        if process_id in row:
            raise ValueError("duplicate process")
        row[process_id] = dict(cells)

    def freeze(self, *, overhead_ns):
        if self._frozen is not None:
            raise RuntimeError("matrix already frozen")
        if type(overhead_ns) not in (int, float) or not isfinite(overhead_ns) or overhead_ns < 0:
            raise ValueError("invalid dispatch overhead")
        decisions = {}
        for key, sessions in self._rows.items():
            if len(sessions) < 3:
                decisions[key] = Decision(BASELINE, "insufficient_processes")
                continue
            candidates = []
            for route in ROUTES[1:]:
                rows = list(sessions.values())
                # Require every process to pass median and p99 gates after
                # adding measured dispatch overhead; no memory-only promotion.
                if all((r[route].median_ns + overhead_ns) <= .95*r[BASELINE].median_ns
                       and (r[route].p99_ns + overhead_ns) <= r[BASELINE].p99_ns
                       and r[route].peak_bytes <= r[BASELINE].peak_bytes for r in rows):
                    candidates.append((median((r[route].median_ns + overhead_ns)
                                              /r[BASELINE].median_ns for r in rows), route))
            decisions[key] = (Decision(min(candidates)[1], "conservative_measured_gain")
                              if candidates else Decision(BASELINE, "risk_or_gain_gate"))
        self._frozen = MappingProxyType(decisions)
        return dict(decisions)

    def choose(self, n, batch, *, environment):
        if self._frozen is None:
            raise RuntimeError("freeze before dispatch")
        if environment != self.environment:
            return Decision(BASELINE, "environment_mismatch")
        return self._frozen.get(features(n, batch), Decision(BASELINE, "unknown_workload"))
