"""Opt-in, actual continuous range-query streams with paid layout transitions.

A stream consists of successive segments over ONE mutable list. At every
segment boundary the policy may keep its live backend or rebuild a new layout
from the current backend state. No backend is rebuilt when the choice is
unchanged. Online route lookup, initial construction, updates, queries, and
actual layout rebuilds are all in the measured end-to-end time.

The oracle is a separate Direct implementation. It is not used to decide any
route; the held-out route table is frozen before checks/timings. These are
research workloads, not arbitrary Python interception.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from .range_routes import ROUTES, RangeWorkload, make_backend, make_case
from .stable_memory import simple_rule, workload_key


@dataclass(frozen=True)
class StreamResult:
    checksum: int
    setup_ns: int
    switch_ns: int
    switches: int
    constructed: int
    route_trace: tuple[str, ...]
    segments: int


def stream_schedule(size: int, *, changes=(1, 8), queries=(1, 8, 32),
                    spans=("short", "wide"), ticks=2) -> tuple[RangeWorkload, ...]:
    """Deterministic alternation; workload mix chosen before any timing."""
    if type(size) is not int or size <= 0 or type(ticks) is not int or ticks <= 0:
        raise ValueError("positive size and ticks required")
    cases = [RangeWorkload(size, ch, q, ticks, span)
             for q in queries for span in spans for ch in changes]
    if not cases or len({workload_key(w) for w in cases}) != len(cases):
        raise ValueError("nonempty distinct stream workload segments required")
    return tuple(cases)


def run_stream(schedule: tuple[RangeWorkload, ...], *, cycles: int,
               policy: str, choices: dict[str, str] | None = None,
               case_cache: dict[str, tuple] | None = None) -> StreamResult:
    if type(schedule) is not tuple or not schedule:
        raise ValueError("nonempty tuple schedule required")
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be positive")
    if policy not in ("simple", "stable", "direct"):
        raise ValueError("invalid policy")
    n = schedule[0].size
    if any(w.size != n for w in schedule):
        raise ValueError("stream cannot silently change source-array size")
    if policy == "stable" and (type(choices) is not dict or
       any(workload_key(w) not in choices or choices[workload_key(w)] not in ROUTES
           for w in schedule)):
        raise ValueError("stable policy requires a verified frozen route table")

    if case_cache is None:
        case_cache = {workload_key(w):make_case(w) for w in schedule}
    if any(workload_key(w) not in case_cache for w in schedule):
        raise ValueError("missing precomputed input case")
    original, _ = case_cache[workload_key(schedule[0])]
    backend = None
    current_route = None
    setup_ns = switch_ns = count = constructed = 0
    checksum = 0
    route_trace = []
    for cycle in range(cycles):
        for slot, w in enumerate(schedule):
            # Route selection must happen during the timed stream, not in an
            # uncounted preprocessing stage. The stable table is read-only.
            if policy == "simple":
                route = simple_rule(w)
            elif policy == "stable":
                route = choices[workload_key(w)]
            else:
                route = "direct"
            if backend is None or route != current_route:
                start = perf_counter_ns()
                # Backend owns its current Python list. Its values, NOT the
                # original input, are transferred into the next representation.
                backend = make_backend(route, original if backend is None else backend.data)
                elapsed = perf_counter_ns() - start
                setup_ns += elapsed
                constructed += 1
                if current_route is not None:
                    switch_ns += elapsed
                    count += 1
                current_route = route
            route_trace.append(route)
            _, ticks = case_cache[workload_key(w)]
            for changes, queries in ticks:
                for index, value in changes:
                    # Unique epoch values prevent repeated cycles becoming
                    # silently no-op updates on the same mutable state.
                    backend.update(index, value + 1009 * (cycle * len(schedule) + slot))
                for left, right in queries:
                    s, q = backend.query(left, right)
                    checksum += s + q
    return StreamResult(checksum, setup_ns, switch_ns, count,
                        constructed, tuple(route_trace), cycles * len(schedule))


def witness(schedule: tuple[RangeWorkload, ...], *, cycles: int,
            case_cache: dict[str, tuple] | None = None):
    """Independent witness: a long-lived direct backend, no learned routes."""
    return run_stream(schedule, cycles=cycles, policy="direct",
                      case_cache=case_cache).checksum


__all__ = ["StreamResult", "stream_schedule", "run_stream", "witness"]
