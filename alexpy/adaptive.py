"""Opt-in rule-based dispatcher. No automatic performance promotion.

Only exact built-in lists and tuples are inspected without invoking user hooks.
The specialized big-int path uses the existing C candidate's exact-int gate.
The public AlexPy dispatcher remains unchanged.
"""
from __future__ import annotations

from . import _native, sum_squares


def route(values):
    """Return a deterministic *hypothesis*, not a measured optimal decision.

    Do not inspect a custom iterable/subclass: even len/getitem may have side
    effects, and its iteration semantics may differ from list storage.
    """
    if type(values) not in (list, tuple):
        return "python-direct"
    n = len(values)
    if n == 0:
        return "zero"
    first = values[0]
    if n == 1 and type(first) is int:
        return "singleton-int"
    if (_native is not None and hasattr(_native, "sum_squares_bigints")
            and type(first) is int
            and (first.bit_length() > 62 or (n >= 10 and first.bit_length() >= 29))):
        return "bigint-native"
    return "current"


def adaptive_sum_squares(values):
    """Correctness-preserving opt-in adapter under narrow assumptions.

    Dynamic rules are intentionally *not* installed in sum_squares(); CI and
    independent guardset benchmarks must validate any performance promotion.
    """
    decision = route(values)
    if decision == "zero":
        return 0
    if decision == "singleton-int":
        return values[0] * values[0]
    if decision == "python-direct":
        return sum(x * x for x in values)
    if decision == "bigint-native":
        result = _native.sum_squares_bigints(values)
        if result is not NotImplemented:
            return result
    return sum_squares(values)


__all__ = ["route", "adaptive_sum_squares"]
