"""Opt-in exact-integer experiments; the public sum_squares dispatcher is unchanged.

Candidate A: mixed-precision C accumulation for plain list/tuple[int].
Candidate B: Python-level zero/singleton dispatch for exact containers and ints.

All unrecognised inputs delegate to the existing public implementation so
subclass iteration, custom multiplication, errors and iterable consumption
retain the original semantics. A candidate is NOT automatically promoted by
seeing a favorable microbenchmark result.
"""
from __future__ import annotations

from . import _native, sum_squares


def bigint_native(values):
    """Single-pass native mixed-precision accumulation where type-safe."""
    if _native is not None and type(values) in (list, tuple):
        result = _native.sum_squares_bigints(values)
        if result is not NotImplemented:
            return result
    return sum_squares(values)


def small_dispatch(values):
    """Avoid C call on exact empty/singleton lists or tuples of exact ints."""
    if type(values) in (list, tuple):
        if not values:
            return 0
        if len(values) == 1:
            value = values[0]
            if type(value) is int:
                return value * value
    return sum_squares(values)


CANDIDATES = {"current": sum_squares, "bigint_native": bigint_native,
              "small_dispatch": small_dispatch}
