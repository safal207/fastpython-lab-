"""Opt-in owned, versioned q-buffer with incremental aggregate maintenance.

This is deliberately *not* a drop-in replacement for a mutable Python list:
all mutations must pass through ``apply_patch``; external aliases cannot be
tracked and no writable buffer view is exposed. No automatic promotion.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass

from .memory_candidates import _buffer_native

_MIN_Q = -(1 << 63)
_MAX_Q = (1 << 63) - 1


def _exact_q(value: object) -> int:
    if type(value) is not int:
        raise TypeError("all values must be exact int (not bool or subclasses)")
    if not _MIN_Q <= value <= _MAX_Q:
        raise OverflowError("values must fit in native signed 64-bit integers")
    return value


@dataclass(frozen=True)
class Transition:
    tick: int
    version: int
    changed_indices: tuple[int, ...]
    delta_sum: int
    delta_sum_squares: int


class OwnedQBuffer:
    """Snapshot an exact list/tuple of signed q ints and maintain sums in O(k).

    ``apply_patch`` is the only supported mutation API; the original source
    list is copied and never observed again. ``verify`` is intentionally O(N)
    and is NOT called on the benchmark hot path. Internal ``_data`` is private,
    not an adversary-proof boundary against introspection.
    """

    __slots__ = (
        "_data", "_total", "_squares", "_tick", "_version", "_dirty",
        "_trace", "_history",
    )

    def __init__(self, values: list[int] | tuple[int, ...], *, trace: bool = False):
        if type(values) not in (list, tuple):
            raise TypeError("source must be an exact list or tuple")
        if type(trace) is not bool:
            raise TypeError("trace must be bool")
        if array("q").itemsize != 8:
            raise RuntimeError("native signed q layout is not 8 bytes")
        checked = [_exact_q(value) for value in values]
        self._data = array("q", checked)
        self._total = sum(checked)
        if _buffer_native is not None:
            self._squares = _buffer_native.sum_squares_linear(self._data)
            if self._squares is NotImplemented:
                self._squares = sum(v * v for v in checked)
        else:
            self._squares = sum(v * v for v in checked)
        self._tick = 0
        self._version = 0
        self._dirty = ()
        self._trace = trace
        self._history: list[Transition] = []

    def __len__(self) -> int:
        return len(self._data)

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def version(self) -> int:
        return self._version

    @property
    def dirty(self) -> tuple[int, ...]:
        return self._dirty

    @property
    def history(self) -> tuple[Transition, ...]:
        return tuple(self._history)

    @property
    def data_bytes(self) -> int:
        """Raw payload bytes only; excludes Python object and bookkeeping."""
        return len(self._data) * self._data.itemsize

    def at(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("index must be an exact integer in range")
        return self._data[index]

    def to_list(self) -> list[int]:
        """Explicit O(N) copy; never returns the mutable backing array."""
        return list(self._data)

    def aggregates(self) -> tuple[int, int]:
        """Return (sum, sum of squares), each from cached O(1) state."""
        return self._total, self._squares

    def variance_numerator(self) -> int:
        """n * sum(x**2) - sum(x)**2; exact integer numerator."""
        return len(self) * self._squares - self._total * self._total

    @staticmethod
    def dependencies() -> dict[str, tuple[str, ...]]:
        """Logical dependency graph, not an observed hardware execution graph."""
        return {
            "data[index]": ("sum", "sum_squares"),
            "sum": ("variance_numerator",),
            "sum_squares": ("variance_numerator",),
        }

    def apply_patch(self, changes: dict[int, int]) -> Transition:
        """Validate whole patch, then atomically update aggregates per tick.

        A rejected patch leaves data, version, tick, and history unchanged.
        O(k) for k supplied indices; stable length, distinct keys (dict).
        """
        if type(changes) is not dict:
            raise TypeError("patch must be an exact dict[index, int]")
        prepared: list[tuple[int, int, int]] = []
        for index, new_value in changes.items():
            if type(index) is not int or not 0 <= index < len(self):
                raise IndexError("patch index must be an exact integer in range")
            value = _exact_q(new_value)
            old = self._data[index]
            if old != value:
                prepared.append((index, old, value))
        return self._commit_prepared(prepared)

    def _commit_prepared(self, prepared: list[tuple[int, int, int]]) -> Transition:
        """Internal commit: requires fully validated exact (index,old,new).

        Caller must hold exclusive ownership of this buffer and have verified
        each entry with ``_exact_q`` and current stored value. Deliberately
        private: bypassing validation would void the atomicity contract.
        """
        # Big-int deltas are computed before any modification. They cannot
        # overflow Python int even if the q array's squared sum exceeds 64 bit.
        delta_sum = sum(new - old for _, old, new in prepared)
        delta_squares = sum(new * new - old * old for _, old, new in prepared)
        for index, _old, new in prepared:
            self._data[index] = new
        self._total += delta_sum
        self._squares += delta_squares
        self._tick += 1
        if prepared:
            self._version += 1
        self._dirty = tuple(sorted(index for index, _, _ in prepared))
        transition = Transition(
            self._tick, self._version, self._dirty, delta_sum, delta_squares
        )
        if self._trace:
            self._history.append(transition)
        return transition

    def verify(self) -> bool:
        """Independent full-state mirrors, O(N), for tests / debug only."""
        values = self._data
        return (self._total == sum(values)
                and self._squares == sum(v * v for v in values)
                and self._squares == sum(pow(v, 2) for v in values)
                and self.variance_numerator() >= 0)


__all__ = ["OwnedQBuffer", "Transition"]
