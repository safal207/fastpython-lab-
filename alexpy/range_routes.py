"""Measured alternatives for range sum + sum-of-squares with point updates.

These are *different complete, equivalent query plans*, not optional DAG edges.
Each owns a copy of exact integer input. No public AlexPy API is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

ROUTES = ("direct", "blocks", "fenwick", "prefix")


def _index(index, length):
    if type(index) is not int or not 0 <= index < length:
        raise IndexError("update index must be an exact int in range")


def _range(start, stop, length):
    if (type(start) is not int or type(stop) is not int
            or not 0 <= start <= stop <= length):
        raise IndexError("query range must satisfy 0 <= start <= stop <= size")


def _int(value):
    if type(value) is not int:
        raise TypeError("only exact Python int elements are supported")
    return value


class Direct:
    """Strong no-index baseline: C sum + Python square generator."""

    def __init__(self, values):
        self.data = [_int(x) for x in values]

    def update(self, index, value):
        _index(index, len(self.data))
        self.data[index] = _int(value)

    def query(self, start, stop):
        _range(start, stop, len(self.data))
        segment = self.data[start:stop]
        return sum(segment), sum(x*x for x in segment)


class Blocks(Direct):
    def __init__(self, values):
        super().__init__(values)
        self.width = max(8, isqrt(max(len(self.data), 1)))
        self.totals = []
        self.squares = []
        for i in range(0, len(self.data), self.width):
            part = self.data[i:i+self.width]
            self.totals.append(sum(part))
            self.squares.append(sum(x*x for x in part))

    def update(self, index, value):
        _index(index, len(self.data))
        new = _int(value)
        old = self.data[index]
        self.data[index] = new
        bucket = index // self.width
        self.totals[bucket] += new-old
        self.squares[bucket] += new*new-old*old

    def query(self, start, stop):
        _range(start, stop, len(self.data))
        total = square = 0
        i = start
        while i < stop and i % self.width:
            x = self.data[i]
            total += x
            square += x*x
            i += 1
        while i + self.width <= stop:
            b = i // self.width
            total += self.totals[b]
            square += self.squares[b]
            i += self.width
        while i < stop:
            x = self.data[i]
            total += x
            square += x*x
            i += 1
        return total, square


class Fenwick(Direct):
    def __init__(self, values):
        super().__init__(values)
        n = len(self.data)
        self.sums = [0]*(n+1)
        self.squares = [0]*(n+1)
        for i, x in enumerate(self.data, 1):
            self.sums[i] += x
            self.squares[i] += x*x
            parent = i+(i & -i)
            if parent <= n:
                self.sums[parent] += self.sums[i]
                self.squares[parent] += self.squares[i]

    def update(self, index, value):
        _index(index, len(self.data))
        new = _int(value)
        old = self.data[index]
        self.data[index] = new
        ds, dq = new-old, new*new-old*old
        i = index+1
        while i < len(self.sums):
            self.sums[i] += ds
            self.squares[i] += dq
            i += i & -i

    def _prefix(self, stop):
        s = q = 0
        while stop:
            s += self.sums[stop]
            q += self.squares[stop]
            stop -= stop & -stop
        return s, q

    def query(self, start, stop):
        _range(start, stop, len(self.data))
        s1, q1 = self._prefix(start)
        s2, q2 = self._prefix(stop)
        return s2-s1, q2-q1


class Prefix(Direct):
    """Rebuild lazy prefix sums once after an update, O(1) queries."""

    def __init__(self, values):
        super().__init__(values)
        self.prefix_s = self.prefix_q = None

    def update(self, index, value):
        _index(index, len(self.data))
        self.data[index] = _int(value)
        self.prefix_s = self.prefix_q = None

    def query(self, start, stop):
        _range(start, stop, len(self.data))
        if self.prefix_s is None:
            s, q = [0], [0]
            for x in self.data:
                s.append(s[-1]+x)
                q.append(q[-1]+x*x)
            self.prefix_s, self.prefix_q = s, q
        return self.prefix_s[stop]-self.prefix_s[start], self.prefix_q[stop]-self.prefix_q[start]


_CLASSES = {"direct":Direct,"blocks":Blocks,"fenwick":Fenwick,"prefix":Prefix}


def make_backend(route, values):
    try:
        cls = _CLASSES[route]
    except (KeyError, TypeError):
        raise ValueError("unknown route") from None
    return cls(values)


@dataclass(frozen=True)
class RangeWorkload:
    size: int
    changes: int
    queries: int
    ticks: int
    span: str

    def __post_init__(self):
        if (any(type(x) is not int or x <= 0 for x in
                (self.size, self.changes, self.queries, self.ticks))
                or self.changes > self.size or self.span not in ("short", "wide")):
            raise ValueError("invalid range workload")

    def cell(self):
        return (self.size.bit_length()-1, self.changes.bit_length()-1,
                self.queries.bit_length()-1, self.ticks.bit_length()-1,
                0 if self.span == "short" else 1)


def make_case(workload):
    if not isinstance(workload, RangeWorkload):
        raise TypeError("expected RangeWorkload")
    n = workload.size
    original = [((i*17+13)%283)-141 for i in range(n)]
    ticks = []
    for t in range(workload.ticks):
        # All indices are distinct within a tick, even for composite sizes.
        changed = [((i * 131 + t * 19) % n) for i in range(n)]
        # Duplicate pattern for some n: use ordered uniqueness + full scan.
        changed = list(dict.fromkeys(changed))
        if len(changed) < workload.changes:
            changed.extend(i for i in range(n) if i not in changed)
        patch = [(i, 1_000 + 31*t + j) for j, i in enumerate(changed[:workload.changes])]
        queries = []
        for j in range(workload.queries):
            if workload.span == "wide":
                length = max(1, n // 2 + ((j*7+t*11) % (n - n//2 + 1)))
            else:
                length = min(n, 1 + (j*13+t*3) % 24)
            start = (j*37+t*23) % (n-length+1)
            queries.append((start, start+length))
        ticks.append((patch, queries))
    return original, ticks


def run_workflow(route, workload, *, case=None):
    original, steps = case if case is not None else make_case(workload)
    backend = make_backend(route, original)
    checksum = 0
    for changes, queries in steps:
        for index, value in changes:
            backend.update(index, value)
        for a, b in queries:
            s, q = backend.query(a, b)
            checksum += s+q
    return checksum


def checked_routes(workload, *, case=None):
    case = make_case(workload) if case is None else case
    expected = run_workflow("direct", workload, case=case)
    for route in ROUTES[1:]:
        if run_workflow(route, workload, case=case) != expected:
            raise AssertionError(f"correctness gate rejected {route}")
    return expected
