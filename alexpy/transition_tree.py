"""Opt-in, order-preserving QA transition summary, inspired by Bardo LOGOS.

This is not an implementation or performance claim for Bardo's RTL/LOGOS.
The binary tree retains *exact* first-K failing test identifiers plus counts
and arbitrary-precision duration. The ordinary set-based index is a strong
control. Both wrap the SAME NativeBatchQA engine, and neither is the default.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest

from .qa_batch import _prepare
from .qa_lab import NOT_RUN, FAIL, QASnapshot
from .qa_native import NativeBatchQA


@dataclass(frozen=True, slots=True)
class Node:
    counts: tuple[int, int, int, int, int]
    witnesses: tuple[int, ...]


_ZERO = Node((0, 0, 0, 0, 0), ())


def _merge(left: Node, right: Node, limit: int) -> Node:
    # Exact sorted witnesses because left span precedes right span.
    return Node(tuple(a + b for a, b in zip(left.counts, right.counts)),
                (left.witnesses + right.witnesses)[:limit])


class TransitionTree:
    """Staged O(k log n) leaf updates, order-preserving first-K witnesses.

    prepare() never mutates the tree. commit() only assigns to existing lists.
    The leaves hold no arbitrary history and no probabilistic collision-prone
    digest: they preserve exact counts and exact failing IDs for top-K query.
    """
    __slots__ = ("n", "base", "limit", "nodes", "statuses", "durations")

    def __init__(self, n: int, limit: int = 4):
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        self.n = n
        self.limit = limit
        self.base = 1 << (n - 1).bit_length()
        self.statuses = [NOT_RUN] * n
        self.durations = [0] * n
        self.nodes = [_ZERO] * (self.base * 2)
        for i in range(n):
            self.nodes[self.base + i] = Node((1, 0, 0, 0, 0), ())
        for i in range(self.base - 1, 0, -1):
            self.nodes[i] = _merge(self.nodes[i * 2], self.nodes[i * 2 + 1], limit)

    def prepare(self, latest: dict[int, tuple[int, int]]):
        """All allocations happen before the native authoritative commit."""
        staged = {}
        changed = {}
        for idx, (status, duration) in latest.items():
            if self.statuses[idx] == status and self.durations[idx] == duration:
                continue
            counts = [0, 0, 0, 0, duration]
            counts[status] = 1
            staged[self.base + idx] = Node(tuple(counts), (idx,) if status == FAIL else ())
            changed[idx] = (status, duration)
        for idx in sorted({(self.base + i) >> shift
                           for i in changed
                           for shift in range(1, self.base.bit_length())}, reverse=True):
            staged[idx] = _merge(staged.get(idx * 2, self.nodes[idx * 2]),
                                 staged.get(idx * 2 + 1, self.nodes[idx * 2 + 1]), self.limit)
        return staged, changed

    def commit(self, prepared):
        staged, changed = prepared
        for idx, node in staged.items():
            self.nodes[idx] = node
        for idx, (status, duration) in changed.items():
            self.statuses[idx] = status
            self.durations[idx] = duration

    @property
    def root(self) -> Node:
        return self.nodes[1]

    def verify(self) -> bool:
        counts = [0, 0, 0, 0, 0]
        witnesses = []
        for i, (status, duration) in enumerate(zip(self.statuses, self.durations)):
            counts[status] += 1
            counts[4] += duration
            if status == FAIL and len(witnesses) < self.limit:
                witnesses.append(i)
        return self.root == Node(tuple(counts), tuple(witnesses))


class TreeNativeQA:
    """Research path: same C engine as control plus exact ordered tree.

    On a native success all preparatory allocations are done; list assignment
    commits the mirror. Native errors do not mutate the auxiliary tree.
    """
    __slots__ = ("engine", "tree")

    def __init__(self, suites, suite_count, *, limit=4):
        self.engine = NativeBatchQA(suites, suite_count)
        self.tree = TransitionTree(len(self.engine.suites), limit)

    @property
    def native(self):
        return self.engine.native

    def apply_batch(self, events):
        latest = _prepare(events, self.tree.n)
        prepared = self.tree.prepare(latest)
        transition = self.engine.apply_batch(events)
        self.tree.commit(prepared)
        return transition

    def snapshot(self) -> QASnapshot:
        return self.engine.snapshot()

    def witnesses(self) -> tuple[int, ...]:
        return self.tree.root.witnesses

    def verify(self):
        return (self.engine.verify() and self.tree.verify()
                and self.snapshot().total == self.tree.root.counts)


class SetNativeQA:
    """Equal-native strong control: Python set + top-K extraction at report."""
    __slots__ = ("engine", "status", "failing", "limit")

    def __init__(self, suites, suite_count, *, limit=4):
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be positive")
        self.engine = NativeBatchQA(suites, suite_count)
        self.status = [NOT_RUN] * len(self.engine.suites)
        self.failing = set()
        self.limit = limit

    @property
    def native(self):
        return self.engine.native

    def apply_batch(self, events):
        latest = _prepare(events, len(self.status))
        # Prepare changes before native commit; no claim of transactional
        # resilience to arbitrary MemoryError on Python set.add.
        additions = tuple(idx for idx, (st, _) in latest.items() if st == FAIL)
        removals = tuple(idx for idx, (st, _) in latest.items() if st != FAIL)
        transition = self.engine.apply_batch(events)
        for idx in removals:
            self.failing.discard(idx)
        for idx in additions:
            self.failing.add(idx)
        for idx, (st, _) in latest.items():
            self.status[idx] = st
        return transition

    def snapshot(self):
        return self.engine.snapshot()

    def witnesses(self):
        return tuple(nsmallest(self.limit, self.failing))

    def verify(self):
        status, _ = self.engine._native.dump() if self.engine.native else (self.engine._fallback._status, ())
        return (self.engine.verify() and self.status == list(status)
                and self.failing == {i for i, st in enumerate(self.status) if st == FAIL})


__all__ = ["Node", "TransitionTree", "TreeNativeQA", "SetNativeQA"]
