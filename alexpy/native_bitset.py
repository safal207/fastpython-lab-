"""Opt-in exact native bitset index over unchanged, authoritative NativeBatchQA.

The bitmap retains only failed flags and the first-K diagnostic IDs, no
probabilistic approximation. A preallocated sparse word directory supports
fast queries across large mostly-healthy test suites. No default promotion.
"""
from __future__ import annotations

from .qa_batch import _prepare
from .qa_lab import FAIL
from .qa_native import NativeBatchQA

try:
    from ._qa_bitset import WitnessBits
except ImportError:
    WitnessBits = None


class BitsetNativeQA:
    __slots__ = ("engine", "index")

    def __init__(self, suites, suite_count, *, limit=4):
        if WitnessBits is None:
            raise RuntimeError("_qa_bitset C extension is required")
        self.engine = NativeBatchQA(suites, suite_count)
        if not self.engine.native:
            raise RuntimeError("_qa_native C extension is required")
        self.index = WitnessBits(len(self.engine.suites), limit)

    @property
    def native(self):
        return True

    def apply_batch(self, events):
        latest = _prepare(events, len(self.engine.suites))
        staged = self.index.stage(latest)
        transition = self.engine.apply_batch(events)
        self.index.commit(staged)
        return transition

    def witnesses(self):
        return self.index.witnesses()

    def snapshot(self):
        return self.engine.snapshot()

    def verify(self):
        if not self.engine.verify() or not self.index.verify():
            return False
        status, _ = self.engine._native.dump()
        return (self.index.dump() == bytes(int(st == FAIL) for st in status)
                and self.index.failed_count == self.snapshot().total[FAIL])


__all__ = ["BitsetNativeQA", "WitnessBits"]
