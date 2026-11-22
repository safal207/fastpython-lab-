"""Experimental compact C witness index over the unchanged NativeBatchQA.

The authoritative native accumulator remains unchanged. Staging validates and
allocates the witness patch *before* the native QA transaction. After a native
commit, applying an immutable staged patch has no allocation path. No default
route is modified; a missing compiled witness extension is explicit.
"""
from __future__ import annotations

from .qa_batch import _prepare
from .qa_lab import FAIL
from .qa_native import NativeBatchQA

try:
    from ._qa_witness import WitnessTree
except ImportError:
    WitnessTree = None


class CTreeNativeQA:
    __slots__ = ("engine", "index")

    def __init__(self, suites, suite_count, *, limit=4):
        if WitnessTree is None:
            raise RuntimeError("_qa_witness C extension is required")
        self.engine = NativeBatchQA(suites, suite_count)
        if not self.engine.native:
            raise RuntimeError("_qa_native C extension is required")
        self.index = WitnessTree(len(self.engine.suites), limit)

    @property
    def native(self):
        return True

    def apply_batch(self, events):
        latest = _prepare(events, len(self.engine.suites))
        stage = self.index.stage(latest)
        transition = self.engine.apply_batch(events)
        # stage is immutable; commit does not allocate or access Python event
        # attributes, preserving consistency after the native QA commit.
        self.index.commit(stage)
        return transition

    def witnesses(self):
        return self.index.witnesses()

    def snapshot(self):
        return self.engine.snapshot()

    def verify(self):
        if not self.engine.verify() or not self.index.verify():
            return False
        status, _ = self.engine._native.dump()
        failed_flags = self.index.dump()
        return (failed_flags == bytes(int(st == FAIL) for st in status)
                and self.index.failed_count == self.snapshot().total[FAIL])


__all__ = ["CTreeNativeQA", "WitnessTree"]
