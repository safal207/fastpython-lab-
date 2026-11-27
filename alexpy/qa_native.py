"""Opt-in native QA batch engine with an explicit safe Python fallback.

Validation, last-write-wins coalescing, counters, and arbitrary-precision
aggregate durations occur inside the extension. No native module available:
fall back to the strong Python batch implementation, never claim native speed.
"""
from __future__ import annotations

from .qa_batch import QABatchTransition, StrongPythonBatchQA
from .qa_lab import QASnapshot, scan_reference, _layout

try:
    from ._qa_native import QAState as _QAState
except ImportError:
    _QAState = None


class NativeBatchQA:
    __slots__ = ("suites", "_native", "_fallback", "last_batch")

    def __init__(self, suites, suite_count):
        self.suites = _layout(suites, suite_count)
        if _QAState is None:
            self._native = None
            self._fallback = StrongPythonBatchQA(suites, suite_count)
        else:
            self._native = _QAState(suites, suite_count)
            self._fallback = None
        self.last_batch = QABatchTransition(0, 0, 0, 0, (), ())

    @property
    def native(self):
        return self._native is not None

    @property
    def tick(self):
        return self._native.tick if self.native else self._fallback.tick

    @property
    def version(self):
        return self._native.version if self.native else self._fallback.version

    def apply_batch(self, events):
        if self._native is None:
            result = self._fallback.apply_batch(events)
        else:
            result = self._native.apply_batch(events)
        self.last_batch = result
        return result

    def snapshot(self):
        if self._native is None:
            return self._fallback.snapshot()
        return QASnapshot(*self._native.snapshot_raw())

    def verify(self):
        if self._native is None:
            return self._fallback.snapshot() == scan_reference(
                self.suites, len(self._fallback._counts),
                self._fallback._status, self._fallback._durations)
        status, durations = self._native.dump()
        return scan_reference(self.suites, len(self.snapshot().suites),
                              status, durations) == self.snapshot()


__all__ = ["NativeBatchQA"]
