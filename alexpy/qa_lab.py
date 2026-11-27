"""Experimental QA-event dashboard: explicit test -> suite -> global DAG.

The evidence source is a deterministic *synthetic* event stream modeling CI
retries and status changes, not a claim to have ingested private QA logs.
Both optimized implementations are compared to an independent full scan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .persistent import OwnedQBuffer

MAX_DURATION = (1 << 63) - 1
NOT_RUN, PASS, FAIL, SKIP = range(4)


@dataclass(frozen=True, slots=True)
class QAEvent:
    test_id: int
    status: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class QASnapshot:
    # (not-run, passed, failed, skipped, sum of latest duration in ms)
    total: tuple[int, int, int, int, int]
    suites: tuple[tuple[int, int, int, int, int], ...]


def _layout(suites: Sequence[int], suite_count: int) -> tuple[int, ...]:
    if type(suite_count) is not int or suite_count < 1:
        raise ValueError("suite_count must be a positive exact integer")
    if type(suites) not in (list, tuple) or not suites:
        raise ValueError("non-empty exact list/tuple of suite IDs required")
    if any(type(i) is not int or not 0 <= i < suite_count for i in suites):
        raise ValueError("invalid suite ID")
    return tuple(suites)


def _validate(event: QAEvent, n: int) -> None:
    if type(event) is not QAEvent:
        raise TypeError("event must be an exact QAEvent")
    if type(event.test_id) is not int or not 0 <= event.test_id < n:
        raise IndexError("invalid test ID")
    if type(event.status) is not int or event.status not in (PASS, FAIL, SKIP):
        raise ValueError("status must be PASS, FAIL or SKIP")
    if (type(event.duration_ms) is not int or not 0 <= event.duration_ms <= MAX_DURATION):
        raise ValueError("duration must be an exact nonnegative signed-q integer")
    if event.status == SKIP and event.duration_ms != 0:
        raise ValueError("skipped tests must have zero duration")


class StrongPythonQA:
    """Strong independent Python baseline: O(1) incremental counters.

    It is intentionally not an O(N) strawman. Report materialization costs
    O(number of suites) and are included in benchmark timings.
    """

    __slots__ = ("suites", "_status", "_durations", "_counts", "_global", "tick", "version")

    def __init__(self, suites: Sequence[int], suite_count: int):
        self.suites = _layout(suites, suite_count)
        self._status = [NOT_RUN] * len(self.suites)
        self._durations = [0] * len(self.suites)
        self._counts = [[0, 0, 0, 0, 0] for _ in range(suite_count)]
        for suite in self.suites:
            self._counts[suite][NOT_RUN] += 1
        self._global = [len(self.suites), 0, 0, 0, 0]
        self.tick = self.version = 0

    def apply(self, event: QAEvent) -> None:
        _validate(event, len(self.suites))
        idx = event.test_id
        previous = self._status[idx]
        old_duration = self._durations[idx]
        self.tick += 1
        if previous == event.status and old_duration == event.duration_ms:
            return
        suite_counts = self._counts[self.suites[idx]]
        suite_counts[previous] -= 1
        suite_counts[event.status] += 1
        suite_counts[4] += event.duration_ms - old_duration
        self._global[previous] -= 1
        self._global[event.status] += 1
        self._global[4] += event.duration_ms - old_duration
        self._status[idx] = event.status
        self._durations[idx] = event.duration_ms
        self.version += 1

    def snapshot(self) -> QASnapshot:
        return QASnapshot(tuple(self._global), tuple(tuple(c) for c in self._counts))


class AlexPyQAGraph:
    """Opt-in sparse dependency graph backed by the owned signed-q duration cache.

    Edges are test[index] -> suite[index] -> global; only an affected suite
    and the global counters change. No arbitrary mutation of a source list
    is detected. Buffer copies once on construction and updates by explicit
    patches. This prototype does not automatically instrument CI frameworks.
    """

    __slots__ = ("suites", "_status", "_buffer", "_counts", "_global",
                 "tick", "version", "last_affected_suites")

    def __init__(self, suites: Sequence[int], suite_count: int):
        self.suites = _layout(suites, suite_count)
        self._status = [NOT_RUN] * len(self.suites)
        self._buffer = OwnedQBuffer([0] * len(self.suites))
        self._counts = [[0, 0, 0, 0, 0] for _ in range(suite_count)]
        for suite in self.suites:
            self._counts[suite][NOT_RUN] += 1
        self._global = [len(self.suites), 0, 0, 0, 0]
        self.tick = self.version = 0
        self.last_affected_suites: tuple[int, ...] = ()

    @property
    def data_bytes(self) -> int:
        return self._buffer.data_bytes

    def apply(self, event: QAEvent) -> None:
        _validate(event, len(self.suites))
        idx = event.test_id
        previous = self._status[idx]
        old_duration = self._buffer.at(idx)
        self.tick += 1
        if previous == event.status and old_duration == event.duration_ms:
            self.last_affected_suites = ()
            return
        suite = self.suites[idx]
        # OwnedQBuffer validates and commits the duration before other caches
        # change. Remaining operations are unbounded Python integers.
        self._buffer.apply_patch({idx: event.duration_ms})
        counts = self._counts[suite]
        counts[previous] -= 1
        counts[event.status] += 1
        counts[4] += event.duration_ms - old_duration
        global_counts = self._global
        global_counts[previous] -= 1
        global_counts[event.status] += 1
        global_counts[4] += event.duration_ms - old_duration
        self._status[idx] = event.status
        self.version += 1
        self.last_affected_suites = (suite,)

    def snapshot(self) -> QASnapshot:
        return QASnapshot(tuple(self._global), tuple(tuple(c) for c in self._counts))

    def verify(self) -> bool:
        """Slow independent mirror; never called in the timed hot loop."""
        states = self._status
        duration = self._buffer.to_list()
        return (self._buffer.verify()
                and scan_reference(self.suites, len(self._counts), states, duration)
                == self.snapshot()
                and self._buffer.aggregates()[0] == self._global[4])

    @staticmethod
    def dependencies() -> dict[str, tuple[str, ...]]:
        return {"test[index]": ("suite[suite_of[index]]",),
                "suite[id]": ("global",)}


def scan_reference(suites: Sequence[int], suite_count: int,
                   statuses: Sequence[int], durations: Sequence[int]) -> QASnapshot:
    """O(N) witness, intentionally separate from both incremental algorithms."""
    if len(suites) != len(statuses) or len(statuses) != len(durations):
        raise ValueError("layout/state length mismatch")
    totals = [[0, 0, 0, 0, 0] for _ in range(suite_count)]
    overall = [0, 0, 0, 0, 0]
    for suite, status, duration in zip(suites, statuses, durations):
        totals[suite][status] += 1
        totals[suite][4] += duration
        overall[status] += 1
        overall[4] += duration
    return QASnapshot(tuple(overall), tuple(tuple(x) for x in totals))


__all__ = ["QAEvent", "QASnapshot", "StrongPythonQA", "AlexPyQAGraph",
           "scan_reference", "PASS", "FAIL", "SKIP", "NOT_RUN", "MAX_DURATION"]
