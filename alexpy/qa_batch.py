"""Opt-in atomic QA-event batches over an owned duration buffer.

A batch is one transaction and one visible snapshot boundary. Every event,
including overwritten retries, is validated before any mutation. Repeated
IDs follow last-write-wins *within the batch*: intermediate statuses are not
observable and cannot be used to reproduce per-event notifications.

No automatic promotion. Keep StrongPythonBatchQA as a strong control, not
merely the old per-event Python baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

from .qa_lab import AlexPyQAGraph, StrongPythonQA, QAEvent, _validate


@dataclass(frozen=True, slots=True)
class QABatchTransition:
    tick: int
    version: int
    received_events: int
    distinct_test_ids: int
    changed_indices: tuple[int, ...]
    affected_suites: tuple[int, ...]


def _prepare(events: list[QAEvent] | tuple[QAEvent, ...], n: int):
    """Fully validate the batch; retain final state per test in insertion order."""
    if type(events) not in (list, tuple):
        raise TypeError("events must be an exact list/tuple of QAEvent")
    latest: dict[int, tuple[int, int]] = {}
    for event in events:
        _validate(event, n)
        latest[event.test_id] = (event.status, event.duration_ms)
    return latest


class StrongPythonBatchQA(StrongPythonQA):
    """Independent O(unique IDs) Python baseline with atomic batch semantics."""

    __slots__ = ("last_batch",)

    def __init__(self, suites, suite_count):
        super().__init__(suites, suite_count)
        self.last_batch = QABatchTransition(0, 0, 0, 0, (), ())

    def apply_batch(self, events: list[QAEvent] | tuple[QAEvent, ...]):
        latest = _prepare(events, len(self.suites))
        changed: list[int] = []
        affected: set[int] = set()
        for idx, (status, duration) in latest.items():
            previous, old_duration = self._status[idx], self._durations[idx]
            if previous == status and old_duration == duration:
                continue
            suite = self.suites[idx]
            counts = self._counts[suite]
            counts[previous] -= 1
            counts[status] += 1
            counts[4] += duration - old_duration
            global_counts = self._global
            global_counts[previous] -= 1
            global_counts[status] += 1
            global_counts[4] += duration - old_duration
            self._status[idx] = status
            self._durations[idx] = duration
            changed.append(idx)
            affected.add(suite)
        self.tick += 1
        if changed:
            self.version += 1
        self.last_batch = QABatchTransition(
            self.tick, self.version, len(events), len(latest),
            tuple(sorted(changed)), tuple(sorted(affected))
        )
        return self.last_batch


class AlexPyBatchQAGraph(AlexPyQAGraph):
    """One validated signed-q buffer commit and one sparse QA DAG update.

    Validation precedes all writes. The private _commit_prepared path is used
    only with exact validated IDs, signed-q values and captured current values;
    this is intentionally an internal research optimization, not public API.
    """

    __slots__ = ("last_batch",)

    def __init__(self, suites, suite_count):
        super().__init__(suites, suite_count)
        self.last_batch = QABatchTransition(0, 0, 0, 0, (), ())

    def apply_batch(self, events: list[QAEvent] | tuple[QAEvent, ...]):
        latest = _prepare(events, len(self.suites))
        data = self._buffer._data
        changed: list[tuple[int, int, int, int]] = []
        durations_patch: list[tuple[int, int, int]] = []
        affected: set[int] = set()
        for idx, (status, duration) in latest.items():
            previous, old_duration = self._status[idx], data[idx]
            if previous == status and old_duration == duration:
                continue
            changed.append((idx, previous, status, duration - old_duration))
            if duration != old_duration:
                durations_patch.append((idx, old_duration, duration))
            affected.add(self.suites[idx])
        # _prepare validated *all* events. Build deltas before committing to
        # avoid partial state for malformed batches. Integer sums cannot overflow.
        self._buffer._commit_prepared(durations_patch)
        for idx, old_status, new_status, delta_ms in changed:
            suite_counts = self._counts[self.suites[idx]]
            suite_counts[old_status] -= 1
            suite_counts[new_status] += 1
            suite_counts[4] += delta_ms
            global_counts = self._global
            global_counts[old_status] -= 1
            global_counts[new_status] += 1
            global_counts[4] += delta_ms
            self._status[idx] = new_status
        self.tick += 1
        if changed:
            self.version += 1
        self.last_affected_suites = tuple(sorted(affected))
        self.last_batch = QABatchTransition(
            self.tick, self.version, len(events), len(latest),
            tuple(sorted(idx for idx, *_ in changed)), self.last_affected_suites
        )
        return self.last_batch


__all__ = ["QABatchTransition", "StrongPythonBatchQA", "AlexPyBatchQAGraph"]
