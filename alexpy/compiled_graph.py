"""Opt-in compiled sparse frontier for the exact signed-q dependency DAG.

Compile once after declaration. Numeric node IDs replace string-keyed heap
items and typed C-buffer initialization avoids Python-per-element at calls.
Rejected patches never update cached graph values. All graph rules
remain explicit; this is not automatic tracing or a CPython optimization.
"""
from __future__ import annotations

import heapq

from .dependency_graph import DependencyGraph, GraphTransition, Node, Metric
from .memory_candidates import _buffer_native
from .persistent import _exact_q


class CompiledDependencyGraph(DependencyGraph):
    """Frozen registration and sparse propagation with numeric node IDs.

    compile() is a required explicit stage. After compilation, no node
    can be added. C-backed square-window initialization is used when the
    optional native extension is present; the exact Python fallback remains.
    """

    __slots__ = ("_compiled", "_ids", "_ordered", "_cached", "_parents",
                 "_offspring", "_bucket_ids")

    def __init__(self, values: list[int] | tuple[int, ...], *,
                 bucket_size: int = 256, trace: bool = False):
        super().__init__(values, bucket_size=bucket_size, trace=trace)
        self._compiled = False
        self._ids: dict[str, int] = {}
        self._ordered: tuple[Node, ...] = ()
        self._cached: list[int] = []
        self._parents: tuple[tuple[int, ...], ...] = ()
        self._offspring: tuple[tuple[int, ...], ...] = ()
        self._bucket_ids: dict[int, tuple[int, ...]] = {}

    def _register(self, node: Node, initial: int) -> None:
        if self._compiled:
            raise RuntimeError("graph is compiled; registrations are frozen")
        super()._register(node, initial)

    def add_window(self, name: str, start: int, stop: int,
                   metric: Metric) -> None:
        """Equivalent declaration with fewer Python object/index operations."""
        self._name_available(name)
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= len(self._buffer)):
            raise ValueError("window endpoints must satisfy 0 <= start <= stop <= n")
        if type(metric) is not str or metric not in ("sum", "squares"):
            raise ValueError("metric must be sum or squares")
        segment = self._buffer._data[start:stop]
        if metric == "sum":
            initial = sum(segment)
            kind = "window_sum"
        else:
            if _buffer_native is not None:
                initial = _buffer_native.sum_squares_linear(segment)
                if initial is NotImplemented:
                    initial = sum(v * v for v in segment)
            else:
                initial = sum(v * v for v in segment)
            kind = "window_squares"
        self._register(Node(name, kind, (), start, stop), initial)
        for bucket in range(start // self._bucket_size,
                            (stop - 1) // self._bucket_size + 1):
            if stop > start:
                self._buckets.setdefault(bucket, set()).add(name)

    def compile(self) -> None:
        """Freeze declarations and generate contiguous, topological IDs."""
        if self._compiled:
            return
        self._ordered = tuple(self._nodes.values())
        self._ids = {node.name: i for i, node in enumerate(self._ordered)}
        self._cached = [self._values[n.name] for n in self._ordered]
        self._parents = tuple(tuple(self._ids[p] for p in n.dependencies)
                              for n in self._ordered)
        self._offspring = tuple(tuple(sorted(self._ids[c] for c in self._children[n.name]))
                                for n in self._ordered)
        self._bucket_ids = {b: tuple(sorted(self._ids[name] for name in names))
                            for b, names in self._buckets.items()}
        self._compiled = True

    def apply_patch(self, changes: dict[int, int]) -> GraphTransition:
        if not self._compiled:
            raise RuntimeError("call compile() after registering graph nodes")
        if type(changes) is not dict:
            raise TypeError("patch must be an exact dict[index, int]")
        source = self._buffer._data
        prepared: list[tuple[int, int, int]] = []
        for index, proposed in changes.items():
            if type(index) is not int or not 0 <= index < len(source):
                raise IndexError("patch index must be an exact integer in range")
            new = _exact_q(proposed)
            old = source[index]
            if new != old:
                prepared.append((index, old, new))

        # Candidate window changes are created before committing anything.
        window_deltas: dict[int, int] = {}
        examined: set[int] = set()
        for index, old, new in prepared:
            delta_sum, delta_sq = new - old, new*new - old*old
            for ident in self._bucket_ids.get(index // self._bucket_size, ()):
                examined.add(ident)
                node = self._ordered[ident]
                if node.start <= index < node.stop:
                    delta = delta_sum if node.kind == "window_sum" else delta_sq
                    window_deltas[ident] = window_deltas.get(ident, 0) + delta

        pending = list(window_deltas)
        heapq.heapify(pending)
        scheduled = set(pending)
        staged: dict[int, int] = {}
        touched: list[str] = []
        evaluated: list[str] = []
        while pending:
            ident = heapq.heappop(pending)
            node = self._ordered[ident]
            before = self._cached[ident]
            if node.kind in ("window_sum", "window_squares"):
                touched.append(node.name)
                updated = before + window_deltas[ident]
            else:
                parents = self._parents[ident]
                if not any(p in staged for p in parents):
                    continue
                evaluated.append(node.name)
                if node.kind == "add":
                    updated = before + sum(staged[p] - self._cached[p]
                                           for p in parents if p in staged)
                elif node.kind == "variance":
                    left, right = parents
                    total = staged.get(left, self._cached[left])
                    squares = staged.get(right, self._cached[right])
                    updated = (node.stop-node.start)*squares - total*total
                else:
                    raise AssertionError("invalid compiled node kind")
            if updated != before:
                staged[ident] = updated
                for child in self._offspring[ident]:
                    if child not in scheduled:
                        heapq.heappush(pending, child)
                        scheduled.add(child)

        # OwnedQBuffer validates/commits the same patch once. Python int
        # operations above cannot overflow; a validation failure leaves all
        # cached nodes untouched.
        source_event = self._buffer.apply_patch(changes)
        for ident, value in staged.items():
            self._cached[ident] = value
            self._values[self._ordered[ident].name] = value
        event = GraphTransition(
            tick=source_event.tick, version=source_event.version,
            changed_indices=source_event.changed_indices,
            examined_windows=len(examined), touched_windows=tuple(touched),
            evaluated_derived=tuple(evaluated),
            updated_nodes=tuple(self._ordered[ident].name for ident in staged),
        )
        self._last = event
        if self._trace:
            self._history.append(event)
        return event


__all__ = ["CompiledDependencyGraph"]
