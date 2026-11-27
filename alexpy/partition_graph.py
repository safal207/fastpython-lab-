"""Opt-in structure-specialized dependency DAG (exact signed-q input only).

A compiler may recognize a proven graph template: disjoint, bucket-aligned
windows of sum and sum-of-squares, one variance per window, and a final sum
of the window totals. That template needs no generic dirty-frontier heap.
Other DAG structures always use CompiledDependencyGraph.apply_patch().

This is a narrow compiler optimization, NOT tracing arbitrary Python code.
"""
from __future__ import annotations

from .compiled_graph import CompiledDependencyGraph
from .dependency_graph import GraphTransition
from .persistent import _exact_q


class PartitionDependencyGraph(CompiledDependencyGraph):
    """Use a prevalidated partition plan when declaration matches exactly."""

    __slots__ = ("_partition_plan", "_global_id")

    def __init__(self, values, *, bucket_size=256, trace=False):
        super().__init__(values, bucket_size=bucket_size, trace=trace)
        self._partition_plan = None
        self._global_id = None

    @property
    def specialized(self) -> bool:
        return self._partition_plan is not None

    def compile(self) -> None:
        super().compile()
        # Do not optimize an arbitrary graph by guessing its semantics.
        # The descriptor proof checks both data coverage and DAG edges.
        nodes = self._ordered
        width = self._bucket_size
        length = len(self._buffer)
        count = (length + width - 1) // width
        if not count or len(nodes) != 3 * count + 1:
            self._partition_plan = None
            self._global_id = None
            return
        plan = []
        for part in range(count):
            start, stop = part * width, min(length, (part + 1) * width)
            triple = nodes[3*part:3*part+3]
            if not (triple[0].kind == "window_sum"
                    and triple[1].kind == "window_squares"
                    and triple[2].kind == "variance"
                    and all(n.start == start and n.stop == stop for n in triple)
                    and not triple[0].dependencies and not triple[1].dependencies
                    and triple[2].dependencies == (triple[0].name, triple[1].name)):
                self._partition_plan = None
                self._global_id = None
                return
            plan.append((3*part, 3*part+1, 3*part+2, stop-start))
        global_id = 3*count
        global_node = nodes[global_id]
        if not (global_node.kind == "add" and
                global_node.dependencies == tuple(nodes[3*p].name for p in range(count))):
            self._partition_plan = None
            self._global_id = None
            return
        # Each index must belong to exactly one declared partition. The
        # prior checks establish this by exact [start, stop) alignment.
        self._partition_plan = tuple(plan)
        self._global_id = global_id

    def apply_patch(self, changes: dict[int, int]) -> GraphTransition:
        if not self._compiled:
            raise RuntimeError("call compile() after registering graph nodes")
        plan = self._partition_plan
        if plan is None:
            return super().apply_patch(changes)
        if type(changes) is not dict:
            raise TypeError("patch must be an exact dict[index, int]")
        source = self._buffer._data
        width = self._bucket_size
        deltas: dict[int, tuple[int, int]] = {}
        prepared: list[tuple[int, int, int]] = []
        for index, proposed in changes.items():
            if type(index) is not int or not 0 <= index < len(source):
                raise IndexError("patch index must be an exact integer in range")
            new = _exact_q(proposed)
            old = source[index]
            if new == old:
                continue
            prepared.append((index, old, new))
            part = index // width
            ds, dq = deltas.get(part, (0, 0))
            deltas[part] = ds + (new-old), dq + (new*new-old*old)

        updated: dict[int, int] = {}
        touched: list[str] = []
        evaluated: list[str] = []
        global_delta = 0
        any_sum_changed = False
        for part, (ds, dq) in sorted(deltas.items()):
            s_id, q_id, v_id, size = plan[part]
            # These two windows were examined even if aggregate deltas cancel.
            touched.extend((self._ordered[s_id].name, self._ordered[q_id].name))
            old_s, old_q = self._cached[s_id], self._cached[q_id]
            new_s, new_q = old_s + ds, old_q + dq
            if ds:
                updated[s_id] = new_s
                global_delta += ds
                any_sum_changed = True
            if dq:
                updated[q_id] = new_q
            if ds or dq:
                evaluated.append(self._ordered[v_id].name)
                new_v = size * new_q - new_s*new_s
                if new_v != self._cached[v_id]:
                    updated[v_id] = new_v
        if any_sum_changed:
            evaluated.append(self._ordered[self._global_id].name)
            if global_delta:
                updated[self._global_id] = self._cached[self._global_id] + global_delta

        # Commit occurs only after all indices, types and derived arithmetic
        # have been validated/staged. OwnedQBuffer owns the source of truth.
        source_event = self._buffer._commit_prepared(prepared)
        for ident, value in updated.items():
            self._cached[ident] = value
            self._values[self._ordered[ident].name] = value
        event = GraphTransition(
            tick=source_event.tick, version=source_event.version,
            changed_indices=source_event.changed_indices,
            examined_windows=2*len(deltas), touched_windows=tuple(touched),
            evaluated_derived=tuple(evaluated),
            updated_nodes=tuple(self._ordered[i].name for i in sorted(updated)),
        )
        self._last = event
        if self._trace:
            self._history.append(event)
        return event


__all__ = ["PartitionDependencyGraph"]
