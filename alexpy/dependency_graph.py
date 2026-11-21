"""Experimental, declarative dependency DAG over an owned signed-q buffer.

All mutations are explicit patches. Registration creates an acyclic graph:
windows read input indices, and derived nodes reference already registered
nodes. Invalid patches are rejected before committing any graph state.

This is a bounded DAG + dirty-frontier experiment, not automatic tracing of
arbitrary Python code and not a replacement for CPython's execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Literal

from .persistent import OwnedQBuffer, _exact_q

Metric = Literal["sum", "squares"]
NodeKind = Literal["window_sum", "window_squares", "add", "variance"]


@dataclass(frozen=True)
class Node:
    name: str
    kind: NodeKind
    dependencies: tuple[str, ...] = ()
    start: int = 0
    stop: int = 0


@dataclass(frozen=True)
class GraphTransition:
    tick: int
    version: int
    changed_indices: tuple[int, ...]
    examined_windows: int
    touched_windows: tuple[str, ...]
    evaluated_derived: tuple[str, ...]
    updated_nodes: tuple[str, ...]


class DependencyGraph:
    """Maintain exact integer aggregate DAG via sparse, value-aware updates.

    Declared range windows are indexed by coarse buckets; a patch examines
    only windows in buckets containing changed indices. Downstream nodes are
    evaluated in registration/topological order only when an input *value*
    changed. Every node retains its cached exact integer result.

    After construction, external input aliases cannot mutate the owned data.
    `verify` is a full O(N) observer, outside the measured hot path.
    """

    __slots__ = ("_buffer", "_nodes", "_values", "_buckets", "_bucket_size",
                 "_children", "_rank", "_last", "_trace", "_history")

    def __init__(self, values: list[int] | tuple[int, ...], *,
                 bucket_size: int = 256, trace: bool = False):
        if type(bucket_size) is not int or bucket_size <= 0:
            raise ValueError("bucket_size must be an exact positive int")
        if type(trace) is not bool:
            raise TypeError("trace must be bool")
        self._buffer = OwnedQBuffer(values)
        self._nodes: dict[str, Node] = {}
        self._values: dict[str, int] = {}
        self._buckets: dict[int, set[str]] = {}
        self._bucket_size = bucket_size
        self._children: dict[str, set[str]] = {}
        self._rank: dict[str, int] = {}
        self._last: GraphTransition | None = None
        self._trace = trace
        self._history: list[GraphTransition] = []

    @property
    def version(self) -> int:
        return self._buffer.version

    @property
    def tick(self) -> int:
        return self._buffer.tick

    @property
    def last_transition(self) -> GraphTransition | None:
        return self._last

    @property
    def history(self) -> tuple[GraphTransition, ...]:
        return tuple(self._history)

    @property
    def data_bytes(self) -> int:
        return self._buffer.data_bytes

    def to_list(self) -> list[int]:
        """Return a copy, not a writable alias to the source."""
        return self._buffer.to_list()

    def value(self, name: str) -> int:
        return self._values[name]

    def snapshot(self) -> dict[str, int]:
        return self._values.copy()

    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((parent, node.name) for node in self._nodes.values()
                     for parent in node.dependencies)

    def _register(self, node: Node, initial: int) -> None:
        self._rank[node.name] = len(self._nodes)
        self._nodes[node.name] = node
        self._values[node.name] = initial
        self._children[node.name] = set()
        for parent in node.dependencies:
            self._children[parent].add(node.name)

    def _name_available(self, name: str) -> None:
        if type(name) is not str or not name:
            raise ValueError("node name must be a nonempty str")
        if name in self._nodes:
            raise ValueError(f"duplicate node: {name}")

    def add_window(self, name: str, start: int, stop: int,
                   metric: Metric) -> None:
        """Register range [start, stop), metric sum or sum of squares."""
        self._name_available(name)
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= len(self._buffer)):
            raise ValueError("window endpoints must satisfy 0 <= start <= stop <= n")
        if type(metric) is not str or metric not in ("sum", "squares"):
            raise ValueError("metric must be sum or squares")
        kind: NodeKind = "window_sum" if metric == "sum" else "window_squares"
        values = (self._buffer.at(i) for i in range(start, stop))
        initial = sum(values) if metric == "sum" else sum(v*v for v in values)
        node = Node(name, kind, (), start, stop)
        self._register(node, initial)
        if stop > start:
            for bucket in range(start // self._bucket_size,
                                (stop - 1) // self._bucket_size + 1):
                self._buckets.setdefault(bucket, set()).add(name)

    def add_sum(self, name: str, dependencies: tuple[str, ...]) -> None:
        """Add exact integer combination; parents must already exist."""
        self._name_available(name)
        if (type(dependencies) is not tuple or not dependencies
                or any(type(parent) is not str for parent in dependencies)
                or len(set(dependencies)) != len(dependencies)
                or any(parent not in self._nodes for parent in dependencies)):
            raise ValueError("dependencies must be a nonempty tuple of distinct existing nodes")
        self._register(Node(name, "add", dependencies),
                       sum(self._values[parent] for parent in dependencies))

    def add_variance(self, name: str, sum_node: str, squares_node: str) -> None:
        """Exact n*sum(x**2)-sum(x)**2 for two matching range nodes."""
        self._name_available(name)
        if sum_node not in self._nodes or squares_node not in self._nodes:
            raise ValueError("variance dependencies must already exist")
        left, right = self._nodes[sum_node], self._nodes[squares_node]
        if (left.kind != "window_sum" or right.kind != "window_squares"
                or (left.start, left.stop) != (right.start, right.stop)):
            raise ValueError("variance needs same-range sum and squares windows")
        n = left.stop - left.start
        self._register(Node(name, "variance", (sum_node, squares_node),
                            left.start, left.stop),
                       n*self._values[squares_node] - self._values[sum_node]**2)

    def to_dot(self) -> str:
        """Stable Graphviz DAG; last updated nodes are distinguished visually."""
        updated = set(self._last.updated_nodes) if self._last else set()
        touched = set(self._last.touched_windows) if self._last else set()
        lines = ["digraph AlexPyDependencies {", "  rankdir=LR;",
                 '  label="Declared dependencies, last patch frontier";']
        import json
        for name, node in self._nodes.items():
            ident = "n" + str(self._rank[name])
            color = ("#cbead0" if name in updated else
                     "#f6e3b4" if name in touched else "#e5e7eb")
            label = json.dumps(f"{name}\n{node.kind}")
            lines.append(f'  {ident} [label={label}, style=filled, fillcolor="{color}"];')
        for parent, child in self.edges():
            lines.append(f'  n{self._rank[parent]} -> n{self._rank[child]};')
        return "\n".join(lines + ["}", ""])

    def apply_patch(self, changes: dict[int, int]) -> GraphTransition:
        """Compute candidate DAG changes before a single committed buffer patch.

        Index check, exact-int type gate and all derived arithmetic occur
        before the backing buffer is mutated. Nodes remain unchanged if their
        numerical outputs remain unchanged (including sign-flip square sums).
        """
        if type(changes) is not dict:
            raise TypeError("patch must be an exact dict[index, int]")
        prepared: list[tuple[int, int, int]] = []
        for index, proposed in changes.items():
            if type(index) is not int or not 0 <= index < len(self._buffer):
                raise IndexError("patch index must be an exact integer in range")
            value = _exact_q(proposed)
            previous = self._buffer.at(index)
            if previous != value:
                prepared.append((index, previous, value))

        # Index touched cells to relevant windows, avoiding full-graph scans.
        window_deltas: dict[str, list[tuple[int, int]]] = {}
        examined: set[str] = set()
        for index, old, new in prepared:
            for name in self._buckets.get(index // self._bucket_size, ()):
                examined.add(name)
                node = self._nodes[name]
                if node.start <= index < node.stop:
                    window_deltas.setdefault(name, []).append((old, new))

        staged: dict[str, int] = {}
        touched: list[str] = []
        evaluated: list[str] = []
        # Node ranks are topological: any dependency precedes its child.
        # The heap contains only affected windows and their descendants.
        queue = [(self._rank[name], name) for name in window_deltas]
        heapq.heapify(queue)
        scheduled = set(window_deltas)
        while queue:
            _, name = heapq.heappop(queue)
            node = self._nodes[name]
            if node.kind in ("window_sum", "window_squares"):
                touched.append(name)
                subset = window_deltas[name]
                if node.kind == "window_sum":
                    delta = sum(new-old for old, new in subset)
                else:
                    delta = sum(new*new-old*old for old, new in subset)
                updated = self._values[name] + delta
            else:
                if not any(parent in staged for parent in node.dependencies):
                    continue
                evaluated.append(name)
                if node.kind == "add":
                    delta = sum(staged[parent] - self._values[parent]
                                for parent in node.dependencies if parent in staged)
                    updated = self._values[name] + delta
                elif node.kind == "variance":
                    left, right = node.dependencies
                    total = staged.get(left, self._values[left])
                    squares = staged.get(right, self._values[right])
                    updated = (node.stop-node.start)*squares - total*total
                else:
                    raise AssertionError("unexpected graph node kind")
            if updated != self._values[name]:
                staged[name] = updated
                for child in self._children[name]:
                    if child not in scheduled:
                        heapq.heappush(queue, (self._rank[child], child))
                        scheduled.add(child)

        # Under the owned-buffer contract this operation has already been
        # validated; commit once, then publish the already computed DAG values.
        source_event = self._buffer.apply_patch(changes)
        self._values.update(staged)
        event = GraphTransition(
            tick=source_event.tick,
            version=source_event.version,
            changed_indices=source_event.changed_indices,
            examined_windows=len(examined),
            touched_windows=tuple(touched),
            evaluated_derived=tuple(evaluated),
            updated_nodes=tuple(staged),
        )
        self._last = event
        if self._trace:
            self._history.append(event)
        return event

    def verify(self) -> bool:
        """Independently rebuild the full DAG; O(N + sum(range widths))."""
        if not self._buffer.verify():
            return False
        data = self._buffer.to_list()
        rebuilt: dict[str, int] = {}
        for name, node in self._nodes.items():
            if node.kind == "window_sum":
                value = sum(data[node.start:node.stop])
            elif node.kind == "window_squares":
                value = sum(pow(x, 2) for x in data[node.start:node.stop])
            elif node.kind == "add":
                value = sum(rebuilt[x] for x in node.dependencies)
            else:
                left, right = node.dependencies
                value = ((node.stop-node.start)*rebuilt[right]
                         - rebuilt[left]*rebuilt[left])
            rebuilt[name] = value
        return rebuilt == self._values


__all__ = ["DependencyGraph", "GraphTransition", "Node"]
