"""CAUSAL-002: dynamic output demand on an immutable exact-integer DAG.

Explicit opt-in. Dormant cache entries may be stale; only the active closure
is certified. On activation, parents are validated in topological order and
exact parent values (not hashes or timestamps) authorize reuse.
"""
from dataclasses import dataclass
from heapq import heappop, heappush
from .causal_engine import Graph, _evaluate

ROUTES = ('eager', 'active_only', 'lazy_dirty', 'causal')

@dataclass(frozen=True)
class DemandTransition:
    evaluated: tuple[int, ...]
    reused: tuple[int, ...]
    equal_boundary: tuple[int, ...]
    invalidated: int

class DynamicCausalEngine:
    """Fixed graph, variable requested output tuple; single-threaded contract.

    eager: all nodes current, value-aware forward propagation.
    active_only: same propagation in active cone, discard dormant nodes.
    lazy_dirty: invalidate cached descendants eagerly, refresh dirty demanded
      nodes in topological order with exact parent-value reuse.
    causal: propagate only in active cone, validate dormant nodes on demand.
    The cache is bounded by the fixed graph, not by number of transitions.
    No rollback guarantee for allocation failure or private state mutation.
    """
    def __init__(self, graph, values, mode='causal'):
        if type(graph) is not Graph or type(mode) is not str or mode not in ROUTES:
            raise ValueError('invalid graph or mode')
        if (type(values) not in (tuple, list) or len(values) != graph.inputs
                or any(type(v) is not int for v in values)):
            raise ValueError('exact integer input vector required')
        self._graph = graph
        self._mode = mode
        self._values = dict(enumerate(values))
        self._signatures = {}
        self._dirty = set()
        self._children = {}
        for i, node in enumerate(graph.nodes, graph.inputs):
            for p in node.parents:
                self._children.setdefault(p, []).append(i)
        self._outputs = ()
        self._live = set()
        self.tick = self.version = self.demand_version = 0
        if mode == 'eager':
            self._live = set(range(graph.inputs, graph.inputs+len(graph.nodes)))
            for i in sorted(self._live):
                self._values[i] = _evaluate(graph.nodes[i-graph.inputs], self._values.__getitem__)
        self.select(graph.outputs)
        self.demand_version = 0

    @property
    def outputs(self): return self._outputs
    @property
    def live(self): return frozenset(self._live)
    @property
    def cached_nodes(self): return len(self._values)-self._graph.inputs

    def read(self):
        return tuple(self._values[i] for i in self._outputs)

    def _cone(self, outputs):
        seen = set(); stack = list(outputs); n = self._graph.inputs
        while stack:
            i = stack.pop()
            if i < n or i in seen: continue
            seen.add(i); stack.extend(self._graph.nodes[i-n].parents)
        return seen

    def _refresh(self, ids, invalidated=0):
        evaluated, reused, equal = [], [], []
        values, sigs = self._values, self._signatures
        for i in sorted(ids):
            node = self._graph.nodes[i-self._graph.inputs]
            if self._mode in ('causal', 'lazy_dirty'):
                signature = tuple(values[p] for p in node.parents)
                if i in sigs and sigs[i] == signature:
                    reused.append(i)
                    self._dirty.discard(i)
                    continue
                sigs[i] = signature
            v = _evaluate(node, values.__getitem__)
            if i in values and values[i] == v: equal.append(i)
            values[i] = v; evaluated.append(i)
            self._dirty.discard(i)
        return DemandTransition(tuple(evaluated), tuple(reused), tuple(equal), invalidated)

    def select(self, outputs):
        n = self._graph.inputs + len(self._graph.nodes)
        if (type(outputs) is not tuple or not outputs or
                any(type(i) is not int or not 0 <= i < n for i in outputs) or
                len(set(outputs)) != len(outputs)):
            raise ValueError('invalid requested outputs')
        if outputs == self._outputs:
            return DemandTransition((), (), (), 0)
        if self._mode == 'eager':
            event = DemandTransition((), (), (), 0)
        else:
            live = self._cone(outputs)
            # Overlapping active nodes were exact before select; all parents
            # of newly active nodes are processed first by rank.
            newly_active = live - self._live
            if self._mode == 'lazy_dirty':
                # Clean cached nodes already have a valid invalidation proof.
                pending = {i for i in newly_active if i not in self._values or i in self._dirty}
                event = self._refresh(pending)
                event = DemandTransition(event.evaluated, tuple(sorted(set(event.reused) | (newly_active-pending))), event.equal_boundary, 0)
            else:
                event = self._refresh(newly_active)
            if self._mode == 'active_only':
                for i in self._live-live: del self._values[i]
            self._live = live
        self._outputs = outputs
        self.demand_version += 1
        return event

    def apply(self, patch):
        if type(patch) is not dict: raise TypeError('exact dict required')
        changed = {}
        for i, v in patch.items():
            if type(i) is not int or not 0 <= i < self._graph.inputs:
                raise IndexError('input index out of range')
            if type(v) is not int: raise TypeError('exact integer required')
            if self._values[i] != v: changed[i] = v
        self._values.update(changed)
        self.tick += 1; self.version += bool(changed)
        if not changed: return DemandTransition((), (), (), 0)
        if self._mode == 'lazy_dirty':
            # A cached descendant implies its ancestors have been cached.
            # Already-dirty nodes have already dirtied their descendants;
            # refreshing a node always refreshes all of its dirty ancestors.
            stack = list(changed); count = 0
            while stack:
                for child in self._children.get(stack.pop(), ()):
                    if child in self._values and child not in self._dirty:
                        self._dirty.add(child); stack.append(child); count += 1
            return self._refresh(self._dirty & self._live, count)
        queue, scheduled = [], set()
        def enqueue(i):
            for child in self._children.get(i, ()):
                if child in self._live and child not in scheduled:
                    scheduled.add(child); heappush(queue, child)
        for i in changed: enqueue(i)
        evaluated, equal = [], []
        while queue:
            i = heappop(queue); node = self._graph.nodes[i-self._graph.inputs]
            if self._mode == 'causal':
                self._signatures[i] = tuple(self._values[p] for p in node.parents)
            v = _evaluate(node, self._values.__getitem__); evaluated.append(i)
            if v == self._values[i]: equal.append(i)
            else:
                self._values[i] = v; enqueue(i)
        return DemandTransition(tuple(evaluated), (), tuple(equal), 0)
