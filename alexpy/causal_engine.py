"""CAUSAL-001: explicit exact-integer DAG with a fixed demand cone.

Research only. No arbitrary callbacks, hidden inputs, floating point or
live graph/output changes. All routes have identical observable outputs.
"""
from dataclasses import dataclass
from heapq import heappop, heappush

MODES = ('full', 'dirty', 'value', 'slice_full', 'slice_dirty', 'causal')


@dataclass(frozen=True)
class Node:
    op: str
    parents: tuple[int, ...]


@dataclass(frozen=True)
class Graph:
    inputs: int
    nodes: tuple[Node, ...]
    outputs: tuple[int, ...]

    def __post_init__(self):
        if type(self.inputs) is not int or self.inputs < 1:
            raise ValueError('positive input count required')
        if type(self.nodes) is not tuple or type(self.outputs) is not tuple:
            raise TypeError('immutable tuples required')
        for i, n in enumerate(self.nodes, self.inputs):
            if (type(n) is not Node or type(n.op) is not str or
                    n.op not in ('sum', 'max', 'square', 'positive') or
                    type(n.parents) is not tuple or not n.parents or
                    any(type(p) is not int or not 0 <= p < i for p in n.parents) or
                    len(set(n.parents)) != len(n.parents) or
                    (n.op in ('square', 'positive') and len(n.parents) != 1)):
                raise ValueError('invalid topological node')
        if (not self.outputs or any(type(i) is not int or not 0 <= i < self.inputs+len(self.nodes)
                                    for i in self.outputs) or
                len(set(self.outputs)) != len(self.outputs)):
            raise ValueError('invalid requested outputs')

    def cone(self):
        seen = set(self.outputs)
        stack = list(self.outputs)
        while stack:
            i = stack.pop()
            if i >= self.inputs:
                for p in self.nodes[i-self.inputs].parents:
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
        return tuple(sorted(i for i in seen if i >= self.inputs))


@dataclass(frozen=True)
class Transition:
    evaluated: tuple[int, ...]
    equal_boundary: tuple[int, ...]
    changed: tuple[int, ...]


def _evaluate(node, get):
    if node.op == 'sum':
        return sum(get(p) for p in node.parents)
    if node.op == 'max':
        return max(get(p) for p in node.parents)
    x = get(node.parents[0])
    return x*x if node.op == 'square' else int(x > 0)


class CausalEngine:
    """Fixed graph, fixed requested outputs, owned exact integer inputs.

    `value` is the strong full-DAG value-aware control; `slice_dirty`
    isolates the effect of output slicing without equality cutoffs.
    `slice_full` is a static pruned full-recompute control.
    """
    def __init__(self, graph, values, mode='causal'):
        if type(graph) is not Graph or type(mode) is not str or mode not in MODES:
            raise ValueError('invalid graph or mode')
        if (type(values) not in (list, tuple) or len(values) != graph.inputs or
                any(type(v) is not int for v in values)):
            raise ValueError('exact integer input vector required')
        self.graph = graph
        self.mode = mode
        self.live = graph.cone() if mode in ('slice_full', 'slice_dirty', 'causal') else tuple(range(graph.inputs, graph.inputs+len(graph.nodes)))
        self._values = dict(enumerate(values))
        self._children = {}
        for i in self.live:
            node = graph.nodes[i-graph.inputs]
            self._values[i] = _evaluate(node, self._values.__getitem__)
            if mode not in ('full', 'slice_full'):
                for p in node.parents:
                    self._children.setdefault(p, []).append(i)
        self.tick = self.version = 0

    def read(self):
        return tuple(self._values[i] for i in self.graph.outputs)

    def apply(self, patch):
        if type(patch) is not dict:
            raise TypeError('exact dict required')
        staged = {}
        for i, v in patch.items():
            if type(i) is not int or not 0 <= i < self.graph.inputs:
                raise IndexError('input index out of range')
            if type(v) is not int:
                raise TypeError('exact integer required')
            if self._values[i] != v:
                staged[i] = v
        source_changed = bool(staged)
        evaluated, equal = [], []
        def get(i):
            return staged[i] if i in staged else self._values[i]
        def evaluate(i):
            v = _evaluate(self.graph.nodes[i-self.graph.inputs], get)
            evaluated.append(i)
            if v == self._values[i]:
                equal.append(i)
                return False
            staged[i] = v
            return True
        if self.mode in ('full', 'slice_full'):
            for i in self.live:
                evaluate(i)
        else:
            queue, scheduled = [], set()
            def enqueue(i):
                for child in self._children.get(i, ()):
                    if child not in scheduled:
                        scheduled.add(child)
                        heappush(queue, child)
            for i in staged:
                enqueue(i)
            while queue:
                i = heappop(queue)
                changed = evaluate(i)
                if changed or self.mode in ('dirty', 'slice_dirty'):
                    enqueue(i)
        # All patch validation and integer evaluation precede commit.
        # Not a guarantee of transactional rollback on out-of-memory.
        event = Transition(tuple(evaluated), tuple(equal), tuple(sorted(staged)))
        self._values.update(staged)
        self.tick += 1
        self.version += source_changed
        return event
