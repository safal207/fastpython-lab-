"""Genesis PHYSARUM-001: bounded, opt-in flow solver for alternative paths.

Undirected positive weighted graph, unit source/sink current. Solve grounded
weighted Laplacian for pressures; update conductance toward absolute flow.
This is a numerical algorithm analogy, NOT a biological or physical energy
measurement. Approximate path proposals are not correctness proofs for
execution plans. Never modifies a dependency DAG or the public dispatcher.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math

from .ant_memory import ROUTES, Workload

FLOOR = 1e-6


@dataclass(frozen=True)
class FlowResult:
    path: tuple
    cost: float
    iterations: int
    converged: bool
    residual: float
    conductance: tuple


def graph(n, edges, source, sink):
    if type(n) is not int or not 2 <= n <= 64:
        raise ValueError("bounded graph: 2..64 vertices")
    if (type(source) is not int or type(sink) is not int
            or not 0 <= source < n or not 0 <= sink < n or source == sink):
        raise ValueError("invalid terminals")
    result = {}
    adjacent = [set() for _ in range(n)]
    for a, b, length in edges:
        if (type(a) is not int or type(b) is not int or not 0 <= a < n
                or not 0 <= b < n or a == b
                or type(length) not in (int, float) or not math.isfinite(length)
                or not 1e-6 <= length <= 1e6):
            raise ValueError("invalid edge")
        key = min(a,b), max(a,b)
        if key in result:
            raise ValueError("duplicate undirected edge")
        result[key] = float(length)
        adjacent[a].add(b)
        adjacent[b].add(a)
    seen, todo = {source}, [source]
    while todo:
        for b in adjacent[todo.pop()]:
            if b not in seen:
                seen.add(b)
                todo.append(b)
    if len(seen) != n:
        raise ValueError("connected graph required")
    return tuple((a,b,result[a,b]) for a,b in sorted(result))


def _linear(a, b):
    """Small dense Gaussian elimination with partial pivoting."""
    n = len(b)
    for i in range(n):
        k = max(range(i,n), key=lambda k: abs(a[k][i]))
        if not math.isfinite(a[k][i]) or abs(a[k][i]) < 1e-20:
            raise ArithmeticError("ill-conditioned pressure system")
        a[i], a[k] = a[k], a[i]
        b[i], b[k] = b[k], b[i]
        pivot = a[i][i]
        for j in range(i+1,n):
            scale = a[j][i]/pivot
            if scale:
                for k in range(i+1,n):
                    a[j][k] -= scale*a[i][k]
                b[j] -= scale*b[i]
            a[j][i] = 0.
    x = [0.]*n
    for i in range(n-1,-1,-1):
        x[i] = (b[i]-sum(a[i][j]*x[j] for j in range(i+1,n)))/a[i][i]
    if not all(math.isfinite(v) for v in x):
        raise ArithmeticError("nonfinite pressures")
    return x


def _flow(n, edges, conductance, source, sink):
    ids = {v:i for i,v in enumerate(v for v in range(n) if v != sink)}
    a = [[0.]*(n-1) for _ in range(n-1)]
    b = [0.]*(n-1)
    b[ids[source]] = 1.
    for (u,v,l), d in zip(edges,conductance):
        g = d/l
        if u != sink:
            a[ids[u]][ids[u]] += g
        if v != sink:
            a[ids[v]][ids[v]] += g
        if u != sink and v != sink:
            a[ids[u]][ids[v]] -= g
            a[ids[v]][ids[u]] -= g
    x = _linear(a,b)
    p = [0. if v == sink else x[ids[v]] for v in range(n)]
    q = tuple(d/l*(p[u]-p[v]) for (u,v,l),d in zip(edges,conductance))
    balance = [0.]*n
    for (u,v,_), f in zip(edges,q):
        balance[u] += f
        balance[v] -= f
    residual = max(abs(v-(1. if i==source else -1. if i==sink else 0.))
                   for i,v in enumerate(balance))
    if not math.isfinite(residual) or residual > 1e-6:
        raise ArithmeticError("flow conservation failed")
    return p,q,residual


def solve(n, edges, source, sink, *, initial=None, iterations=96, dt=.5, tolerance=1e-6):
    """Return a bounded approximate path, not an exact shortest-path claim.

    Initial conductance is aligned with canonical sorted undirected edges.
    Extraction follows positive downhill flow (strictly decreasing pressure).
    Nonconvergence is explicit; no hidden Dijkstra fallback.
    """
    edges = graph(n,edges,source,sink)
    return _solve_validated(n, edges, source, sink, initial=initial,
                            iterations=iterations, dt=dt, tolerance=tolerance, flow=_flow)


def _solve_validated(n, edges, source, sink, *, initial, iterations, dt, tolerance, flow):
    """Shared conductance update and path extraction; caller validates graph."""
    if (type(iterations) is not int or not 1 <= iterations <= 512
            or type(dt) not in (int,float) or not 0 < dt <= 1
            or type(tolerance) not in (int,float) or not 0 < tolerance < 1):
        raise ValueError("invalid solver budget")
    d = [1.]*len(edges) if initial is None else list(initial)
    if len(d) != len(edges) or any(type(x) not in (int,float) or
            not math.isfinite(x) or not FLOOR <= x <= 1. for x in d):
        raise ValueError("invalid conductance state")
    converged = False
    for step in range(1,iterations+1):
        _,q,_ = flow(n,edges,d,source,sink)
        new = [max(FLOOR,min(1.,(1-dt)*old+dt*abs(f))) for old,f in zip(d,q)]
        delta = max(abs(a-b) for a,b in zip(d,new))
        d = new
        if delta <= tolerance:
            converged = True
            break
    p,q,residual = flow(n,edges,d,source,sink)
    outgoing = [[] for _ in range(n)]
    for (u,v,l),f in zip(edges,q):
        if f > 0 and p[u] > p[v]:
            outgoing[u].append((f,v,l))
        elif f < 0 and p[v] > p[u]:
            outgoing[v].append((-f,u,l))
    path,cost = [source],0.
    while path[-1] != sink:
        if not outgoing[path[-1]] or len(path) >= n:
            raise ArithmeticError("no conserved downhill path")
        _,v,l = max(outgoing[path[-1]], key=lambda x:(x[0],-x[1]))
        if v in path:
            raise ArithmeticError("cyclic path")
        path.append(v)
        cost += l
    return FlowResult(tuple(path),cost,step,converged,residual,tuple(d))


def plan_graph(costs):
    """Parallel full-plan alternatives, not required dependency edges."""
    if not costs or any(r not in ROUTES for r in costs):
        raise ValueError("only registered alternatives")
    allowed = tuple(r for r in ROUTES if r in costs)
    sink = len(allowed)+1
    edges = []
    for i,r in enumerate(allowed,1):
        c = costs[r]
        if type(c) not in (int,float) or not math.isfinite(c) or not 2e-6 <= c <= 2e6:
            raise ValueError("invalid plan cost")
        edges.extend(((0,i,c/2),(i,sink,c/2)))
    return allowed,sink+1,edges,sink


class PhysarumMemory:
    """Bounded per-workload conductance memory with exact topology isolation.

    Only previously observed conductance from the same workload cell and
    allowed-route set is reused. Blend with a uniform prior to avoid permanent
    starvation. No unseen costs are read and no automatic activation occurs.
    """
    def __init__(self, capacity=32):
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("invalid memory capacity")
        self.capacity = capacity
        self._cells = OrderedDict()

    def choose(self, workload, costs, *, iterations=32, verified=False):
        if type(workload) is not Workload or verified is not True:
            raise ValueError("only verified complete alternative plans")
        allowed,n,edges,sink = plan_graph(costs)
        key = workload.cell(),allowed
        prior = self._cells.get(key)
        initial = None if prior is None else tuple(.75*d+.25 for d in prior)
        result = solve(n,edges,0,sink,initial=initial,iterations=iterations)
        route = allowed[result.path[1]-1]
        if route not in costs or len(result.path)!=3 or not math.isclose(result.cost,costs[route]):
            raise AssertionError("invalid plan proposal")
        self._cells[key] = result.conductance
        self._cells.move_to_end(key)
        if len(self._cells)>self.capacity:
            self._cells.popitem(last=False)
        return route,result

    @property
    def cells(self):
        return len(self._cells)


def propose(costs, *, iterations=32):
    allowed,n,edges,sink = plan_graph(costs)
    result = solve(n,edges,0,sink,iterations=iterations)
    route = allowed[result.path[1]-1]
    if len(result.path)!=3 or not math.isclose(result.cost,costs[route]):
        raise AssertionError("invalid plan proposal")
    return route,result
