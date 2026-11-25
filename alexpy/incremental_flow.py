"""PHYSARUM-002: opt-in sparse warm-start pressure solver.

Reuse conductance by canonical edge identity, including partial topology
changes; use previous grounded pressures as a PCG initial guess only.
Every accepted flow passes a conservation check; direct elimination is the
explicit fallback. No approximate flow is silently accepted. A valid path
is not necessarily a shortest path. No DAG or default dispatcher changes.
"""
from dataclasses import dataclass
import math

from .physarum import FLOOR, graph, _flow, _solve_validated


@dataclass(frozen=True)
class Diagnostics:
    pressure_solves: int
    cg_steps: int
    direct_solves: int
    pressure_reuses: int
    edge_reuses: int
    exact_flow_hits: int


class _Pressure:
    def __init__(self, n, edges, source, sink, prior=None):
        self.ids = {v:i for i,v in enumerate(v for v in range(n) if v!=sink)}
        self.n, self.source, self.sink = n,source,sink
        self.edges = edges
        self.links = tuple((self.ids.get(u,-1),self.ids.get(v,-1),l)
                           for u,v,l in edges)
        self.x = None if prior is None else list(prior)
        self.cached_d = self.cached_result = None
        self.solves = self.steps = self.direct = self.reuses = self.hits = 0

    def __call__(self,n,edges,d,source,sink):
        self.solves += 1
        key = tuple(d)
        if self.cached_d == key:
            self.hits += 1
            return self.cached_result
        size = n-1
        g = [v/l for v,(_,_,l) in zip(d,self.links)]
        diag = [0.]*size
        for (i,j,_), w in zip(self.links,g):
            if i>=0: diag[i] += w
            if j>=0: diag[j] += w

        def mul(x):
            out = [0.]*size
            for (i,j,_),w in zip(self.links,g):
                a = x[i] if i>=0 else 0.
                b = x[j] if j>=0 else 0.
                f = w*(a-b)
                if i>=0: out[i] += f
                if j>=0: out[j] -= f
            return out

        def dot(a,b):
            return sum(u*v for u,v in zip(a,b))

        rhs = [0.]*size
        rhs[self.ids[source]] = 1.
        x = [0.]*size if self.x is None else self.x.copy()
        if self.x is not None:
            self.reuses += 1
        residual = [a-b for a,b in zip(rhs,mul(x))]
        z = [r/a for r,a in zip(residual,diag)]
        v = z.copy()
        rz = dot(residual,z)
        for _ in range(2*size):
            if max(map(abs,residual)) <= 1e-9:
                break
            av = mul(v)
            den = dot(v,av)
            if not math.isfinite(den) or den<=0 or not math.isfinite(rz) or rz<=0:
                break
            alpha = rz/den
            x = [a+alpha*b for a,b in zip(x,v)]
            residual = [a-alpha*b for a,b in zip(residual,av)]
            self.steps += 1
            z = [r/a for r,a in zip(residual,diag)]
            new = dot(residual,z)
            if not math.isfinite(new) or new<0:
                break
            v = [a+(new/rz)*b for a,b in zip(z,v)]
            rz = new

        # Recompute actual edge fluxes and residual, never trust recursive CG
        # residual alone. Include the grounded sink in the conservation test.
        p = [0. if i==sink else x[self.ids[i]] for i in range(n)]
        q = tuple(w*(p[u]-p[v]) for (u,v,_),w in zip(edges,g))
        balance = [0.]*n
        for (u,v,_),f in zip(edges,q):
            balance[u] += f
            balance[v] -= f
        err = max(abs(b-(1. if i==source else -1. if i==sink else 0.))
                  for i,b in enumerate(balance))
        if (not all(math.isfinite(v) for v in p+list(q))
                or not math.isfinite(err) or err>1e-8):
            self.direct += 1
            p,q,err = _flow(n,edges,d,source,sink)
        self.x = [p[i] for i in range(n) if i!=sink]
        self.cached_d,self.cached_result = key,(p,q,err)
        return self.cached_result


class IncrementalFlowMemory:
    """One bounded graph-session cache; no cross-terminal pressure reuse.

    Conductance uses a uniform blend to allow previously weak routes to
    recover. Changed topology matches edges by endpoints, never by index.
    Invalid input or solver failure leaves the previous state intact.
    """
    def __init__(self, *, pressure=True, blend=.25):
        if type(pressure) is not bool or not math.isfinite(blend) or not 0<blend<=1:
            raise ValueError("invalid memory configuration")
        self.pressure,self.blend = pressure,blend
        self._state = None

    def solve(self,n,edges,source,sink,*,iterations=32,dt=.5,tolerance=1e-6):
        edges = graph(n,edges,source,sink)
        identity = n,source,sink
        prior = self._state
        old_d,old_p = ({},None) if prior is None or prior[0]!=identity else prior[1:]
        reused = sum((u,v) in old_d for u,v,_ in edges)
        initial = tuple((1-self.blend)*old_d.get((u,v),1.)+self.blend
                        for u,v,_ in edges)
        engine = _Pressure(n,edges,source,sink,old_p) if self.pressure else None
        result = _solve_validated(n,edges,source,sink,initial=initial,
                    iterations=iterations,dt=dt,tolerance=tolerance,
                    flow=engine if engine is not None else _flow)
        self._state = (identity,{(u,v):d for (u,v,_),d in zip(edges,result.conductance)},
                       None if engine is None else tuple(engine.x))
        info = (Diagnostics(result.iterations+1,0,result.iterations+1,0,reused,0)
                if engine is None else Diagnostics(engine.solves,engine.steps,
                    engine.direct,engine.reuses,reused,engine.hits))
        return result,info
