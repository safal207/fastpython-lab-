"""PHYSARUM-003: experimental selective forgetting and exact route trust.

All edge weights are converted from the validated binary64 representation to
a common dyadic integer scale. A feasible potential with sink label zero is
a rigorous lower bound; equality with a valid path certifies optimality.
An independent exact Dijkstra is paid for whenever cached proof is insufficient.
No tolerance-based promotion, no DAG edits, no default activation.
"""
from collections import deque
from dataclasses import dataclass
import heapq
import math

from .physarum import graph, _flow, _solve_validated


@dataclass(frozen=True)
class Certificate:
    scale: int
    labels: tuple


@dataclass(frozen=True)
class TrustResult:
    path: tuple
    cost: float
    certificate: Certificate
    source: str
    budget: int
    iterations: int
    invalidated_edges: int
    reused_edges: int
    proposal_accepted: bool
    exact_solves: int


def _weights(edges):
    ratios = [(u,v,*w.as_integer_ratio()) for u,v,w in edges]
    scale = max(d for _,_,_,d in ratios)
    return scale,{(u,v):a*(scale//d) for u,v,a,d in ratios}


def _path_cost(weights,path,source,sink,n):
    if (type(path) is not tuple or not 2<=len(path)<=n
            or any(type(v) is not int or not 0<=v<n for v in path)
            or path[0]!=source or path[-1]!=sink or len(set(path))!=len(path)):
        return None
    total=0
    for u,v in zip(path,path[1:]):
        w=weights.get((min(u,v),max(u,v)))
        if w is None:
            return None
        total+=w
    return total


def _feasible(cert,n,weights,scale,sink):
    return (type(cert) is Certificate and type(cert.scale) is int
            and cert.scale>0 and type(cert.labels) is tuple
            and len(cert.labels)==n and all(type(x) is int and x>=0 for x in cert.labels)
            and cert.labels[sink]==0
            and all(abs(cert.labels[u]-cert.labels[v])*scale<=w*cert.scale
                    for (u,v),w in weights.items()))


def _proves(cert,weights,scale,path,source,sink,n):
    total=_path_cost(weights,path,source,sink,n)
    return (total is not None and _feasible(cert,n,weights,scale,sink)
            and total*cert.scale==cert.labels[source]*scale)


def verify_certificate(n,edges,source,sink,path,certificate):
    edges=graph(n,edges,source,sink)
    scale,weights=_weights(edges)
    return _proves(certificate,weights,scale,path,source,sink,n)


def _exact(n,weights,scale,source,sink):
    """Full reverse Dijkstra, exact integer arithmetic; no Physarum state."""
    adj=[[] for _ in range(n)]
    for (u,v),w in weights.items():
        adj[u].append((v,w)); adj[v].append((u,w))
    dist={sink:0}
    next_hop={}
    heap=[(0,sink)]
    while heap:
        cost,u=heapq.heappop(heap)
        if cost!=dist[u]:
            continue
        for v,w in adj[u]:
            c=cost+w
            if v not in dist or c<dist[v]:
                dist[v]=c
                next_hop[v]=u
                heapq.heappush(heap,(c,v))
    path=[source]
    while path[-1]!=sink:
        path.append(next_hop[path[-1]])
    cert=Certificate(scale,tuple(dist[v] for v in range(n)))
    return tuple(path),cert


def exact_route(n,edges,source,sink):
    edges=graph(n,edges,source,sink)
    scale,weights=_weights(edges)
    path,cert=_exact(n,weights,scale,source,sink)
    if not _proves(cert,weights,scale,path,source,sink,n):
        raise ArithmeticError("exact route proof failed")
    return path,cert.labels[source]/scale,cert


def affected_edges(n,old,new,*,radius=1):
    """Multi-source bounded wave over union of old and new topologies.

    A changed weight, insertion or deletion starts a wave at both endpoints.
    Radius counts hops. This is a heuristic for forgetting conductance only;
    it never serves as a proof of independence or correctness.
    """
    if type(radius) is not int or not 0<=radius<=8:
        raise ValueError("invalid wave radius")
    changed={e for e in old.keys()|new.keys() if old.get(e)!=new.get(e)}
    adj=[set() for _ in range(n)]
    for u,v in old.keys()|new.keys():
        adj[u].add(v); adj[v].add(u)
    seen={v for e in changed for v in e}
    queue=deque((v,0) for v in sorted(seen))
    while queue:
        v,d=queue.popleft()
        if d==radius:
            continue
        for u in sorted(adj[v]):
            if u not in seen:
                seen.add(u); queue.append((u,d+1))
    return {e for e in new if e[0] in seen or e[1] in seen}


class AdaptiveForgetting:
    """One bounded session; old state committed only after exact proof.

    Unchanged/compatible routes can be served from a verified certificate.
    Otherwise allocate 1/2/4/8 outer steps based on the affected edge fraction,
    and pay for exact Dijkstra if the cached certificate cannot prove the
    proposal. Failed approximate proposals never reinforce conductance.
    """
    def __init__(self,*,radius=1,blend=.25):
        if type(radius) is not int or not 0<=radius<=8:
            raise ValueError("invalid radius")
        if type(blend) not in (int,float) or not math.isfinite(blend) or not 0<blend<=1:
            raise ValueError("invalid blend")
        self.radius,self.blend=radius,blend
        self._state=None

    def solve(self,n,edges,source,sink):
        edges=graph(n,edges,source,sink)
        scale,weights=_weights(edges)
        identity=n,source,sink
        current={(u,v):w for u,v,w in edges}
        state=self._state
        if state is None or state[0]!=identity:
            old,conductance,old_path,cert={}, {}, (), None
        else:
            _,old,conductance,old_path,cert=state
        dirty=affected_edges(n,old,current,radius=self.radius)
        reused=sum(e in conductance and e not in dirty for e in current)
        initial=tuple(1. if (u,v) in dirty else
                      self.blend+(1-self.blend)*conductance.get((u,v),1.)
                      for u,v,_ in edges)
        budget=iterations=exact_solves=0
        accepted=False
        if _proves(cert,weights,scale,old_path,source,sink,n):
            path=old_path
            origin="cached_path"
            learned=initial
        else:
            fraction=len(dirty)/len(edges)
            budget=1 if fraction==0 else 2 if fraction<=.25 else 4 if fraction<=.6 else 8
            proposal=None
            try:
                proposal=_solve_validated(n,edges,source,sink,initial=initial,
                    iterations=budget,dt=.5,tolerance=1e-6,flow=_flow)
                iterations=proposal.iterations
            except ArithmeticError:
                iterations=budget # conservative attempted-work accounting
            if proposal is not None and _proves(cert,weights,scale,proposal.path,source,sink,n):
                path=proposal.path
                origin="cached_proof"
                accepted=True
            else:
                path,cert=_exact(n,weights,scale,source,sink)
                exact_solves=1
                accepted=proposal is not None and _proves(cert,weights,scale,proposal.path,source,sink,n)
                if accepted:
                    path=proposal.path
                origin="fresh_proof" if accepted else "exact_fallback"
            learned=proposal.conductance if accepted else (1.,)*len(edges)
        if not _proves(cert,weights,scale,path,source,sink,n):
            raise ArithmeticError("route proof failed")
        total=_path_cost(weights,path,source,sink,n)
        self._state=(identity,current,{(u,v):d for (u,v,_),d in zip(edges,learned)},path,cert)
        return TrustResult(path,total/scale,cert,origin,budget,iterations,len(dirty),reused,accepted,exact_solves)
