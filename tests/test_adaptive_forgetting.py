"""PHYSARUM-003: frozen policy, six unseen seeded streams, exact rational oracle."""
from fractions import Fraction
import json
import platform
import random
import statistics
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

from alexpy.physarum import solve
from alexpy.adaptive_forgetting import (AdaptiveForgetting, Certificate,
    affected_edges, exact_route, verify_certificate)
from test_physarum import quantile

MODES=("cold","dijkstra","adaptive")
HOLDOUT=(("mesh",120301),("mesh",120302),("ring",120303),
         ("ring",120304),("sparse",120305),("sparse",120306))


def scenario(family,seed):
    rng=random.Random(seed)
    n=16 if family=="mesh" else 18 if family=="ring" else 24
    edges={(i,i+1):rng.uniform(1,6) for i in range(n-1)}
    if family=="mesh":
        edges.update({(i,i+4):rng.uniform(1,6) for i in range(n-4)})
    elif family=="ring":
        edges[(0,n-1)]=rng.uniform(4,14)
        edges.update({(i,i+3):rng.uniform(2,9) for i in range(n-3)})
    else:
        edges.update({(i,j):rng.uniform(1,9) for i in range(n)
                      for j in range(i+2,n) if rng.random()<.09})
    rows=[]
    for tick in range(24):
        phase=tick%8
        if phase in (2,3):
            k=rng.choice(sorted(edges)); edges[k]*=.35 if phase==2 else 2.7
        elif phase==4:
            candidates=[e for e in sorted(edges) if e[1]-e[0]>1]
            if candidates: del edges[rng.choice(candidates)]
        elif phase==5:
            candidates=[(i,j) for i in range(n) for j in range(i+2,n) if (i,j) not in edges]
            if candidates: edges[rng.choice(candidates)]=rng.uniform(1,4)
        elif phase==6:
            for k in rng.sample(sorted(edges),min(6,len(edges))):
                edges[k]*=rng.choice((.4,2.5))
        rows.append((n,tuple((u,v,w) for (u,v),w in sorted(edges.items())),0,n-1))
    return rows


def oracle(args):
    n,edges,s,t=args
    dist=[None]*n; dist[s]=Fraction(0)
    for _ in range(n-1):
        changed=False
        for u,v,w in edges:
            w=Fraction(w)
            for a,b in ((u,v),(v,u)):
                if dist[a] is not None:
                    c=dist[a]+w
                    if dist[b] is None or c<dist[b]:
                        dist[b]=c; changed=True
        if not changed: break
    return dist[t]


def path_cost(args,path):
    n,edges,s,t=args
    if (not path or path[0]!=s or path[-1]!=t or len(path)>n
            or len(set(path))!=len(path)):
        raise AssertionError("invalid path")
    weights={(min(u,v),max(u,v)):Fraction(w) for u,v,w in edges}
    return sum((weights[min(u,v),max(u,v)] for u,v in zip(path,path[1:])),Fraction(0))


def execute(mode,rows,timed):
    mem=AdaptiveForgetting() if mode=="adaptive" else None
    out=[]; times=[]
    for args in rows:
        start=time.perf_counter_ns()
        if mode=="adaptive":
            r=mem.solve(*args)
            path,cert=r.path,r.certificate
            info=(r.source,r.budget,r.iterations,r.invalidated_edges,
                  r.reused_edges,r.proposal_accepted,r.exact_solves)
        elif mode=="cold":
            r=solve(*args,iterations=32)
            path,cert=r.path,None
            info=("cold",32,r.iterations,0,0,False,0)
        else:
            path,_,cert=exact_route(*args)
            info=("dijkstra",0,0,0,0,False,1)
        elapsed=time.perf_counter_ns()-start
        if timed: times.append(elapsed)
        out.append((path,cert,info))
    return out,times


def benchmark(family,seed):
    rows=scenario(family,seed)
    expected=[oracle(args) for args in rows]
    runs={m:[] for m in MODES}
    for i in range(5):
        k=(seed+i)%3
        for mode in MODES[k:]+MODES[:k]:
            runs[mode].append(execute(mode,rows,True))
    data={}
    for mode in MODES:
        out=runs[mode][0][0]
        ratios=[]; optimal=0
        for args,ref,(path,cert,_) in zip(rows,expected,out):
            cost=path_cost(args,path)
            if cost<ref: raise AssertionError("cost below exact oracle")
            if mode!="cold" and (cost!=ref or not verify_certificate(*args,path,cert)):
                raise AssertionError("uncertified or nonoptimal route")
            optimal+=cost==ref
            ratios.append(float(cost/ref))
        for other,_ in runs[mode]:
            if other!=out: raise AssertionError("nondeterministic session")
        tracemalloc.start()
        try:
            checked,_=execute(mode,rows,False)
            peak=tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        if checked!=out: raise AssertionError("memory run differs")
        times=[x for _,t in runs[mode] for x in t]
        infos=[v[2] for v in out]
        data[mode]=dict(n=len(rows),optimal=optimal,worst_ratio=max(ratios),
            mean_ratio=statistics.mean(ratios),median_ns=statistics.median(times),
            p95_ns=quantile(times,.95),p99_ns=quantile(times,.99),
            peak_traced_bytes=peak,samples=len(times),
            sources={k:sum(v[0]==k for v in infos) for k in sorted({v[0] for v in infos})},
            budgets={str(k):sum(v[1]==k for v in infos) for k in sorted({v[1] for v in infos})},
            iterations=sum(v[2] for v in infos),invalidated_edges=sum(v[3] for v in infos),
            reused_edges=sum(v[4] for v in infos),accepted=sum(v[5] for v in infos),
            exact_solves=sum(v[6] for v in infos),samples_ns=times,
            cost_ratios=ratios,paths=[list(x[0]) for x in out],trace=[list(v) for v in infos])
    return dict(family=family,seed=seed,results=data)


class AdaptiveForgettingTests(unittest.TestCase):
    def test_certificate_negative_controls(self):
        args=(4,[(0,1,1),(1,3,1),(0,2,2),(2,3,2)],0,3)
        path,cost,cert=exact_route(*args)
        self.assertEqual(cost,2)
        self.assertTrue(verify_certificate(*args,path,cert))
        for bad in ((0,2,3),(0,1,0,1,3),(0,3)):
            self.assertFalse(verify_certificate(*args,bad,cert))
        self.assertFalse(verify_certificate(*args,path,Certificate(1,(4,3,2,0))))
        self.assertFalse(verify_certificate(*args,path,Certificate(0,cert.labels)))

    def test_exact_binary_weights(self):
        args=(4,[(0,1,.1),(1,3,.2),(0,2,.15),(2,3,.15)],0,3)
        path,_,cert=exact_route(*args)
        self.assertEqual(path_cost(args,path),oracle(args))
        self.assertTrue(verify_certificate(*args,path,cert))
        other=(0,1,3) if path==(0,2,3) else (0,2,3)
        self.assertFalse(verify_certificate(*args,other,cert))

    def test_wave_union_and_no_change(self):
        old={(0,1):1,(1,2):1,(2,3):1,(3,4):1}
        new=dict(old); new[(0,1)]=2
        self.assertEqual(affected_edges(5,old,new,radius=0),{(0,1),(1,2)})
        self.assertEqual(affected_edges(5,old,new,radius=1),{(0,1),(1,2),(2,3)})
        self.assertEqual(affected_edges(5,old,old),set())
        new=dict(old); del new[(1,2)]
        self.assertEqual(affected_edges(5,old,new,radius=0),{(0,1),(2,3)})

    def test_cache_reversal_and_atomic_failure(self):
        mem=AdaptiveForgetting(radius=0)
        edges=[(0,1,1),(1,3,1),(0,2,4),(2,3,4)]
        self.assertEqual(mem.solve(4,edges,0,3).cost,2)
        again=mem.solve(4,list(reversed(edges)),0,3)
        self.assertEqual((again.source,again.iterations,again.exact_solves),("cached_path",0,0))
        changed=[(0,1,5),(1,3,5),(0,2,1),(2,3,1)]
        self.assertEqual(mem.solve(4,changed,0,3).path,(0,2,3))
        prior=mem._state
        with self.assertRaises(ValueError):
            mem.solve(4,[(0,1,1)],0,3)
        self.assertIs(mem._state,prior)
        r=mem.solve(4,changed,3,0)
        self.assertEqual(r.reused_edges,0)
        self.assertTrue(verify_certificate(4,changed,3,0,r.path,r.certificate))

    def test_fallback_on_solver_failure(self):
        mem=AdaptiveForgetting()
        args=(4,[(0,1,1),(1,3,1),(0,2,4),(2,3,4)],0,3)
        with patch("alexpy.adaptive_forgetting._solve_validated",side_effect=ArithmeticError("injected")):
            r=mem.solve(*args)
        self.assertEqual(r.source,"exact_fallback")
        self.assertEqual(r.exact_solves,1)
        self.assertFalse(r.proposal_accepted)
        self.assertTrue(verify_certificate(*args,r.path,r.certificate))
        self.assertTrue(all(d==1 for d in mem._state[2].values()))

    def test_invalid_config(self):
        for params in (dict(radius=-1),dict(radius=True),dict(blend=0),dict(blend=float("nan"))):
            with self.assertRaises(ValueError):
                AdaptiveForgetting(**params)

    def test_independent_holdouts(self):
        rows=[benchmark(f,s) for f,s in HOLDOUT]
        report=dict(schema="alexpy-physarum-003/1",environment=platform.platform(),
            python=platform.python_version(),rows=rows,default_promotion=False,
            timing_scope="full per-graph solve incl proof/fallback; external oracle excluded",
            memory_scope="tracemalloc peak full session incl retained outputs, external oracle excluded",
            scope="synthetic graphs; exactness over validated binary64 edge weights")
        Path("physarum003-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        for row in rows:
            for v in row["results"].values():
                for key in ("samples_ns","cost_ratios","paths","trace"): v.pop(key)
        print("ALEXPY_PHYSARUM003 "+json.dumps(report,separators=(",",":")),flush=True)
