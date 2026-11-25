"""PHYSARUM-002: independent dynamic graph families, fixed iteration budgets.

All policies see the same exact graph at each step; no future weights.
Measured latency includes validation, pressure solves, memory and extraction.
Approximate proposals are checked for feasibility and scored against an
independent shortest-path oracle, not promoted on numerical convergence.
"""
import json
import math
import platform
import random
import statistics
import time
import tracemalloc
import unittest
from pathlib import Path

from alexpy.physarum import solve, graph, _flow
from alexpy.incremental_flow import IncrementalFlowMemory, _Pressure
from test_physarum import bellman, exact_path, assert_path, quantile

BUDGETS = (1,2,4,8,16,32)
MODES = ("cold","conductance","pressure")


def scenario(family,seed):
    rng=random.Random(seed)
    n=16 if family=="grid" else 20 if family=="ladder" else 24
    edges={(i,i+1):rng.uniform(1,5) for i in range(n-1)}
    if family=="grid":
        edges.update({(i,i+4):rng.uniform(1,5) for i in range(12)})
    elif family=="ladder":
        edges.update({(i,i+2):rng.uniform(1,5) for i in range(n-2)})
    else:
        edges.update({(i,j):rng.uniform(1,8) for i in range(n)
                      for j in range(i+2,n) if rng.random()<.12})
    rows=[]
    for tick in range(12):
        if tick%4==1:
            for key in rng.sample(sorted(edges),3):
                edges[key]*=rng.choice((.4,2.5))
        elif tick%4==2:
            removable=[e for e in sorted(edges) if e[1]-e[0]>1]
            if removable:
                del edges[rng.choice(removable)]
            new=[(i,j) for i in range(n) for j in range(i+2,n) if (i,j) not in edges]
            if new:
                edges[rng.choice(new)]=rng.uniform(1,5)
        # tick%4==3 and next tick%4==0 deliberately repeat topology/weights
        rows.append((n,tuple((u,v,w) for (u,v),w in sorted(edges.items())),0,n-1))
    return rows


def oracle(args,result,expected):
    n,edges,s,t=args
    assert_path(n,edges,s,t,result)
    if result.cost+1e-8<expected:
        raise AssertionError("path cost below exact oracle")
    return result.cost/expected


def execute(rows,expected,mode,budget,*,timed):
    mem=None if mode=="cold" else IncrementalFlowMemory(pressure=mode=="pressure")
    times,ratios,iterations,steps,direct,reused,hits,paths=[],[],[],[],[],[],[],[]
    for args,opt in zip(rows,expected):
        start=time.perf_counter_ns()
        if mem is None:
            result=solve(*args,iterations=budget)
            info=None
        else:
            result,info=mem.solve(*args,iterations=budget)
        elapsed=time.perf_counter_ns()-start
        if timed:
            times.append(elapsed)
        ratios.append(oracle(args,result,opt))
        paths.append(result.path)
        iterations.append(result.iterations)
        steps.append(0 if info is None else info.cg_steps)
        direct.append(result.iterations+1 if info is None else info.direct_solves)
        reused.append(0 if info is None else info.edge_reuses)
        hits.append(0 if info is None else info.exact_flow_hits)
    return dict(samples_ns=times,cost_ratios=ratios,iterations=iterations,
                cg_steps=steps,direct_solves=direct,edge_reuses=reused,
                exact_flow_hits=hits,paths=paths)


def benchmark(family,seed):
    rows=scenario(family,seed)
    expected=[bellman(*args) for args in rows]
    for args,ref in zip(rows,expected):
        if not math.isclose(exact_path(*args),ref):
            raise AssertionError("independent shortest-path controls disagree")
    results=[]
    for budget in BUDGETS:
        order=MODES[seed%3:]+MODES[:seed%3]
        for mode in order:
            # Three independent memory lifetimes on the same graph stream,
            # not independent workloads. Seeds/families are separate streams.
            runs=[execute(rows,expected,mode,budget,timed=True) for _ in range(3)]
            base=runs[0]
            for run in runs[1:]:
                if run["paths"]!=base["paths"]:
                    raise AssertionError("nondeterministic paths")
            tracemalloc.start()
            try:
                checked=execute(rows,expected,mode,budget,timed=False)
                peak=tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            if checked["paths"]!=base["paths"]:
                raise AssertionError("memory replay differs")
            times=[t for run in runs for t in run["samples_ns"]]
            ratios=base["cost_ratios"]
            results.append(dict(mode=mode,budget=budget,n=len(rows),
                optimal=sum(abs(x-1)<1e-9 for x in ratios),
                mean_cost_ratio=statistics.mean(ratios),worst_cost_ratio=max(ratios),
                median_ns=statistics.median(times),p95_ns=quantile(times,.95),
                p99_ns=quantile(times,.99),samples=len(times),
                peak_traced_bytes=peak,mean_iterations=statistics.mean(base["iterations"]),
                cg_steps=sum(base["cg_steps"]),direct_solves=sum(base["direct_solves"]),
                edge_reuses=sum(base["edge_reuses"]),exact_flow_hits=sum(base["exact_flow_hits"]),
                runs=runs))
    return dict(family=family,seed=seed,results=results)


class IncrementalFlowTests(unittest.TestCase):
    def test_sparse_pressure_against_direct(self):
        for family,seed in (("grid",81091),("ladder",81092),("random",81093)):
            for n,edges,s,t in scenario(family,seed):
                edges=graph(n,edges,s,t)
                d=tuple(.2+(i%5)*.15 for i in range(len(edges)))
                engine=_Pressure(n,edges,s,t)
                p,q,res=engine(n,edges,d,s,t)
                _,ref,_=_flow(n,edges,d,s,t)
                self.assertLessEqual(res,1e-6)
                self.assertLess(max(abs(a-b) for a,b in zip(q,ref)),1e-6)
                self.assertEqual(engine(n,edges,d,s,t),(p,q,res))
                self.assertEqual(engine.hits,1)

    def test_partial_topology_identity_and_atomic_failure(self):
        mem=IncrementalFlowMemory()
        edges=[(0,1,1),(1,3,1),(0,2,2),(2,3,2)]
        r,info=mem.solve(4,edges,0,3,iterations=8)
        self.assertEqual(info.edge_reuses,0)
        old=mem._state
        with self.assertRaises(ValueError):
            mem.solve(4,[(0,1,1)],0,3)
        self.assertIs(mem._state,old)
        changed=[(1,2,1),(2,3,2),(0,1,3),(1,3,1)]
        r,info=mem.solve(4,changed,0,3,iterations=4)
        self.assertEqual(info.edge_reuses,3)
        oracle((4,changed,0,3),r,bellman(4,changed,0,3))
        _,info=mem.solve(4,changed,3,0,iterations=2)
        self.assertEqual(info.edge_reuses,0)

    def test_warm_pressure_and_conductance_reuse(self):
        mem=IncrementalFlowMemory()
        edges=[(0,1,1),(1,3,1),(0,2,4),(2,3,4),(1,2,5)]
        for _ in range(2):
            r,info=mem.solve(4,edges,0,3,iterations=32)
            self.assertEqual(r.cost,2)
        self.assertEqual(info.edge_reuses,5)
        self.assertGreater(info.pressure_reuses,0)
        self.assertEqual(info.pressure_solves,r.iterations+1)

    def test_negative_parameters_and_terminal_reset(self):
        with self.assertRaises(ValueError):
            IncrementalFlowMemory(blend=0)
        with self.assertRaises(ValueError):
            IncrementalFlowMemory(blend=float("nan"))
        m=IncrementalFlowMemory()
        with self.assertRaises(ValueError):
            m.solve(2,[(0,1,1)],0,1,iterations=0)
        self.assertIsNone(m._state)

    def test_independent_budget_sweep(self):
        rows=[benchmark(family,seed) for family,seed in
              (("grid",91001),("ladder",91002),("random",91003))]
        report=dict(schema="alexpy-physarum-002/1",environment=platform.platform(),
            python=platform.python_version(),default_promotion=False,rows=rows,
            timing_scope="full solve per graph; p99 nearest rank, 36 observations per cell",
            memory_scope="tracemalloc peak of full 12-step replay, includes oracle allocations",
            scope="synthetic graph costs; no real execution or energy claim")
        Path("physarum002-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        for row in rows:
            for r in row["results"]:
                r.pop("runs")
        print("ALEXPY_PHYSARUM002 "+json.dumps(report,separators=(",",":")),flush=True)
