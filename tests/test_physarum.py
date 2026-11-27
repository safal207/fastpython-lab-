"""PHYSARUM-001: exact path controls and independent seeded routing streams.

Five policies see the same complete current synthetic cost vector, no future
data. Ant/wave reuse actual AlexPy selectors; they optimize historical evidence,
not general graph shortest paths. On parallel complete plans, exact Dijkstra
reduces to min(cost). No real CPU-energy or execution-speedup claim.
"""
import heapq
import json
import math
import platform
import random
import time
import tracemalloc
import unittest
from pathlib import Path
from statistics import median

from alexpy.ant_memory import ROUTES, Workload, AntMemoryGraph
from alexpy.wave_memory import WaveMemoryGraph
from alexpy.physarum import solve, graph, propose, PhysarumMemory

POLICIES = ("dijkstra", "ant", "wave", "physarum", "physarum_matrix")


def exact_path(n, edges, source, sink):
    adj = [[] for _ in range(n)]
    for u,v,w in graph(n,edges,source,sink):
        adj[u].append((v,w))
        adj[v].append((u,w))
    best = {source:0.}
    queue = [(0.,source)]
    while queue:
        cost,u = heapq.heappop(queue)
        if cost != best[u]:
            continue
        if u == sink:
            return cost
        for v,w in adj[u]:
            c = cost+w
            if c < best.get(v,math.inf):
                best[v] = c
                heapq.heappush(queue,(c,v))
    raise AssertionError("unreachable sink")


def bellman(n, edges, source, sink):
    dist = [math.inf]*n
    dist[source] = 0.
    for _ in range(n-1):
        for a,b,w in edges:
            dist[a] = min(dist[a],dist[b]+w)
            dist[b] = min(dist[b],dist[a]+w)
    return dist[sink]


def assert_path(n, edges, source, sink, result):
    weights = {(min(u,v),max(u,v)):w for u,v,w in edges}
    assert result.path[0] == source and result.path[-1] == sink
    assert len(result.path) == len(set(result.path))
    total = sum(weights[min(a,b),max(a,b)] for a,b in zip(result.path,result.path[1:]))
    assert math.isclose(total,result.cost)
    assert result.residual <= 1e-6
    assert result.cost+1e-8 >= bellman(n,edges,source,sink)


def stream(seed):
    rng = random.Random(seed)
    cells = (Workload(128,32,1,2,4), Workload(512,64,4,2,4),
             Workload(2048,128,8,4,8))
    rows = []
    # Separate seeds from any earlier QA experiments. Change optimal route
    # every 30 steps; revisit cells; costs are synthetic normalized units.
    for tick in range(120):
        cell = cells[tick%3]
        regime = tick//30
        winner = (regime+tick%3)%len(ROUTES)
        costs = {r: (1. if i==winner else 1.6+.4*((i-winner)%4))
                 *rng.uniform(.96,1.04) for i,r in enumerate(ROUTES)}
        # All policies receive exactly the same available alternatives.
        if tick%11==7:
            costs.pop("dense")
        rows.append((cell,costs,regime))
    return rows


def policy(name):
    ant = AntMemoryGraph() if name=="ant" else None
    wave = WaveMemoryGraph(mode="light") if name=="wave" else None
    memory = PhysarumMemory(capacity=8) if name=="physarum_matrix" else None
    def choose(cell,costs):
        if name=="dijkstra":
            # For parallel complete-plan graph, exact shortest path = min.
            return min(costs,key=lambda r:(costs[r],ROUTES.index(r))),0
        if ant is not None or wave is not None:
            model = ant if ant is not None else wave
            model.advance()
            for r,c in costs.items():
                model.observe(cell,r,elapsed_ns=c,reference_ns=costs["manual"],verified=True)
            return model.choose(cell,allowed=tuple(costs)),0
        if memory is not None:
            route,result = memory.choose(cell,costs,iterations=32,verified=True)
        else:
            route,result = propose(costs,iterations=32)
        return route,result.iterations
    return choose


def execute(name, rows, *, timed):
    choose = policy(name)
    times, ratios, iterations, choices = [], [], [], []
    for cell,costs,regime in rows:
        t = time.perf_counter_ns()
        route,iters = choose(cell,costs)
        if timed:
            times.append(time.perf_counter_ns()-t)
        if route not in costs:
            raise AssertionError("disallowed route")
        ratios.append(costs[route]/min(costs.values()))
        iterations.append(iters)
        choices.append(route)
    return times,ratios,iterations,choices


def quantile(xs,p):
    return sorted(xs)[math.ceil(p*len(xs))-1]


def benchmark(seed):
    rows = stream(seed)
    results = {}
    names = POLICIES[seed%len(POLICIES):]+POLICIES[:seed%len(POLICIES)]
    for name in names:
        execute(name,rows[:6],timed=False)
        times,ratios,iters,choices = execute(name,rows,timed=True)
        tracemalloc.start()
        try:
            _,other,_,other_choices = execute(name,rows,timed=False)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        if choices!=other_choices or ratios!=other:
            raise AssertionError("nondeterministic route selection")
        delays = []
        for start in (30,60,90):
            # First three consecutive optimal decisions in new regime.
            end = next((i for i in range(start,start+28)
                        if all(v<=1+1e-10 for v in ratios[i:i+3])),None)
            delays.append(None if end is None else end-start+3)
        results[name] = dict(median_ns=median(times),p95_ns=quantile(times,.95),
                             p99_ns=quantile(times,.99),peak_traced_bytes=peak,
                             optimal=sum(r<=1+1e-10 for r in ratios),
                             decisions=len(rows),mean_cost_ratio=sum(ratios)/len(ratios),
                             adaptation_cycles=delays,mean_iterations=sum(iters)/len(iters),
                             samples_ns=times,cost_ratios=ratios,choices=choices)
    return dict(seed=seed,results=results)


class PhysarumTests(unittest.TestCase):
    def test_unique_short_path_and_conservation(self):
        edges = [(0,1,1),(1,3,1),(0,2,4),(2,3,4),(1,2,5)]
        r=solve(4,edges,0,3)
        assert_path(4,edges,0,3,r)
        self.assertEqual(r.path,(0,1,3))
        self.assertEqual(r.cost,exact_path(4,edges,0,3))
        self.assertTrue(r.converged)

    def test_random_connected_graphs_against_exact_oracle(self):
        rng=random.Random(55031)
        for n in (5,8,12):
            for _ in range(8):
                edges=[(i,i+1,rng.randint(1,9)) for i in range(n-1)]
                edges += [(i,j,rng.randint(1,12)) for i in range(n)
                          for j in range(i+2,n) if rng.random()<.25]
                self.assertEqual(exact_path(n,edges,0,n-1),bellman(n,edges,0,n-1))
                result=solve(n,edges,0,n-1,iterations=32)
                assert_path(n,edges,0,n-1,result)

    def test_negative_controls(self):
        for edges in ([(0,1,1)],[(0,1,0),(1,2,1)],
                      [(0,1,1),(1,0,1),(1,2,1)],
                      [(0,1,float("nan")),(1,2,1)]):
            with self.assertRaises(ValueError):
                solve(3,edges,0,2)
        with self.assertRaises(ValueError):
            solve(2,[(0,1,1)],0,1,initial=[0])
        with self.assertRaises(ValueError):
            propose({"unregistered":1})
        with self.assertRaises(ValueError):
            propose({"manual":-1})
        mem=PhysarumMemory(capacity=1)
        cell=Workload(128,32,1,2,4)
        with self.assertRaises(ValueError):
            mem.choose(cell,{"manual":1},verified=False)
        self.assertEqual(mem.cells,0)

    def test_memory_mask_eviction_and_reversal(self):
        mem=PhysarumMemory(capacity=1)
        cell=Workload(128,32,1,2,4)
        route,_=mem.choose(cell,{"manual":1,"dense":4},verified=True)
        self.assertEqual(route,"manual")
        route,_=mem.choose(cell,{"manual":4,"dense":1},verified=True)
        self.assertEqual(route,"dense")
        route,_=mem.choose(cell,{"manual":1},verified=True)
        self.assertEqual(route,"manual")
        self.assertEqual(mem.cells,1)

    def test_independent_stream_benchmark(self):
        rows=[benchmark(seed) for seed in (71001,71002,71003)]
        for row in rows:
            self.assertEqual(row["results"]["dijkstra"]["optimal"],120)
            for v in row["results"].values():
                self.assertEqual(v["decisions"],120)
                self.assertGreaterEqual(v["mean_cost_ratio"],1)
        report=dict(schema="alexpy-physarum/1",environment=platform.platform(),
                    python=platform.python_version(),rows=rows,
                    synthetic_costs=True,default_promotion=False,
                    scope="planning cost only; no actual execution or energy claim")
        Path("physarum-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        for row in rows:
            for v in row["results"].values():
                for k in ("samples_ns","cost_ratios","choices"):
                    v.pop(k)
        print("ALEXPY_PHYSARUM "+json.dumps(report,separators=(",",":")),flush=True)
