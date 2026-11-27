"""Frozen synthetic protocol: PYTHONPATH=. python tests/causal001_bench.py --output result.json."""
import argparse, gc, json, math, os, platform, random, statistics, subprocess, sys, time, tracemalloc, unittest
from pathlib import Path
from alexpy.causal_engine import Graph, Node, CausalEngine, MODES
from test_causal_engine import oracle, check_transition

CASES = ('narrow', 'stable_boundary', 'shared', 'dense_all_outputs')
SEEDS = (51901, 51902, 51903)
STEPS = 400

def fixture(case, seed):
    rng = random.Random(seed)
    width, depth = 32, 12
    inputs = 2*width
    nodes, tails = [], []
    for lane in range(width):
        p = inputs+len(nodes)
        parents = (2*lane,2*lane+1) if case != 'shared' or lane == 0 else (0,2*lane,2*lane+1)
        nodes.append(Node('sum', parents))
        for d in range(depth):
            nodes.append(Node('positive' if d == 0 else 'sum',(p,)))
            p += 1
        tails.append(p)
    nodes.append(Node('sum', tuple(tails[:4])))
    outputs = tuple(tails) if case == 'dense_all_outputs' else (tails[0], inputs+len(nodes)-1) if case == 'shared' else (tails[0],)
    g = Graph(inputs,tuple(nodes),outputs)
    initial = [rng.randrange(1,20) for _ in range(inputs)]
    current = initial.copy()
    patches = []
    for _ in range(STEPS):
        if case == 'stable_boundary':
            k = rng.randrange(1,7)
            patch={0:current[0]+k, 1:current[1]-k}
        else:
            chosen = rng.sample(range(inputs), inputs if case=='dense_all_outputs' else 4 if case=='shared' else 2)
            patch={i:rng.randrange(-20,21) for i in chosen}
        for i,v in patch.items(): current[i]=v
        patches.append(patch)
    return g,initial,patches

def quantiles(samples):
    order=sorted(samples)
    return dict(median_ns=statistics.median(order),p95_ns=order[math.ceil(.95*len(order))-1],p99_ns=order[math.ceil(.99*len(order))-1])

def worker(seed):
    result={}
    for case in CASES:
        graph,values,patches=fixture(case,seed)
        engines={m:CausalEngine(graph,values,m) for m in MODES}
        current=values.copy();totals={m:0 for m in MODES};test=unittest.TestCase();expected=[]
        before=oracle(graph,current)
        for patch in patches:
            for i,v in patch.items(): current[i]=v
            after=oracle(graph,current)
            expected.append(tuple(after[i] for i in graph.outputs))
            for m,engine in engines.items():
                event=engine.apply(patch)
                check_transition(test,graph,engine,before,after,event)
                totals[m]+=len(event.evaluated)
            before=after
        for m in MODES:
            e=CausalEngine(graph,values,m)
            for p in patches[:40]: e.apply(p);e.read()
        construction={m:[] for m in MODES}
        for i in range(9):
            for m in MODES[i%len(MODES):]+MODES[:i%len(MODES)]:
                start=time.perf_counter_ns();e=CausalEngine(graph,values,m);e.read()
                construction[m].append(time.perf_counter_ns()-start)
        engines={m:CausalEngine(graph,values,m) for m in MODES}
        samples={m:[] for m in MODES};gc.collect()
        for i,p in enumerate(patches):
            for m in MODES[i%len(MODES):]+MODES[:i%len(MODES)]:
                start=time.perf_counter_ns();engines[m].apply(p);actual=engines[m].read();elapsed=time.perf_counter_ns()-start
                if actual!=expected[i]: raise AssertionError('timed mismatch')
                samples[m].append(elapsed)
        rows={}
        for m in MODES:
            gc.collect();tracemalloc.start()
            try:
                e=CausalEngine(graph,values,m)
                for i,p in enumerate(patches):
                    e.apply(p)
                    if e.read()!=expected[i]: raise AssertionError('memory mismatch')
                peak=tracemalloc.get_traced_memory()[1]
            finally:tracemalloc.stop()
            rows[m]={**quantiles(samples[m]),'evaluations':totals[m],'live_nodes':len(e.live),'build_median_ns':statistics.median(construction[m]),'peak_traced_bytes':peak,'samples_ns':samples[m],'build_samples_ns':construction[m]}
        result[case]={'nodes':len(graph.nodes),'outputs':len(graph.outputs),'steps':len(patches),'routes':rows}
    return {'pid':os.getpid(),'seed':seed,'cases':result}

def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--output');args=p.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker)));return
    runs=[]
    for seed in SEEDS:
        proc=subprocess.run([sys.executable,__file__,'--worker',str(seed)],capture_output=True,text=True,check=True,timeout=120)
        runs.append(json.loads(proc.stdout))
    report={'schema':'causal001/1','environment':{'system':platform.platform(),'python':platform.python_version(),'processor':cpu_model(),'machine':platform.machine()},'protocol':{'timing':'apply + read + transition allocation; construction separate','memory':'tracemalloc peak for one engine construction and replay; shared graph and patches excluded; not RSS','samples_per_route_per_process':STEPS,'synthetic':True,'default_promotion':False},'runs':runs}
    if args.output:Path(args.output).write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
    for case in CASES:
        row={m:{'median_us':round(statistics.median(r['cases'][case]['routes'][m]['median_ns'] for r in runs)/1000,2),'p99_us':round(statistics.median(r['cases'][case]['routes'][m]['p99_ns'] for r in runs)/1000,2),'eval':sum(r['cases'][case]['routes'][m]['evaluations'] for r in runs),'peak_kib':round(statistics.median(r['cases'][case]['routes'][m]['peak_traced_bytes'] for r in runs)/1024,1)} for m in MODES}
        print(case,json.dumps(row))

if __name__=='__main__':main()
