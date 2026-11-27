"""Frozen synthetic dynamic-demand study; no timing acceptance gate."""
import argparse, gc, json, math, os, platform, random, statistics, subprocess, sys, time, tracemalloc
from pathlib import Path
from alexpy.causal_engine import Graph, Node
from alexpy.dynamic_causal import DynamicCausalEngine, ROUTES
from test_dynamic_causal import oracle, validate

CASES=('return_stable','return_stale','rare_switch','shared_overlap','dense_switch')
SEEDS=(62901,62902,62903)
STEPS=300

def fixture(case,seed):
    rng=random.Random(seed);width=24;n=2*width;nodes=[];tails=[]
    for lane in range(width):
        parents=(2*lane,2*lane+1) if case!='shared_overlap' or lane==0 else (0,2*lane,2*lane+1)
        nodes.append(Node('sum',parents));p=n+len(nodes)-1
        for d in range(10):
            nodes.append(Node('positive' if d==0 else 'sum',(p,)));p+=1
        tails.append(p)
    g=Graph(n,tuple(nodes),(tails[0],));initial=[rng.randrange(1,10) for _ in range(n)];v=initial.copy();ops=[]
    for step in range(STEPS):
        lane=(step//(30 if case=='rare_switch' else 3))%width
        if case=='return_stable':
            lane=(step//3)%4; ids=[2*lane,2*lane+1];delta=rng.randrange(1,6);p={ids[0]:v[ids[0]]+delta,ids[1]:v[ids[1]]-delta}
        elif case=='dense_switch':
            p={i:rng.randrange(-10,11) for i in range(n)}
        else:
            ids=rng.sample(range(n),4);p={i:rng.randrange(-10,11) for i in ids}
        for i,x in p.items():v[i]=x
        o=tuple(tails[(lane+j)%width] for j in range(18 if case=='dense_switch' else 5 if case=='shared_overlap' else 1))
        ops.append((p,o))
    return g,initial,ops

def q(samples):
    a=sorted(samples);return {'n':len(a),'median_ns':statistics.median(a),'p95_ns':a[math.ceil(.95*len(a))-1],'p99_ns':a[math.ceil(.99*len(a))-1]}

def step(e,operation):
    p,o=operation;a=e.apply(p);b=e.select(o);return a,b,e.read()

def worker(seed):
    cases={}
    for case in CASES:
        g,v,ops=fixture(case,seed);expected=[];current=v.copy();counts={m:dict(evaluated=0,reused=0,invalidated=0,equal=0) for m in ROUTES};engines={m:DynamicCausalEngine(g,v,m) for m in ROUTES}
        for p,o in ops:
            for i,x in p.items():current[i]=x
            full=oracle(g,current);expected.append(tuple(full[i] for i in o))
            for m,e in engines.items():
                old=e._signatures.copy();a,b,actual=step(e,(p,o));validate(e,g,current,o)
                if actual!=expected[-1]:raise AssertionError('oracle mismatch')
                # Reuse at demand switch uses exact cached parent values.
                if m=='causal':
                    for i in b.reused:
                        if old[i]!=tuple(full[p] for p in g.nodes[i-g.inputs].parents):raise AssertionError('invalid reuse')
                for t in (a,b):
                    counts[m]['evaluated']+=len(t.evaluated);counts[m]['reused']+=len(t.reused);counts[m]['invalidated']+=t.invalidated;counts[m]['equal']+=len(t.equal_boundary)
        for m in ROUTES:
            e=DynamicCausalEngine(g,v,m)
            for op in ops[:30]:step(e,op)
        builds={m:[] for m in ROUTES}
        for i in range(8):
            for m in ROUTES[i%4:]+ROUTES[:i%4]:
                t=time.perf_counter_ns();e=DynamicCausalEngine(g,v,m);e.read();builds[m].append(time.perf_counter_ns()-t)
        engines={m:DynamicCausalEngine(g,v,m) for m in ROUTES};samples={m:[] for m in ROUTES};switch=[o!=(g.outputs if i==0 else ops[i-1][1]) for i,(_,o) in enumerate(ops)];gc.collect()
        for i,op in enumerate(ops):
            for m in ROUTES[i%4:]+ROUTES[:i%4]:
                t=time.perf_counter_ns();a,b,actual=step(engines[m],op);samples[m].append(time.perf_counter_ns()-t)
                if actual!=expected[i]:raise AssertionError('timed mismatch')
        rows={}
        for m in ROUTES:
            gc.collect();tracemalloc.start()
            try:
                e=DynamicCausalEngine(g,v,m)
                for i,op in enumerate(ops):
                    if step(e,op)[2]!=expected[i]:raise AssertionError('memory mismatch')
                peak=tracemalloc.get_traced_memory()[1]
            finally:tracemalloc.stop()
            rows[m]={**counts[m],**q(samples[m]),'switch':q([s for s,k in zip(samples[m],switch) if k]),'peak_traced_bytes':peak,'cached_nodes':e.cached_nodes,'build_median_ns':statistics.median(builds[m]),'samples_ns':samples[m],'build_samples_ns':builds[m]}
        cases[case]={'nodes':len(g.nodes),'steps':len(ops),'switches':sum(switch),'switch_flags':switch,'routes':rows}
    return {'pid':os.getpid(),'seed':seed,'cases':cases}

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--output');a=p.parse_args()
    if a.worker is not None:print(json.dumps(worker(a.worker)));return
    runs=[json.loads(subprocess.run([sys.executable,__file__,'--worker',str(s)],capture_output=True,text=True,check=True,timeout=120).stdout) for s in SEEDS]
    try:cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration):cpu=platform.processor()
    report={'schema':'causal002/1','environment':{'system':platform.platform(),'python':platform.python_version(),'processor':cpu},'protocol':{'timing':'apply + select + read + both traces, build separate','memory':'tracemalloc peak, shared graph and operations excluded, not RSS','synthetic':True,'default_promotion':False},'runs':runs}
    if a.output:Path(a.output).write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
    for c in CASES:
        print(c,json.dumps({m:{'median_us':round(statistics.median(r['cases'][c]['routes'][m]['median_ns'] for r in runs)/1000,3),'p99_us':round(statistics.median(r['cases'][c]['routes'][m]['p99_ns'] for r in runs)/1000,3),'eval':sum(r['cases'][c]['routes'][m]['evaluated'] for r in runs),'reuse':sum(r['cases'][c]['routes'][m]['reused'] for r in runs),'peak_kib':round(statistics.median(r['cases'][c]['routes'][m]['peak_traced_bytes'] for r in runs)/1024,2)} for m in ROUTES}))
if __name__=='__main__':main()
