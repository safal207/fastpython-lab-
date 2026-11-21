"""Full-path list -> array('q') -> repeated use -> optional list benchmark.

Every call includes allocation, type validation and conversion.  No global
speedup claims or hidden automatic promotion.  AB/BA pairs and multiple fresh
processes are encouraged.  Reported break-even points are empirical, bounded
within measured repetition values, not extrapolated to all programs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from .amortization import reference_batch, packed_batch, repacked_batch
from .memory_candidates import _buffer_native

SIZES = (8, 4096, 32768)
REPETITIONS = (1, 2, 4, 8, 16, 32)
PROFILES = ('small', 'large')
CANDIDATES = ('pack_once', 'repack_each')


def sample_values(size: int, profile: str):
    if size < 0 or profile not in PROFILES:
        raise ValueError('invalid workload')
    if profile == 'small':
        return [(i % 17) - 8 for i in range(size)]
    return [(1 << 40) + (i % 7) for i in range(size)]


def mirrors(values, k: int, materialize: bool):
    # Two independent Python expressions for the scalar result.
    a = sum(x * x for x in values) * k
    b = sum(map(lambda x: pow(x, 2), values)) * k
    if type(a) is not int or a != b:
        raise AssertionError('Python mirrors disagree')
    return (a, list(values)) if materialize else a


def _measure(fn, loops: int):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn()
    return time.perf_counter_ns() - start


def paired(control, candidate, *, expected, pairs: int, target_seconds: float,
           epoch: int):
    if pairs < 1 or target_seconds <= 0 or not math.isfinite(target_seconds):
        raise ValueError('invalid timing controls')
    for label, fn in (('control', control), ('candidate', candidate)):
        if fn() != expected:
            raise AssertionError(f'{label} failed before measurement')
    for fn in (control, candidate):
        fn(); fn()
    calibration = max(1, _measure(control, 1))
    loops = max(1, min(2000, int(target_seconds * 1e9 / calibration)))
    points=[]
    for i in range(pairs):
        if (i+epoch) % 2:
            b = _measure(candidate, loops)
            a = _measure(control, loops)
            order='BA'
        else:
            a = _measure(control, loops)
            b = _measure(candidate, loops)
            order='AB'
        points.append({'order':order,'reference_ns':a/loops,
                       'candidate_ns':b/loops})
    baseline=statistics.median(x['reference_ns'] for x in points)
    trial=statistics.median(x['candidate_ns'] for x in points)
    return {'loops':loops,'pairs':points,'baseline_ns':baseline,
            'candidate_ns':trial,'speedup':baseline/trial if trial else None}


def measurements(*, sizes=SIZES, repetitions=REPETITIONS, profiles=PROFILES,
                 epochs=2, pairs=7, target_seconds=.002,
                 candidates=CANDIDATES):
    if _buffer_native is None:
        raise RuntimeError('native buffer extension required for meaningful experiment')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    sizes=tuple(sizes); repetitions=tuple(repetitions); profiles=tuple(profiles)
    candidates=tuple(candidates)
    if not sizes or not repetitions or not profiles or any(n<0 for n in sizes) or any(k<1 for k in repetitions):
        raise ValueError('invalid states')
    if not candidates or len(set(candidates)) != len(candidates) or any(name not in CANDIDATES for name in candidates):
        raise ValueError('unknown or duplicate candidate')
    states=[(n,p,k,ret) for n in sizes for p in profiles
            for k in repetitions for ret in (False, True)]
    out=[]
    for epoch in range(epochs):
        sequence=states if epoch % 2 == 0 else list(reversed(states))
        for n,profile,k,ret in sequence:
            values=sample_values(n,profile)
            expected=mirrors(values,k,ret)
            ref=lambda:reference_batch(values,k,materialize=ret)
            candidate_functions={
                'pack_once':lambda:packed_batch(values,k,materialize=ret),
                'repack_each':lambda:repacked_batch(values,k,materialize=ret),
            }
            observations={name:paired(ref,candidate_functions[name],expected=expected,pairs=pairs,
                      target_seconds=target_seconds,epoch=epoch)
                for name in candidates}
            out.append({'epoch':epoch,'size':n,'profile':profile,
                        'repetitions':k,'materialize':ret,'payload_bytes':n*8,
                        'observations':observations})
    return out


def summarize(cases):
    groups={}
    for case in cases:
        key=(case['size'],case['profile'],case['repetitions'],case['materialize'])
        for name,obs in case['observations'].items():
            groups.setdefault((*key,name),[]).append(obs['speedup'])
    out=[]
    for (size,profile,k,ret,name),ratios in sorted(groups.items()):
        if any(x is None or not math.isfinite(x) or x<=0 for x in ratios):
            raise ValueError('invalid speedup')
        out.append({'size':size,'profile':profile,'repetitions':k,
                    'materialize':ret,'candidate':name,'epochs':len(ratios),
                    'epoch_speedups':ratios,'median_speedup':statistics.median(ratios),
                    'repeated_gain_10pct':len(ratios)>=2 and all(x>1.1 for x in ratios),
                    'repeated_regression_10pct':len(ratios)>=2 and all(x<.9 for x in ratios)})
    return out


def empirical_break_even(summary, *, threshold=1.10):
    """First *measured* K above threshold, no extrapolated fitted crossing.

    Conservative monotonic condition: all greater *measured* K must also
    pass, so a single lucky isolated result cannot define break-even.
    """
    groups={}
    for row in summary:
        if row['candidate']!='pack_once':
            continue
        key=(row['size'],row['profile'],row['materialize'])
        groups.setdefault(key,[]).append(row)
    ans=[]
    for (n,p,ret),rows in sorted(groups.items()):
        rows.sort(key=lambda x:x['repetitions'])
        viable=None
        for i,row in enumerate(rows):
            if all(z['epochs']>=2 and all(s>threshold for s in z['epoch_speedups'])
                   for z in rows[i:]):
                viable=row['repetitions']
                break
        ans.append({'size':n,'profile':p,'materialize':ret,
                    'first_measured_repetitions_at_least_10pct_gain':viable,
                    'measured_repetitions':[r['repetitions'] for r in rows],
                    'scope':'observed K values only; not an analytical guarantee'})
    return ans


def report(**kwargs):
    cases=measurements(**kwargs)
    summary=summarize(cases)
    result={
        'schema_version':'alexpy.amortization.v1',
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':sys.version.split()[0], 'platform':platform.platform(),
                       'machine':platform.machine()},
        'comparison':'repeat K sum_squares on exact list vs pack q once or K times',
        'workload_invariants':'read-only exact list[int] in int64; packed values reused within call',
        'materialize_true':'both reference and candidate return a newly created list',
        'scope':'end-to-end per batch on the same input; fresh process validation needed',
        'promotion_gate':'BLOCKED_UNTIL_MULTIHOST_REPLICATION_AND_GUARDSET',
        'hardware_counters':'NOT_MEASURED',
        'cases':cases,'summary':summary,'empirical_break_even':empirical_break_even(summary),
    }
    raw=json.dumps(result,sort_keys=True,separators=(',',':'))
    result['sha256_without_digest']=hashlib.sha256(raw.encode()).hexdigest()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',default='8,4096,32768')
    parser.add_argument('--repetitions',default='1,2,4,8,16,32')
    parser.add_argument('--profiles',default='small,large')
    parser.add_argument('--candidates',default='pack_once,repack_each')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--pairs',type=int,default=7)
    parser.add_argument('--target-seconds',type=float,default=.002)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args(argv)
    try:
        outcome=report(sizes=tuple(map(int,args.sizes.split(','))),
            repetitions=tuple(map(int,args.repetitions.split(','))),
            profiles=tuple(args.profiles.split(',')),epochs=args.epochs,
            pairs=args.pairs,target_seconds=args.target_seconds,
            candidates=tuple(args.candidates.split(',')))
    except ValueError as e:
        parser.error(str(e))
    if args.json:
        args.json.write_text(json.dumps(outcome,indent=2)+'\n',encoding='utf-8')
    for row in outcome['empirical_break_even']:
        print(f"n={row['size']:6} {row['profile']:5} return_list={row['materialize']} "
              f"K_at_10pct={row['first_measured_repetitions_at_least_10pct_gain']}")
    print('promotion_gate:',outcome['promotion_gate'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
