"""Isolated process tracemalloc audit of equal-native QA witness routes."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from .bardo_bridge_bench import host

ROUTES=('c_tree','set','python_tree')
CASES=('medium-sparse','large-sparse')


def audit(sessions=3,smoke=False):
    rows=[]
    for case in (CASES[:1] if smoke else CASES):
        for route in ROUTES:
            trials=[]
            for _ in range(sessions):
                cmd=[sys.executable,'-m','alexpy.native_tree_bench',
                     '--memory-worker',route,'--memory-case',case]
                p=subprocess.run(cmd,capture_output=True,text=True,check=True)
                trials.append(json.loads(p.stdout))
            peaks=[r['peak_traced_bytes'] for r in trials]
            digests=[r['digest'] for r in trials]
            if len(set(digests))!=1:raise AssertionError('replay digests differ')
            rows.append({'case':case,'route':route,'sessions':sessions,
                         'peak_traced_bytes':peaks,
                         'median_peak_traced_bytes':statistics.median(peaks),
                         'digest':digests[0]})
    return {'schema':'alexpy-bardo-native-memory/1','host':host(),'rows':rows,
            'note':'Peak traced Python allocations in isolated interpreters; not RSS or full C heap memory.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sessions',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--json',type=Path)
    args=p.parse_args()
    if args.sessions<1:p.error('sessions must be positive')
    r=audit(args.sessions,args.smoke)
    s=json.dumps(r,indent=2)
    if args.json:args.json.write_text(s+'\n',encoding='utf-8')
    print(s)
if __name__=='__main__':main()
