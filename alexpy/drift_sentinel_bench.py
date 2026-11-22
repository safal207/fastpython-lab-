"""Frozen train/heldout probe-cadence study with hidden route changes.

This measures synthetic decision-regret units, *not* wall-clock acceleration.
The controller API receives one current observation only on a paid probe.
Controls are selected on training seeds, heldout remains unavailable until
policies and parameters are frozen. No decisions depend on regime labels.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from .drift_sentinel import ProbeCosts,DriftSentinel,BardoTrustControl

TICKS=1536
KINDS=("stable","abrupt","reversal","bursts","mixed")
TRAIN_SEEDS=tuple(41001+113*i for i in range(12))
HOLDOUT_SEEDS=tuple(91001+173*i for i in range(24))
FIXED_INTERVALS=(8,16,32,64,128)
COSTS=ProbeCosts()


def hidden_states(seed:int, kind:str, ticks=TICKS):
    """Evaluation-only hidden environment, never passed to a controller."""
    if kind not in KINDS or type(seed) is not int or ticks<200:
        raise ValueError("invalid hidden-stream parameters")
    r=random.Random(seed+sum((i+1)*ord(c) for i,c in enumerate(kind)))
    s=[True]*ticks
    if kind=="stable":
        return tuple(s)
    if kind=="abrupt":
        switch=r.randrange(ticks//4,ticks*3//4)
        return tuple(t<switch for t in range(ticks))
    if kind=="reversal":
        begin=r.randrange(ticks//4,ticks//2)
        end=r.randrange(ticks*3//4,ticks*7//8)
        return tuple(not (begin<=t<end) for t in range(ticks))
    if kind=="bursts":
        for _ in range(10):
            start=r.randrange(ticks-55);length=r.randrange(3,55)
            for i in range(start,min(ticks,start+length)):s[i]=False
        return tuple(s)
    active=True
    for t in range(ticks):
        q=(.0006 if t<ticks//3 else .03 if t<2*ticks//3 else .008)
        if r.random()<q:active=not active
        s[t]=active
    return tuple(s)


def evaluate(policy:str,series:tuple[bool,...],costs=COSTS):
    """Only this evaluator may read hidden truth between paid observations."""
    last=False;due=0;probes=0;cost=0.;stale=0;missed=0;lag=[];flip_at=None
    controller=(DriftSentinel(costs) if policy=="sentinel" else
                BardoTrustControl(costs) if policy=="bardo_trust" else None)
    fixed=None if controller or policy not in (f"fixed{n}" for n in FIXED_INTERVALS) else int(policy[5:])
    if controller is None and fixed is None:raise ValueError("invalid policy")
    for t,truth in enumerate(series):
        if t and truth!=series[t-1]:flip_at=t
        if t>=due:
            if controller:
                step=controller.observe(t,truth)
            else:step=fixed
            last=truth;due=t+step;probes+=1;cost+=costs.probe
            if flip_at is not None:
                lag.append(t-flip_at);flip_at=None
        chosen=(controller.use_fast(t) if isinstance(controller,BardoTrustControl) else last)
        if chosen and not truth:cost+=costs.stale;stale+=1
        elif not chosen and truth:cost+=costs.missed;missed+=1
    return {"cost":cost,"probes":probes,"stale_ticks":stale,"missed_ticks":missed,
            "mean_detect_lag":statistics.mean(lag) if lag else None}


def collect(seeds,policies):
    return [{"seed":seed,"kind":kind,"results":{p:evaluate(p,hidden_states(seed,kind)) for p in policies}}
            for seed in seeds for kind in KINDS]


def train_select():
    policies=tuple(f"fixed{i}" for i in FIXED_INTERVALS)
    rows=collect(TRAIN_SEEDS,policies)
    mean_costs={p:statistics.mean(row["results"][p]["cost"] for row in rows) for p in policies}
    return min(policies,key=lambda x:(mean_costs[x],x)),mean_costs


def run(smoke=False):
    selected,train_costs=train_select()
    policies=(selected,"sentinel","bardo_trust","fixed8","fixed128")
    seeds=HOLDOUT_SEEDS[:4] if smoke else HOLDOUT_SEEDS
    rows=collect(seeds,policies)
    summary={}
    for p in policies:
        costs=[r["results"][p]["cost"] for r in rows]
        ratios=[r["results"][p]["cost"]/max(1,r["results"][selected]["cost"]) for r in rows]
        summary[p]={"mean_cost":statistics.mean(costs),"median_cost":statistics.median(costs),
                    "median_ratio_vs_train_fixed":statistics.median(ratios),
                    "win_fraction_vs_train_fixed":sum(a<b for a,b in zip(costs,(r["results"][selected]["cost"] for r in rows)))/len(rows),
                    "p90_ratio":sorted(ratios)[int(.9*(len(ratios)-1))]}
    per_kind={kind:{p:statistics.mean(r["results"][p]["cost"] for r in rows if r["kind"]==kind)
                    for p in policies} for kind in KINDS}
    return {"schema":"alexpy-drift-sentinel/1","smoke":smoke,"ticks":TICKS,
            "train_seeds":list(TRAIN_SEEDS),"holdout_seeds":list(seeds),
            "frozen_selected_fixed":selected,"train_mean_costs":train_costs,
            "policies":list(policies),"summary":summary,"per_kind_mean_cost":per_kind,
            "rows":rows,
            "limit":"Synthetic economic decision units; no CPU acceleration claims. Heldout seeds not used for tuning."}


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--json",type=Path);args=ap.parse_args(argv)
    data=json.dumps(run(args.smoke),ensure_ascii=False,indent=2)
    if args.json:args.json.write_text(data+"\n",encoding="utf-8")
    print(data)
if __name__=="__main__":main()
