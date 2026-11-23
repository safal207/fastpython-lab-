"""Deterministic, read-only JUnit XML replay into three experimental QA indices.

A real pytest-generated JUnit XML file can be read without credentials or
external network. This adapter does not change NativeBatchQA or fabricate test
failures. Report parsing and oracle construction are outside timed kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from .qa_lab import QAEvent, PASS, FAIL, SKIP, NOT_RUN, scan_reference
from .native_bitset import BitsetNativeQA
from .native_transition_tree import CTreeNativeQA
from .transition_tree import SetNativeQA
from .bardo_bridge_bench import digest_step


def parse_junit(path: Path, *, max_bytes=8_000_000):
    """Return deterministic (suite layout, event log, metadata); no test names logged.

    Key test identity is (classname, name), and a reappearing key becomes an
    ordinary retry. Status priority is error/failure > skipped > passed.
    """
    data=Path(path).read_bytes()
    if not data or len(data)>max_bytes:
        raise ValueError("JUnit report is empty or exceeds size limit")
    root=ET.fromstring(data)
    if root.tag not in ("testsuite","testsuites"):
        raise ValueError("expected JUnit testsuite or testsuites root")
    entries=list(root.iter("testcase"))
    if not entries:
        raise ValueError("JUnit report contains no testcases")
    ids={}
    suites={}
    layout=[]
    events=[]
    for item in entries:
        classname=item.get("classname", "")
        name=item.get("name")
        if not name:
            raise ValueError("JUnit testcase missing name")
        key=(classname,name)
        if key not in ids:
            ids[key]=len(ids)
            if classname not in suites:suites[classname]=len(suites)
            layout.append(suites[classname])
        duration=item.get("time","0")
        try:
            sec=float(duration)
        except ValueError as err:
            raise ValueError("invalid testcase time") from err
        if not (0.0<=sec<=(2**63-1)/1000):
            raise ValueError("test duration out of bounds")
        status=FAIL if any(child.tag in ("failure","error") for child in item) else (
            SKIP if any(child.tag=="skipped" for child in item) else PASS)
        ms=0 if status==SKIP else int(round(sec*1000))
        events.append(QAEvent(ids[key],status,ms))
    return layout,events,{"testcases":len(events),"unique_tests":len(ids),
        "suite_count":len(suites),"source_sha256":hashlib.sha256(data).hexdigest(),
        "retried_tests":len(events)-len(ids)}


def compare(path, *, batch=64):
    if type(batch) is not int or batch<=0:raise ValueError("batch must be positive")
    suites,events,meta=parse_junit(Path(path))
    count=meta["suite_count"]
    routes={"bitset":BitsetNativeQA,"c_tree":CTreeNativeQA,"set":SetNativeQA}
    engines={name:cls(suites,count) for name,cls in routes.items()}
    statuses=[NOT_RUN]*len(suites)
    durations=[0]*len(suites)
    digests={name:0 for name in routes}
    for pos in range(0,len(events),batch):
        group=events[pos:pos+batch]
        for event in group:
            statuses[event.test_id],durations[event.test_id]=event.status,event.duration_ms
        transitions={name:eng.apply_batch(group) for name,eng in engines.items()}
        if len(set(transitions.values()))!=1:raise AssertionError("transition mismatch")
        expected=scan_reference(suites,count,statuses,durations)
        witnesses=tuple(i for i,st in enumerate(statuses) if st==FAIL)[:4]
        for name,eng in engines.items():
            if eng.snapshot()!=expected or eng.witnesses()!=witnesses or not eng.verify():
                raise AssertionError(f"JUnit oracle mismatch: {name}")
            digests[name]=digest_step(digests[name],(eng.snapshot(),eng.witnesses()))
    if len(set(digests.values()))!=1:raise AssertionError("JUnit report digests differ")
    result={"schema":"alexpy-bardo-junit-replay/1",**meta,
            "batches":(len(events)+batch-1)//batch,
            "final_counts":list(scan_reference(suites,count,statuses,durations).total),
            "failure_witness_ids":list(witnesses),
            "digest":next(iter(digests.values())),
            "all_routes_match":True,
            "note":"Report reflects actual pytest execution; not a timed production benchmark"}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--json",type=Path)
    a=p.parse_args(argv)
    res=compare(a.report,batch=a.batch)
    value=json.dumps(res,indent=2)
    if a.json:a.json.write_text(value+"\n",encoding="utf-8")
    print(value)


if __name__=="__main__":main()
