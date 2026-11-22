"""Import a narrow, documented subset of public GitHub Actions job records.

This is a *real format* adapter, not an agent with GitHub credentials:
its input is an explicit JSON snapshot of the Actions jobs API.
Incomplete jobs are not reported as completed QA outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from .qa_lab import AlexPyQAGraph, QAEvent, PASS, FAIL, SKIP

RESULTS = {"success": PASS, "failure": FAIL, "timed_out": FAIL,
           "skipped": SKIP}
_MATRIX_RE = re.compile(r"^[^()]+\(([^,]+),\s*[^)]+\)$")


def _timestamp(iso: str) -> datetime:
    if type(iso) is not str:
        raise ValueError("job timestamp must be a string")
    date = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("job timestamp must be timezone aware")
    return date


def extract_jobs(payload: dict):
    """Return stable suite/job IDs and chronological completed QA events.

    A job name represents one logical test; repeated names are retries.
    Matrix-runner (first argument) becomes the suite, otherwise 'other'.
    Unsupported conclusions and in-progress jobs are explicitly omitted.
    Durations measure job wall clock, not individual unit-test CPU time.
    """
    if type(payload) is not dict or type(payload.get("jobs")) is not list:
        raise TypeError("expected GitHub Actions JSON with a jobs array")
    rows=[]
    for job in payload["jobs"]:
        if type(job) is not dict:
            raise TypeError("job entries must be objects")
        if job.get("status") != "completed" or job.get("conclusion") not in RESULTS:
            continue
        name=job.get("name")
        if type(name) is not str or not name.strip():
            raise ValueError("completed job must have a nonempty name")
        start,finish=_timestamp(job.get("started_at")),_timestamp(job.get("completed_at"))
        delta=finish-start
        milliseconds=((delta.days*86400 + delta.seconds)*1000
                      +delta.microseconds//1000)
        if milliseconds < 0:
            raise ValueError("job finish precedes start")
        match=_MATRIX_RE.match(name)
        suite=match.group(1).strip() if match else "other"
        rows.append((finish, name, suite, RESULTS[job["conclusion"]],
                     0 if job["conclusion"]=="skipped" else milliseconds))
    rows.sort(key=lambda r: (r[0],r[1]))
    suite_names=tuple(sorted({r[2] for r in rows}))
    test_names=tuple(sorted({r[1] for r in rows}))
    suite_ids={s:i for i,s in enumerate(suite_names)}
    test_ids={s:i for i,s in enumerate(test_names)}
    suite_of=[0]*len(test_names)
    for _,name,suite,_,_ in rows:
        suite_of[test_ids[name]]=suite_ids[suite]
    events=[QAEvent(test_ids[name],status,duration) for _,name,_,status,duration in rows]
    return suite_names,test_names,tuple(suite_of),tuple(events)


def summarize_jobs(payload: dict):
    suite_names,test_names,suite_of,events=extract_jobs(payload)
    if not suite_of:
        return {"jobs":0,"suites":{},"total":(0,0,0,0,0)}
    engine=AlexPyQAGraph(suite_of,len(suite_names))
    for event in events:
        engine.apply(event)
    if not engine.verify():
        raise AssertionError("QA jobs fail independent mirror")
    snapshot=engine.snapshot()
    return {"jobs":len(test_names),"events":len(events),
            "suites":{name:snapshot.suites[i] for i,name in enumerate(suite_names)},
            "total":snapshot.total,
            "duration_scope":"GitHub job wall clock (ms); latest completed run per name"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args(argv)
    report=summarize_jobs(json.loads(a.input.read_text(encoding="utf-8")))
    result=json.dumps(report,indent=2,ensure_ascii=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(result+"\n",encoding="utf-8")
    print(result)
    return 0


if __name__=="__main__": raise SystemExit(main())
