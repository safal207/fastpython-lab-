# AlexPy 2.0 research track: QA Performance Lab (mission 19)

**Status: correctness PASS, performance promotion BLOCKED.** This is an
experimental application-oriented harness for QA/CI event aggregation, not a
replacement for existing Python counters and not a claim of a 2.0 release.

## Workload and boundaries

An event updates the *latest outcome* of one test: `PASS`, `FAIL` or `SKIP`,
plus its last duration in milliseconds. Repeated test IDs model CI reruns.
The graph is `test ID -> suite -> global`. Every registered test belongs to
exactly one suite. Querying a dashboard materializes the full immutable
per-suite summary `(not_run, pass, fail, skip, duration_ms)`.

`StrongPythonQA` is a **strong O(1)-per-event** Python control using plain
lists and incremental suite/global counters. `AlexPyQAGraph` uses the existing
`OwnedQBuffer` for signed 64-bit durations and explicitly updates only the
owning suite and global counters. `scan_reference` independently replays a
fresh O(N) scan and checks every dashboard. The timed path includes state
construction, updates, dashboard snapshots and identical digest operations.

The synthetic event generator is deterministic; it models test reruns and
suite aggregation, **not** a measured production CI workload. Distribution:
76% pass, 18% fail, 6% skip, random durations and rerun IDs. Four scenarios
change suite count, report frequency, and number of events. `QAEvent`
requires exact types and valid duration; invalid events are rejected before
mutation, and external mutation of the input suite list cannot affect state.

The reference importer `alexpy.qa_actions` accepts a narrow JSON snapshot of
**real GitHub Actions jobs** (`status`, `conclusion`, ISO timestamps, and job
name). Incomplete or unsupported jobs are omitted. A completed job name is
a logical test ID, so repeated names model reruns and only the last completed
outcome remains in the dashboard. Duration is **job wall clock**, not CPU time
or time of individual Python unit tests. The checked-in sanitized fixture
contains six real jobs from AlexPy's successful PR workflow run #45039994988.
A six-job example validates parsing semantics; **it is far too small to claim
throughput improvement**.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.qa_actions --input evidence/qa-actions-jobs-sample.json \
  --output evidence/qa-actions-sample-summary.json
python -m alexpy.qa_bench --sessions 3 --pairs 5 \
  --json evidence/qa-lab-local-py313.json
# Quick CI check, advisory only:
python -m alexpy.qa_bench --smoke --sessions 2 --pairs 2 --json qa-lab-ci.json
```

The benchmark uses independent child interpreter processes, alternates AB/BA
order per paired observation, checks both candidates against a full-scan
witness **before** timing, then checks the output digest on each timed run.
No optimization is automatically enabled; the API remains opt-in.

## Local results (CPython 3.13.5, Linux, three child processes)

Each reported speedup is the median of three independent **per-process**
medians of five paired times. Values below 1 mean graph is slower than the
strong Python control. Times include all updates and materialized reports.

| Scenario | Tests | Events | Reports every | Python median ms | Graph median ms | Graph / Python speedup |
|---|---:|---:|---:|---:|---:|---:|
| small frequent | 256 | 1200 | 1 (3 reads) | 4.96 | 7.72 | **0.642x** |
| medium mixed | 4096 | 6000 | 16 (4 reads) | 7.79 | 21.74 | **0.362x** |
| large infrequent | 16384 | 12000 | 128 (8 reads) | 15.35 | 43.63 | **0.351x** |
| medium every event | 4096 | 4000 | 1 (1 read) | 13.17 | 22.77 | **0.571x** |

All four cases were slower by more than 10% in **each** independent process.
The raw paired times, independent-process medians and `tracemalloc` peaks are
in `evidence/qa-lab-local-py313.json`. `tracemalloc` reports traced Python
allocations only, **not total RSS or system memory use**. Its median peaks on
the large case were ~412 KiB for plain Python and ~646 KiB for the graph.
This is a targeted workload, not a general claim about Python speed.

A separate, **untimed profiler** of 6000 medium-scenario events attributed
roughly half the graph's cumulative time to `OwnedQBuffer.apply_patch` and
`_commit_prepared`. A fresh transition object, a sorted dirty-index tuple,
and repeated validation are necessary for the general buffer's published
contract, but costly when dispatched one event at a time. This is a plausible
optimization target, not evidence to bypass validation unsafely.

## Decision, QA risks and next experiment

**Do not promote.** A specialized batch API or a C loop should be considered
only if it preserves exact status counts, duration totals, replay ordering,
all-or-nothing validation on invalid batches, and the ability to independently
mirror results. Next experiment: compare batch processing fairly against
**both** an equivalently batched strong Python control and the current
per-event graph; account for queueing latency, ingestion time and reporting.

Not evaluated: production GitHub API rate limits, network fetch latency,
streaming from a CI provider, concurrency, multi-process state sharing,
newly discovered test IDs, non-integer durations, or measurements on a real
customer's workloads. No background monitoring or auto-import is performed.
