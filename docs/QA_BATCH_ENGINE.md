# AlexPy 2.1 — QA Batch Engine (mission 20)

Status: **experimental, opt-in only**. No generic Python acceleration or default promotion.

## Problem and implementation

`AlexPyQAGraph.apply(QAEvent)` created a dictionary and an `OwnedQBuffer.Transition`
for every test-result update. This dominated simple O(1) counter updates.

`AlexPyBatchQAGraph.apply_batch(list_or_tuple)` now validates **all** events first,
coalesces retries of the same test ID (last event wins), then applies one
`OwnedQBuffer._commit_prepared` for the net duration changes and updates only
the affected suite and global counters. `StrongPythonBatchQA` is an independent,
strong **Python batch** baseline with the same coalescing and atomic validation.
Both expose `QABatchTransition`, including tick, version, changed IDs and affected suites.

An empty batch advances the batch tick but not the data version. A status-only
change advances QA version but not `OwnedQBuffer`'s duration-data version. A
batch whose **final** values equal the previous values does not advance version.

### Contract limitations

- A batch is **one atomic transaction**, not a drop-in substitute for per-event
  state/notification visibility. Intermediate retries are not observable;
  reports must occur at batch boundaries.
- Every event is validated, including overwritten intermediates. Invalid test ID,
  status, duration, skipped-test duration, subclassed integer, or unexpected event
  aborts the whole batch without modifying counters, buffer, tick or version.
- Source `QAEvent` duration must be exact int in `[0, 2**63-1]`; aggregate uses
  arbitrary precision Python int and cannot overflow.
- `_commit_prepared` is **private** and only safe because staging checks exact
  validated indices and typed values before calling it. This is a research-only
  coupling; keep future refactors covered by mirrored tests.
- We do not automatically monitor arbitrary mutable lists or CI providers.

## Independent evidence

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.qa_batch_bench --smoke --sessions 2 --pairs 2 --batch-sizes 1,8,64 --json qa-batch-smoke.json
python -m alexpy.qa_batch_bench --sessions 3 --pairs 3 --batch-sizes 1,8,64,256 --json qa-batch-results.json
```

Each of four synthetic workloads replays four routes:

1. `strong_event`: O(1) Python per event (previous best control)
2. `alexpy_event`: previous owned-buffer AlexPy per event
3. `strong_batch`: optimized Python coalescing in a single atomic batch (**primary control**)
4. `alexpy_batch`: new owned-buffer graph using a single batch commit

The timed scope includes construction, per-batch slicing, validation, all updates,
report snapshots and hashing. Source fixture generation is outside timing for all
routes. Checks run outside timing, but **fail closed** before measurements:
every report matches all four routes, and nine independent O(N) full scans are
spread over each sequence (including the final state). Full-stream report digest
is checked on every timed replay.

Full test: 4 workloads × 4 batch sizes = **16 configurations**; **3 fresh Python
processes**, **3 alternated route-order pairs** per configuration.
Measurements made on the local CPython 3.13.5 Linux runner; cannot be extrapolated
to different devices or customer workloads. Source is deterministic **synthetic**
QA events, not private logs. Repeated report snapshots reflect the same semantics
as the event-by-event paths *only at batch boundaries*.

### Results (ratios of total times, higher is better)

| Case | Batch 1: vs strong Python batch | Batch 64: vs strong Python batch | Batch 256: vs old AlexPy graph |
| --- | ---: | ---: | ---: |
| 256 tests, frequent reports | 0.631× | 0.633× | 3.711× |
| 4,096 tests, mixed retries | 0.596× | 0.644× | 2.360× |
| 16,384 tests, sparse reports | 0.660× | 0.641× | 2.244× |
| 4,096 tests, 96 active IDs | 0.611× | 0.673× | 4.672× |

Across the **16 configuration-level medians**, the median speedup of the new
AlexPy batch path over optimized Python batch is **0.632×** (regression);
its median speedup over the prior AlexPy per-event path is **2.006×**.
For all 12 configurations with batch sizes 8, 64 or 256, the new path beat old
AlexPy by >10% in **all three processes**. Against optimized Python batch,
it lost by >10% on **all 16 configurations in all three processes**.

Never cherry-pick the ratio to the older AlexPy graph as evidence of speedup
beyond well-written Python. **No auto-promotion.**

### Profiling

A cProfile spot check (4,096 tests, 8,000 events, 96 hot IDs, batches of 256)
shows substantial time in `_prepare` + `_validate` (shared by both batch paths),
with additional AlexPy time in `OwnedQBuffer._commit_prepared`, squared-sum
maintenance, sorting, and initialization. `tracemalloc` diagnostic values are
recorded separately in `evidence/qa-batch-memory-local-py313.json`, but do not
represent full native/RSS memory usage; avoid definitive comparisons from its
single noisy sample.

## Next hypothesis

Try a **specialized native batch kernel** with a flat typed duration/status
representation that performs validated O(k) updates and only QA-required
aggregates; compare end-to-end including packing/conversion and maintain a
strong Python batch baseline. Profile before moving code into C. A pure C
microkernel speedup without end-to-end win is not sufficient for promotion.

## Evidence

- `evidence/qa-batch-results-local-py313.json` — all raw pairs, three processes
- `evidence/qa-batch-memory-local-py313.json` — limited diagnostic
- `tests/test_qa_batch.py` — atomicity, duplicate IDs, invalid intermediate,
  unsigned-ish boundaries, random mirrors and status-only changes
- `tests/test_qa_batch_bench.py` — deterministic workloads and fair digests
