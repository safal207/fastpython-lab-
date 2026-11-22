# AlexPy 2.2 — Native Batch Kernel (mission 21)

**Research-only opt-in CPython C extension. Do not auto-promote.**

## Hypothesis

The QA Batch Engine from mission 20 improved on the per-event AlexPy graph,
but was slower than an independently optimized Python batch implementation.
Profiled costs included Python event validation, typed-buffer transition
objects, Python aggregate maintenance and repeated per-event work.

`NativeBatchQA` is a **special-purpose accumulator**, not a generic interpreter
or an optimizer for arbitrary Python. The extension keeps private C status,
duration and suite-index arrays, prevalidates the entire list or tuple of
`QAEvent` objects, collapses retries (last event wins), stages unbounded Python
integer totals, and commits only after successful validation and staging.
Updates touch only changed test IDs and affected suite/global counters.

The extension is optional. If it is not installed, `NativeBatchQA` delegates to
`StrongPythonBatchQA`; this fallback **must not** be counted as a native result.
The benchmark refuses to run without the compiled extension.

## Behavioral contract

- Input is an exact `list` or `tuple` of exact `QAEvent` objects. IDs, status,
  duration and suite labels must be exact `int` (not subclasses/bools).
- `duration_ms` is in `[0, 2**63 - 1]`; skipped events have zero duration.
- All events are checked, **including intermediate overwritten retries**.
  Any malformed input rejects the entire batch without changing the exposed
  tick, version, counters or last transition.
- Initial state and retries match `StrongPythonBatchQA`: batch tick increments
  even for empty/no-op batches, but version increments only if the final state
  of any test changes. Intermediate retry state is not observable.
- QA totals (including total duration across tests and suites) are
  **arbitrary-precision Python integers**. Signed-64 per-test values do not
  overflow the final sum.
- The C extension holds the GIL for each update. No arbitrary list mutation
  is intercepted; all updates must go through `apply_batch`.
- The public wrapper returns `QABatchTransition`, `QASnapshot` and supports
  independent `scan_reference` verification. Direct use of the private
  `_qa_native.QAState` is unsupported.
- This tests *batch-boundary equivalence*, not intermediate per-event alerts.

## Repeatable commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.qa_native_bench --smoke --sessions 2 --pairs 2 --batch-sizes 1,8,64 --json qa-native-smoke.json
python -m alexpy.qa_native_bench --sessions 3 --pairs 5 --batch-sizes 1,8,64,256 --json qa-native-full.json
```

Every of the 16 full scenarios compares 3 implementations:

1. `strong_batch`: independent optimized Python batch, **primary control**.
2. `alexpy_batch`: mission 20 typed OwnedQBuffer batch graph.
3. `native_batch`: mission 21 compact native C accumulator and Python wrapper.

Time includes new engine creation, slicing the pre-generated event stream into
batches, all validation and arithmetic, reporting snapshots and digest.
Synthetic fixture generation is outside the timed scope equally for all
algorithms. Each scenario is prevalidated against full independent scans at
nine distributed checkpoints and all three implementations are matched at
**every report boundary**. The same digest is checked after every timed run.
No data is skipped to obtain an apparent speedup.

The session source is deterministic synthetic QA events, **not client logs**.
Three fresh CPython processes and five rotated-order pairs per configuration
reduce, but do not eliminate, environmental and statistical uncertainty.
Results are ratios of *whole replay durations* for comparable workloads;
`median(per-configuration ratios)` is **not** aggregate throughput.

### Local Linux CPython 3.13.5 / AMD EPYC 9V74

| Metric across 16 configuration-level medians | Result |
|---|---:|
| Native / strong optimized Python batch | **1.802× median** |
| Native / old AlexPy batch | **2.833× median** |
| Native / strong Python range | **1.314–2.084×** |
| Cases with >10% improvement in all 3 processes vs optimized Python | **16/16** |
| Tests with the native extension compiled | **282/282 PASS** |

This is a concrete end-to-end improvement on this **synthetic** workload and
this machine. It does *not* imply faster arbitrary Python, generic QA pipelines,
production CI ingest, or other CPU families. The initial module builds in CI
on Ubuntu, Windows and macOS; the advisory benchmark must be checked for each
runner separately. No automatic candidate promotion.

## Limitations and next steps

- `QABatchTransition` materialization and the Python dataclass events still
  cost time. One possible follow-up is an explicit typed-input batch protocol
  to avoid per-event attribute access, *including encoding/packing costs*.
- Peak RSS and native heap allocation have **not** been quantified; Python
  `tracemalloc` does not cover all C-managed allocations. This experiment does
  not support a memory-efficiency claim.
- C-API allocation failures are staged before commit for observable state
  atomicity. Fatal interpreter termination cannot be recovered.
- Validate on actual, properly anonymized CI result streams from a consenting
  project before considering production integration.

## Files

- `alexpy/_qa_native.c` — special-purpose atomic native state
- `alexpy/qa_native.py` — public opt-in API with safe fallback
- `alexpy/qa_native_bench.py` — mirrored full-workflow paired benchmark
- `tests/test_qa_native.py` — input contracts, atomicity, bigint, randomized mirrors
- `tests/test_qa_native_bench.py` — deterministic digest and smoke checks
- `evidence/qa-native-results-local-py313.json` — original paired samples and session-level ratios
