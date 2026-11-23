# AlexPy × Bardo 2.7 — native witness density and failure stress

**Status: research only.** Frozen comparison based on AlexPy v2.6, initial
head `5188804940ef45a44144eb53adbd83f41a15aaeb`. No modification to the
native QA accumulator, main route, or drift calibration policy.

## Question

Does the v2.6 exact C segment-tree witness index still beat a strong Python
failure set on previously unmeasured **failure-heavy**, **dense**, **churn**,
**zero-failure**, and **frequent-diagnostic** workloads?

## Strong comparator and scope

Both routes use identical `NativeBatchQA` C accumulator, suite layout and
identical deterministic `QAEvent` lists. Only their *witness indices* differ:

- `c_tree`: opt-in `CTreeNativeQA` and exact compact C segment tree;
- `set`: `SetNativeQA` and Python `set` with `heapq.nsmallest(K)`.

Each reported time includes construction, batch slicing, validation, native
QA mutation, index update, snapshot creation and witness extraction. The
independent O(N) source-of-truth scan runs **outside timing** at checkpoints,
not just after the last batch. Changes to QA transition counts, epoch or
witness ordering are not permitted. Each route's report digest must match.

## Workloads (seed `312080191`)

| Case | Tests | Events | Batch | Approx. FAIL share | Reporting |
|---|---:|---:|---:|---:|---|
| dense-mostly-fail | 4096 | 8192 | 512 | 90% | every 4 batches |
| dense-wide | 16384 | 12288 | 2048 | 85% | every 4 batches |
| reversible-flips | 4096 | 8192 | 512 | alternating states | every 2 batches |
| hot-retries | 4096 | 8192 | 64 | 55%, 64 active tests | every 4 batches |
| frequent-diagnostics | 4096 | 6000 | 64 | 55% | every batch × 8 reads |
| zero-failures | 4096 | 6000 | 64 | 0% | every 4 batches |
| all-failures | 4096 | 8192 | 512 | 100% | every 4 batches |

The failures are **synthetic**, not observed customer incidents. The holdout
here is by workload shape: no claim of an untouched statistical holdout after
inspecting these outcomes. Independent processes reduce timing instability but
are not independently sampled hardware.

## Local environment and measurements

Linux CPython 3.13.5, AMD EPYC 9V74. Three separate interpreter processes,
three alternating-order paired measurements per process. Medians of per-process
speedup (`set elapsed / C-tree elapsed`):

| Case | Median speedup |
|---|---:|
| dense-mostly-fail | 1.266× |
| dense-wide | 1.240× |
| reversible-flips | 1.275× |
| hot-retries | 1.218× |
| frequent-diagnostics | 4.363× |
| zero-failures | 1.210× |
| all-failures | 1.256× |

All seven cases were faster in **all three** local processes. This does not
establish the same ordering on other CPUs or operating systems; CI now runs
separate small stress probes on Ubuntu, macOS and Windows.

### Isolated traced memory

Three isolated processes per route and case; peak Python allocator tracing,
not total process RSS, and not CPU timing under a profiler:

| Case | C-tree (median KiB) | Python set (median KiB) | C/set ratio |
|---|---:|---:|---:|
| dense-wide | 893.6 | 1374.6 | 0.650 |
| frequent-diagnostics | 151.7 | 301.3 | 0.504 |
| zero-failures | 151.7 | 147.3 | 1.030 |

The zero-failure case is a **negative memory result**: the C tree still owns a
fixed array even if no test fails; the Python set does not. A relative speed
win is not a memory win on every workload.

## Reproduce

```sh
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.native_witness_stress --sessions 3 --pairs 3 \
  --json evidence/native-witness-stress-local-py313.json
python -m alexpy.native_witness_stress --smoke --sessions 2 --pairs 2 \
  --json native-witness-stress-smoke.json
python -m alexpy.native_witness_stress --memory-route c_tree --memory-case dense-wide
python -m alexpy.native_witness_stress --memory-route set --memory-case dense-wide
```

Archive `evidence/native-witness-stress-*-local-py313.json` retains raw timing,
process host metadata, traces and SHA-256 provenance in the summary. Do not
interpret CI green checks alone as cross-host performance superiority.

## Decision and next falsification

Preserve the native witness as **opt-in**. The stress test gives further support
for the current segment-tree implementation on this AMD host, not for enabling
it by default. Next test must use new data and actual QA-report load, include
full peak RSS, and evaluate a direct O(1) first-K tracking alternative with
comparably strong correctness gates. The current Bardo calibration-drift policy
is independent and remains unpromoted after its previous negative result.
