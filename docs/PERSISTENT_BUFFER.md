# AlexPy v0.9 — long-lived owned buffers, sparse transitions, mirrors

## Scope

This research prototype studies *incremental computation*, not interpreter
replacement or physical CPU improvement. It is inspired by COSMIC's dirty
frontier, Bardo's transition representation, CaPU's fail-closed state gate,
and ContractGraph-QA's independent replay, but imports no code from them.

`OwnedQBuffer` accepts **only** exact `list`/`tuple` containers whose elements
are exact signed 64-bit `int`s. It copies and owns an `array('q')`. The source
may be mutated later without affecting the owned state. Only `apply_patch`
mutates the owned buffer. There is deliberately no writable view or direct
exposure of internal storage through the public API; Python's private
attribute convention is not a hard access-control boundary.

## State graph and update equation

A logical state has coordinates `(size, modified_positions, reads, tick,
version)` and edges from each changed `data[index]` to cached `sum` and
`sum_squares`, and from those to `variance_numerator`.

For one patch `P = {(i, new_value)}`:

- `delta_sum = Σ (new_value - old_value)`
- `delta_sum_squares = Σ (new_value² - old_value²)`
- `sum' = sum + delta_sum`, `sum_squares' = sum_squares + delta_sum_squares`
- `variance_numerator = N * sum_squares - sum²`

Python `int` keeps these cached totals exact even when they exceed 64 bits.
Each patch validates all indices and values **before** mutation, rejecting
bools, non-integers, out-of-range values and unsupported input types.
No-op patches advance `tick` but not the data `version`.

The dirty frontier only tracks explicit changes. The system cannot observe
arbitrary writes to a caller's source list, hidden mutation through private
attributes, or concurrent writer effects. If a wider Python API is ever
provided, mutation invalidation must be designed separately.

## Mirrors and complexity

`verify()` independently checks totals using `sum`, generator multiplication,
and `pow`. It costs O(N), so it is used as a sampled diagnostic or test gate,
**not** inside each measured operation. It does not provide a cryptographic
commitment to externally mutable memory or a concurrency proof.

`apply_patch` is O(k) in number of supplied positions; `aggregates()` and
`variance_numerator()` are O(1); creating an owned buffer is O(N); `to_list()`
creates an explicit O(N) copy. The actual latency of large Python integer
aggregates, object allocation, and caches is empirical and non-constant in
bit width; complexity here is in element count.

## Fair end-to-end controls

`alexpy.persistent_bench` compares three routes with exactly the same
positions updated, read count, and final numeric checksum:

1. `list_full`: copy Python list, apply patches, recalculate `sum`,
   `sum_squares` using the existing AlexPy C acceleration, and variance.
2. `array_full`: allocate `array('q')`, apply the same patches, recalculate
   `sum`, `sum_squares` with the native C buffer kernel, and variance.
3. `owned_incremental`: construct `OwnedQBuffer`, apply same patches,
   retrieve cached aggregates and derived variance.

Timing includes **construction, mutations, and queries** in every route.
Patches are generated outside timing identically for all routes. Numeric
correctness is checked between all routes **before** timing. Paired routes
change order across epochs; sample arrays and median workflow times are kept.
Independent fresh Python interpreter *processes*, not independent machines,
are used for repeated observations. `list_full` and `array_full` are both
important baselines: improvements against only a slow Python-only expression
would exaggerate the contribution of incremental state maintenance.

Suggested local invocation:

```sh
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.persistent_bench --sizes 64,4096,32768 \
  --changes 1,32 --reads 1,8,64 --ticks 4 --epochs 2 \
  --pairs 7 --sessions 3 --target-ms 1.5 --json persistent.json
```

A reproducible graph may plot any measured state as `(size, changes,
reads, time)` and compare matched neighbors differing on one axis. The
previous v0.4/v0.5 space-time/fractal modules remain *descriptive*; no
fractal law or cache causality is assumed here.

## Failure and promotion gates

- Unsupported source semantics are rejected rather than silently bypassed.
- State is not invalidated on ordinary list edits; only explicit patches.
- `verify()` is not free and is excluded from performance results.
- Benchmarks measure one aggregate family, not general Python execution.
- Result is a research API with `promotion = BLOCKED_EXPERIMENTAL` even if
  one case is faster. All regressions and noise are retained in JSON.
- Cross-machine, cross-OS and memory-pressure replication is required before
  recommending adoption beyond the named workload.

## Observed local results — CPython 3.13.5, Linux, one host

Two exploratory runs were recorded; the **recheck** includes 3 fresh Python
processes, 2 epochs × 5 order-rotated paired samples per state, 3 patch
steps, changes per step `k={1,32}`, reads per step `{1,8,32}`, and
`size={4096,32768}`. Ratios below are `list_full / owned_incremental`
medians across processes. Larger than 1 means incremental won; smaller than
1 means it lost. **Every workflow time includes construction.**

| Size | k | Reads / patch | Ratio vs list full recompute | Implication |
|---:|---:|---:|---:|---|
| 4096 | 1 | 1 | 0.28× | Buffer bookkeeping loses |
| 4096 | 1 | 8 | 2.34× | Repeated reads start paying |
| 4096 | 1 | 32 | 8.74× | Strong local positive case |
| 4096 | 32 | 1 | 0.27× | Sparse update overhead loses |
| 4096 | 32 | 8 | 2.02× | Positive on this workload |
| 4096 | 32 | 32 | 7.03× | Positive on this workload |
| 32768 | 1 | 1 | 0.29× | Bookkeeping loses |
| 32768 | 1 | 8 | 2.41× | Positive on this workload |
| 32768 | 1 | 32 | 8.61× | Positive on this workload |
| 32768 | 32 | 1 | 0.30× | Bookkeeping loses |
| 32768 | 32 | 8 | 2.22× | Positive on this workload |
| 32768 | 32 | 32 | 8.18× | Positive on this workload |

The stronger `array_full` control was also measured in every case, using the
existing `_buffer_native` C kernel for each query. Full raw records include
all 3 processes, all epoch/pair samples, both controls, SHA-256, platform
labels, and exact workload states. See
`evidence/persistent-recheck-local-py313.json`. The one-axis 12-cell graph
(20 edges) is in `evidence/persistent-grid.dot`.

**Not yet established:** general Python acceleration, CPU cache causality,
thread-safe concurrent mutation, performance on multiple machines, or a
stable cross-platform crossover threshold. The measured workload repeatedly
asks for cached mathematical aggregates; a program doing unrelated work has
no reason to inherit these ratios. Preserve negative results and keep
`alexpy.sum_squares` unchanged.
