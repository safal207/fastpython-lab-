# AlexPy 1.6 — Measured competing plans and sparse transition memory

**Research-only.** This experiments with an *owned, exact-int* range-query workload. It does not accelerate arbitrary Python code, does not alter `alexpy.sum_squares`, and does not simulate physical wave propagation or cache misses. No algorithm is auto-promoted.

## Why change the workload?

On the earlier partition-DAG workload, the manual incremental baseline won nearly all configurations. That is not a useful way to test whether remembered evidence helps choose different plans. This new *explicitly separate* problem issues point updates and variable `[start, stop)` range queries for both sum and sum of squares. Four correct alternatives have genuinely different cost profiles:

| Route | Preprocessing | Update | Query | Notes |
|---|---|---|---|---|
| `direct` | `O(n)` copy | `O(1)` | `O(window)` | Strong no-index baseline; Python's built-in `sum` plus exact square generator |
| `blocks` | `O(n)` | `O(1)` | `O(sqrt(n) + n/sqrt(n))` | Precomputed sums for full blocks; scans partial edges |
| `fenwick` | `O(n)` | `O(log n)` | `O(log n)` | Two binary indexed trees; pure Python |
| `prefix` | lazy `O(n)` per first post-update query | invalidates cache `O(1)` | `O(1)` once built | Good when many queries follow relatively few updates |

Preprocessing, data copying, all patch updates and all query readings are **inside the end-to-end timing boundary**. Two computed metrics are verified against an independent direct reference for every scenario and again during timing. Signed and arbitrarily large Python integers are supported; custom ints, booleans, floats, invalid indices, and unsupported mutation semantics are rejected. No external alias can mutate the owned buffer.

## Protocol

- **Training sizes:** 128, 512, 2048; **held-out sizes:** 384, 1536. Coarse log2 size buckets do not overlap.
- Variations: 1/8 point changes per tick; 1/8/32 range queries; short/wide windows; four ticks.
- Three independent CPython 3.13.5 Linux processes, seven paired AB/BA-like order-rotating samples per scenario.
- 36 training and 24 held-out configurations per process. The selector is **frozen before held-out route timings**. There is no learning from holdout measurements.
- Controls: always `direct`; global fixed winner selected using TRAIN-only total latency; fixed feature heuristic (`prefix` only for at least 16 wide queries); ant-style decayed affinity; wave-style uniform-grid min-plus potential; nondeployable hindsight oracle from heldout times.
- Policy lookup time is included in the end-to-end policy timings. Training and policy compilation costs are reported separately.
- The selector learns *normalized* per-route `log(latency/direct_latency)`, so an observation on a small array is not mistaken for the absolute time of a large one. Unverified plans are quarantined. Sparse matrix size `O(#observed state, plan pairs)`, not a dense tensor. Wave propagation uses a uniform **feature-distance heuristic**; do not mistake it for hardware propagation.

An earlier **exploratory pilot** included sizes 256 and 1024. The subsequent 384/1536 sizes were not timed during that pilot, but model/rule choices followed exploration. Therefore this is a prospective size holdout **within the chosen design**, not a fully independent preregistered or external validation. No threshold tuning or training on 384/1536 was done after the final run.

## What does a measured transition mean?

The `switch_probe` starts a backend on a training input, applies one real patch and query batch, copies its owned data into a new backend and times two phases:

1. `setup_ns`: actual time to construct the new backend after a plan switch (copy + tree/block layout build);
2. `tick_ns`: actual time to execute the next patch and query batch under the new workload.

The source snapshot is taken *before* these timers, and is explicitly excluded. Thus the measured setup cost understates end-to-end switching if the caller also needs to create a snapshot. This limitation is documented rather than hidden. All switched results are cross-checked with the direct reference on the identical input state. Switch edges are directed and route-specific, recorded only on training sizes; the cost of changing `blocks → fenwick` need not equal `fenwick → blocks`. **The switch cost is not currently part of plan scoring**, because the benchmark selects one plan for an entire workload; incorporating it would conflate static route selection with a stateful migration policy.

The log2 grid is an indexing convenience; it does not claim a fractal law or physical shortest path. No L1/L2/L3 hardware counter evidence is claimed.

## Results, one local host (CPython 3.13.5, Linux)

Across the three held-out processes, the hindsight winners varied: `direct` (43 of 72 process-scenarios), `prefix` (17), `blocks` (11), `fenwick` (1). Thus the experimental workload does have genuine competing alternatives.

| Control/selector | Median per-scenario speed ratio vs train-selected fixed `blocks` | Scenarios with >10% gain in *all* three processes | Scenarios with >10% loss in *all* three processes |
|---|---:|---:|---:|
| Fixed direct | 1.37× | 12/24 | 8/24 |
| Preregistered simple feature rule | **1.82×** | 17/24 | 5/24 |
| Ant-style memory | 1.19× | 11/24 | 4/24 |
| Wave-style memory | 0.80× | 7/24 | 10/24 |

The simple rule **outperforms the learned selectors overall** in this experiment. The wave selector over-generalizes from high-query training cells to low-query heldout cells and often pays to build `prefix` structures when few queries occur. It is not proven that graph memory improves route selection beyond simple features.

The switch probes record **48 real directed route-transition measurements per process**. On the first process, median switch setup cost was approximately 21 µs for `direct → prefix`, 21 µs for `prefix → direct`, 148 µs for `blocks → fenwick`, and 53 µs for `fenwick → blocks`. These are one-host timings with considerable range; use the full per-process JSON rather than treating them as universal.

Training took approximately 493 ms per process; policy compilation was approximately 5.8 ms. These figures are not amortized into the speedup table and **must** be considered in any deployment or repeated-workload claim.

These ratios are distributions of *per-scenario medians*, **not** whole-application throughput improvements or statistical confidence intervals. Source code and raw paired timings are in `evidence/range-competition-local-py313.json`; a compact summary is in `evidence/range-competition-summary-local-py313.json`. Unsupported input sizes and APIs require independent validation. Negative results are retained.

## Reproduction

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.range_bench --train-sizes 128,512,2048 --holdout-sizes 384,1536 --changes 1,8 --queries 1,8,32 --spans short,wide --ticks 4 --pairs 7 --sessions 3 --json range-competition.json
```

No global CPython or native C speedup is established. Next possible experiments: register real transitions between changing workload regimes, charge both snapshot and rebuilding costs, then test whether a policy actually beats a simple rule in a new host-independent holdout.
