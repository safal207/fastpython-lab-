# Genesis CAUSAL-002 — dynamic causal space

Experimental and explicit opt-in. Parent commit `ce2744636e7bf5ad9ba3626bf4af131bdf138f95`. CAUSAL-001 fixed-output implementation and all existing defaults are unchanged.

## Contract and validity argument

The immutable, topologically ordered, exact-integer DAG from CAUSAL-001 is fixed; only the tuple of requested outputs changes. No hidden state, arbitrary functions, floating-point, topology mutation, threading or mutation of private state. Input aliases are copied. Ordinary invalid inputs and demands are rejected before state changes. Allocation failure rollback is not promised; arbitrary-precision integers can consume unbounded resources. Cache entry count is bounded by the fixed graph, not by elapsed steps. Byte usage is not bounded independently of integer magnitude.

Only the active ancestor closure is guaranteed current. In `causal`, update propagates along active edges and stops at equal exact values. A dormant node's cache may be stale. Upon demand change, newly activated nodes are processed in topological order; their parents are current by induction. A previously computed result is reused iff its saved tuple of exact parent values equals the current tuple; otherwise recompute. Overlapping active nodes were already valid. There is no clock-based or hash-based authorization. This supports an exact A→B→A return even if intermediate values changed while a node slept. An equal-boundary computation updates its parent signature even if the output remains equal. Reuse trace is inspectable with the graph and old/new state, not a standalone certificate or proof against tampering.

Four controls share graph semantics and transition trace shape. `eager` keeps every derived node current via value-aware propagation. `active_only` keeps overlapping active nodes, uses the same forward value-aware updates, discards dormant nodes, and computes newly activated nodes. `lazy_dirty` eagerly marks cached descendants dirty (stopping at already dirty/never cached nodes), checks only dirty or missing nodes upon activation, and uses exact parent-value signatures to avoid unnecessary arithmetic. `causal` has no dormant invalidation traversal, and checks signatures of newly active cached nodes on activation. All routes have a prebuilt reverse adjacency index. None uses live topology changes or online route promotion.

## Protocol

```
python -m unittest discover -s tests -p 'test_*causal*.py' -v
PYTHONPATH=. python tests/causal002_bench.py --output causal002-results.json
```

Windows PowerShell: set `$env:PYTHONPATH='.'` before the second command. Dedicated CI on Ubuntu, Windows, macOS with Python 3.13 archives raw results. Timing is advisory. Three fresh interpreters, seeds 62901–62903; five synthetic families, 264 derived nodes each; 300 steps each. Each step applies a patch, then selects outputs, then reads them. Rotating route order; warmup; time includes both trace allocations; construction separate. Untimed validation reconstructs all node values using an independent full DAG oracle, verifies the active cone and signatures and validates dormant reuse; timed and memory runs check requested output each step. Peak tracemalloc includes engine construction and replay, excludes prebuilt graph and operations, is not RSS. Repeated cheap unary operations intentionally stress metadata cost rather than simulate production Python.

## Local outcome

14/14 combined CAUSAL-001/002 targeted tests passed. New randomized test covers 1,200 demand/patch operations over four modes. Benchmark validates 18,000 route-steps (each includes apply, select and read). Environment Linux AMD EPYC 9V74, CPython 3.13.5. See `evidence/causal002-summary-local.json` for exact per-case aggregate metrics and raw-file digest. Raw full samples distributed in research archive and uploaded by CI.

Compared with `active_only`, `causal` cuts derived evaluations from 3,870 to 990 in stable-return workload and from 3,994 to 2,827 in stale-return. But median latency is slower on stable return, nearly tied on stale return, and slower on overlap/dense cases. All five p99s are reported; dense p99 is better than active_only on this run, not better than eager. On stable return peak traced memory 61.39→123.58 KiB. Compared with `lazy_dirty`, the causal route is faster in all five observed medians, but that is not sufficient to beat the stronger active-only control. No universal acceleration or production workload claim. No automatic activation.

Next: test a lower-cost validation representation (revision stamps as a separate exact control, with explicit A→B→A trade-off), bounded dormant memory and more expensive deterministic node work on independently seeded workloads. Measure break-even including validation, construction and cache memory, and retain active_only, lazy_dirty and eager controls. Do not select or tune a winner from the existing holdouts.
