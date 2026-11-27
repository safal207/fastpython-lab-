# Genesis CAUSAL-001 — fixed demand cone and exact boundaries

**Experimental, explicit opt-in.** Baseline HEAD `a9cfd0a66f327e51ac4de1c8403dddc91e261272`. No changes to the existing native kernels, graph APIs, defaults, or Physarum. This is a fixed-output exact-integer DAG experiment, not a new physics result or general Python accelerator.

## Model and contract

The existing `DependencyGraph` already performs value-aware propagation. CAUSAL-001 isolates the additional benefit of compiling the ancestor closure of requested outputs. Inputs are owned exact integers; nodes are immutable topological declarations of `sum`, `max`, `square`, or `positive`. No arbitrary callbacks, hidden mutable state, floats, dynamic edges or output switching. Changing requested outputs requires a fresh engine. Treat graph and engine configuration as fixed; no concurrent calls or private-state mutation. Unbounded bigint cost and pathological resource consumption remain possible; rollback on allocation failure is not guaranteed. Invalid ordinary patches are rejected before commit.

For deterministic pure nodes, a skipped node in the output closure is valid when all immediate parent values are unchanged. Nodes outside that closure cannot contribute to any requested output. Topological ordering makes this an inductive argument; equality is exact integer equality, not a hash or probabilistic confidence. The transition records evaluated nodes, equal boundaries, and changed nodes. This trace is not a standalone cryptographic certificate: checking it requires the graph and old/new state. No runtime full-graph scan is performed on the hot path.

## Controls

All routes expose the same requested values and return the same kind of transition trace (the sets of visited nodes differ).

- `full`: evaluate every declared derived node.
- `dirty`: full DAG, propagate through all descendants of changed inputs, including unchanged intermediate results.
- `value`: full DAG, stop propagation at exact equal boundaries (strong control reflecting existing AlexPy value-awareness; not a direct benchmark of the older range-specific implementation).
- `slice_full`: full recomputation of only the static output closure, no heap.
- `slice_dirty`: static output closure plus ordinary dirty propagation.
- `causal`: static output closure plus exact equal-boundary cutoff.

## Reproduction

```
python -m unittest discover -s tests -p test_causal_engine.py -v
# POSIX:
PYTHONPATH=. python tests/causal001_bench.py --output causal001-results.json
```

On Windows PowerShell, set `$env:PYTHONPATH='.'` before the second command. The new advisory CI workflow runs the same protocol on three operating systems and archives all raw samples; it never gates on timing.

Frozen seeds 51901–51903, each in a new interpreter; 400 patches for each of four synthetic fixtures, 417 derived nodes; six routes. Validate every transition and independently reconstruct the full DAG, all cached live values, exact cone, changed values and equal boundaries. Benchmark rotates route order per patch, warms routes, times update + requested read + trace allocation, separately measures construction and peak traced allocations. Raw data includes p95/p99 nearest-rank samples, build samples and process IDs. Shared graph and patches preallocated; tracemalloc is not total RSS. No production workload or native implementation speed claim.

## Local outcome (Linux, AMD EPYC 9V74, CPython 3.13.5)

Seven targeted unit tests passed (including 600 random patch transitions across six routes and a 5000-node chain); benchmark verified 28,800 route transitions. Median of per-process medians: value-aware → CAUSAL 11.73 → 2.17 us (narrow), 2.89 → 2.84 us (stable boundary), 16.77 → 3.26 us (shared), 183.89 → 181.41 us (dense). Exact counts and unrounded measurements in `evidence/causal001-summary-local.json`; raw timing data in the research bundle `causal001-local.json` with SHA-256 recorded in the summary, and new per-runner CI artifacts.

Narrow and shared fixtures show a material reduction in evaluations vs full-DAG value-aware control. The stable boundary fixture is already optimized by the strong control, and CAUSAL p99 is worse (5.14 → 10.80 us). In dense workload, static pruned full recompute is faster than CAUSAL and uses less traced memory. No promotion; no automatic routing. Study dynamic demand changes, validity of dormant cached states, bounded metadata and unseen graph families next, retaining static sliced and value-aware baselines. All performance results are advisory, not portable across CPUs or a scientific novelty claim.
