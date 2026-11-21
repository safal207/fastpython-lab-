# AlexPy 1.2 — guarded partition-compiler experiment

## Research question

Does proving an immutable graph topology once remove enough event-queue and
name-lookup overhead to make a declarative dependency DAG competitive with a
manual incremental algorithm, **without weakening the exact-int result**?

## Algorithm and fail-closed gate

`alexpy.partition_graph.PartitionDependencyGraph` extends the frozen compiled
DAG. `compile()` checks the **entire** declared topology, not just sampled
states: one pair of sum and squared-sum windows for each contiguous nonempty
bucket-aligned partition, one same-range variance node per pair, and one
terminal additive node depending on all window sums. Node order, boundaries,
kind, dependencies, and unique coverage are all checked.

If any condition fails—including overlapping windows, extra nodes, empty
input, noncanonical order—the class delegates each patch to the existing
`CompiledDependencyGraph` implementation. Other graph programs are unchanged.

For a proven partition, `apply_patch()` validates the exact dict, exact-int
indices and signed-64-bit values, accumulates the changed sums and squares per
partition, updates the affected variance nodes, and updates the global sum
exactly once. It skips the generic dirty-frontier heap. Numeric values use
Python's arbitrary-precision `int`, so a sum of squares larger than 2^64
remains exact. No value is committed before all patch validation and
arithmetic. An internal `_commit_prepared` method in `OwnedQBuffer` removes
duplicate validation for the already checked triples. This helper is private:
callers must not bypass the normal `apply_patch` API.

The legacy event, version/tick, unchanged-node suppression, trace and
independent `verify` semantics are retained. Exact equality of snapshots and
events is tested across randomized patches, overflow boundaries, cancellation,
sign flips, invalid patches, and fallback DAGs.

## Reproduce

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.partition_bench --sizes 4096,32768 --changes 1,16 \
  --ticks 2,32 --reads 1,8 --sessions 3 --epochs 2 --pairs 5 \
  --json partition-results.json
```

Benchmark compare four semantically identical workflows: **dense** full
recalculation, **manual** hand-written sparse update, **compiled** generic
frozen graph, and **partition** checked fast-path graph. The timed boundary
includes data conversion, creation/registration, compilation, patches, and
all result reads. Order rotates and reverses within each session; three fresh
CPython processes run the same test grid. Checksum agreement is mandatory
before and during timings. Ratios >1 indicate the partition candidate is
faster. Samples are raw nanoseconds and SHA-256 is included for inspectability.

## Bounded local observations (CPython 3.13.5, Linux, one host)

16 configurations, three independent interpreter processes, two epochs times
five paired samples per route and process:

- Against the prior compiled DAG: **7/16 scenarios** showed >=10% speedup in
  all three processes; median of per-case median ratios ~**1.03x**, full range
  ~**0.95–1.53x**. Small workloads frequently show no gain.
- Against manual incremental: manual won **16/16 scenarios**, often by a wide
  margin. This specialization does **not** close the entire performance gap.
- Against dense recomputation: large multi-read cases often win, but several
  brief two-tick cases lose because compilation/construction dominates.
- No artificial increase of benchmark budgets or loosened correctness gate.

All raw data: `evidence/partition-results-local-py313.json`.
Machine-readable compact summary: `evidence/partition-summary-local-py313.json`.

## Causal limits and next gate

A fast path tied to a proven graph shape does not prove hardware/cache
causality, a fractal advantage, automatic Python tracing, or whole-program
acceleration. There is no user-visible API promotion; `alexpy.sum_squares`
remains unchanged. The opt-in class may still be slower than a handwritten
algorithm. Next experiment: factor construction vs patch vs result-read cost,
profile allocation, and consider a **native batched patch engine**, but only
if it passes the same correctness, cross-platform and guardset checks.
