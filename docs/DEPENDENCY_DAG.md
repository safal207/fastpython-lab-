# AlexPy mission 10 — declared dependency DAG and dirty-frontier execution

## Scope / semantics

`alexpy.dependency_graph.DependencyGraph` is an **opt-in research prototype** for
an owned signed-64-bit integer buffer. It does not replace or intercept the
normal Python `list`, arbitrary call graphs or CPython bytecode. Existing
`alexpy.sum_squares` is unchanged. Users must register dependency nodes before
execution, and all mutations must use `apply_patch({index: new_exact_int})`.
The original list is snapshotted, and `to_list()` returns a separate copy.
No writable buffer interface is exposed. Private Python attributes, racing
threads, process-shared memory, native aliases and external mutation of the
backing store are **outside** the stated contract.

## Example

```python
from alexpy.dependency_graph import DependencyGraph

g = DependencyGraph([1, 2, 3, 4, 5, 6, 7, 8], bucket_size=2)
for name, start, stop in (("left", 0, 4), ("right", 4, 8)):
    g.add_window(name + "_sum", start, stop, "sum")
    g.add_window(name + "_sq", start, stop, "squares")
    g.add_variance(name + "_var", name + "_sum", name + "_sq")
g.add_sum("total", ("left_sum", "right_sum"))

transition = g.apply_patch({6: 12})
assert "left_sum" not in transition.updated_nodes
assert g.verify()  # O(N) full-state witness; do not call in hot path
```

Graph nodes support (1) exact window sums, (2) exact sums of squares,
(3) additive derived nodes and (4) exact unnormalized variance numerator
`n*sum_squares - sum**2` for a matching window pair. The graph can share
parents across multiple descendants and cover overlapping windows.
A node's value is propagated **only when its numerical value changes**:
changing `x` to `-x` touches the sum-of-squares window but does not invalidate
any descendant that depends only on the square sum. The DAG is acyclic by
construction: a new derived node can only refer to existing parents.

## Scheduling model

1. Validate the complete patch (indices must be exact `int` and inside the
   array, values must be exact signed-q `int`; reject `bool` or subclasses).
2. Determine the relevant range windows using coarse index buckets. This
   chooses a *candidate frontier*, then exact interval checks prune it.
3. Compute potential window deltas and propagate through a heap ordered by
   the nodes' topological registration indices. Only descendants of changed
   cached values can be enqueued. Unrelated branches do not run.
4. Atomically commit the validated patch to the owned buffer and publish the
   already calculated integer node results. Failed **validation** preserves
   the old data, tick, graph values and history. A hard process failure during
   memory allocation or object mutation is not a crash-safety guarantee.
5. Optionally call `verify()` to rebuild all results from a separate full
   expression, intentionally outside timing.

Complexity is approximately `O(k * relevant-bucket-fanout + active DAG edges
* log(active nodes))` in the sparse-update case, plus exact-big-int arithmetic
costs. This is **not** universal O(k): densely overlapping windows, wide
fanout and many changed indices can require processing much of the DAG. The
bucket index costs memory proportional to registered interval coverage.
The graph does not discover new Python dependencies at runtime and does not
establish CPU-cache causality, quantum behavior or physical fractal dynamics.

## Strong, fairly scoped controls

The benchmark measures three complete routes on the **same updates and
queries**, including initialization each time:

- `dense`: an existing AlexPy `sum_squares` native kernel is used for each
  dense window query; all declared results are recomputed.
- `manual`: a strong hand-coded conventional incremental algorithm maintains
  per-window sums and squares using direct index arithmetic; it uses no DAG.
- `graph`: `DependencyGraph` owns an array, uses bucketed dirty propagation,
  and reads every window's variance plus a shared global sum.

Run:

```bash
python -m unittest discover -s tests -v
python -m alexpy.dependency_bench --sizes 4096,32768 --changes 1,16 \
  --ticks 2,32 --reads 1,8 --sessions 3 --epochs 2 --pairs 5 \
  --json dependency-results.json
```

Each case verifies matching exact integer checksums **before** the timed
runs. Each fresh process alternates paired execution order. JSON preserves
all per-route samples, medians, checksums, platform details, source graph
counts and a SHA-256 digest. Session spreads are descriptive, not statistical
proof. The registration and array-conversion costs are included. Test machines,
thermal state, memory allocation and interpreter versions may alter results.

## Gate / research outcome

**No automatic promotion.** The graph has real correctness and selective
execution behavior, but its own scheduling and registration overhead must be
compared with the strong manual incremental control. A speedup relative to
full recomputation alone does not show superiority over strong conventional
incremental methods. Claims of a generalized faster Python engine are not
supported by this experiment.

Research lineage: COSMIC-ORGANICS sparse scheduling and witness design,
BardoCompute transition representation, CaPU guarded updates, ContractGraph-QA
state-transition test design. No processor speed claim is transferred here.

## Local pilot / reproducibility (CPython 3.13.5, Linux x86-64)

A 16-cell experiment measured size {4,096, 32,768}, changed indices per tick
{1, 16}, ticks {2, 32}, queries per tick {1, 8}; window width 256.
Three new interpreter processes, two epochs and five paired route observations
per process. For n=32,768, k=1, ticks=32, reads=8, medians of session ratios:

- dense full recomputation / graph: **5.12x** (range 5.12–5.14);
- conventional hand-coded incremental / graph: **0.168x** (range 0.168–0.170),
  meaning the conventional incremental control is ~5.94x faster than the DAG.

For n=32,768, k=1, ticks=2, reads=1, full dense / graph was **0.054x**
(a major regression), chiefly because registering and initializing a full
node set costs more than a few dense calculations. These are illustrative
single-host results; other workloads and machines require independent testing.
The prototype is **not** generally faster than well-designed incremental code.
See `evidence/dependency-results-local-py313.json` and `evidence/dependency-graph-local.dot`.
