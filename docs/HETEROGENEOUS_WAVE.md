# AlexPy v1.5 — heterogeneous wavefront and memory matrix

## Hypothesis

A finite five-dimensional workload grid can represent *alternative, verified*
execution plans. Nonuniform transition penalties may help move evidence from
measured states to previously unseen states more accurately than a constant
Manhattan-distance penalty. The dependency-DAG execution order is never changed.

This is **not** a physical simulation of sound, light, or processor cache.
The synthetic default coefficients do not measure hardware cache misses:

- Axes: `log2(size), log2(bucket width), log2(changes), log2(ticks), log2(reads)`.
- All steps are between adjacent integer cells; graph edges are directed.
- Each step's default penalty is the preregistered axis weight times a
  nonnegative size-dependent multiplier. The multiplier becomes non-unit
  above the *hypothesized* size coordinate 9, not an observed cache boundary.
- `observe_transition()` can replace default costs for a directed edge with
  a mean **verified elapsed/reference ratio**, clipped into `[.05,20]`.
  Edge timing must be independently measured; the main blind benchmark does
  not fabricate or infer edge latency from endpoint execution time.
- Quarantined directed transitions are blocked. The source/action memory and
  the transition memory remain sparse (proportional to measured evidence).

## Algorithms

- **Heterogeneous light:** for each correctness-gated source plan, compute
  `seed_log_loss + shortest_path_cost(source,target)`, then take the minimum.
  Reverse-edge Dijkstra computes the path in the finite L1 ball of radius
  `max_hops` around the requested target. Nonnegative weights are required.
- **Heterogeneous sound:** combine independent source log-loss signals with
  `exp(-shortest_path_cost)` attenuation and a finite-sample confidence
  penalty. This is a score, not a solution of the acoustic wave equation.
- **Uniform light:** existing constant-cost min-plus / Manhattan shortcut.
- **Ant memory:** existing decaying sparse pheromone matrix.
- **Manual:** strong fixed end-to-end execution control.

To reduce compile costs, Dijkstra stops after the selected measured sources
have been settled. When no transition-specific evidence or blocks exist, the
same reverse-distance field can be cached across plan choices. Cache is
bounded and cleared on a transition observation or quarantine.

The graph search never ignores mandatory dependency edges, and it never
rewrites `alexpy.sum_squares()`.

## Independent verification

- The Dijkstra distances are checked against an independent Bellman–Ford
  relaxation over the same finite directed five-dimensional grid.
- With uniform axis weights and zero size slope, heterogeneous light gives
  the same distances as the analytical homogeneous min-plus solution.
- Invalid inputs, no-op/stale decisions, directed edge isolation, cache
  invalidation, quarantine and finite-numerical inputs have unit tests.
- Each candidate route is independently checked against a dense reference
  before timing on an unseen workload. No held-out observation trains the
  router. Every selected plan is frozen before execution timings begin.

## Observations — local Linux / CPython 3.13.5

Three separate interpreter processes, paired route order, eight held-out
states (`size` 256 and 512, `changes` 1 and 4, `reads` 1 and 8, `width` 64,
`2` ticks). Training sizes 128 and 1024 were withheld from the test split.
The figures below are exploratory process-median ratios against *manual*.
Higher is faster; `1.0` is equal speed. They vary with host noise.

| Candidate policy | Initial, uncached | With stopped search | Cached geometry |
|---|---:|---:|---:|
| Ant | 0.881x | 0.899x | 0.927x |
| Uniform light | 0.877x | 0.922x | 0.931x |
| Heterogeneous light | 0.925x | 0.924x | 0.938x |
| Heterogeneous sound | 0.915x | 0.918x | 0.928x |

Every candidate selected `manual` in all eight held-out states in all
three sessions. No verified >10% speedup over manual. This means the table
primarily describes policy-call overhead and timing noise, **not the benefit
of a better chosen algorithm**. Runtime policy inference is memoized; route
execution, including that lookup, is included in the timings.

**Offline policy compilation for all eight holdout states**, including all
four learning policies, had medians about `460.5ms` (initial), `413.3ms`
(early-stop), and `151.2ms` (shared distance geometry), each measured in
three separate processes. This is not a formal apples-to-apples causal
speedup, because the sessions were run sequentially under varying load.
Preparation is kept outside the online execution loop and reported separately.

The fitted evidence does **not** include independently measured edge penalties.
Such evidence requires new controlled transition timing experiments and a
wider distribution of workloads where multiple safe plans can beat manual.

## Reproduce

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests -q
python -m alexpy.hetero_bench --train-sizes 128,1024 \
  --holdout-sizes 256,512 --changes 1,4 --reads 1,8 \
  --width 64 --ticks 2 --pairs 3 --sessions 3 \
  --json hetero-report.json
```

The raw evidence is retained in the research archive. GitHub CI runs a
smaller *advisory* smoke variant on Python 3.15, while full semantics are
checked across the existing OS/Python test matrix.

**Promotion gate:** BLOCKED_EXPERIMENTAL. To justify activation, test actual
verified heterogeneous edge measurements, include full preparation and
memory footprint, show improvements on independently held-out workloads,
and cross-check correctness under each supported platform.
