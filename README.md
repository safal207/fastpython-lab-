# AlexPy 0.1 — Python Performance Lab

**Status:** research prototype, not a Python interpreter or CPython fork yet.

Our first measured hypothesis: a specialized native C loop can outperform
`sum(x*x for x in values)` when `values` is a plain `list[int]` or `tuple[int]`.
This is a **single-operation speedup**, **not** a claim that "Python is N× faster".
The experiment preserves Python semantics by returning to the normal Python
implementation for unsupported input types or values outside the signed
64-bit fast path.

## What it contains

- `alexpy/_native.c`: original CPython C extension, handwritten native loop.
- `alexpy/__init__.py`: safe dispatcher/fallback, arbitrary precision preserved.
- `alexpy/benchmark.py`: warm-ups, repeated wall-time samples, median,
  baseline and backend, JSON export with hardware and interpreter metadata.
- `tests/`: correctness tests including large ints, subclasses, and iterables.
- `.github/workflows/ci.yml`: Linux, Windows, macOS; Python 3.13 and 3.15;
  correctness gates and benchmark artifact.

## Build & test (Python 3.13–3.15)

Requires a C compiler supported by your Python installation.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.benchmark --require-native --sizes 1000 100000 --json benchmarks.json
```

Do not compare results obtained on different hardware/interpreters, and do not
substitute Python fallback results for a native performance comparison.
The baseline is the **Python generator expression**, not NumPy, a precompiled
kernel, or the entire CPython interpreter. Sample sizes, run counts and
crossover points affect the reported speedup.

## Important limitations

- Only the `sum_squares` function is optimized; all other Python runs normally.
- No new Python syntax, parser, JIT, or modified interpreter core yet.
- The native fast path is designed for exact `list[int]`/`tuple[int]` objects;
  uncommon inputs fall back to Python for correctness.
- This module does not add parallel execution or eliminate the GIL.
- Performance on a particular PC is not evidence of a universal speedup.

## Roadmap

1. **v0.1 (this repository)**: correctness, C extension, reproducible benchmark.
2. **v0.2**: profile benchmark bottlenecks; add comparison with optimized
   baselines, CPython 3.15 and PyPy; track noise and CI regressions.
3. **v0.3**: separate CPython fork experiment; measure a specific bytecode
   dispatch/optimizer change while running `test.regrtest` and pyperformance.
4. **v0.4**: third-party reproduction and a technical report with wins *and*
   regressions.

See [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the acceptance criteria.

## License

MIT. This repository contains original code, no vendored CPython files.

## v0.2 state-transition graph explorer

A correctness-first experimental search instrument is now available:

```bash
python -m alexpy.graph_search --require-native --cases 12 --repeat 9 \
  --json graph-results.json --dot graph-states.dot
```

It uses pairwise test design, boundary values, one-axis state transitions,
prewarm/steady observations, independent-expression mirrors, and candidate-only
statistical verdicts. Read [docs/GRAPH_EXPLORATION.md](docs/GRAPH_EXPLORATION.md).
Inspired by prior COSMIC/MORPHOS, CaPU, BardoCompute and ContractGraph-QA
research; their performance results are not imported as Python performance evidence.

## v0.2 regression triage: fresh-process replications

The previous graph found four potential regressions. A separate diagnostic
traces which **code path** each input takes, compares one-axis controls that
actually change the workload, and checks repeated paired timings in new
Python interpreter processes:

```bash
python -m alexpy.regression_triage --sessions 3 --repeat 9 \
  --target-seconds .004 --json regression-triage.json
```

The output preserves raw samples and distinguishes a code-path explanation
from causal evidence of timing. See [docs/REGRESSION_TRIAGE.md](docs/REGRESSION_TRIAGE.md).

## v0.3 opt-in A/B interventions (not enabled by default)

Two separately testable alternatives isolate the exact integer overflow
fallback and small-container call overhead. Neither changes `sum_squares`:

```bash
python -m unittest discover -s tests -v
python -m alexpy.intervention --sessions 3 --repeat 9 \
  --target-seconds .004 --json intervention-results.json
```

The experimental methods are `alexpy.candidates.bigint_native` and
`alexpy.candidates.small_dispatch`. On CPython 3.13.5 / one Linux host, both
improved selected overflow/singleton workloads but both caused repeatable
regressions on guardset cases. No candidate is promoted. See
[docs/INTERVENTION.md](docs/INTERVENTION.md) for raw-measurement procedure,
negative controls and the promotion gate.

## Space-time + multiscale investigation (opt-in)

`alexpy.adaptive.adaptive_sum_squares` experiments with a conservative rule-based
route for exact built-in containers; the public API remains unchanged.
`alexpy.spacetime` adds adjacent graph transitions through explicit time,
first/second discrete derivatives of log speedup, and a finite-sample multiscale
roughness **fractal-axis proxy** (not a fractal dimension).
Run `python -m alexpy.spacetime --json spacetime.json` after building the extension.
See [precise definitions and limitations](docs/SPACETIME_FRACTAL.md).

## v0.5 lattice-grid causal hypotheses and blind length holdout

A new opt-in `alexpy.lattice_lab` treats the workload configuration as a
discrete 4D grid plus experimental time, with adjacent-cell contrasts,
footprint **estimates** and multi-scale differences. Length groups 8 and 1024
are held out from training and used to compare 1-hop and multi-scale
regression forecasts against a constant median baseline. This method is
**inspired by numerical grids**, not an implementation of lattice QCD or
proof of CPU-cache or fractal behavior.

```bash
python -m alexpy.lattice_lab --epochs 2 --repeats 5 --loops 25 --json lattice.json
```

See [docs/LATTICE_LAB.md](docs/LATTICE_LAB.md). Public API unchanged.


## v0.6 native CPU memory-access probe and cache hypotheses (opt-in)

`alexpy.memory_lab` adds an **independent native C** pointer-chase and sequential
memory-read experiment, reported cache topology, space/time transitions,
finite differences and optional hardware miss counters. It tests memory
working-set hypotheses, **not** interpreter acceleration. Hardware counter
access failures are recorded explicitly as `UNAVAILABLE`, not made-up counts.

```bash
python -m alexpy.memory_lab --sizes-kib 4 32 256 2048 16384 \
  --epochs 2 --repeats 5 --target-accesses 750000 \
  --hardware-counters --json memory-lab.json
```

See [experimental protocol, limitations and independent evidence](docs/MEMORY_LAB.md).
This does not change public `alexpy.sum_squares` or promote any candidate.


## v0.7 zero-copy typed-buffer traversal (opt-in; guardset only)

`alexpy.memory_candidates` adds **separate**, deliberately narrow candidates
for exact `array('q')` and contiguous one-dimensional native signed-64
`memoryview` input: sequential buffer reading, 64-way interleaved traversal,
and conversion to `list` *inside* the timed callable as a comparison control.
Each candidate falls back to the original `sum_squares` for unsupported types.
The public dispatcher is untouched. No inputs are converted before timing.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.layout_bench --epochs 3 --pairs 9 \
  --sizes 0,1,8,128,4096,32768,262144 --json layout.json
```

On one CPython 3.13.5 Linux host, three fresh processes independently saw
roughly **12x acceleration** for sequential native traversal of 4,096 small
signed integers *already stored as* `array('q')` vs the Python generator on
that very same input. Large 64-bit values generally gain far less. Pre-copying
large-value arrays into Python lists can be slower. These figures **do not**
include the cost of choosing/creating the array representation upstream and
are **not** a claim about arbitrary Python programs or CPU speed.
See [protocol, raw evidence and limitations](docs/LAYOUT_TRAVERSAL.md).
Candidate promotion remains blocked pending cross-OS and multi-host review.

## v0.8 end-to-end layout amortization (opt-in)

`alexpy.amortization` and `alexpy.amortization_bench` measure the *entire*
workflow when converting exact `list[int]` / `tuple[int]` into `array('q')`:
validation, allocation, transfer, K repeated calculations, and optional
conversion *back into a fresh list*. The control computes the same output
using the existing AlexPy list path; a separate negative control repacks
before every calculation. No public dispatcher is changed.

```bash
python -m alexpy.amortization_bench --sizes 8,4096 \
  --repetitions 1,8,32,64 --profiles small,large \
  --epochs 2 --pairs 7 --candidates pack_once \
  --target-seconds .0003 --json amortization.json
```

In three local Python 3.13.5 Linux processes, for 4,096 small ints, pack-once
was ~0.06× at K=1 but ~1.78× at K=64 when no return-list copy was needed.
When a new list must be returned, K=64 did **not** consistently improve speed;
large 64-bit integers also failed to show a consistent full-path gain at K=64.
These are local bounded observations, **not** a generally faster Python.
See [docs/AMORTIZATION.md](docs/AMORTIZATION.md) and
[evidence summary](evidence/amortization-summary-local-py313.json).

## v0.9 owned persistent q-buffer with sparse state transitions (opt-in)

`alexpy.persistent.OwnedQBuffer` copies an exact built-in `list[int]` or
`tuple[int]` of signed 64-bit numbers into a long-lived `array('q')`. It
maintains exact Python-integer **sum** and **sum of squares** incrementally:
`apply_patch({index: new_value})` updates just the selected indices, and
`variance_numerator()` is derived from those two cached aggregates. Each patch
records a logical tick and version; the optional trace records a sparse dirty
frontier. `verify()` independently recomputes mirrors when requested.

**Not a replacement for ordinary mutable lists**: external list mutations are
not watched, mutable backing views are never exposed, only explicit patches
are supported, and no promise of thread-safe access or tamper-proof storage is
made. No change to the public `alexpy.sum_squares` dispatcher.

End-to-end comparisons include object construction, all updates, and all
aggregate reads, against both ordinary list recalculation and native q-buffer
full recalculation. They show when sparse maintenance is useful and when
construction or patch overhead dominates, not how much all Python improves.

```sh
python -m alexpy.persistent_bench --sizes 64,4096,32768 --changes 1,32 \
  --reads 1,8,64 --ticks 4 --sessions 3 --json persistent.json
```

See [the v0.9 research protocol](docs/PERSISTENT_BUFFER.md).

## Mission 10: opt-in dependency DAG, sparse patches and strong controls

`alexpy.dependency_graph.DependencyGraph` supports declared sum, square-sum,
variance-numerator and additive dependency nodes over an owned signed-q buffer.
Per-index patches propagate only through graph nodes whose numeric values change;
unrelated intervals remain cached. A full independent `verify()` checks results.
The implementation does **not** trace arbitrary Python functions or replace the
normal `sum_squares` dispatcher.

```bash
python -m unittest discover -s tests -v
python -m alexpy.dependency_bench --sizes 4096,32768 --changes 1,16 \
  --ticks 2,32 --reads 1,8 --sessions 3 --json dependency-results.json
```

**Important:** The strong hand-coded incremental comparator **beat** the DAG
in the 16-cell local benchmark, even where the DAG beat full recomputation. See
[docs/DEPENDENCY_DAG.md](docs/DEPENDENCY_DAG.md) and raw evidence. All
performance-based dispatch promotion remains blocked.

## v1.1 compiled DAG frontier — opt-in, strong controls

`alexpy.compiled_graph.CompiledDependencyGraph` is an opt-in, frozen version of
`DependencyGraph`. It compiles name-based dependencies into numeric node IDs
and initializes typed range sums without repeated Python index checks.
Registration is frozen by an explicit `compile()` call. Every patch is still
validated and committed by the owned buffer; cached results are published only
after the patch succeeds. `verify()` remains an independent dense mirror.

The `alexpy.compiled_bench` protocol compares four equivalent end-to-end
routes (dense, hand-optimized manual, previous DAG, compiled DAG), **including
construction and compilation**, with paired reversed order and independent
processes. No automatic default promotion; all results are workload-specific.

```bash
python -m unittest discover -s tests -v
python -m alexpy.compiled_bench --sizes 4096,32768 --changes 1,16 \
  --ticks 2,32 --reads 1,8 --sessions 3 --epochs 2 --pairs 5 \
  --json compiled-results.json
```

See [compiled DAG methodology](docs/COMPILED_DAG.md) and checked-in local
pilot measurements at `evidence/compiled-results-local-py313.json`.

## v1.2 structure-specialized partition graph (opt-in; not promoted)

`alexpy.partition_graph.PartitionDependencyGraph` detects an **exact,
nonoverlapping partition template** at `compile()` time. For this proven
shape, patches bypass the generic heap-based dirty frontier and maintain
cached windows, variance and global sum using only changed partitions. Other
DAGs safely delegate to the preceding compiled algorithm.

The end-to-end `alexpy.partition_bench` compares dense, manual incremental,
previous compiled DAG, and guarded partition DAG with exact checksum gates.
On one local Linux CPython 3.13.5 host (16 configurations, 3 fresh processes),
7 configurations had >=10% speedup over the previous graph in every process;
**the manual incremental control still beat the new graph in all 16 cases**.
This is **not** generally faster Python and is **not** enabled by default.

```bash
python -m unittest discover -s tests -v
python -m alexpy.partition_bench --sizes 4096,32768 --changes 1,16 \
  --ticks 2,32 --reads 1,8 --sessions 3 --epochs 2 --pairs 5 \
  --json partition-results.json
```

[Methodology, constraints, negative results](docs/PARTITION_COMPILER.md)
([raw local evidence](evidence/partition-results-local-py313.json)).

## v1.3: Ant-Memory Graph — sparse matrix for verified execution alternatives

The opt-in `alexpy.ant_memory` combines a graph of **eligible complete plans**
with a sparse state-by-route matrix of bounded pheromone scores, lazy aging,
space/time transition counts, failure quarantine, and compiled decision cache.
A blind-holdout bench `alexpy.ant_bench` compares this policy against the
strong fixed manual incremental algorithm, fixed partition and the retrospective
oracle; all routes must match a dense checksum before measuring.

The **result is negative so far**: the ant policy selected the strong manual
algorithm on all eight held-out cases and did not deliver a confirmed gain.
Compiling decision lookups reduced selection overhead; the public API and
automatic runtime dispatch remain unchanged. See
[docs/ANT_MEMORY_GRAPH.md](docs/ANT_MEMORY_GRAPH.md) and the raw JSON evidence.

```bash
python -m alexpy.ant_bench --sessions 3 --pairs 4 --json ant-memory.json
```

## Experimental wavefront memory — light / sound propagation analogy

`alexpy.wave_memory` adds an **opt-in offline** multi-source selector on top of
previously verified alternative execution plans. A light-style front minimizes
arrival cost on a uniform five-dimensional workload grid. A sound-style field
pools distance-decayed evidence from multiple training cells. A Dijkstra
reference checks the compiled min-plus uniform-grid shortcut. The wavefront
**never** changes obligatory dependency-DAG edges, executes unverified routes,
or replaces the public `sum_squares` API. Unknown workloads default to the
strong manual plan. Neither algorithm simulates physical light or sound.

```bash
python -m unittest discover -s tests -q
python -m alexpy.wave_bench --train-sizes 128,1024 --holdout-sizes 256,512 \
  --changes 1,4 --reads 1,8 --width 64 --ticks 2 \
  --pairs 4 --sessions 3 --json wave-memory.json
```

On local CPython 3.13.5 Linux, 168 tests passed, but **neither wave variant
showed any confirmed >10% improvement** over the manual control on eight
blind length holdouts. Training+compilation and occasional runtime
regressions are part of the measured cost. See
[WAVE_MEMORY.md](docs/WAVE_MEMORY.md) for the results and limitations.


## v1.5 — heterogeneous wave-memory research (opt-in)

`alexpy.hetero_wave.HeterogeneousWaveMemory` extends the earlier light/sound
planning experiments from a constant-cost state lattice to **bounded directed
nonnegative edge costs**. Its default axis weights and working-set penalty
are explicitly *heuristics*, NOT hardware cache-miss measurements. Adjacent
edges may be updated only with separately timed and verified transition
ratios; incorrect transitions are quarantined. Dijkstra is checked against an
independent Bellman–Ford implementation in the tests. Distance geometry can
be reused across plans, and learned policies remain offline / opt-in.

```bash
python -m alexpy.hetero_bench --train-sizes 128,1024 \
  --holdout-sizes 256,512 --changes 1,4 --reads 1,8 \
  --width 64 --ticks 2 --pairs 3 --sessions 3 --json hetero-report.json
```

See [docs/HETEROGENEOUS_WAVE.md](docs/HETEROGENEOUS_WAVE.md).
Observed 8-state holdout (Linux, Python 3.13.5) still did **not** beat the
strong fixed manual control. The stable public API has not been changed.


## AlexPy 1.6: verified competing range-query plans and measured transitions

An **opt-in, separate** range-query research workload has four complete
alternatives (`direct`, `blocks`, `fenwick`, `prefix`). The `range_memory`
module stores sparse correctness-gated timing evidence and records actual
directed layout-switch setup and tick costs in training data. Policies are
frozen before held-out workload execution. The public `sum_squares` remains
unchanged, and **neither ant nor wave learning beats a strong simple feature
rule** on the current holdout. This negative result is preserved.

```bash
python -m alexpy.range_bench --train-sizes 128,512,2048 --holdout-sizes 384,1536 --changes 1,8 --queries 1,8,32 --spans short,wide --ticks 4 --pairs 7 --sessions 3 --json range-competition.json
```

See [range transition methods and limitations](docs/RANGE_TRANSITION_MEMORY.md).

## AlexPy 1.7: noise-aware route memory, measured switching and profile isolation

`alexpy.stable_memory.StableRouteMemory` is an **opt-in offline** router for the
four verified range-sum+square backends. It normalizes timings against the
same-process `simple_rule` baseline, counts independent **processes** (not
iterations) as evidence, penalizes the worst observed ratio and size
extrapolation, and defaults to `simple_rule` if a material gain has not been
replicated. The exact device/runtime profile is required for training and
execution. An independent switch model only recommends rebuilding a live
layout if projected savings over a stated future horizon exceed setup cost.

```bash
python -m unittest discover -s tests -q
python -m alexpy.stable_bench --train-sizes 128,512,2048 \
  --holdout-sizes 384,1536 --changes 1,8 --queries 1,8,32 \
  --ticks 4 --pairs 5 --sessions 3 --device-tag my-local-machine \
  --json stable-memory.json
```

**Local research finding:** the stable selector beat the simple rule on four
specific wide-range held-out cases, but did **not** show meaningful overall
median acceleration across all 24 holdouts. Model calibration is not free,
and it is not validated across real hardware devices. Details and negative
results: [docs/STABLE_ROUTE_MEMORY.md](docs/STABLE_ROUTE_MEMORY.md).

## AlexPy 1.8: real stream economics and cross-runner probes (advisory)

`alexpy.stream_economics` executes a **continuous mutable input** over a
changing workload schedule; an algorithm change rebuilds its actual state
from the previous live backend, and that full cost is in the online timer.
`alexpy.stream_bench` trains conservative route memory in three distinct
processes on one host, freezes its holdout decisions, checks complete-stream
checksums against a direct witness, and measures the paid break-even horizon.

```bash
python -m alexpy.stream_bench --train-sizes 128,512,2048 \
  --holdout-sizes 384,1536 --changes 1,8 --queries 1,8,32 \
  --spans both --ticks 2 --pairs 3 --sessions 3 \
  --cycles 1,8,32,128 --reps 3 --device-tag local-benchmark-device \
  --json stream-economics.json
```

Offline training + frozen table compilation are charged once; online time
includes actual layout switches, initial construction, route choice, updates
and queries. Only sizes (and coarse cells) absent from training are evaluated.
The report does not claim a verified payback beyond the largest **measured**
horizon. Linear payback projections are explicitly labeled speculative.

CI also measures a bounded smoke workload **separately** on Ubuntu, Windows
and macOS runners (Python 3.13). Each runner trains its own memory and uploads
a hardware/OS-tagged JSON artifact; timings cannot be pooled across hosts,
and one CI OS alone does not establish any particular CPU speedup.

Full protocol, limitations and local results: [docs/STREAM_ECONOMICS.md](docs/STREAM_ECONOMICS.md).

## AlexPy 1.9: long-horizon paid payback and CPU-profile isolation

`alexpy.stream_horizon` rechecks a frozen route table on actual long-lived
streams. It refuses to import timings from a different CPU/Python/OS profile,
mirrors the whole stream checksum against a direct reference and requires
**three independent processes** to cover the measured calibration cost before
reporting payback. A new AMD EPYC calibration is intentionally separate from
the prior Intel Xeon results. No general-Python or cross-device speedup is
claimed. See [docs/STREAM_LONG_HORIZON.md](docs/STREAM_LONG_HORIZON.md).

A follow-up three-process experiment on newly calibrated AMD EPYC 9V74 observed paid payback at 8192 tested cycles. This is a specific sustained range-query workload, not general Python acceleration.

## AlexPy QA Performance Lab (experimental, mission 19)

`alexpy.qa_lab` implements an explicit `test -> suite -> global` dependency
model for CI reruns, retries and QA dashboards. It mirrors a strong
incremental Python baseline and an independent full-scan reference.
`alexpy.qa_actions` imports a checked-in, sanitized six-job sample from a
real public GitHub Actions run. The **performance** experiments use a separate,
fully disclosed **synthetic** QA event generator, not production logs.

```bash
python -m unittest discover -s tests -q
python -m alexpy.qa_actions --input evidence/qa-actions-jobs-sample.json
python -m alexpy.qa_bench --sessions 3 --pairs 5 --json qa-lab.json
```

The initial local experiment failed the performance gate on all four
scenarios: AlexPy was 0.35–0.64x as fast as *strong* Python counters. This is
important negative evidence. The C/native backend is not auto-promoted,
and neither this result nor old synthetic speedups demonstrate application-wide
acceleration. See [full protocol](docs/QA_PERFORMANCE_LAB.md).

### AlexPy 2.1 — experimental QA Batch Engine (mission 20)

`alexpy.qa_batch.AlexPyBatchQAGraph` consumes an exact list or tuple of
`QAEvent` values as one atomic batch. All events are validated before any
change; repeated test IDs retain the last valid state. Intermediate updates
are not visible until the batch closes. The corresponding
`StrongPythonBatchQA` provides a *strong batch Python baseline*.

Local CPython 3.13.5 benchmark (16 configurations, 3 independent processes,
3 paired route rotations per configuration): the batch graph is about **2.006×**
faster than the previous per-event AlexPy graph by median of configuration
ratios, but only **0.632×** as fast as optimized batch Python; **no default
promotion**. All intermediate measurements and caveats are in
[`docs/QA_BATCH_ENGINE.md`](docs/QA_BATCH_ENGINE.md).

```bash
python -m alexpy.qa_batch_bench --smoke --sessions 2 --pairs 2 --batch-sizes 1,8,64 --json qa-batch.json
```
