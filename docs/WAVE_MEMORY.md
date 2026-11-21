# AlexPy: light/sound-inspired wavefront planning (research prototype)

## Safety and meaning

We do **not** simulate electromagnetic/acoustic wave equations or claim a quantum,
processor-wide, or CPython-wide speedup. The analogy concerns dissemination of
measured *evidence* across a parameter lattice. The dependency DAG determines
mandatory computation and may not be skipped. Four separately verified whole
algorithms (`manual`, `partition`, `compiled`, `dense`) are the only actions.

Five-dimensional grid cell:
`(floor(log2(size)), floor(log2(width)), floor(log2(changes)),
floor(log2(ticks)), floor(log2(reads)))`.

Memory is sparse over observed `(grid_cell, algorithm)` entries: each contains
mean log performance ratio `log(manual_time/algorithm_time)`, number of
correctness-gated samples and experimental epoch. An invalid checksum
quarantines an action for that exact cell. No quarantine can be cleared by
more performance samples; restart a fresh memory object after revalidation.
Unseen cells without nearby evidence default to `manual`.

## Light: minimum arrival time (fast route)

Let `L_i(a)=-mean(log(manual_time/algorithm_time))` be the normalized,
measured cost of an alternative plan `a` at seed cell `i`. On uniform
axis-neighbor edges of constant nonnegative cost `lambda`:

`T_a(s) = min_{i within max_hops} [L_i(a) + lambda * Manhattan(i,s)
                               + age_penalty * age_i]`.

A reference implementation propagates a front with a priority queue
(Dijkstra). The compiled version uses the exact L1 min-plus identity for the
uniform lattice and stores only observed sources; it does not allocate an
N-by-N transition matrix. Randomized equivalence tests check this shortcut.
On *nonuniform* graph weights this simplification is **not valid** and the
reference shortest-path algorithm is required.

## Sound: damped multi-source field

Independent validated evidence contributes a positive amplitude
`w_i = exp(-lambda*d(i,s)) * n_i/(n_i+2)`.
The proposal uses a weighted **average** of observed log-costs plus a penalty
for unsupported cells. This is an *acoustic-inspired heuristic*, not a physical
interference model: phase, reflections, dispersion and sound waves are not
simulated, and source amplitudes are not interpreted as correctness proofs.

Both methods require at least 5% predicted benefit to displace `manual`,
and require explicit `allowed` plan masks. A cached policy decision is
prepared before online use and invalidated on new evidence or epoch updates.

## Test design

Three fresh Python interpreter processes, a training split of sizes 128/1024
and a holdout of 256/512; changes 1/4, reads 1/8, width=64, ticks=2, 4 paired
rotated samples in the final protocol. Training and holdout sizes map to
distinct log-grid cells. Each candidate's end-to-end checksum is gated
against `dense` before timing. Selections are frozen before any holdout
route measurement. Online comparisons include dispatch overhead and whole
workflow construction/patching/reading. Offline training and frozen-policy
preparation timings are reported separately. The hindsight oracle is never
used for selection. Fixed `manual` remains the strong control.

### Local results (CPython 3.13.5, Linux; 3 fresh processes)

| Policy | Median manual / policy over 8 holdouts | Confirmed >10% gains | Confirmed >10% slowdowns |
|---|---:|---:|---:|
| Ant-Memory | 0.904x | 0 | 0 |
| Light (compiled min-plus) | 0.863x | 0 | 2 |
| Sound (damped evidence) | 0.905x | 0 | 3 |

In the tested grid, the light/sound planners selected `manual` for nearly
all held-out cells. Thus the observed slowdowns are primarily selection
and timing overhead, **not evidence of better route selection**. Results
vary with sample count, CPython process and host; these are exploratory and
do not justify promoting any router.

The committed, auditable summary and frozen decisions are in
`evidence/wave-memory-summary-local-py313.json`; complete session-level raw
samples for both protocols ship with the downloadable research archive.
The earlier reference-Dijkstra run used a different number of paired samples:
**do not attribute the entire between-run timing change to the optimization**.

Independent microbench on one warm host, a 3-hop field and 12 source cells:
Dijkstra reference ~141 microseconds/lookup vs compiled direct min-plus
~19 microseconds/lookup (median per-call, eight batches of 300 calls).
This tests *policy calculation only*, not Python application throughput.
No physical CPU cache, light-speed or acoustical conclusion follows.

## Publication gate

`alexpy.sum_squares()` unchanged. This experiment is opt-in only.
Before considering runtime deployment, measure amortized offline costs,
confidence intervals, workload drift and independent second-machine results.
For workload parameter grids with variable edge weights the compiled shortcut
must be disabled in favor of the generic shortest-path reference.
