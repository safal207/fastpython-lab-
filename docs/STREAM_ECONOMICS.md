# AlexPy 1.8 — real long-lived stream economics and cross-host evaluation

## Question and advance-registered comparison

Can a pre-trained, opt-in route memory earn back its **full calibration cost**
when successive query segments share one mutable input array? Compare:

1. `simple`: the strong fixed *rule* used in 1.6–1.7; no learning.
2. `stable`: a conservative offline router trained only on **other sizes**.
3. `direct`: one fixed unindexed algorithm as an additional negative control.

All paths produce `sum(segment), sum(x*x for x in segment)` after updates.
No Python interpreter or uninstrumented program is sped up automatically.

## Protocol

- Train on `size=128,512,2048`, hold out `size=384,1536`. Coarse state
  coordinates (log2-sized buckets) must be disjoint as well.
- Train all four verified backends (`direct`, `blocks`, `fenwick`, `prefix`)
  with alternating order and three **distinct interpreter processes**.
  The conservative selector requires a replicated advantage. No holdout
  timestamps are read before the route table is frozen and hashed.
- The same deterministic schedule traverses a variety of changes, query
  counts (1, 8, 32) and spans (short/wide), in repeated *cycles* over ONE
  **persistent** mutable backend. The source is never reset between segments.
  Update values include a unique epoch term so cycles are not trivial no-ops.
- A route change creates a **new real** backend from `old_backend.data` and
  includes its construction time (also reported separately). Keeping a route
  retains its existing representation, never rebuilding gratuitously.
- Input schedule and query lists are generated **before** timing for both
  algorithms, eliminating unrelated input-creation noise. Policy lookup,
  initial index creation, backend update, all queries, real layout rebuilds,
  and checksum accumulation are included in the online timed section.
- A separate continuous `Direct` witness checks the complete stream checksum;
  all candidate backends must exactly match. Individual semantic APIs had
  also passed differential tests in prior releases.
- Evaluate in three fresh interpreter processes, paired order rotated and
  reversed across repetitions. Compare *within-process* measurements,
  report raw samples, do not pool absolute times across environments.
- Charge the **measured wall time** of all three calibration processes,
  serialization and freezing the decision table exactly once to the trained
  policy. Evaluation-worker startup, scheduling and reference checks are not
  charged to either online policy. The on-device amortization comparison is
  `sum(median(simple stream) - median(stable stream)) - calibration_wall` for
  each measured cycle count, combining the two held-out array-size streams.
- A *verified paid break-even* requires net savings >=0 **and** positive
  total online savings for **every** independent process. If not observed,
  report `null`. A threshold obtained by linear extension of the longest
  tested horizon is a conjecture only, not a measured result.

## Local run — CPython 3.13.5, Linux, Intel Xeon Platinum 8573C

Source file: `evidence/stream-economics-local-py313.json`. Three training and
three independent holdout worker processes, three A/B repetitions. The
calibration wall time was about **3.18 seconds** on this host.

| Cycles per size stream | Sum of online savings, 384 + 1536 elements | Paid savings after calibration |
| ---: | ---: | ---: |
| 1 | 0.5 ms | negative |
| 8 | 4.6 ms | negative |
| 32 | 22.8 ms | negative |
| 128 | 84.1 ms | negative |

At 128 cycles, for `n=384`, the trained route was ~**1.19x** faster online;
for `n=1536`, ~**1.35x** faster online, relative to the strong rule. Both
ran fully with rebuilds and query checks; the trained memory switched route
**511** times and the simple rule **255** times per 128-cycle size stream.
Even so, the savings **did not** recover the 3.18 s calibration cost. A
simple linear extension of the last horizon is around **4,800 cycles**
for this workload mix and device, but this is only a *projection*. It has
not been reproduced or shown to be a break-even threshold. A separate
512-cycle high-cost attempt did not finish within its bounded run budget;
no result from that attempt is presented as evidence.

Individual process measurements, route choices, build times, and frozen
SHA-256 are recorded in the JSON, not just aggregates. One very short
`n=384`, 1-cycle run had a negative gain in one process, showing noise
and the danger of overinterpreting small workloads.

## Actual different hardware/OS — CI experiment

At Python 3.13 each OS in the test matrix separately trains and evaluates
its own memory on the **same workload specification**, with a unique
`device_tag` incorporating the GitHub run id. The new jobs run on hosted
Linux, Windows and macOS machines. Each writes `stream-economics.json`
containing host name/architecture/processor when available, timestamps,
independent process samples, calibration and a paid net result. Artifacts are
published as `alexpy-stream-<os>-py313-<run_id>`.

Do not assume these are controlled comparisons of identical hardware, nor
transfer memory learned on one runner to another. CI outcomes will become
cross-environment *measurements* only after all three runner jobs actually
finish and their JSON artifacts are reviewed. Existing Linux-only results
are not proof for Windows/macOS or another physical CPU.

## Threats and limits

- A cost-focused synthetic workload, fixed deterministic query/update
  distribution, not a representative sampling of arbitrary user programs.
- `min_processes=3` are **independent interpreters on the same host**, not
  three physically independent computers; CPU load, cache, governor and VM
  contention can correlate outcomes.
- Rebuilt algorithm representations are Python lists and indexes, not a
  zero-copy layout. Both alternatives face the same mutations and queries.
- Planning model trained on different sizes may not extrapolate; unsafe or
  unverified routes fail closed. No dynamic online re-learning is allowed.
- Calibration must be rerun for new CPU/interpreter/OS profile and whenever
  input distribution materially changes. Device identity relies on an
  operator-supplied tag; architecture alone is not sufficient.
- Stream preparation is excluded symmetrically; real application data
  collection can add overhead not measured here.
- Longer horizons, memory pressure, CPU-energy use, and hardware counters
  remain untested. Cross-runner telemetry is advisory and never auto-promotes.

## Reproduce

```bash
python -m unittest discover -s tests -q
python -m alexpy.stream_bench \
  --train-sizes 128,512,2048 --holdout-sizes 384,1536 \
  --changes 1,8 --queries 1,8,32 --spans both \
  --ticks 2 --pairs 3 --sessions 3 --cycles 1,8,32,128 \
  --reps 3 --device-tag local-benchmark-device \
  --json stream-economics.json
```

This is a **research-only optional module**. The public `sum_squares` path,
existing graph algorithms and error semantics are unchanged.
