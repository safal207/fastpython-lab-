# AlexPy × Bardo 2.6 — native witness tree + calibration drift falsification

**Status:** exploratory only. No changes to `NativeBatchQA`, `qa_native.py`, the default route, or any Bardo repository. Separate opt-in experiment, no automatic promotion.

## Sources and hypothesis boundaries

- BardoCompute pinned software reference: `f50490e8194f0e5c7aeeba251f2575eb8155ed54`.
- Bardo LOGOS ordered tree (`58dbf365235d03b0e78abe54551546d220f8c0f8`) and Calibration Trust (`f50490e…`) are inspiration, not code or performance inherited by AlexPy.
- The preceding Python tree on AlexPy `7b97ceb31e609d6b93da857941375ff59654b6a4` lost to a Python set on all prior workloads. The Bardo-inspired route was intentionally not promoted.
- Hypothesis A: an exact C segment tree for *failure locations only* can beat a strong Python set atop the **same unchanged** Native QA accumulator in workflows with incremental reports.
- Hypothesis B: a paid EWMA of observed calibration changes can choose probe cadence better than a fixed interval trained on separate data. Neither oracle labels nor future phase boundaries are available to policies.

These are distinct experiments: native tree timings are physical CPU measurements; the drift study uses **synthetic decision-cost units**, not wall-clock time or evidence of faster Python.

## A. Compact native failure witness tree

The separate `_qa_witness` C extension holds a byte flag per test and a `uint32_t` count per segment-tree node. It does **not** copy the authoritative native statuses, durations or suite totals; it stores only whether each test is currently `FAIL`. Witness queries return the exact first K failing IDs in ascending index order. All merges are exact; no collision-prone digest or lossy top-K replacement is used.

Per batch:
1. Python `_prepare` validates **every** event, including overwritten events, and produces the latest state for each test ID.
2. `WitnessTree.stage` validates the complete candidate patch and allocates immutable serialized updates.
3. The existing `NativeBatchQA.apply_batch` validates and atomically commits the authoritative QA transaction.
4. `WitnessTree.commit` fully validates the staged byte buffer **before** changing any tree node. Its update phase performs no Python allocations or attribute lookups.

The witness mirror is a research-internal companion to an authoritative native state, not an independent public transaction log; direct misuse of staging methods is outside the wrapper contract. Failures in Python validation or in native QA before commit leave the witness index unchanged. The `verify` method compares all failed flags, count and test-suite snapshots with an independent scan.

Controls: the old Python LOGOS-inspired tree and the **strong** Python `set` + `heapq.nsmallest(K)` witness index use the exact same `NativeBatchQA`. The benchmark measures *construction + all event processing + report materialization + witness extraction*. Three independent Python processes, three rotating-order paired samples each, four reused v2.5 synthetic workload families; correctness checks are untimed.

### Local Linux Python 3.13.5 / AMD EPYC 9V74

| Workload | C tree / strong Python set (speed ratio) | Old Python tree / C tree (time ratio) |
|---|---:|---:|
| small-hot | 1.489× | 6.46× |
| medium-sparse | 1.327× | 16.86× |
| large-sparse | 1.171× | 21.25× |
| query-heavy | 1.947× | 12.82× |

Medians across three independent process-level ratios. Each process has three paired measurements. **These are reused, already-inspected workload shapes, not an independent blind validation family.** The C index is faster on all four on this host, but is not yet proven better on other CPU/OS profiles or real production test logs.

### Isolated-process memory

`tracemalloc` peak in KiB; medians of three isolated process runs per route:

| Workload | Native C index | Native QA + Python set | Native QA + Python tree |
|---|---:|---:|---:|
| medium-sparse | 151.8 | 184.8 | 1329.3 |
| large-sparse | 569.7 | 591.9 | 4483.7 |

These are *traced allocations*, not resident set size. The C tree's `PyMem_Calloc` buffers are tracked by the chosen allocator, but external unmanaged memory or allocator fragmentation is not fully represented.

## B. Drift Sentinel — independent negative result

The new `DriftSentinel` sees only paid current-route observations; it never receives hidden regime labels, complete route histories or future changes. Its fixed, untuned-on-holdout heuristic is a bounded `EWMA` of newly observed changes per elapsed tick, with `alpha=.35`, hazard floor `0.00005`, probe-cost 2, stale loss 7, missed gain 3 and allowed intervals 8–128 ticks. These values are engineering assumptions and not calibrated CPU timings.

Controls include the **unchanged** Bardo-inspired v2.5 `probe_decision` policy, fixed intervals 8/16/32/64/128, and the strongest fixed interval **selected only on a training family** of 12 seeds × 5 hidden regimes. The policy code is frozen for evaluation on a disjoint 24 seeds × 5 regimes (120 sequences, 1536 ticks each). Hidden regimes include stable, abrupt change, reversal, intermittent bursts and mixed random transition rates. The controller is not told which regime is active, and an even number of unobserved flips can evade detection.

**Measured result (synthetic units; lower is better):**

| Policy | Heldout mean regret cost |
|---|---:|
| Best train-selected fixed interval 16 | **454.15** |
| New Drift Sentinel | 797.71 |
| Prior Bardo-inspired trust probe | 1221.03 |
| Fixed 8 (control) | 516.98 |
| Fixed 128 (control) | 1047.58 |

The new policy wins in some stable conditions, but loses badly around short changes and bursts. Its overall cost is 1.76× the strong training-selected fixed policy. **Reject deployment, adaptive probe promotion and claims of calibrated speed benefits.** The family is seed-heldout but not an independently preregistered production benchmark; this is research falsification, not a generalized impossibility proof.

## Commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.native_tree_bench --smoke --pairs 1 --json c-tree-smoke.json
python -m alexpy.native_tree_bench --sessions 3 --pairs 3 --json c-tree-full.json
python -m alexpy.native_tree_memory_audit --sessions 3 --json c-tree-memory.json
python -m alexpy.drift_sentinel_bench --json drift-heldout.json
```

The complete JSON measurements, SHA-256 checksums, exact host profile, and test count are in the `evidence/` folder. CI repeats smoke tests on Linux, Windows and macOS but is not equivalent to a new full performance validation on all those platforms.

## Promotion gates / next falsification

1. Retain the baseline C QA accumulator, unchanged and enabled only where already explicitly selected.
2. Validate the C witness index against the same strong Python set on *fresh* workload shapes and distinct actual CPUs, including small, dense, failure-heavy and empty-event workloads; report time **and** RSS.
3. Stress-test failures and provenance matching (epoch / state reconstruction) before exposing external mutation APIs.
4. For drift detection, measure probe surprise on a fresh benchmark with separate train/tuning/test families, and compare to a training-selected fixed control including p90 and worst-case loss. Do not tune on this heldout family.
5. A CI pass alone does not imply performance superiority or product readiness.
