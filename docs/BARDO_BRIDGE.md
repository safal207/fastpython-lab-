# AlexPy × Bardo: Transition-Memory Tree and Calibration Trust — opt-in research

**Source boundary:** AlexPy source: PR #1, branch `feat/graph-search-mirrors-v02`;
Bardo reference: [`f50490e8194f0e5c7aeeba251f2575eb8155ed54`](https://github.com/safal207/BardoCompute/tree/f50490e8194f0e5c7aeeba251f2575eb8155ed54), with a separate learning track at [`0b9da9e61eed562797473c8a841902247a2aa946`](https://github.com/safal207/BardoCompute/tree/0b9da9e61eed562797473c8a841902247a2aa946).
Bardo's `main` is an index; it does **not** include the reference code. The new modules are independent experimental implementations based on the publicly documented ideas, not a Bardo hardware integration, licensing change, or promotion.

## Questions and preregistered (pre-evaluation) controls

1. Does a binary ordered summary retaining the **first four exact failing test IDs** reduce end-to-end QA dashboard costs when compared with the **same native C batch engine plus a Python failure set**? Other fields (all four status counters, total duration) must be exact; no lossy hash can stand in for evidence. Compare identical input, same C engine, same snapshots, same report schedule, same witness output, randomized retries, exact reference scan at checkpoints, and report allocated memory separately.
2. Can a Bardo-inspired time/drift-dependent trust gate beat fixed probing under a known synthetic drift family **without seeing hidden future states**? Compare fixed intervals 8 and 64 and per-step probing. Price probes; report negative results. A simulated gain is **not** measured processor throughput. Calibration model: product of sample, age, drift and forecast-error confidences, zero on host mismatch; probe cadence approximated by `sqrt(2 × probe cost / stale regret rate)`.

No modification of the default `NativeBatchQA`, native C or existing production paths. A route is never promoted merely because a model or another repository reports benefit.

## Exact test implementation

- `alexpy/transition_tree.py`: order-preserving binary aggregation, staged O(k log n) changed-leaf recomputation before native commit; exact sorted first-K failing IDs, arbitrary-precision duration sums.
- `alexpy/calibration_gate.py`: strictly past/present inputs and conservative fail-closed hardware isolation. Benchmark formula is a hypothesis rather than an empirical calibration.
- `alexpy/bardo_bridge_bench.py`: independent reference-scan witness and equal-native opponent; construction, validation, slicing, update, snapshots and witness extraction are timed. Training or event generation are **not** timed and are identical across routes. Paired alternating-order medians, independent processes, separate `tracemalloc` measurement.
- QA events are synthetic, generated using a fixed seed by `alexpy.qa_batch_bench.generate`. All benchmark scenarios use an opt-in compiled native QA extension.

```sh
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.bardo_bridge_bench --smoke --pairs 2 --json bardo-smoke.json
python -m alexpy.bardo_bridge_bench --pairs 4 --sessions 3 --json bardo-full.json
python -m alexpy.bardo_bridge_bench --memory-worker native_set --memory-case large-sparse
python -m alexpy.bardo_bridge_bench --memory-worker native_tree --memory-case large-sparse
```

## Local evidence: 2026-11-23, Python 3.13 / Linux

**Every result below is for the stated local interpreter and experiment, not another host or Bardo hardware.**

| Full QA case | Tree throughput relative to equal-native set | Status |
|---|---:|---|
| small-hot: 256 tests, 1800 events, batch=8 | 0.233× | slower |
| medium-sparse: 4096 tests, 3200 events, batch=64 | 0.079× | slower |
| large-sparse: 16384 tests, 4500 events, batch=64 | 0.051× | slower |
| query-heavy: 4096 tests, 3200 events, report every batch ×8 reads | 0.162× | slower |

Median of **three independent process ratios** with four paired measurements each. Raw figures: `evidence/bardo-bridge-full-local-py313.json`.

| Case | Equal-native failure set, `tracemalloc` peak | Same native engine + binary tree, peak |
|---|---:|---:|
| medium-sparse | 202.1 KiB | 1,573.3 KiB |
| large-sparse | 618.1 KiB | 4,756.6 KiB |

Three independent isolated processes each, no mixing with timed runs. This is **tracked Python allocation peak**, not RSS or the complete physical memory consumed by C. Raw data: `evidence/bardo-bridge-memory-local-py313.json`. The machine-readable digest/host record is in `evidence/bardo-bridge-summary-local-py313.json`.

## Calibration trust: independent symbolic experiment, NOT CPU benchmark

Synthetic net cost is `probe_count × 2` plus per-step marginal result: using the specialized route gives `-3` while advantageous and `+7` after it becomes stale; the safe route is `0`. 1024 deterministic steps. Abrupt shift at tick 512; reversal at tick 768. Hidden environment is used only for paid probe revelations and downstream cost accounting; the decision rule receives no future labels. The control never uses real QA timings. Bardo's calibration trust equations are intentionally not assumed optimal for this workload.

| Environment | Fixed 8 | Fixed 64 | Probe every step | Adaptive trust |
|---|---:|---:|---:|---:|
| Stable | -2816 | -3040 | -1024 | -3042 |
| Single shift | -1280 | **-1504** | +512 | -946 |
| Reversal | -2048 | **-2272** | -256 | -1708 |

Lower is better. The adaptive gate saves one probe in the stable simulation but **loses materially** to fixed64 when the environment changes. The current heuristic is NOT qualified for AlexPy route memory; drift evidence alone arrives too late and the confidence fallback can reject useful work. These deterministic scenarios are a debugging protocol, not independent statistical proof.

## Scientific limits and next falsification

- Tree is a Python implementation. It can preserve exact information but does not prove faster computation or reduced data bandwidth; its memory and end-to-end time *regressed*. Do not replace the proven native batch path.
- The base C engine's transaction is atomic. The experimental tree pre-stages all required allocations; the independent Python-set control does not claim full process-level atomicity under arbitrary `MemoryError` after native commit.
- `first four failing IDs` is a deliberately bounded witness API; it is not complete failure diagnostics. The complete statuses remain accessible through the native engine.
- The tree's full root counters cover **global**, not per-suite totals; suite totals come from the unchanged C engine.
- The calibration simulator compares synthetic decision policies, not physical clock cycles. Both its drift onset and benefit/loss constants are a test fixture, never injected into model decisions.
- A next experiment could build an **optional** compact C tree and compare against both a Python failure set and a sorted-index control, with memory/RSS gates; independently redesign calibration drift detection from historical error and re-test on unseen regimes.
- Do not copy Bardo's hardware performance claims into AlexPy. No physical FPGA/CPU comparison has been performed.
