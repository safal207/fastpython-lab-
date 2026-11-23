# AlexPy × Bardo 2.8 — failure bitmap vs native segment tree

**Research only; no automatic promotion.** Parent research head: `5188804940ef45a44144eb53adbd83f41a15aaeb` (2.6). The 2.7 synthetic density stress experiment was archived locally but not verified on GitHub before this development; its source and evidence are included for preservation in the same publication.

## Question and explicit alternatives

Can an exact packed two-level **native failure bitset** save memory while preserving transactional semantics and outperforming a C segment tree on the entire QA workload? No new learning rule; Bardo is an inspiration for compact representations, not a benchmark guarantee.

- `bitset`: `alexpy._qa_bitset.WitnessBits` over unchanged `NativeBatchQA`, 64-bit failure words and a directory of nonempty 64-word groups; first-K failure IDs found in increasing order.
- `c_tree`: existing `alexpy._qa_witness.WitnessTree` over *identical* `NativeBatchQA`; exact first-K failure IDs.
- `set`: strong ordinary Python `set` plus `heapq.nsmallest(K)` over *identical* `NativeBatchQA`.

Every route receives identical `QAEvent` objects, identical batches and identical report schedules. Timed regions include state creation, Python validation, C-batch application, witness staging and updates, all report materializations and first-K reads. Full independent scanning of true state is done before timing and at several checkpoints. Exact transitions, witnesses, summary snapshots and report digests must match. No tree implementation is promoted or tuned using this evaluated family.

The bitmap stages its private `bytes` patch *before* the authoritative C-batch commit. It validates the entire patch before mutation; then commit has no dynamic allocations. Thus an invalid input rejected by `_prepare`, or a failure in native QA before commit, does not modify the witness index. `verify()` checks both the bitmap's summary directory and the independently dumped QA statuses.

## Performance protocol

Reuses the **previously inspected** seven 2.7 density scenarios (0%, 55%, 85%, 90%, 100% failures; status reversals; hot retries; frequent witness reporting). This is a regression/comparative family, **not a fresh statistical holdout**. Three independent Python interpreters, three alternating-order paired samples per case, on one Linux CPython 3.13.5 **Intel Xeon Platinum 8573C** runner. The 2.7 AMD EPYC data is a separate experiment; do not pool its durations with Intel.

| Case | Bitset / C tree speed | Bitset / Python set speed |
|---|---:|---:|
| dense-mostly-fail | 1.080× | 1.323× |
| dense-wide | 0.953× | 1.229× |
| reversible-flips | 0.808× | 1.093× |
| hot-retries | 1.103× | 1.261× |
| frequent-diagnostics | 0.940× | 4.516× |
| zero-failures | 0.958× | 1.170× |
| all-failures | 0.944× | 1.149× |

A value over 1 means the **bitset is faster**. Medians of per-interpreter median timing ratios. The bitset wins *only two* of the seven scenarios against the C tree by median, and no performance superiority over C tree is supported. Its benefit is lower memory and strong performance relative to the Python-set control.

### Memory and RSS boundaries

Three isolated processes per engine/shape. `tracemalloc` records peak Python-tracked allocation; Linux `/proc/self/status` records instantaneous RSS before/after the workload and the process high-water mark. **A before/after RSS delta is not a complete RSS peak**; repeated trials and Python runtime allocation can shift results.

| Case | Bitset tracked peak | C tree tracked peak | Python set tracked peak | Bitset RSS delta | C tree RSS delta |
|---|---:|---:|---:|---:|---:|
| dense-wide | 870 KiB | 1011 KiB | 1492 KiB | 1120 KiB | 1216 KiB |
| frequent-diagnostics | 125 KiB | 158 KiB | 312 KiB | 76 KiB | 112 KiB |
| zero-failures | 126 KiB | 156 KiB | 157 KiB | 96 KiB | 116 KiB |

All numbers are medians, rounded. Cross-host native allocations or Windows RSS must not be inferred from these Linux measurements.

## Genuine QA-report compatibility

Generated `evidence/alexpy-real-pytest-junit.xml` with a **real local pytest run** on actual AlexPy unit tests (not a fabricated fixture):

```bash
python -m pytest -q tests/test_native_bitset.py tests/test_native_transition_tree.py tests/test_native_witness_stress.py tests/test_junit_replay.py --junitxml=evidence/alexpy-real-pytest-junit.xml
python -m alexpy.junit_replay evidence/alexpy-real-pytest-junit.xml --batch 7 --json evidence/bitset-real-junit-replay.json
```

22 distinct executed test cases were ingested across 4 suites (0 FAIL, 0 SKIP). All three index routes produced identical snapshots, witness lists and report digests. The original XML SHA-256 and test counts are saved in replay and summary records. This small all-pass report validates integration/semantics but **does not prove production performance or failure localization on real failing reports**. Independently authored malformed/FAIL/SKIP/retry XML fixtures exercise those parser paths in unit tests.

## Reproduce and CI

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.witness_bitset_bench --smoke --sessions 1 --pairs 2 --json bitset-smoke.json
python -m alexpy.witness_bitset_bench --sessions 3 --pairs 3 --json evidence/bitset-full-local-py313.json
python -m alexpy.witness_bitset_bench --memory --sessions 3 --json evidence/bitset-memory-local-py313.json
python -m alexpy.junit_replay evidence/alexpy-real-pytest-junit.xml --batch 7 --json bitset-junit.json
```

`evidence/bitset-summary-local-py313.json` links raw timing, memory, genuine pytest XML, JUnit replay and smoke through SHA-256. GitHub Actions builds the bitset extension and checks bounded smoke/JUnit replay on Python 3.13 for Linux, macOS and Windows, recording artifacts independently. A green CI confirms correctness and smoke completion, not that the bitmap accelerates other hardware.

## Decision / next falsification

**Do not enable bitset or C tree by default.** Continue with strong existing C batch QA. The C segment tree remains faster for most of this reused family, while the bitmap saves memory across measured shapes. A future new workload must preregister the trade-off, include genuine multi-run QA reports with actual failures, measure p95/p99 and external RSS with a separate monitor, and compare at least one specialized ordered native failure vector against the bitmap and tree.

The failed drift-sentinel policy from 2.6 remains independent and unpromoted.
