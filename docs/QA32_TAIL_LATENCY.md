# QA 3.2 — Tail-Latency Stability (protocol frozen before new measurements)

Base: `7f8cc3ea314224c6af27473bd0b87d53059fd3b4`. QA 3.1 audits 12 artifacts at the earlier SHA. Two further macOS Python 3.13 artifacts, 15096216818 (run 45775923611) and 15095893287 (run 45775915115), give historical 29-test/batch-1 adaptive/direct p99 ratios 1.160 and 0.904 (300 samples each; medians 1.015 and 1.018). This weakens the earlier repeated-tail signal; it is not proof of no regression.

## Question
Does a frozen route-matrix fallback introduce reproducible end-to-end tail latency, or are earlier excursions dominated by runner noise?

## Frozen experiment
* Same exact QA semantic oracle and report contract as QA 3.0. Compare direct C tree, the same tree selected through a deliberately frozen equal-cell matrix, and bitset. Equal cells are a control to force the observed fallback, not trained evidence.
* Historical case (29 tests, batch 1, seed 8101); fresh seeds 320031 (29/1), 320033 (31/1), 320039 (29/8). Fresh cases are held out from previous QA30 training; they remain synthetic.
* Five sequential independent subprocesses per case, 2000 full replays per route per process, 20 warmups, all six execution orders shuffled once per six-pair block. Store every raw time, order, p95/p99 (nearest rank) and lookup measurement.
* Verify independent scan reference for every group before timing, and compare report fingerprints during timed replays. No tuning on this family. No live migration.
* Separate three fresh memory workers per route/case. Record tracemalloc peak and process-lifetime maximum RSS before/after on Linux/macOS (not engine RSS). Windows RSS is explicitly unavailable. Imports and fixture construction contribute to RSS. No memory measurements inside latency timing.
* Target macOS Python 3.13, Ubuntu and Windows Python 3.13 as platform controls. Machine-specific results are not pooled. All timing evidence is advisory.
* A repeatable candidate is p99(dispatch)/p99(direct)>1.05 in ALL five full-protocol processes. This is a conservative screening rule, not statistical proof. Report median and p95 even if p99 passes. No promotion or code change based solely on a candidate.
* Compare bitset descriptively; do not train on holdout measurements. A negative result is valid evidence.

Run from repository root after compiling extensions:
```sh
PYTHONPATH=. python tests/qa32_tail_latency.py --output qa32-results.json
```
CI uses an explicit project-root PYTHONPATH. Raw JSON is uploaded per runner. Tests exercise full semantics with a six-pair smoke, never a timing assertion.
No changes to native kernels, dispatch policy, defaults, or main. Do not merge this draft.
