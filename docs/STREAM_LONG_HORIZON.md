# AlexPy 1.9 — measured, paid, long-horizon stream experiment

## Purpose

The v1.8 research measured gains on streams of at most 128 cycles and projected
an amortization threshold. This follow-up measures **complete streams** of
8,192 cycles using frozen route decisions, with the previous real calibration
wall time charged once. It must never mistake a projected threshold for an
observed paid payback.

## Critical cross-hardware finding

The archived v1.8 evidence identified an **Intel Xeon Platinum 8573C**.
The later local environment reports **AMD EPYC 9V74**. Attempting to reuse the
Intel route choices was **correctly rejected** with `host mismatch: processor`.
We independently trained again on AMD; those choices and timings are stored in
`evidence/stream-horizon-amd-calibration.json`. This is a new single-CPU-profile
calibration; the Intel training is never imported into the AMD decisions.

The profile gate checks reported platform, architecture, CPU, Python version
and interpreter. It is **not** cryptographic hardware attestation, and machine
names do not establish matching thermal or virtualization conditions.

## Control and accounting

- Training: sizes 128, 512 and 2048, three independent interpreter processes,
  each measuring four exact range-query algorithms, then compiling a table for
  held-out sizes 384 and 1536. Freeze SHA-256 before the longer-stream tests.
- Evaluation: 128 and 8,192 cycles of exactly the same deterministic workload
  specification used in the v1.8 study; counts 1,8 updates and 1,8,32 range
  queries, short and wide windows, two ticks per workload. *The horizons were
  chosen after inspecting earlier projections*, so this is a follow-up, not
  a fresh blinded benchmark. Held-out sizes were already used in 1.8.
- Three separate Python processes, two paired repetitions for each policy.
  Paired order alternates. Compare each process separately and preserve raw
  per-policy nanosecond samples.
- The direct implementation independently recomputes the **whole continuous
  stream checksum** before any measurements. Both policies must match it or
  the worker aborts.
- Both alternatives pay for data layout construction, every real rebuild,
  every update, every query and their policy lookup. Schedule preparation and
  the independent correctness witness are symmetrically outside the timer.
- Offline calibration is a **historical measured cost on AMD**, charged once
  against the tested savings. It is not re-executed for every long-horizon
  worker. A stronger break-even gate requires each of three workers' sum of
  savings across both sizes to cover this charge, not merely the median.
- Runtime environment is not pinned to exclusive hardware. CPU frequency,
  neighboring workloads, memory allocation and thermal conditions may differ.
  Gains on another CPU must be independently measured; the Linux, Windows,
  macOS CI smoke artifacts are not numerical proof of portability.

## Reproduction

First build C extension and run complete correctness suite:

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests -q
```

Calibrate separately on the current host (never carry timing evidence from
another CPU over the environment gate):

```bash
python -m alexpy.stream_bench --train-sizes 128,512,2048 \
  --holdout-sizes 384,1536 --changes 1,8 --queries 1,8,32 \
  --spans both --ticks 2 --pairs 3 --sessions 3 --cycles 1,8 \
  --reps 2 --device-tag amd-epyc-newcalibration-v19 \
  --json evidence/stream-horizon-amd-calibration.json
```

Then measure real long streams on exactly that CPU/runtime profile:

```bash
python -m alexpy.stream_horizon \
  --baseline evidence/stream-horizon-amd-calibration.json \
  --cycles 128,8192 --repeats 2 --sessions 3 \
  --output evidence/stream-horizon-amd-128-8192.json
```

If host identity changes, the script stops instead of silently importing a
foreign device's learned performance rankings.

## Result and decision

The raw result includes `first_observed_paid_cycles`, strict per-process net
savings and the original calibration charge. Any conclusion must quote these
**measured** numbers and must not enable automatic acceleration of general
Python execution. This is a specialized range-query workload only.

## Independent GitHub CI artifacts actually inspected

Downloaded six distinct `stream-economics.json` artifacts from the two GitHub
Actions runs 44969437669 and 44969442135 for v1.8, each with three independent
interpreter processes **on its own runner**. See
`evidence/stream-cross-host-ci.json` for the per-run raw sample excerpts,
reported processors, frozen route distributions and source JSON SHA-256.

| Hosted environment | Training wall-time in two runs | First measured paid payback (<=16 cycles) |
| --- | ---: | --- |
| Ubuntu, Xeon and EPYC | 324 ms / 302 ms | neither |
| Windows, reported AMD | 618 ms / 752 ms | neither |
| macOS, ARM | 437 ms / 274 ms | neither |

All six runner studies used a **smaller smoke workload** with four held-out
routing contexts (`direct` for three, `prefix` for one) and horizons 1, 4, 16.
Some individual online ratios exceed 1.0 and others fall below 1.0. There is
no defensible average across these distinct hardware and OS workloads, and
no evidence that payback was reached within the tested horizon. These are
actual retrieved artifacts, not simply successful CI job statuses. The full
original ZIPs are preserved separately in the research bundle.

## Actual AMD EPYC results (three independent interpreters)

Calibration measured on AMD: **2.550 seconds**. Route choices: direct=16, blocks=3, fenwick=0, prefix=5.

| Cycles | Online gain over both sizes (median across processes) | Paid net after calibration | Paid in every process? |
| ---: | ---: | ---: | --- |
| 128 | 65.6 ms | -2484.7 ms | no |
| 8,192 | 4945.8 ms | +2395.5 ms | yes |

The **first tested, paid** horizon is 8,192 cycles. It is not a measured minimum threshold, because intermediate horizons were not exhaustively tested.

| Size | Cycles | Baseline online time | Memory online time | Baseline / memory |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 85.1 ms | 77.3 ms | 1.100× |
| 1,536 | 128 | 230.3 ms | 164.5 ms | 1.400× |
| 384 | 8,192 | 5620.3 ms | 4869.8 ms | 1.154× |
| 1,536 | 8,192 | 15230.9 ms | 10867.8 ms | 1.401× |

The recorded checksum is identical between all candidate policies and the full independent direct reference for each process/horizon. Raw nanoseconds and layout switching costs are in the full source. These results may be distorted by scheduler contention; reuse on other CPUs is rejected.
