# AlexPy 0.6 — memory hierarchy and cache-boundary experiment

**Status:** independent opt-in hardware microbenchmark, not a CPython optimization and not evidence that the whole Python interpreter is faster.

This experiment tests a narrow causal *hypothesis*: increases in the working set can change native memory-access latency, and reported cache capacities may help choose follow-up sizes. It does **not** establish cache-miss causality without independent hardware counters and environmental controls.

## Exact protocol

Install the package, including the independent C extension:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.memory_lab \
  --sizes-kib 4 32 256 2048 16384 65536 \
  --epochs 2 --repeats 5 --target-accesses 750000 \
  --hardware-counters --json memory-lab.json
```

A smaller portable CI smoke run uses `4 64 512 4096` KiB and fewer samples. The Python public `alexpy.sum_squares` dispatch stays unchanged.

### Workloads

- **Random dependent pointer chasing:** native C creates one deterministic cycle over a Fisher–Yates shuffled `uint32_t` index array. Access `cursor = successor[cursor]` carries a dependency into the next address and is hard for a conventional CPU to prefetch. With `N` nodes, exactly `N` steps return to index 0, checked before the timer starts. `ns / access = median(walk_duration) / number_of_steps`.
- **Sequential indexed read control:** native C reads the **same** array sequentially with volatile loads and a checksum. An independent mathematical oracle verifies `N(N-1)/2` for one pass. We report `index_bytes_read / observed_time` as **effective indexed throughput**. This is not measured DRAM bus bandwidth.
- Buffer allocation, deterministic permutation creation, and first-touch warmup are outside timed loops. All probes are native C and are independent of the AlexPy `sum_squares` optimization.
- At each footprint, repeat paired native operations in alternating AB/BA order; traverse footprint sizes in reverse on alternate epochs. Raw times and epochs are preserved.
- Reported L1/L2/L3 capacities and shared-CPU lists are read from Linux `sysfs` when available. They are **metadata** only, and on shared/virtualized systems may not describe usable cache capacity.

### Graphs, time and derivatives

Each cell has coordinates `(working_set_bytes, epoch, wall_timestamp_ns, access_pattern)` and measured latency/throughput. Within an epoch, neighboring sizes produce **spatial** edges. Reobserving an identical size in the next epoch produces a **temporal** edge; different sizes are never treated as adjacent in time by measurement order alone.

For sizes `b_1, b_2` and dependent-read latency `L(b)` the bounded first finite difference is:

`[ln L(b_2) - ln L(b_1)] / [ln b_2 - ln b_1]`.

This is the slope of an observed curve, **not** a physical derivative of a continuous system and not a measured fractal dimension. A crossing of a reported cache boundary is recorded as a **hypothesis** only.

### Hardware miss counters — strict availability gate

`--hardware-counters` attempts Linux `perf_event_open` for: L1D read misses, LLC read misses, and generic cache misses. The calls can fail due to unavailable events, system policy, virtualization, or permissions; when they do, the JSON contains `count: null` and an `UNAVAILABLE` status, **never zero or an estimated miss count**. Measurements use separate runs; do not subtract counts to synthesize L2 misses. Portable L2 read-miss events are not defined by the generic Linux API, so L2 is marked `UNAVAILABLE_MODEL_SPECIFIC_EVENT_REQUIRED` until a named validated vendor-specific event is added.

Direct hardware counters are optional and cannot block correctness CI. Run on a dedicated, pinned bare-metal host with access to validated counters for stronger cache-causality claims. Export event encodings, perf kernel/version, CPU model, exact workload and thread placement; repeat across processes/days.

### Local environment and exploratory observations — 2026-11-21

CPU model reported: AMD EPYC 9V74, Linux, CPython 3.13.5, shared/virtualized environment. OS reports L1D 32 KiB, L2 1 MiB, and shared L3 32 MiB. Six footprint sizes (4 KiB to 64 MiB), 2 epochs, 5 repeats each, 750k target accesses. The biggest ring automatically uses at least one full cycle, so larger footprints have more accesses (normalization is per access).

- Random-dependent access for 4 KiB: about **1.4 ns/access** in both epochs.
- For 64 MiB: **142.4** and **134.6 ns/access** in the two epochs.
- For 16 MiB: about **32.8 ns/access** in this run, versus **54.9 ns/access** in an earlier shorter-size run. This discrepancy is retained as evidence of drift/noise, not deleted.
- Sequential effective throughput in this experiment: roughly **11.7–14.6 GB/s**, **not DRAM bandwidth**.
- L1D/LLC/generic hardware miss counters: **UNAVAILABLE (ENOENT)** in this container. L2 hardware misses: **NOT SUPPORTED BY GENERIC EVENT**.

These numbers describe this environment and do not establish L1/L2/L3 transitions, CPU efficiency, cache-miss causality, or any speedup of Python. CI on GitHub will measure different hardware and must not be compared as a controlled A/B result.

## Promotion / decision gate

No runtime-optimization candidate may be enabled based on this benchmark alone. A further claim needs reproducible CPU pinning, thermal/load isolation, validated hardware counters including model-specific events, interpreter-level A/B tests on the same workloads, and at least one confirmatory unseen workload. Include counterexamples and negative results in the record.
