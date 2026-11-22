# Native Batch QA — platform and resident-memory audit (v2.3, opt-in)

The v2.2 C kernel was already tested in the GitHub Actions matrix on **Ubuntu, Windows, and macOS**, Python **3.13 and 3.15**. This version adds a *separate, platform-local* allocation and resident-memory protocol. It does **not** enable native QA routing automatically.

## Sources and honest scope

The supplied `evidence/qa-actions-anonymized-replays.json` contains 24 completed public workflow-job records drawn from four runs of this repository (links are in the JSON). The only persisted job fields are public operating-system/Python matrix labels, completion status, outcome and start/finish times; the timestamps were shifted by one constant offset to an arbitrary reference date. Runner names, GitHub job identifiers, user identifiers and logs were **not** included. Durations and relative chronology remain identical to the public source. All 24 observed results were successful. Repeated job names are retries, so the state model holds **6 logical jobs** across **3 suite groups**. These are real **GitHub Actions job metadata**, not individual unit-test logs, private enterprise QA output, or a production performance workload.

The native, strong-Python and previous AlexPy engines are checked against an independent O(N) full scan at report boundaries, including observed retries and the last-write-wins batch contract. All 24 job events have independent traceable public provenance. No external GitHub calls occur during CI audit execution; the sanitized snapshot is committed and deterministic.

## Reproduce the memory audit

Build the native module first:

```sh
python -m pip install -e .
python -m unittest discover -s tests -q
python -m alexpy.qa_field_audit --cases smoke-retries,medium-mixed,recorded-actions --batch-sizes 64 --sessions 3 --json qa-field-audit.json
```

For a shorter cross-platform CI protocol:

```sh
python -m alexpy.qa_field_audit --cases smoke-retries,recorded-actions --batch-sizes 64 --sessions 2 --json qa-field-audit.json
```

Each route runs in an isolated new interpreter with a fixture generated **before** instrumentation. The parent separately validates every route against the exact mirror and supplies the expected digest to its child. A mismatch causes the audit to fail. Routes are rotated between sessions. The native route refuses to benchmark if the C extension is absent.

The audit reports:

- `peak_traced_bytes`: peak allocations tracked by `tracemalloc` since construction. This can include CPython's C allocations through tracked allocators, but **not all native allocations**; it is *not* whole-process RAM.
- `rss_peak_bytes`: the process high-water mark, including Python, imports and common fixture data. Linux uses `ru_maxrss * 1024`; macOS uses byte-based `ru_maxrss`; Windows uses `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize`.
- `rss_current_delta_bytes`: change in current resident/working-set memory during the measured replay, available on Linux and Windows. This can be negative, rounded by pages, or unavailable (e.g. on macOS); it is not an exclusive engine heap size.
- `rss_highwater_growth_bytes`: change in the process high-water value. Often zero if earlier imports dominated maximum RSS. **Do not interpret zero as zero allocation.**
- `instrumented_ns`: execution duration while `tracemalloc` is running. **Never compare to the separate uninstrumented speed benchmark**: tracing adds unequal overhead.

Fixture generation, the independent correctness witness and process startup are excluded from engine allocation measurement; initialization, package slicing, batch application and snapshots are included. The C module is installed by `setup.py` on every platform matrix cell and explicitly checked for presence. CI uploads an independent artifact for each OS/Python runner; platform memory results must **not be pooled**.

## Local Linux results (CPython 3.13.5)

Three independent processes per route. `evidence/qa-field-audit-summary-local-py313.json` contains the tabulated medians, and the three `qa-field-audit-*-local.json` files hold the underlying measurements.

| Workload | Peak Python-traced memory: native / strong Python / old AlexPy | Native comparison |
|---|---|---|
| 256 tests, 240 retries, batch 64 | 22.1 / 11.2 / 15.5 KiB | More memory than strong Python |
| 4096 tests, 6000 events, batch 64 | 216.9 / 114.8 / 160.6 KiB | ~1.89x traced allocation vs strong Python |
| 24 public job records, 6 logical jobs, batch 4 | 3.5 / 2.6 / 4.4 KiB | Too small for performance claims |

For 4096 tests, the median change in current process RSS was approximately native **212 KiB**, strong Python **116 KiB**, old AlexPy **204 KiB**. These are noisy page-granularity observations and should be treated as supporting diagnostics rather than hardware-independent memory limits.

The v2.2 uninstrumented speed study reported approximately **1.80x** median against strong batch Python on synthetic data in three processes; the present audit does **not** rerun that performance study on every hardware target. A speed/memory Pareto evaluation is required before adoption.

## Deployment gates

1. Verify native module compilation, full correctness suite and fail-closed transaction behavior on all CI targets.
2. Verify memory and latency on the target hardware and representative real, sanitized QA datasets. The 24 public job metadata events are an integration smoke test, **not** that workload.
3. Keep C-accelerated routing opt-in. Set a workload-specific resident-memory budget and ensure correctness before promoting any algorithm. In particular, do not claim the native batch kernel uses less memory than optimized Python.
4. For production validation, request explicitly authorized and properly anonymized JUnit/CI logs from the application owner. This study did not ingest or fabricate any private logs.
