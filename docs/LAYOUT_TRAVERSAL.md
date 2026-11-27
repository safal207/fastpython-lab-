# AlexPy 0.7 — zero-copy typed-buffer access and layout controls

**Scope:** experimental `sum_squares` for exact `array('q')` or contiguous,
1D, native signed-long-long `memoryview`. No CPython interpreter rewrite, no
changes to `alexpy.sum_squares`, no general Python acceleration claim.

## Hypothesis and controlled alternatives

We study a 64-bit, densely packed integer buffer compared to the **same**
buffer iterated as Python objects. Baseline: current `alexpy.sum_squares(src)`
which uses a Python generator for typed arrays. Competing paths:

- `linear`: native C `PyObject_GetBuffer` read-only traversal in memory order,
  accumulating exact 64-bit squares and promoting to Python arbitrary-precision
  integers as needed. No per-element Python integer materialization on the fast
  path; the GIL stays held and the buffer is released on all exits.
- `interleaved64`: same arithmetic, intentionally changes traversal order by
  visiting elements at 64-element strides. This is a control for locality. It
  visits every element once and never changes mathematical results for exact ints.
- `convert_list`: creates a new list **inside timing** then calls the existing
  optimized list kernel. This measures whether a representation change pays
  for itself; conversion outside timing would incorrectly inflate speedup.

The C method independently checks 1D, contiguous, itemsize 8, native `q`
format. Unsupported formats, strides and buffers return `NotImplemented`.
The Python wrapper only invokes C for exact `array` and exact `memoryview`.
An `array` subclass, float data or strided view follows the unchanged public
Python path, including any overridden iteration behavior. Inputs are not
mutated. Concurrency with external buffer writers is not addressed.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.layout_bench \
  --sizes 0,1,8,128,4096,32768,262144 --epochs 3 --pairs 9 \
  --target-seconds .002 --json layout.json
```

`layout.json` retains paired AB/BA samples, warmup, calibration loop counts,
per-epoch medians, footprint *payload estimates*, adjacent-size finite
differences and SHA-256 of a canonical JSON object. It fails closed on
incorrect results against both a generator expression and a second Python
`pow` oracle, before timing. Phase ordering reverses per epoch; repeated
processes are required for independent checks. All results are exploratory:
no CPU pinning, confidence certification, or hardware cache counters in this
protocol. `hardware_counters=NOT_MEASURED` is explicit. Numeric derivatives
are descriptive log-size differences, **not** physical derivatives or
fractal dimensions. Estimated data length and measured latency do not by
themselves establish L1/L2/L3 cache miss causality.

## Local evidence — CPython 3.13.5, Linux, 2026-11-21

Three separate interpreter processes, each with 2–3 reversed epochs and
nine paired samples, benchmarked the same signed-q array for each candidate.
Medians below are the medians of the three per-process summary medians.
`>1x` is faster; `<1x` is slower than the current AlexPy baseline on arrays.

| Pre-existing typed buffer | Sequential C | Interleaved64 C | Convert-to-list inside timing |
|---|---:|---:|---:|
| 4,096 small ints, `array('q')` | 12.17× | ~12× | ~2.7× |
| 32,768 small ints, `array('q')` | 12.74× | ~15× | ~2.9× |
| 262,144 small ints, `array('q')` | 11.98× | 8.99× | ~2.5× |
| 4,096 large signed-64 ints, `array('q')` | 1.26× | ~1.2× | ~1.1× |
| 262,144 large signed-64 ints, `array('q')` | 1.21× | ~1.2× | 0.84× |
| 1 large signed-64 int, `array('q')` | ~1.8× | ~1.8× | 0.67× |

The `interleaved64` variation is **not** consistently slower or faster than
linear; short runs and changing cache/prefetcher conditions limit causal
interpretation. Speedup at larger sizes is also affected by Python object
creation on the baseline and cannot be attributed solely to cache geometry.
Input `array('q')` construction cost was excluded *for all variants*; end-to-end
performance of converting an arbitrary Python list into an array may be worse.
That conversion is not part of the optimized path and was not benchmarked.

Evidence files in the local lab: `layout-local-py313.json`,
`layout-process-2-py313.json`, `layout-process-3-py313.json`.
The CI workflow records a separate portable smoke experiment on Python 3.15;
compare its results only within the CI host. Avoid comparing unrelated CPUs.

## QA / review gate

- Unit tests check 64-bit boundaries, accumulator overflow, negative values,
  exact numeric results, float fallback, non-contiguous views, subclass
  iteration overrides, generators, and mutation across calls.
- Benchmark holds the exact same input for all variants and includes copy
  overhead inside the conversion candidate.
- Preserve counterexamples: large-value list conversion is reproducibly slower.
- **No automatic promotion**: `BLOCKED_UNTIL_MULTIHOST_GUARDSET_REVIEW`.
- Further work: independent bare-metal hosts, pinned threads, CPU counters,
  multiple Python versions, explicit memory-allocation/representation cost,
  and full-repository tests on all supported OSes before considering defaults.

Lineage: COSMIC sparse-state comparison and witness concepts inform the test
design; the Bardo/CaPU/ContractGraph- QA experiments are research references,
not imported code or inherited evidence of CPU performance.
