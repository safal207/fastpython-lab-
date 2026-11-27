# AlexPy v0.8 — representation switching and reuse economics

## Narrow question

If a Python program already owns a **read-only exact `list[int]`** and repeats
`sum_squares` **K** times, when does converting once to contiguous
`array('q')` pay off? What if a **new Python list** must be returned at the end?

This is a benchmark of **full workflows**, *not* a speed claim about a new
Python interpreter or a generic automatic list optimizer.

## Three execution paths

1. `reference_batch`: use existing `alexpy.sum_squares` on the list **K** times;
   when requested, return `list(values)` at the end.
2. `packed_batch`: validate exact built-in `list`/`tuple` of signed 64-bit
   **exact ints**, allocate and fill `array('q')` once, execute the same sum K
   times using `_buffer_native.sum_squares_linear`, and optionally return
   `list(buffer)`; the conversion is **inside** the timed callable.
3. `repacked_batch`: negative control; validate/pack again before each pass,
   optionally reconstruct at the end. It intentionally destroys the benefit
   of reuse and exposes conversion overhead.

Both timed workflows compute the **same Python `int` total** and optionally
return a **fresh equal list**. The source does not change during execution.
Unsupported/side-effecting types are **rejected as `NotImplemented`** by the
packed candidates; this is a research interface, not a drop-in polymorphic
replacement. Subclasses, custom `__iter__`, out-of-range and non-int values
must never be silently converted. The stable `alexpy.sum_squares` is untouched.

## Cost model (hypothesis, not a fitted physical law)

For a stable input and integer K:

- List: `T_list(K) ≈ K * T_list_eval + (return_list ? T_list_copy : 0)`
- Pack once: `T_pack(K) ≈ T_validate+convert + K*T_buffer_eval + (return_list ? T_unpack : 0)`
- Pack every time: `T_repack(K) ≈ K*(T_validate+convert+T_buffer_eval) + T_unpack`

**Use full measured timings** instead of inferring speedup from kernel-only
latencies. Actual overhead is nonlinear with profile, input size, cache, memory
allocator, frequency, large-integer arithmetic, and interpreter version.

The search grid has axes **size × number of reuses × integer profile ×
output materialization × experimental epoch**. Every state records AB/BA
paired nanoseconds, start/finish metadata and candidate route. Correctness
mirrors (`sum(x*x...)`, `pow`) must pass before measuring. The optional
`empirical_break_even` reports only the earliest measured K for which **all
larger measured K in that session** beat baseline by 10% in **every epoch**;
it is *not* a continuous or universally valid crossover point.

## Reproduction

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.amortization_bench \
  --sizes 8,4096 --repetitions 1,8,32,64 \
  --profiles small,large --epochs 2 --pairs 7 \
  --target-seconds .0003 --candidates pack_once \
  --json amortization-packA.json
# Restart the Python interpreter and run exactly the same command again.
```

For the negative control add `--candidates pack_once,repack_each` (slower);
keep that distinction in any comparison. A lightweight CI smoke test records
JSON as advisory evidence; a CI green check does **not** constitute a
statistically supported performance promotion.

## Local evidence: CPython 3.13.5, Linux, one host

Three independent Python process launches ran the **same pack-once protocol**:
2 epochs × 7 AB/BA pairs, two sizes, two integer profiles, K=1/8/32/64,
with and without list materialization. `evidence/amortization-summary-local-py313.json`
contains medians, the min/max across processes and strict three-process pass
flags. Full raw JSON for all sessions is kept in the supplied research archive.

Representative three-process medians (baseline / full packed workflow):

| Size | Profile | K | Materialize list | Ratio | Three-process >10% gain? |
|---:|---|---:|---|---:|---|
| 4,096 | small | 1 | no | ~0.06× | no, major regression |
| 4,096 | small | 8 | no | ~0.38× | no |
| 4,096 | small | 32 | no | ~1.05× | no |
| 4,096 | small | 64 | no | ~1.78× | yes, this measured point only |
| 4,096 | small | 64 | yes | ~0.92× | no |
| 4,096 | large | 64 | no | ~0.98× | no |
| 8 | small | 32 | no | ~1.33× | yes, but only tiny-input overhead |
| 8 | large | 32 | yes | ~1.51× | yes, small absolute times |

**Negative control:** Two further fresh processes ran both candidates and
showed `repack_each` at 4,096 small ints, K=64, no list output at only
~0.05–0.06× of the existing list route. Converting on every iteration is
catastrophically worse for that workload.

## Limitations and promotion gate

- All results so far: **one host**, no CPU affinity isolation, power/caches
  not pinned, no L1/L2/L3 performance counters, and no second implementation
  or third-party independent runner.
- The full profile grid is **not** all Python; no mutation or arbitrary
  objects, no concurrency guarantee, and no memory-pressure stress test.
- 3 fresh processes provide a useful repeated observation, **not** three
  independent machines. Differences as large as 1.5–2.6× between processes
  occurred in one configuration: do not hide this noise.
- The negative-control runs use a different candidate schedule than the
  three pack-only runs; do not pool their medians without labeling this.
- The proposal does not change the default dispatcher, cannot claim that
  `array('q')` universally improves Python, and does not promote runtime
  dispatch rules based on these samples.
- **`BLOCKED_UNTIL_MULTIHOST_REPLICATION_AND_GUARDSET`** remains mandatory.

Future controlled experiment: vary k over a denser range, compare *keeping*
arrays between independent API operations (when the program can naturally
retain them), measure memory occupancy, test mutation/invalidation costs,
and repeat on Python 3.15 and an actual second host.
