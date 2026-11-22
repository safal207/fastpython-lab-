# AlexPy 1.7 — noise-aware route memory and switch economics

**Status: opt-in research only.** No changes to public `sum_squares` or CPython.
The intent is to test whether memory of verified *competing algorithms* can
improve the choice of a complete range-query plan. It must never suppress an
obligatory dependency-DAG edge or bypass arithmetic correctness.

## Evidence and route selection

`StableRouteMemory` observes all four `range_routes` backends, after
`checked_routes` confirms their final result against `direct`. It accepts
**only positive, same-process route timings for all four backends**. Repeated
measurements are aggregated to one median ratio *per independently launched
Python process*. Multiple repetitions inside a process do not count as
independent confirmations. Duplicate process IDs are rejected.

The reference is the strong, fixed, preregistered feature heuristic:

- `prefix` when there are at least 16 wide range queries per step;
- `direct` otherwise.

For a previously unseen size, the router uses the nearest *trained size*
with **exactly matching** update/query counts, tick count and span category.
It evaluates the log of the algorithm's time relative to the local reference.
A candidate can replace the reference only when at least three separate
training processes agree on sufficiently strong gain under the bound:

```text
risk = max(median(log_ratios) + 2*MAD(log_ratios),
           worst_observed_log_ratio)
       + 0.07 * abs(log2(heldout_size / training_size))
select alternative only when risk < -log(1.10)
```

This is a **conservative engineering bound, not a statistical confidence
interval or a generalization guarantee**. A single losing independent run
can veto a switch. A quarantined route is never scored using that training
cell. The resulting table is frozen **before holdout timings are obtained**.
The runtime performs only a fixed dictionary lookup.

## Hardware/runtime isolation

Each evidence collection has an explicit environment string incorporating
Python implementation/version, OS, architecture and an **operator-supplied
`device_tag`**. If the runtime differs, the router returns the safe reference
rule rather than using another device's timing profile. These tags are not
cryptographic hardware identifiers and do not establish cross-device
transfer validity. Real phones, other CPUs and new Python runtimes were
**not** measured. Such validation remains necessary.

## Switching costs

Training additionally measures **directed layout transitions** using
`range_bench.switch_probe`: time to construct the new representation from
the current data snapshot (`setup_ns`) and time for one target tick with
updates and queries (`tick_ns`). A separate switch adviser uses paired
independent processes; its predicted benefit over an announced horizon is

```text
savings(horizon) = horizon * (tick_old - tick_new) - setup_new
```

It reduces the prediction by `2*MAD` across processes and requires at least
10% savings relative to the incumbent before recommending a rebuild. The
old representation is assumed already live; the input snapshot copy and
interpreter/environment changes are **not** included. Multi-tick predictions
assume the target workload stays similar. The adviser does not perform
switching and is not a runtime optimizer.

## Blind performance protocol

Commands (locally tested on Linux with CPython 3.13.5):

```bash
python -m unittest discover -s tests -q
python -m alexpy.stable_bench \
  --train-sizes 128,512,2048 --holdout-sizes 384,1536 \
  --changes 1,8 --queries 1,8,32 --spans both --ticks 4 \
  --pairs 5 --sessions 3 --device-tag cpu-local-py313 \
  --json evidence/stable-memory-full-local-py313.json
```

Training runs in **three separate processes** across 36 workload
configurations. Holdout is evaluated in *three other processes* across 24
configurations, using previously frozen decisions. The held-out sizes fall
into coarse grid cells distinct from those of training sizes.

Five policies are measured on identical deterministic workloads:
`fixed_train` (best fixed plan by training totals), `simple_rule`, legacy
`ant`, legacy `wave` and the new `stable` router. The benchmark rotates
execution order and includes each policy's table lookup and full workload
execution in timing. It also separately measures all four raw backend
implementations to record an after-the-fact oracle. Raw samples, checksums,
choices, gates and the frozen-map SHA-256 are preserved. Oracle values are
**never available to the router before evaluation**.

### Reproduced local results (three process runs)

| Selector | Median speedup vs `simple_rule` across 24 scenarios |
|---|---:|
| `fixed_train` | 0.568× |
| `simple_rule` | 1.000× |
| `ant` | 0.995× |
| `wave` | 0.884× |
| `stable` | **1.004×** |

These are **medians of per-scenario relative speeds**, not total runtime
ratios, confidence intervals or population estimates. No overall practical
speedup can be claimed from 1.004×.

The risk-aware selector changed the reference choice in only four scenarios:

| Held-out size | Changes/tick | Queries/tick | Reference → stable | Median speedup |
|---|---:|---:|---|---:|
| 384 | 1 | 8 wide | direct → blocks | 2.081× |
| 384 | 8 | 8 wide | direct → prefix | 2.183× |
| 1536 | 1 | 8 wide | direct → blocks | 2.939× |
| 1536 | 8 | 8 wide | direct → blocks | 2.965× |

No stable choice had a >10% slowdown in **all three** independent processes;
for comparison, 6 ant choices, 11 wave choices and 18 fixed-train choices
had such repeated >10% regressions. These counts apply **only** to this
deterministic held-out workload and hardware. Variability is substantial;
for one 384-element case the raw oracle varied by process.

Cost: total timed **calibration + switch probing** was about **1.297 s
summed over three training processes** (approximately 1.104 s for algorithm
measurements and 0.192 s for transition probes). Frozen-policy compilation
cost **8.65 ms**. Subprocess startup and Python runtime initialization were
not included in those timers; amortization into deployed traffic was **not
proven**. Online selection is included in held-out policy timings.

## QA and reproducibility

216 local `unittest` cases passed on CPython 3.13.5, including 23 new tests.
They exercise unverified-evidence rejection, profile mismatch, repeated
process rejection, stable and noisy evidence, quarantine, holdout isolation,
short- and long-horizon transition decisions, malformed timings and full
four-backend mirroring. The bounded CI smoke uses **three distinct processes**
with small sizes and does not set any performance promotion threshold.

**Limitations:** one actual machine, a deterministic synthetic workload,
restricted range-sum algorithms, small sample counts and operationally
provided device tags. Decisions and measurements are advisory. No claim
that the learner accelerates arbitrary Python or transfers between hardware.
