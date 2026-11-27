# AlexPy regression triage: paths, controlled graph edges and independent sessions

**Scope:** the specialized `sum_squares` API only. This research does not measure
or claim faster execution of arbitrary Python programs. `alexpy.regression_triage`
extends `graph_search` and does **not** alter the native C kernel.

## Four preregistered targets

The earlier exploratory 24-case run nominated these slowdowns:

| Graph state | Initial speedup relative to Python | Hypothesized path |
| --- | ---: | --- |
| `tuple|1|huge|steady` | 0.72x | native eligibility check fails; Python fallback |
| `list_subclass|0|small|steady` | 0.76x | wrapper directly uses Python fallback |
| `list_subclass|1|mixed|steady` | 0.79x | wrapper directly uses Python fallback |
| `list_subclass|8|small|steady` | 0.88x | wrapper directly uses Python fallback |

These are **candidates from one exploratory run**, not independent findings.
Exact input profiles and axes are defined in `alexpy/graph_search.py`.

## Stage 1: deterministic execution-path witness

Before timing, the harness requires all three arithmetic results to agree:

1. Python generator: `sum(x*x for x in values)`;
2. separate syntax: `sum(pow(x, 2) for x in values)`;
3. AlexPy entrypoint: `sum_squares(values)`.

Then an independent Python model of the C eligibility checks predicts
`NATIVE_SUPPORTED`, `UNSUPPORTED_CONTAINER`, `OUTSIDE_INT64_SQUARE`,
`INT64_ACCUMULATOR_OVERFLOW`, or an unsupported element type. Its predicted
fast-path decision must agree with the actual native return value or the
harness **refuses to time anything**. These are software expression witnesses,
not independent hardware attestations. The model's check of the C code path
is separate from timing and cannot itself establish a performance cause.

## Stage 2: change one dimension at a time

For each target, measure one-hop graph controls in the container dimension
and one **meaningfully changed input** in the arithmetic profile or length
dimension. At length 0, profiles produce identical empty inputs; for one
item, `small` and `mixed` both initially contain zero. Such no-op comparisons
are deliberately excluded. For each edge, the output records both states,
the dimension changed, code-path classifications and speed ratios.

## Stage 3: independent-process replication

Each of three *new Python processes* performs 9 paired, alternating AB/BA
timing samples. State order is shuffled deterministically by session. All
raw samples and per-process medians are retained. A reproducible candidate
regression requires **all three** process medians to be at least 10% slower
than the generator baseline. With fewer than three sessions or mixed results,
no replicated regression label is permitted. This threshold is deliberately
simple and is **not a statistical significance test**.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.regression_triage --sessions 3 --repeat 9 \
    --target-seconds .004 --json regression-triage.json
```

`regression-triage.json` contains the environment, raw samples, individual
process medians, controls, checks and a SHA-256 integrity digest. CI runs the
same diagnostic on Python 3.15 / Ubuntu and uploads it as a separate artifact.
Performance labels are informational; CI requires correctness tests to pass,
not any particular speed ratio.

## Interpretation and next controlled intervention

- For list subclasses, the public API avoids the C loop by design to retain
  Python's subclass behavior. Additional wrapper overhead is a plausible
  explanation for the slowdown on tiny inputs, but an optimized kernel cannot
  transparently remove the Python API call overhead.
- For huge integer tuples, native code scans an element and returns
  `NotImplemented`, after which Python reevaluates the expression. The
  eligibility rejection is deterministic; the *size* of its performance cost
  still requires controlled intervention and hardware-level profiling.
- Before changing the API or native implementation, compare a branch with
  a narrow early-rejection guard against the unchanged branch, using the same
  machine, equal inputs, randomized runs and correctness tests. Report
  regressions alongside wins. Do **not** use a change that breaks subclasses,
  integer precision or iterator side effects.

Related research: [COSMIC-ORGANICS sparse scheduler](https://github.com/safal207/COSMIC-ORGANICS/blob/main/morphos/sparse_scheduler.py), [COSMIC-ORGANICS multimirror](https://github.com/safal207/COSMIC-ORGANICS/blob/main/morphos/multimirror.py), [CaPU state machine](https://github.com/safal207/CaPU/blob/main/STATE_MACHINE.md), [ContractGraph-QA](https://github.com/safal207/ContractGraph-QA). These are **design references**, not imported runtime dependencies or transferable performance evidence.
