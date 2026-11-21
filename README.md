# AlexPy 0.1 — Python Performance Lab

**Status:** research prototype, not a Python interpreter or CPython fork yet.

Our first measured hypothesis: a specialized native C loop can outperform
`sum(x*x for x in values)` when `values` is a plain `list[int]` or `tuple[int]`.
This is a **single-operation speedup**, **not** a claim that "Python is N× faster".
The experiment preserves Python semantics by returning to the normal Python
implementation for unsupported input types or values outside the signed
64-bit fast path.

## What it contains

- `alexpy/_native.c`: original CPython C extension, handwritten native loop.
- `alexpy/__init__.py`: safe dispatcher/fallback, arbitrary precision preserved.
- `alexpy/benchmark.py`: warm-ups, repeated wall-time samples, median,
  baseline and backend, JSON export with hardware and interpreter metadata.
- `tests/`: correctness tests including large ints, subclasses, and iterables.
- `.github/workflows/ci.yml`: Linux, Windows, macOS; Python 3.13 and 3.15;
  correctness gates and benchmark artifact.

## Build & test (Python 3.13–3.15)

Requires a C compiler supported by your Python installation.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.benchmark --require-native --sizes 1000 100000 --json benchmarks.json
```

Do not compare results obtained on different hardware/interpreters, and do not
substitute Python fallback results for a native performance comparison.
The baseline is the **Python generator expression**, not NumPy, a precompiled
kernel, or the entire CPython interpreter. Sample sizes, run counts and
crossover points affect the reported speedup.

## Important limitations

- Only the `sum_squares` function is optimized; all other Python runs normally.
- No new Python syntax, parser, JIT, or modified interpreter core yet.
- The native fast path is designed for exact `list[int]`/`tuple[int]` objects;
  uncommon inputs fall back to Python for correctness.
- This module does not add parallel execution or eliminate the GIL.
- Performance on a particular PC is not evidence of a universal speedup.

## Roadmap

1. **v0.1 (this repository)**: correctness, C extension, reproducible benchmark.
2. **v0.2**: profile benchmark bottlenecks; add comparison with optimized
   baselines, CPython 3.15 and PyPy; track noise and CI regressions.
3. **v0.3**: separate CPython fork experiment; measure a specific bytecode
   dispatch/optimizer change while running `test.regrtest` and pyperformance.
4. **v0.4**: third-party reproduction and a technical report with wins *and*
   regressions.

See [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the acceptance criteria.

## License

MIT. This repository contains original code, no vendored CPython files.

## v0.2 state-transition graph explorer

A correctness-first experimental search instrument is now available:

```bash
python -m alexpy.graph_search --require-native --cases 12 --repeat 9 \
  --json graph-results.json --dot graph-states.dot
```

It uses pairwise test design, boundary values, one-axis state transitions,
prewarm/steady observations, independent-expression mirrors, and candidate-only
statistical verdicts. Read [docs/GRAPH_EXPLORATION.md](docs/GRAPH_EXPLORATION.md).
Inspired by prior COSMIC/MORPHOS, CaPU, BardoCompute and ContractGraph-QA
research; their performance results are not imported as Python performance evidence.

## v0.2 regression triage: fresh-process replications

The previous graph found four potential regressions. A separate diagnostic
traces which **code path** each input takes, compares one-axis controls that
actually change the workload, and checks repeated paired timings in new
Python interpreter processes:

```bash
python -m alexpy.regression_triage --sessions 3 --repeat 9 \
  --target-seconds .004 --json regression-triage.json
```

The output preserves raw samples and distinguishes a code-path explanation
from causal evidence of timing. See [docs/REGRESSION_TRIAGE.md](docs/REGRESSION_TRIAGE.md).
