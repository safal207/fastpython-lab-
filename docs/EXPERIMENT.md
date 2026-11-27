# Experimental protocol / Протокол эксперимента

## Hypothesis H1

A C loop specialized for exact `list[int]` can improve `sum(x*x for x in values)`
for large input sizes. It may perform **worse** for small inputs due to extension
call overhead. We keep and report both cases.

## Fair comparison

1. Build extension under the *same* CPython process and interpreter version.
2. Verify identical outputs before timing.
3. Warm up each implementation, record >=9 samples and medians.
4. Record OS, architecture, CPU count, interpreter/build and input sizes.
5. Do not cherry-pick runs; keep raw JSON and reproduce on another machine.
6. Add benchmarks against `sum(map(...))`, NumPy, PyPy as distinct comparisons.
7. Present source of speedup (native fused loop, specialization), not as a
   universal engine optimization.

## Acceptance criteria for v0.1

- [x] Native extension implements a specialized operation.
- [x] Fallback preserves arbitrary-precision and dispatch semantics.
- [x] Test suite checks boundary cases and correct results.
- [x] Benchmark exports raw measurements and environments.
- [ ] GitHub CI results confirmed on Python 3.15 across three OSes.
- [ ] Independent repetition on second physical machine.

## Before calling anything "a new Python engine"

- Change CPython's parser, compiler, runtime, or interpreter core (in a fork).
- Document exact source-level changes and compatibility limitations.
- Run CPython's test suite and pyperformance with controlled machine setup.
- Compare multiple realistic workloads, including regressions.

The v0.1 release is **not** such a fork.
