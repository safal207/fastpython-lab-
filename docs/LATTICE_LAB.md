# AlexPy lattice lab: spatial, temporal and multi-scale test design

Motivation: a **numerical lattice** such as the one used in lattice QCD discretizes
continuous domains so neighboring configurations can be studied systematically.
We borrow the *grid organization*, **not** QCD equations, quantum-computing
claims, spacetime physics, or a claim that spatial neighbors are physical neighbors.

Run after building the C extension:

```bash
python -m alexpy.lattice_lab --epochs 2 --repeats 5 --loops 25 --json lattice.json
```

## What is discretized?

- Grid coordinates: `(container, length, arithmetic profile, phase)`; defaults have
  `3 × 6 × 4 × 2 = 144` possible configurations.
- Fifth coordinate: **experiment epoch**. Same-state points measured in consecutive
  epochs form time edges; changing state between measurements does NOT create a
  physical time edge.
- Spatial edge: one controlled axis changes to a documented graph neighbor. It
  measures a *configuration contrast*; it is not a causal intervention because
  system temperature, cache, scheduler and order can still affect latency.
- `footprint_bytes_est`: `sys.getsizeof` container plus unique element objects.
  This is a **rough footprint proxy** and not measured RSS, cache misses or
  CPU energy. Thresholds for L1/L2/L3 are unknown.
- Multi-scale axis: median absolute difference in log-speedup at Manhattan
  grid distances 1, 2 and 4. Insufficient pairs produce `null`; it is not
  a Hurst exponent, fractal dimension or proof of self-similarity.

## Predictive protocol (guard against data leakage)

Before evaluating the grid, hold out **all configurations of lengths 8 and
1024**, regardless of container or arithmetic profile. Fit no parameters on
these lengths. Compare the absolute prediction error of (i) median training
speedup, (ii) local 1-hop neighbor average, and (iii) distances up to 4 hops.
Report false negatives for regressions of 10% or worse; retain poor outcomes.
The entire protocol is exploratory until repeated on independent hardware and
against a preregistered untouched set of workloads.

The first run is not a forecast of arbitrary Python programs. It only tests
one opt-in `sum_squares` dispatcher relative to an expression baseline.

## Guardrails

- Correctness verified against two Python expressions **before** timing.
- AB/BA order alternated and the sequence of states reversed across epochs.
- Every measurement records time, state, decision route and estimated footprint.
- No promotion into `alexpy.sum_squares`; no global Python speedup claims.
- No cache-effect or hardware causality claims without independent CPU counters,
  reproducible pinning and controls.

Relationship to the user's other research: COSMIC's sparse scheduling, multi-
mirror and witness experiments informed graph coverage and mirrored checks;
CaPU and ContractGraph-QA inspired fail-closed gates. No historical processor
benchmark or hardware result is transferred to AlexPy.
