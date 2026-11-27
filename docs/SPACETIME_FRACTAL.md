# AlexPy 0.4: spatial, temporal, multiscale and derivative axes

This is **exploratory performance QA**, not a fractal processor or a faster Python interpreter.

## Four related but distinct objects

1. **Space:** each existing graph node is a tuple `(container, length, numeric_profile, phase)`. A valid edge may change **one** axis. It does not prove causal performance impact; it defines a controlled comparison.
2. **Time:** an observation has `tick` (ordered graph-transition index) and monotone `timestamp_ns` (elapsed clock). One step is not one second. Prewarm/steady remains an experimental label and is **not** a complete cold-start characterization.
3. **Derivatives:** for `L[t] = ln(reference_ns / candidate_ns)`, `d1[t] = L[t]-L[t-1]`, and `d2[t] = d1[t]-d1[t-1]`. Tick derivatives use graph steps; additional `d1_per_second` and `d2_per_second2` use monotone timestamps, including unequal intervals. They describe a measured workload trajectory, **not** causal physical acceleration. Negative d1 means the advantage is shrinking along that controlled route.
4. **Fractal / scale axis:** at hop scale `s`, use `R(s) = median(|L[t]-L[t-s]|)`. Record `s = 1,2,4`, and exploratory log-log slopes `ln(R(b)/R(a))/ln(b/a)` where both are positive. These are **finite-sample multiscale roughness signatures**; **not** a fractal dimension, Hurst exponent or validated long-range scaling law.

Missing scales are explicit nulls. Never manufacture a derivative or scale estimate from fewer points. Flat traces have undefined log-log slopes, not zero by fiat.

## Safety and evidence gates

- Verify two algebraically different expressions and the candidate against exact-int data before timing. Both witnesses still rely on the same CPython arithmetic implementation; they are not independent hardware proofs.
- Preserve custom `list` subclass iteration, rich numeric protocols, errors and generator consumption; no custom hook may be called while choosing a route.
- The experimental route chooses exact empty/singleton fast cases and tests mixed-precision C on selected exact containers. The public `alexpy.sum_squares` is unchanged.
- Every sampled point records the route, graph state and timestamps. Interventions are measured in alternating AB/BA order. Follow up with independent process/machine evidence; do not auto-promote.
- Differences across graph nodes **mix workload-change effects with measurement time**, so cannot by themselves establish a temporal dynamical law. Fixed-state repeated runs are a separate required control for future research.
- The algorithm is inspired by ideas in COSMIC-ORGANICS (sparse graph frontier / multimirror), BardoCompute (state transitions), and ContractGraph-QA (correctness gate). It neither imports nor validates their hardware performance claims.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.spacetime --path targeted --repeats 7 --loops 200 --json spacetime.json
# Or --path walk --hops 15 for a less directed adjacent-state walk
```

`spacetime.json` includes raw baseline/candidate medians per node and a compact graph-walk signature. It is not a pre-registered release benchmark. Future: fixed-state longitudinal controls, independent repeated workers, CPU counters, cache and branch profiles, then a pre-registered no-regression decision rule.
