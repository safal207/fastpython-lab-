# AlexPy: state graph / spatial and temporal transition QA (v0.2 research)

This is **performance exploration for a specialized `sum_squares` kernel**, not a new Python interpreter or a measured CPU design. The graph represents test configurations, **not physical processor topology**.

## Axes and test-design methods

| Axis | Discrete states | Test design purpose |
| --- | --- | --- |
| Container / data shape (space) | `list`, `tuple`, `list_subclass` | Equivalence classes; dispatch boundary |
| Length / workload scale (space) | `0`, `1`, `8`, `16`, `1024`, `10000` | Boundary value analysis; crossover and overhead |
| Integer values / arithmetic domain (space) | `small`, `mixed`, `huge`, `acc_overflow` | Overflow transitions; fallback conditions; negative values |
| Benchmark stage (time) | `prewarm`, `steady` | Before/after explicit warmup; **NOT** an OS cold-start |

Edges change **exactly one axis**; length/layout traverse adjacent discrete levels. The full directed conceptual grid is represented by an undirected DOT graph. Candidate selection greedily covers previously-unseen **pairs** of spatial axes (not exhaustive). Follow-up suggestions are single-axis neighbors of measured cases; performance gains/regressions increase search priority.

## Mirror and gate chain

```text
state → data → exact-int mirror A: sum(x*x) ─┐
                 independent syntax mirror B: sum(pow(x,2)) ─┤
                 AlexPy native or fallback ────────────────┘
                                 ↓ all agree?
                          BLOCK / benchmark
                                 ↓
              prewarm observation → steady A/B–B/A pairs
                                 ↓
                        exploratory bootstrap CI
                                 ↓
                  candidate gain / regression / inconclusive
                                 ↓
                  next unmeasured graph neighbors
```

`sum(x*x)` and `sum(pow(x,2))` follow **different arithmetic expressions** but **share CPython and host hardware**. They are *not* independent hardware observers. The comparison is performed outside timed regions. A mismatch blocks benchmarking. A short or noisy run cannot become a verified gain.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m alexpy.graph_search --require-native --cases 12 --repeat 9 \
  --json graph-results.json --dot graph-states.dot
# Optionally render the graph with Graphviz:
# dot -Tsvg graph-states.dot -o graph-states.svg
```

`graph-results.json` records all raw paired observations, expected fast-path behavior, platform, source SHA in GitHub Actions, candidate verdicts, follow-up states and a canonical SHA-256 checksum. `graph-states.dot` can be viewed with Graphviz. The checksum is **integrity metadata**, not third-party attestation or independent validation.

The paired bootstrap is **exploratory only**; repeated timing samples may be correlated, and this is not a confidence guarantee. Confirm any gain on a dedicated machine, with randomized independent sessions, additional optimized baselines (NumPy and PyPy when relevant), energy/perf counters and explicit regressions before promotion.

## Why these earlier repositories matter

- [COSMIC-ORGANICS](https://github.com/safal207/COSMIC-ORGANICS): sparse frontier and transition-state ideas; inspect `morphos/sparse_scheduler.py`, `morphos/multimirror.py`, `morphos/witness.py`. This lab **adapts research concepts**, not their source or claimed speedups.
- [CaPU](https://github.com/safal207/CaPU): state gate/commit vocabulary inspires correctness-first verdicts; **no CaPU runtime integration or security guarantee**.
- [BardoCompute](https://github.com/safal207/BardoCompute): compare transitions as observed costs, not automatically beneficial computation.
- [ContractGraph-QA](https://github.com/safal207/ContractGraph-QA): small counterexamples, state-machine coverage and fail-closed evidence capture.

### Known gaps

- Current search scans one operation and one machine per run; it does not optimize CPython dispatch, CPU scheduling, cache policy or JIT.
- The graph currently varies *input configurations*, not memory placement, CPU frequency, cache topology, thread affinity or process-level cold-start.
- No physical speed, power or energy advantage has been demonstrated by COSMIC / CaPU / Bardo integration here.
- The default candidate threshold is 10% (bootstrap lower bound), but **candidate != independently verified**.
- Next steps: add stability checks across independent process restarts, pyperformance control tasks, more arithmetic kernels, property-based/MCDC tests and optional performance counter adapters.
