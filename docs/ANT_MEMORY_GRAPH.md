# AlexPy 1.3 — Ant-Memory Graph: sparse experience atop a verified execution DAG

## Purpose and limits

The dependency DAG defines mandatory dataflow; it is **not** modified or
traversed probabilistically. The ant-inspired matrix chooses among four
preapproved, semantically equivalent *complete* execution plans (`manual`,
`partition`, `compiled`, `dense`), and only after the concrete workload has
passed the independent dense-reference correctness check. A true multi-ant
optimizer or a CPython-wide runtime optimizer is **not** claimed.

Grid cell = five coarse coordinates `(log2(size), log2(width),
log2(changes), log2(ticks), log2(reads))`. The sparse matrix stores only
observed `(cell, plan)` edges. Values: bounded pheromone, measurement count,
mean log speed ratio vs the manual baseline and last experimental epoch.
A separate sparse `(prior_cell, current_cell, plan)` counter remembers
transitions in space and time but is **not** interpreted as a speed advantage.

Training reward is clamped measured `manual_ns / candidate_ns`, added to a
lazily evaporating pheromone (rho=0.1 for experiments). Deterministic
exploitation selects the highest score; optional seeded probabilistic choice
allows exploration *offline*. Unseen states interpolate nearby log2 cells
within a finite radius; outside that support the strong manual plan wins.
Failed checksum gates quarantine the `(cell, plan)` pair. A compiled lookup
caches deterministic decisions and is invalidated on any new training or
quarantine. This avoids repeating graph-neighbor searches on every call.

**Safety boundary:** Application code must enumerate/validate the legally
available plans for its actual graph. Matrix memory never proves plan
compatibility; default `alexpy.sum_squares` remains unchanged.

## Blind holdout experiment

Predeclared parameters: training lengths 4096 and 32768, held-out lengths
8192 and 16384, changes 1/8, reads 1/8, ticks=4, width=256, three fresh
CPython 3.13.5 Linux processes. 8 training cells, 8 held-out cells, each
training cell checks all four candidates against an independent dense result.
Training captures alternating-order per-route timings before updating
pheromones. **All eight held-out decisions are frozen before any held-out
execution is timed.** Held-out evaluation repeats AB/BA-style order rotations,
measuring the entire workflow including policy selection; it also measures a
strong fixed manual baseline, fixed partition baseline and a retrospective
oracle that must never be used for deployment.

Run from repository root after `python -m pip install -e .`:

```bash
python -m unittest discover -s tests -q
python -m alexpy.ant_bench --sessions 3 --pairs 4 --json ant-memory.json
```

The protocol is end-to-end for **each benchmark workflow** (construction,
patches, reads). One-time training and policy compilation are reported
separately and **not included** in per-workflow online times. Full deployment
cost must amortize those one-time expenses, and may be negative.

## Local results; evidence not a general speedup claim

- Naive neighbor lookup (3 sessions, 3 pairs each): 0/8 holdouts were >10%
  faster than manual in all processes; 4/8 were >10% slower in all processes.
  Median manual/ant over eight holdouts: ~0.816.
- Compiled cached lookups (3 sessions, 4 pairs each): 0/8 confirmed >10%
  faster, 0/8 confirmed >10% slower across all processes; median manual/ant
  over eight holdouts ~0.961. One state had an exploratory >10% median
  advantage but did not satisfy all-process confirmation.
- All eight held-out decisions in both protocols chose `manual`. Thus there
  is **no demonstrated routing advantage** from ant-memory; the gain is
  reduced *selection overhead* only, with normal experimental variability.
- 8 training cells, 32 observed state-action edges, 7 distinct visited
  space-time transition edges. Serialized state ~3.9 KB (not a measurement
  of resident Python heap memory).
- Training ~0.39–0.40 s in the earlier protocol, ~0.48–0.49 s in the cached
  protocol, and freezing the eight held-out decisions cost ~156–158 microseconds
  in the cached protocol. These are separate costs, and benchmarks use
  different pairing counts: **do not attribute the entire between-run
  difference to caching**, as noise/pair counts changed.
- `evidence/ant-memory-local-py313.json` and
  `evidence/ant-memory-compiled-local-py313.json` include raw samples,
  checksums, decisions, training timings, control measurements, and SHA-256
  checksums.

## Research gates

No public auto-promotion. Next evaluate workloads on which *different*
verified plans win, quantify regret against the hindsight oracle, evaluate
multiple machines, and test cold/warm and workload-drift costs. Include
training and buffer conversion in amortized deployment decisions. Guard against
sampling bias: no test-cell measurements may update training memory.

Inspired by the idea of an ant pheromone matrix and COSMIC's sparse-frontier
and witness experiments; neither quantum hardware nor fractal geometry nor
a CPU cache effect is inferred from these software numbers.
