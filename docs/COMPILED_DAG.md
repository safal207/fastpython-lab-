# AlexPy 1.1: compiled dependency DAG

This opt-in experiment reduces registration and scheduling overhead in the
previous declarative `DependencyGraph`. It is not a faster Python interpreter.

## Implementation

`CompiledDependencyGraph` accepts the same `add_window`, `add_sum` and
`add_variance` declarations. The explicit `compile()` step freezes the graph,
assigns numeric IDs, and prepares parent/child arrays and bucket indexes.

Range square sums initialize through the existing native signed-q buffer
kernel if available. Other aggregate operations retain exact Python integer
arithmetic. Patches are staged and validated before a single owned-buffer
commit; cached nodes are published only after a successful commit. The
previous graph remains unchanged as a control.

Usage:

```python
from alexpy.compiled_graph import CompiledDependencyGraph

graph = CompiledDependencyGraph([1, 2, 3, 4], bucket_size=2)
graph.add_window("sum_left", 0, 2, "sum")
graph.add_window("sum_right", 2, 4, "sum")
graph.add_sum("total", ("sum_left", "sum_right"))
graph.compile()
assert graph.value("total") == 10
graph.apply_patch({2: 30})
assert graph.value("total") == 37
assert graph.verify()
```

No arbitrary Python mutations are tracked. Data are an owned exact signed
64-bit array; sums and squares may exceed 64 bits and remain exact Python
integers. External aliases do not mutate the owned buffer.

## Strong four-way comparison

```bash
python -m unittest discover -s tests -q
python -m alexpy.compiled_bench --sizes 4096,32768 --changes 1,16 \
    --ticks 2,32 --reads 1,8 --sessions 3 --epochs 2 --pairs 5 \
    --json compiled-results.json
```

The benchmark includes construction, buffer conversion, declarations,
compilation, patches and all reads. Every route must return the same exact
checksum. It uses three fresh Python processes, alternating order in two
epochs, five paired samples per epoch and per-route medians. Raw sessions
with sample arrays are included in the downloadable research archive.

The baselines are: `dense` full recomputation (using existing AlexPy C
kernels), `manual` strong hand-coded incremental maintenance, `graph`
original DAG and `compiled` the new graph.

### Local results, CPython 3.13.5, Linux

Numbers below mean baseline time divided by compiled time: values above 1
favor compiled; values below 1 favor baseline. These are medians of three
process-level ratios.

| Elements | Changes/tick | Ticks | Reads/tick | Old DAG / compiled | Manual / compiled | Dense / compiled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1 | 2 | 1 | 3.33x | 0.12x | 0.16x |
| 4,096 | 1 | 32 | 8 | 2.19x | 0.40x | 9.43x |
| 4,096 | 16 | 32 | 8 | 1.83x | 0.25x | 4.63x |
| 32,768 | 1 | 2 | 1 | 3.39x | 0.12x | 0.18x |
| 32,768 | 1 | 32 | 8 | 2.66x | 0.44x | 13.68x |
| 32,768 | 16 | 32 | 8 | 2.37x | 0.39x | 10.74x |

Across all 16 measured configurations the old-DAG-to-compiled ratio ranged
from 1.83x to 3.48x. **The manual incremental implementation was faster in
all 16 cases** and dense full recomputation won for short workloads.

A separate 20-repetition diagnostic for 32,768 elements found median graph
build time about 18.24 ms legacy vs 5.12 ms compiled. Updating 32 patches
took about 0.454 vs 0.408 ms. Thus most improvement currently comes from
initialization, not proven dramatic faster sparse frontier scheduling.

## Safety and review gate

- Existing public `alexpy.sum_squares` and original DAG are unchanged.
- Differential randomized patches, overlapping ranges, sign flips,
  64-bit extremes, failed-patch rollback and complete independent mirrors
  are covered by local unit tests (134/134 passed in the author's environment).
- CI runs correctness on Windows, macOS and Linux with CPython 3.13/3.15;
  a smaller four-route benchmark on Linux is advisory only.
- Do not promote this implementation to a default until it reliably beats
  both the manual control and the required real workloads.
- These measurements do not establish CPU-cache causality, fractal physical
  behavior, a whole-Python speedup or a new processor architecture.

This work borrows sparse-frontier and independent-witness design ideas from
COSMIC-ORGANICS, CaPU and ContractGraph-QA; no hardware performance claim is
transferred.
