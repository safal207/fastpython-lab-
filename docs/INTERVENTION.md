# AlexPy candidate interventions — preregistered A/B research

## Question and scope

Does removing the native `NotImplemented` + Python rerun for huge exact
integers improve `sum_squares`? Can an exact zero/singleton fast path remove
call overhead? Neither intervention constitutes a Python interpreter fork.

`alexpy.candidates` exposes **two opt-in** implementations:

- `bigint_native`: a C extension method preserves the fast signed-64-bit loop
  and promotes to `PyLong` if a square or accumulated sum exceeds the range.
  Exact plain `list`/`tuple` with exact `int` elements only. For other types,
  it returns `NotImplemented` and delegates to the existing public function.
- `small_dispatch`: directly handles zero and singleton exact built-in
  sequences with exact `int` elements; otherwise delegates to the existing
  public function. It never inspects `list` subclasses.

The stable `alexpy.sum_squares` remains **unchanged**. No optimization is
promoted because it wins on just one workload.

## Semantics and failure conditions

The gate compares the generator expression, `pow` witness and candidate
outputs, before timings. Unit tests cover arbitrary-precision integers,
64-bit accumulator overflow, custom integer `__mul__`, overridden iteration,
iterator consumption, falsey/float inputs, and exception propagation. Code
paths for list subclasses always fall back to Python. For unsupported
mixed-type sequences the native helper may re-read previous *exact integers*
but must not replay user-defined arithmetic; a later nonexact element is
handled by the ordinary baseline expression.

## Predeclared graph and measurement

Fourteen steady states cover the four historical regression candidates plus
zero, singleton, large, overflow, long small-integer and subclass controls.
They are already defined by `State(container, length, profile, phase)` in
`alexpy.graph_search`. Each of three fresh Python interpreter processes takes
nine within-process paired samples per state and candidate. Timings are
recorded in nanoseconds per call; execution order rotates and alternates.
The raw sample arrays, process medians, environment and SHA-256 digest are
retained in the JSON report. The 10% classification rule is from
`regression_triage.assess`. This is **not** independent hardware or a formal
causal estimate; running variants in the same process does not isolate all
cache/thermal/clock effects.

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests -v
python -m alexpy.intervention --sessions 3 --repeat 9 --target-seconds .004 --json intervention-results.json
```

## Local observations — CPython 3.13.5, Linux, one host

Two separate complete experiments, both three processes × nine pairs.
Relative ratios compare each candidate to the unchanged AlexPy entrypoint.

| Graph state | Candidate A bigint C | Candidate B small dispatch |
|---|---:|---:|
| tuple / 1 / huge | 2.06×, 2.10× | 2.12×, 2.07× |
| list / 1 / huge | 2.11×, 2.07× | 2.19×, 2.11× |
| tuple / 8 / huge | 1.55×, 1.55× | 0.92×, 0.91× |
| list / 16 / accumulator overflow | 2.57×, 2.65× | 0.94×, 0.94× |
| list / 0 / small | 1.09×, 1.02× | 1.54×, 1.49× |
| list subclass / 0 / small | 0.77×, 0.76× | 0.78×, 0.78× |
| list / 8 / small | 0.99×, 0.97× | 0.61×, 0.61× |

Values are rounded cross-process median ratios (first run, second run).
**Both candidates cause repeatable regressions in the guardset.** They stay
opt-in and `promotion_gate=BLOCKED_GUARDSET_REGRESSIONS` by design. A single
successful scenario does not justify switching the public default. The
subclass regressions reflect another Python function call in the candidate
wrapper; neither implementation solves that issue.

## Next experiment

Pre-register a narrow dispatcher with **no additional work for subclasses**,
check that it preserves custom `__iter__`/`__mul__`, and compare it with both
unchanged control and explicit C candidates. Then validate on CPython 3.15,
Windows, macOS and a second physical machine, report negative outcomes, and
only consider enabling it if the complete guardset passes.

Research roots: `COSMIC-ORGANICS/morphos/sparse_scheduler.py` (strong controls),
`morphos/multimirror.py` (multiple-scale checks), `CaPU/STATE_MACHINE.md`
(fail-closed transitions), `ContractGraph-QA` (counterexamples). These are
architectural analogies, not dependencies or transferred performance claims.
