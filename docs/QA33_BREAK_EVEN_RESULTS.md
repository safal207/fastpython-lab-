# QA 3.3 — observed results and independent audit

Evaluated exact branch head `f6413998a6a84850d316f5b63ee2de4e54e858d3`. Both CI runs [45897081427](https://github.com/safal207/fastpython-lab-/actions/runs/45897081427) and [45897089396](https://github.com/safal207/fastpython-lab-/actions/runs/45897089396) succeeded. The latter is a pull_request run recording temporary merge SHA `0d8a9a508594c1b8b0e02300c2a06c10fac91fbc`, with branch head as parent and the identical Git tree `067286ce3c3b2896ab116a8c5ba88cb05d8eb9fb`. The two runs each cover Linux/macOS/Windows with Python 3.13 for the full protocol, and 3.15 for correctness only.

## Independent recomputation
Six original archives: 12 plans x 3 training and 3 holdout processes per archive, 200 samples per route. Checked raw nearest-rank p95/p99, medians, traced-memory medians, balanced order blocks, phase/plan consistency, distinct PIDs within each phase, lookup p99 and max training lookup p99. Recomputed all 72 decisions and matched all stored holdout ratios and savings. Total 216,000 full-session samples and 864,000 lookup samples. Archive hashes below. This audit checks reported measurement consistency, not independent execution of the engines.

| Measure | Observed |
|---|---:|
| Selected bitset | 0/72 |
| Failed >=5% median gain gate in training | 72/72 |
| Failed p95 gate in training | 56/72 |
| Failed p99 gate in training | 61/72 |
| Failed traced memory gate in training | 0/72 |
| Holdout gates passed | 0/72 |
| Holdout process-plan pairs with smaller traced bitset peak | 215/216 |
| Holdout process-plan pairs with bitset median >=5% faster | 0/216 |
| Calibration wall time | 6.905–11.143 s |
| Maximum training lookup p99 per artifact | 200–1917 ns |

Medians of per-process ratios (72 per OS, two runs), lower better:

| OS | bitset/tree median | p95 | p99 | traced peak | adaptive/tree median |
|---|---:|---:|---:|---:|---:|
| macOS | 0.9984 | 1.0051 | 0.9882 | 0.8120 | 1.0025 |
| Linux | 0.9987 | 0.9996 | 1.0005 | 0.7974 | 1.0017 |
| Windows | 0.9998 | 0.9979 | 1.0001 | 0.8033 | 1.0021 |

Bitset p99 > tree by 5% in 17/72 macOS, 8/72 Linux, 15/72 Windows process-plan pairs. No holdout plan-run would pass all direct bitset gates even without lookup. No calibration payback number is admissible. At n=4096, median traced peak ratios across the six plans range 0.6807–0.7453; at n=128, 0.8793–0.9597. These are descriptive statistics for a frozen synthetic family, not universal thresholds.

## Evidence manifest (SHA-256 of original artifact ZIP)
| Run | Artifact ID | SHA-256 |
|---|---|---|
| 45897081427 | 15154097350 | b8a9e5e39f4694624007612e2b7c95c6def851c85ce4e229cf3b8f3155eee601 |
| 45897081427 | 15154621201 | 05c8034f5dd5120b47e72d4ccac5f47c8a4cf74016cae0b7698b9599e4745237 |
| 45897089396 | 15153563389 | 6de2497ec5c91debb79408d57b3cc2073fb04109da51550e7739713d07c00b5d |
| 45897089396 | 15154920414 | 76bc30f10551c1beb97db7303b607b313d58bc812c70403124c1f37962e8f087 |
| 45897081427 | 15154022447 | 017c6f98fbfa2008b102f963b7b5b396cfbbdf758e4116c22661a6b77b498628 |
| 45897089396 | 15154661359 | 45249994141006efaa8b5a4051b385e5142761ff3cbea9e3cbbd70ab715445f4 |

Artifact retention is finite. Peak traced allocations are not total RSS or a hard memory bound. Each p99 uses only 200 samples; shared CI hardware noise remains. The two workflows use the same synthetic seed family; they are repeated measurements, not new independent workload families. Negative speed result must be retained. No production promotion.

## Decision and next research question
Keep QA33 opt-in; do not activate automatic routing. QA34 Memory-Budget Admission: investigate whether bitset provides predictable memory reduction under explicitly declared budgets, using fresh synthetic seeds, more sensitive separate-process live RSS measurement where available, and separately specified latency guardrails. Distinguish exact index allocation bounds from measured whole-engine memory. Do not equate traced peaks with enforceable process memory limits. Compare direct selection and fallback/rejection; no silent loss of diagnostics or live migration. Freeze a protocol before measuring; no threshold tuning on holdout. This is a proposed direction, not an implemented result.
