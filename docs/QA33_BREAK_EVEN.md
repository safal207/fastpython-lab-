# QA 3.3 — Adaptive Break-Even (frozen protocol)

Base: `6c363649b89c23ddf43ea2fc22e6cdc9dcfe042b`. QA 3.2 green on both runs 45893137138 and 45893143885. Its fixed fallback experiment does not establish a useful alternative route.

Question: can a fresh-session route selected using declared size, batch horizon and diagnostic cadence repay its lookup cost, remain safe in tail latency, and ultimately repay runner-local calibration?

* Frozen 12 plans: test count 128/4096; batch size 32; batch horizons 1/8/32; report every 1/8 batches (always report at end). Synthetic random retries and nonzero durations. Different seeds for each training/holdout process. Plans must be known up front; no information from future actual outcomes used for selection.
* Compare exact C-tree and bitset on the same fixtures, independently validate every batch against QA29 scan oracle, compare complete requested snapshots/witnesses outside timed interval for every replay. Same report schedule for all routes. Include construction and destruction as incurred by replay; time the complete fresh session, not browser execution.
* Three independent training processes and three independent holdout processes per runner, 200 replays per route per plan in balanced randomized orders. Keep raw samples/order and per-process results. 200 samples gives a noisy p99 estimate (two upper observations); all timing gates advisory, not a significance test.
* Train exact-plan table only, no interpolation. Require all training processes to show >=5% median gain and no p95/p99 increase after adding the maximum measured lookup p99, and no increase in peak traced allocation. Missing/insufficient evidence -> tree. New experimental lookup is an exact-plan dictionary, not the existing QA30 dispatcher. Scope to same runner invocation; no persistence or automatic activation.
* Freeze before holdout. Measure actual adaptive lookup + replay alongside direct routes. Require >=5% median gain, no p95/p99 increase and no traced allocation increase in all three holdout processes; failures are scientific negative results, not CI failures. Do not retune. No live state migration.
* Distinguish marginal session break-even from total calibration amortization. If an alternative passes, report ceil(total measured calibration wall time / smallest positive per-process median holdout savings) as an illustrative conservative repeat-session count, conditional on stationary workload and cost; no guaranteed payback. If none pass, no break-even demonstrated. Report only tested horizons, no monotonicity assumption.
* Memory: separate tracemalloc replay, not process RSS or native malloc. Existing QA32 RSS limitations remain. No production claim.
* CI Python 3.13 on Linux/macOS/Windows; Python 3.15 correctness smoke. Separate runner reports, never pool absolute timing. No default changes, no merge.

Run after native build: `PYTHONPATH=. python tests/qa33_break_even.py --output qa33-results.json`.

Implementation note: the full frozen 12-plan table is present in each measurement worker; table construction is charged to calibration, not to the per-session lookup. The pre-existing QA30 selector is unchanged. The first protocol commit was superseded before inspecting any QA33 measurements to correct the table-size control.
