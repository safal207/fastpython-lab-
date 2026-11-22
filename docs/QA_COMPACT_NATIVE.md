# AlexPy 2.4: memory-conscious native QA batch kernel

Experimental opt-in CPython C backend. Previously QAState kept full-length
seen, indices, pending status and duration arrays even for tiny patches.
Those are now per-batch rather than resident. A sparse open-addressing table
stores one integer per hash slot, mapping a test ID to its staged entry.
For large batches (> half of tests, upper-bounded by batch length), the kernel
uses direct temporary indexing. Pending entries are sorted in place and
transitions retain their canonical order. This threshold is an uncalibrated
heuristic, not hardware-tuned.

Every input event is validated before mutation, including overwritten
intermediate events. Last valid status per test wins, SKIP implies zero
duration, per-test duration is exact signed int64 and aggregate durations
remain arbitrary precision. Allocations and transition construction complete
before mutation. Invalid batches roll back completely. The user-facing API
of NativeBatchQA is unchanged; no general-purpose Python optimization is
claimed.

## Independent local research

Linux CPython 3.13.5; 16 synthetic workload configurations; three
fresh Python processes for each C binary, three alternating route-order
measurements per configuration. Total time includes engine initialization,
validation, updates, slicing and reporting. Original and compact C builds
were measured sequentially, so small percentage changes can be host drift.

Median compact / strong Python batch = **1.770x**, all 16 configurations
more than 10% faster in every compact process. Median compact / original
C kernel = **0.971x** (roughly 3% slower, not robustly significant).

Isolated tracemalloc peak medians (KiB):

| Tests / batch | Original C | Compact C | Strong Python |
| --- | ---: | ---: | ---: |
| 4096 / 64 | 217.6 | 119.5 | 114.8 |
| 16384 / 64 | 834.2 | 436.2 | 423.1 |
| 4096 / 4096 | 374.5 | 350.4 | 410.6 |

Sparse memory savings are 45-48%; dense savings about 6%. Earlier sparse-only
prototype regressed on the dense scenario (374.5 to 415.9 KiB), motivating
the new direct-index path and in-place sorting. tracemalloc is NOT a complete
physical-memory census; current RSS deltas were measured separately.

The downloadable research archive includes raw per-process JSON for both
source versions, speed and memory, including the intermediate negative case.
The checked-in summary is evidence/qa-compact-summary-local-py313.json.
Do not promote this optimization by default.

## Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -q
    python -m alexpy.qa_native_bench --sessions 3 --pairs 3 --batch-sizes 1,8,64,256 --json qa-speed.json
    python -m alexpy.qa_field_audit --cases medium-mixed,large-sparse --batch-sizes 64 --sessions 3 --json qa-memory.json
    python -m alexpy.qa_field_audit --cases medium-mixed --batch-sizes 4096 --sessions 2 --json qa-dense.json

For the old C implementation, use separate source checkout b28c87d.
Do not import two native builds in one interpreter. CI tests Python 3.13
and 3.15 on Ubuntu, Windows and macOS; no production QA log study here.
