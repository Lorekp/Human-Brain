# Persistent language/action v1 resource review

The completed session and saved audit took **19.662127 seconds** together:
18.9237751 seconds for the run and 0.7383519 seconds for the audit. Their observed
process-tree peaks were 364.921875 and 35.44921875 MiB respectively. This fits
the selected complete 900-second / 2048-MiB envelope. The run used an 840-second
watchdog and reserved 60 seconds for audit; these were limits, not measured cost
projections.

This review aggregates saved JSON and source-bound work counters. It hashes the
final checkpoint as bytes but never deserializes it, imports a learner, reruns an
audit, or executes an action. The saved audit covers behavior and state custody;
neither audit establishes a replay of neuronal arithmetic.

## Recorded time and exposure

| Route | Attempts / seconds | Teacher rows / seconds | Ingest rows / seconds |
| --- | ---: | ---: | ---: |
| Online | 72 / 4.9351526 | 24 / 0.6998856 | 13 / 0.0702908 |
| Learning disabled | 72 / 4.7040654 | 24 / 0.6798916 | 13 / 0.0703564 |
| Simple reference | 72 / 2.0342246 | 24 / 0.0020336 | 13 / 0.0006197 |
| Total | 216 / 11.6734426 | 72 / 1.3818108 | 39 / 0.1412669 |

The 144 edit attempts each have an actual execution receipt, including unchanged
workspaces following rejected or unsupported proposals. Their executor intervals
sum to 6.0069075 seconds, of which 5.5301012 seconds are nested child intervals.
These times are already included in the attempt column. There were 78 accepted
SPLICEs, 37 accepted STOPs, 25 rejected SPLICEs and four unsupported proposals;
no child timeout or output limit was reached. The audit's `actual_executions: 0`
describes the saved auditor's own exposure, not the session.

The three disjoint row timer sums total 13.1965203 seconds. Each timer ends before
its journal append. Completion reports 18.0978235 seconds, leaving 4.9013032
seconds for unpartitioned initialization, serialization, hashing, summaries,
checkpoint export and other work. The outer monitor adds 0.8259516 seconds beyond
that completion timer. There is no separate model-only or journal-only timer;
the residual must not be labeled entirely as either. The longest recorded
attempt was 0.1390979 seconds, learning-disabled recovery X00 attempt three.

Each neuronal route appended 437 raw bytes plus 16 EOS symbols to its single
tape, reaching 453 of 512 slots. The simple reference received the same notes.
The two neuronal routes made 136 free queries and 48 corrective queries together;
eight unsupported requests caused no neuronal query. Their corrections offered
208 reader symbols including EOS and 72 action-head labels. Only the online
route enabled the 24 address and 24 codec update groups and 12 updates per action
head. The disabled route still performed its declared corrective forward work.

The saved counters total 528 contiguous feature-block evaluations, 57,008,128
lag pairs, 1,083,154,432 coincidence thresholds and 3,078,438,912 coincidence input
applications. Initial competition evaluated 9,852,628 pair comparators; the full
codec banks evaluated 55,672 slots and 14,307,704 output potentials. These are
operation counts, not resident cell populations, isolated timings or newly
verified arithmetic. The 216,039,424 reported feature-assembly bytes are
cumulative allocations across calls, not peak memory.

## Storage and limits

Initialization loaded the authenticated previous payload once: 109,896,320
tensor bytes (104.8053 MiB), 1,480 tensors spanning five model containers and 360
tapes. It retained only the selected learned numeric model state, discarded all
old tapes, and constructed the two current models. Their combined model tensors
occupy 1,268,864 bytes; two new 512-slot tapes add 1,650,704 bytes. The final
checkpoint therefore declares 2,919,568 tensor bytes and occupies 2,931,675 bytes
on disk. This is prior-trained initialization, not fresh training.

The three journals occupy 4,416,521 bytes; the largest individual attempt line
is 35,864 bytes. Serialized sizes do not bound the Python object graph retained
by the runner. The monitor samples live owned process peak working sets every
500 ms, so brief child peaks can be missed. No per-stage peak or allocation
profile exists.

The fixed workload has ample observed headroom. A speed change is not warranted
by these measurements before addressing useful language/action behavior. The
executor is a material measured interval, but this does not isolate startup or
justify an implementation change. Nor does a roughly 20-second completed
session establish a worst-case bound for longer streams, larger workspaces or
longer instructions. Existing query and output limits remain explicit; no
cross-cohort or causal speed comparison is made.

## Evidence bindings

Exact route totals, work dictionaries, categories and all eleven input hashes
are in `.pytest-tmp/persistent-language-action-v1-resource-review.json`, SHA-256
`4842eb4949171f7b0a52da3c06516b68f71e4f82a7783c5a643ca37d6fd7cd45`.
The run source was frozen at `032d908`; the saved source inventory is bound by
the completion receipt.

| Evidence | SHA-256 |
| --- | --- |
| Run report | `b71126d1c2150c5b550e87f1aac3446dd04838d9a435df7baec39685d964fda1` |
| Completion | `f167a75e394857df7248e94b357f366f4ec4d9fc8ba728b538df4d6fc873275d` |
| Run monitor | `7267a200e39ff7a993b55e5b2819af60ffccac37ceb71f83feeab8450363f259` |
| Saved audit | `366a355aa2742d2390ada33d72e01cd3fce3c5472f832389a52a0661e6f25652` |
| Audit monitor | `651d9df83171539438b5d58d839cc63f571a5a56537499ed9ce1f18da157e0a7` |
| Final checkpoint bytes | `e0c581ab907310e846a256fbd868c667e93c8d6cf96626c60b1177d3cdf3da7b` |
