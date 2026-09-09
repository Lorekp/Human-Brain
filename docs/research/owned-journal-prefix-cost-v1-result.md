# Owned journal-prefix cost result

2026-09-09. **Completed matched journal-path measurement** at source revision
`195bc0e4dcf8ab2d1c65d28170f93b415265030f`. The
[saved receipt](../../.pytest-tmp/stable-credit-journal-cost-v1/receipt.json)
records identical row and frontier results for ordinary prefix verification and
owned append journals on the same authenticated V8 schedule. This measures the
selected journal path, not complete checkpoint or learner speed.

Both lanes copy the same 270 saved rows across 265 boundary positions:
192 attempts, six teacher records and 72 ingests, totaling 420,178,380 bytes per
lane. These are historical journal bytes used for storage measurement, not new
learner exposure. The run constructs zero models, decodes zero tensors and
executes zero task children. This report reads only the saved cost and launch
receipts; it does not repeat either lane.

| Recorded scope/counter | Ordinary | Owned |
|---|---:|---:|
| Complete scheduled journal path, seconds | 369.0633679 | 24.3657044 |
| Included append interval, seconds | 9.1783096 | 8.7093226 |
| Fresh final verification, seconds | 0.4833017 | 0.5635055 |
| Scheduled full-prefix hash bytes read | 276,856,555,770 | 0 |
| Scheduled full-prefix passes / cached checks | 3,969 passes | 3,969 cached checks |
| Actual newly written tail bytes authenticated | — | 420,178,380 |
| Avoided scheduled prefix-hash bytes | 0 | 276,856,555,770 |
| Fresh final bytes read | 420,178,380 | 420,178,380 |
| Fresh final full passes | 3 | 3 |
| Actual journal bytes written | 420,178,380 | 420,178,380 |

The owned lane uses three native writers with 270 write-through completions,
578 actual tail reads, 8,481 identity checks and 1,587 freeze scopes. All three
writers close. The ordinary lane records 270 writes and 270 fsync calls. Native
write-through is not relabeled fsync. Each lane also reads 420,178,380 saved row
bytes to supply its writes; these are separate from prefix verification and
fresh final reads. The append interval is contained in the scheduled path and
must not be added to it again. These software byte counters are not measured
physical-device traffic.

Comparison of the saved records confirms equality of all 270 row
coordinates/lengths/SHA256 values, all 265 boundary frontier records, and the
fresh final frontiers. After writer closure, both lanes perform complete fresh
reads with the following matching results:

| Journal | Rows | Bytes | Final SHA256 |
|---|---:|---:|---|
| attempts | 192 | 311,424,915 | `3efbc004382b496602325d9c77b2f5a6e3fcb8148f322c31161fce7ef7a81a5c` |
| teachers | 6 | 4,279,179 | `e81b8cc340341605bc7a31a56268536f9d34273a2e0892974ad826473be5e9aa` |
| ingests | 72 | 104,474,286 | `f8506bfbc14eeaab4e2a98f50234adeb4c0e9d65937781bbf7e336c2925fc362` |

The observed scheduled-path ratio is 15.1468×, with 344.6976635 fewer seconds
in the owned lane. This is one serial pair in fixed **ordinary then owned**
order, without counterbalanced repeats or controlled OS cache state. Both lanes
use the same source bytes and boundary schedule, but the later lane may benefit
from prior reads. The result supports a measured reduction for this selected
path under these conditions. It does not establish a general latency ratio,
physical I/O reduction, complete checkpoint speedup or broader-session speedup.
Tensor hashing, actor snapshots, payload serialization and actual learning/action
work are outside the measured schedule.

Authentication takes a separate 1.7873505 seconds, reading 456,401,339 bytes in
1,073 passes and checking 711 checkpoint inventory entries. The internal elapsed
time is 396.2719530 seconds through final checks, excluding receipt write/fsync.
The [outer monitor](../../.pytest-tmp/stable-credit-journal-cost-v1.launch.json)
reports exit zero, **396.7159483 seconds and 97.71484375 MiB**, within its
900-second/4096-MiB limits. Its memory measure is the largest sampled simultaneous
sum of live owned process peak working sets at 500 ms intervals, not a sum of
independent process maxima; brief processes can be missed. The measured process
itself reports 84.29296875 MiB. The clock is `perf_counter` backed by
`QueryPerformanceCounter`, with 100 ns reported resolution. Both lane copies
retain 840,356,760 bytes before receipt publication, below the 1-GiB artifact cap.

The saved receipt binds cost-source SHA256
`8a8ea09e39026541358892fa1cd7ff60d70e2081a4bc7c0b2a2d963221161de9`
for `experiments/r4_integrated/persistent_journal_prefix_cost_v1.py`, and
`b46f4f222634fae2d19ebb6e7c9dd6485b1a52dfd472c853e871bb6d70abcd1d`
for the owned journal implementation. Its complete source map and authenticated
V8 bank identity remain in that receipt. The receipt SHA256 is
`1a98f93d9f567e06a17da8db2628e8d2759d878322e2ff8cd10b4ef3780ec302`;
the outer launch receipt SHA256 is
`9b3318a411f7370ab334aa9173cc52048b9cd75c58aee4699559d1b441bb5128`.
