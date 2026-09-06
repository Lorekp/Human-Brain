# Neuronal splice online v1: completed resource evidence

2026-09-06. Saved JSON/scalar review only. No neuronal model, checkpoint,
action, audit replay or timing probe ran for this review. The fixed scientific
comparison is closed; these observations do not change its admission or guards.

The full run took **84.7433100 s**, followed by **11.3985175 s** for its passed
independent audit: **96.1418275 s combined**, versus **418.095431802003 s**
admitted (372.734282006361 run + 45.361149795642 audit). Remaining combined
time to the 900 s ceiling was **803.8581725 s**.

The run's observed owned-tree peak was **1234.234375 MiB**, the audit's
**481.8359375 MiB**. Their maximum is **266.265625 MiB** below the admitted
1500.5 MiB envelope and **813.765625 MiB** below the 2048 MiB ceiling. Parent
and worker individual peaks were 743.7890625 and 630.2578125 MiB; their sum,
1374.046875 MiB, is conservative and is not an observed simultaneous peak.
The audit's separate admitted memory component was 524.87890625 MiB.

## Where the time went

All 1,024 acquisition pairs are included. The projection column is the frozen
512-times-sum-of-four-guarded-maxima term, before fixed overhead and 30 s reserve.

| Acquisition stage | Actual seconds | Share of run | Projected stage seconds |
|---|---:|---:|---:|
| Model, validation and custody hashing | 18.635976800 | 21.9911% | 161.959526405 |
| Apply, execute, grade and action receipt | 41.244499600 | 48.6699% | 85.710438393 |
| Learning receipt and timing-row publication | 2.775853300 | 3.2756% | 49.114624009 |
| Remaining monitored run time | 22.086980300 | 26.0634% | Included in the separate residual/reserve |

Model time splits into 8.494032900 s before execution and 10.141943900 s
for correction plus final state hashes. It does not isolate neuronal kernels.
The nested executor interval accounts for 40.912304000 s of acquisition action
time; it includes workspace materialization, process execution and verification,
so it is not a measurement of child startup alone. Journal publication before
the small timing-row append took 2.739942900 s; the aggregate append difference
was 0.035910400 s.

The endpoint package took 14.752137700 s, containing 14.101262900 s of worker
wall time. Worker totals were 2.079138500 s model, 0.467641800 s receipt,
8.041913100 s waiting for the parent, and 0.499436100 s reading-panel work.
Endpoint executor intervals totaled 7.816872200 s. These nested scopes overlap:
particularly parent execution and worker waiting must not be added again.
Audit acquisition-pair validation totaled 2.908093300 s; the remaining
8.490424200 s includes its other validation, loading and monitor overhead.

## Coverage and limits

The passed audit reports 1,216 actual verifications, 1,024 learning pairs,
1,920 reader offers, 2,560 action labels, 192 endpoint proposals and 160 old-panel
calls. Acquisition used 1,124 free reader decisions; endpoints used 268.

Cost caps supplied all model and journal maxima. They were deliberately much
larger than ordinary geometry. Actual acquisition action maxima exceeded their
cost guards in **296/1,024** cases (learned 42, action-frozen 68, slow-frozen 114,
memory-disabled 72). Per-arm actual maxima were respectively
79.263100, 63.957100, 71.507400 and 95.678500 ms; cost guards were
44.404400, 42.054200, 39.285000 and 41.659600 ms. The aggregate estimate covered
the completed workload, but sampled maxima were not per-call upper bounds.
No cause such as contention is inferred from these observations.

The learning journal contained 33,424,356 bytes across 1,024 rows. Its largest
line was 82,799 bytes and largest reported writer tree 154,712 bytes, below the
prospective 1 MiB guards. Action journal size was 4,751,817 bytes/1,216 rows.
All four parent journals were streamed here and their complete hashes/counts
checked against the report; no growing row history was retained.

**No speed change is worthwhile now on this evidence.** Removing duplicate
worker serialization could save only part of the entire 0.467641800 s receipt
component (0.5518% of run time). Execution is the larger component, but current
headroom supplies no resource blocker and saved intervals do not identify a
specific safe executor improvement. Preserve the implementation and prioritize
the scientific outcome; no new profiling or optimization is selected here.

## Evidence bindings

The report/completion/audit chain, both passed monitor statuses and actual
interpreter/redirector creation windows were checked from saved JSON. All six
cost-decision input hashes were rechecked. This is a resource interpretation of
the completed independent audit, not another scientific audit.

| Artifact | SHA-256 |
|---|---|
| `.pytest-tmp/neuronal-splice-online-v1-run/report.json` | `862918bf7116d88baf337965c3e8177fad02efb40021628b0248afcba6264a4f` |
| `.pytest-tmp/neuronal-splice-online-v1-run/completion.json` | `86727a06c0fc89e4f9bfb9dd2152babe9a6176d725c496e8e357b2712c233a62` |
| `.pytest-tmp/neuronal-splice-online-v1-run-independent-audit.json` | `96cbf7f155beb8efea0f3541cce535a7ab727661293e93e9525f0312af7afe6d` |
| `.pytest-tmp/neuronal-splice-online-v1-run.launch.json` | `7c833e2b53db4b5136a4b40a725101be1b9417557daf1e1734a7368cac613aea` |
| `.pytest-tmp/neuronal-splice-online-v1-run-audit.launch.json` | `a0778a737c07215dbabcfa599d1339e44809bf8c43351b9c3362fd620fad89fb` |
| `benchmarks/neuronal_splice_online_v1.cost_decision.json` | `4002465df81acc7ed99a8ec3ea625dd664e71d2e6ea7c61b8cdc5c166ca4a749` |
| `.pytest-tmp/neuronal-splice-online-v1-cost/report.json` | `76c48ac8eea6257f850356a02d3ea0d2953a715ee86319a7a86544956474ed27` |

The authenticated run source manifest binds checker
`ac7bc876bc72187daccb5bd148507366d20244b60b17c1264221ac1325e218b5`.
Durations above retain nine decimal places where useful; admission retained its
original full-precision JSON numbers. Resource envelopes are engineering
estimates, not formal allocator or worst-case timing proofs.
