# Closed competitive splice comparison: resource review

Evidence-only review, 2026-09-06, of frozen source
`db39a6cdcb331cb69352fbb56cc0344d505808a5`. The full run and independent audit
finished in **135.7729329 seconds**, with a maximum observed owned-tree peak of
**1,522.5390625 MiB**. Completed admission projected 583.707702804179 seconds
and 1,879.48828125 MiB: actual use was 447.934769904179 seconds and
356.94921875 MiB below those projections. The fixed experiment is closed.
No speed change is warranted by this review alone.

The review reads saved JSON and streams existing receipts. It loads no tensor
checkpoint, constructs no model, runs no neuronal computation or action, and
does not replay the independent audit. Input hashes, scalar extraction and work
totals are in [the saved summary](../../.pytest-tmp/neuronal-splice-online-v2-resource-summary.json),
SHA `e75a7f12eeeac35b391ffa8c0f1d5a2973134a526fab37c1a3fe36c473c03caf`.
Three extraction corrections concerning Windows path spelling, receipt status
spelling and the worker journal directory are recorded there; none changed
benchmark artifacts or caused experimental exposure.

## Disjoint timing account

| Full workload interval | Seconds |
|---|---:|
| Acquisition model and custody work, M | 25.316956300 |
| Acquisition apply/execute/grade/action receipt, A | 54.754877300 |
| Acquisition learning receipt and timing publication, J | 9.175232900 |
| Endpoint package/export/synchronous exchanges | 21.905398400 |
| Other work inside the parent report | 6.800449600 |
| Run watchdog duration minus parent report | 2.505284400 |
| Independent audit watchdog duration | 15.314734000 |
| **Total** | **135.772932900** |

The run watchdog records 120.4581989 seconds; the parent reports
117.952914499998. The audit's internal timer is 14.132959700000 seconds.
Its regular learning-pair checks total 5.840004200072 seconds, nested within
that audit duration. M includes source-defined validation, hashing, prediction
and corrective teaching; it is not a pure neural-kernel timer. J includes full
compaction, guards, serialization, flushing, hashing and timing-row publication.
The raw timing rows reproduce the reported M/A sums; their J values stop before
their own append, so the complete J totals come from the bound report.

Endpoint worker wall time is 20.969966600009 seconds, nested inside the
21.905398400006-second package/export interval. Worker subintervals total
2.558979799869 model, 1.243088500007 reading, 2.405642100057 receipt and
11.223453199957 parent-wait seconds. Parent wait overlaps actual action execution;
these numbers must not be added to the disjoint table. The saved timers do not
separately isolate checkpoint writing, source packaging, checkpoint loading or
process startup. The remaining parent interval includes the 0.472808399994-second
pre-reading model panel plus ingestion, setup and other custody/publication work.

## Measured bottleneck and matched arms

Acquisition A is 40.33% of complete run-plus-audit time, compared with M at
18.65% and J at 6.76%. The qualified executor's saved intervals sum to
54.320715300186 seconds for 1,280 acquisition verifications and
10.195576899961 seconds for 240 endpoint verifications. Those intervals include
materialization, child execution and verification; they do not measure child
startup separately. Acquisition executor median/p95/max were
41.00645/53.09230/82.89720 ms; endpoint values were
40.51485/55.23130/84.70480 ms. No contemporaneous evidence identifies contention
or a particular internal function as the cause of outliers.

| Arm, 256 acquisition pairs each | M seconds | A seconds | J seconds | Lesson JSONL bytes |
|---|---:|---:|---:|---:|
| learned | 4.958393 | 10.923072 | 2.114581 | 20,418,764 |
| action_frozen | 5.115584 | 10.961116 | 2.133973 | 20,523,860 |
| slow_frozen | 5.158916 | 10.916241 | 2.469819 | 24,854,395 |
| memory_disabled | 5.167702 | 10.985052 | 1.713433 | 11,481,117 |
| marginal_reference | 4.916362 | 10.969395 | 0.743427 | 8,496,666 |

Learned competitive free receipts report 304,416 comparator pairs and 317,024
threshold evaluations. Learned and marginal each evaluate 12,608 occupied codec
slots during acquisition free recall. Their cursor advances are 628 and 2,516,
respectively, but free output lengths and action paths differ. M does not show a
measured competitive speed advantage. Complete competition traces add storage
and validation work; their larger J is not evidence that any trace may be omitted.
There is no aggregate v1-versus-v2 causal speed comparison here.

The reusable executor is the largest measured interval, but these receipts do
not identify a supported internal change. Keep its qualified isolation and all
neuronal/work receipts unchanged. Prioritize the learning review; do not start
an optimization experiment or repeat this cost merely to seek a faster sample.

## Projection, storage and coverage

Independent scalar arithmetic reproduces the admitted run projection:
404.030259206891 seconds from replicated separate M/A/J maxima, plus
66.195528199947 doubled fixed residual and 30 seconds of guard, totaling
500.225787406838. The separately finished cost-audit monitor gives
83.481915397341 projected audit seconds. Measured cost and its audit took
34.3996267 and 11.1646145 seconds. Their observed peaks were
1,524.40234375 and 553.96875 MiB. The run memory estimate uses the larger of
the cost owned-tree peak and parent-plus-worker peaks, then adds 24 MiB of
receipt allowance and 128 MiB reserve. Audit adds 24+32 MiB to its measured peak,
giving 609.96875 MiB. These are engineering estimates, not allocator guarantees.

Full acquisition action maxima exceeded the selected raw cost maxima by
1.575–1.789 times across all five arms. Model and journal maxima stayed below
their respective cost/cap maxima. The final aggregate stayed comfortably within
the doubled estimate and watchdog ceilings; sampled maxima are not exact
per-action upper bounds.

The full audit records 1,280 learning pairs, 4,018 acquisition free byte/EOS
decisions, 2,400 reader offers, 3,200 action labels, 240 endpoint proposals with
868 payload decisions, 200 reading-panel calls and 1,520 actual verifications.
The cost included 20 regular pairs, five disposable caps and four category
actions, with all 240 endpoints and 200 panel calls, totaling 269 verifications.
Both retained 360 tapes and 239,657,600 tensor bytes (228.5553 MiB), authenticated
by the completed audit; this review did not reload that payload.

The six full JSONL streams total 119,643,199 bytes (114.1006 MiB), including
85,774,802 lesson bytes and 14,935,392 endpoint proposal/acknowledgement bytes.
The largest full line was 229,422 bytes and largest reported row tree 612,924
bytes. The cost cap reached a 340,650-byte line and 1,041,093-byte row tree.
All are below the prospectively fixed 2 MiB per-line/per-tree guards. Serialized
file size and tensor payload are not resident-memory measurements. Full audit
peak was 551.79296875 MiB; full run peak was 1,522.5390625 MiB.

## Bound evidence

The summary authenticates all six cost-decision inputs, the full report,
completion, admission and monitors, and every analyzed JSONL byte hash. It checks
the report/audit/admission/source hash relationships and completed statuses.
Scientific validity, process creation custody and tensor invariants remain claims
of the already completed independent audit, not a newly executed audit here.

| Receipt | SHA-256 |
|---|---|
| Full report | `52edd52de06776a253de4462c796ebe74f44fcaa1dd6659185f1c475ed644de8` |
| Full independent audit | `d31238776795984c6ac58fa9c6b2549323e603b795a652044bb01dacd16322db` |
| Full run monitor | `b20c6f96d6530014834dbbfd92aba72d10af3777a4dead9831daeb852a3e0292` |
| Full audit monitor | `d0e670125ee7bff70d33a40fd6ed12c460edb967a83b135887b53a84c0dc2bf7` |
| Completed cost decision | `fcd5657ae350ea5842ffe433b3724dc29f7a101fd18c9090bffda9e37ffb1e06` |
