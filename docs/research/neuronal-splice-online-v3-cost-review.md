# Completed byte-coincidence comparison: resource review

Saved-evidence review, 2026-09-06. Frozen source:
`4840c24d8e4265eaddeaea032231d52d2af3185d`.

The full workload and independent audit completed in **157.9033546 seconds**,
with a maximum observed owned-process-tree peak of **1,047.4375 MiB**. Admission
allowed 673.2005430127208 projected seconds and 1,300.0625 MiB, below the fixed
900-second/2,048-MiB ceilings. The observed margins against that projection were
515.2971884 seconds and 252.625 MiB. No speed change is needed to accommodate this
completed workload. Resource success does not support adopting the feature:
the separately banked fresh-task result is 4/32 versus 5/32 for its reference.

## Completed scope and admission

| Phase | Outer seconds | Observed tree peak, MiB |
| --- | ---: | ---: |
| Complete cost | 34.1233798 | 1,148.0625 |
| Independent cost audit | 15.0050991 | 444.80078125 |
| Full acquisition and endpoints | 136.0286610 | 1,047.4375 |
| Independent full audit | 21.8746936 | 444.23828125 |

The completed cost decision projected 572.1458468179275 run seconds plus
101.05469619479335 audit seconds. Cost included 20 regular acquisitions, five
authored geometry caps, 240 endpoints and four execution-category fixtures:
269 actual verifications and 200 old-reader calls. Full execution included
1,280 acquisitions, 240 endpoints and the same 200 reader calls: 1,520 actual
verifications. Its audit counted 2,400 reader teaching offers and 3,200 action
labels. Cost/full reader queries were 485/2,680, including free and teaching
queries; category fixtures were not neuronal predictions.

Each phase constructed five parent and five sequential worker models. Cost
also made five disposable cap clones. Full retained 72 freshly ingested tapes
per arm, totaling 20,950 raw-byte/EOS symbols; cost added 640 cap symbols.
Original-prior and final-payload loads were included in both cost and full,
and in their audits. No historical model was executed as a reference.

## Where the full time went

| Disjoint run stage | Seconds | Share of run |
| --- | ---: | ---: |
| Acquisition model work, M | 34.3144661 | 25.23% |
| Acquisition apply/execute/grade/receipt, A | 56.3761838 | 41.44% |
| Acquisition learning journal, J | 12.1658126 | 8.94% |
| Remaining outer time | 33.1721985 | 24.39% |

The residual includes initialization, ingestion, panels, export and endpoint
exchange. The saved export/exchange interval was 22.7758203 seconds; worker
wall time was 22.2486045 seconds. These are nested intervals, not additional
disjoint charges. Worker summaries included 3.2797169 seconds of endpoint model
work, 11.2173446 waiting for parent execution/acknowledgment, 1.7088372 reading
and 2.6995532 receipt work. The audit spent 6.8737663 seconds checking regular
pairs within its 21.8746936-second outer duration.

All five full-run action maxima exceeded their unmultiplied sampled cost
guards. The largest was learned index 7 at 131.829 ms; its model interval was
105.029 ms, slightly above the 99.998-ms cap-derived model guard. Complete
journal maxima reached 345.129 ms for slow-frozen and 327.216 ms for the
reference. Raw timing rows locate the corresponding partial-journal maxima at
indices 177 and 178; they do not independently identify the complete-J maximum
because their own append is timed afterward. These outliers demonstrate why
sampled maxima plus reserves are engineering estimates, not per-call bounds.
There is no contemporaneous evidence here assigning them to contention or I/O.

The learned/reference mean model intervals were approximately 28.78/20.64 ms
per acquisition. Those whole-lesson measurements include different generated
trajectories and feedback work. They are not isolated feature-kernel timings,
and aggregate elapsed times from earlier cohorts are not causal speed baselines.

## Storage and bounded receipts

Each model contains 634,432 tensor bytes. Candidate Boolean-key tapes contain
206,344 bytes each; graded-key tapes contain 656,904. Four candidate arms and
one graded arm, with 72 tapes each, therefore retain exactly **109,896,320
tensor bytes (104.8053 MiB)**. Against five graded arms at the same shapes, the
nominal tape reduction is 129,761,280 bytes (123.75 MiB). This is allocation
arithmetic, not an isolated RSS measurement.

The full learning journal held 1,280 rows totaling 98,698,173 bytes. Its largest
line/tree were 179,072/369,405 bytes. Across the reported full journals, the
largest line/tree were 233,621/635,982 bytes in the worker endpoint journal.
The prospective 2-MiB line/tree guards and common 24-MiB buffer allowance were
unchanged. Complete competition traces remained serialized; fixed feature
gate arrays were not added to journals. Their implementation arithmetic was
covered by the already banked source/component QA and audited work receipts,
not recomputed by this resource review.

## Decision and evidence limits

Retain the existing qualified executor and complete streamed receipts. Action
execution is the largest measured acquisition stage, but its timer combines
materialization, child execution, grading and receipt publication. These saved
totals cannot identify which reusable source optimization would pay off.
No speed implementation or additional profiling is warranted by this closed,
budget-compliant negative feature result alone.

This review used saved JSON summaries and streamed only the small 238,877-byte
timing journal. It checked the completed report/audit/source hash chains and
timing metadata, without importing project validators, rerunning audits,
loading checkpoints, evaluating neurons or executing actions. All input hashes
and exact derived scalars are in
`.pytest-tmp/neuronal-splice-online-v3-resource-review-v1.json`, SHA256
`4be24b6abf75028c556692bbe051f1bb1b91de58f1ce0c727e444e08d5e04aab`.

Authoritative final decision:
`benchmarks/neuronal_splice_online_v3.decision.json`, SHA256
`807e256dbb750a41e24a00997bee89dbfb7ad4d6e2ee7ab3dc556686f93770a8`.
Completed cost decision SHA256:
`59120c40d6c91800188c4812f6ddfd9915d1bcd111b659e8d0e05c6b808eeeac`.
Full report SHA256:
`423ebbf80219ff4e67e3dca7699782467f74581ee86c31756233d0de3ca70b18`.
Independent full audit SHA256:
`0484aeed3f3b1b05a0c0ee357cb846b6c1d8dfa86c83dfc94c2c2915f0b80474`.
