# Persistent language/action v2: completed resource review

Status: saved-evidence review, 2026-09-07. The completed run and independent
saved review took **117.971022 seconds combined**, with a maximum observed
owned-process-tree peak of **406.921875 MiB**. This is 13.11% of the 900-second
complete budget and 19.87% of the 2,048-MiB ceiling. Runtime is not currently a
reason to postpone broader learning development. These measurements establish
resource feasibility for this completed workload, not useful competence.

The result bank is
[persistent_language_action_v2.result.json](../../benchmarks/persistent_language_action_v2.result.json),
source revision `5d9c3027533f5516102604ebd0ade20f6abae90d`.
This review streamed saved JSONL once, read saved scalar reports, and hashed
files. It did not import project modules, deserialize checkpoints, run neurons,
execute actions, replay the independent auditor, or repeat QA.

## Measured stages

| Scope | Online seconds | Learning-disabled seconds | Simple reference seconds | Total seconds |
|---|---:|---:|---:|---:|
| Attempts, 72 per route | 46.8589929 | 46.5904618 | 1.7961476 | 95.2456023 |
| Corrections, 24 per route | 7.0540504 | 7.0290789 | 0.0040127 | 14.0871420 |
| New-document ingests, 2 per route | 0.0123053 | 0.0106797 | 0.0002194 | 0.0232044 |

The 294 row timers sum to **109.3559487 seconds**. The completion clock is
114.9276552 seconds, leaving **5.5717065 seconds unallocated** among initialization,
baseline restoration, journal encoding/flush/fsync, final summaries, checkpoint
export, source/file hashing and other orchestration. The monitor adds another
0.7849836 seconds beyond that clock, producing **115.7126388 seconds** for the
run. The separately monitored saved review took **2.2583832 seconds** and peaked
at **205.03515625 MiB**. Serial memory peaks are compared by maximum, not added.

Attempt time includes actual inference, receipt conversion, state hashing,
grading and execution when applicable. Teacher time likewise includes its
receipt conversion and custody checks. Neither is an isolated neural-kernel
timer. Journal append occurs after each corresponding row timer is captured.
There is no separate measured initialization, pure model-compute or journal
stage, so allocating the residual more finely would invent evidence.

The **144 actual workspace verifications** consumed **5.2321315 seconds** inside
attempt timers; their nested child intervals sum to **4.8261622 seconds**. Do
not add either quantity again. There were no recorded timeouts or output-limit
failures. Application acceptance was 0/48 online, 1/48 learning-disabled and
48/48 simple; rejected proposals still received unchanged-workspace verification.
Acceptance is an execution category, not task correctness.

Across the two neuronal routes, attempt time was 93.4494547 seconds, including
3.5318557 seconds of execution. Its **89.9175990-second non-executor remainder**
is the dominant measured stage; it still includes conversion and hashing. The
largest attempt was online `X05`, 0.8051979 seconds; the largest correction was
online `A06-E`, 0.5832017 seconds. These are observed maxima, not future bounds.

## Actual circuit work and retained state

Each neuronal route made 68 supported free calls; four overlength requests per
route remained unsupported. The saved work contains **1,108 output decisions
per neuronal route**, 2,216 total, with a fresh full-address competition at
each output. There were **277,776,960 pair-comparator evaluations** in total.
Free output evaluated 2,216 selected conditional populations, contributing
166,297,504 output-synapse products and 18,153,472 fixed signed afferent
applications. All candidate addresses remained eligible.

Each route received **98 byte/EOS teacher offers** and 36 action labels in its
24 corrections. Teaching evaluated **96,280 candidate-prefix populations**
across both routes: 54 offers per route at 480 occupied slots, then 44 at 505.
This gives exactly **7,225,236,320 conditional-output forward products**, the
same number of local output-update products, and **788,725,760 fixed signed
afferent applications**. The disabled route calculates the offered local
updates but publishes none; its similar teacher time is consistent with that
source-defined workload. This is not an isolated estimate of publication cost.

Free address construction records 1,206,845,440 lag-pair evaluations and
22,930,063,360 AND/match threshold evaluations. Teacher address construction
records 122,798,080 lag pairs and 2,333,163,520 thresholds. Named coincidence
buffer allocation counters total 73,147,191,296 bytes for free calls and
7,442,832,512 bytes for teachers; these are cumulative allocation-volume
counters, **not simultaneous storage or measured memory traffic**. The full
teacher's larger output matrix workload therefore does not imply that teaching
dominates elapsed time: there are far fewer teacher prefixes than free outputs,
and the output products are batched matrix operations. No kernel-level timing
is available to attribute the free-stage time specifically to coincidence,
competition, validation, allocation or hashing.

Initialization loaded the prior checkpoint once: **2,919,568 declared tensor
bytes**, two saved model containers and two saved tape containers. It selected
the prior online state and constructed two new combined models/readers/action
circuits, preserving old numeric tensors and restoring the same 453-slot tape
and experience counter. New prefix/output weights and fixed wiring change
the free function; numerical prior preservation is not functional parity.
This run did not reload the older 109,896,320-byte five-arm source checkpoint.

The two retained models total **2,625,216 tensor bytes**. Two allocated
512-slot tapes contribute **1,650,704 bytes**, yielding **4,275,920 final tensor
bytes**. Both memories grew 453→480→505 slots through two appended documents
(50 raw bytes plus two EOS each), with no reset or per-task tape selection.
The actual final checkpoint file is **4,291,715 bytes**; its hash was checked
without deserialization. Tensor-byte counts come from the authenticated
initialization/report and source shapes, not a new tensor audit.

The runner also retains all 216 parsed attempt summaries for final metrics;
its history storage is not constant. `attempts.jsonl` is 71,698,136 bytes,
`teachers.jsonl` 5,132,048 bytes, and `ingests.jsonl` 5,904 bytes. Maximum observed
lines were 555,600, 206,905 and 1,194 bytes respectively. JSON file size,
Python-object retention, temporary circuit activity and tensor payload are
different quantities. The monitor samples live owned process peaks at 500 ms;
brief children can be missed, so the observed peak is not a proof of every
instantaneous allocation bound.

## Comparison and next use

The [v1 resource review](persistent-language-action-v1-cost-review.md) measured
19.662127 seconds complete and 364.921875 MiB. V2 is **5.9999115 times** that
elapsed total and **1.1150931 times** that peak. Both have 216 attempts and
144 actual verifications, but this is not a controlled performance comparison:
the manifest, prior load, retained memory, output trajectories, baseline
restoration and learning mechanism changed. V1 ingested from empty to 453 slots
and offered 104 teacher symbols per neuronal route; v2 continues to 505 and
offers 98. V2 also reretrieves every output and teaches all candidates with the
new conditional population. The ratio is descriptive, not an isolated speed
regression caused by any one change.

No additional speed experiment is justified by this completed budget alone.
If longer sessions make memory retention limiting, one reusable source change
is to replace the runner's retained full attempt rows with only the scalar
outcome fields needed for final metrics, leaving full circuit receipts in the
append-only journal. That changes host history storage, not neuronal selection,
learning or receipt contents. Its benefit is prospective and unmeasured; it
does not explain or repair the observed capability failures.

## Evidence identity

All **288 source files** matched the saved source inventory at review time.
Completion hashes matched the three journals, report, initialization, manifest,
inventory and final checkpoint. The saved auditor binds the same completion
and run monitor and reports zero model constructions, checkpoint loads and
executions. The independent review's scientific scope excludes replay of
teacher conditional arithmetic and neuron/features.

| Evidence | SHA-256 |
|---|---|
| Run monitor | `b17da5b7c6b39e6c5560930c9aa41de43f379bdb2ba00a098ab70efe5160daf9` |
| Review monitor | `4bdc33c20d871c999a56868a6b10f8f8804e8e77dc68e5df990f25137a296b69` |
| Report | `04f2d46ac72ff9c4bfbadcdfe6caf0944463e63903644fd9035109da9b5b8526` |
| Completion | `716159dae22bb90adf17caf70250c184b6039cfde314649d1c83ed05c31e4a3e` |
| Saved review | `ea54da72c5b245f3c1019e84d66eac8129d55298cd436db6c8e75318225fed12` |
| Source inventory | `54816da83b461f228e353dffe2c8d91a11589ce9abb4f4343e7136cd595c87b2` |
| Final checkpoint | `bdc2db89692fee5d49b305935020199c06e8cf0282324e5c1a15bfe6fd9db860` |

The full input hashes, per-route stages and all aggregated work counters are in
`.pytest-tmp/persistent-language-action-v2-resource-review.json`, SHA-256
`688470c219a741cddb3362d9e025c872707e2d3413fa00bb2ff4693eb4810c03`.
A preliminary display-only inspection stopped on a null execution field;
the corrected saved-data aggregation completed without changing evidence.
