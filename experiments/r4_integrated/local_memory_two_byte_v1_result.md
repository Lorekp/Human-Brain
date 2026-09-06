# Two-byte document-memory return fails useful acquisition

2026-09-06. **The full fit and saved audit complete, but the fixed useful
criterion fails.** Tagged, aggregate and frozen-writer controls all score
2/8 fit, 1/4 unseen, 1/4 delayed and 0/4 interference. There is no learned-writer
advantage over the frozen writer and no tag-specific advantage. Preserve the
qualified output component, close this integrated fit and review the saved
failure before selecting another bounded intervention. No automatic extra fit
or gate advancement follows.

Frozen source: `a94b466f1b2c34e83b53d19e43f9430e52191648`.
The [recipe](local_memory_two_byte_v1.md) fixes the new two-byte roster, five
learning controls, twelve retained memories per arm and original executed-action
criterion. The [LMS output qualification](local_context_lms_v1_result.md)
establishes learning over authored factor inputs; it does not transfer directly
to continuous raw-document memory contexts. This return starts output weights
fresh and removes the authored answer factors.

## Full acquisition outcome

The [final decision](../../benchmarks/local_memory_two_byte_v1.decision.json)
binds the successful full-run custody and failed capability criterion, separately
from the immutable prospective cost decision.

| Arm | Fit / 8 | Unseen / 4 | Delayed / 4 | Interference / 4 |
| --- | ---: | ---: | ---: | ---: |
| Tagged | 2 | 1 | 1 | 0 |
| Aggregate | 2 | 1 | 1 | 0 |
| Writer/maps frozen | 2 | 1 | 1 | 0 |
| Fully frozen | 0 | 0 | 0 | 0 |
| Writes disabled | 1 | 0 | 0 | 0 |

The required tagged 8/4/4/4 endpoint is absent. Its unseen score ties the frozen
writer, so superiority to fully frozen and writes-disabled is insufficient.
The entire tagged final output list matches writer/maps-frozen across all twenty
checks. Outputs are nonconstant: this is not the earlier constant-phrase result.

All three memory-enabled updating arms produce `ca,ca` on the identical-question
`ac/ca` counterfactual swap, and `cd,cd` on the `bd/db` swap. The delayed answers
are identical to their corresponding immediate answers. After unrelated writes,
the same `k00` question changes from `ca` to `da` in both retained memories;
the other pair stays `cd,cd`. That intervention establishes retained-state
influence, but its nuisance sensitivity does not establish correct key/value
binding. Updating arms generate EOS on all final cases; fully frozen emits
`00 00` and stops at the cap. Final raw bytes for every arm/case are preserved
in the decision record.

The saved matrix diagnostic finds distinct tagged memories despite equal swap
outputs: relative Frobenius differences are 0.05007377 and 0.07063044, with
cosines 0.99874565 and 0.99750278. After unrelated writes the corresponding
relative differences shrink to 0.01836849 and 0.02593323, with cosines
0.99983191 and 0.99966373. Writes-disabled matrices remain zero. Frozen-writer
maps stay fixed while its query/output weights learn. These observations do
not locate a saturation mechanism or prove identical memories; the needed
forward context/activation gains were not saved.

The saved-only diagnostic is `.pytest-tmp/local-memory-two-byte-v1-diagnostics.json`,
SHA-256 `7e5d5cd23e9013a78616bc4ec34c7f1b97048921ccec6e9a6892cb8729edaf13`.
It uses authenticated saved outputs and matrices without model inference or
new teaching. Its separate 2.157-second diagnostic cost is not included in the
monitored full-run audit total below.

## Learning curves and full exposure

Loss is half the sum of all 257 squared output-potential residuals, not
cross-entropy or teacher mistake rate. Each loss follows that lesson's genuine
first attempt; these changing-weight curves do not add endpoint probes.

| Arm | First exact / 1,024 | First 128 exact | Last 128 exact | First 128 mean loss | Last 128 mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tagged | 121 | 13 | 24 | 0.2753090434 | 0.2511994546 |
| Aggregate | 123 | 13 | 25 | 0.2753446987 | 0.2481911236 |
| Writer/maps frozen | 118 | 13 | 23 | 0.2752817171 | 0.2509717680 |
| Fully frozen | 0 | 0 | 0 | 0.5 | 0.5 |
| Writes disabled | 100 | 12 | 13 | 0.2767246755 | 0.2576939370 |

Every updating arm has one initial cap stop and 1,023 EOS stops; frozen has
1,024 cap stops. All eight consecutive 128-lesson exact/loss blocks are saved
in the decision. The modest late gains do not establish convergence or justify
extending the closed budget. They also do not distinguish a weak memory
representation from ineffective upstream credit.

The full run counts 5,120 first attempts and 15,360 byte/EOS teacher offers,
including frozen. All 3,947,520 scalar residual opportunities are counted;
3,158,016 are enabled. All 12,288 enabled decoder steps change weights.
There are 39,780 nonzero residual offers, of which 36,708 are enabled. Generation
uses 15,360 first-attempt decisions. Byte traversals total 256,000 ingestion,
266,240 acquisition query and 19,240 final query steps, followed by 100 action
cases and two supplied motor demonstrations. These corrections train from byte
labels, not from execution feedback. The three discarded cost attempts add
sixty lessons and 180 teacher offers beyond these full-fit counts.

## Full-run resources and audit

The full run's watchdog records **255.179305 seconds** and a process-tree peak
of **957.93359375 MiB**. The independently monitored saved audit takes
**3.2826283 seconds** at **416.81640625 MiB**. Run plus monitored audit is
**258.4619333 seconds**; their sequential memory peaks are not summed. Both
fit within their recorded watchdog bounds. The audit's internal 2.6614786-second
timer excludes startup and is not substituted for its outer measurement.

Each arm stores 809,804 model tensor bytes, including 33,024 fixed input-code
bytes. The five models and twelve retained memories per arm total 186,556,588
source-free tensor bytes. Source-derived eligibility remains counted even
without raw documents. The output submodule contains 644 neurons and 74,787
learned synapses, with 602,952 serialized bytes. Its 4,396-byte step includes
its 12-byte next clock; its 2,056-byte residual and 768-byte context-feedback
vector are additional. The 74,016 context-return connections share the existing
forward weights and allocate no separate weight array. Python, transport copies,
transient arithmetic and native executor overhead are covered by monitoring.

The passing full saved audit authenticates 1,135 source files, five artifact
hashes and 102 execution receipts. It verifies matching source-free states and
read-only outputs, all five controls, complete lesson hash chains and actual
counters, discarded cost weights, zero initial LMS output weights, and exact
initial input-code/memory/query/broadcast tensors against authenticated v5.
It does not recompute the whole local learning trajectory or re-execute actions.
Audit passage verifies custody and accounting, not useful acquisition.

| Full evidence | SHA-256 |
| --- | --- |
| `run/report.json` | `57f6ff8330e90c536be40809579ff6c31917ea23076e711a5d61a6d1727c40c5` |
| `run/completion.json` | `a9d34214a8e5ecd227d4496f09e328f80f650411910c64b964e0657b05f28181` |
| Run watchdog | `4a0edf8c5c93e4256460a923763cb30499ec9fe7bd56974b634f98533a071bc9` |
| Independent run audit | `1ef342ddac2cabae3195d23007354ec8dd70b29171a1cc7c56f2bb49216c80a0` |
| Run audit watchdog | `6737f0a253d3ed3cff5548fa2ad00e2415064b3483d0654a16224e8a47fca808` |

These reside under `.pytest-tmp/local-memory-two-byte-v1-run` or its adjacent
`-run-watchdog.launch.json`, `-run-independent-audit.json`, and
`-run-audit-watchdog.launch.json`. The decision binds their exact paths/hashes,
all saved artifact hashes and the complete source-manifest fingerprint.

## Engineering readiness

The final affected QA selection passes 94 tests: 93 ordinary checks and one
separate fresh-process transport check. Receipts are
`.pytest-tmp/two-byte-qa-final.xml` and
`.pytest-tmp/two-byte-transport-qa.xml`; doctor is
`.pytest-tmp/dev/doctor-j6q_14qs/receipt.json`. An earlier fixture failure is
preserved in `.pytest-tmp/two-byte-initial-qa.xml`: the transport test's old
primitive-type restriction rejected the new scalar learning-rate metadata.
Correcting that fixture introduced no model change. The recipe records small
disposable fixture exposure and the ten untrained worker predictions.

## Initial invocation failure, preserved

Root supplied a relative `--output` path to the initial cost launch. The
existing action helper changes the worker directory to `source-free` and
forwards checkpoint/request/output paths. Relative paths consequently acquired
a second `.pytest-tmp/.../source-free` prefix. The worker raised
`FileNotFoundError` while opening its checkpoint, before model loading.
The checkpoint existed in the intended directory and matched its command hash.

This attempt completed **20 disposable lessons and 60 byte/EOS teacher offers,
with zero actions**. All its weights are discarded. Its external watchdog
records 9.1791626 seconds and a 608.76171875 MiB process-tree peak. The failed
attempt remains additional engineering exposure and cost; it is not silently
replaced by later attempts and supplies no acquisition result.

The repair changes only invocation paths: use absolute output/admission paths
and a fresh retry directory. Frozen source, architecture, rule, seed, roster,
budget and projection formula remain identical. No model change or new
threshold was justified by this filesystem error. The preserved correction is
`.pytest-tmp/local-memory-two-byte-v1-cost-invocation-repair.json`, SHA-256
`fa0bf50e996a4b9e762302127775b409a0a8f631bece56c84ec6c8316b087850`.

## Second attempt: complete workload, inadmissible launch

The retry completes in **24.6570224 seconds**, with an external process-tree peak
of **914.4765625 MiB**. These are measured disposable-run resources, not the
prospective full-acquisition projection. Including the failed invocation, the
two cost attempts take 33.8361850 seconds before independent audit; sequential
memory peaks are not summed.

The retry records twenty first attempts, sixty byte/EOS offers, 15,420 scalar
residual offers (12,336 enabled), 48 enabled and changed decoder steps, 84
nonzero residual offers (72 enabled), and sixty first-generation decisions.
It processes 1,000 ingestion byte steps, 1,040 acquisition query byte steps and
19,240 final query byte steps. All 100 final action cases and two fixed motor
demonstrations complete. These cost-model outputs are not full-fit performance.
Both attempts together offer forty disposable lessons and 120 byte/EOS labels.

The retry nevertheless fails launch admission: root omitted the parent's
required `-X utf8` prefix. The frozen `validate_cost_launch` requires the exact
prefix, so workload completion cannot admit these receipts. Root briefly
relaxed the independent checker incorrectly; its saved `status=passed` output
is invalid admission evidence. The original strict checker has been restored.
Neither the missing flag nor the invalid checker is reinterpreted as a permitted
configuration change. The second attempt's twenty lessons, sixty offers,
100 actions and two motor demonstrations remain counted; all weights are discarded.

The invalid audit is
`.pytest-tmp/local-memory-two-byte-v1-cost-independent-audit.json`, SHA-256
`5015c5e78dc4319636fb942e097d001805c55aef116cc92107c24d3c0e25ad9d`.
The correction is `.pytest-tmp/two-byte-cost-utf8-repair.json`, SHA-256
`e4a5e5ae26419285f35a05374409fd0c908315278b03afe0e3c260976416b0e4`.

## Third attempt: strict cost qualification and full-fit admission

The fresh `local-memory-two-byte-v1-cost-qualified` directory uses absolute
paths and the required `-X utf8 -m experiments.r4_integrated.local_memory_two_byte_v1
cost` prefix. Frozen scientific sources, configuration, thresholds and budget
formula remain unchanged. Its complete cost workload passes in **23.5347427
seconds** at an external process-tree peak of **915.22265625 MiB**.

The restored strict saved-only audit passes in 2.576265 internal seconds and
independently recomputes a **630.7949631693597-second** projection, below the
900-second ceiling. Its terms are 555.6045823693275 seconds projected learning,
45.190380800032166 seconds other work, and the unchanged 30-second allowance.
All timing samples are strictly positive, with the original 2x safety factor.
The admitted peak upper bound is 915.22265625 MiB, below 2,048 MiB. Audit internal
time is separately reported; it is not an external startup-inclusive timer.

All three cost attempts count: sixty disposable lessons and 180 byte/EOS
teacher offers, with 200 completed action cases and four motor demonstrations.
Their external run times sum to 57.3709277 seconds; audits are additional.
No weights from any cost attempt initialize acquisition.

The qualified audit is
`.pytest-tmp/local-memory-two-byte-v1-cost-qualified-audit.json`, SHA-256
`61fbf2de4b69e5cb876e8d7e4aeaecac0f6f33939f4fd5de450f51f713355957`.
The fresh full admission is `.pytest-tmp/local-memory-two-byte-v1.run-admission.json`,
also preserved in the [cost decision](../../benchmarks/local_memory_two_byte_v1.cost_decision.json);
both hash to `0943d8545f568f9e13a7b00eb24117c009946f734ff3c8f7359186f8cce01a7f`.
This strict qualified receipt supersedes neither invalid attempt nor its costs;
it supplies the valid prospective basis for one fresh full run.

Acquisition subsequently completed at `.pytest-tmp/local-memory-two-byte-v1-run`
as recorded above. Cost admission establishes feasibility under its formula,
not acquisition success. The prospective cost decision retains its original
`full_fit_ran: false`; this final result is a separate retrospective record.

## Evidence custody

| Evidence | SHA-256 |
| --- | --- |
| Initial `cost/report.json` | `fa6c057bab1f3d0f12807ca78d38e365fc86de89d9e8ad01ec5b811a8345b520` |
| Initial `cost/completion.json` | `b09f3e4c54b9a1ab2a8330b30c5b6176a8832c1902091ec47d89f08b74f09435` |
| Initial cost watchdog | `708eb630f72767b6736baa929345c3c2f84c0ce51db7414301c96478fde14899` |
| Retry `cost-retry/report.json` | `85f3efc56e51fc7470eafb07211519bd1efacf95418eebc4336a35af77e545da` |
| Retry `cost-retry/completion.json` | `cc590e92e9477fc67133d36b945f6ea631f226149c7e8b5407a9752198fb44cc` |
| Retry cost watchdog | `c7f6554b8fd98e8ea1d45a950729688512040fd203dffa3f8eaf77f8eecb0da4` |

Artifact roots are `.pytest-tmp/local-memory-two-byte-v1-cost` and
`.pytest-tmp/local-memory-two-byte-v1-cost-retry`; watchdogs are the adjacent
`local-memory-two-byte-v1-cost-watchdog.launch.json` and
`local-memory-two-byte-v1-cost-retry-watchdog.launch.json`. Both use cost
admission SHA-256
`9f2bb656437a5e360f3a9d8a7a91162dad89fd690345754296c24d08d947ad79`.
Qualified cost/admission and full acquisition receipts are bound separately
above. The relaxed-checker passing record remains invalid admission evidence.

## Interpretation and closure

The unchanged tagged criterion required: 8/8 fit and 4/4 each for unseen values,
delayed reads and interference, with strict unseen-value advantage over
writer/maps-frozen, fully frozen and writes-disabled controls. Report
aggregate-credit superiority separately. Identical-question counterfactual
swaps test retained memory use; final labels never train the learner.

The delayed condition consists of 32 unrelated read queries over a fixed saved
snapshot. It tests invariance under read distraction, not elapsed biological
time, new learning, task switching or forgetting under slow-weight updates.
Interference appends unrelated document writes to a copy of the novel memory.
The four withheld combinations are already exposed development material.

The host-authored occurrence motor still supplies matching and action-schema
selection. Correct neural bytes reaching real execution do not establish
entirely neuronal action selection. Teacher byte corrections remain separate
from execution feedback, autonomous recovery and general software capability.
The output component qualified only for this return. The present result fails
useful acquisition under its own conditions; it authorizes no extra training,
architecture adoption, integrated gate or protected coding evaluation. Preserve
the measured failure and use one bounded review of the shared memory/action
problem to select subsequent work.
