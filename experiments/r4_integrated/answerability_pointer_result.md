# Separate answerability: acquisition and executed continuation

Completed 2026-09-06 local time at source
`0c1261be7280a7616562d17521d6670fa34f8f01`. Mechanical completion and the saved-only
audit pass; **acquisition, integration and joint qualification fail**. The
[contract](../../docs/research/answerability-pointer-development.md),
[admission](../../benchmarks/answerability_pointer_acquisition_v1.admission.json),
[full report](../../benchmarks/answerability_pointer_acquisition_v1.json) and
[completion](../../benchmarks/answerability_pointer_acquisition_v1.completion.json)
preserve the fixed recipe and evidence. No architecture is adopted or gate passed.

## Reading result

The four fresh policies reuse the same 8,192 supervised SQuAD2 TRAIN questions,
eight epochs and frozen MiniLM inputs. The 256 development questions are now
project-exposed; this is not fresh adjudication. Old-objective comparison uses
authenticated saved results, with identical inherited initial tensors and RNG,
without another old fit or forward pass.

| Policy | Initial development supported / STOP | Final development supported / STOP | Final TRAIN supported / STOP |
| --- | --- | --- | --- |
| Rate cells | 0/192 · 31/64 | 25/192 · 8/64 | 28/96 · 7/32 |
| Frozen producer, trained heads | 0/192 · 31/64 | 0/192 · 0/64 | 2/96 · 0/32 |
| Trained recurrence bypass | 0/192 · 31/64 | 15/192 · 9/64 | 28/96 · 10/32 |
| BiLSTM | 0/192 · 29/64 | 23/192 · 3/64 | 56/96 · 10/32 |

Against the [old objective](token_pointer_result.md), cells gain seven supported
answers but lose eighteen correct STOPs: thirteen newly correct/six lost supported
IDs, and one newly correct/nineteen lost STOP IDs. BiLSTM's supported total stays
23/192 while STOP falls from 28/64 to 3/64. Neither reaches the 48/64 STOP floor;
cells also miss the 48/96 TRAIN floor and the required gains against bypass.
Separating the loss/readout did not produce a useful balanced improvement.

Cells still emit 159 wrong supported spans, 93 disjoint from every accepted
answer, alongside eight supported STOPs. The previous figures were 129, 74 and
45 respectively. This saved-position diagnostic describes error geometry; it
does not establish the semantic cause or recoverability of STOP outputs.

## Return to executed actions

The same retained-source session supplies 23 offers per updating arm and an exact
frozen continuation mate. Episode/late questions are excluded from acquisition
training at question, paragraph and article levels, but their inputs appeared
in earlier project probes/caches. The two ambiguity exclusions remain unchanged.

| Policy | Initial old supported / STOP | Final old supported / STOP | Episode + late supported / STOP |
| --- | --- | --- | --- |
| Cells | 2/12 · 0/4 | 1/12 · 4/4 | 0/8 · 1/2 |
| Frozen cells | 2/12 · 0/4 | 2/12 · 1/4 | 1/8 · 1/2 |
| Frozen producer / mate | 0/12 · 0/4 | 0/12 · 0/4 | 0/8 · 0/2 |
| Bypass | 2/12 · 0/4 | 1/12 · 4/4 | 1/8 · 2/2 |
| Frozen bypass | 2/12 · 0/4 | 2/12 · 0/4 | 1/8 · 1/2 |
| BiLSTM | 9/12 · 0/4 | 0/12 · 4/4 | 2/8 · 2/2 |
| Frozen BiLSTM | 9/12 · 0/4 | 9/12 · 0/4 | 2/8 · 1/2 |
| External sensory cosine | 0/12 · 0/4 | 0/12 · 0/4 | 0/8 · 0/2 |

Cells lose **both** initially correct supported old actions to pooled STOP; the
final 1/12 is a different newly correct item. Their original correct-paragraph
answers survive, with oracle support improving 4/12 to 7/12. BiLSTM loses all
nine initial supported old actions; every final old prediction is STOP. Six of
those nine answers remain correct in oracle probes, while oracle support falls
10/12 to 6/12. Frozen mates preserve all initially correct supported old actions
under the same enlarged source pool. Saved outputs establish these changes, but
absent logits cannot identify every STOP as a negative gate rather than a tie.

No updating policy gains fresh supported successes over its frozen mate, and
none solves either late supported task. Pooling and feedback stability remain
substantial limitations alongside weak acquisition. All 468 primary actions,
468 oracle probes and 92 offers complete. Twenty-three external teacher repairs
and two motor demonstrations bring actual child executions to 493; decoys remain
unchanged. Teacher repairs are not learner successes or autonomous recovery.

## Cost, verification and next work

The run takes **3,184.905 seconds**, including 137.987 seconds for integration,
with **2,036.148 MiB** peak parent working set. Existing input/cache dependencies
remain retained and accounted in the report; new encoder calls/cache bytes are
zero. Twelve saved states occupy 23,532,851 bytes. New artifacts before the report
total 34,475,929 bytes; report/completion add 5,958,415 and 446 bytes. Cache reuse
prevents interpreting runtime differences as a simulation speedup.

The saved-only audit verifies original annotations/UTF-8 positions, all 8,192
updates and 262,144 presentations, 3,584 local predictions, paired old/new IDs,
checkpoint/optimizer metadata, frozen lineages, qualification flags, and the
action/oracle/teacher records with 1,410 workspace files. It does not replay model
forwards, gradients, optimizer trajectories, encoding or child executions.

| Artifact | SHA256 |
| --- | --- |
| Full report | `07ed7192bcded0739d18697b250ab26f07bf61e637d9a4a9f495b3f2a2379c62` |
| Completion | `e8588976a2bac68c1e6f5a921cf1b76ae69de5868131dd307e6b8c9862ca94c2` |
| Checkpoints | `1d6e2c2d8c00f99bd57ca89bbe94b3845f8cfef58695aa89f547bb98168d588b` |
| Saved audit, `.pytest-tmp/answerability-pointer-saved-audit-v1.json` | `a8b4c83bcc80caa086a9f56c0b8fbc7300772e11bcb4366f2218f4ca69743728` |

This closes the comparison without another threshold, epoch or objective sweep.
The user's [parallel capability development](../../docs/research/persistent-memory-byte-development.md)
addresses independent missing memory, byte-output and integration paths. Its
component tests are not a reading result. Preserve the abstention and retention
failures as requirements when testing learning in the persistent action loop.
