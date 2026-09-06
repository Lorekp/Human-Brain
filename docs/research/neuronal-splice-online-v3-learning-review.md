# Coincidence v3: correct payloads, failed operation transfer

Completed saved-evidence review, 2026-09-06. The fixed run is closed. The
[result bank](../../benchmarks/neuronal_splice_online_v3.decision.json) records
**4/32 fresh tasks for the coincidence learner versus 5/32 for the graded
competitive reference**, both with 14/16 retained actions and 20/20 old reading
before and after learning. The strict criterion and reference-superiority
component fail. No additional rounds, altered decoding or endpoint-specific
repair follows this result.

The component result is more specific: **all 28 candidate fresh SPLICE payloads
are correct and terminate with EOS**, including the empty deletion payload.
For all 27 nonempty payloads, the intended literal start is the unique highest
saved score and the actual selected address. Yet 23 of those edit tasks receive
STOP. All four KEEP tasks receive SPLICE. The remaining edit failure, O2-0,
selects the wrong start/end despite correct file, operation and payload. These
27 operation errors and one span failure account for all 28 failed fresh tasks;
there are no candidate fresh path, payload, missing-EOS, rejection or process
errors. Correct pre-operation payloads do not count as successful edits.

## What the saved comparison establishes

| Fresh diagnostic | Coincidence learned | Graded competitive reference |
|---|---:|---:|
| Correct SPLICE payloads /28 | 28 | 24 |
| Intended nonempty start uniquely ranked first and selected /27 | 27 | 23 |
| Wrong operation /32 | 27 | 26 |
| Wrong path /28 | 0 | 0 |
| Wrong start /28; wrong end /28 | 1; 1 | 1; 1 |
| Missing payload EOS /28; rejection /32; process error /32 | 0; 0; 0 | 0; 0; 0 |
| Successful tasks: D, C, V, O, each /8 | 0, 2, 0, 2 | 2, 1, 0, 2 |
| Complete fresh contrast pairs /16 | 0 | 1 |

The graded reader selects the wrong record at unique target rank2 for C1-0,
C1-1, C3-0 and C3-1, emitting `dd` instead of `ca`, `cacaca`, `db` and
`dbdbdb`. The candidate selects and emits these intended literals correctly;
only C1-1 becomes a successful edit. Conversely, the reference completes both
D0 tasks while the candidate issues STOP with the correct full proposed edit.
Both systems retrieve all eight V literals, including the four five-document
interference cases, but issue STOP throughout that factor. This cohort therefore
does not show a remaining V-address defect in either trained reader.

Candidate successes are C0-1, C1-1, O1-0 and O3-0. Reference successes are
D0-0, D0-1, C0-1, O1-0 and O3-0. Both fail retained T01 by selecting a wrong
record at target rank2 (`cc` instead of `ba`), and T08 by choosing the wrong
start/end with the correct `aa` payload. All other retained tasks pass.

The action-frozen control also produces all 28 fresh edit payloads correctly,
but makes no successful fresh edit. It misses 26 files and all 28 canonical
starts/ends. The all-slow-frozen control misses 26 payloads; memory-disabled
misses 27. Their fresh successes are respectively one, two and one explicit
STOP, with zero successful edits. Thus reader learning can yield useful payload
transfer while frozen action selection remains ineffective. These controls do
not identify a unique cause of the trained operation head's errors.

## Acquisition, preservation and locality

| Arm | Correct actual first attempts in rounds 1–8, each /32 | Total /256 |
|---|---|---:|
| Coincidence learned | 2, 6, 14, 21, 23, 26, 28, 28 | 148 |
| Action frozen | 4, 3, 3, 5, 5, 5, 5, 5 | 35 |
| All slow frozen | 4, 4, 4, 4, 4, 4, 4, 4 | 32 |
| Memory disabled | 5, 10, 10, 10, 12, 14, 16, 15 | 92 |
| Graded reference | 2, 6, 12, 15, 22, 26, 28, 28 | 139 |

The nine additional candidate acquisition successes precede correction and are
real actions, not teacher outputs. Both learned systems finish at 28/32 per
round, yet transfer poorly to the fresh operation contexts. Learned, action-frozen,
slow-frozen and reference old-reader panels each remain 20→20, with every
pre-correct identity preserved; disabled memory remains 0→0. Those initial
reading successes are not newly acquired during these rounds.

The audit records 1,280 same-instance learning actions, 2,400 reader offers,
3,200 action labels, 240 teacher-free endpoints and 200 separate panel calls.
Actual free payload decisions are 3,772 during acquisition and 676 at endpoints.
The enabled learned/reference groups publish start/end192, operation256 and
reader codec/address192 times each. Action-frozen publishes only reader groups;
slow-frozen publishes none; disabled address stays unchanged while its codec
group can change through biases. Audit custody and source/ordinary QA support
these statements without reconstructing unsaved intermediate weight tensors.

The original v6 numerical prior is shared; its address meaning changes under
coincidence. Corrective evidence uses the same persistent reader and action
synapses after actual execution, with fixed forward features and direct local
output errors. There is no backpropagation through coincidence/history/winners.
Different payload histories can change operation teaching, so the matched
integrated experiment is not an isolated final-layer or frozen-policy test.

## One next bounded question

Prioritize a **source/math review of operation-decision generalization**: can
the current local operation neuron distinguish requested action intent across
changed raw paths, contexts and span/payload combinations, and does its teacher
phase train the same relevant distinctions used while acting? Preserve the
reader and raw action interface during that review; no further reader repair is
motivated by the fresh payload evidence here.

The current `operation_features` uses query×selected-context and query×path,
span×payload, lengths and EOS, followed by normalization and two locally trained
outputs. It has no standalone query activity. This makes context-dependent
intent representation a concrete question, not proof that a missing feature
caused these outcomes. Shared false STOP across D/V and missed KEEP contrast
with familiar acquisition success, but saved outcomes alone cannot distinguish
representation limits, contextual correlations, normalization or training
coverage. Review generic neural input ownership and separability before selecting
one change; do not supply a host SET/KEEP parser, operation oracle, span mask,
endpoint-specific feature, extra epochs or sweep. No implementation or new runtime
is selected by this review. A successor needs a newly preregistered cohort.

## Evidence and analysis boundary

Frozen source: `4840c24d8e4265eaddeaea032231d52d2af3185d`. The independently
audited fixed run plus audit took 157.9033546 seconds, within its completed
admission; timing detail belongs to the separate resource review.

The root-reviewed stdlib extractor made one pass over the 1,520 action rows
and 480 endpoint proposal/acknowledgement rows, authenticated exact hashes and
EOF counts, and compared saved grades/rounds with the finalized audit. It loaded
no checkpoint/model, imported no project helper, made no forward/teacher/action,
and did not rerun the audit. Target starts are external annotations from the
authored raw-document grammar, never input to neuronal execution. Empty payloads
and STOP tasks have no inferred canonical nonempty target rank. Diagnostic prior
mass is not the committed policy's probability, and ranks imply no counterfactual
generation or causal improvement claim.

| Evidence | SHA256 |
|---|---|
| Final decision bank | `807e256dbb750a41e24a00997bee89dbfb7ad4d6e2ee7ab3dc556686f93770a8` |
| Full report | `423ebbf80219ff4e67e3dca7699782467f74581ee86c31756233d0de3ca70b18` |
| Full independent audit | `0484aeed3f3b1b05a0c0ee357cb846b6c1d8dfa86c83dfc94c2c2915f0b80474` |
| Roster | `9b8f03657c1604b7e8e286e12b7534db750a5ad29ab9c872f3481d58bf2aabd5` |
| Actions JSONL | `f74b7fe5d52294f3a0db2ca9990b6f033704e6913d31daba4cc606c82323c481` |
| Worker endpoint JSONL | `4ce10e3c7ea412e68a921a59cb463ff599a6d03cfe94058d96a5110313751657` |
| `.pytest-tmp/coincidence-saved-learning-summary-v1.json` | `b7eaeb80b649095751b0e6f278d6be885a02e3accd004a933a0ea17515dd1c1d` |

The exposed v3 cohort is closed development evidence. Useful literal payload
selection is a narrower result than reliable raw editing or general coding.
