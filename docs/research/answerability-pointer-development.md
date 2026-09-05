# Separate answerability: one public-development comparison

Closed 2026-09-06 local time: [result and saved-evidence audit](../../experiments/r4_integrated/answerability_pointer_result.md).
Mechanical completion passes; acquisition and integration qualification fail.
The recipe below remains the contract run at source `0c1261be`; no rerun follows.

Selected 2026-09-05 after the [bounded primary-source review](token-pointer-followup-review.md).
Preparation started 21:20 UTC, with a two-hour budget ending 23:20 UTC, and
completed within that budget. The [full-run admission](../../benchmarks/answerability_pointer_acquisition_v1.admission.json)
freezes the reviewed implementation, readiness and measured cost receipts.
The [previous comparison](../../experiments/r4_integrated/token_pointer_result.md)
and its original contracts remain closed and unchanged at source `9d73bba`.

## Failure, hypothesis and scope

Cells learned 24/96 supported TRAIN probes versus BiLSTM's 53/96; both failed
the STOP floor. Cells answered only 18/192 untaught supported questions. All
eight supported fresh cell actions selected the right source, but seven chose
wrong spans. This does not isolate recurrence, calibration or endpoint coupling.

Test whether separating question-conditioned answerability credit from
conditional span acquisition improves both useful extraction and abstention.
Keep the cell dynamics, generic MiniLM inputs, endpoint inventory and supervised
experience fixed. This is one objective/readout package, not a paper reproduction
or an isolation of the new head from its loss. Better STOP scores alone fail.

This comparison reuses the SAME 8,192 original TRAIN questions and now-exposed
256 development probes. It is repeatable public development, not a newly sealed
cohort, fresh adjudication, answer-free acquisition or source-absent memory.
The original split remains article-disjoint between training and probes; prior
probe inspection prevents describing the probe as untouched. No new data is
selected, no threshold is fitted, and no best epoch or seed is chosen.

## Computation and controls

The reader's existing question-conditioned modeled states `M` feed a learned
attention pool: `a=softmax(M*w_a)` over valid tokens; `z=w_z*sum(a*M)+b`.
Local examples pool one paragraph. Integration pools ALL valid modeled tokens
across supplied documents, while recurrent processing remains paragraph-local.
Initialize `w_a=0`, Xavier `w_z`, `b=0` on seed+2; preserve all old producer and
endpoint initialization tensors and the matched private dropout stream exactly.
The inherited NULL scalar stays frozen and unused in this objective.

Train mean per-example `BCE(z,supported)+supported*span_NLL`. The span term
marginalizes unique original accepted pairs with separate endpoint softmaxes
containing no NULL. STOP labels supply no span target. On an all-STOP offer,
endpoint parameters and warmed Adam states remain unchanged; the shared
producer still learns from answerability. This does not remove all interference.
Decode STOP at `z<=0`; otherwise return the unique maximum legal within-source
endpoint sum. Empty inventories and exact ties STOP. No answer-length filter.

Train four fresh policies: cells, identical-initial frozen producer with learned
heads, trained nonrecurrent bypass, and BiLSTM. All use the same eight epochs,
256 batches/epoch, 24 supported + 8 STOP examples/batch, Adam .001, clipping 1,
dropout .2 and seed 20260905. Keep initial, post-acquisition and final complete
states, including optimizer/RNG. Frozen continuation mates clone the exact
post-acquisition states. The fixed sensory-only action reference is unchanged.

Use saved old-objective results for paired diagnosis, with no new old-objective
fit or inference. Before fitting, authenticate their complete archive and compare
every inherited initial tensor plus the dropout RNG against each new policy.
Report gains and losses on the same question IDs separately for supported/STOP.

## Readiness, budget and return to action

Before the broad comparison, require independent numerical-gradient checks,
masked local/pooled loss and decoding, warmed all-STOP optimizer behavior,
frozen producer, old checkpoint identity, read-only inference and exact restore.
Use the existing focused test file. Then run one disposable sixteen-question
fit: cells and BiLSTM each get 256 batch-16 updates, final 12/12 supported and
4/4 STOP required, with 900 seconds/4 GiB limits. These states never seed broad
training. A failed fit closes this preparation without automatic extra steps.

The broad comparison retains the prior margins: cell supported gain >=20/192
and balanced-accuracy gain >=5 percentage points against its initial state,
trained frozen producer and trained bypass; cell and BiLSTM STOP >=48/64.
BiLSTM must improve >=20 supported and >=5 balanced points over its initial
state. BOTH cells and BiLSTM must now fit >=48/96 supported TRAIN probes at
epoch eight. All criteria are development readiness, not architecture adoption.
Probe the fixed 128 TRAIN questions at epochs 1/4/8 and the 256 development
questions initially/finally. No additional score-dependent diagnostic forwards.

Reuse the authenticated `.runtime/token-pointer-acquisition-v1/word-cache`.
Count its retained disk/RAM costs and inherited input/model dependencies; new
encoding calls and new cache bytes are zero. Measure disposable synthetic update,
prediction and pool costs before admission. The observed prior 3,637-second run
provides a reality check, not a timing guarantee. Admit one seed/launch only if
the conservative projection fits four hours and 4 GiB peak parent memory, with
96 MiB new artifacts and at least 1 GiB free disk remaining. Preserve all archives.

Return every final policy and frozen mate to the SAME 28-item retained-source
Python session, whether the local readiness criterion passes or fails. Preserve
the two ambiguity exclusions, 23 offers per updating arm, first-attempt/teacher
distinction, old retention, decoys and oracle diagnostics. Broad STOP labels
remain paragraph-local; only the reviewed pool authorizes pool-wide teaching.
The existing session criterion is unchanged. Report actual child executions.

Afterward, decide from extraction, abstention and later action benefit together.
If only BiLSTM becomes useful, record evidence against this cell recipe. If
neither improves, compare alternatives without extending epochs automatically.
Return to the [persistent source-absent seam](balanced-evidence-to-action-session-v1.md#source-absent-memory-and-action-interface);
stored answer bytes are episodic payload, not demonstrated internalization.
The authentic exam scorer remains ready; a structured learner binding is still
missing and official response formats must remain intact.

## Completed preparation

The [disposable fit](../../benchmarks/answerability_pointer_small_fit_v1.json)
from `0639e498` passes: both cells and BiLSTM finish with 12/12 supported and
4/4 STOP, in 151.047 seconds at 381.152 MiB peak parent memory. Its saved-only
audit verifies annotations, 512 updates/8,192 presentations, 64 predictions,
checkpoint/optimizer metadata and receipt hashes; it does not replay logits or
gradients. All 35 focused numerical/state tests pass, including exact synthetic
compatibility with the old objective. The synthetic cost measurement projects
9,978.134 seconds and 3,127,239,168 bytes, below the full-run caps. These checks
admit the development comparison; no broader reading or integration pass follows.
