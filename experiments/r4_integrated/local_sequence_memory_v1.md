# Learned retrieval over neuronal sequence memory

Status: prospective implementation; no cost, acquisition or endpoint run.
The selected [design](../../docs/research/local-sequence-memory-review.md),
committed at d9a38d9, owns the model's local math and publication order.

## One replacement and five controls

Use one 128-slot neuronal tape of raw sensory bytes plus document EOS, with
generic preceding 32-lag keys over the existing 32-dimensional fixed byte codes.
All slots and lags are accessible to one learned query projection. Every start
branch advances through forward adjacency; document EOS absorbs continuation.
The shared 257-way conditional codec has a generic 50-percent identity prior,
not a special punctuation rule. Prefix beliefs follow emitted bytes during free
generation and supplied bytes during the separate teacher traversal. Source
documents end in a period before EOS, so copying a two-byte value requires
learning a period-to-stop correction; no host skips it.

The five ordered arms are `learned`, `address_frozen`, `slow_frozen`,
`writes_disabled`, and `byte_reference`. The first four share initial arrays.
Address/query frozen retains plastic codec and normal sensory association;
all-slow frozen still acquires sensory associations. Writes-disabled advances
allocation/boundary clocks with zero keys and sensory weights. It retains counted
layout state. The strongest a1ab multiscale byte baseline uses the explicit
`local_memory_event_reference.py` archive, fresh seed 733. Its old full trajectory
must reproduce exactly. The failed 1935060 event learner is not rerun; its explicit
archive exists only for historical ordinary fixtures. No main compatibility mode,
dummy old writer tags or obsolete aggregate control remains.

Candidate byte teaching reports `sequence_mixture_negative_log_likelihood`. The address
error is initial prior minus final prefix posterior, exact for held codec/query
features. Codec credit uses each current prefix posterior and is an explicitly
local filtering approximation, not the full delayed sequence gradient. Hold all
slow parameters across the three teacher positions, sum local contributions,
validate proposals and publish the codec/address/query groups. Report three
teacher offers separately from enabled and actually changed group publications.
The byte baseline retains its original half-squared-potential loss and bytewise
LMS updates; do not compare its numeric loss directly with sequence NLL.

## Fixed workload and admission

Preserve seed 733, slow rate 0.01, the exact ordered 1024-lesson raw roster, twelve
retained memories per arm, and 20 source-absent checks: 8 fit, 4 novel, 4 delayed,
4 interference. Novel values are exposed development combinations. First attempts
precede teachers. A complete fit has 5,120 first attempts, 15,360 byte/EOS offers,
100 actual Python actions and two motor demonstrations. No target start-address,
selected span or source string enters the query-only worker.

Delay invokes only a fresh query/address traversal, without codec generation or
teaching. Interference appends unrelated documents within the 128-slot bound.
The sparse stored cursors implement declared logical neuronal populations; they
do not select a teacher-matching substring. Retained literal sensory associations
are episodic storage, not semantic compression.

Fresh cost uses the same four maximum-length selected lessons per arm: 20 lessons,
60 teacher offers, five complete capped predictions,100 actions and two demos.
For each cap, ingest exactly one 127-byte period-only document and its EOS, all
five arms including the byte reference. Candidate tapes occupy all 128 slots.
Count this extra cost-only ingestion separately; it has no teachers and its
weights/state are discarded. Temporarily force a normal non-EOS output through
ordinary output parameters, run the unchanged `max_bytes=2` decoder, and verify
exact restoration of every slow tensor. There is no host generation shortcut.
The cap query is the existing `What is the value of k00?` byte string.

Keep the original twice(max complete lesson plus cap) projection for 1,024 lessons
per arm, twice other measured cost and 30 seconds allowance, within 900 seconds and 2,048 MiB. Cap
ingestion belongs to other measured cost; cap prediction is also explicitly
projected. Root owns native preparation, absolute paths and the exact `-X utf8` launcher, source
freeze, independent cost audit, admission and the single conditional fit.
Cost weights never initialize acquisition. Any cost/custody failure stops launch.

## Complete accounting and evidence

Record raw document/query bytes and EOS, sensory/key association opportunities,
allocation/adjacency work, all evaluated occupied start branches, query-cell and
eligibility steps, address projection/scores/normalizations, codec branch passes
and 257-neuron potential/probability arrays, mixture reductions, prefix updates,
cursor initializations and advances, teachers and group publications. Sparse gathers count the
logical populations/connectivity and actual stored arrays. No unrecorded codec
replay is permitted: current conditional probabilities supply current error.

Separate first generation, teacher traversal, capped cost, final ingestion,
query-only distraction, free final generation and real execution. Keep per-arm
input/work counts, model/state snapshots, before/after hashes, prefix receipts,
source manifests and immutable output/action receipts. Output row timing includes
the preceding distractor work when its work count does; enclosing worker timing
also includes restore, serialization and validation. Process-tree monitoring
covers all imports, snapshots, copies, logging and temporary arrays. Compact
decision receipts retain mixture probabilities, prefix beliefs and cursors in
the external report; their Python objects and final JSON serialization remain
inside the measured budget. The learner does not retain these histories.
The full conditional cache includes sensory activity, logits, log probabilities
and probabilities: 921,088 bytes at 128 branches. Named tensor payloads are
not process-peak estimates; allocation overlap, errors, proposals and receipts
also contribute. `sequence_model_accounting` records those scopes.

Qualification requires learned 8/8 fit and 4/4 in each remaining stratum, both novel
assignment swaps correct, and strict novel superiority over address-frozen,
slow-frozen, writes-disabled and byte-reference. A system gain without the control
advantage is reported separately. No automatic rate/temperature/epoch or new
component benchmark follows failure. The host occurrence motor remains supplied
narrow matching/action scaffolding, not learned general neuronal action selection.

## Ordinary QA before freeze

Run doctor and reviewed affected tests in the existing two test files. Use small
authored streams for the tape, all 257 codec identities, EOS/forward continuation,
same-prefix filtering and independent scalar local error/update arithmetic.
Verify source-absent independent state, pre-update publication order, frozen
controls, no backward/history, finite proposals, baseline/reference trajectories,
cap restoration, actual work counts and a fresh query-only worker. No full TRAIN,
cost workload or endpoint roster is evaluated by this implementation lane.
Record all successful and failed QA exposures before freeze.


### Saved implementation receipts

Doctor `.pytest-tmp/dev/doctor-wu0fcchb/receipt.json` passes before our QA;
the main author also records `doctor-_en59rh8`, and the tape author
`doctor-i4fkithq`. The first new end-to-end fixture passes in 2.23 seconds
(`sequence-e2e-v1.xml`). An intended narrow `-k sequence_` selector then matched
the existing test filename and executed all 160 cases: 156 pass, four fail in
15.30 seconds (`sequence-focused-v1.xml`). Preserve that selection mistake and
the full receipt; this was ordinary authored QA, not an admitted acquisition.

Three failures were historical factories still importing the replaced main
learner. The high-resolution clock fixture now constructs the explicit byte
archive. Two duplicate historical child-worker tests are retired because their
frozen entrypoints internally import the obsolete main contract; changing only
their parent fixture cannot fix that routing. Their original tests remain in
Git, and the new five-arm child fixture covers independent PID, twelve supplied
states per arm, strict query-only request fields, source/checkpoint hashes,
unchanged weights/state and real codec generation. That new worker passed in
the broad receipt without native action construction.

The fourth failure was an authored expectation, not model arithmetic: for the
initial identity codec over `ab.`, first and final teacher posteriors coincide,
while the middle posterior differs. The corrected middle-versus-final check
retains the independent scalar current-prefix update oracle, complete-likelihood
address finite difference and pre-publication query-signal comparison. Only the
two recovered tests and the new retired-event archive's two dtype parity tests
run afterward: four pass in 2.82 seconds (`sequence-recovery-v2.xml`). The archive
matches its 1935060 Git class and original dependency trajectory for three small
lessons per model/dtype. Strongest a1ab baseline parity also passes in the broad
receipt. Dynamic old-source loading is QA-only; runtime references are static.

Two new no-model cost/custody fixtures pass in 1.80 seconds
(`sequence-cost-schema-v1.xml`), preserving positive cap samples, the original
2x projection and 30-second allowance, the 900-second/2,048-MiB limits and exact
UTF-8 launcher/PID ownership checks. There is no second full-file execution.
Together these receipts verify all 162 current decoder-file cases. Tape QA adds
eight distinct cases (`sequence-tape-unit-v1.xml`, 1.74 seconds); only its three
strengthened lag-window cases rerun (`sequence-tape-window-v2.xml`, 1.61 seconds).
The total is 170 distinct affected cases, including 257 codec identities,
held-sequence math, controls, generic longer continuation, source-free transport,
capacity forcing/restoration and real work counters. Ruff and diff checks pass.

The exact newly added integration subset is recorded in
`.pytest-tmp/sequence-integration-qa-exposure.json` (SHA256
`bb90732c75a8acf85043bb9a87ce970cd1ddeda594d2cf3afbc1e978060ccdd6`).
Across initial, failed and successful attempts it has 25 successful ingestions,
704 raw-byte/EOS steps (571 candidate tape slots and 133 baseline writer steps),
51 actual query traversals with 442 controller steps, 44 teacher offers including
disabled offers, and 31 free traversals including one disclosed authored initial
address fixture. Five cap ingestions contribute 635 raw bytes plus five EOS;
the fresh worker contributes ten free calls, 18 decisions and zero teachers.
Two standalone identity-codec checks add 516 branch evaluations. The additional
event-archive parity tests contribute twelve lessons and 36 teacher offers;
the repaired old clock fixture adds one query/cap and no ingestion or teacher.

These are explicitly subset counts derived from fixed fixture control flow.
The broad receipt also executes all pre-existing disposable fixtures, including
their separate autograd comparison tests. Its complete case list and failures
remain in XML; variable first-decision counts were not recorded globally and
are not invented. No action, admitted TRAIN fit or full endpoint was run.
The tape author's separate exposure file has SHA256
`21f82354903489fe578bbe44946e5a2d1e01f17fe0d16fe6e2525d75216d4ef0`:
679 successful slots (660 bytes plus 19 EOS), one discarded partial normalization
failure, 336 actual cursor advances and 168 independent dense-reference advances,
with no learner/query/codec/teacher/action calls.
