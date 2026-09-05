# Contextual exploration: executed own-error feedback

Status: preregistered public TRAIN development, 2026-09-05. One fixed run;
no protected evaluation. Results are recorded separately after execution.

## Failure and hypothesis

The binding comparison at source `46b6870`, recorded in `fe720c1`, gave BINDING
one effective native update. Its other 149 new/replay offers already satisfied
their margins. All 48 fallback attempts used external cosine because thresholded
raw inference abstained. The learner's unique unthresholded leaders were wrong
on every supported final query, but none had been an executed negative for its
query. Lowering evaluation STOP would therefore expose wrong answers.

Test whether executing those actual leaders during training supplies useful
feedback. Change only exploration after a discrete neuronal STOP: try the unique
maximum over all raw candidate drives, even below STOP. Use external cosine only
when there is no unique raw candidate. Allow at most one fallback. A correct
exploration earns PASS without a fabricated negative. A wrong executed leader
followed by a teacher PASS supplies a violated ranking margin.

## Fixed architecture and exposures

Keep both GAIN16 and BINDING feature policies, candidate inventory, native
learning rate/rule, STOP drive, discrete teaching, learned occurrence motor, and
all evaluation policies from the binding protocol unchanged. The new core API is
explicitly labeled `raw_candidate_exploration`; it is not an inference-policy
replacement. Frozen mates receive identical documents and certified feedback.
No answer labels enter the proposal, feature construction, or replay selection.

The fixture `contextual_exploration_cases.json` openly references the binding
fixture's 16 teaching questions and fixed 48-attempt schedule. Reuse is deliberate
development exposure. Its six new transfer documents provide eight supported
questions and two unsupported probes, fixed before model outputs. No transfer
answer, rank diagnostic, or original-exam result affects training. Transfer is
fresh for this run, not independent adjudication or an uncontaminated benchmark.

Retain at most 32 actual edited FAIL/teacher-PASS correction pairs, at most four
distinct older receipt replays per correction, and 128 replay offers. Complete
the 32nd pair's allowed rehearsal, then stop updates while recording all 48
primary attempts. No added epochs, tuning, or training after transfer begins.
Record differing on-policy negatives between GAIN16 and BINDING.

The public binding runner may receive a small explicit reusable configuration
extension, with its default choices, labels and budgets preserved. Historical
exact reproduction uses its recorded source commit. New entrypoint, fixtures,
protocol and executable sources are frozen in a source commit before the run;
record their hashes, encoder files and loaded native-library identity.

## Measurements and controls

Record thresholded raw reference and raw proposal separately at fallback;
identify proposals that bypass STOP. For each actual wrong raw leader accepted
for correction, check teacher-minus-leader drive is at most zero before learning
(only numerical tolerance allowed), and record the gap after correction and the
actual native update. Receipt validity and identical frozen exposure remain
mandatory. Changed weights alone do not constitute success.

At the same final checkpoint, execute all 16 retention and 10 transfer questions
through seven existing policies: updating/frozen discrete neurons,
updating/frozen interpolated neurons, updating mapped score, updating thresholded
raw score, and external cosine. BINDING also executes its existing disabled and
position-collapsed feature ablations. Keep true-tie STOP behavior. All references
share the learned motor: these are evidence-selection controls, not a complete
whole-system core-disabled comparison. Frozen updating contrasts measure learning
under this exposure stream; comparison with the old exploration policy is
historical, not a concurrent causal comparison of exploration policies.

Add read-only evaluator diagnostics at that checkpoint: unique unthresholded
top candidate, leader count and drive; best correct occurrence rank and drive
over all candidates emitting the expected value; best-correct minus best-wrong
drive; and best-correct minus fixed STOP drive. Unsupported questions have no
correct candidate. Assert that diagnostics leave the checkpoint unchanged.
Keep these answer-aware diagnostics outside training and replay. Executed
outcomes remain authoritative: wrong ranking, correct-but-below-STOP and readout
collision are different failure modes.

Report discrete and interpolated neuronal outcomes separately, with neither
selected retrospectively as a new policy. Targets for either declared policy
remain at least 6/8 supported fresh edits, 12/16 retained edits, and both
properties for three fresh entities. Report unsupported abstention/error
separately. Raw-score success alone cannot pass a neuronal target. Preserve
decoy integrity, source-absent checks, and the original untuned public KS2
question/rubric diagnostic; human-marking-pending answers remain pending.

Report ingestion/query/training runtime, memory, retained raw and derived bytes,
native updates, offers, action counts and checkpoint lineage. External encoder
and retained source/vector contributions remain disclosed. Teacher corrections
are not autonomous recovery. These narrow generated development cases cannot
pass the integrated roadmap gate, G0, R5 or R6.

## Budget and decision

One fixed two-arm comparison, with two hours for implementation and diagnosis.
Run focused behavior QA before the frozen run; do not rerun successful checks
without an unresolved concern. Preserve results and failures in new immutable
artifacts. If informative own-error updates still fail transfer, compare the
measured ranking/STOP failures before choosing a new bounded mechanism. If
correct leaders remain below STOP, consider an explicit action-versus-STOP
learning contract in a later experiment; do not silently reduce the threshold.
