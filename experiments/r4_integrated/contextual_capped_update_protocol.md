# Capped corrective updates in the integrated learner

Status: preregistered public TRAIN development, 2026-09-05. One comparison;
no protected evaluation. Results belong in a separate immutable artifact.

## Failure, hypothesis and decision

The [affine readout comparison](contextual_affine_readout_result.md) recovered
BINDING's saved raw selections through neuronal execution: 4/8 fresh and 2/16
retained edits. Six certified pair margins became negative after later feedback;
pair 26 fell from +17.218 to -8.665 in the final rehearsal. Conversely, nine
failed old queries satisfied all their recorded contrasts. Both interference
and incomplete competitor coverage remain plausible contributors.

Test whether capping each corrective step at its remaining unit-margin deficit
improves retention without sacrificing fresh executed performance. This reduces
unnecessary overshoot at a given state; it does not guarantee preservation of
other margins or global ranking. The earlier conditioning diagnostic tested an
uncapped deficit rule on a different finite ledger: zero wrong pairs but 23
subunit margins, versus none for its fixed-rate gain-16 comparator. Smaller
steps may lose useful protective margin or learn too slowly. All 116 applied
rates in that study were already below one (0.001994 to 0.712581), so adding this
cap would not change its trajectory. The reason for the present test is the new
BINDING geometry and integrated on-policy workload, not a novel fix to that
earlier negative result.

The primary comparison is updating affine-neuronal first-attempt execution:
retained success must increase over the concurrent fixed-rate arm without a
decrease in supported fresh success to support adopting this change. Report
tradeoffs, ties and unsupported errors explicitly. Broader development targets
remain 6/8 fresh, 12/16 retained, and both properties correct for three fresh
entities. Meeting these tiny generated-case targets alone does not pass the
integrated roadmap gate or establish independent generalization.

## Exactly one changed learning policy

Both arms use the same BINDING representation, interaction gain 16, initial zero
evidence synapses, fixed learned occurrence motor and base learning rate 1.
For accepted-minus-rejected eligibility `d`, let `n=d·d` and
`g=(w_GO-w_NOGO)·d`. The existing native correction changes the gap by
`2*eta*n/sqrt(1+n)`. Compare:

- FIXED: the existing comparison rule and its current unit-margin update gate.
- CAPPED: when `g<1` and `n>0`, use
  `eta_eff=min(1,(1-g)*sqrt(1+n)/(2*n))`; otherwise skip the update.

Native code still performs every synapse mutation through its existing scalar
rate argument. Discard the scratch bias as before. Do not alter absolute
single-decision feedback, feature construction, candidate inventory, thresholds,
replay selection or the native ABI. The new opt-in policy must bind checkpoint
and feedback identity; old default numerical behavior and identity stay intact.
Record the policy, squared norm, before/after gap, effective learning rate, cap
activation and actual native updates. A frozen exposure can report its proposed
rate separately, but must report zero applied rate and no weight mutation.

Use the validated affine currents and interpolated GLIF race for primary
training and evaluation in both arms. Its fixed tonic STOP and true-tie behavior
are unchanged. After primary STOP, execute at most one unique raw candidate
proposal over the entire inventory, including below STOP. Use external cosine
only if no unique raw candidate exists. A correct fallback is a learner action,
not a negative example. Teacher success remains separately labeled.

## Exposure, controls and freshness

The new `contextual_capped_update_cases.json` links the openly reused binding
teaching fixture: sixteen requests, four documents and the same counterbalanced
48-encounter schedule. Fix six new transfer documents before any model output:
eight supported requests covering simple facts, nearby entities, repeated values
and paraphrased/reordered relations, plus two unsupported probes. Preserve the
existing stratum structure. New instances are public development, not reserved
adjudication; prior teaching and exam exposure are disclosed.

Keep the [exploration feedback contract](contextual_exploration_protocol.md):
at most 32 actual edited FAIL/teacher-PASS pairs, four distinct older pair replays
per correction, and 128 replay offers. Complete the last admitted correction's
rehearsal, then stop updates while recording all 48 encounters. Do not invent
negative feedback from a score diagnostic or apply transfer/exam feedback.

Each updating learner has a same-seed, same-policy frozen mate receiving its
identical documents and certified feedback exposure. Frozen weights never change.
The two updaters have equal budgets but may generate different on-policy
negatives, correction counts and trajectories; report these differences. This
tests the overall learning policy under the fixed training process, not the
isolated effect of a step on a shared realized ledger.

Execute updating/frozen affine selection, updating thresholded raw-score and
external-cosine references on all retention and transfer requests. Historical
discrete/interpolated or binding ablation diagnostics may be retained by the
shared runner, but cannot replace the primary comparison or select a winner.
All policies share the learned motor: these controls isolate evidence selection
and do not constitute a whole-system core-disabled or external-only control.
Disclose the frozen external encoder and retained source/vector contributions.

Use the original public KS2 questions, passage and official rubric unchanged.
Ingest the passage once per learner for all requested policies. Keep answers
requiring human marking pending; do not tune or repair scores using this run.

## Verification and measurements

Before model outputs, commit the reviewed core, tests, runner, fixture and this
protocol, and record source hashes, encoder files and the loaded native library.
Focused QA must establish real native capped arithmetic, default parity, frozen
and no-op behavior, valid feedback custody and opt-in policy identity. Structural
runner preflight must verify the roster, budget and policy routing. Preserve
interrupted runs and any subsequent harness corrections separately.

Report first attempts, exploration, teacher demonstrations and later evaluation
separately, including actual Python execution, source absence and decoy integrity.
Freeze weights during transfer, retention and exam answering. Record checkpoint
lineage, native updates, admitted corrections and replay offers by arm.

At the final checkpoint, record all certified pair margins and per-query global
ranking: best correct occurrence rank, correct-minus-wrong gap and gap above
STOP, along with the actual selected occurrence. Count failed retained queries
whose recorded pair margins are all satisfied (unit margin, tolerance `1e-10`)
separately from those with an erased pair; also disclose queries with no recorded
pair. Report negative, nonnegative subunit and satisfied margins separately.
Count correct-to-wrong and wrong-to-correct transitions in encountered-query
thresholded raw selections around new updates and rehearsal, with the compared
source sets and denominators. Report correct-to-STOP and STOP-to-correct subsets
separately: lost threshold clearance is not necessarily a ranking reversal. These evaluator
diagnostics are read-only and never determine candidate or replay selection;
they are not additional executed successes or proof of a single failure cause.

Report total/arm training and query runtime, parent memory limits, retained raw
and derived bytes, encoder/ledger costs and actual work. Count overhead from
diagnostics and child execution; do not claim isolated kernel speedups.

## Budget and next decision

One new step policy, one two-arm integrated comparison, and two hours for
implementation and diagnosis starting 02:03 UTC. No extra epochs, rate search,
threshold change, candidate filtering, selected ablations or training on the new
transfer cohort. If execution fails to improve, preserve the negative evidence
and compare stability against competitor coverage before choosing another
mechanism. Return the decision to the shared integration plan. Protected G0,
R5 and R6 evidence and all prior frozen artifacts remain unchanged.
