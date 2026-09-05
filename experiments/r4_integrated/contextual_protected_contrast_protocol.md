# Preserve learned contrasts during the integrated session

Contract fixed for public TRAIN development, 2026-09-05. Freeze reviewed source
and fresh instances before one comparison; no protected evaluation is involved.

## Failure, hypothesis and decision

The [reservation comparison](contextual_reserved_feedback_result.md) improves
fresh executed edits from 3/8 to 4/8 and retention from 2/16 to 4/16. Every
executed training failure receives feedback, yet 12/42 admitted pair margins end
negative, eleven after being positive at their last offer. Five final wrong
leaders are previously admitted negatives, five are unexecuted old occurrences
and two come from fresh sources. Interference is actionable but is not the only
established failure. One correct old answer is lost after fresh ingestion with
unchanged weights.

Test whether preventing later changes along already learned contrast directions
improves retained execution. The [primary-source review](../../docs/research/contextual-interference-control-review.md)
owns the research attribution and limits. This is an OGD-inspired conventional
global training operator, explicitly disclosed in the hybrid system. Native
placement does not establish that neurons discovered its algorithm. Neuronal
readout synapses still learn from authenticated actions and the affine GLIF race
controls selection; frozen encoder, external stores and learned motor retain
their existing disclosures.

Adopt only if PROTECTED executes more retained edits than concurrent RESERVED
without fewer supported fresh successes. Targets remain at least 6/8 fresh and
12/16 retained, including both properties for three fresh entities. Report
unsupported behavior separately. Preserving recorded margins alone is not a
capability pass; stalling new learning is a possible negative result. These
generated cases cannot establish independent generalization or pass a roadmap gate.

Budget one implementation and one two-arm comparison, with at most two hours for
implementation/diagnosis starting **03:21 UTC**. Do not extend the experiment into
rate sweeps, basis eviction, extra teaching passes or a host-solver fallback.
The earlier finite TRAIN host witness and its weights are excluded.

## Same integrated workload and controls

Both arms use BINDING gain 16, fixed comparison rate 1, the existing learned motor,
affine currents and interpolated GLIF, sixteen openly reused teaching requests,
four teaching documents and the fixed 48-encounter schedule. After primary STOP,
permit one unique raw-candidate exploration; external cosine is used only if
there is no unique raw candidate. Correct exploration is not a negative example.

Both use the [RESERVED allocation](contextual_reserved_feedback_protocol.md):
one possible new actual edited-FAIL/teacher-PASS pair per encounter and at most
150 total pair offers. After encounter t and its new admission, if any, reserve
48-t future corrections and rehearse at most
`min(4, distinct eligible old pairs, 150-used-(48-t))` in the existing cyclic order.
Exclude that encounter's new pair from rehearsal. A native no-op still consumes
its offer; report underspend without padding. Keep all 48 teacher executions.

RESERVED keeps `comparison_protection_policy="none"`. PROTECTED uses
`comparison_protection_policy="orthogonal_certified"`, capacity 48. Each updating
learner has a same-policy frozen mate with identical certified exposure. Frozen
mates disable both synaptic mutation and learned auxiliary-state growth. The
updaters may yield different on-policy negatives; this is not an identical-ledger
comparison. Default/core behavior outside the opt-in path must remain unchanged.

The new `contextual_protected_contrast_cases.json` fixes six fresh documents,
eight supported queries in the existing four strata and two unsupported queries.
These instances must be disjoint from prior comparison substitutions and fixed
before model outputs. No fresh or exam label may influence learning, consolidation
or replay. Preserve all eleven existing primary/reference/ablation policies,
the same motor and one exam helper call per arm. These controls isolate evidence
selection and the training operator, not the entire neuronal system.

## Exact protection semantics

Let `d=accepted_eligibility-rejected_eligibility` be the original certified
same-query contrast, `v=w_GO-w_NOGO` and `Q` an orthonormal basis for protected
original contrasts. Start with zero synapses, empty basis and empty protection
records. Do not alter the decision, occurrence or original eligibility hashes.

For each actual new or replay offer, native code projects d against the current
basis using two ordered Gram-Schmidt sweeps, writing projected direction p to
scratch. The competitive gate and normalizer continue to use **original d**.
If the original margin is below one and p is not numerically exhausted, apply
`Δw_GO=p/sqrt(1+d·d)` and `Δw_NOGO=-Δw_GO`. The rate remains one; no normalization
by projected norm or compensating step increase is allowed. Keep the persistent
bias zero and discard the existing scratch pair-bias update.

An originally satisfied margin has no synaptic update. A numerically exhausted
projection also has no synaptic or scratch-bias update. Always record whether
the original gate or projection caused the no-op. With an empty basis and a
non-exhausted contrast, weights and scratch bias must match the original native
function exactly. Frozen mates may receive exposure/custody records but do not
construct a learned basis or change any learned synapse.

Only the current offer's original contrast may be newly protected, and only
after its resulting margin is at least one. Do not scan labels or other ledger
pairs for extra consolidation. Add its independent normalized residual to Q in
first-certification order; a dependent contrast needs an original-delta and
provenance record but no extra column. Repeated certification of the same
query/source-occurrence/delta identity does not append another learned record.
Capacity overflow is an error, not an eviction or silent loss of protection.
Weak or negative contrasts remain unprotected. No absolute-feedback update may
bypass this policy: reject it in the opt-in learner before any mutation.

Bind policy, basis, raw protected deltas, stable identities and first-certification
provenance into checkpoint identity. Keep source ancestry suitable for authentic
older-source replay. Original scoring eligibility and its hash remain separate
from projected-direction hash, basis-before/after hashes, norms and actual mutation.
Export auxiliary arrays and records with the saved synaptic checkpoint so an
independent audit can reproduce protected margins and projection history.

## Float64 checks and declared no-ops

With D the feature dimension and eps the float64 machine epsilon, use the fixed
exhaustion/dependence threshold `64*eps*D*max(1, ||d||)` on residual norm. A residual
at or below it is treated as zero, reported as such and never amplified. Use this same
rule for basis insertion. No threshold is tuned against model outputs.

For every protected raw delta d_i, verify
`abs(d_i·p) <= 64*eps*D*max(1, ||d_i||*||p||)`.
After mutation, verify its actual margin change is at most
`128*eps*D*max(1, abs(m_before), ||v_before||*||d_i||, ||v_after||*||d_i||)`.
Also verify normalized basis orthogonality with absolute tolerance `64*eps*D`.
Retain maxima, tolerances and certification margins, not just pass flags.
Invalid/nonfinite state or a failed numerical check must raise and preserve an
error artifact; it must not be scored as a successful preservation result.

In real arithmetic, protected contrasts have zero margin change and the current
contrast gains `2*d·p/sqrt(1+d·d)=2*||p||²/sqrt(1+d·d)`. Audits must use actual
original/projected directions and native mutation, allowing declared floating
tolerances rather than assuming the old unprojected gap formula.
Protection applies only to certified directions in the fixed feature space.
It can lock in early overshoot, obstruct beneficial transfer or exhaust useful
directions; it does not protect unseen competitors or absolute STOP clearance.

## Measurements and source freeze

Record new/replay offers, native synaptic updates, original-gate no-ops,
projection stalls, basis rank and certification count, and all teacher, primary
and fallback outcomes. Save original and projected direction custody plus enough
state to reconstruct every update. Check frozen zero synapses and zero learned
basis/delta payload; exposure receipts outside inference are counted separately.

At final training weights, save all sixteen read-only TRAIN-query scores/ranks
before fresh ingestion and repeat them after ingestion. Assert both synapses and
auxiliary learned state are unchanged; only retained source state may grow. Use
these probes to separate learned interference from newly introduced competition.
Keep per-query final leaders, actual executed-FAIL steps, admitted negative pair
IDs/margins and the existing final raw-rank diagnostics. Probes never select
teaching/replay/consolidation. Execute the existing 572 transfer/retention policy
actions, preserving source absence, decoys and read-only state.

Use the original official KS2 passage, questions and marking policy unchanged,
with one ingestion per learner and pending human marks preserved. No exam
feedback is provided. Report first-attempt learning, later transfer and retained
skills separately; teacher successes are not autonomous recovery.

Count packed basis/raw-delta arrays, provenance, scratch, external ledger and
encoder bytes without double-counting shared storage. At D=1621 and capacity 48,
each packed 48-row float64 matrix costs 622,464 bytes; frozen arrays have zero
rows. Avoid a dense D-by-D projector. Report O(KD) projection/check work, phase
and total wall time and parent memory scope. Equal offers do not mean equal compute.

Before the run, verify synthetic original-gate/normalizer parity, protected-margin
preservation, dependent/conflicting directions, exhausted projections, frozen
auxiliary state, receipt custody and baseline identity. Run the required native
suites after source stabilizes. Commit reviewed sources, protocol and fresh
cohort before one experiment; bind source/native/encoder identities in its artifact.
Preserve interruptions and harness corrections separately. Return the outcome,
including failure or budget exhaustion, to the active integration plan.
