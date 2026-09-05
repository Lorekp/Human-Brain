# Covariance-aware acquisition in the persistent Python session

Public TRAIN development contract, 2026-09-05. Preparation starts **05:35 UTC**
from clean `56709d4e4c68a23f1607e91b735e3a8c7c3b1f3f`. Freeze reviewed source,
this protocol and the new fixture before one comparison. No learner/encoder
preview of this comparison's TRAIN or fresh fixtures is allowed; existing public
synthetic, encoder-contract, arithmetic and custody QA are allowed.

## Failure, hypothesis and decision

The [curriculum result](contextual_curriculum_result.md) retains only 4/16 old
actions in either arm. Both lose twelve authentic TRAIN contrasts that previously
reached unit margin. Query-form sensitivity and uncovered competitors coexist
with this measured erasure. The [acquisition review](../../docs/research/contextual-acquisition-review.md)
selects one covariance-aware learning rule to test whether different update
geometry improves useful acquisition with the current fixed representation.
The hypothesis does not assert sufficient fresh representation capacity.

Compare **FIXED**, the compact RESERVED baseline, with **COVARIANCE**, the
pairwise AROW adaptation below. Adopt COVARIANCE only if its primary affine
neuronal policy has strictly more of the sixteen executed retained successes,
and no decrease in supported fresh total, either four-request familiar/broader
fresh stratum, or the two correct unsupported STOPs. Report paired gains/losses
and both-property entity counts. Margin fit alone cannot satisfy adoption.
Absolute targets remain 6/8 fresh, 12/16 retained and both properties for three
fresh entities; relative adoption is not the full integrated milestone or a gate.

Allow at most two hours of preparation/diagnosis from the recorded start, then
one sequential FIXED/COVARIANCE comparison within thirty minutes total wall
time including child waits. No regularization/covariance/rate search, additional
teacher dose, extra epochs or post-output fixture adjustment. Preserve any
interruption before a separately recorded infrastructure recovery decision.
Close this candidate at its budget boundary, including a negative result.

## Fixed inputs, actions and authentic experience

Both arms use `contextual_curriculum_compact_train.json`: the same sixteen
facts/queries, four source documents and original 48-entry schedule. Ingest each
document once. Preserve unique teacher occurrence custody, generic one-to-three-
word candidates and 48-word windows. No answer/entity/span filter is permitted.
`contextual_covariance_cases.json` supplies one newly frozen common transfer
roster: six documents, four familiar supported requests, four broader supported
requests, and two unsupported requests. Each supported stratum has two entities
with both properties and at least one multiword answer. Review unambiguous
support, candidate reachability and substitutions absent from prior cohorts
statically; no learner output may influence authoring.

Keep the frozen MiniLM encoder, BINDING features at gain 16, zero initial evidence
weights, learned literal-edit motor, affine currents and interpolated GLIF action
path unchanged. Both arms disable protected-contrast projection. FIXED keeps the
original rate-one comparison update. No solver witness or prefitted covariance
enters either learner. The frozen encoder is externally pretrained. The motor is
a previously learned neuronal literal-edit path, shared scaffolding for this
evidence-learner comparison; evidence weights learn which source occurrence to use.

Keep [RESERVED allocation](contextual_reserved_feedback_protocol.md): exactly
48 teacher executions per completed arm, at most one new executed edited-FAIL /
separately executed teacher-PASS pair per encounter, and at most 150 total new
plus replay offers, including gate no-ops. At step t reserve 48−t potential new
corrections; replay up to four distinct eligible older pairs in existing cyclic
order, excluding a newly admitted pair. Do not pad underspend. Primary STOP may
use the existing unique raw-candidate exploration, then external cosine only if
no unique raw candidate exists. Correct exploration provides no negative pair.
No additional retries, teacher executions or automatic ledger-wide supervision.

Each updater has its own same-seed frozen mate receiving exactly that arm's
documents and certified original/replay feedback. Updating arms can generate
different negatives and mutation counts; equal budgets do not mean identical
ledgers or compute. Teacher repairs do not count as autonomous success.

## Exact covariance learning policy

Use `comparison_step_policy="arow"`, float64 dimension D=1621, covariance S=I,
fixed regularization r=1, and constructor learning rate one with protection off.
Reject other rate/protection combinations rather than silently composing rules.
For original authenticated same-query contrast d=accepted−rejected and effective
weight difference v=w_GO−w_NO, compute m=v·d from current weights. The reference is
[AROW equations 7/9 and Figure 1](https://proceedings.neurips.cc/paper/2009/file/8ebda540cbcc4d7336496819a46a1b68-Paper.pdf#page=4),
adapted by treating d as the input and +1 as its executed comparison label.

When learning is enabled, d is nonzero and m<1, use only pre-update values:

```text
p = S d
u = d·p
beta = 1 / (u + 1)
alpha = (1 - m) * beta
w_GO' = w_GO + (alpha / 2) p
w_NO' = w_NO - (alpha / 2) p
S' = S - beta p pᵀ
```

Keep persistent bias zero. The zero-delta, satisfied-margin (m>=1), and frozen
cases change neither weights nor covariance. Offers still receive custody records
and count against the same budget. Report gate status, actual weight mutation
and covariance mutation separately; rounding must not fabricate a synaptic change.
Early no-ops need not compute p/u/alpha/beta; mark uncomputed fields explicitly
rather than presenting placeholders as measured transformed directions.
Never apply the old `2d/sqrt(1+d·d)` formula to COVARIANCE receipts.

This policy changes direction and magnitude together. Its own expected gap is
`m_after = m + alpha*u`; for finite positive r it approaches, rather than enforces,
unit margin. It provides no exact old-margin preservation or generalization
guarantee. It is conventional global learning arithmetic implemented in the
neuronal training path, not an algorithm discovered by cells or biological proof.
Inference ranks only existing learned weights through the unchanged GLIF; no
covariance-conditioned inference, new uncertainty STOP or answer lookup is allowed.

## State, custody and resource accounting

Bind the opt-in policy, r, covariance bytes and learned weights into checkpoint
identity. Preserve source ancestry for exact older-occurrence replay; learning
must not change original source/candidate identity. Reject absolute-feedback
bypasses before mutation. Original eligibility/delta hashes stay distinct from
covariance-transformed direction hashes. Record before/after checkpoint, synapse
and covariance hashes, original gap/norm, u, alpha, beta, p hash/norm, gate/no-op
reason and actual mutation. Save original decisions and enough arrays to reproduce
all offers independently. Do not export a different covariance than inference's
learner owns. FIXED defaults, weights and old receipt semantics remain unchanged.
The experiment explicitly exports learning state with
`checkpoint_arrays(include_learning_state=True)`; a covariance hash without its
saved matrix is insufficient. Cache identity only when mutation invalidation is
guaranteed, and independently verify cached hashes against saved array bytes.

COVARIANCE and its frozen mate start with the same S=I; frozen S stays exactly I
and its learned covariance change is zero. Count the full initial allocation even
for frozen controls. FIXED has no covariance allocation. A dense D² float64 matrix
costs 21,021,128 bytes (20.05 MiB); count scratch and temporary copies separately.
Matrix-vector/outer-product work is O(D²) per applied offer, with 2,627,641 entries
per dense operation. These counts are not measured runtime. Report retained
raw/vector storage, covariance, ledger and encoder bytes without overlap, phase
and total wall time, and parent-memory scope excluding child memory.

## Numerical validity and independent audit

Runtime update validation requires finite inputs/state and outputs, positive u
for nonzero d and positive denominator. Symmetric paired entries use the same
rank-one subtraction; check symmetry at state validation and in the saved audit,
within `64*eps*D*max(1,maxabs(S))`, where eps is float64 machine epsilon. Invalid state
raises and preserves an error artifact; no jitter, clipping, eigenvalue flooring,
symmetrization repair or silent skip may replace a failed check. These checks
belong at update/state boundaries, not new matrix scans during inference.
No repeated Cholesky factorization is required per offer.

Afterward independently reconstruct each offer from saved original deltas and
the declared recurrence, without loading a learner, encoder or native updater.
Use `abs(actual-reference) <= 1e-10 + 1e-10*abs(reference)` for weight/covariance
entries, direction components and recorded scalar identities, including the
own-gap equation. Report maximum errors and counts. Gate decisions still use
strict m<1, not these tolerances: report near-unit margins separately and resolve
any reconstruction disagreement before scientific interpretation. Numerical
thresholds must not be widened after outputs. Final reconstructed/saved covariance
must pass one independent Cholesky check, with symmetry checked first and no jitter.

Frozen, gate-no-op and zero-delta learned-state equality is byte-exact. Fresh
ingestion, diagnostics, evaluation and exam must preserve both trained weights
and covariance exactly; only source storage can grow. Check exact FIXED trained
array parity with the preceding compact RESERVED checkpoint. Audit original pair
margins and exact-query/occurrence joins; no tolerant join or answer-byte matching.
Pure-query weights and persistent bias remain zero; covariance rows for the
zero-delta pure-query block retain their initial identity structure.

## Integrated measurements and freeze

Record first attempts, fallback and teacher outcomes, new/replay offers, actual
updates, final pair margins and admitted versus never-executed wrong leaders.
Measure actual/common old-query scores before and after fresh ingestion. Both
arms now use the same compact questions, so actual/common probes must agree at
the same checkpoint and are not independent query-form evidence. Probe answers
never select feedback, rehearsal or covariance updates.

Retain all eleven existing policies and all 572 transfer/retention executions;
verify raw/affine occurrence agreement, source absence, materialized edits, decoys
and read-only checkpoints. Source retention internally is not semantic compression.
Keep official KS2 Q6a/b unchanged, one helper call per arm and one passage ingestion
per learner across its exam policies. No exam feedback; human rubric marks stay
pending. Repeated items are regression checks, not fresh examination evidence.
New official material requires a separate [reading manifest](../../docs/research/reading-comprehension-evaluation.md).

Before freeze, verify native arithmetic with synthetic correlated contrasts,
conditional covariance updates, fixed-path parity, covariance custody, frozen
state and real FFI calls, plus required native suites. Verify harness routing,
strict serialization and failure preservation without project previews. Freeze
source/native/encoder identities and save trained arrays with the one artifact.
No projected protection, covariance variants or other representation change may
be added during the run. Return the evidence to the [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md):
reduced erasure with persistent form failures supports a binding review; fitted
contrasts with missing competitors supports a coverage review. Failure of this
one candidate does not prove either cause or authorize a sequence of tweaks.
