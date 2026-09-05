# One projected-bilinear binding comparison

Public TRAIN development contract, 2026-09-05. Preparation starts **06:45 UTC**
from clean `f944fb10888a8748e33410c55fc0ef9695ef4f7d`. Freeze this protocol,
reviewed source, projection identity and three input fixtures before one run.
No learner/encoder preview of these TRAIN or fresh inputs. Existing synthetic
arithmetic, encoder-contract, custody and inert harness QA remain allowed.

## Failure, hypothesis and decision

The [covariance curriculum result](contextual_covariance_curriculum_result.md)
retains positive admitted comparisons but loses question-form and span accuracy.
VARIED acquires 13/16 taught forms yet only 5/16 common forms; every supported
fresh selection uses the right source and a wrong span/relation. Earlier TRAIN
gold values were single words, so multiword errors do not establish inability
to learn boundaries. The [binding review](../../docs/research/contextual-learned-binding-review.md)
selects one association-block test under matched new mixed-length supervision.

Compare **BASELINE** and **PROJECTED**, both using adopted AROW. PROJECTED adds
the fixed-projection bilinear block below. Hypothesis: learnable cross-coordinate
query/span associations improve actual transfer beyond merely supplying missing
demonstrations. This does not assert learned parsing, role variables, sufficient
fresh representability or a pure parameter-count effect.

Adopt PROJECTED only for strictly higher primary affine supported fresh executed
success, with no decrease in common sixteen-request executed retention, either
four-request familiar/broader fresh stratum, or two correct unsupported STOPs.
Report paired gains/losses and complete entities. Preserve separate absolute
targets: 6/8 fresh, 12/16 retained and both properties for three fresh entities.
Relative adoption is not the integrated gate; unsupported ties at zero do not
establish abstention. Diagnostic ablations cannot replace the primary outcome.

Allow at most two hours of preparation from the recorded start and one sequential
BASELINE/PROJECTED comparison within thirty minutes total wall time, including
child waits. No rank, seed, normalization, gain, rate, wording, value-length or
epoch search. Preserve interrupted evidence before a separately recorded recovery
decision. Failure closes this candidate: no automatic AROMA, projection variant
or extra practice successor. The earlier curriculum decisions remain closed.

## Fixed projection and unchanged learning

Keep the frozen MiniLM encoder, original BINDING features at gain16, generic
one-to-three-word candidates and 48-word windows, neuronal literal-edit motor,
affine currents and interpolated GLIF. Both start with zero evidence synapses,
`comparison_step_policy="arow"`, learning rate1, protection off, float64 S=I,
r=1. BASELINE uses `projected_bilinear=False`, D=1621; PROJECTED uses True,
requires candidate interactions, and has D=2645. The existing native AROW
algorithm is unchanged; there is no gradient through the encoder or projections.

Generate two immutable 384×32 float64 matrices exactly once per learner:

```text
rng = numpy.random.default_rng(0)
for matrix in (Pq, Ps), in that order:
    A = rng.standard_normal((384, 32))
    Q, R = numpy.linalg.qr(A, mode="reduced")
    signs = where(diag(R) < 0, -1.0, +1.0)  # zero maps to +1
    matrix = contiguous_float64(Q * signs[None, :])
```

For the existing unit query vector q and unit span vector s, append
`16 * outer(q @ Pq, s @ Ps).reshape(1024, order="C")` after the original 1621
features. Do not renormalize projected vectors or change existing feature blocks.
Teacher, inference and replay eligibility use the same ordering and scale.
Record recipe, NumPy version, matrix shapes/dtype/bytes/hashes and bind them into
representation, source and checkpoint identity. Same-policy frozen mates use
identical fixed matrices; incompatible projections cannot share certified custody.

The [adopted recurrence](contextual_covariance_protocol.md#exact-covariance-learning-policy)
remains binding at each arm's D. For original accepted-minus-rejected delta d,
m=(wGO−wNO)·d<1, use p=S d, u=d·p, beta=1/(u+1), alpha=(1−m)beta,
GO/NO changes ±alpha p/2, and S←S−beta ppᵀ using pre-update state. Frozen,
zero-delta and satisfied-margin offers change no learned bytes. Keep persistent
bias zero; no absolute-feedback bypass, covariance-scored inference, solver
weights, numerical repair or extra normalization. Original delta and transformed
AROW direction remain separately authenticated; neither is a learned projection.

## Matched teaching and prospective evaluation

Freeze `contextual_projected_binding_train.json`, shared byte-for-byte across
arms: sixteen facts, eight single-word/eight two-word positives (four of each
length per property), four documents and a 48-encounter schedule. Freeze
`contextual_projected_binding_common_train.json` with the same facts,
values, identities, documents and order but different sixteen question forms.
Freeze `contextual_projected_binding_cases.json` referencing common TRAIN, with six fresh
documents, eight supported requests (four familiar/four broader, two entities
and both properties per stratum) and two unsupported requests. Include reachable
multiword values in both strata. Register exact file hashes in source inventory.
This is prospective authored development, not a fresh version of spent cases.

Reviewed SHA256 identities, in actual/common/cases order:

```text
e45e9497e4d8edfe228fce5c1c1ba976a939ec41f95087793e80279486c61291
4e2e68a95e22ffe07f256caaed6bc5097a245d3159ac40ce8ad9df048d1c30c0
671fc26009ca1a3e70b8ae7677b7fcdac4e975e02d523aece083ed764d56149a
```

Only actual TRAIN supplies ingestion and teaching questions; common TRAIN
supplies old-query execution/ranks and before/after-fresh probes. Ingest each
actual document once per arm. Verify unique teacher occurrences, unambiguous
relations and whole-span reachability statically. No teacher/entity/property/span
filter or silent occurrence selection. Both arms receive identical supervision
scope, including length variation; this is not reopening the old wording sweep.

Keep [RESERVED allocation](contextual_reserved_feedback_protocol.md): exactly
48 encounters and 48 teacher executions per arm; at most one newly executed
edited-FAIL/separate teacher-PASS pair per encounter; at most 150 total offers
including replay and native no-ops. Reserve 48−t potential later corrections;
replay up to four distinct eligible older pairs in unchanged cyclic order,
excluding the new pair. No padding, extra epochs, teachers or retries. Primary
STOP retains unique raw-candidate exploration, then external cosine only if no
unique raw candidate exists. Correct exploration does not create a negative.

Each arm's same-policy frozen mate receives that arm's exact documents and
original/replay exposures. Equal budgets do not imply identical on-policy
negatives, ledgers, updates or runtime. Teacher repairs are not autonomous success.
The external pretrained encoder and receipt-certified credit are disclosed;
evidence synapses learn preferences and feed the shared previously learned motor.

## Diagnostics, custody and numerical audit

Retain eleven existing policies per arm and add only PROJECTED's two executed
raw-score ablations: `projected_bilinear_ablation="disabled"` zeros the new block;
`"diagonal_only"` retains its 32 entries i=j. These refer to projected coordinates,
not the original encoder diagonal. They leave original BINDING features intact.
Total transfer/retention executions are 624; the 52 primary raw/affine occurrence
comparisons are unchanged. Ablations are read-only, cannot supply feedback and
cannot satisfy adoption. Dependence on a block does not exclude a capacity effect.

Save actual-form/common-form probes at the same checkpoint and after fresh
ingestion. They are raw diagnostics, not additional executed successes. Coverage
requires exact query and occurrence identity; a zero common-query join does not
mean an underlying fact was untaught. Probe answers never schedule learning.
Report whole-span successes conditional on correct source, multiword and relation
errors, pair-margin histories and unencountered leaders. Preserve source-absent
execution, selected/materialized edits and decoys for all policies.

Save decisions, original eligibility, ledger/feedback chains and full learned
arrays with `checkpoint_arrays(include_learning_state=True)`. NPZ has sixteen
keys: BASELINE's six weights/bias/covariance arrays and PROJECTED's ten including
updating/frozen `projected_query_matrix` and `projected_span_matrix`. Projection
bytes remain fixed through teaching, ingestion, diagnostics and exam. Frozen S
stays I and synapses stay zero; all evaluation preserves trained learned bytes.
New teaching means no prior COMPACT checkpoint parity claim. Require unchanged
default features/receipt identity and meaningful synthetic baseline parity instead.

Independently replay saved offers without learner/encoder/native calls, using
exact folded state for recorded hashes and a separate numerical recurrence.
Audit the projection recipe and appended block from saved q/span eligibility
coordinates; do not substitute tolerant features into authenticated decisions.
Use `abs(actual-reference) <= 1e-10 + 1e-10*abs(reference)` throughout. Gates remain
strict m<1; report near-unit margins and resolve disagreements without widening
tolerances. Validate finite state, symmetry within `64*eps*D*max(1,maxabs(S))`,
and one independent final Cholesky per matrix without jitter or repair. Frozen
and no-op equality is byte-exact; pure-query weights/bias stay zero and the
zero-delta pure-query covariance block retains initial identity.

PROJECTED's covariance costs 55,968,200 bytes versus BASELINE's 21,021,128;
two projection matrices add 196,608 bytes per projected learner. Count frozen
allocations, scratch/export/audit copies, expanded eligibilities, ledger, source,
encoder and candidates without overlap. Report phase/total time, native work and
parent-memory scope separately from excluded child memory. Size is not runtime.

Before freeze, review core feature/identity/default-parity tests and new inert
routing/serialization/export checks; reuse attested unchanged-native QA. Keep
one unchanged official KS2 Q6a/b helper call per arm, existing exam policies only,
one passage ingestion per learner, no feedback and pending human rubric marks.
These repeated checks are not independent comprehension. Fresh official reading
remains a separately sourced/frozen [companion manifest](../../docs/research/reading-comprehension-evaluation.md).
Close the single comparison with evidence and claim limits; no post-output tuning.
