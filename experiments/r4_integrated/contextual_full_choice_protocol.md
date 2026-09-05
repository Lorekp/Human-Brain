# One full-choice acquisition attempt

Public TRAIN development protocol, 2026-09-05. Preparation starts **07:32 UTC**
from clean `9be1ae71ff72377fec29d6fe6c8a5b5f2875fee6`: at most two hours of
preparation, then one four-pass acquisition run within two hours total wall time.
**Numerical policy fixed after synthetic preflight: learning rate 1/256,
joint gradient norm bound 1, zero GO/NO and STOP initialization.**
No learner/encoder preview of this comparison's inputs. Existing synthetic
arithmetic, encoder-contract and inert custody/routing QA are allowed.

## Question and boundary

The [interface review](../../docs/research/contextual-trainable-interface-review.md)
identifies correct recorded pair margins alongside wrong current competitors.
Test whether explicit full-choice supervision can acquire a broader declared
TRAIN package in the persistent reading -> selection -> real Python edit ->
execution loop. This does not isolate objective effects against historical AROW:
training, budgets and STOP supervision change together. No fresh transfer cohort,
representation-capacity conclusion or protected gate is part of this attempt.

Keep the frozen external encoder, original gain16 BINDING features (D=1621),
generic one-to-three-word candidates, source windows and previously learned
neuronal literal-edit motor. No projected bilinear block, recurrent producer,
entity/property predicates, answer-conditioned candidate filters or encoder
updates. The new learning rule is conventional supervised SGD over fixed neural
features and a learned STOP score; it is not AROW or local biological plasticity.

## Teacher authority and full-choice objective

For fixed TRAIN candidate occurrences plus STOP, let C be the complete output
inventory. A teacher explicitly authorizes exact UTF-8 answer bytes or STOP.
For an answer, G contains **every occurrence emitting those exact bytes**; for
STOP, G contains only STOP. Empty supported G is a contract failure. Do not
silently choose one duplicate or teach source provenance from answer equality.

```text
L = logsumexp(z[C]) - logsumexp(z[G])
a[c] = softmax(z[C])[c] - 1[c in G] * softmax(z[G])[c]
g_v = sum(candidate c, a[c] * feature[c])
g_stop = a[STOP]
```

Candidate scores are feature dot (GO minus NO); STOP is a separate learned
scalar. Initialize GO/NO and STOP to zero. The effective parameter coordinates
are v=GO minus NO and STOP, with one joint L2 gradient clip; apply the candidate
step symmetrically to GO/NO by minus/plus half the effective-v step. Keep bias
unchanged at zero. Stable log-sum-exp uses finite current scores. The precise parameter
coordinates, joint gradient clipping and fixed SGD step are recorded below. No momentum, covariance,
margin gate, extra optimizer or adaptive step schedule is implicit in this loss.
Every scheduled lesson, including already-correct attempts, may train once.
Unexecuted candidates receive surrogate gradients, never fabricated FAIL labels.
A teacher PASS authenticates the performed repair; the separate exact-output
teaching contract authorizes this stronger imitation objective. Neither proves
that every other possible program edit would fail.

Each encounter first records an actual label-blind learner decision. A selected
literal goes through the shared motor and real execution in a source-absent
workspace; STOP records an actual unchanged-workspace no-action decision. Then
record the independently supplied teacher action: a supported literal is really
edited/executed even if the learner was correct; unsupported teaching records
explicit authorized STOP and unchanged workspace, without inventing a repair or
calling learner STOP its own target. Keep selection correctness, edit execution,
test result and teacher authority distinct. Teacher repairs never count as
learner success. Reject invalid authority/custody before any learning mutation.

## Fixed sources, schedule and controls

Freeze one new package with 64 requests: 48 supported and 16 unsupported, at
least sixteen source/task bundles, one-, two- and three-word answers, both roles,
alternate question forms, competing entities and incidental lexical overlap.
Unsupported items include entities present without the requested relation.
Assignments must be consistent across the complete source pool. Freeze fixture
bytes, source IDs, spans, task baselines, groups and all schedules before outputs.
These authored tasks are reusable development; they are not official exams.
Freeze contextual_full_choice_train.json at SHA256
`9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059`.
Its sixteen TRAIN documents contain 3,195 raw bytes and 1,458 candidates; the
four growth documents add 934 bytes and 423 candidates. These are source counts,
not semantic candidate filters.

Ingest the **complete TRAIN source pool once before teaching**. Every original
or replay offer scores exactly this same candidate universe. No source addition,
answer change or source-conditioned curriculum occurs during acquisition.
Run four declared passes over the 64 requests: 256 encounters and 256 teacher
opportunities, including 192 supported teacher executions and 64 explicit STOP
teaching receipts. At most one deterministic older lesson replay follows each
encounter; no replay before a certified older lesson exists, no same-encounter
self-replay, no loss/rank/label-driven replay selection, and no additional teacher
execution for replay. After every encounter except the first, replay the immediately preceding
encounter's immutable lesson: 255 replays and 511 offers per learner. Adjacent
tasks differ in the frozen 64-request order, including pass boundaries.
The old 48/150 budget does not apply. Count encounters, learner executions,
teacher executions/no-actions, replay offers and nonzero updates separately.

Run an identically initialized no-learning mate with the same source inventory,
fixed features, candidate/STOP family, motor, teacher authority and original/replay
offers. It computes diagnostics but changes no learned bytes. Its own label-blind
choices and actual outcomes are independent of the updater. An external-only
selector uses the same candidate inventory and motor. It selects the unique
maximum span/query unit-vector cosine, regardless of sign or magnitude, with
normalization floor 1e-12. It returns STOP only for an empty inventory or exact
maximum tie; no positive threshold or learned STOP competitor participates.
Full feature rows supply its decision eligibility but do not choose the winner.
It receives no gradient credit. Neither control substitutes teacher answers
into inference. The historical
AROW results remain contextual evidence, not a matched objective control.

Use learned-weight raw selection and affine/interpolated neuronal readout as
separate recorded policies with identical candidate and learned STOP scores.
Keep unique-earliest selection and ties as STOP; do not break occurrence ties by
index or by teacher bytes. The primary readout is affine/interpolated, with the same frozen exact-tie
rule as the core API. Record raw/affine choices alongside updater/frozen attempts
without extra execution. Raw/affine disagreement is a readout diagnosis, not extra supervision.
Inference sees query plus retained sources/state, never G, labels or teacher logs.

## Acquisition probes and stopping decision

After the final scheduled update, disable learning and execute all actual TRAIN
requests plus frozen alternate-form requests. Alternate forms add no teaching.
Report exact supported executions, correct unsupported STOPs and paired control
outcomes, separated by role, answer length and declared alternate-form groups.
Require **44/48 supported actual-TRAIN executions, 15/16 unsupported STOPs, and
at least 80% in every declared role/length/alternate-form group**; use integer
ceil(0.8*n), not rounded percentages. Each actual/alternate phase requires
26/32 for each pooled role (supported plus STOP), 13/16 for each positive answer
length, and 7/8 for each of the eight declared form groups. Actual and alternate
groups are evaluated separately, never pooled into a larger denominator.
The declared unsupported answerability group also requires 13/16 in every phase;
the separate actual-query global STOP threshold remains stricter at 15/16.

Each of the actual and alternate-form probe sets has 64 requests. Three policies
(updater affine, no-learning affine, external) produce 768 teaching first attempts
and 768 evaluation executions across actual/alternate before/after growth.
The 256 teacher actions bring the total to 1,792 executions, excluding two motor
demonstrations. For teacher STOP, execute the unchanged workspace with its
declared stale baseline as the expected setting and bind teacher_kind=
authorized_absent_evidence_stop, answer=null and workspace_unchanged=true.
Its passed result certifies no-action preservation only, not an unsupported repair.

Then ingest exactly four frozen unrelated growth documents without feedback and
repeat the same probes. Require the acquisition thresholds still hold and no
more than five percentage points loss over the same 64 actual requests (at most
three net successes lost). Report alternate-form growth changes separately and
retain their 80% thresholds. Source growth may change storage/checkpoint identity
but must leave learned parameters, optimizer state and counters byte-exact.

Record updater score-inventory diagnostics for all 64 queries in each of four
post-teaching phases (256 read-only inventories): authorized target mass, selected
leader and worst wrong competitor. Labels are evaluator-only in these diagnostics;
they cannot choose exposure or alter learning. They add phase cost, no executions.

An acquisition pass earns consideration of a separately frozen fresh comparison,
not adoption over historical AROW or a general reading/coding capability claim.
Failure closes this configured attempt: report optimization/coverage, readout,
question-form and source-growth errors without automatically adding epochs,
rates, labels, projections or a recurrent successor. A conditional trainable
producer requires a separate decision and an identical output-family control.
Preserve terminal/interrupted evidence; no invisible retry or post-output tuning.
On failure, retain completed rows and, when available, the last learned arrays
and audit basis in a partial NPZ. A partial artifact is not a completed outcome.

## Minimal auditable state and numerical policy

Use fixed learning rate **1/256** and joint gradient norm bound **1.0**, chosen
on synthetic numerical/acquisition QA before TRAIN outputs. This is no guarantee
of monotone loss or task fit, and permits no fixture-conditioned adjustment. Use FullChoiceEvidenceMemory with a sealed training
pool, teacher_lesson custody and observe_teacher replay; growth rejects all
later offers. Frozen yoking accepts the updater's certified lesson while binding
its own current parent, proposed gradient and effective learning rate zero.

Export a compact audit basis: normalized candidate vectors/window contexts,
candidate occurrence/byte identities, interaction-window positions/words, and
each distinct query's normalized sentence/word vectors and word metadata. Rebuild
the fixed 1621 features independently; do not archive full feature matrices per
offer. Retain raw sources, queries, frozen encoder identity, configuration and
array shapes/dtypes/hashes. These evaluator exports are measured artifact bytes,
not hidden inference memory. Each offer also stores its full raw 1622-coordinate
gradient so exact published-state replay can be separated from independent
numerical feature/gradient verification without new encoder/model/native calls.

Each offer binds lesson ID, original teacher authority/execution receipt IDs,
query/source-universe/feature identities, explicit target indices, current
parent/learned-state hash, loss, raw gradient/vector/hash/norm, clipping factor
and resulting state hash. The authenticated receipt binds the target set; target
probability exp(-loss) and applied update norm are independently derivable, not
additional required stored fields. Replay references the immutable original lesson
and recomputes current scores/gradient; old pair eligibility is not current
credit. Save initial/final full parameters and enough ordered receipts to replay
all states. Publication order is norm=norm(g), c=min(1,clip/norm) (c=1 at
zero norm), applied=g*c, change=(0.5*eta)*applied[:-1], GO-=change,
NO+=change, STOP-=eta*applied[-1]. Frozen uses effective rate zero. No per-offer full covariance or dense state history is needed.

Independent audit reconstructs target sets from recorded authorized bytes,
checks original/replay order and authentic receipt joins, recomputes stable loss,
gradients/clipping/SGD and every available state hash from saved arrays. Compare
an independently ordered reference within `1e-10 + 1e-10*abs(reference)`;
report ambiguous score/tie boundaries without widening the frozen decision rule.
Require finite state, exact no-learning/inference/no-action mutation invariants,
initial-state equality and complete source/feature/teacher custody. If arithmetic
order prevents bit-exact independent reconstruction, state that explicitly and
verify stored state bytes with their recorded hashes; never relabel tolerance
agreement as exact identity. Synthetic finite differences cover the marginal
duplicate target, STOP target, clipping and frozen paths before freeze.

Count learner/frozen parameters, STOP scalar, scratch, candidate feature caches,
raw sources, encoder, motor, teacher/replay ledger and audit exports; report phase
wall times and execution counts. Authentication/replay arrays are not neurons.
Official reading remains a separately sourced/frozen companion under the
[reading plan](../../docs/research/reading-comprehension-evaluation.md); repeated
pending KS2 marks would not be independent generalization. This acquisition-only
run opens no new exam content or fresh/protected evaluation.
