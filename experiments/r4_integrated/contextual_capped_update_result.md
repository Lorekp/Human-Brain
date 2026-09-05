# Capped corrections fail the integrated comparison

Completed 2026-09-05 from source `2c6218254e530bc39e47f236872d3d6302e9e8be`,
following the [frozen protocol](contextual_capped_update_protocol.md). The
[artifact](../../benchmarks/contextual_capped_update_v1.json) and
[saved synapses](../../benchmarks/contextual_capped_update_v1.npz) preserve the
single completed run. **Do not adopt the capped rule:** it reduced fresh and
retained neuronal execution. The existing default remains unchanged.

| Primary affine neuronal policy | Fresh edits | Retained edits | Unsupported STOPs |
| --- | ---: | ---: | ---: |
| FIXED, updating | 3/8 | 2/16 | 0/2 |
| CAPPED, updating | 0/8 | 1/16 | 0/2 |
| Either matched frozen learner | 0/8 | 0/16 | 2/2 |
| External cosine, either arm | 0/8 | 0/16 | 0/2 |

Neither updater handles both properties for any fresh or retained entity. FIXED
solves fresh javu transport, nevo transport and veju audit; it retains falu audit
and horo transport. CAPPED retains only dofi audit. Its one retained STOP is a
failure, not a solved task. The other capped failures and all fresh failures have
wrong raw selections; lowering STOP is not a remedy.

All 52 updating affine selections and executed outcomes match their raw-score
references. The old saturated readouts still hide FIXED's successes; CAPPED's
old and affine readouts give the same scores. Predeclared feature ablations are
recorded without retrospective selection: removing the binding block gives
CAPPED 3/16 retention, while collapsing positions gives FIXED 5/16. Neither is
adopted from this spent cohort. No generalization or protected gate passes.

## Exposure and learning

Both arms use the same BINDING features, gain 16, rate ceiling 1, affine primary
teaching, motor, sixteen openly reused requests and 48 encounters. Each admits
32 actual edited FAIL/teacher-PASS pairs and offers 118 older replays: 150 pair
offers and 300 feedback events, identically exposed to its frozen mate. The
newly named transfer instances preserve the previous four development strata;
they do not establish independent language generalization.

| Teaching measurement | FIXED | CAPPED |
| --- | ---: | ---: |
| First-attempt successes / 48 | 6 | 8 |
| Fallback attempts / successes | 4 / 0 | 8 / 0 |
| Teacher executions / successes | 48 / 48 | 48 / 48 |
| Admitted neuronal / raw / external negatives | 28 / 3 / 1 | 25 / 6 / 1 |
| Native updates: new + replay | 32 + 48 | 32 + 88 |
| Wrong encounters after pair budget exhausted | 10 | 8 |

CAPPED's extra teaching successes do not survive final retention. The updaters
generate different on-policy negatives and trajectories despite equal budgets;
this is a learning-policy comparison, not an identical-ledger step-size test.
All 120 capped native updates use rates below one: minimum 0.00033323, median
0.08156 and maximum 0.21379. FIXED uses rate one for its 80 native updates. Its
final weight array exactly matches the prior exploration BINDING array; the new
fresh scores come from different transfer instances, not changed fixed weights.

## Stability and feedback coverage

| Final certified pair margins | FIXED | CAPPED |
| --- | ---: | ---: |
| Negative | 6 | 8 |
| Nonnegative, below unit margin | 0 | 12 |
| Satisfied, tolerance 1e-10 | 26 | 12 |
| Minimum margin | -15.668 | -3.046 |

Every final negative pair was positive at its first and last recorded offer.
The cap reduces the worst negative magnitude but does not improve the number
of erased constraints or executed retention. Unit-margin fit alone is secondary
to ordering and actual actions.

Repeated encountered-query probes compare the same retained sources around each
correction block. These are thresholded raw selections, not independent tasks:

| Arm / stage | Comparisons | Correct to incorrect / initially correct | Incorrect to correct / initially incorrect |
| --- | ---: | ---: | ---: |
| FIXED new correction | 381 | 44 / 66 | 57 / 315 |
| FIXED rehearsal | 381 | 40 / 79 | 28 / 302 |
| CAPPED new correction | 388 | 53 / 76 | 54 / 312 |
| CAPPED rehearsal | 388 | 39 / 77 | 37 / 311 |

Only one loss, during capped rehearsal, becomes STOP. STOP-to-correct accounts
for one gain in each FIXED stage and one after a new CAPPED correction. All
remaining losses choose a wrong answer. Correction blocks occur at
different steps in the shared teaching stream, making these descriptive comparisons.

FIXED's fourteen failed retained queries split into nine whose recorded pairs
all fit and five with an erased pair. None of those nine final wrong leaders was
an **admitted** negative for its query. Seven were actually executed incorrectly
later, at steps 38, 39 and 43-47, after the pair budget had been exhausted; two
were never executed. Overall, only two of fourteen final wrong leaders had been
admitted. All belong to the training inventory, and fresh ingestion changes no
FIXED old selection. This is direct evidence of incomplete feedback coverage,
alongside interference, rather than proof of a representation-capacity limit.

CAPPED's fifteen failures include three with all recorded pairs satisfied, six
with erased pairs, five with other subunit margins, and one with no pair. Only
two final wrong leaders were admitted negatives; six were ever executed wrong.
Five leaders come from newly ingested transfer sources. Fresh ingestion causes
no additional correct-to-incorrect outcomes, but changes some wrong selections.

## Verification, exam and resource cost

All 47 focused core tests pass with no skips. A separate old-source comparison
preserves eight default states and 56 decision hashes across both feature
settings, plus native comparison arrays and existing receipt fields. Structural
QA verifies routing, single exam ingestion, ledger accounting and STOP reporting.
All 33 declared source files match the committed bytes; encoder, native DLL,
checkpoint and artifact identities verify. There were no interrupted runs.

Independent receipt algebra verifies all 600 updater/frozen receipts and all 64
final pair margins. The worst own-gap formula error is 1.15e-14; accumulating
the recorded steps reconstructs saved weights within 1.23e-15. Frozen weights,
pure-query weights and persistent bias remain zero. All 572 transfer/retention
executions preserve decoys and have zero live documents. Checkpoint lineage and
read-only evaluation checks pass; each arm receives identical frozen feedback.

The original KS2 passage is ingested once per learner. FIXED's affine answers
are `tenrec that live` and `tenrec`; CAPPED answers `fragile` twice. Each updater
has zero confirmed marks and two marks pending official-rubric human scoring.
Frozen answers STOP for 0/2. No question, rubric or marking policy was changed.

The complete comparison takes **284.18 seconds**, including diagnostics and
child waits. Parent peak working set is **331.16 MiB**, final private memory
589.70 MiB and traced Python peak 117.03 MiB; child memory is excluded. Teaching
takes 87.29/90.77 seconds for FIXED/CAPPED, transfer plus retention 39.56/40.44,
and the exam 11.18/11.27. Shared motor setup takes 0.18 seconds and encoder load
0.33. These totals do not isolate a learning-rule speed effect.

Each learner ends with eleven externally retained documents, 4,705 raw bytes,
2,298 candidates and 23 windows. Counted retained array/byte payload is 7,256,335
bytes, excluding Python object overhead. Each external correction ledger contains
829,952 bytes of eligibility floats within its roughly 2.24 MB compact serialized
record; these sizes overlap and must not be added. The frozen ONNX encoder is
90,405,214 bytes, separately retained. All controls share the learned motor and
isolate evidence selection; whole-system core-disabled evidence remains absent.

Artifact SHA-256: `0da5647faa19aa3383288a8879bd74f8dbdc917522ebe940b685a0d8c97fd9e1`.
NPZ SHA-256: `4dbfa505b71ee310c2519cd49d1175a4518fe8fb783b3f8dd318534434bc9309`.

## Next decision

Retain the fixed rule and affine action path. One short TRAIN-only feasibility
diagnostic should check whether the existing features can jointly rank the
sixteen taught answers above all wrong candidates in the four training documents.
Keep any host-solver solution outside the learner and use no transfer or exam
labels. A verified feasible witness would rule out impossibility on that finite
roster, not prove learnability or generalization; solver failure is inconclusive.
Then return to an integrated comparison of feedback allocation, specifically the
late executed errors currently excluded while older pairs consume replay offers.
The [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns that bounded work. Do not tune the cap, select ablations, or reopen gates.
