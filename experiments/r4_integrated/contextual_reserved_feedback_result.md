# Reserving feedback improves the integrated baseline

Completed 2026-09-05 from source `c5a14eabb0597685feedd2973e04c47f05b375b0`,
following the [frozen protocol](contextual_reserved_feedback_protocol.md).
The [artifact](../../benchmarks/contextual_reserved_feedback_v1.json) and
[saved synapses](../../benchmarks/contextual_reserved_feedback_v1.npz) preserve
the single completed run. **Adopt RESERVED for subsequent development:** it
improves retention without reducing fresh execution under the same offer budget.
Absolute capability remains weak; no roadmap or protected gate passes.

| Primary affine neuronal policy | Fresh edits | Retained edits | Unsupported STOPs |
| --- | ---: | ---: | ---: |
| BASELINE, updating | 3/8 | 2/16 | 0/2 |
| RESERVED, updating | 4/8 | 4/16 | 0/2 |
| Either matched frozen learner | 0/8 | 0/16 | 2/2 |
| External cosine, either arm | 0/8 | 0/16 | 0/2 |

RESERVED solves both properties for one fresh entity, tavi, and one retained
entity, cevu; BASELINE solves no complete pair. RESERVED's other fresh successes
are raku audit and vumi audit; its other retained successes are gemi transport
and itva transport. BASELINE solves raku transport, tavi transport and vumi audit,
and retains falu audit and horo transport. Both updaters act wrongly on both
unsupported requests. These tiny generated development cases do not establish
independent language generalization, and teacher repairs are not learner recovery.

All 52 updating affine selections and executed outcomes match their raw-score
references. The saturated legacy readouts stop throughout. The predeclared
position-collapsed ablation gives BASELINE 2/8 fresh and 5/16 retained, RESERVED
2/8 and 4/16; disabling the new binding block gives 0/8 and 2/16 or 3/16.
These remain diagnostic results, not retrospective architecture selections.

## What changed and what was learned

Both arms use fixed rate-one updates, BINDING gain 16, the existing learned
motor, identical sixteen teaching requests and 48 encounters, and six fresh
documents fixed before outputs. Only feedback allocation changes. BASELINE's
trained weight array exactly reproduces the previous FIXED array.

| Teaching measurement | BASELINE | RESERVED |
| --- | ---: | ---: |
| New pairs + certified older replays | 32 + 118 | 42 + 108 |
| Pair offers / unused budget | 150 / 0 | 150 / 0 |
| Native updates: new + replay | 32 + 48 | 42 + 39 |
| First-attempt successes / 48 | 6 | 6 |
| Fallback attempts / successes | 4 / 0 | 4 / 0 |
| Teacher executions / successes | 48 / 48 | 48 / 48 |
| Admitted neuronal / raw / external negatives | 28 / 3 / 1 | 38 / 3 / 1 |
| Executed failures excluded by admission budget | 10 | 0 |

RESERVED admits every executed training failure. Its ten corrections after step
36 occur at steps 37 and 39–47. Rehearsal also runs after correct encounters at
12, 17, 24, 28, 38 and 48. Every allocation leaves one possible new correction
for each remaining encounter; cyclic replay excludes the newly admitted pair.
Each updater and its frozen mate receive identical certified feedback: 150 pair
offers, recorded as 300 feedback events per learner, with zero frozen updates.
The two updating arms have different on-policy ledgers and rehearsal timing;
the improvement cannot be attributed to correction count alone.

## Remaining interference and coverage

| Final certified pair margins | BASELINE | RESERVED |
| --- | ---: | ---: |
| Negative | 6 | 12 |
| Nonnegative, below unit margin | 0 | 4 |
| Satisfied, tolerance 1e-10 | 26 | 26 |
| Minimum | -15.668 | -14.424 |

All final negative pairs had become positive after their first offer. Eleven of
RESERVED's twelve were positive even after their last offer, as were all six
BASELINE pairs. Later updates demonstrably erase learned contrasts despite the
modest gain in executed retention. The remaining RESERVED pair was already
negative after its last offer; a single fixed update need not restore ordering.

Exact occurrence joins distinguish final wrong leaders from other failed pairs:

| Final wrong leader category | BASELINE, 14 failures | RESERVED, 12 failures |
| --- | ---: | ---: |
| Executed failure, never admitted | 9 | 0 |
| Admitted negative that now outranks its teacher | 2 | 5 |
| Old occurrence never executed as this query's failure | 3 | 5 |
| Occurrence introduced by fresh ingestion | 0 | 2 |

With the same final weights and only the training store, the last recorded
read-only query probes are correct on 2/16 BASELINE and 5/16 RESERVED requests.
Fresh ingestion changes no BASELINE selection. In RESERVED, `guava` at candidate
429 displaces the correct abri transport answer, reducing retained execution to
4/16; a new `yopi` occurrence also replaces an already-wrong enko transport
leader. No weight update occurs during this ingestion or evaluation. Missed
competitors and source-growth interference remain alongside erased corrections.

Repeated same-source query probes record BASELINE new/rehearsal losses of
44/40 and gains of 57/28 over 381 comparisons per stage. RESERVED records 84/22
losses and 72/37 gains over 541/632 comparisons. No loss becomes STOP; one gain
in each BASELINE stage and one RESERVED rehearsal gain starts from STOP. These
overlapping diagnostic observations have different schedules and denominators;
they are not independent tasks or a causal comparison of stage effectiveness.

## Verification, examination and cost

Synthetic QA preserves full baseline teaching traces, verifies reservation
arithmetic and underspend, distinct cyclic replay, replay-only transitions and
authentic feedback custody. Independent source review found no blockers. The
core and native implementation were unchanged, so the previously passing 47
focused tests were not repeated without a new concern.

All 33 declared source files match the committed bytes; encoder, native library,
checkpoint and artifact hashes verify. Independent arithmetic checks all 600
updater/frozen receipts and 74 final pair margins, reconstructing weights within
1.23e-15 and recorded gaps within 5.07e-14. Frozen weights, pure-query weights and
persistent bias remain zero. All 572 transfer/retention executions preserve
decoys, materialize the selected action and have zero live documents. Same-parent
feedback, frozen exposure and read-only evaluation checks pass. No run was retried.

The unchanged official KS2 passage is ingested once per learner. BASELINE emits
`tenrec that live` and `tenrec`; RESERVED emits `vibrations` twice. Each updater
has zero confirmed marks and two items pending official-rubric human scoring;
frozen answers STOP for 0/2. No exam feedback or question/rubric tuning occurred.

The complete comparison takes **308.06 seconds**, including diagnostics and
child waits. Parent peak working set is **338.55 MiB**, final private memory
593.89 MiB and traced Python peak 118.56 MiB; child memory is excluded. Teaching
takes 100.44/101.38 seconds, transfer plus retention 40.08/39.63, and the exam
11.45/11.24 for BASELINE/RESERVED. Shared motor setup takes 0.22 seconds and
encoder loading 0.37. These measurements do not isolate a policy speed effect.

Each learner ends with eleven externally retained documents, 4,708 raw bytes,
2,298 candidates and 23 windows. Counted retained array/byte payload is 7,256,389
bytes, excluding Python object overhead. BASELINE/RESERVED external ledgers use
2,238,122/2,939,039 compact serialized bytes, including overlapping eligibility
payloads of 829,952/1,089,312 bytes; do not add those measurements. The frozen
ONNX encoder is separately retained at 90,405,214 bytes. Controls share the
learned motor and isolate evidence selection, not the whole neuronal system.

Artifact: 10,517,556 bytes, SHA-256
`b834382e06ffeaf9c9d5d13fe0a3e1db123a99d57eb94c4018826bb202619a12`.
NPZ: 27,862 bytes, SHA-256
`b645b543252ad16d5a2d0a9deb0c816acb3ff3704c06feb1d973c89b1eda7ab9`.

## Next decision

Keep fixed updates, affine actions and the RESERVED allocation for development.
Use one bounded primary-source/native-code review of interference control before
choosing at most one further integrated comparison. Preserve representation,
authentic feedback and the 150-offer budget; account for any added storage and
compute. Separately report erased admitted contrasts, unexecuted competitors and
retention before/after fresh ingestion. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md) owns
that decision. Do not deploy the host feasibility witness, tune on this spent
fresh cohort, select ablations retrospectively or reopen protected gates.
