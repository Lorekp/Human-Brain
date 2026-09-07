# V6 partial saved learning review

Status: failed complete-run attempt; bounded post-stop analysis completed,
2026-09-07. This is **not a completed V6 result or a passed full saved audit**.
The frozen [protocol](persistent-language-action-v6-protocol.md) remains unchanged.

## Evidence and limits

The released run at `46839d7fcea50dc6e805db910a36616b0bd51589` exhausted its
840-second wall allowance: 840.4533474 seconds, sampled peak 890.42578125 MiB,
exit 1. There is no report, completion record, final checkpoint or failure JSON.
Consequently no continuing final model is authenticated by this review, and the
completed-run auditor is not applicable.

Root authenticated the stopped artifacts in
`.pytest-tmp/persistent-language-action-v6-partial-custody.json`
(`eca11dc219eacfd3bbabca40f320da227ec2b370af8e3dfd827ec7bf4cbcc972`).
The manifest SHA is
`9ac45b7754f9ac033cc85aba49d66bee1b471e9a6da7ee26ecd883c25d65051f`.
One authorized stdlib-only extraction then passed its 60-second/2048-MiB
watchdog, taking 3.2539705 seconds and 96.7109375 MiB; its internal analysis
took 2.8882998 seconds. No extractor failure preceded this launch.

The extraction is
`.pytest-tmp/persistent-language-action-v6-partial-learning-extraction.json`, SHA
`f190058222b8f14641bccdab8438599bb120a74212c7a942513413dcb2915854`.
Its source SHA is
`1d726db724f305eeef4d55cefacfdae9042417d556b0247c254f4b8d55dd898f`;
the monitor is
`.pytest-tmp/persistent-language-action-v6-partial-learning-analysis-v1.launch.json`.
It checked the failed launch, custody and source/manifest hashes, stable journal
bytes, complete newline-terminated rows, recorded chronology and the scalar
recognition equations below. It imported no project module and performed no
model, feature, predictor, teacher, executor or checkpoint replay. This review's
task grades are taken from durable records; its scalar checks do not replace the
independent action/file audit or a full model-state audit.

There are 182 durable attempts, 18 correction rows and 36 ingestion rows, with no
incomplete journal tails. All three routes completed the first 60 episodes.
Only online's first two attempts on recovery episode C00-E follow them. The next
expected row is online C00-E attempt 3. Work interrupted before another durable
row is **unknown**, rather than zero. No recovery episode has a complete result.

## Outcomes actually reached

The table covers the matched first 60 episodes per route. These are durable
prefix outcomes of a failed run, not complete-protocol scores.

| Encounter or phase | Online | Correction-disabled | Simple reference |
|---|---:|---:|---:|
| Corrected first response | 0/6 | 0/6 | 1/6 |
| Changed-form encounter | 0/6 | 0/6 | 1/6 |
| Exact delayed repeat | 1/6 | 0/6 | 4/6 |
| Fresh transfer | 0/24 | 0/24 | 0/24 |
| Older regressions | 0/18 | 1/18 | 11/18 |

Changed-form encounters also change requirements, so they are not a clean
paraphrase-only test. The six exact repeats have 47 intervening attempts.
Online's sole success is N06-R, the delayed `hold/express` answer. It establishes
one exposed delayed retrieval, not generalized instruction understanding.
Correction-disabled's sole success is older reading R16-R, answering `idle`.
The online/control older-task contrast is therefore 0 versus 1 under their
shared V5-online starting state; it is not a complete historical-forgetting
measurement. No online or disabled fresh reading or fresh edit succeeds.

Both recorded online C00-E recovery attempts fail. There is no terminal attempt
for that episode and no recovery record from the other routes. A recovery score
of 0/12 would incorrectly count unreached work as measured failure.

## What the local learning actually did

Each neural route received six correction occasions: 129 raw target bytes plus
six EOS offers, and nine action-head labels. Online recorded 135 sequential
recruits and local recognition/output updates, published in six transactions.
Disabled computed the same number of offers without advancing its private bank
or gains. Both routes ingested all twelve source documents: 884 raw bytes plus
twelve EOS associations, taking factual memory from 864 to 1760 occupied slots.
These ordinary association writes were enabled in both routes. They supplied
**zero source next-byte teacher symbols**.

The source preserves the old reader and shared router weights. For an event j,
the new current is `s_j(x) = prior_j(x) + g_j * kappa_j(x)`. The independently
prepared input-defined event k receives
`delta_g_k = (1 - s_k(x)) / kappa_k(x)` before the current target byte enters its
outgoing update. At its canonical self-context, the recorded kappa is 0.5.
The extractor checked every recorded teacher's ownership, causal earlier-target
prefix, gain chronology, before/after current, error, division, staged/disabled
flag and selected publication counters. It also recomputed every saved free
owner current from the saved prior, kappa and causally reconstructed gains, and
checked the lower-index maximum. This verifies recorded scalar arithmetic; it
does not reconstruct omitted feature vectors or deserialize the live model.

All 135 online gains are positive and nonzero; their maximum is 4.1041560643.
Each new prepared current reaches one within the scalar tolerance. The last
durably observed online bank has 533 owners, versus disabled's unchanged 398.
Every online routed owner at a teacher step differs from the separately
prepared/taught owner. Unlike V5's shared outcome update, however, V6 explicitly
gives that prepared owner its own recognition association. The absence of
positive credit to newly taught cells is therefore no longer an explanation
for this run's failure. Recognition of an input occurrence still does not
identify its usefulness for another instruction.

## Generation and entry selection

| Recorded generation | Online, 62 responses | Disabled, 60 responses |
|---|---:|---:|
| Byte/EOS decisions | 1832 | 540 |
| EOS responses | 54 | 60 |
| Length-limit responses | 8 | 0 |
| Distinct output strings | 22 | 12 |
| Decisions selecting a new owner | 1827 | 0 |
| Winner differs from saved prior-only ranking | 900 | 0 |

Online has 1574 decisions on the matched first 60 responses; its two extra
recovery attempts account for the remaining 258. Thus the increased generation
length is not solely an unequal-recovery-exposure artifact. The prior-ranking
comparison uses the same actual recorded context at each decision. It is not a
counterfactual generation with recognition switched off. New owners can also
win the frozen-prior ranking, so not all 1827 new-owner selections are separately
attributable to the gain term.

The decisive entry observation is that owner **398**, the first new reading
entry, starts **60/62** online responses. Owner 358 starts one response and owner
452 starts one. The literal `hold/express` appears in 35 online responses:
28 reading answers and seven edit payloads. Its thirteen taught byte/EOS events
occupy owners 398–410. Later reading and edit corrections recruit other owners,
but only 23 distinct owners appear anywhere in online's recorded free paths.
Numerous wrong outputs preserve this reading prefix and then vary or loop;
eight reach the output cap. Disabled instead emits `idle` in 46/60 responses.

This is a change in which irrelevant learned answer dominates, with one correct
repeat. EOS and sequence reproduction are not sufficient evidence of relevance:
the exact same stopped answer is usually wrong, including when emitted as code.
Conversely, malformed suffixes and limits show that per-position selection can
also lose sequence continuity. A latch on the dominant entry alone would retain
the wrong answer more reliably; it would not solve the observed entry failure.

## Supported diagnosis and one next design direction

Private gains remove one specific interference path: a gain update does not
overwrite another owner's parameters. They do not prevent the updated owner
from displacing others in competition. The recorded first-owner concentration
and same-context ranking changes demonstrate that distinction. Recruitment,
new output vectors, recognition gains, and action learning coexist in the
online route, so the records do not isolate one of them as the complete causal
explanation for task accuracy or elapsed time.

The source identifies a narrower representational limitation. An owner's gain
only scales its fixed self-similarity direction. It cannot learn which ordered
instruction relation matters, which shared word is incidental, or how different
requirements select different generated structure. Its self-context metadata
difference is zero, so that new trace learns no positive flag interpretation.
Each teacher-prefix event is independently recruited; there is no learned
sequence-level relational state. Actual wrong generated prefixes then enter
subsequent retrieval, whereas correction uses earlier supplied target bytes.
Those source facts make similarity attraction and prefix drift plausible
mechanisms. The saved scalars do not identify a particular word, feature block
or missing hidden state as their measured cause.

The recommended next **design direction**, not a selected implementation or
experiment, is a causal, locally predictive **ordered-context population** shared
by instruction reading, observed source, and output generation. Its purpose is
to learn reusable contextual relations before calibrating another prototype
score. Retain the existing model, memories and actor as the prior. Let sparse
causal state `a_t` depend only on raw field/token input and prior state; give an
output cell its own current `v_c = w_c · a_t`. When the next naturally observed
symbol arrives, a bounded direct local rule
`delta_w_c = eta * (T_c - v_c) * a_t / (epsilon + ||a_t||²)`
can teach that cell from its own current and presynaptic activity. No error is
transported backward into the state transition, and no event/episode identity or
correct memory address labels the input. Answer corrections use the same causal
path; free generation uses its own emitted prefix and actual tool feedback.

The unresolved design requirement is substantive: the causal state must preserve
useful order and distinguish instruction contexts without simply creating one
new full-context prototype per symbol or recreating a broadly overlapping dense
readout. Its construction and direct local learning must be specified before
implementation. The equations describe the permissible local teaching seam,
not proof that a chosen state representation will understand language or compose
programs. This is a representation-and-experience question; another scalar
recognition clamp or a transition latch is not yet an adequate answer.

Naturally observed source next-symbol learning is a necessary component of
that proposed direction, not something this run already did. Fewer answer
corrections can coexist with more openly counted raw language/program exposure.
The 884 appended bytes here only wrote associations; six answers containing
135 byte/EOS offers are not evidence that unseen syntax was learned. Any future
source teaching must count every causal symbol separately, preserve a matched
learning-disabled route, and use the same broader transfer, retention and actual
action setting. No hidden corpus, replay, additional fit on V6, separate executor
learner, or prerequisite toy-perfect gate follows from this review.
