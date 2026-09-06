# Persistent language/action: first broader session

Status: completed saved-evidence review. The fixed public session is closed;
its examples are now exposed development material. This review performs no
model construction, checkpoint loading, prediction, alternative decoding,
teacher call, task execution or audit replay.

The persistent machinery completed its workload, but the online learner did
not demonstrate useful exact-answer or executable-task performance. Both neural
routes scored zero in every phase. This broader failure is not the earlier
byte task's predominantly false-STOP problem.

| Phase, first attempts | Online | Learning disabled | Simple reference |
| --- | ---: | ---: | ---: |
| Acquisition reading | 0/12 | 0/12 | 12/12 |
| Acquisition edits | 0/12 | 0/12 | 0/12 |
| Fresh reading | 0/8 | 0/8 | 7/8 |
| Fresh edits | 0/8 | 0/8 | 2/8 |
| Recovery edits | 0/8 | 0/8 | 0/8 |
| Delayed reading | 0/4 | 0/4 | 4/4 |
| Delayed edits | 0/4 | 0/4 | 4/4 |

The reference scored 6/12 on supported fresh items, versus 0/12 for each neural
route; its other three fresh successes were on the four deliberately long
instructions. Its lexical retrieval accepts these unchanged instructions,
whereas the neural interface explicitly rejects two reading and two edit
requests over 31 bytes. This asymmetry is disclosed, not silently excluded from
overall totals. Public wording families overlap; this is not protected transfer.

## What the actual outputs show

Across 52 supported first attempts, the online reader/payload generator reached
EOS 40 times and its byte limit 12 times. The full 68-query sequence, including
retries, made 703 byte decisions: 50 EOS endings and 18 limits. The disabled
route made 717 decisions, with 49 EOS endings and 19 limits.

The failures include both selecting the wrong portion of prose and failing to
end an answer at its appropriate boundary. For `A00-R`, “What colour is the
badge?”, the actual output was ` badge is 'blue'`, rather than `blue`. For
`A06-R`, `A08-R` and delayed `T08-R`, the selected address was an exact occurrence
of the requested literal's start, yet the outputs were `0' writes`, `west'`
and `west'`. These are actual saved generations, not hypothetical generations
from substituted addresses. A zero exact score therefore does not establish
that no relevant fact was retrieved; it does establish failure under the frozen
answer contract. No substring-based rescoring is applied here.

External byte annotation finds a nonempty target literal somewhere in available
memory for 49 of the 52 supported first queries. The online winner lies in a
document containing some such occurrence in 27/49, but selects an occurrence's
exact start in only the three cases above. This annotation uses literal presence,
not semantic relevance: repeated digits and values can occur in unrelated notes.
The three remaining cases include an empty deletion, a value absent from notes,
and the composed answer `green; blue`. Literal recall alone does not establish
the ability to compose that answer.

All 30 supported first edit proposals have a replacement different from the
canonical teacher. Only two starts and one end match the canonical coordinates;
20 paths match. Nevertheless, 29/30 operations are SPLICE. Canonical fields are
diagnostics only: the independent grader accepts alternate edits that produce
the required behavior while preserving protected files. Thus an operation-only
change would leave substantial reading, payload and location failures.

Among all 48 online executable attempts, 13 SPLICE proposals are rejected for
missing payload EOS and two long-request attempts verify the unchanged workspace
without a proposal. Forty executions return nonzero; protected-file integrity
fails in 13. These categories overlap. The disabled route has 12 missing-EOS
rejections, 38 nonzero executions and 16 integrity failures. The simple reference
has no proposal rejections or protected-file violations, but 24 nonzero executions.

## Learning, memory and recovery

There are 216 response/action records, 144 actual child executions, 72 correction
records and 39 append records. Each neural route receives the same 24 one-pass
corrections: 104 reader symbols, including 24 EOS offers, and 36 action labels.
Online publishes codec and address changes on all 24 corrections and each action
head on all 12 edit corrections; the disabled route publishes none. Initial
model hashes match, the disabled model hash remains unchanged, and both final
tape hashes match. These are same-instance local updates after durable actual
responses/actions; no backpropagation or teacher-free hidden replay was added.

Learning is therefore active, but a publication is not acquired competence.
All 22 supported first reading outputs are identical between neural routes.
Only one of their 52 first reader/payload outputs differs (`A09-E`); full first
proposals differ in ten episodes. None yields an exact task success. There is
no immediate post-correction probe, so the records cannot distinguish briefly
acquired behavior subsequently lost from behavior never established by one offer.

Both neural routes retain the same append-only 453-slot stream: 437 raw bytes
and 16 EOS slots. Retention repeats occur after 31–47 intervening episode starts,
with initial memory sizes of 22, 100, 229 or 309 slots. All eight neural repeats
failed initially and after delay. Calling this eight forgotten skills would be
incorrect. The reference preserves four initially correct readings and improves
four previously failed edits through its explicitly stored demonstrations.

Every recovery episode uses three actual attempts. All three routes recover
0/8 first failures, and none ends correctly. Each neural episode repeats exactly
the same payload across its three attempts. The model and tape receive no public
tool-feedback encoding or consequence update; only resulting files can change
the next action-head observation. This measures the present recovery limitation,
not a failed trial of a feedback-learning mechanism.

## One next shared priority

Review a **request-conditioned answer construction and termination interface**
for ordinary prose, shared by language responses and action payloads. Keep the
broader reading/action/retention/recovery stream as the integration target,
with bounded corrections, matched disabled and simple references, and new
preregistered material. Do not turn the three diagnostic examples into a new
toy perfection gate or increase repetitions until this public set fits.

The source reason is concrete: the current reader selects a starting address,
then traverses document adjacency. Its codec sees the current raw sensory byte,
not the request, workspace or desired answer boundary. All-address corrective
likelihood can update shared address/codec synapses locally, but an arbitrary
answer-ending EOS is not the same task as reaching a document's EOS. The same
raw byte can legitimately continue one answer and end another. A generic
request-conditioned boundary/content mechanism therefore deserves source/math
review before assuming more local offers can resolve the mismatch.

This is a review priority, not selection of an implementation or another fit.
It must address how any new output population receives a direct local teaching
signal without a supplied target address, a host answer parser, hidden correct
span or reverse-Jacobian. The saved data do not prove that boundary control alone
will solve address discrimination, composition, edit localization or recovery.

## Evidence

The [independent saved review](../../.pytest-tmp/persistent-language-action-v1-review.json)
passed; its SHA256 is
`366a355aa2742d2390ada33d72e01cd3fce3c5472f832389a52a0661e6f25652`.
The [report](../../.pytest-tmp/persistent-language-action-v1-run/report.json) is
`b71126d1c2150c5b550e87f1aac3446dd04838d9a435df7baec39685d964fda1`;
the frozen manifest is
`8ff7e338fb7649d8b230fc54507a1265a406cadf3f183c9c6585cbfa47b62ef8`.
The [saved scalar/byte extraction](../../.pytest-tmp/persistent-language-action-learning-review-v1.json),
SHA256 `ca5c66f5d06c9355d69cafd015d80b972aa8a78fd84a133fc90362e92c6ba5fa`,
binds the complete attempt, teacher and ingest journal hashes and records all
first-attempt annotations. It does not reconstruct neuronal forward arithmetic.
