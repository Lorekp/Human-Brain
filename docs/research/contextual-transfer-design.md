# Transfer after improved acquisition

Decision review, 2026-09-05. Documentation only; no new model execution, fixture,
or implementation. Recommend one COMPACT-versus-VARIED teaching comparison under
the adopted covariance learner. Its protocol and new evaluation cohort must be
frozen before execution. This replaces the automatic representation-first branch
of the [acquisition review](contextual-acquisition-review.md), for the reason below.

## What changed, and what did not

The audited [covariance artifact](../../benchmarks/contextual_covariance_v1.json)
at source `ce5a08c53465467e9f396709a34aa38bd6075997` improves executed retention
from 5/16 to 14/16. Fresh execution remains 2/8, with 1/4 in each stratum and
0/2 correct unsupported STOPs. All 36 admitted comparisons end positive, although
33 remain below unit margin. Both remaining old errors select occurrences never
executed for that exact query. This separates missing competitor coverage from
erasure of those admitted pairs; it does not establish absence of all interference.
Both multiword fresh answers fail, alongside entity/relation and span confusion.
Raw and affine selections agree: another score-map adjustment is not indicated.

In the earlier [fixed-rule curriculum comparison](../../experiments/r4_integrated/contextual_curriculum_result.md),
VARIED improved actual taught-form raw selections from COMPACT's 5/16 to 8/16,
but switching VARIED to common questions lost five and gained two. Both arms
ended with twelve negative TRAIN comparisons that had previously attained unit
margin. Fresh scores tied at 2/8; VARIED regressed on the familiar stratum.
That adoption failure remains closed. AROW now changes a demonstrated acquisition
failure, so testing its interaction with that *unchanged* varied curriculum is
a new hypothesis, not a rerun seeking a favorable cohort for the old learner.

The [TRAIN feasibility witness](../../experiments/r4_integrated/contextual_representation_feasibility_result.md)
only establishes that one compact finite roster is linearly separable with other
weights. It says nothing about fresh representability or native learning reachability.
The current retention gain likewise does not establish transferable relation binding.

## What each alternative would change

The [current features](../../src/humanbrain/contextual_evidence_memory.py) contain
384-dimensional query and span vectors, diagonal query–span/query–window products,
an overlap scalar and 84 fixed candidate-position interaction features: D=1621.
The [frozen encoder](../../src/humanbrain/frozen_context_encoder.py) contextualizes
each input separately. Feedback trains shared GO/NO synapses and covariance;
it does not train the encoder, token pooling, candidate boundaries or positional
kernels. Covariance affects learning, while inference uses learned weights alone.
These are real learned evidence preferences feeding a previously learned neuronal
edit motor, with an external pretrained encoder and receipt-certified credit.

| Option | Discriminating value now | Cost and unresolved confound |
| --- | --- | --- |
| Reuse VARIED under AROW | Tests whether the previously partly acquired wording/form variation transfers when comparisons are retained better. Keeps features, learning rule and motor fixed between arms. | Same dimension and teacher limits; longer documents create more candidates and encoding/ledger work. Changes source wording and question form together, so it tests the curriculum package, not either component alone. |
| Full shared bilinear qᵀWspan | Adds off-diagonal query–span associations absent from the diagonal product block; remains generic and linear in W. | Dense AROW becomes impractical at the expanded dimension below. Low-rank or structured covariance introduces another capacity/learning approximation; it is not the same experiment made cheaper. |
| A broader learned contextual or cell representation | Could learn source/query role structure or span composition that fixed pooling and kernels miss. | Needs an explicit trainable interface, credit path, state budget and enough diverse authentic supervision. Present failures do not identify the required architecture; changing it now would mix acquisition and representation changes. |

For a concrete full-bilinear configuration, replace the existing 384-coordinate
query–span product with 384²=147,456 outer-product coordinates; retain every
other block. Then D=148,693 and a dense float64 covariance contains
22,109,608,249 entries: **176,876,865,992 bytes (164.73 GiB)**. An updater plus
its frozen covariance mate needs 329.46 GiB before scratch, export and audit
copies. The current matrix is 21,021,128 bytes (20.05 MiB); quadratic work and
state grow about 8,414-fold. A two-decision eligibility pair alone becomes
2,379,088 bytes. These are arithmetic costs, not measured runtime predictions.
Appending instead of replacing the product would increase D further to 149,077.

A rank-r factorization W=UVᵀ has 768r factor parameters, but learning both factors
makes the score nonlinear in them. Existing linear contrast eligibility and AROW
semantics would require a new derivation. Fixed factor projections instead change
the available representation. Neither construction guarantees relation binding.
Do not pick a rank, change covariance structure or deploy the host witness here.

## One next integrated comparison

**Hypothesis:** the fixed VARIED curriculum contains reusable information that
the former update rule partly acquired and then erased; adopted AROW can preserve
enough of it to improve fresh action without losing common-query retention.
Confidence is sufficient for one bounded test, not a prediction of success:
observed query-form sensitivity and multiword failures may remain unchanged.

Compare COMPACT-AROW with VARIED-AROW using the exact two previously registered
[TRAIN renditions](../../experiments/r4_integrated/contextual_curriculum_protocol.md).
Use S=I, r=1, gain16, no protection, unchanged affine readout, RESERVED allocation,
and identical facts, order, sixteen cases and 48 encounters. Each arm gets exactly
48 teacher executions and at most 150 offers, including replay. Keep one source
rendition per arm and unique teacher occurrence; never ingest COMPACT passages
into VARIED through the common-probe helper. No extra teachers, reviews, epochs,
paraphrases, answer filters or fresh-label learning. On-policy negatives and native
updates may differ; equal limits do not imply equal executed feedback content.

Use matched frozen mates, external controls and the same neuronal motor. Count
both initial covariance allocation and learned covariance mutation, all executed
attempts and teachers, encoding, candidates, original eligibilities, receipts,
checkpoint payload and time. D remains 1621, but total storage/runtime need not
match because the unchanged VARIED documents are longer.

Freeze one new shared eight-supported/two-unsupported fresh cohort, with four
queries per familiar/broader stratum, before model outputs. No spent-cohort
acceptance or choice among alternative fixtures. Keep the common sixteen old
queries as the executed retention comparison. Separately record read-only actual
taught-form and common-form probes at the same checkpoint, and after fresh
ingestion; these are diagnostics, not additional executed successes. Audit final
original-pair margins and whether current wrong occurrences were actually executed
and admitted for that exact query. Case-name matching cannot establish coverage.

Adopt VARIED only for **strictly higher fresh executed success**, with no decrease
in either fresh stratum, common executed retention or correct unsupported STOPs
against concurrent COMPACT-AROW. Report paired gains/losses, complete entities and
keep the separate absolute gates: 6/8 fresh, 12/16 retained and three complete
fresh entities.
A two-item unsupported tie at zero does not solve abstention. Shared teaching and
one tiny authored cohort support a local decision, not independent replication or
general reading ability. This experiment does not isolate covariance alone.

Limit preparation to two hours and the one comparison to thirty minutes total
model wall time. Freeze configuration and semantic/custody checks first; audit
saved results afterward. Close this curriculum-under-AROW question after the run:
no wording, value-length, epoch or regularization search if it fails. The failure
exit is a bounded learned-binding architecture review, with stable taught-form
acquisition but failed common/fresh forms its strongest evidence; renewed erased
pairs would limit that inference. Keep the two unencountered old leaders separate:
do not add practice to solve them while fresh transfer remains weak. Coverage
does not prove extra teaching would transfer. This review does not replace the
next experiment's frozen protocol.

The unchanged official KS2 items remain a regression check pending human marking.
Fresh authentic reading pressure needs the separate official-source, rights and
partition manifest in [the public-exam plan](reading-comprehension-evaluation.md);
this curriculum experiment neither replaces it nor manufactures exam questions.
