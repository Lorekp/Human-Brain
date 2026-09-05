# Learned binding after the covariance curriculum result

Decision review, 2026-09-05. The orchestrator selects the bounded projected-bilinear
comparison below. No implementation, model calls or new fixtures belong to this
review; the successor protocol must be frozen before execution.

**Closed:** the [frozen comparison](../../experiments/r4_integrated/contextual_projected_binding_result.md)
rejects PROJECTED (fresh 4/8 to 1/8; retention 8/16 to 4/16). Neither arm fully
acquires taught forms. The recommendation below records the original decision,
not a repeat authorization; the roadmap owns the next interface review.

## Failure identified, without claiming impossibility

In the [completed comparison](../../benchmarks/contextual_covariance_curriculum_v1.json),
COMPACT/VARIED execute 1/8 versus 0/8 fresh and 11/16 versus 5/16 common old
requests. Actual taught-form raw selections are 14/16 versus 13/16. At one
VARIED checkpoint, changing only questions loses nine correct selections and
gains one. Both arms retain positive margins on every admitted TRAIN contrast.
VARIED selects the correct source for all eight supported fresh requests but
misses their relation/span; its lost shared success is an overspan. These facts
favor investigating within-source binding and query-form sensitivity, not another
readout correction or treating all remaining failures as erased learning.

All existing teacher golds are single words. Whole multiword candidates are
reachable, but failure to select them does not prove inability to learn boundaries
without positive demonstrations. Unencountered exact-query competitors also remain.
The former [TRAIN feasibility witness](../../experiments/r4_integrated/contextual_representation_feasibility_result.md)
does not certify fresh inputs. Sixteen reused facts support neither an adequate
language curriculum nor a general representation impossibility conclusion.

## Is full structured bilinear learning feasible?

Yes in memory. [Crammer and Chechik, 2012, §4.2/Figure 2](https://arxiv.org/pdf/1206.4639#page=5)
maintain unrestricted mean W and covariance Ω⊗Λ, with separate row/column
factors. This restricts covariance, not W's rank. For query q and difference
d=s+−s−, let a=qᵀΛq, b=dᵀΩd. The mean update is
W←W+max(0,1−qᵀWd)/(r+ab) · Λq dᵀΩ. Factor updates use dimension-dependent
denominators mr+ab and nr+ab; they are not the existing dense AROW recurrence
stored compactly. The paper's cost is O(mn+m²+n²), not O(m²n²).

At m=n=384, W, Λ and Ω require 3,538,944 float64 bytes (3.375 MiB);
splitting W into GO/NO matrices gives 4.5 MiB per learner before scratch/copies.
This is a feasible structured alternative to the previous review's *full dense*
covariance, whose arithmetic estimate remains correct.
However, adding this head to today's 1,621 features needs an explicit joint or
block covariance/update contract. It loses cross-block correlations unless they
are modeled. Replacing the whole head discards existing useful features instead.
Neither is a drop-in unchanged-AROW experiment or established linguistic binding.

## Three options and their learning contracts

| Candidate | What it can test | Principal limit and custody change |
| --- | --- | --- |
| Fixed projected bilinear block plus existing AROW | With fixed Pq/Ps, learn B in (Pq q)ᵀB(Ps s). It permits off-diagonal associations in a fixed subspace while preserving the audited linear learning rule. | Projection can discard relevant directions. Pooled q/span inputs still lack an explicitly learned boundary/role representation. Added capacity or changed conditioning can explain gains. Projection identity and each original expanded eligibility must be authenticated; no gradient-through-encoder claim. |
| Full bilinear head with factored AROMA | Tests full cross-coordinate associations without random compression and with feasible structured second-order state. | Changes covariance geometry together with representation. A joint head requires a derived update, numerical audit and newly bound factor state; old AROW receipt equations cannot certify it. |
| Learned factors or a plastic recurrent representation | Learns the feature producer itself, potentially making query roles, token order and span composition useful across requests. | W=UVᵀ makes scores nonlinear in U/V; old stored linear deltas cease to be current gradients. Requires producer-state identity, original input custody, explicit gradient/credit semantics and matched frozen producer controls. More parameters alone do not create role structure. |

[Fast weights](https://arxiv.org/abs/1610.06258) explicitly target temporary recent
memories; that is not yet evidence for resolving this system's within-source
relation/boundary errors. [Differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a.html)
learns plasticity parameters with backpropagation in recurrent networks. It offers
a concrete route to learned adaptation, but introduces outer training and its
data/compute budget. Neither paper licenses calling sixteen-fact receipt replay
an adequate meta-training distribution. A cell implementation must disclose which
representations and plasticity parameters actually learn, rather than relabel a
fixed feature expansion as learned parsing.

## Recommended smallest discriminating step

Prepare **one projected-bilinear versus current-AROW integrated comparison**,
not a projection sweep. Its advantage is preserving the demonstrated learning
rule while asking whether new cross-coordinate associations improve actual action;
cheapness alone would not justify it. Treat it as a limited representation test,
not the final learned-role architecture. Use one fixed budget of 32 projected
coordinates per side, adding 1,024 products to the unchanged features: D=2,645.
One dense covariance is 55,968,200 bytes (53.38 MiB), 2.66 times current size;
two 32×384 float64 projections add 196,608 bytes. Count frozen mates, expanded
eligibilities, original ledger and transient matrices separately. These are
arithmetic estimates, not observed runtime. Freeze projection construction/seed,
normalization and gain before outputs; do not search them after failure.

Give both learners the **same prospectively declared generic binding TRAIN set**
with single- and multiword positives, varied question roles and opportunities for
authentic wrong-span failures: 48 encounters, exactly 48 teachers and at most
150 offers per arm. Admit negatives only from actual failed actions. Do not selectively
repair the spent cases or author entity/property/length filters. This is a new
supervision scope, disclosed in both arms, not reopening the closed wording sweep.
Comparing both architectures under that common supervision distinguishes a
representation benefit from simply supplying missing demonstrations. No new
teacher or extra epoch may be hidden behind the old 150-offer limit.

Freeze one new common evaluation roster and scoring before training. Require
strict fresh executed improvement with no common-retention, form-stratum or
unsupported regression, and retain the separate absolute gates. Report exact
whole-span success conditional on correct source, common-versus-taught forms,
multiword outcomes, admitted margins and uncovered competitors. Read-only block
disable/diagonal-only ablations can show dependence, not exclude a capacity effect.
If neither model acquires its taught forms, transfer does not identify a binding
limit. If both improve equally, better supervision suffices for that improvement.
If only the new block transfers, it earns a local adoption claim, not learned syntax.

Bound preparation to two hours and one comparison to thirty minutes, subject to
preflight cost checks. Failure closes this candidate: no rank/projection/value
search or automatic AROMA successor. A broader trainable token/role interface then
needs its own concrete credit/state design. Keep authentic official reading as a
separately sourced/frozen [companion track](reading-comprehension-evaluation.md);
repeating pending KS2 items cannot supply independent reading generalization.
