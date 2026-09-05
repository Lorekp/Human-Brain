# Acquisition before another representation change

Research review, 2026-09-05. Documentation only; no learner, native, fixture or
model execution changes. The recommendation below needs its own frozen protocol.
Keep COMPACT/RESERVED; the curriculum comparison did not meet its adoption rule.

## What the evidence identifies

The [curriculum artifact](../../benchmarks/contextual_curriculum_v1.json), source
`681928bce9d8ea7a48f4c616ccb5f22b6edc7002`, gives both arms 2/8 fresh and 4/16
common retained executions, with 0/2 correct unsupported STOPs. COMPACT/VARIED
familiar scores are 1/4 versus 0/4; broader scores are 1/4 versus 2/4. VARIED
therefore fails both strict fresh improvement and familiar nonregression.
Before fresh ingestion, actual taught-query raw selections improve from 5/16 to
8/16, but common original queries are 5/16 in both arms, falling to 4/16 after
ingestion. Switching VARIED's query forms at the same checkpoint loses five
correct selections and gains two. These raw probes are not additional executed
successes. Query-form sensitivity therefore accompanies weight interference.

Both arms end with twelve negative actual-TRAIN pair margins, each previously
at least one. Eleven COMPACT and all twelve VARIED negatives were positive after
their last offer. Maximum last-offer-to-final drops are 33.89 and 61.16. This is
direct interference evidence, not an inference from low task scores. At the
final actual-query probe, COMPACT has five admitted failed occurrences winning
again and six wrong leaders never executed for that query; VARIED has four of
each. Coverage and erasure coexist. The [result report](../../experiments/r4_integrated/contextual_curriculum_result.md)
records the independent margin and exact-query occurrence audits against the
saved artifact.

Earlier interventions narrow the interpretation:

- [Reservation](../../experiments/r4_integrated/contextual_reserved_feedback_result.md)
  removes ten excluded real corrections and modestly improves action, but still
  loses learned contrasts and misses current competitors. Admission is necessary
  for those errors; admitting every encountered failure is not sufficient.
- [Capping](../../experiments/r4_integrated/contextual_capped_update_result.md)
  reduces the worst negative margin but worsens retained/fresh execution.
  [Protection](../../experiments/r4_integrated/contextual_protected_contrast_result.md)
  preserves nineteen certified contrasts and improves retention, yet fails its
  fresh-nonregression rule; [robustness](../../experiments/r4_integrated/contextual_robustness_result.md)
  confirms that adoption failure. Neither result refutes all stability methods.
- The [host feasibility witness](../../experiments/r4_integrated/contextual_representation_feasibility_result.md)
  satisfies all 4,560 compact TRAIN ranking/STOP constraints. Its unrestricted
  weights have norm 604.51. This excludes finite-roster linear impossibility,
  not native reachability, budgeted convergence or fresh representability.

## What neuronal learning currently contributes

[Source features](../../src/humanbrain/contextual_evidence_memory.py) combine
frozen contextual span/query vectors, fixed products and fixed relative-position
interactions. The [encoder](../../src/humanbrain/frozen_context_encoder.py) produces
contextual token vectors separately for query and source; feedback does not train
it, candidate boundaries or the positional kernels. A shared learned GO-minus-NO
weight vector ranks those fixed features, then affine GLIF selects an occurrence.
There is no learned parser or explicit entity/property variable store here.

For a certified contrast d, the fixed [native update](../../src/humanbrain/_native_src/src/lib.rs)
changes the effective weight difference by 2d/sqrt(1+d·d) when its margin gate
fires. Another contrast e consequently changes by 2(e·d)/sqrt(1+d·d).
Opposing feature directions can therefore undo an earlier correction. The
[caller](../../src/humanbrain/contextual_evidence_memory.py) discards pair-bias
updates; pure-query coordinates cancel in same-query contrasts. Neural weights
learn evidence preferences, while externally stored receipts supply credit and
replay custody. Frozen/external controls share the learned motor: their failure
isolates evidence learning, not the contribution of the entire neuronal system.

## Three concrete directions

| Approach | Evidence for and limit | Observation that changes the decision |
| --- | --- | --- |
| More closed-loop practice | Re-executing old requests can expose unseen leaders. However, [revisitation](../../experiments/r4_integrated/contextual_revisit_protocol.md) already introduced three passes; RESERVED already corrects each encountered failure. Extra reviews change teacher/action dose even with the same offer cap. | Favor only if stable acquired contrasts coexist mainly with never-confronted leaders. A fair comparison would match extra review and teacher opportunities in both arms. The current erasure evidence does not justify another practice increase first. |
| Covariance-aware synaptic updates | Learn a metric from authentic contrast directions so correlated features need not receive the same geometric treatment each time. The failed cap changes scalar size; full protection freezes certified directions. Neither tests this soft, history-dependent update geometry. It adds global learning machinery and can still forget. | With unchanged inputs and encounter budgets, fewer erased pairs together with better retained execution would support a learning-geometry bottleneck. Margin improvement without useful action would not justify adoption. Recommended once below. |
| A shared learned bilinear query–span map | Replace the diagonal product score with qᵀWspan, allowing off-diagonal interactions learned from the same executed contrasts. No entity table, semantic span filter or authored relation rule is needed. Query-form sensitivity motivates it, but the TRAIN witness and existing erasure oppose assuming missing capacity is the sole cause. | Strong acquisition with stable current-competitor rankings but failed untouched forms would make this more discriminating. It increases parameters and eligibility/ledger costs, does not contextualize query and source jointly, and offers no guaranteed role binding. Do not add it alongside the next learning-rule trial. |

## Recommended next integrated experiment

**Failure/hypothesis:** native fixed-direction updates repeatedly erase authentic
TRAIN comparisons. One covariance-aware margin learner may retain useful evidence
preferences better with the same representation and feedback opportunities.
This tests a learning rule, not a claim that representation is already adequate.

A concrete candidate is AROW, [Crammer et al., 2009, equations 7/9 and Figure 1](https://proceedings.neurips.cc/paper/2009/file/8ebda540cbcc4d7336496819a46a1b68-Paper.pdf#page=4).
For the effective weight difference v, initialize covariance S=I. Given original
certified delta d and m=v·d<1, use p=S d, b=1/(d·p+r),
v←v+(1−m)b p and S←S−b ppᵀ. Otherwise leave both unchanged.
The paper's classifier becomes our pairwise learner by setting input d and label
+1; this mapping is a proposed adaptation. Freeze r=1 as an untuned convention.
Split the change symmetrically across GO/NO weights; keep persistent bias zero.
This changes direction and size together, approaches rather than guarantees unit
margin, and does not guarantee preservation or transfer. Do not claim a pure
covariance causal effect or biological discovery. Count the full covariance,
scratch, checkpoint and compute costs; a 1621² float64 matrix alone is 20.05 MiB.
Dense matrix-vector and outer-product work are O(D²) per applied offer, each
touching 2,627,641 entries; 150 offers permit 394 million entries per operation.
These are size/work counts, not measured runtime or a scaling claim.

Compare fixed RESERVED with this one candidate: compact sixteen facts, original
48 encounters, exactly 48 teacher executions per arm, maximum 150 offers,
unchanged cyclic allocation, authentic FAIL/PASS custody and frozen mates.
No extra practice, teachers, covariance prefit, solver weights or fresh-label
feedback. Inference uses only learned weights and the unchanged affine GLIF;
covariance changes during authorized learning only. Both controls share the
motor. The candidate's frozen mate starts with the same fixed S=I and keeps it
unchanged: count that allocation, but distinguish it from learned covariance.
The existing fixed-rule arm has no covariance. On-policy negatives, updates and
compute can differ despite equal limits; do not claim identical training exposure.

Freeze one new common familiar/broader fresh roster and common old probes.
Adopt only with strictly better executed retention and no decrease in fresh
total, either fresh stratum or correct unsupported STOPs; retain absolute gates.
Audit original-delta margins, current-leader coverage, own-form/common-form
selections and complete entities. Reduced erasure with improved acquisition but
persisting form loss favors a representation review; fitted pairs with unadmitted
wrong leaders favor coverage. Failure of this configuration proves neither cause.

Limit preparation to two hours and one comparison to thirty minutes total wall
time, with synthetic arithmetic/custody QA before freeze and saved-scalar audits
afterward. Close after this candidate: no r sweep, covariance variants, extra
epochs or automatic mechanism successor. Unsupported abstention remains unsolved;
unchanged KS2 items are regression checks, and fresh official exam items require
a separate manifest. The existing failed adoption decisions remain closed.
