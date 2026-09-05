# TRAIN-only host-linear representation feasibility

Status: preregistered diagnostic, 2026-09-05. One fixed solve, no capability
evaluation. Source, inputs and this protocol are committed before model outputs.

## Question and boundary

The [capped-update result](contextual_capped_update_result.md) left nine FIXED
retained failures despite satisfying all admitted contrasts. Seven of those
wrong leaders were actually executed after the pair budget and received no
learning update. Other failures have erased margins. Before changing features,
test whether the unchanged representation admits a correct joint ranking of the
sixteen taught answers across the four training documents.

This is an external, answer-supervised capacity reference. Any solver witness
stays outside the learner; it is never installed as synapses, executed through
the motor or used on transfer/exam questions. No native learning runs. A verified
witness establishes only finite-TRAIN host-linear representability. It does not
establish native learnability, attainable update trajectories, useful efficiency,
neuronal execution or generalization.

## Inputs and coordinates

`contextual_representation_train.json` contains only the four training documents
in original first-ingestion order and sixteen existing teaching requests/values.
Root extracted this projection from the binding fixture at `d0b0c40`, before
diagnostic model outputs; its provenance records the original file hash. The
diagnostic process reads only the projection, not the original mixed fixture,
prior comparison artifacts, transfer cohorts or exam files.

Construct the unchanged frozen MiniLM encoder and a learning-disabled BINDING
memory with gain 16, fixed comparison policy and rate 1. Ingest only these four
documents through the existing opaque-document interface. Assert 285 complete
candidate occurrences and extract the same 1,621 features for every candidate
under each of the sixteen requests. No candidate filtering, coordinate scaling,
representation ablation or label-conditioned features are allowed.

For each request, require exactly one candidate emitting its existing teacher
value. These are known TRAIN labels, not new feedback. Record every candidate
identity and source span, request and teacher index. Assert feature extraction
does not mutate the ingested checkpoint, synapses or feedback counters.

## Fixed feasibility problem

Let `x_qc` be the unchanged feature row for candidate `c` and query `q`, `a_q`
its teacher index, and `v` a host weight-difference vector. Fix its first 384
pure-query coordinates to zero and keep the persistent bias zero. The remaining
1,237 coordinates have unrestricted signs and magnitude. Let
`tau=atanh((600.1-600)/200)` be the existing raw STOP drive. Require:

```text
v dot (x_q,a_q - x_q,c) >= 1      for each q and every c != a_q
v dot x_q,a_q >= tau + 1         for each q
```

This gives 4,544 answer-ranking inequalities plus sixteen STOP inequalities.
Solve one zero-objective linear program using installed SciPy 1.17.1
`linprog(method="highs")`, free bounds, presolve enabled, a 300-second time
limit, and primal/dual feasibility tolerances `1e-9`. Record actual versions and
solver status/message. These options follow the installed documentation and
the [SciPy HiGHS interface](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
Do not search solver variants, norms, feature policies or thresholds.

Joint infeasibility would concern ranking **and** absolute STOP clearance; it
must not be attributed solely to answer discrimination. Solver status alone
does not verify an infeasibility certificate. Without an independently checked
certificate, report solver-reported infeasibility or numerical/timeout failure
as inconclusive, and return to integration at the budget boundary.

## Witness verification and accounting

Retain the original feature tensor, teacher indices, free-coordinate indices and
any returned witness in a separate NPZ, with hashes for the derived constraints.
Avoid storing redundant copies of reconstructible matrices. Record source and
fixture hashes, encoder files and the loaded checkout-native identity.

Verify finite coordinates, zero fixed coordinates and every inequality against
the original unrounded float64 features, with absolute allowance `1e-7` at unit
margin. Independently recompute each query's actual teacher score, maximum wrong
score, ranking margin and STOP clearance. Require the teacher to be the unique
host raw leader for all sixteen queries. A result is verified feasible only
when all checks pass; a successful solver status alone is insufficient.

Report the smallest ranking/STOP margins, largest residual, weight L2 norm and
maximum magnitude, score ranges, state before/after, solve/feature/verification
runtime, retained bytes and parent peak memory. An unbounded-weight witness may
still cause finite-precision aliases in affine GLIF; do not claim neuronal
execution or resource-bounded capacity without testing those separately.

Native pairwise updates remain positive steps along offered eligibility
differences. An unrestricted host solution need not lie in their attainable
cone or be learnable under the current schedule. The diagnostic does not remove
that distinction merely by fixing pure-query weights and bias.

Use inert synthetic feasible/infeasible examples and invalid witnesses for
preflight QA; do not add project-cohort model runs before the source freeze.
Preserve every terminal outcome without overwriting prior evidence.

## Budget and return

Budget one fixed diagnostic and at most 45 minutes for setup and diagnosis,
starting 02:31 UTC. No native-source, learning-rule or core feature changes.
If a witness verifies, return directly to the integrated feedback-allocation
comparison: test whether later real errors can receive correction within a
declared overall offer budget, while still measuring retention. A concrete
verified representation conflict may justify a narrowly targeted redesign;
inconclusive optimization does not. Do not deploy the witness, run a second
mechanism study automatically or reopen protected evaluations.
