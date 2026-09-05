# Preserving learned contrasts after feedback reservation

Date: 2026-09-05. Bounded primary-source review; no experiment or learner change.
Reviewed next candidate: one opt-in, OGD-inspired native projection
of plasticity directions against already consolidated comparison directions.
This is a disclosed global training operator, not a claim that cells discovered
the projection algorithm. Do not implement a host QP and relabel its solution
as neuronal learning.

## Observed failure and remaining alternatives

The [reservation result](../../experiments/r4_integrated/contextual_reserved_feedback_result.md)
improves fresh edits 3/8→4/8 and retention 2/16→4/16 with 150 offers per arm.
BASELINE uses 32 new/118 replay offers and 80 updates; RESERVED uses 42/108
and 81 updates. Every executed failure receives an admitted correction, yet
12/42 final pair margins are negative; 11 were positive at their last offer.
Five final wrong leaders are admitted-but-erased negatives, five were never
executed old occurrences, and two come from fresh sources. The final TRAIN
weights answer 5/16 old questions correctly before fresh ingestion and 4/16
afterward. These are distinct interference and competitor-coverage failures,
not proof that preserving recorded comparisons will solve every query.
[Artifact](../../benchmarks/contextual_reserved_feedback_v1.json); the result
report records the independent numerical and occurrence audit.

The [TRAIN feasibility witness](../../experiments/r4_integrated/contextual_representation_feasibility_result.md)
excludes finite representational impossibility for its declared constraints;
it neither supplies a learnable trajectory nor authorizes deploying its weights.
Mem0 was searched only for development context; no retrieved result is used as
scientific evidence here. Code discovery used the canonical checkout graph.

## Three relevant approaches

| Approach | Primary mechanism and fit to this failure |
| --- | --- |
| GEM | Projects a proposed gradient subject to previous task-memory loss-gradient inequalities. The paper uses a local linearization; task-average loss protection is not protection of every contrast. Its official implementation solves a conventional dual QP. A direct transplant adds a solver and changes the learning mechanism substantially. [Paper §3, equations 7–11](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), [official implementation](https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py). |
| A-GEM | Replaces the collection of constraints with one sampled average-memory gradient constraint and an inexpensive projection. Individual old contrasts may worsen while the average improves; that does not directly address the observed erased-pair count. [Original paper §4, equations 9–11](https://arxiv.org/pdf/1812.00420). |
| OGD | Stores an orthogonal basis of previous model-output gradients and projects later updates away from that span. Its nonlinear-network argument is local and uses stored Jacobians; limited stored directions and large steps limit protection. Fixed linear comparison features remove that particular Jacobian approximation, making this the most direct small adaptation. [Original paper §3, equation 9 and Algorithm 1](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf). |

## Actual native seam

[`observe_comparison`](../../src/humanbrain/contextual_evidence_memory.py)
authenticates same-query decisions, forms `d = accepted - rejected`, and calls
[`NativeThreeFactorLearner`](../../src/humanbrain/native.py) with zero scratch
bias. The [Rust rule](../../src/humanbrain/_native_src/src/lib.rs),
`humanbrain_three_factor_update`, uses the same eligibility vector for its
competitive gate, norm and synapse mutation. For `v=w_GO-w_NOGO`, it gives
`Δv = 2ηd/sqrt(1+d·d)` when the original pair margin is below one.

Consequently, simply passing a projected vector to that existing ABI is wrong:
it silently tests a different margin and normalizes a different vector. A new
opt-in native entry point must separate original scoring eligibility from the
plasticity direction, preserving the old entry point and default identities.
This is a small isolated numerical extension plus checkpoint/custody work,
not merely a Python option or a reason to rerun the previous experiment.

## One project-specific candidate and its limits

The following is our adaptation and algebraic deduction, not an OGD benchmark claim.
Keep a basis `Q` spanning only authentic contrasts already observed to attain
the existing unit margin, `v·d_i >= 1`, after an admitted new or replay offer.
Do not consolidate weak/negative pairs. Add each independent direction once,
using twice-reorthogonalized Gram–Schmidt in first-certification order;
dependent directions need provenance records but no extra basis column.

For a later authentic offer, test the gate on **original** `d`. If it violates
the unit margin, form `p=(I-QQᵀ)d` and apply in native code
`Δw_GO=ηp/sqrt(1+d·d)`, `Δw_NOGO=-Δw_GO`, with fixed `η=1`.
For every protected `d_i`, `Δ(v·d_i)=0` in real arithmetic. The current
pair improves by `2η||p||²/sqrt(1+d·d)` when `p` is nonzero.
Leave pure-query weights and persistent bias zero. Do not alter the readout,
candidate inventory, replay priorities, teacher contract or offer budget.

Retain original decision/occurrence IDs and `eligibility_delta_sha256` unchanged.
Seal projected-direction hash, basis-before/after hashes, protected contrast IDs,
original/projected norms, original margin and actual mutation separately.
Predeclare residual tolerances from float64 arithmetic; check protected raw
contrast residuals and actual before/after margins, not only basis orthogonality.
Numerical validation failure must abort the attempt, not count a preservation pass.
A numerically exhausted projection is a reported no-op that still consumes its offer.

This protects only directions consolidated so far in the unchanged feature
space. It does not protect unrecorded competitors, action-versus-STOP margins,
or new source competitors. It freezes early overshoot and beneficial changes
along protected directions as well as harmful ones; a feasible inequality
solution need not remain reachable under these stronger equality constraints.
Projection can therefore stall learning despite the earlier feasibility witness.
No rate sweep, automatic basis eviction or fallback host optimization belongs
in this first comparison.

## State, computation and causal accounting

For dimension 1,621 and at most 48 distinct admitted contrasts, an orthonormal
basis needs at most `48*1621*8 = 622,464` bytes. A packed copy of protected raw
deltas would cost another 622,464 bytes; reuse the existing certified ledger
where possible and report copied versus shared storage, scratch arrays and IDs.
Projection and residual checks cost `O(KD)` per offer, basis insertion `O(KD)`;
avoid a dense `D×D` projector or QP. Count these operations and total runtime,
rather than claiming that an unchanged 150-offer count means unchanged compute.

Rust placement does not make Gram–Schmidt a simulated-neuron computation:
it is a conventional global plasticity operator using learned auxiliary memory.
The native rule still mutates readout synapses; the affine GLIF race still
controls selection. Report the hybrid operator contribution separately and
avoid claims of purely local or biologically derived learning.
The roadmap permits this bounded hybrid training experiment with explicit
disclosure; it does not establish neuronal discovery of the projection rule.

Frozen mates must disable **both** synapse mutation and learned basis growth,
while retaining the same certified exposure records outside inference. Verify
zero learned basis rank/payload and zero learned synapses; a learned basis cannot
be hidden behind a zero-weight control. Bind policy, basis and consolidation
provenance into checkpoint lineage. Projection-disabled RESERVED is the matched
algorithmic control; raw-score/external references retain their existing scope
and shared motor, not a whole-system core-disabled claim.

## Smallest integrated decision

After protocol review, compare RESERVED against RESERVED plus this one operator:
same BINDING features, affine actions, authentic FAIL/teacher-PASS pairs,
16 teaching queries/48 encounters, 150 maximum offers and existing motor.
Freeze new transfer instances before outputs; never load oracle weights.
Use focused synthetic native checks for protected-margin preservation, original
gate/normalizer parity with an empty basis, dependent/conflicting directions,
no-op accounting, frozen auxiliary state and custody before one integrated run.
Measure actual fresh/retained edits, protected-margin violations, projection
stalls, unprotected wrong leaders and pre/post-fresh-source retention separately.
Adopt only if retention improves without reducing fresh execution; exact pair
preservation alone is insufficient. If the operator stalls or costs too much,
preserve that result and return to the integrated learning decision.
