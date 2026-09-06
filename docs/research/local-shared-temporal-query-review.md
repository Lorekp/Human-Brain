# Lag-query follow-up: share temporal alignment across sensory-code channels

Completed: the [fixed comparison](../../experiments/r4_integrated/local_shared_temporal_query_v1_result.md)
and independent audit finish in 119.745 seconds at source `82db27e`.
Learned scores 8/8 taught, 4/4 novel, 4/4 delayed and 2/4 interference;
controls score zero and the dense reference exactly reproduces 7/4/4/2.
Strict qualification fails on the two interference checks. Close this fit;
the design below is its historical contract, not permission to repeat it.
Saved evidence motivates a bounded review of single-location recall versus
byte marginalization, with no new policy output or acquisition claim yet.

Root selection: implement this one integrated replacement, then complete ordinary
QA and fresh cost admission before a single fit. The preceding result is closed
at `d60ccab68db0fdf76ccc9ee167d399ea19a087c0`. No new cost or fit is admitted by
this design commit. Use five ordered arms: `learned`, `address_frozen`,
`slow_frozen`, `writes_disabled`, `dense_reference`. The independent temporal-map
peer review and integration review agree on the physical sharing, arithmetic,
streaming lifetimes and resource contract below. Their source-only notes are
`.pytest-tmp/shared-temporal-map-peer-review.md` and
`.pytest-tmp/shared-lag-integration-review.md` (SHA256
`c61b35833fd543ce61fe978fa8be394c4605d86f666b83c2b5bf068b13ccd2d0`).

2026-09-06. Bounded saved-evidence and source/algebra review. Research skill
applied; Carson and Hubble independently checked the proposed local rule and
conditioning. No learner construction, checkpoint forward, new prediction,
teacher offer, fit or parameter trial was performed. This recommendation does
not authorize a workload or advance the gate.

## What the completed comparison establishes

At source `8ef70d6f512caffc0cbc9eac2f19b772298447b2`, learned scores are
7/8 fit, 4/4 novel, 4/4 delayed and 2/4 interference. All three matched controls
score zero. Exact saved references reproduce sequence 4/2/2/1 and byte 5/2/2/0.
Strict qualification therefore fails, while the declared novel retrieval
advantage is present. Both novel assignment swaps now work. Three of four
same-memory, different-question fit pairs produce both answers correctly.
These are meaningful gains in learned question-conditioned retrieval on this
exposed synthetic roster. Literal sensory storage, fixed copying/adjacency and
the supplied occurrence-edit motor remain explicit assistance.

Learned first-attempt success is 472/1,024. Its eight 128-lesson bins are
`0,13,70,69,83,71,81,85`; mean per-offer NLL falls from 2.310500 to 0.338597.
The corresponding sequence reference had 435 first successes. Falling loss
does not establish convergence or warrant an automatic exposure extension.
The four delayed successes repeat the same four stored novel memories.

The saved initial address odds sharpen the endpoint interpretation. Below,
odds always mean first listed target versus second listed target, evaluated
under each of the two questions in the same memory:

| Fit pair | Question 0 log odds | Question 1 log odds | Correct-direction change | Prior TV | Both correct |
|---|---:|---:|---:|---:|---|
| k00 / k01 | 0.110756 | -0.036814 | 0.147570 | 0.036823 | yes |
| k02 / k03 | 0.171669 | 0.124734 | 0.046935 | 0.011686 | no |
| k04 / k05 | 0.043264 | -0.016104 | 0.059368 | 0.014835 | yes |
| k06 / k07 | 0.063144 | -0.007015 | 0.070159 | 0.017623 | yes |

All four changes have the useful sign; three cross zero. The source probability
margin is nevertheless small, including only 0.007015 log odds for correct k07.
This is improved discrimination, not robust separation.

## The three failures have two immediate forms

* **Fit case 3, k03 = bc:** emits bb. The correct value start has prior
  0.467385, versus 0.529476 for k02 = bb (target log odds -0.124734).
  Both branches support the first b. At the second decision the saved mixture
  assigns b 0.473887 and c 0.419112; selecting b concentrates posterior mass
  0.998922 on the wrong bb branch. Initial ranking is already wrong.
* **Interference case 16, k00 = ac:** emits cc. The correct ac start is the
  largest individual prior, 0.214145, but ca/cc/cd starts sum to 0.601461.
  The actual first mixture assigns c 0.537586 versus a 0.191774. After c,
  the next mixture slightly prefers c (0.314002) over d (0.294795) and a
  (0.284024), concentrating on the unrelated cc branch.
* **Interference case 18, k02 = bd:** emits cc. The correct bd start again
  ranks first, at 0.234029; cc/cd starts sum to 0.392253. The first mixture
  assigns c 0.350976, d 0.330511 and b 0.209481. After c, c 0.453282 beats
  d 0.438366 and the cc branch dominates.

All three emit complete source values and then EOS with probability about
0.91576. The failures do not indicate source erasure or a broken EOS path.
The two interference failures are concrete examples of marginal byte pooling:
a group of individually weaker starts can outweigh the correct start. The
fit failure additionally requires stronger correct addressing, not merely a
different pooling rule. Source/target start labels in this analysis were
obtained externally from the saved roster; they were never model inputs or
teaching signals. No counterfactual output was generated.

## One recommended structural comparison

Keep the generic lag query and literal sequence tape, but replace the dense
unconstrained 1,024-by-1,024 address projection with **one shared 32-by-32
temporal projection**, used identically across all 32 sensory-code channels.
Do not change soft competition, mixture/filtering, codec, local rate, tape
capacity or exposure in this comparison.

Let `Z` be the existing energy-normalized query population reshaped to
32 lags by 32 code channels. Let `K_s` be an occupied tape key with the same
shape. Define a learned `B[32,32]`, projected query `Q = Bᵀ Z`, and slot
potential `a_s = <Q,K_s>_F`. All lags and occupied slots participate. The
same existing raw-byte code population is used by documents and questions.
No identifier, word, role, span or answer detector is introduced. The new
prior says that corresponding sensory-code channels should be compared while
temporal alignment is learned. It removes arbitrary cross-channel rotations
from the learned map; this is a real restriction, not a reparameterization of
the full A hypothesis.

The saved result does **not** establish that learning those rotations caused
the failures. This choice is motivated by the now-demonstrated usefulness of
raw lag evidence and the reusable correspondence available between the two
streams. It is a more substantive, smaller hypothesis than another gain or
feedback adjustment. Random 32-dimensional byte codes still have nonzero
cross-byte similarity; sharing does not make an exact symbolic equality test.
The 32-lag truncation and 128-slot tape limit also remain.

For the held codec and teacher sequence, retain the existing initial prior
`p`, final causal posterior `b_T` and slot error `e_s = p_s - b_T,s`.
The projected-query neuron signal is `H = Σ_s e_s K_s`. The local rule is

```
G_B = Z Hᵀ
B_new = B - 0.01 G_B
```

Physical ownership is one 32-input/32-output B circuit time-multiplexed over
the 32 code channels. In channel d, synapse i→j sees presynaptic `Z[i,d]`
and postsynaptic teaching signal `H[j,d]`; its own accumulator adds their
product. B is held while all channels and the complete teacher prefix finish.
Accumulate those local products in float64, finite-check, cast the completed
proposal once to float32, and publish alongside the same held codec group
after all proposals are finite. This is shared circuit reuse, not independent
copies whose weights are tied by an external per-synapse solver. No reverse
temporal differentiation or upstream query learning is added.

There are 1,024 learned B synapses, but 32 applications of that circuit per
query and 32×1,024 local contribution products per teacher sequence. Count
those actual operations, including frozen modes. Address slow-weight storage
falls from 4,194,304 to 4,096 bytes; with otherwise unchanged components the
model tensor count is 567,568 bytes. Tape, codec, validation, receipts and
motor costs must still be measured; these algebraic counts are not timing or
RSS predictions.

Use ordinary private Xavier initialization at seed+4, shape 32×32, with no
additional scale factor. Its entry variance is 1/32; both this B and the old
Xavier A have expected projected-query squared energy `||Z||_F² = 32`.
Their initial competition is **not** matched: score variance becomes
`||Z K_sᵀ||_F²/32`, versus `||Z||_F² ||K_s||_F²/1024` for dense A. Shared
channel geometry changes correlations and can increase particular score
variance. This initialization is a prospective standard-rule choice, not a
result-tuned gain or claim of identical initial behavior.

The prior conditioning argument remains useful but changes form. The held
address loss has Hessian `Cov_p(K) - Cov_b(K)` with operator norm at most one
for unit-norm flattened keys. The B update induces
`ΔQ = -η H ZᵀZ`, not a scalar multiple of H. Because `ZᵀZ` is positive
semidefinite with largest eigenvalue at most 32, η = 0.01 gives

```
ΔL <= -η (1 - η λ_max(ZᵀZ)/2) ||Z Hᵀ||_F²
    <= -0.0084 ||Z Hᵀ||_F².
```

This is constrained, real-arithmetic descent for fixed tape and codec.
`Z Hᵀ` can vanish even when H is nonzero. It proves neither general
learnability, combined codec/address descent, float32 monotonicity nor
cross-question retention. No conditioning trial is needed or proposed.

## Alternatives and the next workload

A hard initial winner would directly remove the demonstrated pooling route,
but preserves case 3's wrong initial ordering and discards the current mixture
semantics. Its counterfactual exact action count is unmeasured. Do not select
it merely because two observed cases happen to rank the right start first.
A contrastive or margin objective could target stronger discrimination, but
requires a separately justified competing sequence/teaching protocol and must
not supply host-labeled target locations. Existing likelihood credit already
has the correct direction in all four paired cases. Neither alternative is
selected for an additional queue.

Recommend one five-arm integrated comparison: shared-B learned,
address-frozen, slow-frozen and writes-disabled, plus an exact static archive
of the immediate dense-A learner at 8ef70d6. The older sequence and byte
comparisons remain authenticated historical evidence; rerunning them adds no
new isolation to this single change. Keep fresh seed733, the same 1,024
lessons per arm and unchanged final raw-document/action roster. Preserve
exact dense-A initial/final scientific state, all 1,024 lesson receipts and
20 final outputs against this completed run, with accounting-only fields
declared separately.

Ordinary scalar/circuit-sharing/interface QA precedes any acquisition. Then
require fresh five-arm cost admission: 20 disposable lessons, 60 teacher
offers, 100 final actions plus two motor demos, and five complete-cap fixtures.
Keep twice the measured separate lesson/journal/cap maxima plus 30 seconds,
the 900-second aggregate run/audit ceiling and 2,048-MiB process-tree ceiling.
Keep streamed scientific JSONL and the completed cost-audit peak in admission.
Use the immediate reference's authenticated full checkpoint payload
`F = 105,236,476 bytes`; do not inherit the obsolete two-reference constant.
Both cost and full auditors must stream its complete 6,144-row historical
journal with the same parse/lifetime scope, even though only its learned arm
is compared. No indexed dictionary retaining all old lesson objects is needed.

The completed lag run's parent peak grew 115.597656 MiB relative to its cost
sample; the earlier run estimate underpredicted that component by 75.40625
MiB after its allowances. Receipts do not identify an allocator or stage as
the cause. Accordingly freeze a **128-MiB run reserve**, rounded above that
observed parent increment, rather than knowingly reuse the insufficient
32-MiB run allowance. Keep 32 MiB for the audit allocator reserve, whose prior
projection covered its observed peak. With the same 262,144-byte row/line
guards, freeze one conservative allowance for both processes: up to four row
objects (two current and two historical) and four line encodings across the
current and reference generators, `E = 2*4*262144 + 4*262144 = 3 MiB`.
Only one selected historical learned row and its current matching row may
be retained across comparisons; advancing generators can transiently retain
the previous and next objects. No history list is allowed. Review and enforce these lifetimes
in the actual implementation and authored streaming QA before freezing.
The resulting prospective estimates are
`P_run = max(cost parent+worker peak, cost owned-tree peak) + 3 + 128 MiB`
and `P_audit = finished cost-audit owned-tree peak + 3 MiB + 2F + 32 MiB`.
Admission requires their maximum to fit 2,048 MiB. These margins acknowledge
observed growth; they do not prove a process-memory bound or identify its cause.
Stop if the fresh time/memory/custody estimate fails. No rate, temperature,
seed, epoch or budget adjustment is proposed.

The prospective useful criterion remains 8/8 fit, 4/4 novel, 4/4 delayed and
4/4 interference, both assignment swaps, all four matched fit query pairs,
and strict novel superiority to the three matched controls. Immediate dense-A
already scores 4/4 novel, so strict superiority to that reference on novel
alone is impossible: require no stratum regression and strictly more overall
successes than its 17/20, alongside absolute 20/20. This is restored completion,
not a novel-superiority claim against dense A. It is a new comparison's declared criterion, not a retrospective
pass for the closed run. On failure, close the sharing hypothesis. On useful
success, return to longer content and varied document/query forms in the same
action loop, rather than another improvement to this roster alone.

## Evidence and source boundaries

* Saved description `.pytest-tmp/local-lag-query-v1-description.json`, SHA256
  `5c5057a9771888c4fc9276fcef05fe06c55e257722e609b47db5a49d2004a0d1`.
* Full saved audit `.pytest-tmp/local-lag-query-v1-run-independent-audit.json`,
  SHA256 `064da6dde81f3faee42f02b68737d74378843aa388df26c069fdfca5ee0dffcf`.
* Actual mixtures and prefix beliefs come from the description-bound
  `local-lag-query-v1-run/source-free/generated.json`, SHA256
  `13e7088652ba9d49220bb891209a68a801d386e18e5834597d063630acd13d7d`.
* Primary implementation sources: `src/humanbrain/local_memory_learner.py`
  (`_query`, `teach`, `_mixture`, `_observe`) and
  `src/humanbrain/neuronal_sequence_memory.py` at 8ef70d6; selected mathematical
  contract `docs/research/local-lag-query-review.md`.

The proposed sharing rule is derived here from those actual interfaces, with
independent peer algebra checks. No new literature-backed performance claim
is made, and no additional literature or model search is needed to distinguish
this one prospective hypothesis from the saved observations.
