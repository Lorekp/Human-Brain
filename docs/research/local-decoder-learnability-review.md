# Local decoder learnability: a bounded comparison with a certified reference

2026-09-06. Primary-source and mathematical review after the closed direct-context
v1/v2 failures. This report is the only edited file; no model, data experiment or
implementation was executed. The roadmap and active plan were reread. The
orchestrator, saved-curve agent and cell-arithmetic agent supplied the independent
analyses identified below. Mem0 was searched as external development context;
older retrieved mechanisms are not evidence for this decoder.

**Recommendation:** do not adopt symmetric feedback or automatically extend
either failed fit. Preregister one learnability bracket: the unchanged default
decoder and a known-learnable local neuronal reference, each under lexical and
fixed permuted order. Use one analytically motivated endpoint, with predeclared
observations at the old exposure. This separates observable budget/order effects
from a practical representation/credit limitation without pretending that a
finite unsuccessful fit proves architectural impossibility.

## What is established, and only two explanations

The [v2 result](../../experiments/r4_integrated/local_context_use_v2_result.md)
and [independent audit](../../.pytest-tmp/local-context-use-v2-independent-audit.json)
preserve exact v1 random-arm reproduction. Correct random feedback fits 4/12
TRAIN and 0/4 withheld combinations; symmetric fits 3/12 and 0/4. Symmetry removes
premature EOS at the final endpoint and lowers teacher loss, yet gives no exact
transfer advantage. Source review found no sign, gain, trace-update or
pre-update-readout implementation error.

The [saved cycle analysis](../../.pytest-tmp/local-context-use-v2-curves.json)
shows why convergence must not be inferred. Between cycles 24–27 and 28–31,
random correct first-attempt success rises from 1/48 to 12/48; symmetric stays
8/48. Both still reduce all three teacher-position losses. On the 24 within-cycle
first-symbol transitions in the last eight cycles, current/previous symbol
matches are 0/12 for random correct and 0/16 for symmetric correct. These are
correlations under a fixed lexical stream, not counterfactual causation or a
proof of last-answer copying. The fully taught token prefix and free-generation
trajectory also differ.

1. **Acquisition/stability within the exposure budget.** Learning may still be
   progressing, while grouped lexical targets produce repeated competing
   updates. A longer, independently fixed exposure and an order intervention
   can positively establish sufficiency if they work. Decreasing teacher loss
   alone does not authorize further epochs or imply eventual exact composition.
2. **Representation and temporal-credit mismatch.** The decoder must use two
   factors differently at different output phases, maintain phase through
   repeated symbols, and learn when to stop. Its full recurrent forward graph
   has cross-neuron paths, whereas its eligible credit follows each neuron's
   self path. Instantaneous symmetric readout feedback repairs one signal
   approximation, not those omitted paths. A structured neuronal temporal
   representation can make the readout learning problem provably separable;
   its success would diagnose a practical missing prior or inefficient credit,
   not prove that the larger decoder lacks an expressive solution.

## Which existing variants are meaningful

| Variant | What it changes or establishes | Decision |
| --- | --- | --- |
| Current recurrent cell | Retains temporal state and permits cross-neuron computation; local traces omit cross-neuron routes. Partial context use is measured, convergence is not. | Keep the default random-feedback version as the existing learner in one comparison; symmetry has not earned adoption. |
| Existing memoryless bypass | Discards state and uses only context plus previous-byte embedding. | Do not run as a positive control. For `aa`, step two and the EOS step have identical fixed-weight inputs but require different outputs. The same contradiction applies to `bb,cc,dd`; no extra exposure can make deterministic bypass inference emit these pairs followed by EOS. |
| Enforced diagonal-self recurrence | Removes off-diagonal recurrent pathways and hence that source of trace mismatch; still retains time. | A coherent possible architecture, but not a known-learnable reference. It changes forward capacity, and online intervening weight updates remain outside the trace derivative. Do not add it as another arm now. |

These are source-derived conclusions about
[StreamingRateCell](../../src/humanbrain/plastic_memory.py), independently checked
by the cell agent. Bypass with symmetric feedback would use exact instantaneous
two-layer supervised derivatives, but exact credit cannot resolve identical
inputs with contradictory output labels. A diagonal initialization is also not
an enforced diagonal model: unconstrained recurrent updates can create
off-diagonal weights.

[Bellec et al., e-prop](https://www.nature.com/articles/s41467-020-17236-y)
distinguish exact local eligibility from approximated available neuronal error
and discuss failure when local forward credit cannot reach the relevant loss.
That motivates inspecting the computation/credit relationship, not treating
symmetry as a convergence guarantee. The
[IndRNN paper, section 3](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Independently_Recurrent_Neural_CVPR_2018_paper.pdf)
provides a primary precedent for independent recurrent neurons, but analyzes and
trains them with BPTT; its successes are not evidence that our proposed local
diagonal variant will fit. No such variant is selected by this review.

[Rigotti et al.](https://www.nature.com/articles/nature12160) show how nonlinear
mixed neuronal selectivity increases the functions available to a linear
readout. This supports using generic phase/context conjunctions as an explicit
representational reference. It does not establish that a hand-authored clock is
learned, that the specific reference below is biologically faithful, or that
this prior suffices for language.

## A local reference with a checkable learning guarantee

Use the same 96 informative context rates as the existing diagnostic. A fixed
three-neuron temporal chain starts on BOS and advances on every decoder decision;
its state supplies phase, not the teacher byte or a host-selected answer.
Create a generic conjunction neuron for **every** phase and **every** context
coordinate, including the currently zero coordinates. The output receives
`x = [phase(3), phase ⊗ context(288)]`: 291 presynaptic features. Do not wire
phase one exclusively to the first factor or phase two to the second factor.
That would supply the factor-selection computation being tested.

All 257 byte/EOS output neurons start with zero synapses. On a wrong teacher-step
winner, the correct output neuron receives a positive teaching signal, the
winning neuron a negative signal, and each affected synapse changes by its own
presynaptic rate times that signal. Other output synapses do not change. This is
the local multiclass perceptron rule, without an optimizer, hidden gradient or
per-synapse external correction. The
[primary multiclass online-learning analysis](https://jmlr.org/papers/volume3/crammer03a/crammer03a.pdf)
establishes mistake-bound reasoning for additive mistake-driven updates on
separable data. The specific certificate below is our derivation for this
reference, not a reported result from that paper.

The clock and conjunctions are deliberately supplied neuronal dynamics. They
are not a learned reader, learned timing policy or answer-free representation.
The learned output associations still come only from explicit byte/EOS labels.
The proof-only separator must never initialize or correct the evaluated learner.
Use the same greedy output competition and genuine learned EOS criterion as the
existing diagnostic; advance the teacher sequence through all three phases even
when its preceding free attempt stopped early.

For exact binary clock/context activity, the orchestrator's certificate is:

```text
||x||² = 3: one active phase unit plus two active conjunctions.

Proof-only U:
  four unit connections from first-phase/first-factor features to their bytes;
  four unit connections from second-phase/second-factor features to their bytes;
  one unit connection from the third phase unit to EOS.
||U||² = 9; correct output exceeds every wrong output by margin 1.

For M teacher-position mistakes from W0 = 0:
  <W_M,U> >= η M
  ||W_M||² <= 2 η² R² M = 6 η² M
  Cauchy–Schwarz therefore gives M <= 2 R² ||U||² = 54.
```

The norm increment bound uses the fact that the mistaken winner's score is at
least the correct neuron's score before the update. A deterministic tie rule is
allowed. Each complete TRAIN cycle offers all twelve pairs and all three teacher
positions. Among 55 complete cycles at least one must have no mistakes; that
cycle causes no updates and covers the complete fixed training set, so subsequent
cycles remain correct. **Choose 64 cycles, 768 lessons, as one preregistered
endpoint above this bound**, not a succession of additional epochs selected from
results. The bound concerns teacher-position mistakes, not complete-answer first
attempts; free inference becomes correct once all fixed phase inputs are correct.

Use integer synaptic ticks, with one tick denoting rate 0.01, so the reference
update/winner ordering is exact rather than claiming a real-arithmetic theorem
for unconstrained floating rounding. No decay, clipping, adaptive step, hidden
parameter update or teacher-dependent feature is permitted. The readout has
257×291 = 74,787 synapses, or 299,148 bytes at int32, plus clock/conjunction state
and fixed connectivity that must be measured and counted. This is calculated
storage, not measured speed.

The certificate guarantees TRAIN learnability under these assumptions. It does
**not** guarantee the four withheld compositions, sample efficiency of the RNN,
reading, or general sequence generation. Several separators can fit the twelve
TRAIN pairs and disagree on the other four. Reference failure on TRAIN would
first implicate implementation or violated assumptions, not justify a sweep.

## One prospective experiment and decisions

The proposed experiment has six trajectories, including two negative controls,
in this fixed per-lesson order:

| Arm | Learner and context | TRAIN order | Teaching |
| --- | --- | --- | --- |
| decoder_lexical | Existing default random-feedback decoder, correct context | Existing lexical order | Enabled |
| decoder_permuted | Same decoder and correct context | Private `Random(734)` permutation of all twelve pairs per cycle | Enabled |
| reference_lexical | Phase/context reference, correct context | Existing lexical order | Enabled |
| reference_permuted | Same reference and correct context | Exactly the decoder's permutations | Enabled |
| reference_zero | Same reference, zero context; phase clock retained | Existing lexical order | Enabled |
| reference_frozen | Same reference and correct context | Existing lexical order | All labels offered, updates disabled |

Freeze both streams before training and keep 64 complete cycles in all six arms.
Use fresh initialization; do not resume the closed checkpoints. The existing
decoder retains seed 735, dimensions, rate 0.01, local traces and per-byte
teaching. Reference zero initialization and different architecture are explicit;
do not claim matched parameter states across architectures. Preserve the same
informative input rates, genuine first attempts and all three teacher offers.

Predeclare read-only exact TRAIN/withheld probes only at lessons 384 and 768,
with no adaptive stop or parameter selection. The lexical decoder at 384 should
reproduce its authenticated old outputs/losses/tensors. Both checkpoints count
as exposure to the evaluator; neither supplies teaching. Totals are 4,608
first attempts, 13,824 teacher offers and 192 checkpoint probe generations.
There are 11,520 enabled teacher-step opportunities; perceptron mistake-triggered
updates and decoder updates are counted separately. The 54-mistake certificate
applies only to the two correct-context references, not the nonseparable zero
context or deliberately frozen control.
`ab,bc,cd,da` remain previously exposed development probes excluded from TRAIN,
including permutations. They are not fresh evaluation.

At the final endpoint, a useful decoder or reference must attain 12/12 TRAIN
and 4/4 withheld exact two-byte-plus-EOS outputs and strictly exceed both
reference controls on withheld exact count. The reference's separate certificate
criterion is 12/12 TRAIN, no more than 54 teacher-position mistakes, and a recorded
complete no-mistake cycle. Certificate passage alone does not earn integration.
Report all six arms even if a control unexpectedly succeeds; this invalidates
the claimed advantage rather than relaxing the rule. The operational
interpretation is fixed in advance:

- If the lexical decoder satisfies the existing exact criterion only at 768,
  the old exposure was insufficient for this fixed case; no structural change
  was necessary for that gain.
- If the permuted decoder succeeds where matched lexical training does not,
  order is a demonstrated intervention for this stream. It does not prove a
  universal optimal curriculum or authorize seed/order searches.
- If the certified reference fits TRAIN and the decoder does not under either
  order by 768, useful computation exists with local neuronal updates within
  the declared budget. Retire further tuning of this decoder candidate and
  prefer the explicit temporal representation for the next scoped return.
  **This does not prove that no larger RNN budget could succeed, or distinguish
  expressivity from credit by itself.** It identifies a practical failure of
  the current computation/learning combination relative to a learnable reference.
- If the reference fits TRAIN but fails composition, keep both facts. Do not
  describe it as the successful output solution or silently teach the probes.
  Any failed reference certificate requires a source/arithmetic investigation.

Allow one reviewed implementation/QA pass and one complete experiment under
120 seconds and 2,048 MiB, including both checkpoints, serialization and audits.
The analytic bound is an exposure justification, not runtime admission; budget
exhaustion closes the attempt. No automatic next endpoint, symmetric adoption,
phase variant, diagonal trial or threshold change follows.

A useful exact compositional result earns one return to document-derived
neuronal context and real actions under a fresh integrated cost admission.
Remove the answer-revealing factor inputs; only the actual retained memory/read
may feed output. Preserve fresh swapped-value, writer-frozen, writes-disabled,
delay/interference and exact executable-action checks. The phase/gating prior
must remain explicit, and two-byte success is not variable-length output or
general coding. Acquiring useful document representations and learned action
selection remain the shared milestone, not tasks supplied by this reference.
