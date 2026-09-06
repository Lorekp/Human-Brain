# Local memory context failure: bounded primary-source review

2026-09-06. Research-only follow-up to the completed v5 development fit. No
model, harness, admission or frozen result was modified; no new learner was run.
The roadmap, active plan, writer-credit contract and prior preflight review were
read. Mem0 was searched as development context only; returned older mechanisms
do not supply evidence here. This is the delegated background research task
specified by the research skill, alongside separate saved-evidence analyses.

**Recommendation:** preserve v5 as a failed acquisition result and perform the
orchestrator's selected direct-context decoder positive control before changing
any learning rule. It is the stronger immediate falsifier of decoder context
use. A forward-maintained decoder context-sensitivity trace is discussed below
only as a conditional future alternative to the earlier random activity gate.
The observed failure is broader than swapped output order; neither feedback
change is selected or authorized by this review.

## Observed result and two explanations

The [saved run report](../../.pytest-tmp/local-memory-byte-v5-run/report.json) and
[independent audit](../../.pytest-tmp/local-memory-byte-v5-run-independent-audit.json)
authenticate 5,120 lessons and 100 final actions. Tagged, aggregate,
writer/maps-frozen and writes-disabled each emit `mavi lune` on all twenty final
queries: 1/8 fit and 0/4 in each fresh, delayed and interference group. Fully
frozen scores zero. Tagged first-attempt acquisition is 72/1,024, compared with
68/1,024 without writes. These descriptive counts provide no useful contextual
or writer-learning advantage. All eighty updating-arm emissions are valid
executed bytes, so EOS/UTF-8/action transport is not the leading repair target.
Audit passage establishes evidence integrity, not acquisition.

Only two explanations are selected:

1. **Context credit does not track how the decoder uses context.** The
   [decoder](../../src/humanbrain/byte_sequence_decoder.py) uses independent
   fixed matrices for hidden and context error. The
   [learner](../../src/humanbrain/local_memory_learner.py) averages context
   broadcasts over teacher bytes before applying upstream eligibility. In real
   arithmetic, `mean(Bc delta_t) = Bc(mean(p_t) - target_byte_histogram)`.
   The target term loses order; more generally this broadcast ignores the
   decoder's state-dependent input gain and how its context synapses affect
   output activity. Teacher prefixes can support next-byte learning while
   context remains unused. Constant free output is compatible with this
   explanation, but does not prove it. Neither the independence of feedback
   matrices nor averaging alone makes learning mathematically impossible.
2. **Forward addressing or retention fails before credit reaches the decoder.**
   Every byte, including common boilerplate, writes through the same initially
   untrained maps. Correlated keys can erase earlier information. End-of-question
   activity can also fail to distinguish keys. The implementation's write tags
   track a fixed-key overwrite approximation; they cannot reconstruct a fact
   absent from the fast matrix. Independent map broadcasts additionally omit
   important effects of changing keys and strength. Distinct saved matrices
   alone would not demonstrate useful question-conditioned reads. This remains
   a competing explanation pending the separate read/query diagnostics.

These explanations can coexist. In particular, equal behavior to
writes-disabled does not distinguish an uninformative memory from a decoder
that ignores informative memory. A learning-rule change should not be selected
solely from the four exposed swapped-value failures.

## What the original methods establish

| Primary source | Applicable result | Boundary for this project |
| --- | --- | --- |
| [Bellec et al., e-prop, equations 1–4 and temporal-credit discussion](https://www.nature.com/articles/s41467-020-17236-y) | Synaptic updates combine a time-indexed neuronal learning signal with a locally maintained eligibility at the same time. Random, symmetric and adaptive feedback are distinct variants. The exact decomposition does not make the available online error approximation exact. | Our self-path cell eligibility is related in form. A once-per-answer independent broadcast through a separate fast-memory writer is not the paper's validated algorithm. Averaging errors before pairing them with changing sensitivities loses information that summing their products retains. The paper also identifies failures when local forward credit cannot reach the relevant loss. |
| [Lillicrap et al., feedback alignment, Methods](https://www.nature.com/articles/ncomms13276) | Fixed random feedback can support learning with local presynaptic activity and postsynaptic nonlinear gain; useful alignment emerges through learning. | Independent feedback is therefore not inherently illegitimate. Its demonstrated settings do not guarantee a locally trained fast writer plus autoregressive byte decoder will align. The article offers code on request; no unofficial implementation was substituted for its evidence. |
| [Nøkland, direct feedback alignment](https://proceedings.neurips.cc/paper_files/paper/2016/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf) | Direct random output-error broadcasts can train separate hidden layers, with local nonlinear factors. | This supports considering direct broadcasts, but does not validate averaging sequence errors across differently conditioned decoder states or our memory-tag approximation. |
| [Schlag, Irie and Schmidhuber, fast-weight programmers](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf) | A delta-style fast-memory update corrects associations and addresses limitations of additive updates; slow networks learn the programming representations. | Fast synaptic updating and acquisition of its writer are different problems. Reusing the forward delta rule does not inherit the paper's trained addressing ability or provide a local slow-learning theorem. |

The [official e-prop evidence-accumulation tutorial](https://github.com/IGITUGraz/eligibility_propagation/blob/master/Figure_3_and_S7_e_prop_tutorials/tutorial_evidence_accumulation_with_alif.py)
has explicit hardcoded-eligibility and autodiff variants, random/symmetric
feedback, and an Adam optimizer. Its default `eprop=False` selects BPTT. Even
the hardcoded branch uses TensorFlow differentiation for specified signals and
comparison checks. Thus its implementation is useful for checking equations,
not a drop-in main learner under this project's rules. The
[official fast-weight training loop](https://github.com/ischlag/fast-weight-transformers/blob/main/synthetic/main.py)
uses `loss.backward()` and Adam; those slow-learning results cannot be claimed
for our local rule. Neither codebase was executed or imported. Sources were
accessed on 2026-09-06; linked repository branches are mutable references.

## Conditional credit alternative, not the next experiment

The earlier `mean(tanh(R h_t) * Bc delta_t)` suggestion adds state dependence,
but a random signed gate can suppress or reverse errors and need not reflect
which context neurons influenced a byte. It adds a projection/scaling choice
without identifying the failing connection. The result justifies investigating
context credit, not selecting that specific gate. A direct symmetric readout
signal would be another diagnostic, but changing hidden feedback simultaneously
would confound the context question. Do not queue both.

If a later isolated credit comparison is warranted, a more grounded option
would replace only the decoder-to-context signal. For existing
decoder cell activity `h_t = 0.5 h_(t-1) + 0.5 tanh(a_t)`, let `Wc` denote its
existing context-input synapses and `Ujj` its recurrent self-synapse. Before the
current teacher update, maintain at each context-input connection:

```text
d_j(t) = 0.5 * (1 - tanh(a_j(t))^2)
c_j(t) = 0.5 + d_j(t) * U_jj(t)
S_jk(t) = c_j(t) * S_jk(t-1) + d_j(t) * Wc_jk(t),  S(0)=0
L_j(t) = [B_hidden delta_t]_j
g_k(t) = sum_j L_j(t) * S_jk(t)
upstream_signal = mean_t g(t)
```

This is our proposed **local input-sensitivity approximation**, not a published
e-prop formula for this architecture. It connects feedback to the same decoder
neurons already receiving error, retaining the product with their changing
context sensitivity before temporal averaging. It introduces no new random
matrix, output ontology, source parser or target-dependent input code. Existing
decoder updates, writer/query eligibility and downstream upstream-signal
projections remain unchanged. Mean normalization and learning rate stay fixed.

Each connection uses its own current weight and trace, the postsynaptic gain,
the neuron's self-carry and a declared neuronal error. Contributions are summed
into one signal per upstream context neuron; writer synapses still compute their
own updates from local eligibility. This requires an explicit return signalling
path from decoder context connections to context neurons. It is not a claim
that fixed asymmetric feedback alone solves weight transport: the current
forward connection weight participates in its locally maintained sensitivity.
No full Jacobian over upstream parameters, externally computed per-synapse
correction, retained-network backward traversal, Adam or BPTT is allowed.

`S` ignores cross-neuron recurrent paths and derivatives through intervening
weight updates. With online decoder changes it is a declared local trace,
not the derivative of the complete learning process. All factors must be
computed before that byte's updates, finite-checked, reset per teacher sequence
and discarded afterwards. For 64 decoder cells and 96 context neurons it needs
6,144 float32 values, 24,576 bytes per active stream, plus reduction temporaries;
this is a calculated storage requirement, not measured runtime. It does not
replace the substantially larger retained writer tags. The frozen path must
remain mutation-free. Unused old fixed feedback can remain serialized and
counted for matched initialization; it must not secretly supply a second signal.

## Selected smallest falsifier and bounded return

The source-free matrices differ in the separately reported saved contrasts;
this rules out identical storage in those contrasts, not identical reads or
decodability. Do not train a probe or hidden-answer classifier to repair the
evaluated learner. Resolve the decoder independently with the root-selected
positive control:

- Keep the current 96-context/32-embedding/64-hidden local decoder, rate 0.01,
  seed 733 and genuine first generation before each correction.
- Encode the first and second symbols from `a,b,c,d` in two explicit four-neuron
  one-hot populations. These eight artificial context features are an
  answer-revealing engineering input, not a learned reader, memory or permissible
  source-absent evaluation input. Remaining context coordinates stay zero.
- Teach twelve combinations, reserving `ab`, `bc`, `cd`, `da`; use 384 lessons
  per arm and exactly two target bytes plus EOS. Compare correct context, zero
  context, independently shuffled context and fully frozen learning, under the
  one fixed 120-second bound. Do not feed held-out targets or previous generated
  answers into learning or adapt the shuffle after observing outcomes.
- Inspect exact first-attempt two-byte/EOS output for every fit and held-out
  combination, with strict held-out advantage over controls. The concrete
  protocol owns its final acceptance rule. Constant output, failure to fit the
  supplied context or absence of the declared control advantage at the endpoint
  rejects this route for this budget; lower teacher loss is insufficient.

This is the **one recommended comparison**, already selected by the root; this
report does not authorize a second feedback or architecture variant. It asks a
necessary engineering question: can this unchanged local decoder learn to use
an unambiguous ordered context at all? It does not test the reader, the actual
fast-memory representation, arbitrary unseen bytes, autonomous action learning
or prose comprehension.

If it fails, an upstream-only signal change cannot repair this direct-context
test because its artificial context is not learned. Diagnose the decoder's
existing local learning/teacher-forcing behavior before selecting a correction.
If it passes, return to actual neuronal reads: measure whether paired questions
and swapped document assignments remain distinguishable at the decoder input.
Exactly erased distinctions place the next work in forward representation;
available but unused distinctions may justify a separately preregistered
context-credit comparison. Neither outcome automatically selects the sensitivity
trace above, and the random activity gate remains unselected.

Any later integrated comparison must preserve raw ingestion before targets,
source removal, query-only worker input, exact generated-byte action scoring and
the writer-frozen/writes-disabled controls. Reserve fresh swapped-value pairs;
do not rename the four exposed v5 novel values as fresh evidence. Keep source,
state, exposure and process receipts, QPC durations and the original factor-two
cost margin. V5's 322-second observed duration does not authorize another run:
measure the complete new workload under a new admission, with the existing
900-second/2,048-MiB integrated ceilings. A focused pass earns this return to
fresh-value, delay, interference and real actions, not another automatic series
of mechanism experiments or relaxed thresholds.
