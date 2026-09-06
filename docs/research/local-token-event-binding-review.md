# Reusable binding structures after the fixed multiscale comparison

Status: bounded research and source reasoning, selected by the orchestrator on
2026-09-06 for concrete interface review. No new model calls, fits or endpoint
probes. This selects one design question, not an admitted acquisition.

## What the completed result establishes

The [completed result](../../experiments/r4_integrated/local_memory_multiscale_v1_result.md)
and saved `local-memory-multiscale-v1-description.json` report tagged, aggregate
and candidate writer-frozen at 5/8 fit, 2/4 novel, 2/4 delayed and 0/4
interference. The scalar reference is 5/8, 1/4, 1/4, 0/4. The gain is one novel
case and its delayed duplicate. Neither same-question assignment pair is jointly
correct. Writer-frozen matches every correctness and the first sixteen outputs;
its two other differences are wrong interference answers. The strict audit
SHA256 is `a9a1ebfec69f909124f8583834469739fdf13b959999ddac6b03aa0dea87dc48`.
Run plus audit took 374.5419774 seconds. Exact scalar-reference and declared
writes-disabled parity make this a controlled fixed-prior gain. They do not
demonstrate useful learned writing, convergence, or improved prefix retention.

The earlier binding localization found contracted identifier contrast before
values and predominantly common-direction responses to swaps across questions.
Those are measurements of the earlier scalar checkpoint, not measurements of
the new multiscale endpoint. Together the records justify reconsidering the
representation and association interface; they do not prove information was
erased or that one architecture below will succeed.

## Three established building blocks, with different missing links

### 1. Generic token events with compositional neuronal content

**Established:** subword tokenization can learn a vocabulary from raw sentences,
without supplied word segmentation. SentencePiece treats whitespace explicitly
and packages its segmentation state. Its stated round trip is for *normalized*
text; default NFKC is not a raw-byte preservation guarantee. This matters for
software identifiers and exact output. Its translation experiments do not
establish local neuronal learning. [Kudo and Richardson, 2018, sections 3.1–3.5](https://aclanthology.org/D18-2012.pdf).

**Project hypothesis:** change the unit of writer admission from every byte to
a generic text event. A neuronal encoder composes the event's ordered bytes;
the persistent writer consumes that neuronal activity once per event. This
separates within-token composition from across-token association. Filler then
costs fewer writer overwrites without choosing which token is an identifier.
All token contents must use the same encoder and writer maps. Learned encodings
must use declared local online rules; a random whole-word lookup alone does not
establish composition of an unseen value's bytes.

The tokenizer is conventional preprocessing, explicitly allowed by the project,
not the learned reasoning system. Freeze it using only admitted training text;
retain arbitrary bytes, spaces and punctuation exactly, including fallback and
streaming flush behavior. Do not introduce identifier/value token types, extract
fields, select answer factors, or merge according to task labels. Compression
could still join an entire familiar record and encourage memorization, so the
interface must bound event length and make every byte recoverable. Generic
segmentation is not itself a binding solution.

### 2. Learned selective admission into neuronal working memory

**Established:** PBWM learns selective update/maintenance using basal-ganglia
gating and an actor–critic mechanism. Different maintained populations can resist
distractors while another population updates. Its 1-2-AX demonstrations use
discrete stimuli. Cortical learning combines Hebbian learning with GeneRec;
the authors describe backpropagation derivatives computed by activation
settling, with minus/plus/update phases. Thus the gating result is a useful
structural precedent, not a drop-in proof for this byte-stream learner or its
present credit rule. [O'Reilly and Frank, 2006, sections 2–3, 5 and appendix A](https://ccnlab.org/papers/OReillyFrank06.pdf).

**Project hypothesis:** a learned neuronal admission gate holds a useful earlier
representation while later content changes. This is different from the current
write-strength beta, which scales a matrix update while both key and value maps
still read the current writer activity. A gate would control the *state supplying
an address*, without a host instruction saying when an identifier ended.

Learning gate timing is the main unresolved problem, not a bookkeeping detail.
Local actor eligibility with an explicit neuronal reward/error is possible in
principle, but an actual proposal needs its stochastic/deterministic gate rule,
temporal credit, exploration cost and state ownership derived before execution.
The PBWM phases should neither be hidden nor automatically declared forbidden:
their local information and execution costs need explicit review. A gate whose
host opens on `kNN` or closes at `has value` would supply the binding solution.

### 3. Temporally separated associative activity and fast synapses

**Established:** fast-weight delta memory writes an outer-product correction
between value and key activities, reducing interference relative to simple
additive storage in the studied settings. The relevant controllers were trained
with global gradient methods; their success is not evidence for our online
controller credit. [Schlag, Irie and Schmidhuber, 2021](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf).

**Project hypothesis:** make a preceding neuronal activity trace available as
the presynaptic associative cue when later content neurons become active.
An update such as `delta M_ij = beta * (v_i - read_i) * cue_j` is local at the
fast synapse. It still needs a learned, generic account of which earlier cue to
maintain and which present activity to associate; maintaining every recent cue
can produce a bag of nearby content rather than an assignment.

Distinguish this *readable forward activity* from an eligibility trace used only
when teaching arrives. E-prop separates local eligibility from the neuron-level
learning signal; improving either does not automatically create a useful forward
binding representation. [Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y).
The existing matrix already supplies fast association. The new structural issue
would be its cue source and timing, not simply another outer-product rule.
Full exact controller-history sensitivities are not a cheap fallback: a naive
14,593-by-64-by-64 float32 derivative costs 239,091,712 bytes per stream before
scratch. This is an algebraic bound for that representation, not a claim that
all better local credit requires it.

## One recommended bounded next step

**Review and qualify a generic token-to-neuron event interface first.** It is
the smallest structural question that can reduce the observed competing byte
updates without immediately requiring a new gate-learning architecture or larger
per-synapse traces. This is an engineering priority inference, not a paper-derived
winner. Retain learned selective admission as the strongest alternative
explanation; do not implement it concurrently.

The next deliverable should specify one lossless, bounded-length segmentation
policy and one neuronal event encoder with explicit local traces. Its contract
must account for every raw-byte encoder advance, every writer event, buffers,
parameters, vocabulary state and resets. Host token IDs may index only declared
input representations; they may not select a writer address, value neuron,
output byte or action. Source removal must remove tokenizer text buffers too.
The query interface must be declared consistently; no silent parsed lookup.

Prefer examining an order-sensitive neuronal recurrence over the existing shared
byte codes, applied uniformly to every token, over opaque whole-lexeme IDs or
mean pooling. The former can preserve order-dependent distinctions; it does not
guarantee injectivity or linear compositional decoding. An arbitrary recurrent
code need not satisfy the additive-feature assumptions of the earlier successful
LMS component certificate. Neither a larger cosine separation nor a lossless host
tokenizer establishes that the actual output can recover unseen byte order.
Do not supply two answer-position slots or select an encoding dimension to fit
the current two-byte answers. Choose dimensions by the reusable interface and
state budget, then hold them fixed.

The follow-up review must resolve whether this encoder is fixed or locally
plastic; a fixed code is a disclosed representation prior. If plastic, specify
the incoming neuronal teaching signal, its pre-update weights and the lifetime
of within-token eligibility before claiming learning across delayed questions.
Resetting encoder activity at every generic event is not permission to erase
credit silently. Writer activity must retain its declared cross-event/document
lifetime. All shared initialization, fixed encoder state and vocabulary data
must be reproducible independently of answers and the fitted endpoint.

Before any neural fit, propose one at-most-60-second, 2048-MiB deterministic
preprocessing qualification using only the admitted training stream and authored
byte-preservation fixtures. Check exact round trips, causal buffering/flush,
bounded event length, unseen-byte fallback, event counts and the distribution of
writer-event gaps. Report bytes *and* events; do not claim shorter writer streams
reduce total simulation time until all encoder work is counted. No vocabulary,
threshold or policy sweep, no target-aware segmentation and no protected cases.
Stop that qualification on broken preservation or insufficiently specified local
encoder credit; do not extend its budget. These checks can reject an unsafe or
ineffective interface, but cannot certify neuronal information retention or
capability.

Only an implementation-reviewed interface should return to the same source-absent document-to-
Python-action workload under fresh cost admission. Keep an eventized writer-frozen
control to separate the new fixed prior from learned writing, the immediate
byte-stream reference, and the existing disabled controls. Count complete
same-query assignment swaps and interference separately from individual answers.
No success on tokenizer fixtures substitutes for those actions. Pretrained
reasoners, host answer parsers, BPTT and global neural fits are outside this
proposed main path. No new observation or fit is authorized by this note.
