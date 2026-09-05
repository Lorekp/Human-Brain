# Acquisition strategy after the query-aligned failure

Decision review, 2026-09-05; historical conditional proposal, not an adopted
implementation. The [one qualification](../../experiments/r4_integrated/frozen_reader_qualification_result.md)
completed after preparation within 11:24–13:24 UTC and **failed**: known-source
answers are 24/48 and 28/48, below 44/48; STOPs also miss their threshold.
The proposed router comparison therefore does not proceed with this reader.
The design below records the rationale and rejected admission path; it does not
authorize replacement readers, threshold tuning or another fit.
The [roadmap](roadmap-reading-to-coding.md) and [active plan](balanced-evidence-to-action-session-v1.md)
own priorities. The [alignment rationale](rate-cell-failure-review.md) remains historical.
The review itself ran no models; the separately frozen qualification owns its
subsequent execution and negative result.

## Evidence and the unresolved question

The [completed comparison](../../experiments/r4_integrated/query_aligned_rate_cell_result.md)
has 0/48 final supported executions in all four treatment phases. Exactly one
conditional gold span wins in each phase and is gate-blocked; threshold changes
cannot repair the other 47 selections. Supported span loss falls across passes,
but that is not successful acquisition. One teaching action succeeds and is
wrong at the final checkpoint. These observations do not isolate catastrophic
forgetting, insufficient representation, or shared-producer interference.
The result also preserves a failed accumulated numerical replay tolerance and
the separate successful same-state numerical diagnosis; neither is erased here.

The current [runner](../../experiments/r4_integrated/query_aligned_rate_cell_python_session.py)
stores authenticated lessons but trains on the current lesson and immediately
previous lesson: 256 encounters and 511 offers. Its inference reads raw source
storage through learned parameters, not the teacher-episode ledger. Thus physical
retention of records, representative rehearsal, useful episodic retrieval and
learning reusable language computation are four different things.

Existing [program-map memory work](../program-map.md) already includes sparse
episodic pointers, exact cross-surface associations and neutral two-slot frame/fact
memory. Earlier contextual runs tested broader replay, AROW and protected contrasts.
A new queue, exact question lookup, or another protection policy is not the missing
innovation. Another randomly initialized reader fitted to these same 64 exposed
requests is currently a poor use of the next experiment budget: several demanding
functions still have to emerge together, while the comparisons cannot locate one
decisive failure. This is a research allocation judgment, not an impossibility proof.

## Three materially different strategies

| Strategy | What changes | Why it might help; principal limitation |
| --- | --- | --- |
| Representative experience rehearsal | Keep raw episodes and recompute current gradients from a deterministic sample of the whole observed history, rather than only the previous lesson. | Tests availability of past practice without changing reading computation. It cannot supply missing reading competence or prove that forgetting caused the current failure. Earlier project replay results prevent treating it as an untried universal cure. |
| Learn reusable reading computation | Broader ordinary language/task experience, potentially separate answerability and span producers, followed by continual adaptation. | Directly pursues reusable neuronal comprehension. It requires a credible acquisition curriculum and substantially more experience than repeated tiny fits; neither another attention variant nor an optimizer sweep is presently justified. |
| Rapid neuronal experience memory with a frozen competent reader | Preserve experiences as inference-addressable evidence; learn which evidence to use from action feedback, while a declared pretrained component supplies language comprehension. | Separates language acquisition from rapid session learning. It risks becoming an ordinary reader wrapper; strong retrieval, conventional-learning and full-corpus controls must be able to reject it. |

Experience replay is a serious reference: Chaudhry et al. found that joint training
on current examples and small episodic memories was strong across their four
supervised continual-learning benchmarks. Those findings do not predict success
on our reading/action stream. [Original paper](https://arxiv.org/abs/1902.10486).

Neural Episodic Control separates slowly changing representations from rapidly
updated experience values, demonstrating faster acquisition in its reinforcement-
learning environments. This supports considering separate timescales, not a claim
that its value dictionary solves language or implements our proposed cell memory.
[Pritzel et al.](https://proceedings.mlr.press/v70/pritzel17a.html).
Fast-weight work likewise motivates a distinction between activity, temporary
synaptic memory and slower learned weights; its recent-past memory is not evidence
of durable semantic abstraction. [Ba et al.](https://arxiv.org/abs/1610.06258).

## Recommendation: test whether learned experience addressing earns its place

Prefer one **conditional experience-first hybrid comparison**, with a real chance
of rejecting the neuronal router. Ask: after authentic action feedback, can cell
memory retain and apply useful experience under changed wording and competing
sources, beyond a capable frozen reader and ordinary retrieval at comparable cost?
Do not simultaneously train a new language producer or slow consolidation system.

As a system class, this is retrieval-augmented question answering. The research
claim is limited to acquired neuronal experience addressing and its useful cost
or behavior contribution. It does not replace the long-term commitment to broader
neuronal learning, or establish that the cells understand language.

One concrete external candidate is `deepset/roberta-base-squad2`, a RoBERTa reader
fine-tuned on SQuAD 2.0, including unanswerable questions. Its model card reports
79.87 exact match on that development set and currently lists CC-BY-4.0. This is
external supervised reading knowledge, including answer extraction and no-answer
behavior, not a neutral encoder or evidence of our cells learning English.
[Official model card](https://huggingface.co/deepset/roberta-base-squad2).
Pin one revision, tokenizer, license and runtime artifacts before use. The project
currently declares ONNX Runtime and tokenizers, not Torch/Transformers; a compatible
export and measured resource envelope remain preparation work, not verified facts.
Public pretraining/QA exposure precludes treating familiar public questions as clean
transfer. Do not replace this candidate automatically if it fails the fixed test.

The proposed cell role is content-addressed selection among retained experience
slots. A slot binds original request, raw source occurrence, before/after workspace
and execution provenance; teacher answer bytes stay in the authorized training/audit
view. The inference payload is the original raw evidence, not a teacher-written
answer summary. Fixed external embeddings may supply sensory keys. Simulated rate
cells integrate query/key inputs and learned query-to-slot synapses; their selected
index controls which evidence reaches the reader. The reader returns a source span
or STOP, then the existing learned literal-edit motor acts and Python executes.

Rapid plasticity must modify those addressing synapses from an actually executed
selected-source outcome. Reward a successful selected source; penalize an executed
failure. A teacher repair alone does **not** identify a correct source. If correction
needs a positive source association, a declared, fixed-budget reader proposal must
itself execute successfully and bind its source occurrence. Otherwise record the
teacher but admit no positive source update. No answer-containment search, hidden
gold-source index, source-type filter or latest-record rule may choose the slot.
Retain the whole authenticated episode; representative replay can reuse certified
outcomes only under an unchanged reader/task contract, without calling it fresh
execution. Freeze the plasticity equation and replay schedule before scored work.

This changes the inference-memory contract and tests acquired addressing, not
answer-free internalization or learned abstraction. Reuse suitable opaque custody,
pointer and motor utilities; do not duplicate the exact two-slot learner or claim
that an episode array is a new neuronal mechanism. Slower structure learning is a
later question only if this simpler persistent loop becomes useful.

## One comparison that can reject the proposal

Use the same frozen reader, raw episodes, embedding cache and motor in every arm:

- Learned cell addressing with authentic feedback and persistent synapses.
- Same cells and episode writes with plasticity disabled; no hidden learned state.
- Core disabled: fixed content-similarity retrieval with identical reader-call limits.
- An independent conventional learned router, such as a small shared MLP scorer of
  query/episode features, with the same inputs, episodes, feedback and update budget.
  Freeze its architecture and account for capacity/compute differences. A mathematical
  re-expression of the cell equations is only a parity check; matching that equivalent
  implementation cannot reject simulated-cell learning.
- The frozen reader over **all** retained raw documents, with complete answer/STOP
  aggregation and measured total cost. Do not restrict this reference to force a
  routing advantage; the present small corpora are cheap enough that it may dominate.

At the final checkpoint, reset or shuffle learned addressing while retaining the
same episode contents: the claimed learning gain should disappear. Count correct
whole-answer executions and unsupported STOPs, not retrieval recall alone. Reader
cost includes chunking, aggregation, all tokens/calls, latency and memory. One
small, declared component check on already exposed development passages can verify
reader output/STOP compatibility before router fitting; its relevant-source input
is an evaluator-only ceiling diagnostic, never routing supervision or fresh evidence.

Nearest-neighbor augmentation can already improve pretrained language models and
adapt them through datastore changes without extra training. This is why ordinary
retrieval is a substantive control, not a deliberately weak baseline.
[Khandelwal et al.](https://arxiv.org/abs/1911.00172).

Proposed development workload: 24 initial acquisition episodes with ordinary
post-first-attempt corrections, followed by ingestion of eight fresh facts without
teacher exposure. Then execute 32 probes: eight unchanged retention requests, eight
changed-wording requests, eight fresh-fact tasks and eight unsupported requests.
All arms receive the same fresh raw records before those first attempts; generic
initial addressing must support new slots without a hidden learned association.
Use varied ordinary prose and executable configuration tasks, including longer
values. No source-truth revision challenge, official/protected material or extra
training follows. Freeze the roster and disclose its relationship to the exposed
64 requests; new names alone are not semantic generalization. Earlier and distracting
sources remain available. Count teachers and positive-association certification
calls separately from first actions; provenance revision remains deferred.

Budget: at most two hours preparation and 45 minutes total for one comparison,
including a bounded numerical/runtime check and the executed stream. No seed,
reader, rate, threshold or replay sweep, retry, extra epoch or post-result subset.
Before freeze, fix context splitting, all-source aggregation, abstention, synaptic
updates, proposal limits, replay exposure and state accounting. Estimate and then
measure frozen-reader bytes, all raw episodes/keys/synapses, auxiliary state, child
execution costs and reader calls; use a 2 GiB parent-memory ceiling initially.
For scale only, 64 slots with 384 float64 addressing weights need 196,608 bytes;
that excludes keys, recurrent state, optimizer, records and the external reader.

Adopt only if at least 20/24 supported final probes execute correctly, each
eight-item supported stratum reaches at least 6/8, and unsupported STOP reaches
7/8. Require at least four additional supported successes over frozen and fixed
retrieval controls. Do not adopt the cell design on a contribution claim if the
independent conventional learned router matches it at comparable cost; this excludes
the mathematical parity implementation above. Against full-corpus reading, require
either four more supported successes at comparable measured cost, or no worse
supported/STOP scores with at
least a twofold measured reduction in total inference cost. These are proposed
decision thresholds to freeze, not measurements or statistical significance claims.

If the frozen reader itself cannot answer the full-corpus tasks adequately, report
that component ceiling; do not diagnose memory failure or silently shop for another
reader. If all-source reading or conventional routing dominates at acceptable cost,
reject the hybrid as an incidental neuronal wrapper. If it passes, return directly
to one broader persistent coding/reading session with retained history. Passing
would establish useful learned episodic control, still not general comprehension,
slow abstraction learning, broad software competence or the final project goal.
