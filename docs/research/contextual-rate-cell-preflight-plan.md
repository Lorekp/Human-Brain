# Rate-cell reading acquisition: bounded preflight candidate

Decision, 2026-09-05. This selects a candidate for preparation and preflight,
not an adopted learner or authorization for a fresh/protected comparison.
The [roadmap](roadmap-reading-to-coding.md) and
[active session plan](balanced-evidence-to-action-session-v1.md) own priorities.
Preserve the [previous interface review](contextual-trainable-interface-review.md)
and all frozen experiments as historical evidence.

## Evidence and choice

The [full-choice attempt](../../experiments/r4_integrated/contextual_full_choice_result.md)
achieved 2/48 supported actual requests, 5/48 alternate forms and 0/16 STOPs.
Whole answers were available, but all final selections were single words.
Supported updates overwhelmed STOP corrections through shared feature weights.
The [joint-ranking diagnostic](../../experiments/r4_integrated/contextual_full_choice_feasibility_result.md)
ended inconclusive after an LP timeout; its incomplete witness failed independent
verification. Neither feasibility nor fixed-feature impossibility was established.

| Candidate | Benefit | Limitation |
| --- | --- | --- |
| Fixed BINDING, separate answerability/span losses | Smallest test of direct STOP/span credit coupling | Leaves token boundaries/roles fixed; improvement primarily establishes readout learning |
| Trainable rate-cell token producer, separated heads | Learns query-conditioned sequence and boundary computation | More implementation; shared producer gradients can still interfere |
| Episodic/protected contrasts | Preserves selected acquired distinctions | Earlier protected margins did not cover final competitors; protection can retain weak or wrong distinctions |

Prepare the second candidate with matched controls. This is a joint representation
and objective design, not a causal claim that objective factorization alone helps.
Failure of the prior configuration and the solver timeout do not prove recurrence
necessary or sufficient. Do not reopen an optimizer, rank or replay sweep.

## Computation, supervision and action

Use the prior 64-cell leaky-rate sketch, processing query then document tokens
with segment indicators and resetting between windows. Learn input/recurrent
weights and span/answerability heads through explicitly disclosed backpropagation
through time. This is conventional gradient training of simulated rate neurons,
not local biological plasticity. GLIF may remain the final action readout.

Frozen MiniLM tokens, UTF-8 offsets and raw source custody remain external input.
The existing encoder exposes these; the evidence store needs a token-retention
seam. Torch is not a repository dependency: the contextual extra supplies
ONNX Runtime and tokenizers. Review existing tools and freeze any implementation
or dependency choice before preparing the scored preflight.

Score contiguous word-boundary spans with learned start, end and pooled-interior
readouts. Include lengths above three words; no entity, property, answer-length
or gold-source filter. Freeze window boundaries and cross-window coverage before
outputs. If cross-window answers are unsupported, declare that limitation.
All controls use exactly the same broadened candidate inventory and live inputs.

Train binary answerability on every explicitly authorized answer/STOP lesson.
Train whole-answer marginal span loss only on supported lessons, normalized over
all available TRAIN spans, including every occurrence of the authorized bytes.
Unsupported lessons do not directly update span-ranking heads; shared producer
updates can still cause interference. Gradients on unexecuted competitors are
surrogate supervision, never fabricated FAIL receipts.

Inference is explicitly hierarchical: **STOP when p_stop >= 0.5; otherwise emit
the unique maximum conditional span score; a span tie also yields STOP.**
Never compare p_stop against (1-p_stop) times an individual span probability.
That would reintroduce candidate-count dependence into the abstention decision.
No threshold tuning follows outputs. The same learned edit motor executes choices;
teacher repairs and authorized unchanged-workspace STOP checks remain separate.

Retain authenticated original queries, sources, authorized outputs and genuine
execution receipts. Replay recomputes current trajectories, logits and gradients
under the current producer and sealed TRAIN pool. Old eligibility is not a current
gradient. Hash learned producer/head state and retained inputs; inference cannot
read labels or teacher ledgers. Growth and evaluation never update parameters.
This remains answer-supervised acquisition over retained raw sources, not the
broader roadmap's answer-free acquisition or source-free internalization.

## Controls and cost contract

- Trainable cells with trainable heads: the candidate treatment.
- Identically initialized frozen cells with trainable heads: isolates learning
  of cell computation from conventional head learning.
- Cell bypass using a fixed projection of frozen token inputs with trainable
  heads: a conventional host-readout reference, with the same output inventory.
- Frozen evidence-learner and external-only controls, sharing sources and the
  learned motor.

Raw/neuronal agreement is a readout check, not proof of learned cell contribution.
Freeze the bypass projection and head interfaces; report parameter/cost differences.
The prior sketch suggests about 29.1 thousand float64 parameters, roughly 233 KB,
before gradients/optimizer state. Frozen float32 tokens cost 1,536 bytes/token.
These are estimates, not measured resources. Count encoder, raw sources, ledgers,
token/trajectory caches, span scores, controls and object overhead separately.
Avoid per-span copies of the old 1,621-dimensional candidate feature matrix.

## Freeze, preflight and decision

Allow **two hours of preparation**, then **45 minutes total preflight: 15 minutes
numerical/cost checks and 30 minutes acquisition**. Before target-scored preflight,
freeze window size/coverage, cell and head initialization, optimizer/update rule,
gradient policy, answerability pooling, bypass projection, memory ceiling,
source/candidate manifests, execution schedule and receipt/replay interfaces.

Numerical checks cover recurrent gradients, duplicate-answer marginalization,
STOP credit, changed-producer replay and four-to-eight-word synthetic answers.
Measure full-size windows/inventory, not another six-candidate demonstration.
Reject before teaching if correctness, memory or projected runtime violates the
frozen contract. One initialization and update policy; no retries or sweeps.

Use the already exposed 64-request package: 48 supported and 16 unsupported,
at most 256 encounters and 511 offers per updating arm. Reuse its actual and
alternate questions openly. Keep matched teachers, replay and observation budgets;
retain updates-off source-growth probes. This is public TRAIN development.

Advance only if treatment reaches 44/48 supported actual executions, 15/16 correct
STOPs and at least 80% in each declared role/length and alternate-form group,
retains those thresholds after growth with at most five percentage points loss,
and exceeds both trainable-head controls by at least eight supported actual
executions without sacrificing STOP performance. Otherwise close the candidate.
Equal success with frozen cells attributes improvement to heads/design rather
than learned cell computation. A pass earns a reviewed integration decision,
not a fresh cohort, official-exam score, broad-comprehension claim or protected run.

## Primary research basis

[Lee et al.](https://arxiv.org/pdf/1611.01436) motivate explicit recurrent whole-span
representations; our smaller cell design is an adaptation, not their implementation.
[Clark and Gardner](https://aclanthology.org/P18-1078.pdf) motivate shared passage
normalization and answer-occurrence marginalization.
[Rajpurkar et al.](https://aclanthology.org/P18-2124.pdf) motivate relevant unsupported
questions with plausible competing answers. These sources do not validate this
architecture, fixed teaching budget or proposed acceptance thresholds.
